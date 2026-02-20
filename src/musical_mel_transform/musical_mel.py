import math
import time
from typing import List, Literal, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.amp as amp

from .conv_fft import ConvFFT


PostTransformType = Literal["db", "log", "log1p", None]


# ──────────────────────────────────────────────────────────────────────────
#  helper – label each frequency with a unique note name (± cents)
# ──────────────────────────────────────────────────────────────────────────
def _hz_to_note(freq: float) -> str:
    midi_f = librosa.hz_to_midi(freq)  # fractional MIDI
    base = int(round(midi_f))  # nearest semitone
    cents = int(round((midi_f - base) * 100))  # −50…+50
    name = librosa.midi_to_note(base, octave=True, unicode=False)
    return name if cents == 0 else f"{name}{'+' if cents > 0 else ''}{cents}¢"


# ──────────────────────────────────────────────────────────────────────────
#  main module
# ──────────────────────────────────────────────────────────────────────────
class MusicalMelTransform(nn.Module):
    """
    Parameters
    ----------
    sample_rate : int
    frame_size  : int      – FFT size / window length
    interval    : float    – step in *semitones* (1.0=semitone, 0.5=quarter-tone…)
    f_min       : float    – lowest note centre
    f_max       : float    – analysis ceiling (default Nyquist)
    passthrough_cutoff_hz : float – above this add identity columns
    norm        : bool     – L¹-normalise every column
    min_bins    : int      – widen triangles so each spans ≥ this many FFT bins
    adaptive    : bool     – adapt filter width to distance from nearest FFT-bin center
    passthrough_grouping_size: int – group this many consecutive pass-through high-frequency bins
    power       : int      – power exponent (1 → magnitude, 2 → power, etc.)
    post_transform_type : str | None – "db", "log", "log1p", or None (raw)
    hop_size    : int | None – required for waveform-mode forward(); None = frame-only
    center      : bool     – pad waveform for centered frames in forward()
    window_periodic : bool – True = periodic window (STFT convention), False = symmetric
    """

    LOG_EPS = 1e-5

    def __init__(
        self,
        sample_rate: int = 44_100,
        frame_size: int = 1_024,
        interval: float = 1.0,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        passthrough_cutoff_hz: float = 20_000,
        norm: bool = True,
        min_bins: int = 2,
        adaptive: bool = True,
        passthrough_grouping_size: int = 3,
        use_conv_fft: bool = False,
        window_type: str = None,
        learnable_weights: str = None,  # None, "fft", "mel"
        power: int = 2,
        post_transform_type: Optional[PostTransformType] = None,
        hop_size: Optional[int] = None,
        center: bool = True,
        window_periodic: bool = True,
    ):
        super().__init__()

        # Use librosa's exact A0 frequency if f_min is not specified
        if f_min is None:
            f_min = librosa.midi_to_hz(librosa.note_to_midi("A0"))

        if f_max is None or f_max > sample_rate / 2:
            f_max = sample_rate / 2

        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.interval = interval
        self.f_min = f_min
        self.f_max = f_max
        self.passthrough_cutoff_hz = passthrough_cutoff_hz
        self.norm = norm
        self.fft_resolution = sample_rate / frame_size
        self.adaptive = adaptive
        self.passthrough_grouping_size = passthrough_grouping_size
        self.use_conv_fft = use_conv_fft
        self.dtype = torch.float
        self.learnable_weights = learnable_weights
        self.power = power
        self.post_transform_type = post_transform_type
        self.hop_size = hop_size
        self.center = center
        self.window_periodic = window_periodic

        # Ensure TorchScript sees this attribute regardless of branch
        self.conv_fft: Optional[ConvFFT] = None

        bw = sample_rate / frame_size
        bin_freqs = np.fft.rfftfreq(frame_size, 1 / sample_rate).astype(np.float64)
        left_edge = bin_freqs - 0.5 * bw
        right_edge = bin_freqs + 0.5 * bw
        n_bins = bin_freqs.size

        # ── musical note centers up to cutoff_hz ────────────────────────
        ratio = 2 ** (interval / 12)
        centers = []
        f = f_min
        while f <= min(f_max, self.passthrough_cutoff_hz):
            centers.append(f)
            f *= ratio
        centers = np.asarray(centers, np.float64)

        # ── build area-overlap basis ────────────────────────────────────
        B = np.zeros((n_bins, centers.size), np.float32)
        r_sqrt = math.sqrt(ratio)

        for k, fc in enumerate(centers):
            fl, fr = fc / r_sqrt, fc * r_sqrt

            effective_min_bins = float(min_bins)
            if self.adaptive:
                dist = np.min(np.abs(fc - bin_freqs))
                norm_dist = dist / (0.5 * bw + 1e-12)  # 0 (on-bin) to 1 (off-bin)
                # Interpolate min_bins: 1.0 for on-bin, `min_bins` for off-bin
                if min_bins > 1:
                    effective_min_bins = 1.0 + (min_bins - 1.0) * norm_dist

            if fr - fl < effective_min_bins * bw:  # ensure at least min_bins width
                extra = 0.5 * (effective_min_bins * bw - (fr - fl))
                fl -= extra
                fr += extra

            area_tri = 0.5 * (fr - fl)
            bins = np.where((right_edge > fl) & (left_edge < fr))[0]

            for j in bins:
                a = max(left_edge[j], fl)
                b = min(right_edge[j], fr)
                if b <= a:
                    continue
                if b <= fc:  # entirely rising edge
                    area = ((b - fl) ** 2 - (a - fl) ** 2) / (2 * (fc - fl))
                elif a >= fc:  # entirely falling edge
                    area = ((fr - a) ** 2 - (fr - b) ** 2) / (2 * (fr - fc))
                else:  # spans the peak
                    rise = ((fc - fl) ** 2 - (a - fl) ** 2) / (2 * (fc - fl))
                    fall = ((fr - fc) ** 2 - (fr - b) ** 2) / (2 * (fr - fc))
                    area = rise + fall
                B[j, k] = area / (area_tri + 1e-12)

        # ── identity columns for high frequencies ───────────────────────
        hi_bins = np.where(
            (bin_freqs >= self.passthrough_cutoff_hz) & (bin_freqs <= f_max)
        )[0]
        if hi_bins.size:
            if self.passthrough_grouping_size > 1:
                n_groups = int(np.ceil(hi_bins.size / self.passthrough_grouping_size))
                hi_B = np.zeros((n_bins, n_groups), np.float32)
                hi_centers = np.zeros(n_groups, np.float64)
                for i in range(n_groups):
                    start = i * self.passthrough_grouping_size
                    end = start + self.passthrough_grouping_size
                    group_bins = hi_bins[start:end]
                    hi_B[group_bins, i] = 1.0
                    hi_centers[i] = bin_freqs[group_bins].mean()
                B = np.concatenate([B, hi_B], axis=1)
                centers = np.concatenate([centers, hi_centers])
            else:
                eye = np.zeros((n_bins, hi_bins.size), np.float32)
                eye[hi_bins, np.arange(hi_bins.size)] = 1.0
                B = np.concatenate([B, eye], axis=1)
                centers = np.concatenate([centers, bin_freqs[hi_bins]])

        # remove any weight on FFT-bin 0 DC offset
        B[0, :] = 0.0

        # normalize if needed
        if norm:
            B /= B.sum(axis=0, keepdims=True) + 1e-12

        # check for negative weights or empty filters
        assert (B.sum(axis=0) == 0).sum() == 0, "Some filters have no weight"
        assert (B < 0).sum() == 0, "Some filters have negative weight"

        # ── constant buffers & labels ────────────────────────────────────
        self.register_buffer("mel_basis", torch.from_numpy(B), persistent=True)
        self.register_buffer(
            "fft_freqs", torch.from_numpy(bin_freqs.astype(np.float32)), persistent=True
        )
        self.register_buffer(
            "mel_freqs", torch.from_numpy(centers.astype(np.float32)), persistent=True
        )
        self.note_names: List[str] = [_hz_to_note(f) for f in centers]
        self.n_mel = centers.size

        if self.use_conv_fft:
            # we will use the windowing in this module, not in the ConvFFT module
            self.conv_fft = ConvFFT(self.frame_size, window_type=None)

        # Create window for windowing the input frames
        if window_type == "hann":
            window = torch.hann_window(frame_size, dtype=self.dtype, periodic=window_periodic)
        elif window_type == "hamming":
            window = torch.hamming_window(frame_size, dtype=self.dtype, periodic=window_periodic)
        else:
            # no windowing
            window = torch.ones(frame_size, dtype=self.dtype)

        self.register_buffer("window", window)

        # setup parameter for learnable weights, depending on the learnable_weights argument
        if self.learnable_weights == "fft":
            self.register_parameter(
                "learnable_weights_param",
                nn.Parameter(torch.randn(self.frame_size // 2 + 1)),
            )
        elif self.learnable_weights == "mel":
            self.register_parameter(
                "learnable_weights_param", nn.Parameter(torch.randn(self.n_mel))
            )
        else:
            self.learnable_weights_param = None

    # ───────────────────────── post-transform helper ───────────────────────
    def _apply_post_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_transform_type == "db":
            multiplier = 20.0 if self.power == 1 else 10.0
            amin = float(torch.finfo(x.dtype).tiny)
            return torchaudio.functional.amplitude_to_DB(
                x,
                multiplier=multiplier,
                amin=amin,
                db_multiplier=0.0,
                top_db=None,
            )
        elif self.post_transform_type == "log":
            return torch.log(x + self.LOG_EPS)
        elif self.post_transform_type == "log1p":
            return torch.log1p(x)
        return x

    # ───────────────────── forward_frame (ONNX-safe) ─────────────────────
    def forward_frame(self, frames: torch.Tensor):
        """
        Frame-level processing (the original forward() interface).

        frames : [B, frame_size]  – time-domain window
        Returns
        -------
        mel      : [B, n_mel]            – musical-mel features
        fft_power: [B, n_fft/2 + 1]      – linear-FFT power
        """
        windowed_frames = frames * self.window
        if self.conv_fft is not None:
            _, _, mag, _ = self.conv_fft.transform(windowed_frames.unsqueeze(1))
            mag = mag.squeeze(1)
        else:
            mag = torch.abs(torch.fft.rfft(windowed_frames, n=self.window.numel()))

        fft_power = mag ** self.power

        if self.learnable_weights == "fft":
            fft_power = fft_power * self.learnable_weights_param

        if fft_power.is_cuda:
            with amp.autocast("cuda", enabled=False):
                mel = fft_power @ self.mel_basis
        else:
            mel = fft_power @ self.mel_basis

        if self.learnable_weights == "mel":
            mel = mel * self.learnable_weights_param

        mel = self._apply_post_transform(mel)
        fft_power = self._apply_post_transform(fft_power)
        return mel, fft_power

    # ───────────────────── forward (waveform mode) ───────────────────────
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Waveform-level processing. Drop-in replacement for MelSpectrogramScaled.

        Requires hop_size to be set at construction time.

        waveform : [B, num_samples] or [num_samples]
        Returns
        -------
        mel : [B, n_mel, num_frames]  (batch dim added if input was 1D)
        """
        if self.hop_size is None:
            raise RuntimeError(
                "MusicalMelTransform.forward() requires hop_size to be set. "
                "Use forward_frame() for frame-level processing, or pass "
                "hop_size= to the constructor for waveform mode."
            )

        squeezed = waveform.dim() == 1
        if squeezed:
            waveform = waveform.unsqueeze(0)

        if self.center:
            pad_amount = self.frame_size // 2
            waveform = F.pad(waveform, (pad_amount, pad_amount))

        # [B, num_frames, frame_size]
        frames = waveform.unfold(1, self.frame_size, self.hop_size)
        batch_size, num_frames, _ = frames.shape

        flat_frames = frames.reshape(batch_size * num_frames, self.frame_size)
        mel, _ = self.forward_frame(flat_frames)

        mel = mel.reshape(batch_size, num_frames, self.n_mel)
        mel = mel.permute(0, 2, 1)
        return mel


# ──────────────────────────────────────────────────────────────────────────
#  plotting helper – visualise lowest filters, add centre dots & guides
# ──────────────────────────────────────────────────────────────────────────
def plot_low_filters(
    bank: MusicalMelTransform,
    bank_idx_to_show: list[int],
    x_max_hz: float = 300.0,
    legend: bool = True,
    x_min_hz: float | None = None,
    title_override: str | None = None,
):
    """
    bank   – MusicalMel instance
    bank_idx_to_show – which filters (by index) to draw
    x_max_hz – right-hand x-axis limit
    legend – show legend if True
    x_min_hz – optional left-hand x-axis limit. If None, starts at 0 Hz.
    """

    B = bank.mel_basis.cpu().numpy()
    f_bins = bank.fft_freqs.cpu().numpy()
    centers = bank.mel_freqs.cpu().numpy()
    names = bank.note_names

    plt.figure(figsize=(9, 4))
    for k in bank_idx_to_show:
        # draw filter
        (line,) = plt.plot(
            f_bins, B[:, k], label=f"{names[k]} ({bank.mel_freqs[k]:.2f} Hz)"
        )
        # dot at centre frequency
        plt.plot(centers[k], 0, marker="o", color=line.get_color(), markersize=10)

    # FFT-bin vertical guides
    lower = 0.0 if x_min_hz is None else x_min_hz
    for f in f_bins[(f_bins <= x_max_hz) & (f_bins >= lower)]:
        plt.axvline(f, ls=":", lw=0.6, color="grey", alpha=0.5)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain")
    if not title_override:
        plt.title(
            f"Selected low-frequency musical-mel filters (FFT N: {bank.frame_size}, res: {bank.fft_resolution:.2f} Hz)"
        )
    else:
        plt.title(title_override)
    if legend:
        plt.legend(ncol=1, fontsize=12, loc="upper right")
    plt.tight_layout()

    plt.xlim(lower, x_max_hz)
    plt.ylim(0, 1)
    plt.show()


class _FrameLevelWrapper(nn.Module):
    """Thin wrapper that routes forward() to forward_frame() for ONNX export."""

    def __init__(self, bank: MusicalMelTransform):
        super().__init__()
        self.bank = bank

    def forward(self, frames: torch.Tensor):
        return self.bank.forward_frame(frames)


def convert_to_onnx(
    bank: MusicalMelTransform,
    onnx_model_save_path: str,
    opset: int = 18,
    dynamic_batch: bool = True,
):
    """
    Convert the bank to an ONNX model (frame-level export).

    Args:
        bank: MusicalMelTransform instance to convert
        onnx_model_save_path: Path to save the ONNX model
        opset: ONNX opset version to use
        dynamic_batch: If True, export with dynamic batch size. If False, fixed batch size = 1.
    """
    wrapper = _FrameLevelWrapper(bank)
    wrapper.eval()

    use_dynamo = bank.use_conv_fft
    dummy_frames = torch.randn(1, bank.frame_size)

    batch_info = "dynamic batch" if dynamic_batch else "fixed batch=1"
    print(
        f"\nExporting to {onnx_model_save_path} with dynamo={use_dynamo}, {batch_info}"
    )

    if use_dynamo and dynamic_batch:
        dynamic_shapes = {
            "frames": (None, bank.frame_size)
        }
        torch.onnx.export(
            wrapper,
            (dummy_frames,),
            onnx_model_save_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["frames"],
            output_names=["mel", "fft_mag"],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            dynamo=use_dynamo,
        )
    else:
        dynamic_axes = (
            {
                "frames": {0: "batch_size"},
                "mel": {0: "batch_size"},
                "fft_mag": {0: "batch_size"},
            }
            if dynamic_batch
            else None
        )

        torch.onnx.export(
            wrapper,
            (dummy_frames,),
            onnx_model_save_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["frames"],
            output_names=["mel", "fft_mag"],
            dynamic_axes=dynamic_axes,
            verbose=False,
            dynamo=use_dynamo,
        )

    print(f"Checking ONNX model...")
    onnx_model = onnx.load(onnx_model_save_path)
    onnx.checker.check_model(onnx_model)

    print(f"Running ONNX model...")
    # do forward pass
    print("=" * 100)
    ort_session = onnxruntime.InferenceSession(onnx_model_save_path)
    print(
        f"Input names for ONNX session: {[inp.name for inp in ort_session.get_inputs()]}"
    )

    onnxruntime_input = {"frames": dummy_frames.detach().numpy()}

    # Warmup
    _ = ort_session.run(None, onnxruntime_input)

    iterations = 100
    timings = []
    for i in range(iterations):
        start = time.time()
        onnxruntime_input["frames"] = np.random.rand(*tuple(dummy_frames.shape)).astype(
            np.float32
        )
        _ = ort_session.run(None, onnxruntime_input)
        elapsed = (time.time() - start) * 1000.0
        timings.append(elapsed)

    print(
        f"ONNX MusicalMel: Forward pass timing: {np.mean(timings):.3f} +/- {np.std(timings):.3f} ms (opset {opset})"
    )

    print("\nInput shapes:")
    for k, v in onnxruntime_input.items():
        print("\t", k, v.shape)

    print("\nOutput shapes:")
    outputs = ort_session.get_outputs()
    results = ort_session.run(None, onnxruntime_input)
    for i, output in enumerate(outputs):
        print("\t", output.name, results[i].shape)


# ──────────────────────────────────────────────────────────────────────────
#  quick demo
# ──────────────────────────────────────────────────────────────────────────
def debug_tuning():
    """Debug the tuning issue"""
    import librosa
    import numpy as np

    print("=== TUNING DEBUG ===")

    # Check librosa's A4 reference
    a4_midi = librosa.note_to_midi("A4")
    a4_freq = librosa.midi_to_hz(a4_midi)
    print(f"Librosa A4: MIDI {a4_midi}, {a4_freq} Hz")

    # Check A0
    a0_midi = librosa.note_to_midi("A0")
    a0_freq = librosa.midi_to_hz(a0_midi)
    print(f"Librosa A0: MIDI {a0_midi}, {a0_freq:.10f} Hz")

    # What does _hz_to_note think about this frequency?
    a0_note_name = _hz_to_note(a0_freq)
    print(f"_hz_to_note({a0_freq:.10f}) = '{a0_note_name}'")

    # Test the round-trip
    a0_back_to_midi = librosa.hz_to_midi(a0_freq)
    print(
        f"Round trip: {a0_freq:.10f} Hz -> {a0_back_to_midi:.10f} MIDI (should be {a0_midi})"
    )

    # Create a simple MusicalMelTransform and check first few notes
    print("\n=== ACTUAL TRANSFORM TEST ===")
    bank = MusicalMelTransform(frame_size=1024, interval=1.0)
    print(f"f_min used: {bank.f_min:.10f} Hz")
    print(f"First 5 frequencies: {bank.mel_freqs[:5].tolist()}")
    print(f"First 5 note names: {bank.note_names[:5]}")

    # Manual calculation of what the frequencies should be
    print(f"\n=== MANUAL CALCULATION ===")
    ratio = 2 ** (1.0 / 12)  # semitone ratio
    for i in range(5):
        expected_freq = bank.f_min * (ratio**i)
        expected_note = _hz_to_note(expected_freq)
        print(f"Note {i}: {expected_freq:.6f} Hz -> '{expected_note}'")


if __name__ == "__main__":
    # Run debug first
    debug_tuning()
    print("\n" + "=" * 80 + "\n")

    SR, N = 44_100, 4096
    bank = MusicalMelTransform(
        frame_size=N,
        interval=1.0,
        norm=True,
        passthrough_cutoff_hz=8_000,
        f_max=12_000,
        adaptive=True,
        passthrough_grouping_size=5,
        use_conv_fft=True,
    )
    # plot_low_filters(bank, bank_idx_to_show=[11, 15], x_max_hz=100)
    plot_low_filters(bank, bank_idx_to_show=[0, 3, 5, 6, 9, 15], x_max_hz=110)
    print(
        f"Stats about our bank: {bank.n_mel} mel bins, {bank.fft_resolution:.2f} Hz resolution"
    )
    # plot_low_filters(bank, bank_idx_to_show=[25, 40, 50, 60, 70, 71, 72, 80, 90, 100, 250, 251, 252, 253, 254, 255], x_max_hz=16000)
    # plot_low_filters(bank, bank_idx_to_show=list(range(bank.mel_basis.shape[1])), x_max_hz=bank.f_max, legend=False)

    # ------------------------------------------------------------------
    # 3) ONNX export and test
    # ------------------------------------------------------------------
    opset = 18
    onnx_model_save_path = f"musical_mel_opset_{opset}.onnx"
    convert_to_onnx(bank, onnx_model_save_path, opset)
