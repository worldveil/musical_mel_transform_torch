#!/usr/bin/env python3
"""Demo script for Musical Mel Transform.

This script demonstrates various features and options of the musical mel transform,
including different parameter settings, visualizations, and performance comparisons.
"""

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

from musical_mel_transform import MusicalMelTransform, convert_to_onnx, plot_low_filters


def _import_matplotlib():
    """Import matplotlib with helpful error message if not available."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting demos. Install it with: "
            "pip install 'musical-mel-transform[plot]' or pip install matplotlib"
        )


def _import_torchaudio():
    """Import torchaudio with helpful error message if not available."""
    try:
        import torchaudio

        return torchaudio
    except ImportError:
        raise ImportError(
            "torchaudio is required for this demo. Install it with: pip install torchaudio"
        )


def create_test_signals(sample_rate: int = 44100, duration: float = 2.0):
    """Create various test signals for demonstration."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)

    signals = {}

    # Musical chord (C major)
    c_major_freqs = [261.63, 329.63, 392.00]  # C4, E4, G4
    chord = sum(0.3 * np.sin(2 * np.pi * f * t) for f in c_major_freqs)
    signals["c_major_chord"] = chord

    # Chromatic scale sweep
    start_freq = 220  # A3
    end_freq = 880  # A5
    sweep = np.sin(2 * np.pi * np.geomspace(start_freq, end_freq, n_samples) * t)
    signals["chromatic_sweep"] = sweep

    # White noise
    noise = 0.1 * np.random.randn(n_samples)
    signals["white_noise"] = noise

    # Harmonic series
    fundamental = 110  # A2
    harmonics = sum(
        (1 / (i + 1)) * np.sin(2 * np.pi * fundamental * (i + 1) * t) for i in range(8)
    )
    signals["harmonic_series"] = harmonics

    return signals


def demo_basic_usage():
    """Demonstrate basic usage of the musical mel transform."""
    print("=== Basic Usage Demo ===")

    # Create transform
    transform = MusicalMelTransform(
        sample_rate=44100,
        frame_size=2048,
        interval=1.0,  # Semitone resolution
        f_min=80.0,
        f_max=8000.0,
        use_conv_fft=True,
    )

    print(f"Created transform with {transform.n_mel} mel bins")
    print(f"FFT resolution: {transform.fft_resolution:.2f} Hz")

    # Generate test signal (must match frame_size)
    test_signal = np.sin(
        2 * np.pi * 440 * np.linspace(0, 1, transform.frame_size)
    )  # A4
    frames = torch.from_numpy(test_signal.astype(np.float32)).unsqueeze(0)

    # Transform
    with torch.no_grad():
        mel_spec, fft_mag = transform.forward_frame(frames)

    print(f"Input shape: {frames.shape}")
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"FFT magnitude shape: {fft_mag.shape}")

    return transform, mel_spec, fft_mag


def demo_parameter_comparison():
    """Compare different parameter settings."""
    print("\n=== Parameter Comparison Demo ===")

    # Test different intervals
    intervals = [0.5, 1.0, 2.0]  # Quarter-tone, semitone, whole-tone
    sample_rate = 44100
    frame_size = 2048

    # Create test signal (C major chord) - use frame_size length
    c_major_freqs = [261.63, 329.63, 392.00]
    t = np.linspace(0, frame_size / sample_rate, frame_size)
    test_signal = sum(0.3 * np.sin(2 * np.pi * f * t) for f in c_major_freqs)
    frames = torch.from_numpy(test_signal.astype(np.float32)).unsqueeze(0)

    plt = _import_matplotlib()

    fig, axes = plt.subplots(len(intervals), 1, figsize=(12, 3 * len(intervals)))
    if len(intervals) == 1:
        axes = [axes]

    for i, interval in enumerate(intervals):
        transform = MusicalMelTransform(
            sample_rate=sample_rate,
            frame_size=frame_size,
            interval=interval,
            f_min=200.0,
            f_max=500.0,
            use_conv_fft=True,
        )

        with torch.no_grad():
            mel_spec, _ = transform.forward_frame(frames)

        # Plot
        mel_freqs = transform.mel_freqs.cpu().numpy()
        axes[i].plot(mel_freqs, mel_spec.squeeze().cpu().numpy(), "o-")
        axes[i].set_xlabel("Frequency (Hz)")
        axes[i].set_ylabel("Magnitude")
        axes[i].set_title(f"Interval: {interval} semitones ({transform.n_mel} bins)")
        axes[i].grid(True, alpha=0.3)

        # Mark the input frequencies
        for freq in c_major_freqs:
            if 200 <= freq <= 500:
                axes[i].axvline(
                    freq, color="red", linestyle="--", alpha=0.7, label=f"{freq:.1f} Hz"
                )
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("parameter_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved parameter comparison plot as 'parameter_comparison.png'")


def demo_filterbank_visualization():
    """Visualize the mel filterbank."""
    print("\n=== Filterbank Visualization Demo ===")

    transform = MusicalMelTransform(
        sample_rate=44100,
        frame_size=2048,
        interval=1.0,
        f_max=8000.0,
        use_conv_fft=False,
        norm=True,
    )

    # Plot low-frequency filters
    plot_low_filters(
        transform,
        bank_idx_to_show=[0, 1, 2, 3, 4, 5, 10, 15, 20, 25],
        x_max_hz=300,
        legend=True,
    )

    # plot highest filters
    plot_low_filters(
        transform,
        bank_idx_to_show=[0, 1, 2, 3, 4, 5],
        x_max_hz=80,
        legend=True,
    )

    plot_low_filters(
        transform,
        bank_idx_to_show=[x + 90 for x in [0, 1, 2, 3, 4, 5]],
        x_max_hz=6750,
        x_min_hz=4850,
        legend=True,
        title_override="High-frequency musical-mel filters (FFT N: 2048, res: 21.56 Hz)",
    )

    transform_passthrough = MusicalMelTransform(
        sample_rate=44100,
        frame_size=2048,
        interval=1.0,
        passthrough_cutoff_hz=5000.0,
        passthrough_grouping_size=3,
        f_max=8000.0,
        use_conv_fft=False,
        norm=True,
    )

    plot_low_filters(
        transform_passthrough,
        bank_idx_to_show=[x + 90 for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
        x_max_hz=6500,
        x_min_hz=4850,
        legend=True,
        title_override="High-frequency musical-mel filters (FFT N: 2048, res: 21.56 Hz, passthrough @ 5kHz / 3 bins)",
    )

    print("Displayed filterbank visualization")


def demo_performance_comparison():
    """Compare performance of different FFT implementations."""
    print("\n=== Performance Comparison Demo ===")

    sample_rate = 44100
    frame_sizes = [256, 512, 1024, 2048, 4096]
    n_iterations = 100

    for frame_size in frame_sizes:
        # Create transforms
        transform_conv = MusicalMelTransform(
            sample_rate=sample_rate,
            frame_size=frame_size,
            use_conv_fft=True,
        )

        transform_torch = MusicalMelTransform(
            sample_rate=sample_rate,
            frame_size=frame_size,
            use_conv_fft=False,
        )

        # Create test signal (use frame_size)
        test_signal = np.random.randn(frame_size).astype(np.float32)
        frames = torch.from_numpy(test_signal).unsqueeze(0)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                transform_conv.forward_frame(frames)
                transform_torch.forward_frame(frames)

        # Benchmark Conv FFT
        conv_times = []
        with torch.no_grad():
            for _ in range(n_iterations):
                start = time.time()
                transform_conv.forward_frame(frames)
                conv_times.append((time.time() - start) * 1000)

        # Benchmark Torch FFT
        torch_times = []
        with torch.no_grad():
            for _ in range(n_iterations):
                start = time.time()
                transform_torch.forward_frame(frames)
                torch_times.append((time.time() - start) * 1000)

        conv_avg = np.mean(conv_times)
        torch_avg = np.mean(torch_times)

        print(f"@ Frame size: {frame_size}")
        print(f"Conv FFT: {conv_avg:.2f} ± {np.std(conv_times):.2f} ms")
        print(f"Torch FFT: {torch_avg:.2f} ± {np.std(torch_times):.2f} ms")
        print(f"Speedup (torch vs convFFT): {conv_avg / torch_avg:.2f}x")
        print("-" * 100)


def demo_onnx_export():
    """Demonstrate ONNX export functionality."""
    print("\n=== ONNX Export Demo ===")

    transform = MusicalMelTransform(
        sample_rate=44100,
        frame_size=1024,
        use_conv_fft=True,  # Required for ONNX export
    )

    # Export to ONNX
    onnx_path = "demo_musical_mel.onnx"
    convert_to_onnx(transform, onnx_path, opset=18)

    print(f"Successfully exported to {onnx_path}")

    # Clean up
    Path(onnx_path).unlink(missing_ok=True)


def demo_musical_analysis():
    """Demonstrate musical analysis capabilities."""
    print("\n=== Musical Analysis Demo ===")

    # Create transform optimized for musical analysis
    transform = MusicalMelTransform(
        sample_rate=44100,
        frame_size=4096,  # Higher resolution for better frequency precision
        interval=0.5,  # Quarter-tone resolution
        f_min=65.0,  # C2
        f_max=4186.0,  # C8
        adaptive=True,
        use_conv_fft=True,
    )

    # Generate musical test signals
    signals = create_test_signals()

    plt = _import_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (name, signal) in enumerate(signals.items()):
        if i >= 4:
            break

        # Truncate or pad signal to match frame_size
        if len(signal) > transform.frame_size:
            signal = signal[: transform.frame_size]
        elif len(signal) < transform.frame_size:
            signal = np.pad(signal, (0, transform.frame_size - len(signal)))

        frames = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            mel_spec, _ = transform.forward_frame(frames)

        mel_freqs = transform.mel_freqs.cpu().numpy()
        mel_values = mel_spec.squeeze().cpu().numpy()

        axes[i].plot(mel_freqs, mel_values, "o-", markersize=3)
        axes[i].set_xlabel("Frequency (Hz)")
        axes[i].set_ylabel("Magnitude")
        axes[i].set_title(f"Musical Analysis: {name.replace('_', ' ').title()}")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xscale("log")

    plt.tight_layout()
    plt.savefig("musical_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved musical analysis plot as 'musical_analysis.png'")


def demo_plot_spectrums(audio: str, top_freq: float):
    """Plot MP3 segments comparing Linear FFT, Torchaudio Mel, and Musical Mel at low freqs."""
    print("\n=== MP3 Plot Demo (≤ top frequency) ===")

    plt = _import_matplotlib()
    torchaudio = _import_torchaudio()

    audio_path = Path(audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Could not find {audio_path}. Pass a valid --audio path.")

    # Load mono @ 44.1k
    y, sr = librosa.load(str(audio_path), sr=44100, mono=True)

    # Analysis params
    frame_size = 2048
    x_max_hz = float(top_freq)

    # Our musical-mel transform configured for real-world f_max (plot may show subset)
    transform = MusicalMelTransform(
        sample_rate=sr,
        frame_size=frame_size,
        interval=1.0,
        f_min=None,
        f_max=16000.0,
        use_conv_fft=True,
        window_type="hann",
    )

    # Match torchaudio mel bin count to the total Musical Mel bins (apples-to-apples)
    f_low = float(transform.f_min)
    n_mels_ta = int(transform.mel_freqs.numel())

    n_stft = frame_size // 2 + 1
    fbanks = torchaudio.functional.melscale_fbanks(
        n_stft,
        n_mels=n_mels_ta,
        f_min=f_low,
        f_max=float(transform.f_max),
        sample_rate=sr,
        norm=None,
        mel_scale="slaney",
    )  # [n_stft, n_mels]
    # L1-normalize columns to match MusicalMelTransform's normalized basis and ignore DC bin weight
    fbanks = fbanks.clone()
    fbanks[0, :] = 0.0
    col_sums = fbanks.sum(dim=0, keepdim=True) + 1e-12
    fbanks = fbanks / col_sums

    # Pick evenly spaced segments
    num_segments = 3
    total = int(len(y) * 0.5)
    if total < frame_size:
        y = np.pad(y, (0, frame_size - total))
        total = len(y)

    centers = np.linspace(frame_size // 2, total - frame_size // 2 - 1, num_segments).astype(int)

    fig, axes = plt.subplots(1, num_segments, figsize=(5 * num_segments, 4), sharey=True)
    if num_segments == 1:
        axes = [axes]

    fft_freqs = transform.fft_freqs.cpu().numpy()
    musical_freqs = transform.mel_freqs.cpu().numpy()

    # Precompute full mel centres for plotting, then slice to <= x_max_hz
    mel_centers_full = librosa.mel_frequencies(
        n_mels=n_mels_ta, fmin=f_low, fmax=float(transform.f_max), htk=False
    )

    for idx, (ax, c) in enumerate(zip(axes, centers)):
        start = int(c - frame_size // 2)
        end = start + frame_size
        frame = y[start:end]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        frames = torch.from_numpy(frame.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            musical_mel, fft_mag = transform.forward_frame(frames)

        # Linear FFT up to x_max_hz (use index via searchsorted for exact alignment)
        k_top_lin = int(np.searchsorted(fft_freqs, x_max_hz, side="right"))
        k_top_lin = max(1, min(k_top_lin, len(fft_freqs)))
        lin_x = fft_freqs[:k_top_lin]
        lin_power = fft_mag.squeeze(0).cpu().numpy()[:k_top_lin]
        lin_power = np.clip(lin_power, 1e-12, None)
        lin_db = 10.0 * np.log10(lin_power)
        lin_y = 10.0 ** ((lin_db - lin_db.max()) / 10.0)  # normalize in dB domain (power)

        # Torchaudio Mel – expects (..., freq). Use 1D input [freq] -> [mel]
        power_spec = fft_mag.squeeze(0).unsqueeze(0)  # [1, freq], power
        power_spec[:, 0] = 0.0
        ta_mel_full = torch.matmul(power_spec, fbanks).squeeze(0).cpu().numpy()  # [n_mels_total]
        # slice torchaudio mel to <= x_max_hz by centers
        k_top_ta = int(np.searchsorted(mel_centers_full, x_max_hz, side="right"))
        k_top_ta = max(1, min(k_top_ta, n_mels_ta))
        mel_centers = mel_centers_full[:k_top_ta]
        ta_mel = ta_mel_full[:k_top_ta]
        ta_mel = np.clip(ta_mel, 1e-12, None)
        ta_db = 10.0 * np.log10(ta_mel)
        ta_y = 10.0 ** ((ta_db - ta_db.max()) / 10.0)

        # Musical Mel up to x_max_hz (index via mel_freqs)
        k_top_mus = int(np.searchsorted(musical_freqs, x_max_hz, side="right"))
        k_top_mus = max(1, min(k_top_mus, len(musical_freqs)))
        mus_x = musical_freqs[:k_top_mus]
        mus_power = musical_mel.squeeze(0).cpu().numpy()[:k_top_mus]
        mus_power = np.clip(mus_power, 1e-12, None)
        mus_db = 10.0 * np.log10(mus_power)
        mus_y = 10.0 ** ((mus_db - mus_db.max()) / 10.0)

        ax.plot(lin_x, lin_y, "-", lw=1.2, label="Linear FFT (bins)")
        ax.plot(mel_centers, ta_y, "o-", ms=3, lw=1.0, label="Torchaudio Mel")
        ax.plot(mus_x, mus_y, "o-", ms=3, lw=1.0, label="Musical Mel")

        t_sec = start / sr
        ax.set_title(f"Segment {idx+1} @ {t_sec:.1f}s")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, x_max_hz)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Normalised magnitude")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_file = "plot_mp3.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved MP3 comparison plot as '{out_file}'")


def demo_plot_spectrogram_comparison(audio: str, top_freq: float, segment_seconds: float):
    """Plot spectrograms per 10s segment: Linear FFT vs Torchaudio Mel vs Musical Mel."""
    print("\n=== MP3 Spectrogram Demo (per segment, ≤ top frequency) ===")

    plt = _import_matplotlib()
    torchaudio = _import_torchaudio()

    audio_path = Path(audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Could not find {audio_path}. Pass a valid --audio path.")

    # Load mono @ 44.1k
    y, sr = librosa.load(str(audio_path), sr=44100, mono=True)

    # Analysis params
    frame_size = 2048
    hop_size = 512
    x_max_hz = float(top_freq)

    # Transform configured for real-world f_max (plot may show subset)
    transform = MusicalMelTransform(
        sample_rate=sr,
        frame_size=frame_size,
        interval=1.0,
        f_min=None,
        f_max=16000.0,
        use_conv_fft=False,
        window_type="hann",
        power=2,
        post_transform_type=None,
    )
    f_low = float(transform.f_min)
    # Match torchaudio mel bin count to total Musical Mel bins
    n_mels_ta = int(transform.mel_freqs.numel())
    n_stft = frame_size // 2 + 1
    fbanks = torchaudio.functional.melscale_fbanks(
        n_stft,
        n_mels=n_mels_ta,
        f_min=f_low,
        f_max=float(transform.f_max),
        sample_rate=sr,
        norm=None,
        mel_scale="slaney",
    )  # [n_stft, n_mels]
    fbanks = fbanks.clone()
    fbanks[0, :] = 0.0
    col_sums = fbanks.sum(dim=0, keepdim=True) + 1e-12
    fbanks = fbanks / col_sums

    freqs_lin = transform.fft_freqs.cpu().numpy()
    k_top_lin = int(np.searchsorted(freqs_lin, x_max_hz, side="right"))
    freqs_lin = freqs_lin[:k_top_lin]
    # Linear FFT frequency edges for y-axis alignment
    bw = float(transform.sample_rate) / float(transform.frame_size)
    lin_edges = np.empty(k_top_lin + 1, dtype=np.float64) if k_top_lin > 0 else np.array([0.0, x_max_hz])
    if k_top_lin > 0:
        lin_edges[0] = max(0.0, freqs_lin[0] - 0.5 * bw)
        if k_top_lin > 1:
            lin_edges[1:-1] = 0.5 * (freqs_lin[:-1] + freqs_lin[1:])
        lin_edges[-1] = min(x_max_hz, freqs_lin[-1] + 0.5 * bw)
    # Ensure strictly positive for log scale
    lin_edges[0] = max(lin_edges[0], max(1e-6, f_low * 0.999))

    freqs_mus = transform.mel_freqs.cpu().numpy()
    k_top_mus = int(np.searchsorted(freqs_mus, x_max_hz, side="right"))
    freqs_mus = freqs_mus[:k_top_mus]
    # Musical mel frequency edges via midpoints of mel_freqs
    if k_top_mus > 1:
        mus_edges = np.empty(k_top_mus + 1, dtype=np.float64)
        mus_edges[1:-1] = 0.5 * (freqs_mus[:-1] + freqs_mus[1:])
        mus_edges[0] = max(0.0, freqs_mus[0] - 0.5 * (freqs_mus[1] - freqs_mus[0]))
        mus_edges[-1] = min(x_max_hz, freqs_mus[-1] + 0.5 * (freqs_mus[-1] - freqs_mus[-2]))
    elif k_top_mus == 1:
        delta = 0.5 * max(1.0, freqs_mus[0] - f_low)
        mus_edges = np.array([max(0.0, freqs_mus[0] - delta), min(x_max_hz, freqs_mus[0] + delta)], dtype=np.float64)
    else:
        mus_edges = np.array([0.0, x_max_hz], dtype=np.float64)
    # Ensure strictly positive for log scale
    mus_edges[0] = max(mus_edges[0], max(1e-6, f_low * 0.999))

    # Torchaudio mel edges and centers over full range
    mel_lo = librosa.hz_to_mel(f_low, htk=False)
    mel_hi = librosa.hz_to_mel(float(transform.f_max), htk=False)
    ta_edges_full = librosa.mel_to_hz(np.linspace(mel_lo, mel_hi, n_mels_ta + 1), htk=False)
    ta_centers_full = librosa.mel_frequencies(n_mels=n_mels_ta, fmin=f_low, fmax=float(transform.f_max), htk=False)
    # slice to <= x_max_hz using centers to keep edges consistent
    k_top_ta = int(np.searchsorted(ta_centers_full, x_max_hz, side="right"))
    k_top_ta = max(1, min(k_top_ta, n_mels_ta))
    ta_edges = ta_edges_full[: k_top_ta + 1]
    # Ensure strictly positive for log scale
    ta_edges[0] = max(ta_edges[0], max(1e-6, f_low * 0.999))

    # Use exactly three segments starting at specified times (seconds)
    seg_len = int(segment_seconds * sr)
    desired_starts_sec = [45.0, 90.0]
    starts_idx = [int(s * sr) for s in desired_starts_sec if int(s * sr) < len(y)]
    if not starts_idx:
        starts_idx = [0]

    n_segments = len(starts_idx)
    fig, axes = plt.subplots(n_segments, 3, figsize=(15, 4 * n_segments), squeeze=False)

    for s, start_idx in enumerate(starts_idx):
        end = min(start_idx + seg_len, len(y))
        seg = y[start_idx:end]
        if len(seg) < frame_size:
            seg = np.pad(seg, (0, frame_size - len(seg)))

        # Frame using sliding_window_view and hop
        windows = np.lib.stride_tricks.sliding_window_view(seg, window_shape=frame_size)
        frames_np = windows[::hop_size]
        if frames_np.size == 0:
            frames_np = seg[:frame_size][None, :]

        frames = torch.from_numpy(frames_np.astype(np.float32))

        with torch.no_grad():
            mel_amp, fft_amp = transform.forward_frame(frames)

        # Prepare spectrograms
        # Linear FFT power -> dB
        lin_spec = fft_amp[:, :k_top_lin].cpu().numpy().T  # [freq, time], power
        lin_spec = np.clip(lin_spec, 1e-12, None)
        lin_db = 10.0 * np.log10(lin_spec)

        # Torchaudio Mel from power -> dB via explicit fbanks
        power_tf = fft_amp  # [time, freq], power
        power_tf[:, 0] = 0.0
        ta_mel_tf = torch.matmul(power_tf, fbanks)  # [time, mel_total]
        ta_mel = ta_mel_tf.cpu().numpy().T  # [mel_total, time]
        ta_mel = np.clip(ta_mel, 1e-12, None)
        # slice to <= top_freq to match ta_edges
        ta_mel = ta_mel[:k_top_ta, :]
        ta_db = 10.0 * np.log10(ta_mel)

        # Musical Mel power -> dB – slice to <= top_freq to match mus_edges
        mus_spec = mel_amp[:, :k_top_mus].cpu().numpy().T  # [mel, time], power
        mus_spec = np.clip(mus_spec, 1e-12, None)
        mus_db = 10.0 * np.log10(mus_spec)

        # Time axis (seconds)
        n_frames = lin_db.shape[1]
        t0 = start_idx / sr
        times = t0 + (np.arange(n_frames) * hop_size) / sr
        t1 = times[-1] if n_frames > 0 else (start_idx + frame_size) / sr

        # Plot each with pcolormesh using true frequency edges for y alignment
        ax_lin = axes[s, 0]
        ax_ta = axes[s, 1]
        ax_mus = axes[s, 2]

        # time edges
        if n_frames > 1:
            t_edges = np.empty(n_frames + 1, dtype=np.float64)
            t_edges[1:-1] = 0.5 * (times[:-1] + times[1:])
            dt = hop_size / sr
            t_edges[0] = max(t0, times[0] - 0.5 * dt)
            t_edges[-1] = min(t1, times[-1] + 0.5 * dt)
        else:
            dt = hop_size / sr
            t_edges = np.array([t0, t0 + dt], dtype=np.float64)

        # y edges for each method computed earlier
        ax_lin.pcolormesh(t_edges, lin_edges, lin_db, shading="auto", cmap="magma", edgecolors='none', linewidth=0, antialiased=False)
        ax_lin.set_title(f"Linear FFT [{t0:.1f}s - {t1:.1f}s]")
        ax_lin.set_ylabel("Hz")
        ax_lin.set_yscale("log")

        ax_ta.pcolormesh(t_edges, ta_edges, ta_db, shading="auto", cmap="magma", edgecolors='none', linewidth=0, antialiased=False)
        ax_ta.set_title("Torchaudio Mel")
        ax_ta.set_yscale("log")

        ax_mus.pcolormesh(t_edges, mus_edges, mus_db, shading="auto", cmap="magma", edgecolors='none', linewidth=0, antialiased=False)
        ax_mus.set_title("Musical Mel")
        ax_mus.set_yscale("log")

        # custom musically-relevant y-ticks
        def _gen_ticks(low: float, high: float):
            ticks = []
            # 20 Hz steps up to 100 Hz
            start = int(np.ceil(max(20.0, low) / 20.0) * 20)
            for v in range(start, int(min(100.0, high)) + 1, 20):
                ticks.append(float(v))
            # 50 Hz steps to 500 Hz
            if high > 100.0:
                start = int(np.ceil(max(150.0, low) / 50.0) * 50)
                for v in range(start, int(min(500.0, high)) + 1, 50):
                    ticks.append(float(v))
            # 100 Hz steps to 1000 Hz
            if high > 500.0:
                start = int(np.ceil(max(600.0, low) / 100.0) * 100)
                for v in range(start, int(min(1000.0, high)) + 1, 100):
                    ticks.append(float(v))
            # 500 Hz steps above 1000 Hz
            if high > 1000.0:
                start = int(np.ceil(max(1500.0, low) / 500.0) * 500)
                # cap at high
                v = start
                while v <= high:
                    ticks.append(float(v))
                    v += 500
            # de-duplicate and sort
            ticks = sorted({t for t in ticks if t > 0})
            return ticks

        yticks = _gen_ticks(f_low, x_max_hz)
        yticklabels = [f"{int(t)}" for t in yticks]

        for ax in (ax_lin, ax_ta, ax_mus):
            ax.set_xlim(times[0], times[-1] if n_frames > 0 else t1)
            ax.set_ylim(f_low, x_max_hz)
            ax.set_xlabel("Time (s)")
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)

    plt.tight_layout()
    out_file = "plot_mp3_spectrograms.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved MP3 spectrogram comparison as '{out_file}'")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Musical Mel Transform Demo")
    parser.add_argument(
        "--demo",
        choices=["all", "basic", "params", "filters", "performance", "onnx", "musical", "plot_spectrums", "plot_spectrogram_comparison"],
        default="all",
        help="Which demo to run",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--audio", type=str, default="data/hawk_DREVO.mp3", help="Path to an audio file (e.g., MP3)")
    parser.add_argument("--top-freq", type=float, default=150.0, help="Upper frequency (Hz) to display")
    parser.add_argument("--segment-seconds", type=float, default=10.0, help="Segment length in seconds for spectrograms")

    args = parser.parse_args()

    print("Musical Mel Transform Demo")
    print("=" * 50)

    try:
        if args.demo in ["all", "basic"]:
            demo_basic_usage()

        if args.demo in ["all", "params"] and not args.no_plots:
            demo_parameter_comparison()

        if args.demo in ["all", "filters"] and not args.no_plots:
            demo_filterbank_visualization()

        if args.demo in ["all", "performance"]:
            demo_performance_comparison()

        if args.demo in ["all", "onnx"]:
            demo_onnx_export()

        if args.demo in ["all", "musical"] and not args.no_plots:
            demo_musical_analysis()

        if args.demo in ["plot_spectrums"] and not args.no_plots:
            demo_plot_spectrums(args.audio, args.top_freq)

        if args.demo in ["plot_spectrogram_comparison"] and not args.no_plots:
            demo_plot_spectrogram_comparison(args.audio, args.top_freq, args.segment_seconds)

        print("\n" + "=" * 50)
        print("Demo completed successfully!")

    except Exception as e:
        print(f"Error during demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
