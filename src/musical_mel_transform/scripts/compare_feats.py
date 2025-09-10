import math

import librosa
import torchaudio
import torch

from musical_mel_transform import MusicalMelTransform


# Analysis params
frame_size = 2048
hop_size = 512
sr = 44100
f_max = 16000.0

audio_frames = torch.randn(4, frame_size)

# Transform configured for real-world f_max (plot may show subset)
transform = MusicalMelTransform(
    sample_rate=sr,
    frame_size=frame_size,
    interval=1,
    f_min=None,
    f_max=f_max,
    adaptive=True,
    use_conv_fft=False,
    window_type="hann",
    # passthrough_cutoff_hz=2000.0,
    # passthrough_grouping_size=3,
)
num_mels = transform.mel_freqs.numel()
musical_bin_centers = transform.mel_freqs

mel_scale = torchaudio.transforms.MelScale(
    n_mels=num_mels,
    sample_rate=sr,
    f_min=float(transform.f_min),
    f_max=float(transform.f_max),
    n_stft=frame_size // 2 + 1,
    mel_scale="slaney",
    norm=None,
)

def hz_to_mel(freq: torch.Tensor | float, *, mel_scale: str = "slaney") -> torch.Tensor:
    """
    Convert frequency in Hz to mel.  Vectorised, differentiable, ONNX-safe.
    """
    f = torch.as_tensor(freq, dtype=torch.float64)

    if mel_scale == "htk":
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    # —— Slaney (Auditory Toolbox) ———————————————
    f_sp = 200.0 / 3                      #  ≈66.67 Hz per mel below 1 kHz
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp      #  15 mels
    logstep = math.log(6.4) / 27.0

    return torch.where(
        f < min_log_hz,
        f / f_sp,
        min_log_mel + torch.log(f / min_log_hz) / logstep,
    )


def mel_to_hz(mel: torch.Tensor | float, *, mel_scale: str = "slaney") -> torch.Tensor:
    """Inverse of hz_to_mel (same notes about vectorisation/ONNX)."""
    m = torch.as_tensor(mel, dtype=torch.float64)

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    return torch.where(
        m < min_log_mel,
        m * f_sp,
        min_log_hz * torch.exp(logstep * (m - min_log_mel)),
    )

mel_scale_type = "slaney"      # "slaney" or "htk"
mel_pts = torch.linspace(
    hz_to_mel(float(transform.f_min), mel_scale=mel_scale_type),
    hz_to_mel(f_max, mel_scale=mel_scale_type),
    num_mels + 2,
    dtype=torch.float64,
)
edges_hz = mel_to_hz(mel_pts, mel_scale=mel_scale_type)
freq_cutoffs = torch.stack((edges_hz[:-2], edges_hz[2:]), 1)   # shape (n_mels, 2)

torchaudio_bin_centers = freq_cutoffs.mean(dim=1)

# Linear FFT bin centres (real-valued FFT up to Nyquist)
fft_bin_centers = torch.fft.rfftfreq(frame_size, d=1.0 / sr)
fft_bin_centers = fft_bin_centers[fft_bin_centers <= f_max]

print(musical_bin_centers.shape)
print(torchaudio_bin_centers.shape)

optimized_transform = MusicalMelTransform(
    sample_rate=sr,
    frame_size=frame_size,
    interval=1,
    f_max=f_max,
    adaptive=True,
    use_conv_fft=False,
    window_type="hann",
    passthrough_cutoff_hz=5000.0,
    passthrough_grouping_size=6,
)

# now calculate the number of bins in each range for both of the bin centers
# and report back to the user with a table

freq_ranges_hz = [
    (0, 150),
    (150, 500),
    (500, 1000),
    (1000, 3000),
    (3000, 6000),
    (6000, 9000),
    (9000, 16000),
]

# Compute and display bin counts per frequency range for both transforms

def count_bins(bin_centers: torch.Tensor, ranges_hz: list[tuple[int, int]]):
    """Return list with number of centres falling within each (low, high] Hz range."""
    counts = []
    for low, high in ranges_hz:
        mask = (bin_centers > low) & (bin_centers <= high)
        counts.append(int(mask.sum()))
    return counts

musical_counts = count_bins(musical_bin_centers, freq_ranges_hz)
torch_counts = count_bins(torchaudio_bin_centers, freq_ranges_hz)
fft_counts = count_bins(fft_bin_centers, freq_ranges_hz)
optimized_bin_centers = optimized_transform.mel_freqs
opt_counts = count_bins(optimized_bin_centers, freq_ranges_hz)

header = ("Range (Hz)", "Musical", "Torchaudio", "LinearFFT", "Optimized")
print("{:<15} {:>10} {:>12} {:>12} {:>12}".format(*header))
print("-" * 65)
for (low, high), m_cnt, t_cnt, f_cnt, o_cnt in zip(freq_ranges_hz, musical_counts, torch_counts, fft_counts, opt_counts):
    range_lbl = f"{low}-{high}"
    print(f"{range_lbl:<15} {m_cnt:>10} {t_cnt:>12} {f_cnt:>12} {o_cnt:>12}")

print("-" * 65)
print(f"{'TOTAL':<15} {sum(musical_counts):>10} {sum(torch_counts):>12} {sum(fft_counts):>12} {sum(opt_counts):>12}")

# —— Plot grouped bar chart ————————————————————————————
import numpy as np
import matplotlib.pyplot as plt

range_labels = [f"{low}-{high}" for low, high in freq_ranges_hz]

x = np.arange(len(range_labels))
bar_w = 0.18

plt.figure(figsize=(11, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.bar(x - 1.5*bar_w, musical_counts, width=bar_w, label=f"Musical Mel (semitones) - {transform.n_mel} feats", color=colors[0])
plt.bar(x - 0.5*bar_w, torch_counts, width=bar_w, label=f"MelScale (torchaudio) - {num_mels} feats", color=colors[1])
plt.bar(x + 0.5*bar_w, fft_counts, width=bar_w, label=f"Vanilla FFT - {len(fft_bin_centers)} feats", color=colors[2])
plt.bar(x + 1.5*bar_w, opt_counts, width=bar_w, label=f"Musical Mel (passthrough @ 5kHz / 6 bins) - {optimized_transform.n_mel} feats", color=colors[3])

# Add count labels
for offset, counts in [(-1.5*bar_w, musical_counts), (-0.5*bar_w, torch_counts), (0.5*bar_w, fft_counts), (1.5*bar_w, opt_counts)]:
    for xpos, cnt in zip(x + offset, counts):
        plt.text(xpos, cnt + 1, str(cnt), ha="center", va="bottom", fontsize=7)

plt.xticks(x, range_labels, rotation=45, ha="right")
plt.xlabel("Frequency Range (Hz)")
plt.ylabel("Number of Bin Centres")
plt.title("Frequency-bin Distribution for Different Transforms @ FFT Size 2048")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()