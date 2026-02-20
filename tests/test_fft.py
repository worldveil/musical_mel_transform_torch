import time

import numpy as np
import onnx
import onnxruntime
import torch
from tqdm import tqdm

from musical_mel_transform.conv_fft import ConvFFT
from musical_mel_transform.musical_mel import MusicalMelTransform

DEFAULT_FRAME_SIZE = 2048


def test_exact_fft_transform_matches_torch_rfft():
    batch_size = 1
    num_channels = 1
    frame_size = DEFAULT_FRAME_SIZE
    iterations = 10

    # Lists to store timing results in milliseconds
    torch_fft_times = []
    conv_fft_times = []

    # Lists to store magnitude differences for summary statistics
    min_mag_diffs = []
    max_mag_diffs = []
    avg_mag_diffs = []
    median_mag_diffs = []
    std_mag_diffs = []
    prop_positive_diffs = []

    # Lists to store signed differences for noise calculation
    all_signed_diffs = []

    conv_fft = ConvFFT(frame_size=frame_size)

    for i in tqdm(range(iterations)):
        signal = torch.randn(batch_size, num_channels, frame_size)

        # Timing Torch rFFT
        start_time = time.time()
        torch_rfft_output = torch.fft.rfft(signal, n=frame_size, dim=-1)
        torch_fft_real = torch_rfft_output.real
        torch_fft_imag = torch_rfft_output.imag
        torch_fft_mag = torch.sqrt(torch_fft_real**2 + torch_fft_imag**2)
        torch_fft_phase = torch.atan2(torch_fft_imag, torch_fft_real)
        torch_fft_times.append((time.time() - start_time) * 1000)  # Convert to ms

        # Timing ConvFFT
        start_time = time.time()
        conv_fft_real, conv_fft_imag, conv_fft_mag, conv_fft_phase = conv_fft.transform(
            signal
        )
        conv_fft_times.append((time.time() - start_time) * 1000)  # Convert to ms

        # Calculate signed magnitude differences (Torch - Conv) for per-iteration reporting
        mag_diff_signed = torch_fft_mag - conv_fft_mag
        min_mag_diff_signed = torch.min(mag_diff_signed).item()
        max_mag_diff_signed = torch.max(mag_diff_signed).item()
        avg_mag_diff_signed = torch.mean(mag_diff_signed).item()
        median_mag_diff_signed = torch.median(mag_diff_signed).item()

        # Store signed differences for noise calculation
        all_signed_diffs.append(mag_diff_signed.detach().cpu().numpy())

        # Calculate absolute differences for summary statistics
        mag_diff_abs = torch.abs(torch_fft_mag - conv_fft_mag)
        min_mag_diff_abs = torch.min(mag_diff_abs).item()
        max_mag_diff_abs = torch.max(mag_diff_abs).item()
        avg_mag_diff_abs = torch.mean(mag_diff_abs).item()
        median_mag_diff_abs = torch.median(mag_diff_abs).item()
        std_mag_diff_abs = (
            torch.std(mag_diff_abs).item() if mag_diff_abs.numel() > 1 else 0.0
        )

        # Calculate proportion of positive differences (Torch > Conv)
        prop_positive = torch.mean((mag_diff_signed > 0).float()).item()

        # Store absolute differences for summary
        min_mag_diffs.append(min_mag_diff_abs)
        max_mag_diffs.append(max_mag_diff_abs)
        avg_mag_diffs.append(avg_mag_diff_abs)
        median_mag_diffs.append(median_mag_diff_abs)
        std_mag_diffs.append(std_mag_diff_abs)
        prop_positive_diffs.append(prop_positive)

        # Allclose assertions
        assert torch.allclose(
            torch_fft_real, conv_fft_real, atol=1e-4
        ), "Real parts are not equal"
        assert torch.allclose(
            torch_fft_imag, conv_fft_imag, atol=1e-4
        ), "Imag parts are not equal"
        assert torch.allclose(
            torch_fft_mag, conv_fft_mag, atol=1e-4
        ), "Magnitude parts are not equal"
        assert torch.allclose(
            torch_fft_phase, conv_fft_phase, atol=5e-4
        ), "Phase parts are not equal"

    # Calculate and print performance statistics
    torch_avg = np.mean(torch_fft_times)
    torch_std = np.std(torch_fft_times)
    conv_avg = np.mean(conv_fft_times)
    conv_std = np.std(conv_fft_times)

    print("\nPerformance Timing Results (in milliseconds):")
    print(f"Torch FFT - Average: {torch_avg:.3f} ms, Std Dev: {torch_std:.3f} ms")
    print(f"Conv FFT  - Average: {conv_avg:.3f} ms, Std Dev: {conv_std:.3f} ms")
    print(f"Difference (Conv - Torch): {(conv_avg - torch_avg):.3f} ms")
    print(f"Torch is faster by: {conv_avg / torch_avg:.2f}x")

    # Calculate and print summary of absolute magnitude differences across all iterations
    print("\nSummary of Absolute Magnitude Differences Across All Iterations:")
    print(
        f"Minimum Magnitude Difference - Min: {np.min(min_mag_diffs):.6f}, Max: {np.max(min_mag_diffs):.6f}, Avg: {np.mean(min_mag_diffs):.6f}, Median: {np.median(min_mag_diffs):.6f}"
    )
    print(
        f"Maximum Magnitude Difference - Min: {np.min(max_mag_diffs):.6f}, Max: {np.max(max_mag_diffs):.6f}, Avg: {np.mean(max_mag_diffs):.6f}, Median: {np.median(max_mag_diffs):.6f}"
    )
    print(
        f"Average Magnitude Difference - Min: {np.min(avg_mag_diffs):.6f}, Max: {np.max(avg_mag_diffs):.6f}, Avg: {np.mean(avg_mag_diffs):.6f}, Median: {np.median(avg_mag_diffs):.6f}"
    )
    print(
        f"Median Magnitude Difference - Min: {np.min(median_mag_diffs):.6f}, Max: {np.max(median_mag_diffs):.6f}, Avg: {np.mean(median_mag_diffs):.6f}, Median: {np.median(median_mag_diffs):.6f}"
    )
    print(
        f"Standard Deviation of Magnitude Difference - Min: {np.min(std_mag_diffs):.6f}, Max: {np.max(std_mag_diffs):.6f}, Avg: {np.mean(std_mag_diffs):.6f}, Median: {np.median(std_mag_diffs):.6f}"
    )
    print(
        f"Proportion of Positive Differences (Torch > Conv) - Min: {np.min(prop_positive_diffs):.3f}, Max: {np.max(prop_positive_diffs):.3f}, Avg: {np.mean(prop_positive_diffs):.3f}, Median: {np.median(prop_positive_diffs):.3f}"
    )

    # Calculate statistics for signed differences to suggest Gaussian noise
    all_signed_diffs = np.concatenate([diff.flatten() for diff in all_signed_diffs])
    mean_diff = np.mean(all_signed_diffs)
    std_diff = np.std(all_signed_diffs)

    print("\nGaussian Noise Parameters to Simulate Conv FFT from Torch FFT:")
    print(f"Mean of Signed Differences (Torch - Conv): {mean_diff:.6f}")
    print(f"Standard Deviation of Signed Differences: {std_diff:.6f}")
    print("Suggested function to add Gaussian noise to Torch FFT magnitudes:")
    print("```python")
    print("def add_conv_fft_noise(torch_fft_mag):")
    print(
        f"    noise = torch.normal(mean={mean_diff:.6f}, std={std_diff:.6f}, size=torch_fft_mag.shape, device=torch_fft_mag.device)"
    )
    print("    return torch_fft_mag + noise")
    print("```")
    print(
        "Note: This function assumes the input torch_fft_mag is a tensor. Adjust the mean and std values based on further testing if needed."
    )


def test_exact_mel_transform_matches_torch_rfft():
    batch_size = 1
    num_channels = 1
    frame_size = DEFAULT_FRAME_SIZE
    iterations = 10

    # Lists to store timing results in milliseconds
    torch_mel_times = []
    conv_mel_times = []

    # creating mel transform with identical kwargs except use_conv_fft=True|False
    kwargs = dict(
        sample_rate=44100,
        frame_size=frame_size,
        interval=1.0,
        f_min=27.5,
        f_max=44100 / 2,
        passthrough_cutoff_hz=10000,
        norm=True,
        min_bins=2,
        power=2,
        adaptive=True,
        passthrough_grouping_size=3,
        learnable_weights=None,
    )
    torch_mel_transform = MusicalMelTransform(use_conv_fft=False, **kwargs)
    conv_mel_transform = MusicalMelTransform(use_conv_fft=True, **kwargs)

    for i in tqdm(range(iterations)):
        signal = torch.randn(batch_size, num_channels, frame_size)

        # Timing Torch Mel Transform (use_conv_fft=False)
        start_time = time.time()
        torch_mel_output, torch_fft_mag = torch_mel_transform.forward_frame(signal.squeeze(1))
        torch_mel_times.append((time.time() - start_time) * 1000)  # Convert to ms

        # Timing Conv Mel Transform (use_conv_fft=True)
        start_time = time.time()
        conv_mel_output, conv_fft_mag = conv_mel_transform.forward_frame(signal.squeeze(1))
        conv_mel_times.append((time.time() - start_time) * 1000)  # Convert to ms

        # Debugging prints
        print(f"Iteration {i + 1}:")
        print(
            "Max mel output difference:",
            torch.max(torch.abs(torch_mel_output - conv_mel_output)),
        )
        print(
            "Max fft magnitude difference:",
            torch.max(torch.abs(torch_fft_mag - conv_fft_mag)),
        )

        # Allclose assertions
        assert torch.allclose(
            torch_mel_output, conv_mel_output, atol=5e-4
        ), "Mel outputs are not equal"
        assert torch.allclose(
            torch_fft_mag, conv_fft_mag, atol=5e-4
        ), "FFT magnitudes are not equal"

    # Calculate and print performance statistics
    torch_avg = np.mean(torch_mel_times)
    torch_std = np.std(torch_mel_times)
    conv_avg = np.mean(conv_mel_times)
    conv_std = np.std(conv_mel_times)

    print("\nPerformance Timing Results (in milliseconds):")
    print(
        f"Torch Mel Transform - Average: {torch_avg:.3f} ms, Std Dev: {torch_std:.3f} ms"
    )
    print(
        f"Conv Mel Transform  - Average: {conv_avg:.3f} ms, Std Dev: {conv_std:.3f} ms"
    )
    print(f"Difference (Conv - Torch): {(conv_avg - torch_avg):.3f} ms")
    print(f"Torch is faster by: {conv_avg / torch_avg:.2f}x")
