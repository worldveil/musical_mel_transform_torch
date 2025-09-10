import numpy as np
import torch


class ConvFFT(torch.nn.Module):
    """
    ONNX compatible FFT. It is slower than torch native FFT :/ but it is accurate!

    For small frame sizes, the speed difference is not too bad, but >=2048 you will start to feel the pain. Run the
    performance demo described in the README to see the difference.

    $ musical-mel-demo --demo performance
    """

    def __init__(self, frame_size: int, window_type: str = None):
        super(ConvFFT, self).__init__()

        self.frame_size = frame_size
        self.cutoff = frame_size // 2 + 1
        self.dtype = torch.float

        # Precompute Fourier basis, matching torch.fft.rfft normalization
        fourier_basis = np.fft.fft(
            np.eye(frame_size), norm="backward"
        )  # Unnormalized DFT
        fourier_basis = np.vstack(
            [
                np.real(fourier_basis[: self.cutoff, :]),
                np.imag(fourier_basis[: self.cutoff, :]),
            ]
        )
        forward_basis = torch.tensor(fourier_basis, dtype=self.dtype).transpose(
            0, 1
        )  # double precision for accuracy

        self.register_buffer("forward_basis", forward_basis)

        # Create Hann window for windowing the input frames
        if window_type == "hann":
            window = torch.hann_window(frame_size, dtype=self.dtype)
        elif window_type == "hamming":
            window = torch.hamming_window(frame_size, dtype=self.dtype)
        else:
            # no windowing
            window = torch.ones(frame_size, dtype=self.dtype)

        self.register_buffer("window", window)

    def num_fft_features(self):
        return self.cutoff

    def transform(self, frames, eps: float = 1e-12):
        """
        Input: [batch, channel, time]
        Output:
            real_part: [batch, channel, (frame_size // 2 + 1)]
            imag_part: [batch, channel, (frame_size // 2 + 1)]
            magnitude: [batch, channel, (frame_size // 2 + 1)]
            phase:     [batch, channel, (frame_size // 2 + 1)]
        """
        batch_size, num_channels, time = frames.shape
        frames = frames.view(-1, time).to(self.dtype)  # compute in double

        # Apply Hann window to reduce spectral leakage
        frames = (
            frames * self.window
        )  # Broadcasting: [batch * channel, frame_size] * [frame_size]

        # Matrix multiplication
        forward_transform = torch.matmul(frames, self.forward_basis)

        # Reshape to [batch, channel, 2 * cutoff]
        forward_transform = forward_transform.view(batch_size, num_channels, -1).to(self.dtype)
        real_part = forward_transform[:, :, : self.cutoff]
        imag_part = forward_transform[:, :, self.cutoff :]

        # Compute magnitude and phase
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        safe_denominator = torch.where(
            torch.abs(real_part) < eps, torch.full_like(real_part, eps), real_part
        )
        phase = torch.atan2(imag_part, safe_denominator)

        return real_part, imag_part, magnitude, phase
