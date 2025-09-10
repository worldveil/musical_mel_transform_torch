"""Tests for the MusicalMelTransform module."""

import numpy as np
import pytest
import torch

from musical_mel_transform.musical_mel import MusicalMelTransform


class TestMusicalMelTransform:
    """Test cases for MusicalMelTransform class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        transform = MusicalMelTransform()

        assert transform.sample_rate == 44100
        assert transform.frame_size == 1024
        assert transform.interval == 1.0
        assert transform.f_min == 27.5
        assert transform.adaptive is True
        assert transform.use_conv_fft is False

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=2048,
            interval=0.5,
            f_min=80.0,
            f_max=8000.0,
            use_conv_fft=True,
        )

        assert transform.sample_rate == 44100
        assert transform.frame_size == 2048
        assert transform.interval == 0.5
        assert transform.f_min == 80.0
        assert transform.f_max == 8000.0
        assert transform.use_conv_fft is True

    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            use_conv_fft=True,
        )

        # Create test input
        batch_size = 2
        test_input = torch.randn(batch_size, transform.frame_size)

        with torch.no_grad():
            mel_spec, fft_mag = transform(test_input)

        # Check shapes
        assert mel_spec.shape[0] == batch_size
        assert fft_mag.shape[0] == batch_size
        assert fft_mag.shape[1] == transform.frame_size // 2 + 1  # frequency dimension
        assert mel_spec.shape[1] == transform.n_mel  # mel dimension

    def test_torch_vs_conv_fft_consistency(self):
        """Test that torch FFT and conv FFT produce similar results."""
        # Same parameters for both transforms
        kwargs = dict(
            sample_rate=44100,
            frame_size=1024,
            interval=1.0,
            f_min=80.0,
            f_max=8000.0,
            norm=True,
        )

        torch_transform = MusicalMelTransform(use_conv_fft=False, **kwargs)
        conv_transform = MusicalMelTransform(use_conv_fft=True, **kwargs)

        # Create test signal
        test_input = torch.randn(1, 1024)

        with torch.no_grad():
            torch_mel, torch_fft = torch_transform(test_input)
            conv_mel, conv_fft = conv_transform(test_input)

        # Results should be close but not identical due to implementation differences
        torch.testing.assert_close(torch_mel, conv_mel, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(torch_fft, conv_fft, rtol=1e-3, atol=1e-4)

    def test_different_intervals(self):
        """Test different musical intervals."""
        intervals = [0.5, 1.0, 2.0]  # quarter-tone, semitone, whole-tone

        for interval in intervals:
            transform = MusicalMelTransform(
                sample_rate=44100,
                frame_size=1024,
                interval=interval,
                f_min=100.0,
                f_max=1000.0,
            )

            test_input = torch.randn(1, 1024)
            with torch.no_grad():
                mel_spec, fft_mag = transform(test_input)

            # Check that we get reasonable outputs
            assert not torch.isnan(mel_spec).any()
            assert not torch.isinf(mel_spec).any()
            assert mel_spec.shape[1] == transform.n_mel  # mel dimension

    def test_frequency_ranges(self):
        """Test different frequency ranges."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=2048,
            f_min=80.0,
            f_max=8000.0,
        )

        # Check that mel frequencies are within expected range
        mel_freqs = transform.mel_freqs.cpu().numpy()
        assert mel_freqs[0] >= transform.f_min
        assert mel_freqs[-1] <= transform.f_max

        # Check that frequencies are increasing
        assert np.all(np.diff(mel_freqs) > 0)

    def test_note_names(self):
        """Test that note names are generated correctly."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            f_min=220.0,  # A3
            f_max=880.0,  # A5
            interval=1.0,  # Semitone
        )

        # Should have note names for each mel bin
        assert len(transform.note_names) == transform.n_mel

        # Note names should be strings
        for name in transform.note_names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_adaptive_vs_fixed(self):
        """Test adaptive vs fixed filter sizing."""
        adaptive_transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            adaptive=True,
            min_bins=2,
        )

        fixed_transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            adaptive=False,
            min_bins=2,
        )

        # Both should work and produce different filterbanks
        assert adaptive_transform.mel_basis.shape == fixed_transform.mel_basis.shape

        # Filterbanks should be different due to adaptive sizing
        assert not torch.allclose(
            adaptive_transform.mel_basis, fixed_transform.mel_basis
        )

    def test_passthrough_grouping(self):
        """Test high-frequency passthrough grouping."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=2048,
            passthrough_cutoff_hz=5000.0,
            passthrough_grouping_size=3,
        )

        # Should have some high-frequency bins
        high_freq_bins = (transform.fft_freqs >= transform.passthrough_cutoff_hz).sum()
        assert high_freq_bins > 0

        # Check that filterbank includes high-frequency components
        assert transform.n_mel > 50  # Should have many bins including high-freq ones

    def test_normalization(self):
        """Test filterbank normalization."""
        norm_transform = MusicalMelTransform(norm=True)
        no_norm_transform = MusicalMelTransform(norm=False)

        # Normalized filterbank should have different properties
        norm_sums = norm_transform.mel_basis.sum(dim=0)
        no_norm_sums = no_norm_transform.mel_basis.sum(dim=0)

        # Normalized filters should have more uniform sums
        norm_std = torch.std(norm_sums)
        no_norm_std = torch.std(no_norm_sums)

        # This is a heuristic check - normalized should be more uniform
        assert norm_std < no_norm_std * 2  # Allow some tolerance

    def test_device_compatibility(self):
        """Test that the transform works on different devices."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = MusicalMelTransform(frame_size=1024)
        transform = transform.to(device)

        test_input = torch.randn(1, 1024, device=device)

        with torch.no_grad():
            mel_spec, fft_mag = transform(test_input)

        assert mel_spec.device == device
        assert fft_mag.device == device

    def test_gradient_flow(self):
        """Test that gradients flow through the transform."""
        transform = MusicalMelTransform(frame_size=1024, use_conv_fft=True)
        test_input = torch.randn(1, 1024, requires_grad=True)

        mel_spec, fft_mag = transform(test_input)
        loss = mel_spec.sum() + fft_mag.sum()
        loss.backward()

        assert test_input.grad is not None
        assert not torch.isnan(test_input.grad).any()

    def test_learnable_weights_none(self):
        """Test initialization and forward pass with learnable_weights=None."""
        transform = MusicalMelTransform(
            frame_size=1024, learnable_weights=None, use_conv_fft=True
        )

        # Should not have learnable_weights_param parameter when learnable_weights=None
        assert transform.learnable_weights_param is None

        # Forward pass should work normally
        test_input = torch.randn(2, 1024)
        with torch.no_grad():
            mel_spec, fft_mag = transform(test_input)

        assert mel_spec.shape[0] == 2
        assert fft_mag.shape[0] == 2
        assert not torch.isnan(mel_spec).any()
        assert not torch.isnan(fft_mag).any()

    def test_learnable_weights_fft(self):
        """Test initialization and forward pass with learnable_weights='fft'."""
        transform = MusicalMelTransform(
            frame_size=1024, learnable_weights="fft", use_conv_fft=True
        )

        # Should have learnable_weights_param parameter with correct shape
        assert hasattr(transform, "learnable_weights_param")
        assert isinstance(transform.learnable_weights_param, torch.nn.Parameter)
        expected_shape = (1024 // 2 + 1,)  # FFT bins
        assert transform.learnable_weights_param.shape == expected_shape

        # Parameter should be randomly initialized (small values around 0)
        assert (
            torch.abs(transform.learnable_weights_param).mean() < 1.0
        )  # Should be small random values
        assert not torch.allclose(
            transform.learnable_weights_param, torch.zeros(expected_shape)
        )  # Should not be all zeros

        # Forward pass should work
        test_input = torch.randn(2, 1024)
        with torch.no_grad():
            mel_spec, fft_mag = transform(test_input)

        assert mel_spec.shape[0] == 2
        assert fft_mag.shape[0] == 2
        assert not torch.isnan(mel_spec).any()
        assert not torch.isnan(fft_mag).any()

    def test_learnable_weights_mel(self):
        """Test initialization and forward pass with learnable_weights='mel'."""
        transform = MusicalMelTransform(
            frame_size=1024, learnable_weights="mel", use_conv_fft=True
        )

        # Should have learnable_weights_param parameter with correct shape
        assert hasattr(transform, "learnable_weights_param")
        assert isinstance(transform.learnable_weights_param, torch.nn.Parameter)
        expected_shape = (transform.n_mel,)  # Mel bins
        assert transform.learnable_weights_param.shape == expected_shape

        # Parameter should be randomly initialized (small values around 0)
        assert (
            torch.abs(transform.learnable_weights_param).mean() < 1.0
        )  # Should be small random values
        assert not torch.allclose(
            transform.learnable_weights_param, torch.zeros(expected_shape)
        )  # Should not be all zeros

        # Forward pass should work
        test_input = torch.randn(2, 1024)
        with torch.no_grad():
            mel_spec, fft_mag = transform(test_input)

        assert mel_spec.shape[0] == 2
        assert fft_mag.shape[0] == 2
        assert not torch.isnan(mel_spec).any()
        assert not torch.isnan(fft_mag).any()

    def test_learnable_weights_fft_affects_output(self):
        """FFT weights with default dB output should add ~+3.01 dB where not floor-limited."""
        transform = MusicalMelTransform(
            frame_size=1024, learnable_weights="fft", use_conv_fft=True
        )

        test_input = torch.randn(1, 1024)

        with torch.no_grad():
            mel_default, fft_default = transform(test_input)

        with torch.no_grad():
            transform.learnable_weights_param.data *= 2.0

        with torch.no_grad():
            mel_modified, fft_modified = transform(test_input)

        # Expect +3.010299957 dB for power doubling (power=2 default → 10*log10)
        delta_db_expected = 10.0 * torch.log10(torch.tensor(2.0))

        # Mask to avoid bins at the floor (-100 dB)
        floor = -100.0
        fft_mask = fft_default > (floor + 1e-6)
        mel_mask = mel_default > (floor + 1e-6)

        torch.testing.assert_close(
            (fft_modified - fft_default)[fft_mask],
            delta_db_expected.expand_as((fft_modified - fft_default)[fft_mask]),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            (mel_modified - mel_default)[mel_mask],
            delta_db_expected.expand_as((mel_modified - mel_default)[mel_mask]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_learnable_weights_mel_affects_output(self):
        """Mel weights with default dB output should subtract ~3.01 dB where not floor-limited."""
        transform = MusicalMelTransform(
            frame_size=1024, learnable_weights="mel", use_conv_fft=True
        )

        test_input = torch.randn(1, 1024)

        with torch.no_grad():
            mel_default, fft_default = transform(test_input)

        with torch.no_grad():
            transform.learnable_weights_param.data *= 0.5

        with torch.no_grad():
            mel_modified, fft_modified = transform(test_input)

        # FFT should be unchanged by mel weights
        torch.testing.assert_close(fft_default, fft_modified, rtol=1e-7, atol=1e-8)

        # Expect -3.010299957 dB for halving power weights
        delta_db_expected = -10.0 * torch.log10(torch.tensor(2.0))
        floor = -100.0
        mel_mask = mel_default > (floor + 1e-6)

        assert not torch.allclose(mel_default, mel_modified)
        torch.testing.assert_close(
            (mel_modified - mel_default)[mel_mask],
            delta_db_expected.expand_as((mel_modified - mel_default)[mel_mask]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_learnable_weights_fft_affects_output_linear(self):
        """Linear scale: doubling FFT weights doubles outputs."""
        transform = MusicalMelTransform(
            frame_size=1024, learnable_weights="fft", use_conv_fft=True, to_db=False
        )

        test_input = torch.randn(1, 1024)

        with torch.no_grad():
            mel_default, fft_default = transform(test_input)

        with torch.no_grad():
            transform.learnable_weights_param.data *= 2.0

        with torch.no_grad():
            mel_modified, fft_modified = transform(test_input)

        assert not torch.allclose(fft_default, fft_modified)
        torch.testing.assert_close(fft_modified, fft_default * 2.0, rtol=1e-5, atol=1e-6)
        assert not torch.allclose(mel_default, mel_modified)
        torch.testing.assert_close(mel_modified, mel_default * 2.0, rtol=1e-5, atol=1e-6)

    def test_learnable_weights_mel_affects_output_linear(self):
        """Linear scale: halving mel weights halves mel outputs while FFT unchanged."""
        transform = MusicalMelTransform(
            frame_size=1024, learnable_weights="mel", use_conv_fft=True, to_db=False
        )

        test_input = torch.randn(1, 1024)

        with torch.no_grad():
            mel_default, fft_default = transform(test_input)

        with torch.no_grad():
            transform.learnable_weights_param.data *= 0.5

        with torch.no_grad():
            mel_modified, fft_modified = transform(test_input)

        torch.testing.assert_close(fft_default, fft_modified, rtol=1e-7, atol=1e-8)
        assert not torch.allclose(mel_default, mel_modified)
        torch.testing.assert_close(mel_modified, mel_default * 0.5, rtol=1e-5, atol=1e-6)

    def test_learnable_weights_gradient_flow(self):
        """Test that gradients flow through learnable weights."""
        for learnable_weights in ["fft", "mel"]:
            transform = MusicalMelTransform(
                frame_size=1024, learnable_weights=learnable_weights, use_conv_fft=True
            )

            test_input = torch.randn(1, 1024, requires_grad=True)

            mel_spec, fft_mag = transform(test_input)
            loss = mel_spec.sum() + fft_mag.sum()
            loss.backward()

            # Input gradients should exist
            assert test_input.grad is not None
            assert not torch.isnan(test_input.grad).any()

            # Learnable weights gradients should exist
            assert transform.learnable_weights_param.grad is not None
            assert not torch.isnan(transform.learnable_weights_param.grad).any()
            assert not torch.allclose(
                transform.learnable_weights_param.grad,
                torch.zeros_like(transform.learnable_weights_param.grad),
            )

    def test_learnable_weights_with_torch_fft(self):
        """Test learnable weights work with torch FFT (not just ConvFFT)."""
        for learnable_weights in ["fft", "mel"]:
            transform = MusicalMelTransform(
                frame_size=1024,
                learnable_weights=learnable_weights,
                use_conv_fft=False,  # Use torch FFT
            )

            test_input = torch.randn(2, 1024)

            with torch.no_grad():
                mel_spec, fft_mag = transform(test_input)

            assert mel_spec.shape[0] == 2
            assert fft_mag.shape[0] == 2
            assert not torch.isnan(mel_spec).any()
            assert not torch.isnan(fft_mag).any()

    def test_learnable_weights_different_frame_sizes(self):
        """Test learnable weights with different frame sizes."""
        frame_sizes = [512, 1024, 2048]

        for frame_size in frame_sizes:
            for learnable_weights in ["fft", "mel"]:
                transform = MusicalMelTransform(
                    frame_size=frame_size,
                    learnable_weights=learnable_weights,
                    use_conv_fft=True,
                )

                # Check parameter shapes
                if learnable_weights == "fft":
                    expected_shape = (frame_size // 2 + 1,)
                else:  # mel
                    expected_shape = (transform.n_mel,)

                assert transform.learnable_weights_param.shape == expected_shape

                # Test forward pass
                test_input = torch.randn(1, frame_size)
                with torch.no_grad():
                    mel_spec, fft_mag = transform(test_input)

                assert not torch.isnan(mel_spec).any()
                assert not torch.isnan(fft_mag).any()

    @pytest.mark.integration
    def test_realistic_audio_signal(self):
        """Test with a realistic musical signal."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=2048,
            interval=1.0,
            f_min=80.0,
            f_max=8000.0,
        )

        # Create a realistic musical signal (chord progression) - use frame_size length
        duration = (
            transform.frame_size / transform.sample_rate
        )  # Duration that matches frame_size
        t = np.linspace(0, duration, transform.frame_size)

        # C major chord: C4, E4, G4
        frequencies = [261.63, 329.63, 392.00]
        signal = sum(0.3 * np.sin(2 * np.pi * f * t) for f in frequencies)

        # Add some harmonics and noise for realism
        for f in frequencies:
            signal += 0.1 * np.sin(2 * np.pi * f * 2 * t)  # Second harmonic
        signal += 0.05 * np.random.randn(len(t))  # Noise

        # Convert to tensor and apply transform
        frames = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            mel_spec, fft_mag = transform(frames)

        # Check that we get reasonable activations
        assert mel_spec.max() > 0.1  # Should have significant energy
        assert not torch.isnan(mel_spec).any()
        assert not torch.isinf(mel_spec).any()

        # The signal should activate multiple mel bins around the chord frequencies
        mel_freqs = transform.mel_freqs.cpu().numpy()
        mel_values = mel_spec.squeeze().cpu().numpy()

        # Find bins that correspond to our input frequencies
        chord_activations = []
        for freq in frequencies:
            # Find closest mel bin
            closest_idx = np.argmin(np.abs(mel_freqs - freq))
            chord_activations.append(mel_values[closest_idx])

        # Chord frequencies should have higher activation than the average
        avg_activation = np.mean(mel_values)
        for activation in chord_activations:
            assert activation > avg_activation * 0.5  # Allow some tolerance

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 32000, 44100, 48000])
    def test_different_sample_rates(self, sample_rate):
        """Test that the transform works with different sample rates."""
        transform = MusicalMelTransform(
            sample_rate=sample_rate,
            frame_size=1024,
            interval=1.0,
            f_min=80.0,
            f_max=min(8000.0, sample_rate / 2.0),  # Ensure f_max <= Nyquist
            use_conv_fft=True,
        )

        # Create test signal
        test_input = torch.randn(1, 1024)

        with torch.no_grad():
            mel_spec, fft_mag = transform(test_input)

        # Check basic properties
        assert mel_spec.shape[0] == 1  # batch dimension
        assert fft_mag.shape[0] == 1  # batch dimension
        assert mel_spec.shape[1] == transform.n_mel  # mel dimension
        assert not torch.isnan(mel_spec).any()
        assert not torch.isinf(mel_spec).any()

        # Check that frequency resolution is appropriate for sample rate
        expected_resolution = sample_rate / 1024
        assert abs(transform.fft_resolution - expected_resolution) < 0.1

        # Check that mel frequencies are within reasonable range
        mel_freqs = transform.mel_freqs.cpu().numpy()
        assert mel_freqs[0] >= transform.f_min
        assert mel_freqs[-1] <= transform.f_max
        assert mel_freqs[-1] <= sample_rate / 2.0  # Below Nyquist

    @pytest.mark.parametrize(
        "sample_rate,frame_size",
        [
            (8000, 512),
            (16000, 1024),
            (22050, 1024),
            (44100, 2048),
            (48000, 2048),
        ],
    )
    def test_sample_rate_frame_size_combinations(self, sample_rate, frame_size):
        """Test various combinations of sample rate and frame size."""
        transform = MusicalMelTransform(
            sample_rate=sample_rate,
            frame_size=frame_size,
            f_max=min(8000.0, sample_rate / 2.0),
        )

        # Create test signal with the correct frame size
        test_input = torch.randn(2, frame_size)  # Batch of 2

        with torch.no_grad():
            mel_spec, fft_mag = transform(test_input)

        # Verify output shapes
        assert mel_spec.shape[0] == 2  # batch dimension
        assert fft_mag.shape[0] == 2  # batch dimension
        assert fft_mag.shape[1] == frame_size // 2 + 1  # frequency dimension

        # Verify no numerical issues
        assert not torch.isnan(mel_spec).any()
        assert not torch.isinf(mel_spec).any()
        assert not torch.isnan(fft_mag).any()
        assert not torch.isinf(fft_mag).any()
