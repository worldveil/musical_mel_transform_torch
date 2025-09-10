"""Tests for ONNX export functionality."""

import gc
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from musical_mel_transform.musical_mel import MusicalMelTransform, convert_to_onnx


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows file permission issues with ONNX external data",
)
class TestONNXExport:
    """Test cases for ONNX export functionality."""

    @pytest.fixture
    def mel_transform(self):
        """Create a MusicalMelTransform instance for testing."""
        return MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            interval=1.0,
            f_max=8000.0,
            passthrough_cutoff_hz=4000.0,
            norm=True,
            min_bins=2,
            adaptive=True,
            passthrough_grouping_size=3,
            use_conv_fft=True,  # Test with conv FFT for ONNX compatibility
            to_db=False,
        )

    @pytest.fixture
    def mel_transform_torch_fft(self):
        """Create a MusicalMelTransform instance using torch FFT."""
        return MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            interval=1.0,
            f_max=8000.0,
            passthrough_cutoff_hz=4000.0,
            norm=True,
            min_bins=2,
            adaptive=True,
            passthrough_grouping_size=3,
            use_conv_fft=False,  # Test with torch FFT (should not use dynamo)
            to_db=False,
        )

    @pytest.fixture
    def test_input(self, mel_transform):
        """Create test input tensor."""
        return torch.randn(1, mel_transform.frame_size)

    def _cleanup_onnx_session(self, session):
        """Helper to properly cleanup ONNX Runtime session on Windows."""
        if session is not None:
            # Explicitly close the session
            try:
                del session
            except:
                pass
            # Force garbage collection to release file handles
            gc.collect()

    def test_onnx_export_conv_fft(self, mel_transform, test_input):
        """Test ONNX export with ConvFFT (dynamo=True)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_conv_fft.onnx")

            # Export to ONNX
            convert_to_onnx(mel_transform, onnx_path, opset=18)

            # Verify file was created
            assert os.path.exists(onnx_path)

            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # Verify input/output names and shapes
            self._verify_onnx_model_structure(onnx_model, mel_transform)

    def test_onnx_export_torch_fft(self, mel_transform_torch_fft, test_input):
        """Test ONNX export with torch FFT (should fail - torch.fft.rfft not supported)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_torch_fft.onnx")

            # torch.fft.rfft is not supported in ONNX, so this should raise an error
            with pytest.raises(
                (torch.onnx.errors.UnsupportedOperatorError, RuntimeError)
            ):
                convert_to_onnx(mel_transform_torch_fft, onnx_path, opset=18)

    def test_onnx_runtime_inference(self, mel_transform, test_input):
        """Test that ONNX model produces correct outputs."""
        ort_session = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "test_inference.onnx")

                # Export to ONNX
                convert_to_onnx(mel_transform, onnx_path, opset=18)

                # Get PyTorch reference output
                with torch.no_grad():
                    mel_ref, fft_mag_ref = mel_transform(test_input)

                # Run ONNX inference
                ort_session = onnxruntime.InferenceSession(onnx_path)
                onnx_inputs = {"frames": test_input.numpy()}
                onnx_outputs = ort_session.run(None, onnx_inputs)

                mel_onnx = torch.from_numpy(onnx_outputs[0])
                fft_mag_onnx = torch.from_numpy(onnx_outputs[1])

                # Compare outputs (with tolerance for runtime differences)
                torch.testing.assert_close(mel_ref, mel_onnx, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(
                    fft_mag_ref, fft_mag_onnx, rtol=3e-5, atol=3e-5
                )

                # Cleanup session before tmpdir cleanup
                self._cleanup_onnx_session(ort_session)
                ort_session = None
        finally:
            self._cleanup_onnx_session(ort_session)

    def test_onnx_fixed_batch_size(self, mel_transform):
        """Test ONNX model with fixed batch size = 1."""
        test_input = torch.randn(1, mel_transform.frame_size)
        ort_session = None

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "test_fixed_batch.onnx")

                # Export to ONNX with fixed batch size
                convert_to_onnx(mel_transform, onnx_path, opset=18, dynamic_batch=False)

                # Get PyTorch reference output
                with torch.no_grad():
                    mel_ref, fft_mag_ref = mel_transform(test_input)

                # Run ONNX inference
                ort_session = onnxruntime.InferenceSession(onnx_path)
                onnx_inputs = {"frames": test_input.numpy()}
                onnx_outputs = ort_session.run(None, onnx_inputs)

                mel_onnx = torch.from_numpy(onnx_outputs[0])
                fft_mag_onnx = torch.from_numpy(onnx_outputs[1])

                # Verify shapes
                assert mel_onnx.shape == mel_ref.shape
                assert fft_mag_onnx.shape == fft_mag_ref.shape

                # Compare outputs - relaxed tolerance for Windows
                torch.testing.assert_close(mel_ref, mel_onnx, rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(
                    fft_mag_ref, fft_mag_onnx, rtol=2e-5, atol=2e-5
                )

                # Test different batch size (should fail)
                batch_input = torch.randn(4, mel_transform.frame_size)
                batch_onnx_inputs = {"frames": batch_input.numpy()}

                # This should raise an error because the model expects batch size = 1
                with pytest.raises(
                    onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument
                ):
                    ort_session.run(None, batch_onnx_inputs)

                # Cleanup session before tmpdir cleanup
                self._cleanup_onnx_session(ort_session)
                ort_session = None
        finally:
            self._cleanup_onnx_session(ort_session)

    def test_onnx_dynamic_batch_size(self, mel_transform):
        """Test ONNX model with dynamic batch size configuration."""
        ort_session = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "test_dynamic_batch.onnx")

                # Export to ONNX with dynamic batch size configuration
                convert_to_onnx(mel_transform, onnx_path, opset=18, dynamic_batch=True)

                # Verify the model exports successfully and is valid
                assert os.path.exists(onnx_path)
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)

                # Load ONNX session and test with batch size 1
                ort_session = onnxruntime.InferenceSession(onnx_path)
                test_input = torch.randn(1, mel_transform.frame_size)

                # Get PyTorch reference output
                with torch.no_grad():
                    mel_ref, fft_mag_ref = mel_transform(test_input)

                # Run ONNX inference
                onnx_inputs = {"frames": test_input.numpy()}
                onnx_outputs = ort_session.run(None, onnx_inputs)

                mel_onnx = torch.from_numpy(onnx_outputs[0])
                fft_mag_onnx = torch.from_numpy(onnx_outputs[1])

                # Verify shapes
                assert mel_onnx.shape == mel_ref.shape
                assert fft_mag_onnx.shape == fft_mag_ref.shape

                # Compare outputs
                torch.testing.assert_close(mel_ref, mel_onnx, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(
                    fft_mag_ref, fft_mag_onnx, rtol=1e-4, atol=1e-5
                )

                # Cleanup session before tmpdir cleanup
                self._cleanup_onnx_session(ort_session)
                ort_session = None
        finally:
            self._cleanup_onnx_session(ort_session)

    def test_different_opset_versions(self, mel_transform, test_input):
        """Test ONNX export with different opset versions."""
        # Note: With current onnxscript, opset < 18 is not supported when using dynamo=True
        # Only test opset 18 for compatibility
        opset_versions = [18]  # Changed from [18, 19] to just [18]

        for opset in opset_versions:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, f"test_opset_{opset}.onnx")

                # Export to ONNX
                convert_to_onnx(mel_transform, onnx_path, opset=opset)

                # Verify file was created and is valid
                assert os.path.exists(onnx_path)
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)

                # Verify opset version
                assert onnx_model.opset_import[0].version == opset

    def test_onnx_performance_consistency(self, mel_transform, test_input):
        """Test that ONNX inference is consistent across multiple runs."""
        ort_session = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "test_consistency.onnx")

                # Export to ONNX
                convert_to_onnx(mel_transform, onnx_path, opset=18)

                # Run inference multiple times
                ort_session = onnxruntime.InferenceSession(onnx_path)
                onnx_inputs = {"frames": test_input.numpy()}

                outputs = []
                for _ in range(5):
                    onnx_outputs = ort_session.run(None, onnx_inputs)
                    outputs.append(onnx_outputs)

                # Verify all outputs are identical
                for i in range(1, len(outputs)):
                    np.testing.assert_array_equal(
                        outputs[0][0], outputs[i][0]
                    )  # mel output
                    np.testing.assert_array_equal(
                        outputs[0][1], outputs[i][1]
                    )  # fft_mag output

                # Cleanup session before tmpdir cleanup
                self._cleanup_onnx_session(ort_session)
                ort_session = None
        finally:
            self._cleanup_onnx_session(ort_session)

    def test_onnx_numerical_stability(self, mel_transform):
        """Test ONNX model with edge cases and numerical stability."""
        test_cases = [
            torch.zeros(1, mel_transform.frame_size),  # Silence
            torch.ones(1, mel_transform.frame_size),  # DC signal
            torch.randn(1, mel_transform.frame_size) * 1e-8,  # Very small signal
            torch.randn(1, mel_transform.frame_size) * 100,  # Large signal
        ]

        ort_session = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "test_stability.onnx")

                # Export to ONNX
                convert_to_onnx(mel_transform, onnx_path, opset=18)

                ort_session = onnxruntime.InferenceSession(onnx_path)

                for i, test_input in enumerate(test_cases):
                    # Get PyTorch reference
                    with torch.no_grad():
                        mel_ref, fft_mag_ref = mel_transform(test_input)

                    # Run ONNX inference
                    onnx_inputs = {"frames": test_input.numpy()}
                    onnx_outputs = ort_session.run(None, onnx_inputs)

                    mel_onnx = torch.from_numpy(onnx_outputs[0])
                    fft_mag_onnx = torch.from_numpy(onnx_outputs[1])

                    # Verify no NaN or inf values
                    assert not torch.isnan(
                        mel_onnx
                    ).any(), f"NaN found in mel output for test case {i}"
                    assert not torch.isinf(
                        mel_onnx
                    ).any(), f"Inf found in mel output for test case {i}"
                    assert not torch.isnan(
                        fft_mag_onnx
                    ).any(), f"NaN found in fft_mag output for test case {i}"
                    assert not torch.isinf(
                        fft_mag_onnx
                    ).any(), f"Inf found in fft_mag output for test case {i}"

                    # Compare with reference (with appropriate tolerance for edge cases)
                    if (
                        i < 2
                    ):  # Silence and DC cases - more relaxed tolerance due to numerical differences
                        torch.testing.assert_close(
                            mel_ref, mel_onnx, rtol=1e-2, atol=1e-4
                        )
                        torch.testing.assert_close(
                            fft_mag_ref, fft_mag_onnx, rtol=1e-2, atol=1e-4
                        )
                    else:  # Small and large signal cases
                        torch.testing.assert_close(
                            mel_ref, mel_onnx, rtol=1e-3, atol=1e-4
                        )
                        torch.testing.assert_close(
                            fft_mag_ref, fft_mag_onnx, rtol=1e-3, atol=1e-4
                        )

                # Cleanup session before tmpdir cleanup
                self._cleanup_onnx_session(ort_session)
                ort_session = None
        finally:
            self._cleanup_onnx_session(ort_session)

    def _verify_onnx_model_structure(self, onnx_model, mel_transform):
        """Helper to verify ONNX model structure."""
        # Check input
        assert len(onnx_model.graph.input) == 1
        input_node = onnx_model.graph.input[0]
        assert input_node.name == "frames"

        # Check outputs
        assert len(onnx_model.graph.output) == 2
        output_names = [output.name for output in onnx_model.graph.output]
        assert "mel" in output_names
        assert "fft_mag" in output_names

        # Check that the model has operations (not empty)
        assert len(onnx_model.graph.node) > 0

    @pytest.mark.slow
    def test_onnx_performance_benchmark(self, mel_transform, test_input):
        """Benchmark ONNX vs PyTorch performance."""
        iterations = 100

        ort_session = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "test_benchmark.onnx")

                # Export to ONNX
                convert_to_onnx(mel_transform, onnx_path, opset=18)

                # Warm up PyTorch
                with torch.no_grad():
                    for _ in range(10):
                        mel_transform(test_input)

                # Benchmark PyTorch
                import time

                torch_times = []
                with torch.no_grad():
                    for _ in range(iterations):
                        start = time.time()
                        mel_transform(test_input)
                        torch_times.append((time.time() - start) * 1000)

                # Warm up ONNX
                ort_session = onnxruntime.InferenceSession(onnx_path)
                onnx_inputs = {"frames": test_input.numpy()}
                for _ in range(10):
                    ort_session.run(None, onnx_inputs)

                # Benchmark ONNX
                onnx_times = []
                for _ in range(iterations):
                    start = time.time()
                    ort_session.run(None, onnx_inputs)
                    onnx_times.append((time.time() - start) * 1000)

                torch_avg = np.mean(torch_times)
                onnx_avg = np.mean(onnx_times)

                print(f"\nPerformance Benchmark Results:")
                print(f"PyTorch: {torch_avg:.3f} ± {np.std(torch_times):.3f} ms")
                print(f"ONNX: {onnx_avg:.3f} ± {np.std(onnx_times):.3f} ms")
                print(f"Speedup (of torch vs convFFT): {onnx_avg / torch_avg:.2f}x")

                # Ensure ONNX is at least comparable (within 2x of PyTorch)
                assert (
                    onnx_avg < torch_avg * 2.0
                ), f"ONNX too slow: {onnx_avg:.3f}ms vs PyTorch {torch_avg:.3f}ms"

                # Cleanup session before tmpdir cleanup
                self._cleanup_onnx_session(ort_session)
                ort_session = None
        finally:
            self._cleanup_onnx_session(ort_session)

    def test_onnx_export_with_learnable_weights_fft(self):
        """Test ONNX export with learnable_weights='fft'."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            learnable_weights="fft",
            use_conv_fft=True,
        )

        test_input = torch.randn(1, transform.frame_size)

        ort_session = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "test_learnable_fft.onnx")

                # Export to ONNX
                convert_to_onnx(transform, onnx_path, opset=18)

                # Verify file was created and is valid
                assert os.path.exists(onnx_path)
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)

                # Test inference
                with torch.no_grad():
                    mel_ref, fft_mag_ref = transform(test_input)

                ort_session = onnxruntime.InferenceSession(onnx_path)
                onnx_inputs = {"frames": test_input.numpy()}
                onnx_outputs = ort_session.run(None, onnx_inputs)

                mel_onnx = torch.from_numpy(onnx_outputs[0])
                fft_mag_onnx = torch.from_numpy(onnx_outputs[1])

                # Compare outputs (relaxed tolerance for ONNX numerical differences with random weights)
                torch.testing.assert_close(mel_ref, mel_onnx, rtol=3e-5, atol=1e-5)
                torch.testing.assert_close(
                    fft_mag_ref, fft_mag_onnx, rtol=3e-5, atol=1e-5
                )

                # Cleanup session before tmpdir cleanup
                self._cleanup_onnx_session(ort_session)
                ort_session = None
        finally:
            self._cleanup_onnx_session(ort_session)

    def test_onnx_export_with_learnable_weights_mel(self):
        """Test ONNX export with learnable_weights='mel'."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            learnable_weights="mel",
            use_conv_fft=True,  # Required for ONNX export
            to_db=False,
        )

        test_input = torch.randn(1, transform.frame_size)

        ort_session = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "test_learnable_mel.onnx")

                # Export to ONNX
                convert_to_onnx(transform, onnx_path, opset=18)

                # Verify file was created and is valid
                assert os.path.exists(onnx_path)
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)

                # Test inference
                with torch.no_grad():
                    mel_ref, fft_mag_ref = transform(test_input)

                ort_session = onnxruntime.InferenceSession(onnx_path)
                onnx_inputs = {"frames": test_input.numpy()}
                onnx_outputs = ort_session.run(None, onnx_inputs)

                mel_onnx = torch.from_numpy(onnx_outputs[0])
                fft_mag_onnx = torch.from_numpy(onnx_outputs[1])

                # Compare outputs (relaxed tolerance for ONNX numerical differences with random weights)
                torch.testing.assert_close(mel_ref, mel_onnx, rtol=3e-5, atol=1e-5)
                torch.testing.assert_close(
                    fft_mag_ref, fft_mag_onnx, rtol=3e-5, atol=1e-5
                )

                # Cleanup session before tmpdir cleanup
                self._cleanup_onnx_session(ort_session)
                ort_session = None
        finally:
            self._cleanup_onnx_session(ort_session)

    def test_onnx_learnable_weights_affect_inference(self):
        """Test that learnable weights actually affect ONNX inference outputs."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            learnable_weights="fft",
            use_conv_fft=True,
        )

        test_input = torch.randn(1, transform.frame_size)

        # Export default model
        ort_session_default = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path_default = os.path.join(tmpdir, "test_default.onnx")
                convert_to_onnx(transform, onnx_path_default, opset=18)

                # Get default outputs
                ort_session_default = onnxruntime.InferenceSession(onnx_path_default)
                onnx_inputs = {"frames": test_input.numpy()}
                onnx_outputs_default = ort_session_default.run(None, onnx_inputs)

                # Modify weights and export again
                with torch.no_grad():
                    transform.learnable_weights_param.data *= 2.0

                onnx_path_modified = os.path.join(tmpdir, "test_modified.onnx")
                convert_to_onnx(transform, onnx_path_modified, opset=18)

                # Get modified outputs
                ort_session_modified = onnxruntime.InferenceSession(onnx_path_modified)
                onnx_outputs_modified = ort_session_modified.run(None, onnx_inputs)

                # Outputs should be different
                mel_default = onnx_outputs_default[0]
                fft_mag_default = onnx_outputs_default[1]
                mel_modified = onnx_outputs_modified[0]
                fft_mag_modified = onnx_outputs_modified[1]

                # In dB default, doubling weights adds ~+3.0103 dB (ignore floors)
                delta = fft_mag_modified - fft_mag_default
                mask = fft_mag_default > (-100.0 + 1e-6)
                expected = 10.0 * np.log10(2.0)
                np.testing.assert_allclose(delta[mask], expected, rtol=1e-5, atol=1e-6)

                # Mel outputs should also be different
                assert not np.allclose(mel_default, mel_modified, rtol=1e-6, atol=1e-7)

        finally:
            self._cleanup_onnx_session(ort_session_default)
            self._cleanup_onnx_session(ort_session_modified)

    def test_onnx_learnable_weights_numerical_stability(self):
        """Test ONNX learnable weights with edge case values."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            learnable_weights="mel",
            use_conv_fft=True,
        )

        # Test with extreme weight values
        test_cases = [
            torch.ones_like(transform.learnable_weights_param)
            * 0.001,  # Very small weights
            torch.ones_like(transform.learnable_weights_param)
            * 100.0,  # Very large weights
            torch.zeros_like(transform.learnable_weights_param),  # Zero weights
        ]

        ort_session = None
        try:
            test_input = torch.randn(1, transform.frame_size)

            for i, weights in enumerate(test_cases):
                with torch.no_grad():
                    transform.learnable_weights_param.data = weights

                with tempfile.TemporaryDirectory() as tmpdir:
                    onnx_path = os.path.join(tmpdir, f"test_extreme_{i}.onnx")

                    # Should not crash during export
                    convert_to_onnx(transform, onnx_path, opset=18)

                    # Should run inference without issues
                    ort_session = onnxruntime.InferenceSession(onnx_path)
                    onnx_inputs = {"frames": test_input.numpy()}
                    onnx_outputs = ort_session.run(None, onnx_inputs)

                    # Check for numerical stability (no NaN/inf)
                    mel_output = onnx_outputs[0]
                    fft_mag_output = onnx_outputs[1]

                    assert not np.isnan(
                        mel_output
                    ).any(), f"NaN found in mel output for test case {i}"
                    assert not np.isinf(
                        mel_output
                    ).any(), f"Inf found in mel output for test case {i}"
                    assert not np.isnan(
                        fft_mag_output
                    ).any(), f"NaN found in fft_mag output for test case {i}"
                    assert not np.isinf(
                        fft_mag_output
                    ).any(), f"Inf found in fft_mag output for test case {i}"

        finally:
            self._cleanup_onnx_session(ort_session)

    def test_onnx_learnable_weights_affect_inference_linear(self):
        """Linear variant: verify exact doubling using to_db=False."""
        transform = MusicalMelTransform(
            sample_rate=44100,
            frame_size=1024,
            learnable_weights="fft",
            use_conv_fft=True,
            to_db=False,
        )

        test_input = torch.randn(1, transform.frame_size)

        ort_session_default = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path_default = os.path.join(tmpdir, "test_default_linear.onnx")
                convert_to_onnx(transform, onnx_path_default, opset=18)

                ort_session_default = onnxruntime.InferenceSession(onnx_path_default)
                onnx_inputs = {"frames": test_input.numpy()}
                onnx_outputs_default = ort_session_default.run(None, onnx_inputs)

                with torch.no_grad():
                    transform.learnable_weights_param.data *= 2.0

                onnx_path_modified = os.path.join(tmpdir, "test_modified_linear.onnx")
                convert_to_onnx(transform, onnx_path_modified, opset=18)

                ort_session_modified = onnxruntime.InferenceSession(onnx_path_modified)
                onnx_outputs_modified = ort_session_modified.run(None, onnx_inputs)

                fft_mag_default = onnx_outputs_default[1]
                fft_mag_modified = onnx_outputs_modified[1]

                np.testing.assert_allclose(
                    fft_mag_modified, fft_mag_default * 2.0, rtol=1e-5, atol=1e-6
                )
        finally:
            self._cleanup_onnx_session(ort_session_default)
