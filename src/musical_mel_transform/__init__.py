"""Musical Mel Transform - A PyTorch-based musical mel-frequency transform for audio processing.

This package provides efficient, ONNX-compatible implementations of mel-frequency transforms
optimized for musical audio analysis and processing.
"""

__version__ = "0.1.1"
__author__ = "Will Drevo"
__email__ = "will.drevo+github@gmail.com"

from .conv_fft import ConvFFT
from .musical_mel import MusicalMelTransform, PostTransformType, convert_to_onnx, plot_low_filters

__all__ = [
    "ConvFFT",
    "MusicalMelTransform",
    "PostTransformType",
    "convert_to_onnx",
    "plot_low_filters",
]
