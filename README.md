# 3D-Corpus

3D-Corpus is a feature extraction and processing pipeline for audio datasets, designed for visualization in 3D space.

## Main Features

- Audio file loading and preprocessing
- Onset detection-based audio segmentation
- MFCC, Spectral Centroid, and Chroma feature extraction
- GPU acceleration support (Apple Silicon MPS backend and CUDA)
- Asynchronous processing and optimized batch processing

## System Requirements

- Python 3.10 or higher
- Apple Silicon Mac (M1/M2/M3) or CUDA-compatible GPU
- 32GB RAM recommended

## Installation

1. Clone the repository:
```bash
git clone https://github.com/unohee/3d-corpus.git
cd 3d-corpus
```

2. Create and activate a virtual environment:
```bash
python -m venv research_env
source research_env/bin/activate  # macOS/Linux
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Feature Extraction (CPU Version)

```bash
python featureExtractor.py ./path/to/audio/folder
```

### GPU-Accelerated Feature Extraction (MPS/CUDA Version)

```bash
python featureExtractor_torch.py ./path/to/audio/folder
```

### Interactive TUI Interface

To use the text-based user interface for selecting datasets:

```bash
python curses_interface.py
```

### Command-line Options

You can also run the pipeline with various options:

```bash
python run.py ./path/to/audio/folder [options]
```

Available options:
- `--download-only`: Only download the FSD50K dataset without extracting features
- `--no-onset`: Disable onset detection and extract features for entire audio files
- `--save-splits`: Save onset-split audio files to disk
- `--output-dir DIR`: Specify the directory to save split audio files (default: splitted_files)

## Implemented Feature Extraction

1. **MFCC (Mel-Frequency Cepstral Coefficients)**
   - 13 MFCC coefficients
   - 40 mel filter banks
   - 256 frame size, 256 hop length

2. **Spectral Centroid**
   - Center frequency of the spectrum
   - 256 frame size, 256 hop length

3. **Chroma Features**
   - 12 semitone bins
   - 256 frame size, 256 hop length

## Performance Optimizations

- Asynchronous I/O processing
- Batch processing optimization
- GPU memory management
- Transformer caching
- Vectorized operations
- Multi-processing and multi-threading

## Code Structure

The codebase is organized into several well-documented Python modules:

- `featureExtractor_torch.py`: GPU-accelerated feature extraction with PyTorch
- `featureExtractor.py`: CPU-based feature extraction with librosa
- `curses_interface.py`: Text-based user interface for dataset selection
- `run.py`: Command-line interface with FSD50K dataset download capabilities

All functions include detailed docstrings with parameter descriptions and return value information.

## Dataset Structure

```
dataset/
├── [dataset_name].pkl          # Original audio buffers
└── [dataset_name]_features.pkl # Extracted features
```

## Feature Normalization

The extracted features are normalized to consistent 1D array formats:
- MFCC: Multi-dimensional to 1D array (dimension reduction)
- Spectral Centroid: Array to scalar value
- Chroma: Multi-dimensional to 1D array

## Reference Datasets

- FSD50K dataset: [https://zenodo.org/record/4060432](https://zenodo.org/record/4060432)

## License

MIT License

## References

- librosa: [https://librosa.org/](https://librosa.org/)
- torchaudio: [https://pytorch.org/audio/](https://pytorch.org/audio/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/) 