# TinyStories Language Model

## Contents
- [Overview](#overview)
- [Files and Dependencies](#files-and-dependencies)
  - [Core Files](#core-files)
  - [Data Files](#data-files-automatically-downloaded)
  - [For Inference](#for-inference)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Features](#features)
- [Example Usage](#example-usage)
- [Command Line Options](#command-line-options)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

This repository contains an implementation of a small GPT-style language model trained on the TinyStories dataset using Modal for cloud-based training.

## Overview

The project implements a transformer-based language model that can:
- Train on the TinyStories dataset (a collection of simple stories)
- Generate new text in a similar style
- Run efficiently in the cloud using Modal's GPU infrastructure

## Files and Dependencies

### Core Files
- `tinystories_modular.py` - Main implementation with modular design for training and inference
- `tinystories_data.py` - Data downloading and preprocessing utilities
- `volume_cleanup.py` - Utility for cleaning up Modal volume files
- `transfer_model.py` - Utility for transferring models between local and Modal volume

### Data Files (automatically downloaded)
- From HuggingFace:
  - `tok105.tar.gz` - Pre-tokenized training data
  - `tok105.bin` - Tokenizer model
- From llama2.c GitHub:
  - `model.py` - Model implementation
  - `export.py` - Model export utilities
  - `configurator.py` - Configuration handling
  - `tokenizer.py` - Tokenizer implementation
  - `run.c` - C inference program (compiled automatically)

### For Inference
Required files (must exist from previous training):
- `tok105.bin` - Tokenizer model
- `out/model.bin` - Trained model output
- `run` - Compiled C inference program

Note: All files are automatically managed in the Modal volume `tinystories-volume`. The setup function in `tinystories_modular.py` handles downloading and compiling all necessary files.

## Requirements

- modal
- A Modal account and token

Note: Other dependencies (torch, numpy, sentencepiece) are automatically installed in the Modal image.

## Getting Started

1. Install Modal:
```
pip install modal
```

2. Set up Modal:
```
modal token new
```

3. Create a Modal volume:
```
modal volume create tinystories-volume
```

4. Run the training:
```
modal run tinystories_modular.py
```

Or run inference with a custom prompt:
```
modal run tinystories_modular.py --command inference --prompt "Once upon a time"
```

## Model Architecture

The model is a small GPT-style transformer with:
- Vocabulary size: 105 tokens
- Embedding dimension: 128
- Number of layers: 5
- Number of attention heads: 8
- Context length: 256 tokens
- Total parameters: ~1M

## Configuration

The model uses these default parameters:
```python
# Training parameters
vocab_size = 105
dim = 128
n_layers = 5
n_heads = 8
n_kv_heads = 4
batch_size = 32
```

## Features

- **Modular Design**: Separate functions for setup, training, and inference
- **Automatic Setup**: Downloads and prepares all required files
- **Progress Monitoring**: Shows download and extraction progress
- **GPU Support**: Runs on Modal's T4 GPU infrastructure
- **Simple Interface**: Easy-to-use command line interface
- **Efficient Training**: Uses PyTorch for optimal performance

## Example Usage

The code provides three main modes of operation:

1. Training:
```bash
modal run tinystories_modular.py --command train
```

2. Inference:
```bash
modal run tinystories_modular.py --command inference --prompt "Once upon a time"
```

3. Model Transfer:
```bash
# Download model from Modal volume to local
modal run transfer_model.py --action download --path out/model.bin

# Upload model from local to Modal volume
modal run transfer_model.py --action upload --path out/model.bin
```

## Command Line Options

### Main Script (tinystories_modular.py)
- `--command`: Either "train" or "inference" (default: "train")
- `--prompt`: Initial text for story generation when using inference (default: "Once upon a time")

### Transfer Script (transfer_model.py)
- `--action`: Either "download" or "upload" (default: "download")
- `--path`: Local path for model file (default: "out/model.bin")

## Directory Structure

After running, your Modal volume will contain:
```
/data/
  ├── tok105.bin            # Tokenizer model
  ├── tok105.tar.gz         # Training data archive
  ├── tok105/              # Extracted training data
  │   └── data*.bin        # Individual training files
  ├── run.c                # C inference program source
  ├── run                  # Compiled inference program
  ├── model.py             # Model implementation
  ├── export.py            # Model export utilities
  ├── configurator.py      # Configuration handling
  ├── tokenizer.py         # Tokenizer implementation
  └── out/                 # Training outputs
      └── model.bin        # Trained model
```

Note: The directory structure is automatically managed by the setup function in `tinystories_modular.py`. Files are downloaded and organized as needed.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License

## Acknowledgments

- Based on [Andrej Karpathy's](https://github.com/karpathy/llama2.c) implementation
- Uses the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- Cloud implementation powered by [Modal](https://modal.com/)