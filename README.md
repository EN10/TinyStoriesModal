# TinyStories Language Model

This repository contains an implementation of a small GPT-style language model trained on the TinyStories dataset using Modal for cloud-based training.

## Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture & Configuration](#model-architecture-and-configuration)
- [Files and Dependencies](#files-and-dependencies)
  - [Core Files](#core-files)
  - [Data Files](#data-files-automatically-downloaded)
  - [For Inference](#for-inference)
  - [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Example Usage](#example-usage)
  - [Command Line Options](#command-line-options)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The project implements a transformer-based language model that can:
- Train on the TinyStories dataset (a collection of simple stories)
- Generate new text in a similar style
- Run efficiently in the cloud using Modal's GPU infrastructure

## Model Architecture & Configuration

The model is a small GPT-style transformer with these default parameters:
- Vocabulary size: 105 tokens
- Embedding dimension: 128
- Number of layers: 5
- Number of attention heads: 8
- Number of key/value heads: 4
- Context length: 256 tokens
- Batch size: 32
- Total parameters: ~1M

These parameters can be modified in the configuration:
```python
# Model and training parameters
vocab_size = 105
dim = 128
n_layers = 5
n_heads = 8
n_kv_heads = 4
batch_size = 32
```

## Files and Dependencies

### Core Files
- `tinystories_modular.py` - Main implementation with modular design for training and inference
- `tinystories_data.py` - Data downloading and preprocessing utilities
- `tinystories_trainer_v1.py` - Alternative training implementation with more detailed logging
- `volume_cleanup.py` - Utility for cleaning up Modal volume files
- `transfer_model.py` - Utility for transferring models between local and Modal volume
- `tinystories.ipynb` - Jupyter notebook for interactive training and testing

### Data Files (automatically downloaded)
From HuggingFace (`tok105` files):
- `tok105.tar.gz` - Pre-tokenized training data
- `tok105.bin` - Tokenizer model

From llama2.c GitHub:
- `train.py` - Training script
- `model.py` - Model implementation
- `tinystories.py` - Dataset handling
- `tokenizer.py` - Tokenizer implementation
- `export.py` - Model export utilities
- `configurator.py` - Configuration handling
- `run.c` - C inference program

### For Inference
Required files (must exist from previous training):
- `tok105.bin` - Tokenizer model
- `out/model.bin` - Trained model output
- `run` - Compiled C inference program

Note: All files are automatically managed in the Modal volume `tinystories-volume`. The setup function in `tinystories_modular.py` handles downloading and compiling all necessary files.

## Directory Structure

After running, your Modal volume will contain:
```
/data/
  ├── tok105.bin            # Tokenizer model
  ├── tok105.model          # SentencePiece tokenizer model
  ├── tok105.vocab          # Tokenizer vocabulary
  ├── tok105.tar.gz         # Training data archive
  ├── tok105/              # Extracted training data
  │   └── data*.bin        # Individual training files
  ├── run.c                # C inference program source
  ├── run                  # Compiled inference program
  ├── model.py             # Model implementation
  ├── train.py             # Training script
  ├── tinystories.py       # Dataset handling
  ├── export.py            # Model export utilities
  ├── configurator.py      # Configuration handling
  ├── tokenizer.py         # Tokenizer implementation
  ├── train.pt             # Processed training data
  ├── val.pt               # Processed validation data
  ├── checkpoint.pt        # Latest training checkpoint
  ├── model_best.pt        # Best model during training
  └── out/                 # Training outputs
      └── model.bin        # Trained model
```

Note: The directory structure is automatically managed by the setup function in `tinystories_modular.py`. Files are downloaded and organized as needed.

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