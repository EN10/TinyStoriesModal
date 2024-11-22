# TinyStories Language Model

This repository contains an implementation of a small GPT-style language model trained on the TinyStories dataset using Modal for cloud-based training.

## Overview

The project implements a transformer-based language model that can:
- Train on the TinyStories dataset (a collection of simple stories)
- Generate new text in a similar style
- Run efficiently in the cloud using Modal's GPU infrastructure

## Files

- `tinystories.py`: Main implementation using Modal for cloud training
- No requirements.txt file is needed since dependencies are handled by Modal image

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
modal run tinystories.py
```

Or run with --detach for long-running training sessions that continue after disconnection:
```
modal run --detach tinystories.py
```

Or run inference:
```
modal run tinystories.py inference "Once upon a time"
```

## Model Architecture

The model is a small GPT-style transformer with:
- Vocabulary size: 105 tokens
- Embedding dimension: 128
- Number of layers: 5
- Number of attention heads: 8
- Context length: 256 tokens
- Total parameters: ~1M

## Training Data

The model uses the TinyStories dataset, specifically the tok105 tokenized version from Hugging Face:
- Dataset: [enio/TinyStories](https://huggingface.co/datasets/enio/TinyStories)
- Pre-tokenized with a vocabulary of 105 tokens
- Data files are automatically downloaded during training/inference
- Includes train.pt and val.pt for training and validation

## Configuration

Key hyperparameters can be adjusted in the `MODEL_CONFIG` dictionary in `tinystories.py`:

```python
MODEL_CONFIG = {
    'vocab_size': 105,
    'dim': 128,
    'n_layer': 5,
    'n_head': 8,
    'block_size': 256,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'max_iters': 5000,
    'eval_interval': 500,
    'eval_iters': 200,
}
```

## Features

- **Efficient Training**: Uses PyTorch's native MultiheadAttention for optimal performance
- **Cloud Support**: Runs on Modal's T4 GPU infrastructure with 1-hour timeout
- **Pre-tokenized Data**: Uses SentencePiece tokenizer with a small, efficient vocabulary
- **Progress Monitoring**: Regular evaluation of training and validation loss
- **Text Generation**: Dedicated inference function for text generation
- **Automatic Data Management**: Downloads and manages required dataset files

## Example Usage

The code provides two main modes of operation:

1. Training:
```bash
modal run tinystories.py
```

2. Inference:
```bash
modal run tinystories.py inference "Once upon a time"
```

The inference command accepts an optional prompt to start the story generation.

## Performance

Typical training metrics:
- Training time: ~1 hour on T4 GPU
- Final validation loss: ~2.5-3.0
- Memory usage: ~2GB GPU memory

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