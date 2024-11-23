# Simple TinyStories Training

A simplified script for training and running the TinyStories language model using Modal.

## Setup

1. Install Modal:
```bash
pip install modal
```

2. Set up Modal:
```bash
modal token new
```

3. Create volume:
```bash
modal volume create tinystories-volume
```

## Usage

### Training
To train the model:
```bash
modal run simple.py --command train
```

This will:
1. Download required files:
   - tok105.tar.gz (training data)
   - tok105.bin (tokenizer)
   - Training scripts from llama2.c
2. Extract training data
3. Train the model with:
   - Vocabulary size: 105
   - Model dimension: 128
   - 5 layers, 8 heads
   - Batch size: 32
   - 10 training iterations

### Inference
To generate text:
```bash
modal run simple.py --command inference --prompt "Once upon a time"
```

This will:
1. Compile the inference program (run.c)
2. Generate text using:
   - Temperature: 0.8
   - Max tokens: 256
   - Your provided prompt

## Files Used

1. **Training Data**
   - tok105.tar.gz: Pre-tokenized training stories
   - tok105.bin: Binary tokenizer file

2. **Scripts from llama2.c**
   - train.py: Training script
   - model.py: Model definition
   - configurator.py: Configuration handling
   - export.py: Model export utilities
   - tokenizer.py: Tokenizer implementation
   - run.c: Inference program

## Directory Structure

After running, your `/data` directory will contain:
```
/data/
  ├── tok105.bin
  ├── tok105.tar.gz
  ├── run.c
  ├── run (compiled)
  ├── train.py
  ├── model.py
  ├── configurator.py
  ├── export.py
  ├── tokenizer.py
  ├── tinystories.py
  ├── out/
  │   └── model.bin
  ├── data/
  │   └── tok105/ -> ../tok105/
  └── tok105/
      └── (training files)
```

## Requirements
- modal
- A Modal account and token
- GPU access on Modal (T4 used for training)

All other dependencies (torch, numpy, etc.) are automatically installed in the Modal image. 