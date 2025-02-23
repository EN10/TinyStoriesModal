# This application is designed to run in a Modal environment where training and inference are handled remotely.

import os
from pathlib import Path
import wget
import modal
import torch
import sys

# Add /data to the Python path
sys.path.append('/data')

# Initialize Modal app and volume
app = modal.App("tinystories-modded-nanogpt")
volume = modal.Volume.from_name("modded-nanogpt-volume")

# Create image with required packages and tools
image = (modal.Image.debian_slim()
         .pip_install("torch", "huggingface-hub", "tokenizers", "tqdm", "wget", "numpy")
         .run_commands("apt-get update", "apt-get install -y wget"))

# Define files to download with their source URLs
FILES = {
    "train_gpt.py": "https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/refs/heads/master/train_gpt.py",
}

TINYSTORIES_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz",
    "validation": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
}

# Set default RANK if not set
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"

# Set default WORLD_SIZE if not set
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "8"  # Set to 8 for distributed training

if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"  # Default to 0 for single GPU training

if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"  # Set master address for distributed training
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12345"  # Set master port for distributed training

class Hyperparameters:
    # data
    train_files = "data/tinystories/train_0.bin"  # input .bin to train on
    val_files = "data/tinystories/val_0.bin"  # input .bin to eval validation loss on
    val_tokens = 10485760  # tokens for validation
    train_seq_len = 1024  # sequence length during training
    val_seq_len = 2048  # sequence length during validation
    # optimization
    num_iterations = 1  # number of iterations to run
    cooldown_frac = 0.4  # fraction of training spent cooling down the learning rate
    # architecture
    vocab_size = 50257
    num_layers = 4  # Reduced from 6
    num_heads = 4  # Reduced from 8
    model_dim = 128  # Reduced from 512
    max_seq_len = 1024
    # evaluation and logging
    val_loss_every = 50  # evaluate val loss every N steps
    save_checkpoint = True

@app.function(image=image, volumes={"/data": volume})
def setup_data():
    from tokenizers import ByteLevelBPETokenizer
    
    # Download modded-nanogpt files
    for file, url in FILES.items():
        if not os.path.exists(f"/data/{file}"):
            print(f"\nDownloading {file}...")
            wget.download(url, f"/data/{file}")
            print()

    # Create data directories
    data_dir = Path("/data/data/tinystories")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and process TinyStories dataset
    if not (data_dir / "train_0.bin").exists():
        print("\nDownloading and processing TinyStories dataset...")
        
        # Download training data
        train_archive = data_dir / "train.tar.gz"
        if not train_archive.exists():
            wget.download(TINYSTORIES_URLS["train"], str(train_archive))
            os.system(f"cd {data_dir} && tar xzf train.tar.gz")
        
        # Download validation data
        val_file = data_dir / "validation.txt"
        if not val_file.exists():
            wget.download(TINYSTORIES_URLS["validation"], str(val_file))
        
        # Initialize tokenizer
        tokenizer = ByteLevelBPETokenizer()
        
        # Collect text files for tokenizer training
        text_files = list(data_dir.glob("*.txt"))

        # Train the tokenizer with the collected text files
        if text_files:
            tokenizer.train(
                files=[str(file) for file in text_files],
                vocab_size=Hyperparameters.vocab_size,
                min_frequency=2,
                special_tokens=["<|endoftext|>"]
            )
        else:
            print("No text files found for training the tokenizer.")
        
        tokenizer.save_model(str(data_dir))
        
        # Process and save training data
        def process_file(input_file: Path, output_file: Path):
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Tokenize and convert to tensor
            encoded = tokenizer.encode(text).ids
            data = torch.tensor(encoded, dtype=torch.uint16)
            
            # Save tensor using torch.save
            torch.save(data, str(output_file))
        
        # Process training files
        for i, txt_file in enumerate(data_dir.glob("*.txt")):
            if txt_file.name != "validation.txt":
                process_file(txt_file, data_dir / f"train_{i}.bin")
        
        # Process validation file
        process_file(val_file, data_dir / "val_0.bin")
        
        print("Dataset processing completed!")

    # Check if train_gpt.py was downloaded successfully
    if not os.path.exists('/data/train_gpt.py'):
        print("Error: train_gpt.py was not downloaded successfully.")
    else:
        print("train_gpt.py downloaded successfully.")

    # List contents of the /data directory
    print("Contents of /data directory:")
    print(os.listdir('/data'))

@app.function(image=image, volumes={"/data": volume}, gpu="T4")
def train():
    import train_gpt
    from train_gpt import GPT, Muon, distributed_data_generator
    import torch.nn.functional as F
    from torch.nn.attention.flex_attention import BlockMask
    
    os.chdir("/data")
    
    # Initialize model and hyperparameters
    hparams = Hyperparameters()
    model = GPT(
        vocab_size=hparams.vocab_size,
        num_layers=hparams.num_layers,
        num_heads=hparams.num_heads,
        model_dim=hparams.model_dim,
        max_seq_len=hparams.max_seq_len
    ).cuda()
    
    # Initialize optimizer
    optimizer = Muon(model.parameters())
    
    # Training loop
    print("Starting training...")
    
    # Initialize data generator
    train_gen = distributed_data_generator(
        hparams.train_files,
        batch_size=32,  # Adjust based on GPU memory
        rank=0,
        world_size=1
    )
    
    # Training loop
    model.train()
    for iter_num in range(hparams.num_iterations):
        # Get batch
        input_seq = next(train_gen).cuda()
        target_seq = input_seq.clone()
        
        # Calculate sliding window size
        sliding_window_num_blocks = torch.tensor([4], device='cuda')  # Adjust based on sequence length
        
        # Forward pass
        loss = model(input_seq, target_seq, sliding_window_num_blocks)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log progress
        if iter_num % 10 == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}")
        
        # Validation
        if iter_num % hparams.val_loss_every == 0:
            model.eval()
            with torch.no_grad():
                val_gen = distributed_data_generator(
                    hparams.val_files,
                    batch_size=1,
                    rank=0,
                    world_size=1
                )
                val_loss = 0
                for _ in range(5):  # Average over 5 validation batches
                    val_seq = next(val_gen).cuda()
                    val_loss += model(val_seq, val_seq, sliding_window_num_blocks).item()
                val_loss /= 5
                print(f"Validation loss: {val_loss:.4f}")
            model.train()
        
        # Save checkpoint
        if hparams.save_checkpoint and iter_num % 100 == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
            }
            torch.save(checkpoint, "checkpoint.pt")
    
    print("Training completed!")

@app.function(image=image, volumes={"/data": volume}, gpu="T4")
def inference(prompt="Once upon a time", max_tokens=100):
    import train_gpt
    from train_gpt import GPT
    from tokenizers import ByteLevelBPETokenizer
    import torch.nn.functional as F
    
    os.chdir("/data")
    
    # Load tokenizer
    tokenizer = ByteLevelBPETokenizer(
        str(Path("/data/data/tinystories/vocab.json")),
        str(Path("/data/data/tinystories/merges.txt"))
    )
    
    # Load model
    model = GPT(
        vocab_size=Hyperparameters.vocab_size,
        num_layers=Hyperparameters.num_layers,
        num_heads=Hyperparameters.num_heads,
        model_dim=Hyperparameters.model_dim,
        max_seq_len=Hyperparameters.max_seq_len
    ).cuda()
    
    # Load checkpoint
    checkpoint = torch.load("checkpoint.pt")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Tokenize prompt
    input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).cuda()
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass with sliding window
            sliding_window_num_blocks = torch.tensor([4], device='cuda')
            logits = model(input_ids, input_ids, sliding_window_num_blocks)
            
            # Sample next token
            next_token = torch.multinomial(
                F.softmax(logits[:, -1, :], dim=-1),
                num_samples=1
            )
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for end of text token
            if next_token.item() == tokenizer.token_to_id("<|endoftext|>"):
                break
    
    # Decode and print generated text
    generated_text = tokenizer.decode(input_ids[0].cpu().tolist())
    print(f"Generated text:\n{generated_text}")
    return generated_text

@app.local_entrypoint()
def main(command="train", prompt="Once upon a time"):
    if command == "train":
        setup_data.remote()
        train.remote()
    elif command == "inference":
        inference.remote(prompt)
    else:
        print("Invalid command. Use 'train' or 'inference'")
