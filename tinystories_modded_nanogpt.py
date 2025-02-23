# This application is designed to run in a Modal environment where training and inference are handled remotely.

import os
from pathlib import Path
import wget
import modal
import torch
import sys
import time
from datetime import datetime

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

@app.function(image=image, volumes={"/data": volume}, gpu="T4", timeout=3600)  # Increased timeout to 1 hour
def train():
    print(f"\n=== Starting training setup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Check if train_gpt.py exists
    train_gpt_path = '/data/train_gpt.py'
    print(f"Checking for train_gpt.py at {train_gpt_path}...")
    if os.path.exists(train_gpt_path):
        print("train_gpt.py found.")
    else:
        print("ERROR: train_gpt.py not found!")
        raise FileNotFoundError("train_gpt.py not found at expected path.")
    
    print("Importing required modules...")
    try:
        import train_gpt
        from train_gpt import GPT, Muon, distributed_data_generator
        import torch.nn.functional as F
        from torch.nn.attention.flex_attention import BlockMask
        print("Successfully imported all modules")
    except Exception as e:
        print(f"Error importing modules: {str(e)}")
        raise
    
    print("\nChanging to /data directory...")
    os.chdir("/data")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of current directory: {os.listdir('.')}")
    
    # Initialize model and hyperparameters
    print("\nInitializing hyperparameters...")
    try:
        hparams = Hyperparameters()
        print("Successfully created hyperparameters")
        print(f"Training files path: {hparams.train_files}")
        print(f"Validation files path: {hparams.val_files}")
        
        # Check if training files exist
        print("\nChecking data files...")
        if os.path.exists(hparams.train_files):
            print(f"Training file exists at {hparams.train_files}")
        else:
            print(f"WARNING: Training file not found at {hparams.train_files}")
        
        if os.path.exists(hparams.val_files):
            print(f"Validation file exists at {hparams.val_files}")
        else:
            print(f"WARNING: Validation file not found at {hparams.val_files}")
    except Exception as e:
        print(f"Error initializing hyperparameters: {str(e)}")
        raise

    print("\nInitializing model...")
    try:
        print("Creating GPU model with parameters:")
        print(f"- vocab_size: {hparams.vocab_size}")
        print(f"- num_layers: {hparams.num_layers}")
        print(f"- num_heads: {hparams.num_heads}")
        print(f"- model_dim: {hparams.model_dim}")
        print(f"- max_seq_len: {hparams.max_seq_len}")
        
        model = GPT(
            vocab_size=hparams.vocab_size,
            num_layers=hparams.num_layers,
            num_heads=hparams.num_heads,
            model_dim=hparams.model_dim,
            max_seq_len=hparams.max_seq_len
        ).cuda()
        print("Successfully created and moved model to GPU")
        
        # Print CUDA memory usage
        if torch.cuda.is_available():
            print(f"\nCUDA Memory Summary:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

    print("\nInitializing optimizer...")
    try:
        optimizer = Muon(model.parameters())
        print("Successfully created optimizer")
    except Exception as e:
        print(f"Error initializing optimizer: {str(e)}")
        raise

    print("\nInitializing data generator...")
    try:
        train_gen = distributed_data_generator(
            hparams.train_files,
            batch_size=32,
            rank=0,
            world_size=1
        )
        print("Successfully created data generator")
        
        # Try to get first batch to verify data loading works
        print("\nTesting data generator with first batch...")
        input_seq = next(train_gen)
        print(f"Successfully loaded first batch with shape: {input_seq.shape}")
    except Exception as e:
        print(f"Error with data generator: {str(e)}")
        raise

    print("\n=== Starting training loop ===")
    
    # Training metrics
    start_time = time.time()
    running_loss = 0.0
    best_val_loss = float('inf')
    
    # Training loop
    model.train()
    try:
        for iter_num in range(hparams.num_iterations):
            batch_start_time = time.time()
            
            # Get batch
            print(f"\nIteration {iter_num + 1}/{hparams.num_iterations}")
            print("Loading batch...", end=" ")
            input_seq = next(train_gen).cuda()
            target_seq = input_seq.clone()
            print(f"Batch shape: {input_seq.shape}")
            
            # Calculate sliding window size
            sliding_window_num_blocks = torch.tensor([4], device='cuda')
            
            # Forward pass
            print("Forward pass...", end=" ")
            loss = model(input_seq, target_seq, sliding_window_num_blocks)
            running_loss += loss.item()
            
            # Backward pass and optimize
            print("Backward pass...", end=" ")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time = time.time() - batch_start_time
            
            # Log progress
            if iter_num % 10 == 0:
                avg_loss = running_loss / (iter_num + 1)
                elapsed_time = time.time() - start_time
                print(f"\nStats:")
                print(f"- Iteration: {iter_num}/{hparams.num_iterations}")
                print(f"- Loss: {loss.item():.4f}")
                print(f"- Average loss: {avg_loss:.4f}")
                print(f"- Batch time: {batch_time:.2f}s")
                print(f"- Total training time: {elapsed_time:.2f}s")
            
            # Validation
            if iter_num % hparams.val_loss_every == 0:
                print("\nRunning validation...")
                model.eval()
                with torch.no_grad():
                    val_gen = distributed_data_generator(
                        hparams.val_files,
                        batch_size=1,
                        rank=0,
                        world_size=1
                    )
                    val_loss = 0
                    for val_batch in range(5):
                        print(f"Validation batch {val_batch + 1}/5...", end=" ")
                        val_seq = next(val_gen).cuda()
                        val_loss += model(val_seq, val_seq, sliding_window_num_blocks).item()
                    val_loss /= 5
                    print(f"\nValidation loss: {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"New best validation loss: {best_val_loss:.4f}")
                model.train()
            
            # Save checkpoint
            if hparams.save_checkpoint and iter_num % 100 == 0:
                print("\nSaving checkpoint...")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'training_time': time.time() - start_time
                }
                torch.save(checkpoint, "checkpoint.pt")
                print("Checkpoint saved successfully")
    
        total_time = time.time() - start_time
        print(f"\n=== Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Final loss: {loss.item():.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

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
