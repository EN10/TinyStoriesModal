import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from datetime import datetime

# Update Modal stub to App
app = modal.App("tinystories-training")
volume = modal.Volume.from_name("tinystories-volume")
image = (modal.Image.debian_slim()
         .pip_install("torch", "numpy", "sentencepiece")
         .run_commands("apt-get update", "apt-get install -y wget"))

# Model configuration
MODEL_CONFIG = {
    'vocab_size': 105,  # From tok105
    'dim': 128,
    'n_layer': 5,
    'n_head': 8,
    'block_size': 256,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'max_iters': 1000,
    'eval_interval': 200,   # Every ~5-10 minutes. Lower (e.g., 100) for more frequent checkpoints, higher (e.g., 500) for faster training
    'eval_iters': 200,      # 50-200 is typical. Lower (e.g., 50) for faster eval, higher (e.g., 400) for more stable metrics
    'num_train_files': 10,  # Number of training files to use (max 45)
    'val_percent': 10,      # Percentage of training files to use for validation (e.g., 10 = 10%)
}

class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(dim, n_head, dropout=0.1, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, n_layer, n_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, dim),
            'wpe': nn.Embedding(block_size, dim),
            'drop': nn.Dropout(0.1),
            'blocks': nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)]),
            'ln_f': nn.LayerNorm(dim),
        })
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(t, device=idx.device))
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_tar_file_sizes(tar_path):
    """Get file sizes from tar archive"""
    import subprocess
    print("Running tar command to list archive contents...", flush=True)
    result = subprocess.run(['tar', '-tvf', tar_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to read tar contents: {result.stderr}")
    
    print("Parsing archive contents...", flush=True)
    file_sizes = {}
    total_lines = len(result.stdout.splitlines())
    for i, line in enumerate(result.stdout.splitlines(), 1):
        if i % 1000 == 0:  # Log progress every 1000 lines
            print(f"Processed {i:,}/{total_lines:,} entries...", flush=True)
        
        parts = line.split()
        size = int(parts[2])
        filename = parts[-1]
        if filename.startswith('tok105/') and filename.endswith('.bin'):
            file_sizes[os.path.basename(filename)] = size
    
    print(f"Found {len(file_sizes)} data files in archive", flush=True)
    return file_sizes

def download_data():
    os.makedirs("/data", exist_ok=True)
    
    # Check what files we already have
    existing_files = set(os.listdir("/data"))
    print(f"Existing files: {existing_files}")
    
    # Download tokenizer model file if needed
    tokenizer_files = {
        "tok105.model": "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/tok105.model",
    }
    
    for file, url in tokenizer_files.items():
        if file not in existing_files:
            print(f"Downloading {file}...")
            result = os.system(f"cd /data && wget {url}")
            if result != 0:
                raise RuntimeError(f"wget failed with exit code {result} for {file}")
            if not os.path.exists(f"/data/{file}"):
                raise RuntimeError(f"Failed to download {file}")
            print(f"Successfully downloaded {file}")
    
    # Process pre-tokenized data files if needed
    if not all(f in existing_files for f in ["train.pt", "val.pt"]):
        print("Processing pre-tokenized data...")
        
        # Calculate number of validation files based on percentage
        num_train = MODEL_CONFIG['num_train_files']
        num_val = max(1, round(num_train * MODEL_CONFIG['val_percent'] / 100))
        print(f"\nUsing {num_train} files for training and {num_val} files for validation ({MODEL_CONFIG['val_percent']}%)")
        
        required_files = ([f"data{str(i).zfill(2)}.bin" for i in range(num_train)] + 
                         [f"data{str(i).zfill(2)}.bin" for i in range(45, 45 + num_val)])
        
        existing_data_files = set()
        if os.path.exists("/data/tok105"):
            existing_data_files = set(os.listdir("/data/tok105"))
        
        # Download pre-tokenized data if needed
        tar_file = "tok105.tar.gz"
        if tar_file not in existing_files:
            print("Downloading pre-tokenized data...")
            data_url = "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/tok105.tar.gz"
            result = os.system(f"cd /data && wget {data_url}")
            if result != 0:
                raise RuntimeError(f"wget failed with exit code {result}")
        else:
            print(f"Using existing {tar_file}")
        
        # Get expected file sizes from tar
        print("Checking file sizes in archive...")
        expected_sizes = get_tar_file_sizes(f"/data/{tar_file}")
        
        # Check for missing or incomplete files
        missing_files = []
        incomplete_files = []
        for file in required_files:
            if file not in existing_data_files:
                missing_files.append(file)
            else:
                actual_size = os.path.getsize(f"/data/tok105/{file}")
                expected_size = expected_sizes[file]
                if actual_size != expected_size:
                    print(f"File {file} has incorrect size: {actual_size} bytes (expected {expected_size})")
                    incomplete_files.append(file)
        
        files_to_extract = missing_files + incomplete_files
        if files_to_extract:
            print(f"\nNeed to extract {len(files_to_extract)} files:")
            print(f"- Missing: {len(missing_files)} files")
            print(f"- Incomplete: {len(incomplete_files)} files")
            
            # Create tok105 directory if it doesn't exist
            os.makedirs("/data/tok105", exist_ok=True)
            
            # Extract files one by one for better logging
            print("\nExtracting files...")
            for file in files_to_extract:
                print(f"Extracting {file}...", end='', flush=True)
                extract_cmd = f"cd /data && tar -xf {tar_file} tok105/{file}"
                result = os.system(extract_cmd)
                if result != 0:
                    print(" Failed!")
                    raise RuntimeError(f"tar extraction failed for {file} with exit code {result}")
                
                # Verify extracted file
                if not os.path.exists(f"/data/tok105/{file}"):
                    print(" File not found after extraction!")
                    raise RuntimeError(f"Failed to extract {file}")
                
                actual_size = os.path.getsize(f"/data/tok105/{file}")
                expected_size = expected_sizes[file]
                if actual_size != expected_size:
                    print(f" Size mismatch: {actual_size} bytes (expected {expected_size})")
                    raise RuntimeError(f"Extracted file {file} has incorrect size")
                print(f" Success ({actual_size:,} bytes)")
        else:
            print("All required data files are present and complete")
        
        print("\nConverting data to tensors...")
        try:
            # Process training data
            train_tokens = []
            total_train_tokens = 0
            print("Processing training files:")
            for i in range(num_train):
                file_num = str(i).zfill(2)
                file_path = f'/data/tok105/data{file_num}.bin'
                if not os.path.exists(file_path):
                    raise RuntimeError(f"Missing training file: {file_path}")
                print(f"Reading file {file_num}...", end='', flush=True)
                with open(file_path, 'rb') as f:
                    data = f.read()
                    train_tokens.extend(list(data))
                    total_train_tokens += len(data)
                print(f" Added {len(data):,} tokens (Total: {total_train_tokens:,})")
            
            print("\nConverting training data to tensor...")
            chunk_size = 10_000_000  # Process 10M tokens at a time
            tensor_chunks = []
            for i in range(0, len(train_tokens), chunk_size):
                print(f"Converting chunk {i//chunk_size + 1}/{(len(train_tokens) + chunk_size - 1)//chunk_size}...", 
                      end='', flush=True)
                chunk = torch.tensor(train_tokens[i:i + chunk_size], dtype=torch.long)
                tensor_chunks.append(chunk)
                print(f" Done ({i + len(chunk):,}/{len(train_tokens):,} tokens)")
            
            print("Concatenating chunks...")
            train_data = torch.cat(tensor_chunks)
            print("Saving training data...")
            torch.save(train_data, '/data/train.pt')
            print(f"Saved training data: {len(train_data):,} tokens")
            
            # Process validation data
            val_tokens = []
            total_val_tokens = 0
            print("\nProcessing validation files:")
            for i in range(45, 45 + num_val):
                file_num = str(i).zfill(2)
                file_path = f'/data/tok105/data{file_num}.bin'
                if not os.path.exists(file_path):
                    raise RuntimeError(f"Missing validation file: {file_path}")
                print(f"Reading file {file_num}...", end='', flush=True)
                with open(file_path, 'rb') as f:
                    data = f.read()
                    val_tokens.extend(list(data))
                    total_val_tokens += len(data)
                print(f" Added {len(data):,} tokens (Total: {total_val_tokens:,})")
            
            print("\nConverting validation data to tensor...")
            # Convert validation data in chunks too if needed
            if len(val_tokens) > chunk_size:
                tensor_chunks = []
                for i in range(0, len(val_tokens), chunk_size):
                    print(f"Converting chunk {i//chunk_size + 1}/{(len(val_tokens) + chunk_size - 1)//chunk_size}...", 
                          end='', flush=True)
                    chunk = torch.tensor(val_tokens[i:i + chunk_size], dtype=torch.long)
                    tensor_chunks.append(chunk)
                    print(f" Done ({i + len(chunk):,}/{len(val_tokens):,} tokens)")
                print("Concatenating chunks...")
                val_data = torch.cat(tensor_chunks)
            else:
                val_data = torch.tensor(val_tokens, dtype=torch.long)
            
            print("Saving validation data...")
            torch.save(val_data, '/data/val.pt')
            print(f"Saved validation data: {len(val_data):,} tokens")
            
            # Cleanup extracted files
            print("\nCleaning up...")
            os.system("rm -rf /data/tok105 /data/tok105.tar.gz")
            
        except Exception as e:
            print(f"Error processing data: {e}")
            raise RuntimeError("Failed to process pre-tokenized data")

@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
def train(fresh_start=False):
    print(f"\n=== Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("\nModel configuration:")
    for k, v in MODEL_CONFIG.items():
        print(f"  {k}: {v}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_data()
    
    # Load data and create model
    print("\nLoading dataset...")
    train_data = torch.load('/data/train.pt').to(device)
    val_data = torch.load('/data/val.pt').to(device)
    print(f"Training data size: {len(train_data):,} tokens")
    print(f"Validation data size: {len(val_data):,} tokens")
    
    print("\nInitializing model...")
    model = GPTLanguageModel(**{k: MODEL_CONFIG[k] for k in 
                               ['vocab_size', 'dim', 'n_layer', 'n_head', 'block_size']}).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
    start_iter = 0
    best_val_loss = float('inf')
    
    # Load previous checkpoint if it exists and fresh_start is False
    checkpoint_files = ['model_best.pt', 'checkpoint.pt', 'model.pt']
    checkpoint_loaded = False
    
    if not fresh_start:
        print("Looking for existing checkpoints...")
        for checkpoint_file in checkpoint_files:
            checkpoint_path = f'/data/{checkpoint_file}'
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_file}...")
                try:
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model_state'])
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    start_iter = checkpoint['iteration']
                    best_val_loss = checkpoint['best_val_loss']
                    checkpoint_loaded = True
                    print(f"Resumed from iteration {start_iter} with best validation loss: {best_val_loss:.4f}")
                    break
                except Exception as e:
                    print(f"Failed to load {checkpoint_file}: {e}")
    else:
        print("Fresh start requested - skipping checkpoint loading")
        # Optionally delete existing checkpoints
        for checkpoint_file in checkpoint_files:
            checkpoint_path = f'/data/{checkpoint_file}'
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                    print(f"Deleted existing checkpoint: {checkpoint_file}")
                except Exception as e:
                    print(f"Failed to delete {checkpoint_file}: {e}")

    if not checkpoint_loaded and not fresh_start:
        print("No valid checkpoint found. Starting training from scratch.")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    def save_checkpoint(iteration, model, optimizer, val_loss, is_best=False):
        checkpoint = {
            'iteration': iteration,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'model_config': MODEL_CONFIG,
        }
        
        # Save regular checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at iteration {iteration}")
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = '/data/model_best.pt'
            torch.save(checkpoint, best_model_path)
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - MODEL_CONFIG['block_size'], (MODEL_CONFIG['batch_size'],))
        x = torch.stack([data[i:i+MODEL_CONFIG['block_size']] for i in ix])
        y = torch.stack([data[i+1:i+MODEL_CONFIG['block_size']+1] for i in ix])
        return x.to(device), y.to(device)
    
    # Training loop
    print("\nStarting training loop...")
    print(f"Training from iteration {start_iter} to {MODEL_CONFIG['max_iters']}")
    start_time = time.time()
    checkpoint_interval = min(500, MODEL_CONFIG['eval_interval'])  # Save every 500 iterations or at eval, whichever is smaller
    
    try:
        for iter in range(start_iter, MODEL_CONFIG['max_iters']):
            # Evaluation
            if iter % MODEL_CONFIG['eval_interval'] == 0:
                model.eval()
                with torch.no_grad():
                    losses = [model(x, y)[1] for x, y in [get_batch('val') 
                             for _ in range(MODEL_CONFIG['eval_iters'])]]
                    val_loss = torch.mean(torch.tensor(losses))
                    
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                    
                    elapsed = time.time() - start_time
                    print(f"\nStep {iter}/{MODEL_CONFIG['max_iters']} ({iter/MODEL_CONFIG['max_iters']*100:.1f}%)")
                    print(f"Validation loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                    print(f"Time elapsed: {elapsed:.2f}s ({elapsed/60:.2f}min)")
                    
                    # Save checkpoint at evaluation
                    save_checkpoint(iter, model, optimizer, val_loss, is_best)
            
            # Regular checkpoint saving
            if iter % checkpoint_interval == 0 and iter != 0:
                save_checkpoint(iter, model, optimizer, best_val_loss)
            
            # Training
            model.train()
            xb, yb = get_batch('train')
            _, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if iter % 100 == 0:
                print(f"Training loss: {loss.item():.4f}", end='\r')
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving checkpoint...")
        save_checkpoint(iter, model, optimizer, best_val_loss)
        print("Checkpoint saved. Exiting...")
        return "Training interrupted but checkpoint saved!"
    
    # Save final model
    print("\nSaving final checkpoint...")
    save_checkpoint(MODEL_CONFIG['max_iters'], model, optimizer, best_val_loss)
    
    # Final stats
    total_time = time.time() - start_time
    print(f"\n=== Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return "Training completed successfully!"

@app.function(image=image, gpu="T4", volumes={"/data": volume})
def inference(initial_context="Once upon a time ", max_new_tokens=256):
    import sentencepiece as spm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = GPTLanguageModel(**{k: MODEL_CONFIG[k] for k in 
                               ['vocab_size', 'dim', 'n_layer', 'n_head', 'block_size']}).to(device)
    
    # Try loading the model from different possible checkpoint files
    checkpoint_files = ['model_best.pt', 'checkpoint.pt', 'model.pt']
    model_loaded = False
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint_path = f'/data/{checkpoint_file}'
            if os.path.exists(checkpoint_path):
                print(f"Loading model from {checkpoint_file}")
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint['model_state'])
                print(f"Model loaded successfully. Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
                model_loaded = True
                break
        except Exception as e:
            print(f"Failed to load {checkpoint_file}: {e}")
    
    if not model_loaded:
        return "Error: No trained model found. Please run training first."
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    try:
        sp.load('/data/tok105.model')
    except:
        download_data()
        sp.load('/data/tok105.model')
    
    print("\nGenerating text...")
    try:
        # Encode input text
        context_tokens = torch.tensor([sp.encode(initial_context)], dtype=torch.long, device=device)
        print(f"Input context: {initial_context}")
        
        # Generate tokens
        model.eval()
        with torch.no_grad():
            generated_tokens = model.generate(context_tokens, max_new_tokens)
        
        # Decode all tokens at once
        all_tokens = generated_tokens[0].tolist()
        generated_text = sp.decode(all_tokens)
        
        # Clean up the text
        import re
        
        # Remove the specific Unicode escape sequence
        generated_text = generated_text.replace('\\342\\201\\207', '')
        # Remove any remaining escape sequences
        generated_text = re.sub(r'\\[0-9]{3}\\[0-9]{3}\\[0-9]{3}', '', generated_text)
        
        # Split into context and generated parts
        context_len = len(initial_context)
        context = generated_text[:context_len]
        generated = generated_text[context_len:]
        
        # Clean up the generated part
        generated = ''.join(generated.split())  # Remove all spaces
        generated = re.sub(r'([.!?,])', r'\1 ', generated)  # Add space after punctuation
        generated = re.sub(r'\s+', ' ', generated).strip()  # Normalize spaces
        
        # Combine context and generated text
        generated_text = context + generated
        
        # Final cleanup
        generated_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', generated_text)
        
        print("\nGenerated text:")
        print("-" * 80)
        print(generated_text)
        print("-" * 80)
        
        return generated_text
        
    except Exception as e:
        print(f"Error during text generation: {e}")
        return f"Error generating text: {str(e)}"

@app.local_entrypoint()
def main(command: str = "train", prompt: str = "Once upon a time", fresh_start: bool = False):
    """
    Main entry point for the application.
    Args:
        command: Either "train" or "inference"
        prompt: Initial text prompt for inference (only used if command is "inference")
        fresh_start: If present, starts training from scratch (only used if command is "train")
    """
    if command == "inference":
        print(f"Generating text from prompt: {prompt}")
        generated_text = inference.remote(initial_context=prompt)
        # Don't print the text here since it's already printed in the inference function
    elif command == "train":
        print("Starting training..." + (" (fresh start)" if fresh_start else ""))
        train.remote(fresh_start=fresh_start)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, inference")
