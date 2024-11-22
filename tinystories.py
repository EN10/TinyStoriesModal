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
    'max_iters': 5000,
    'eval_interval': 500,
    'eval_iters': 200,
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

def download_data():
    os.makedirs("/data", exist_ok=True)
    
    # Check what files we already have
    existing_files = set(os.listdir("/data"))
    
    # Download and extract training data first
    if "data.tar.gz" not in existing_files:
        print("Downloading training data...")
        data_url = "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/data.tar.gz"
        os.system(f"cd /data && wget {data_url}")
    
    if "train.txt" not in existing_files or "val.txt" not in existing_files:
        print("Extracting training data...")
        print("  - Extracting train.txt...")
        print("  - Extracting val.txt...")
        os.system("cd /data && tar -xvzf data.tar.gz")
        # Verify extraction
        if not os.path.exists("/data/train.txt") or not os.path.exists("/data/val.txt"):
            raise RuntimeError("Failed to extract training data files")
    
    # Download tokenizer files if needed
    tokenizer_files = ["tok105.model", "tok105.vocab", "tok105.tar.gz"]
    base_url = "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/"
    
    for file in tokenizer_files:
        if file not in existing_files:
            print(f"Downloading {file}...")
            os.system(f"cd /data && wget {base_url}{file}")
            if not os.path.exists(f"/data/{file}"):
                raise RuntimeError(f"Failed to download {file}")
    
    # Extract tokenizer files only if necessary files don't exist
    if not all(f in existing_files for f in ["tok105.model", "tok105.vocab"]):
        print("Extracting tokenizer files...")
        print("  - Extracting tok105.model...")
        print("  - Extracting tok105.vocab...")
        os.system("cd /data && tar -xvzf tok105.tar.gz")
        if not os.path.exists("/data/tok105.model"):
            raise RuntimeError("Failed to extract tokenizer files")
    else:
        print("Tokenizer files already extracted, skipping extraction.")
    
    # Check if processed data files exist
    if 'train.pt' in existing_files and 'val.pt' in existing_files:
        print("Found existing processed data files, skipping data preparation.")
        return
    
    # Process and save the data
    print("Processing training data...")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load('/data/tok105.model')
    
    # Read and tokenize the text files
    try:
        with open('/data/train.txt', 'r') as f:
            train_text = f.read()
        with open('/data/val.txt', 'r') as f:
            val_text = f.read()
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not read training files: {e}")
            
    # Convert to tensors and save
    train_data = torch.tensor(sp.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(sp.encode(val_text), dtype=torch.long)
    
    torch.save(train_data, '/data/train.pt')
    torch.save(val_data, '/data/val.pt')
    
    print(f"Saved processed data: train ({len(train_data):,} tokens), val ({len(val_data):,} tokens)")

@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
def train():
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
    
    # Load previous checkpoint if it exists
    if os.path.exists('/data/model.pt'):
        print("Loading previous checkpoint...")
        model.load_state_dict(torch.load('/data/model.pt'))
        print("Resumed from previous checkpoint.")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
    
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - MODEL_CONFIG['block_size'], (MODEL_CONFIG['batch_size'],))
        x = torch.stack([data[i:i+MODEL_CONFIG['block_size']] for i in ix])
        y = torch.stack([data[i+1:i+MODEL_CONFIG['block_size']+1] for i in ix])
        return x.to(device), y.to(device)
    
    # Training loop
    print("\nStarting training loop...")
    start_time = time.time()
    best_val_loss = float('inf')
    
    for iter in range(MODEL_CONFIG['max_iters']):
        # Evaluation
        if iter % MODEL_CONFIG['eval_interval'] == 0:
            model.eval()
            with torch.no_grad():
                losses = [model(x, y)[1] for x, y in [get_batch('val') 
                         for _ in range(MODEL_CONFIG['eval_iters'])]]
                val_loss = torch.mean(torch.tensor(losses))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), '/data/model_best.pt')
                
                elapsed = time.time() - start_time
                print(f"\nStep {iter}/{MODEL_CONFIG['max_iters']} ({iter/MODEL_CONFIG['max_iters']*100:.1f}%)")
                print(f"Validation loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                print(f"Time elapsed: {elapsed:.2f}s ({elapsed/60:.2f}min)")
        
        # Training
        model.train()
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Training loss: {loss.item():.4f}", end='\r')
    
    # Save final model
    print("\nSaving final model...")
    torch.save(model.state_dict(), '/data/model.pt')
    
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
    try:
        model.load_state_dict(torch.load('/data/model.pt'))
    except FileNotFoundError:
        return "Error: No trained model found. Please run training first."
    
    # Load tokenizer and generate text
    sp = spm.SentencePieceProcessor()
    try:
        sp.load('/data/tok105.model')
    except:
        download_data()
        sp.load('/data/tok105.model')
    
    context_tokens = torch.tensor([sp.encode(initial_context)], dtype=torch.long, device=device)
    with torch.no_grad():
        generated_tokens = model.generate(context_tokens, max_new_tokens)
    
    return sp.decode(generated_tokens[0].tolist())

@app.local_entrypoint()
def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "inference":
        context = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time "
        print("Generated text:", inference.remote(initial_context=context))
    else:
        train.remote()
