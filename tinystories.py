import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
    files = ["tok105.model", "tok105.vocab", "tok105.tar.gz"]
    base_url = "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/"
    
    for file in files:
        os.system(f"cd /data && wget {base_url}{file}")
    os.system("cd /data && tar -xf tok105.tar.gz")

@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_data()
    
    # Load data and create model
    train_data = torch.load('/data/train.pt').to(device)
    val_data = torch.load('/data/val.pt').to(device)
    model = GPTLanguageModel(**{k: MODEL_CONFIG[k] for k in 
                               ['vocab_size', 'dim', 'n_layer', 'n_head', 'block_size']}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
    
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - MODEL_CONFIG['block_size'], (MODEL_CONFIG['batch_size'],))
        x = torch.stack([data[i:i+MODEL_CONFIG['block_size']] for i in ix])
        y = torch.stack([data[i+1:i+MODEL_CONFIG['block_size']+1] for i in ix])
        return x.to(device), y.to(device)
    
    # Training loop
    for iter in range(MODEL_CONFIG['max_iters']):
        if iter % MODEL_CONFIG['eval_interval'] == 0:
            losses = [model(x, y)[1] for x, y in [get_batch('val') 
                     for _ in range(MODEL_CONFIG['eval_iters'])]]
            print(f"step {iter}: val loss {torch.mean(torch.tensor(losses)):.4f}")
        
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), '/data/model.pt')
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
