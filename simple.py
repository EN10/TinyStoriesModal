import modal
import os

app = modal.App("tinystories-simple")
volume = modal.Volume.from_name("tinystories-volume")
image = modal.Image.debian_slim().pip_install("wget", "torch", "numpy", "tqdm", "pv", "requests", "sentencepiece")

# Files to download with their URLs
FILES = {
    # Training data and tokenizer
    "tok105.tar.gz": "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/tok105.tar.gz",
    "tok105.bin": "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/tok105.bin",
    # Training scripts from llama2.c
    **{f: f"https://raw.githubusercontent.com/karpathy/llama2.c/master/{f}" for f in 
       ["train.py", "model.py", "configurator.py", "export.py", "tokenizer.py", "tinystories.py"]}
}

@app.function(image=image, volumes={"/data": volume}, gpu="T4")
def setup():
    import wget
    os.chdir("/data")
    
    # Download and extract files
    for file, url in FILES.items():
        if not os.path.exists(file):
            print(f"Downloading {file}...")
            wget.download(url, file)
            print()
    
    if not os.path.exists("tok105"):
        print("Extracting training data...")
        os.system("pv tok105.tar.gz | tar xzf -")
    
    # Setup data directory
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/tok105"):
        os.symlink("../tok105", "data/tok105")
    
    # Train model
    print("Starting training...")
    os.system("""
        python train.py --vocab_source=custom --vocab_size=105 --compile=False \
        --dim=128 --n_layers=5 --n_heads=8 --n_kv_heads=4 --batch_size=32 \
        --always_save_checkpoint=True --eval_interval=10 --max_iters=10
    """)

@app.local_entrypoint()
def main():
    setup.remote()
