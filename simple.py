import modal
import os

app = modal.App("tinystories-simple")
volume = modal.Volume.from_name("tinystories-volume")
image = (modal.Image.debian_slim()
         .pip_install("wget", "torch", "numpy", "tqdm", "pv", "requests", "sentencepiece"))

# Files to download
FILES = {
    # HuggingFace files
    "tok105.tar.gz": "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/tok105.tar.gz",
    "tok105.bin": "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/tok105.bin",
    # GitHub files - using master branch
    "train.py": "https://raw.githubusercontent.com/karpathy/llama2.c/master/train.py",
    "model.py": "https://raw.githubusercontent.com/karpathy/llama2.c/master/model.py",
    "configurator.py": "https://raw.githubusercontent.com/karpathy/llama2.c/master/configurator.py",
    "export.py": "https://raw.githubusercontent.com/karpathy/llama2.c/master/export.py",
    "tokenizer.py": "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.py"
}

@app.function(image=image, volumes={"/data": volume}, gpu="T4")
def setup():
    import wget
    print("Current directory:", os.getcwd())
    print("Current files:", ", ".join(os.listdir("/data")))
    
    # Download missing files
    for file, url in FILES.items():
        if not os.path.exists(f"/data/{file}"):
            print(f"\nDownloading {file}...")
            wget.download(url, f"/data/{file}", bar=wget.bar_adaptive)
            print()
    
    # Extract training data if needed
    if not os.path.exists("/data/tok105"):
        print("\nExtracting training data...")
        os.system("cd /data && pv tok105.tar.gz | tar xzf -")
    
    # Run training
    print("\nStarting training...")
    os.chdir("/data")
    print("Training directory:", os.getcwd())
    result = os.system("""
        python train.py --vocab_source=custom --vocab_size=105 --compile=False \
        --dim=128 --n_layers=5 --n_heads=8 --n_kv_heads=4 --batch_size=32 \
        --always_save_checkpoint=True --eval_interval=1 --max_iters=1
    """)
    print("Training", "completed successfully!" if result == 0 else "failed!")

@app.local_entrypoint()
def main():
    setup.remote()
