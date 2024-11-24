import modal
import os

# Initialize Modal app and volume
app = modal.App("tinystories-simple")
volume = modal.Volume.from_name("tinystories-volume")

# Create image with required packages and tools
image = (modal.Image.debian_slim()
         .pip_install("wget", "tqdm", "torch", "numpy", "requests", "sentencepiece"))
        #  .run_commands("apt-get update", "apt-get install -y pv"))

# Define files to download with their source URLs
FILES = {
    # All files using dictionary comprehension
    **{f: f"https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/{f}" for f in 
       ["tok105.tar.gz", "tok105.bin"]},
    **{f: f"https://raw.githubusercontent.com/karpathy/llama2.c/master/{f}" for f in 
       ["train.py", "model.py", "tinystories.py", "tokenizer.py", "export.py", "configurator.py", "run.c"]}
}

@app.function(image=image, volumes={"/data": volume})
def setup_data():
    import wget
    from tqdm import tqdm
    
    # List current contents of data directory
    print("Current files:", ", ".join(os.listdir("/data")))
    
    # Download any missing files with progress bar
    for file, url in FILES.items():
        if not os.path.exists(f"/data/{file}"):
            print(f"\nDownloading {file}...")
            wget.download(url, f"/data/{file}", bar=wget.bar_adaptive)
            print()  # New line after progress bar
    
    # Extract training data archive if not already extracted
    if not os.path.exists("/data/tok105"):
        print("\nExtracting training data...")
        os.system("cd /data && pv tok105.tar.gz | tar xzf -")  # Show extraction progress
    
    # Compile run.c if needed
    if not os.path.exists("/data/run"):
        print("\nCompiling run.c...")
        os.system("cd /data && gcc -O3 -o run run.c -lm")

@app.function(image=image, volumes={"/data": volume}, gpu="T4")
def train():
    # Command to run initial training with specific parameters
    train_cmd = """
    python train.py --vocab_source=custom --vocab_size=105 --compile=False \
      --dim=128 --n_layers=5 --n_heads=8 --n_kv_heads=4 --batch_size=32 \
      --always_save_checkpoint=True --eval_interval=1 --max_iters=1
    """
    
    # Run initial training
    print("\nStarting training...")
    os.chdir("/data")  # Ensure we're in the right directory
    result = os.system(train_cmd)
    if result != 0:
        print("Training failed!")
    else:
        print("Training completed successfully!")

@app.function(image=image, volumes={"/data": volume})
def inference(prompt="Once upon a time"):
    os.chdir("/data")
    print(f"\nGenerating text from prompt: {prompt}")
    cmd = f"./run out/model.bin -z tok105.bin -t 0.8 -n 256 -i \"{prompt}\""
    os.system(cmd)

# Entry point when running with 'modal run'
@app.local_entrypoint()
def main(command="train", prompt="Once upon a time"):
    if command == "train":
        setup_data.remote()
        train.remote()
    elif command == "inference":
        inference.remote(prompt)
    else:
        print("Invalid command. Use 'train' or 'inference'")
