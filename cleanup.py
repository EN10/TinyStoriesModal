import modal
import torch
import os

app = modal.App("tinystories-cleanup")
volume = modal.Volume.from_name("tinystories-volume")
image = (modal.Image.debian_slim()
         .pip_install("torch", "numpy", "sentencepiece"))

@app.function(image=image, volumes={"/data": volume})
def cleanup_data():
    """Clean up the training data by properly decoding and re-encoding the tokens"""
    import sentencepiece as spm  # Import inside function
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    if not os.path.exists('/data/tok105.model'):
        print("Error: tokenizer model not found")
        return
    sp.load('/data/tok105.model')
    
    # Process training data
    if os.path.exists('/data/train.pt'):
        print("Processing training data...")
        train_data = torch.load('/data/train.pt')
        print(f"Loaded training data: {len(train_data):,} tokens")
        
        # Decode to text in chunks
        print("Decoding training tokens...")
        chunk_size = 100000  # Process 100k tokens at a time
        train_text = []
        for i in range(0, len(train_data), chunk_size):
            chunk = train_data[i:i + chunk_size].tolist()
            print(f"Decoding chunk {i//chunk_size + 1}/{(len(train_data) + chunk_size - 1)//chunk_size} "
                  f"({i:,}-{min(i+chunk_size, len(train_data)):,} tokens)...", flush=True)
            decoded_chunk = sp.decode(chunk)
            train_text.append(decoded_chunk)
            
        print("Concatenating decoded chunks...")
        train_text = "".join(train_text)
        print(f"Decoded text length: {len(train_text):,} characters")
        
        # Re-encode properly
        print("Re-encoding training text...")
        train_tokens = sp.encode(train_text)
        print(f"Re-encoded tokens: {len(train_tokens):,}")
        train_tensor = torch.tensor(train_tokens, dtype=torch.long)
        
        # Save cleaned data
        print(f"Saving cleaned training data ({len(train_tensor):,} tokens)...")
        torch.save(train_tensor, '/data/train_clean.pt')
    
    # Process validation data with similar chunking
    if os.path.exists('/data/val.pt'):
        print("\nProcessing validation data...")
        val_data = torch.load('/data/val.pt')
        print(f"Loaded validation data: {len(val_data):,} tokens")
        
        # Decode to text in chunks
        print("Decoding validation tokens...")
        val_text = []
        for i in range(0, len(val_data), chunk_size):
            chunk = val_data[i:i + chunk_size].tolist()
            print(f"Decoding chunk {i//chunk_size + 1}/{(len(val_data) + chunk_size - 1)//chunk_size} "
                  f"({i:,}-{min(i+chunk_size, len(val_data)):,} tokens)...", flush=True)
            decoded_chunk = sp.decode(chunk)
            val_text.append(decoded_chunk)
            
        print("Concatenating decoded chunks...")
        val_text = "".join(val_text)
        print(f"Decoded text length: {len(val_text):,} characters")
        
        # Re-encode properly
        print("Re-encoding validation text...")
        val_tokens = sp.encode(val_text)
        print(f"Re-encoded tokens: {len(val_tokens):,}")
        val_tensor = torch.tensor(val_tokens, dtype=torch.long)
        
        # Save cleaned data
        print(f"Saving cleaned validation data ({len(val_tensor):,} tokens)...")
        torch.save(val_tensor, '/data/val_clean.pt')
    
    print("\nCleanup complete!")

@app.local_entrypoint()
def main():
    cleanup_data.remote()
