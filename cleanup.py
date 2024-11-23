import modal
import torch
import os
import gc
import json
from multiprocessing import Pool, cpu_count

# Configuration
CHUNK_SIZE = 25000  # Number of tokens to process at once
TEXT_CHUNK_SIZE = 500000  # Number of characters to process at once
SAVE_EVERY_N_CHUNKS = 5000  # Save progress every N chunks
TIMEOUT_SECONDS = 3600  # 1 hour timeout
NUM_PROCESSES = max(1, cpu_count() - 1)  # Use all CPU cores except one
ENCODING_BATCH_SIZE = 1000  # Number of text segments to encode at once

app = modal.App("tinystories-cleanup")
volume = modal.Volume.from_name("tinystories-volume")
image = (modal.Image.debian_slim()
         .pip_install("torch", "numpy", "sentencepiece"))

def save_progress(progress_file, progress_data):
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"last_chunk": 0, "encoding_position": 0}

def encode_batch(args):
    """Helper function for parallel encoding"""
    sp, text_batch = args
    return sp.encode(text_batch)

def parallel_encode(sp, text_segments):
    """Encode text segments in parallel using multiple CPU cores"""
    with Pool(NUM_PROCESSES) as pool:
        args = [(sp, segment) for segment in text_segments]
        results = pool.map(encode_batch, args)
    return [token for batch in results for token in batch]

def batch_encode(sp, text, batch_size=ENCODING_BATCH_SIZE):
    """Split text into batches and encode"""
    # Split text into roughly equal segments
    avg_char_per_segment = 1000  # Adjust this based on your text
    segments = []
    
    for i in range(0, len(text), avg_char_per_segment):
        segment = text[i:i + avg_char_per_segment]
        # Ensure we split at word boundaries
        if i + avg_char_per_segment < len(text):
            last_space = segment.rfind(' ')
            if last_space > 0:
                segment = segment[:last_space]
                i = i + last_space
        segments.append(segment)
    
    # Process segments in batches
    all_tokens = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        tokens = parallel_encode(sp, batch)
        all_tokens.extend(tokens)
    
    return all_tokens

@app.function(image=image, volumes={"/data": volume}, timeout=TIMEOUT_SECONDS)
def cleanup_training_data():
    """Clean up the training data by properly decoding and re-encoding the tokens"""
    import sentencepiece as spm
    
    progress_file = '/data/train_cleanup_progress.json'
    progress = load_progress(progress_file)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    if not os.path.exists('/data/tok105.model'):
        print("Error: tokenizer model not found")
        return
    sp.load('/data/tok105.model')
    
    if os.path.exists('/data/train.pt'):
        print("Processing training data...")
        train_data = torch.load('/data/train.pt')
        print(f"Loaded training data: {len(train_data):,} tokens")
        
        chunk_size = CHUNK_SIZE
        text_chunk_size = TEXT_CHUNK_SIZE
        
        # Check if we have partial results
        if os.path.exists('/data/train_partial.txt') and progress["last_chunk"] > 0:
            print(f"Resuming from chunk {progress['last_chunk']}")
            with open('/data/train_partial.txt', 'r', encoding='utf-8') as f:
                train_text = [f.read()]
        else:
            train_text = []
            progress["last_chunk"] = 0
        
        try:
            # Continue decoding from last successful chunk
            for i in range(progress["last_chunk"], len(train_data), chunk_size):
                chunk = train_data[i:i + chunk_size].tolist()
                print(f"Decoding chunk {i//chunk_size + 1}/{(len(train_data) + chunk_size - 1)//chunk_size} "
                      f"({i:,}-{min(i+chunk_size, len(train_data)):,} tokens)...", flush=True)
                decoded_chunk = sp.decode(chunk)
                train_text.append(decoded_chunk)
                
                if (i + chunk_size) % (chunk_size * SAVE_EVERY_N_CHUNKS) == 0:
                    print("\nSaving intermediate progress...")
                    intermediate_text = "".join(train_text)
                    with open('/data/train_partial.txt', 'w', encoding='utf-8') as f:
                        f.write(intermediate_text)
                    progress["last_chunk"] = i + chunk_size
                    save_progress(progress_file, progress)
                    print(f"Saved {len(intermediate_text):,} characters")
                    gc.collect()
                    torch.cuda.empty_cache()
            
            print("Concatenating decoded chunks...")
            train_text = "".join(train_text)
            print(f"Decoded text length: {len(train_text):,} characters")
            
            # Re-encode from last position using batched parallel processing
            print(f"Re-encoding training text using {NUM_PROCESSES} CPU cores...")
            train_tokens = []
            
            for i in range(progress["encoding_position"], len(train_text), text_chunk_size):
                text_chunk = train_text[i:i + text_chunk_size]
                print(f"Processing chunk {i//text_chunk_size + 1}/{(len(train_text) + text_chunk_size - 1)//text_chunk_size}...")
                
                chunk_tokens = batch_encode(sp, text_chunk)
                train_tokens.extend(chunk_tokens)
                
                print(f"Re-encoded {i+len(text_chunk):,}/{len(train_text):,} characters...", flush=True)
                progress["encoding_position"] = i + text_chunk_size
                save_progress(progress_file, progress)
                gc.collect()
                torch.cuda.empty_cache()
            
            train_tensor = torch.tensor(train_tokens, dtype=torch.long)
            
            print(f"Saving cleaned training data ({len(train_tensor):,} tokens)...")
            torch.save(train_tensor, '/data/train_clean.pt')
            
            # Clear progress file after successful completion
            if os.path.exists(progress_file):
                os.remove(progress_file)
            if os.path.exists('/data/train_partial.txt'):
                os.remove('/data/train_partial.txt')
            
        except Exception as e:
            print(f"Error processing training data: {e}")
            raise
    
    print("\nCleanup complete!")
    gc.collect()
    torch.cuda.empty_cache()

@app.function(image=image, volumes={"/data": volume}, timeout=TIMEOUT_SECONDS)
def cleanup_validation_data():
    """Clean up the validation data by properly decoding and re-encoding the tokens"""
    import sentencepiece as spm
    
    progress_file = '/data/val_cleanup_progress.json'
    progress = load_progress(progress_file)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    if not os.path.exists('/data/tok105.model'):
        print("Error: tokenizer model not found")
        return
    sp.load('/data/tok105.model')
    
    if os.path.exists('/data/val.pt'):
        print("Processing validation data...")
        val_data = torch.load('/data/val.pt')
        print(f"Loaded validation data: {len(val_data):,} tokens")
        
        chunk_size = CHUNK_SIZE
        text_chunk_size = TEXT_CHUNK_SIZE
        
        # Check if we have partial results
        if os.path.exists('/data/val_partial.txt') and progress["last_chunk"] > 0:
            print(f"Resuming from chunk {progress['last_chunk']}")
            with open('/data/val_partial.txt', 'r', encoding='utf-8') as f:
                val_text = [f.read()]
        else:
            val_text = []
            progress["last_chunk"] = 0
        
        try:
            for i in range(progress["last_chunk"], len(val_data), chunk_size):
                chunk = val_data[i:i + chunk_size].tolist()
                print(f"Decoding chunk {i//chunk_size + 1}/{(len(val_data) + chunk_size - 1)//chunk_size} "
                      f"({i:,}-{min(i+chunk_size, len(val_data)):,} tokens)...", flush=True)
                decoded_chunk = sp.decode(chunk)
                val_text.append(decoded_chunk)
                
                if (i + chunk_size) % (chunk_size * SAVE_EVERY_N_CHUNKS) == 0:
                    print("\nSaving intermediate progress...")
                    intermediate_text = "".join(val_text)
                    with open('/data/val_partial.txt', 'w', encoding='utf-8') as f:
                        f.write(intermediate_text)
                    progress["last_chunk"] = i + chunk_size
                    save_progress(progress_file, progress)
                    gc.collect()
                    torch.cuda.empty_cache()
            
            print("Concatenating decoded chunks...")
            val_text = "".join(val_text)
            print(f"Decoded text length: {len(val_text):,} characters")
            
            # Re-encode from last position using batched parallel processing
            print(f"Re-encoding validation text using {NUM_PROCESSES} CPU cores...")
            val_tokens = []
            
            for i in range(progress["encoding_position"], len(val_text), text_chunk_size):
                text_chunk = val_text[i:i + text_chunk_size]
                print(f"Processing chunk {i//text_chunk_size + 1}/{(len(val_text) + text_chunk_size - 1)//text_chunk_size}...")
                
                chunk_tokens = batch_encode(sp, text_chunk)
                val_tokens.extend(chunk_tokens)
                
                print(f"Re-encoded {i+len(text_chunk):,}/{len(val_text):,} characters...", flush=True)
                progress["encoding_position"] = i + text_chunk_size
                save_progress(progress_file, progress)
                gc.collect()
                torch.cuda.empty_cache()
            
            val_tensor = torch.tensor(val_tokens, dtype=torch.long)
            
            print(f"Saving cleaned validation data ({len(val_tensor):,} tokens)...")
            torch.save(val_tensor, '/data/val_clean.pt')
            
            # Clear progress file after successful completion
            if os.path.exists(progress_file):
                os.remove(progress_file)
            if os.path.exists('/data/val_partial.txt'):
                os.remove('/data/val_partial.txt')
            
        except Exception as e:
            print(f"Error processing validation data: {e}")
            raise
    
    print("\nCleanup complete!")
    gc.collect()
    torch.cuda.empty_cache()

@app.local_entrypoint()
def main():
    cleanup_training_data.remote()
    cleanup_validation_data.remote()
