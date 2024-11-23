import modal
import os
import subprocess


app = modal.App("tinystories-data")
volume = modal.Volume.from_name("tinystories-volume")
image = (modal.Image.debian_slim()
         .apt_install("wget"))
         
def get_tar_file_sizes(tar_path):
    """Get file sizes from tar archive"""
    print("\nGetting archive contents...")
    
    # Count files with progress indicator
    print("Counting files in archive...")
    process = subprocess.Popen(['tar', '-tvf', tar_path],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    
    file_count = 0
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner_idx = 0
    
    for line in process.stdout:
        file_count += 1
        if file_count % 5 == 0:  # Update every 5 files instead of 100
            spinner_char = spinner[spinner_idx]
            print(f"\r{spinner_char} Counted {file_count:,} files...", end='', flush=True)
            spinner_idx = (spinner_idx + 1) % len(spinner)
    
    print(f"\nTotal files in archive: {file_count:,}")
    
    # Now process with progress bar
    print("\nProcessing archive contents...")
    process = subprocess.Popen(['tar', '-tvf', tar_path],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    
    file_sizes = {}
    processed_count = 0
    last_percent = -1
    
    for line in process.stdout:
        processed_count += 1
        percent = (processed_count * 100) // file_count
        
        # Update progress bar if percentage changed
        if percent != last_percent:
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '=' * filled + '-' * (bar_length - filled)
            print(f"\rProgress: [{bar}] {percent}% ({processed_count:,}/{file_count:,} files)", end='', flush=True)
            last_percent = percent
        
        # Process the file info
        parts = line.split()
        size = int(parts[2])
        filename = parts[-1]
        if filename.startswith('tok105/') and filename.endswith('.bin'):
            file_sizes[os.path.basename(filename)] = size
    
    # Check if process completed successfully
    process.wait()
    if process.returncode != 0:
        print(f"\nError: Failed to read tar contents")
        print(f"Details: {process.stderr.read()}")
        raise RuntimeError("Failed to read tar contents")
    
    print(f"\nFound {len(file_sizes)} data files in archive")
    return file_sizes

def verify_extracted_files(expected_sizes, required_files):
    """Verify that extracted bin files match their expected sizes"""
    if not os.path.exists("/data/tok105"):
        print("tok105 directory not found - files not extracted yet")
        return False, required_files
        
    existing_files = set(os.listdir("/data/tok105"))
    files_to_extract = []
    
    for filename in required_files:
        if filename not in existing_files:
            print(f"Missing file: {filename}")
            files_to_extract.append(filename)
            continue
            
        actual_size = os.path.getsize(f"/data/tok105/{filename}")
        expected_size = expected_sizes[filename]
        if actual_size != expected_size:
            print(f"Size mismatch for {filename}: {actual_size} bytes (expected {expected_size})")
            files_to_extract.append(filename)
        else:
            print(f"Verified {filename}: {actual_size} bytes - ✓")
            
    return len(files_to_extract) == 0, files_to_extract

def extract_files(tar_file, expected_sizes, required_files):
    """Extract specific files and verify them"""
    print("\nExtracting files...")
    os.makedirs("/data/tok105", exist_ok=True)
    
    total_bytes = sum(expected_sizes[f] for f in required_files)
    extracted_bytes = 0
    
    # Extract files one by one for better logging
    for i, file in enumerate(required_files, 1):
        print(f"\nFile {i}/{len(required_files)}: {file}")
        print(f"Progress: {extracted_bytes:,}/{total_bytes:,} bytes ({(extracted_bytes/total_bytes)*100:.1f}%)")
        print(f"  Extracting...", end='', flush=True)
        
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
            print(f"\n  Error: Size mismatch")
            print(f"  - Actual: {actual_size:,} bytes")
            print(f"  - Expected: {expected_size:,} bytes")
            raise RuntimeError(f"Extracted file {file} has incorrect size")
            
        extracted_bytes += actual_size
        print(f" Success!")
        print(f"  - Size: {actual_size:,} bytes")
        print(f"  - Path: /data/tok105/{file}")

def download_tok105_files(num_train=10, val_percent=10):
    os.makedirs("/data", exist_ok=True)
    
    # Calculate required files
    num_val = max(1, round(num_train * val_percent / 100))
    required_files = ([f"data{str(i).zfill(2)}.bin" for i in range(num_train)] + 
                     [f"data{str(i).zfill(2)}.bin" for i in range(45, 45 + num_val)])
    
    print(f"\nWill process {num_train} training files and {num_val} validation files ({val_percent}%)")
    
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
    
    # Download and verify pre-tokenized data
    tar_file = "tok105.tar.gz"
    if tar_file not in existing_files:
        print("Downloading pre-tokenized data...")
        data_url = "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/tok105.tar.gz"
        result = os.system(f"cd /data && wget {data_url}")
        if result != 0:
            raise RuntimeError(f"wget failed with exit code {result}")
        print(f"Successfully downloaded {tar_file}")
    
    # Verify tar file before proceeding
    if not verify_tar_file(tar_file):
        print("Tar file verification failed - will attempt to redownload")
        os.remove(f"/data/{tar_file}")
        print("Downloading pre-tokenized data again...")
        data_url = "https://huggingface.co/datasets/enio/TinyStories/resolve/main/tok105/tok105.tar.gz"
        result = os.system(f"cd /data && wget {data_url}")
        if result != 0:
            raise RuntimeError(f"wget failed with exit code {result}")
        if not verify_tar_file(tar_file):
            raise RuntimeError("Failed to download valid tar file after retry")
    
    # Get expected file sizes and verify/extract files
    expected_sizes = get_tar_file_sizes(f"/data/{tar_file}")
    
    all_verified, files_to_extract = verify_extracted_files(expected_sizes, required_files)
    if not all_verified:
        print(f"\n{len(files_to_extract)} files need to be extracted...")
        extract_files(tar_file, expected_sizes, files_to_extract)
    
    # Cleanup
    # print("\nCleaning up...")
    # os.system(f"rm -f /data/{tar_file}")
    # print("Done!")

def verify_tar_file(tar_file):
    """Verify the tar file size and integrity"""
    print(f"\nVerifying {tar_file}...")
    
    if not os.path.exists(f"/data/{tar_file}"):
        print(f"Error: {tar_file} not found!")
        return False
        
    actual_size = os.path.getsize(f"/data/{tar_file}")
    print(f"Actual tar file size: {actual_size:,} bytes")
    
    # Count files with progress indicator
    print("\nCounting files in archive...")
    process = subprocess.Popen(['tar', '-tvf', f"/data/{tar_file}"], 
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    
    file_count = 0
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner_idx = 0
    
    for line in process.stdout:
        file_count += 1
        if file_count % 5 == 0:  # Update every 5 files instead of 100
            spinner_char = spinner[spinner_idx]
            print(f"\r{spinner_char} Counted {file_count:,} files...", end='', flush=True)
            spinner_idx = (spinner_idx + 1) % len(spinner)
    
    process.wait()
    if process.returncode != 0:
        print(f"\nError during counting: {process.stderr.read()}")
        return False
    
    print(f"\nTotal files in archive: {file_count:,}")
    
    # Test tar file integrity with progress
    print("\nTesting tar file integrity...")
    process = subprocess.Popen(['tar', '-tvf', f"/data/{tar_file}"], 
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    
    verified_count = 0
    last_percent = -1
    for line in process.stdout:
        verified_count += 1
        percent = (verified_count * 100) // file_count
        
        # Update progress bar if percentage changed
        if percent != last_percent:
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '=' * filled + '-' * (bar_length - filled)
            print(f"\rProgress: [{bar}] {percent}% ({verified_count:,}/{file_count:,} files)", end='', flush=True)
            last_percent = percent
    
    # Check if process completed successfully
    process.wait()
    if process.returncode != 0:
        print(f"\nError: Tar file is corrupted")
        print(f"Details: {process.stderr.read()}")
        return False
    
    print(f"\nTar file integrity check passed! (Verified {verified_count:,} files)")
    return True

@app.function(image=image, volumes={"/data": volume})
def download_data(num_train=10, val_percent=10):
    download_tok105_files(num_train, val_percent)

@app.local_entrypoint()
def main():
    download_data.remote()
