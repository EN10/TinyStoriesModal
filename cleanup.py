import os
import argparse
import modal

# Create Modal app and volume
app = modal.App()
volume = modal.Volume.from_name("tinystories-volume")

@app.function(volumes={"/data": volume})
def clean_duplicate_files(directory="/data"):
    """
    Clean up duplicate files and unnecessary txt files in the specified directory.
    Handles tok105.* and data.tar.gz files, and removes large txt files.
    """
    files_removed = 0
    
    # Part 1: Clean up duplicates
    base_files = [
        "tok105.model",
        "tok105.vocab",
        "tok105.tar.gz",
        "data.tar.gz"
    ]
    
    for base_file in base_files:
        matching_files = [f for f in os.listdir(directory) 
                         if f.startswith(base_file.replace('.', '.'))]
        
        if len(matching_files) > 1:
            matching_files.sort()
            print(f"\nFound duplicates for {base_file}:")
            print(f"  Keeping: {matching_files[0]}")
            for duplicate in matching_files[1:]:
                file_path = os.path.join(directory, duplicate)
                try:
                    os.remove(file_path)
                    files_removed += 1
                    print(f"  Removed: {duplicate}")
                except Exception as e:
                    print(f"  Error removing {duplicate}: {str(e)}")
    
    # Part 2: Clean up txt files
    txt_files_to_remove = [
        "train.txt",
        "val.txt",
        "test.txt"
    ]
    
    print("\nChecking for txt files to remove:")
    for txt_file in txt_files_to_remove:
        file_path = os.path.join(directory, txt_file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                files_removed += 1
                print(f"  Removed: {txt_file}")
            except Exception as e:
                print(f"  Error removing {txt_file}: {str(e)}")
    
    if files_removed == 0:
        print("\nNo files found to remove.")
    else:
        print(f"\nTotal files removed: {files_removed}")
    
    return files_removed

if __name__ == "__main__":
    # For local execution
    if os.environ.get("MODAL_ENVIRONMENT") is None:
        parser = argparse.ArgumentParser(description='Clean up duplicate downloaded files.')
        parser.add_argument('--directory', type=str, default="/data",
                          help='Directory to clean (default: /data)')
        
        args = parser.parse_args()
        
        if not os.path.exists(args.directory):
            print(f"Error: Directory '{args.directory}' does not exist.")
            exit(1)
            
        print(f"Cleaning directory: {args.directory}")
        clean_duplicate_files(args.directory)
    else:
        # For Modal execution
        modal.runner.deploy_stub(app)
