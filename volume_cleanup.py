import os
import modal

app = modal.App("delete-files")
volume = modal.Volume.from_name("tinystories-volume")

@app.function(volumes={"/data": volume})
def delete_files(directory="/data"):
    """Delete specific Python files from Modal volume."""
    files_to_delete = [
        "configurator.py",
        "model.py",
        "tokenizer.py",
        "tinystories.py",
        "train.py",
        "export.py",
        "run.c",
        "run"
    ]
    
    files_removed = 0
    print("\nDeleting files:")
    
    for file_name in files_to_delete:
        file_path = os.path.join(directory, file_name)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                files_removed += 1
                print(f"  Removed: {file_name}")
            else:
                print(f"  Not found: {file_name}")
        except Exception as e:
            print(f"  Error removing {file_name}: {str(e)}")
    
    print(f"\nTotal files removed: {files_removed}")
    return files_removed

if __name__ == "__main__":
    modal.runner.deploy_stub(app)
