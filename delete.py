import os
import modal

# Create Modal app and volume
app = modal.App("delete-files")
volume = modal.Volume.from_name("tinystories-volume")

@app.function(volumes={"/data": volume})
def delete_files(directory="/data"):
    """
    Delete specific files from the given directory in the Modal volume.
    """
    files_to_delete = [
        "train.txt",
        "val.txt",
        "test.txt"
    ]
    
    files_removed = 0
    print("\nDeleting specified files:")
    
    for file_name in files_to_delete:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                files_removed += 1
                print(f"  Removed: {file_name}")
            except Exception as e:
                print(f"  Error removing {file_name}: {str(e)}")
        else:
            print(f"  Not found: {file_name}")
    
    if files_removed == 0:
        print("\nNo files were removed.")
    else:
        print(f"\nTotal files removed: {files_removed}")
    
    return files_removed

if __name__ == "__main__":
    # For Modal execution
    modal.runner.deploy_stub(app)
