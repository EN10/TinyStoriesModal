import modal
import os

app = modal.App("tinystories-simple")
volume = modal.Volume.from_name("tinystories-volume")

MODEL_PATH = {"remote": "/data/out/model.bin", "local": "out/model.bin"}

@app.function(image=modal.Image.debian_slim(), volumes={"/data": volume})
def remote_transfer(action: str, data: bytes = None):
    """Handle remote model transfer"""
    if action == "download":
        if not os.path.exists(MODEL_PATH["remote"]):
            raise FileNotFoundError("No model found in Modal volume")
        with open(MODEL_PATH["remote"], "rb") as f:
            return f.read()
    else:  # upload
        os.makedirs(os.path.dirname(MODEL_PATH["remote"]), exist_ok=True)
        with open(MODEL_PATH["remote"], "wb") as f:
            f.write(data)

@app.local_entrypoint()
def main(action="download", path=MODEL_PATH["local"]):
    """Transfer model between local and Modal volume
    Usage: modal run transfer_model.py [--action=download|upload] [--path=model.bin]"""
    try:
        print(f"{action.title()}ing model...")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if action == "download":
            with open(path, "wb") as f:
                f.write(remote_transfer.remote("download"))
        else:
            with open(path, "rb") as f:
                remote_transfer.remote("upload", f.read())

        print(f"Success! {action.title()}ed {os.path.getsize(path):,} bytes")
    except Exception as e:
        print(f"Error: {e}")
