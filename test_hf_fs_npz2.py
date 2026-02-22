from huggingface_hub import HfFileSystem

fs = HfFileSystem()
# Let's try to read a small npz file from a public repo, or just check if fs.open works with np.load
# We can just create a dummy npz locally, upload it to a test repo, or just check the API.
# Actually, let's just check if fs.open returns a file-like object that np.load can use.
# We'll use a known public repo with npz files if possible, or just mock it.
print("HfFileSystem imported successfully")
