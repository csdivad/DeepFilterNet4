from huggingface_hub import HfFileSystem

fs = HfFileSystem()
# Let's try to read a small npz file from a public repo, or just check if fs.open works with np.load
# We can just create a dummy npz locally, upload it to a test repo, or just check the API.
print("HfFileSystem imported successfully")
