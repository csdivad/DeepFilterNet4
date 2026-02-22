# Upload it to a test repo or just use a local file with fsspec
import fsspec
import numpy as np
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
# Let's create a dummy npz locally
np.savez("dummy.npz", a=np.array([1, 2, 3]), b=np.array([4, 5, 6]))


fs_local = fsspec.filesystem("file")
with fs_local.open("dummy.npz", "rb") as f:
    npz = np.load(f)
    print(npz["a"])
