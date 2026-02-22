import io

import numpy as np
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
# Let's create a dummy npz locally
np.savez("dummy.npz", a=np.array([1, 2, 3]), b=np.array([4, 5, 6]))

# Now let's read it using a file-like object
with open("dummy.npz", "rb") as f:
    data = f.read()

f_io = io.BytesIO(data)
npz = np.load(f_io)
print(npz["a"])
print(npz["b"])
