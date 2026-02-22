import fsspec
import numpy as np

# Let's create a dummy npz locally
np.savez("dummy.npz", a=np.array([1, 2, 3]), b=np.array([4, 5, 6]))

fs_local = fsspec.filesystem("file")
f = fs_local.open("dummy.npz", "rb")
npz = np.load(f)
f.close()
try:
    print(npz["a"])
    print("Works after close!")
except Exception as e:
    print("Failed after close:", e)
