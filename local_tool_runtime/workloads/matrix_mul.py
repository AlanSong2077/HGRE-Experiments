import sys, time
import numpy as np
n = int(sys.argv[1])
rng = np.random.default_rng(42)
a = rng.random((n, n), dtype=np.float64)
b = rng.random((n, n), dtype=np.float64)
c = a @ b
print(float(c[0,0]), n)
