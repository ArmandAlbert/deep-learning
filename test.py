import numpy as np
out = np.array([-2, -1, 1, 2])
t = (out <= 0)
out[t] = 0
print(out)
