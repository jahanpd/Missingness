import UAT.datasets as data
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd

X, y, (x_a, x_b) = data.spiral(2048, missing="MNAR", rng_key=1, p = 0.99)

print(X.shape)
print(np.isnan(X).sum(0))
print(X[np.any(np.isnan(X), axis=1), :].shape)

xdrop = X[~np.any(np.isnan(X), axis=1), :]
ydrop = y[~np.any(np.isnan(X), axis=1)]
x0 = xdrop[ydrop == 0, :2]
x1 = xdrop[ydrop == 1, :2]
print(x0.shape, x1.shape)

fig = plt.figure(figsize=plt.figaspect(0.5))
gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(gs[0, 0])
font_size=10
# ax1.set_title('Spiral Dataset', fontsize=font_size)
im = ax1.scatter(x0[:,0],x0[:,1], s=1, label="0")
im = ax1.scatter(x1[:,0],x1[:,1], s=1, label="1")
ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.legend()

plt.show()