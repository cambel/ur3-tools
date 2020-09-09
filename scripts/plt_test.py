import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
# Create Map
cm = plt.get_cmap("RdYlGn")

x = np.random.rand(30)
y = np.random.rand(30)
z = np.random.rand(30)
col = np.arange(30)

# # 2D Plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(x, y, s=10, c=col, marker='o')  

# 3D Plot
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
p3d = ax3D.scatter(x, y, z, s=30, c=col, marker='o')                                                                                

plt.show()