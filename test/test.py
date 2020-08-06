import numpy as np

weight = np.array([0.5,1.,.5,.1,.1,.1])


ft = np.ones(6)
print(np.linalg.norm(ft*weight))
ft = np.zeros(6)
print(np.linalg.norm(ft*weight))
ft = np.array([1,1.,1,0,0,0])
print(np.linalg.norm(ft*weight))
