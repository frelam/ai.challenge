import numpy as np
out1 = np.load('./out1.npy')
out2 = np.load('./out2.npy')
out3 = np.load('./out3.npy')
out4 = np.load('./out4.npy')

out = out1

indices = (-out).argsort()[:,:3]

np.save('./top3.npy',indices)