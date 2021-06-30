import numpy as np
import matplotlib.pyplot as plt

test = np.load("/Users/robin/PycharmProjects/DeepCCA/examples/new_features.npy", allow_pickle=True)

test_data, test_pred, test_label = test[2]

plt.plot(test_data[:,0], test_pred[:,0], 'ro')
plt.show()
