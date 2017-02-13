from utils import *
import numpy as np
dim = 2
c = np.power(2, dim-1)
#c=None
c=16
#g, true_c = generate_gmm_data_random_orthant(dim=dim, num_cluster=c, var=0.01)
g, true_c, mean = generate_gmm_dense_data(num_cluster=c, var=0.001)
#if c==None:
#    c=10
# print estimate_optimal_cluster_size_gmm(g, clusters=range(1, c+1))
# plot_2d(g, color=true_c/np.max(true_c.flatten()), save_path="transformed.png", axis=None, transform=True)
plot_2d(g, save_path="original.png", axis=None,transform=False)
