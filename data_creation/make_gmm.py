import sys
sys.path.append('../')
import os
import numpy as np

import utils

data_folder = "/home/ubuntu/s3-gmm"
num_cluster = 4

data, cluster_assignment, cluster_center = utils.generate_gmm_circle_data(num_data=60000, num_cluster=num_cluster)

to_dump={}
to_dump["X_train"] = data[0:50000,:]
to_dump["Y_train"] = cluster_assignment[0:50000,]
to_dump["X_test"] = data[50000:60000,:]
to_dump["Y_test"] = cluster_assignment[50000:60000,]

to_dump["center"] = cluster_center

np.savez(os.path.join(data_folder, "gmm_"+str(num_cluster)+"_center.npz"), to_dump)
