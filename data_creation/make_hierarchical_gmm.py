import sys
sys.path.append('../')
import os
import numpy as np

import utils

data_folder = "/home/ubuntu/s3-gmm"
l1_radius = 3
l2_radius = 0.3
var = 0.001
angle_1 = np.pi / 2
angle_2 = angle_1 + 2*np.pi/3.0
angle_3 = angle_2 + 2*np.pi/3.0
angles = [angle_1, angle_2, angle_3]

means = []
for l1_angle in angles:
    for l2_angle in angles:
        means.append(np.array(
            [l1_radius * np.cos(l1_angle) + l2_radius * np.cos(l2_angle),
             l1_radius * np.sin(l1_angle) + l2_radius * np.sin(l2_angle)]))
print means
data, cluster_assignment, cluster_center = utils.generate_gmm_dense_data(num_data=60000, means=means, var=var)
utils.plot_2d(data,save_path="data.png", axis=[-4,4,-4,4])
to_dump={}
to_dump["X_train"] = data[0:50000,:]
to_dump["Y_train"] = cluster_assignment[0:50000,]
to_dump["X_test"] = data[50000:60000,:]
to_dump["Y_test"] = cluster_assignment[50000:60000,]

to_dump["center"] = cluster_center

np.savez(os.path.join(data_folder, "gmm_hierarchical.npz"), to_dump)
