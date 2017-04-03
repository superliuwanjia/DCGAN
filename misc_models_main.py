import os
import utils
import numpy as np
import misc_models
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("dataset", "berkeley","dataset name")
flags.DEFINE_string("data_dir", "/home/liuwanjia/Documents/DCGAN/data", "Parent folder dir.")
flags.DEFINE_string("plot_dir", "/home/liuwanjia/Documents/DCGAN/plots", "Parent folder dir.")
flags.DEFINE_string("tag", "PCA", "sub plots folder.")
flags.DEFINE_string("model", "PCA", "Model to fit")
flags.DEFINE_string("visualization", "components","Matrix to visualize")
flags.DEFINE_integer("num_components", 20,"Number of components")
flags.DEFINE_float("sparsity", 0.1,"Number of nonzero coefficients")
FLAGS = flags.FLAGS

def main(_):
    data = os.path.join(FLAGS.data_dir, FLAGS.dataset + ".npz")
    print np.load(data)["X_train"].shape
    if FLAGS.model == "PCA":
        model = misc_models.GMM(num_components=FLAGS.num_components)
    elif FLAGS.model == "ICA":
        model = misc_models.ICA(num_components=FLAGS.num_components)
    elif FLAGS.model == "SC":
        model = misc_models.SC(sparsity=FLAGS.sparsity)

    r = model.train(data) 

    if FLAGS.visualization == "covariances":
        utils.visualize_cov(os.path.join(FLAGS.plot_dir, FLAGS.tag), r)
    elif FLAGS.visualization == "components":
        utils.visualize_matrix(os.path.join(FLAGS.plot_dir, FLAGS.tag), r)

if __name__ == '__main__':
    tf.app.run()


