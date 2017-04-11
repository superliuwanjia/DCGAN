import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("activation", "relu", "nonlinearity")
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate_g", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("learning_rate_d", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_height", 64, "The height of the output images to produce [64]")
flags.DEFINE_integer("output_width", 64, "The width of the output images (2nd dim) to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("z_dim", 100, "Dimension of z. [100]")
flags.DEFINE_integer("gmm_cluster", None, "number of gmm cluster")
flags.DEFINE_integer("gmm_dim", 2, "dimension of gmm data")
flags.DEFINE_float("gmm_var", 0.02, "variance of generated gmm clusters")
flags.DEFINE_float("gmm_scale", 2, "scale of generated gmm clusters")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("network", "DCGAN", "D G architecture choice")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save the image samples [samples]")
flags.DEFINE_string("cluster_est", "gmm", "method to estimate cluster #, gmm or jump")
flags.DEFINE_string("init_type", "orthogonal", "How to initialize linear layer weights")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("visualize_interval", 5, "save generated samples every [5] batches")
flags.DEFINE_integer("g_heuristic", 0, "True for -log(D) g loss ")
flags.DEFINE_integer("g_update", 2, "Two generator update for 1 discriminator update")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    with tf.Session() as sess:
        dcgan = DCGAN(sess,
                          image_size=FLAGS.image_size,
                          batch_size=FLAGS.batch_size,
                          output_width=FLAGS.output_width,
                          output_height=FLAGS.output_height,
                          c_dim=FLAGS.c_dim,
                          z_dim=FLAGS.z_dim,
                          sample_size=FLAGS.batch_size,
                          dataset_name=FLAGS.dataset,
                          is_crop=FLAGS.is_crop,
                          checkpoint_dir=FLAGS.checkpoint_dir,
                          sample_dir=FLAGS.sample_dir,
                          flags=FLAGS)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)

        # Below is codes for visualization
        # OPTION = 1
        #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
