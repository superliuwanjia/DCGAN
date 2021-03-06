from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *


class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, sample_size=64, output_height=64,
                 output_width=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None, num_test_images=0,
                 flags=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_height: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.flags = flags

        self.sess = sess

        self.num_test_images = num_test_images
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        # Attribute y_dim==False indicate no label is used,
        # and the third discriminator-related batch normalization will be used
        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        # Attribute y_dim==False indicate no label is used,
        # and the third generator-related batch normalization will be used
        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.process_input()
        self.prepare_nonlin()
        self.build_model()

    def prepare_nonlin(self):
        '''
        Consider two kinds of nonlinearities: relu and tanh
        :return:
        '''
        if self.flags.activation == "relu":
            self.nl = lrelu
        elif self.flags.activation == "tanh":
            self.nl = tf.nn.tanh
        elif self.flags.activation == "lin":
            self.nl = lin

    def process_input(self):
        '''
        Before feeding input data into model, dealing with input w.r.t. different types of data
        :return:
        '''
        if not os.path.exists(self.flags.sample_dir):
            os.mkdir(self.flags.sample_dir)

        # Load MNIST data
        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()

        # Generate GMM data
        elif self.dataset_name == 'GMM':
            data, label, mean = generate_gmm_data(dim=self.flags.gmm_dim, num_cluster=self.flags.gmm_cluster,
                                                  var=self.flags.gmm_var, scale=self.flags.gmm_scale)

            # TODO: Why do we not set self.c_dim=1 explicitly here?
            self.data_X = data.reshape([data.shape[0], 1, self.flags.gmm_dim, 1])
            self.output_height = 1
            self.cluster_mean = mean
            self.output_width = self.flags.gmm_dim
            plot_2d(data[0:1000, :], save_path=self.flags.sample_dir + "/dataset.png", axis=None, transform=False)

        # Generate GMM Circle data
        elif self.dataset_name == "GMM_CIRCLE":
            data, label, mean = generate_gmm_circle_data(num_cluster=self.flags.gmm_cluster, var=self.flags.gmm_var,
                                                         scale=self.flags.gmm_scale)

            # TODO: Why do we not set self.c_dim=1 explicitly here?
            self.data_X = data.reshape([data.shape[0], 1, self.flags.gmm_dim, 1])
            self.output_height = 1
            self.cluster_mean = mean
            self.output_width = 2
            plot_2d(data[0:1000, :], save_path=self.flags.sample_dir + "/dataset.png", axis=None, transform=False)

        # Generate GMM dense data (need to use gmm_scale argument)
        elif self.dataset_name == "GMM_DENSE":
            data, label, mean = generate_gmm_dense_data(dim=self.flags.gmm_dim, num_cluster=self.flags.gmm_cluster,
                                                        var=self.flags.gmm_var, scale=self.flags.gmm_scale)

            # TODO: Why do we not set self.c_dim=1 explicitly here?
            self.data_X = data.reshape([data.shape[0], 1, self.flags.gmm_dim, 1])
            self.output_height = 1
            self.cluster_mean = mean
            self.output_width = 2
            plot_2d(data[0:1000, :], save_path=self.flags.sample_dir + "/dataset.png", transform=False,
                    axis=[-2, self.flags.gmm_scale + 2, -2, self.flags.gmm_scale + 2])

        elif self.dataset_name == "celebA":
            pass
        # Processing any arbitrary input images
        else:
            # The method glob() will return a list of pathnames which mataches the path patterns
            data = glob(os.path.join("./data", self.dataset_name, "*.jpg"))
            batch_files = data[0:len(data) - self.num_test_images]

            # The method get_image() will return a processed image
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_height,
                               is_grayscale=self.is_grayscale) for batch_file in batch_files]

            # Determine if it is grayscale or colored image
            if (self.is_grayscale):
                self.data_X = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                self.data_X = np.array(batch).astype(np.float32)
        

    def build_model(self):
        """
        Build a model with discriminator and generator and calculate the d-loss and g-loss
        :return:
        """
        # Set placeholders for y, images, sample_images and z
        # y stands for labels of a batch of images

        self.images = tf.placeholder(tf.float32, [self.batch_size] +
                                     [self.output_height, self.output_width, self.c_dim], name='real_images')

        # self.sample_images is not used
        self.sample_images = tf.placeholder(tf.float32, [self.sample_size] +
                                            [self.output_height, self.output_width, self.c_dim], name='sample_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        # Set a list to show relu_state
        self.relu_state = []

        # noise of samples generated, approximated as I - medianfilt(I)   
        self.avg_noise = tf.placeholder(tf.float32, None, name='avg_noise')
        self.avg_noise_sum = scalar_summary("avg_noise_sum", self.avg_noise)

        # noise of training images used, base line
        self.train_avg_noise = tf.placeholder(tf.float32, None, name='train_avg_noise')
        self.train_avg_noise_sum = scalar_summary(
            "train_avg_noise_sum", self.train_avg_noise)

        # The placeholder est_clusters is used for estimating how many clusters in GMM data case
        self.est_clusters = tf.placeholder(tf.float32, None, name='estimated_cluster_count')
        self.est_clusters_sum = scalar_summary("estimated_clusters_count_summary", self.est_clusters)

        # Initialize the average image of all images used during training
        # TODO: Instance attributes should be defined in the __init__() method
        self.cum_image = np.zeros([1, self.output_height, self.output_width, self.c_dim])

        # Initialize the number of images used (in batch size)
        self.num_images_used = 0

        # self.mean_image = tf.placeholder(tf.float32, [1, self.output_height, self.output_width, self.c_dim],
        #                                  name='mean_image')

        # Calculate the output of generator and discriminator separately
        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.images)

        # TODO: What is sampler used for? attribute and method have the same name
        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    
        print "G shape:",self.G.get_shape().as_list()
        print "Z shape:",self.z.get_shape().as_list()
        print "D shape:",self.D_logits_.get_shape().as_list()

        # Calcalute the accuracy of discriminator for real and fake image
        self.real_accu = tf_accuracy(self.D, 1, self.batch_size)
        self.fake_accu = tf_accuracy(self.D_, 0, self.batch_size)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)
        self.real_accu_sum = scalar_summary("real_accu_sum", self.real_accu)
        self.fake_accu_sum = scalar_summary("fake_accu_sum", self.fake_accu)

        # Calcalute the loss for discriminator
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        # Calculate the loss for generator in vanilla GAN, -logD trick and reverse KL, maximum likelihood
        
        likelihood_ratio = self.D_ / (1 - self.D_)
        
        if self.flags.g_objective == "-logDtrick":
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            self.D_logits_, tf.ones_like(self.D_)))
        else:
            self.g_loss = tf.reduce_mean(f_divergence(likelihood_ratio, \
                    option=self.flags.g_objective, alpha=self.flags.alpha)) 
        
        # Put variables w.r.t. discriminator and generator in two different lists so as to do gradients easily
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # Initialize a Saver instance to save all the variables into checkpoint file
        self.saver = tf.train.Saver()

    def train(self, config):
        """
        Train DCGAN
        """

        # np.random.shuffle(data)

        # -----------------------------------------------------------------
        # Training Discriminator!
        # Use AdamOptimizer
        # TODO: Is the index tf.gradients()[0] indicate we just consider one parameter or weight?
        # TODO: Why is using so many instance attributes (e.g. self.d_grads, self.d_vars)?
        d_optim = tf.train.AdamOptimizer(config.learning_rate_d, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)

        # Comupte gradients of d_loss w.r.t. d_vars
        self.d_grads = tf.gradients(self.d_loss, self.d_vars)[0]
        self.d_grads_sum = tf.histogram_summary("d_grad_sum", self.d_grads)

        # -----------------------------------------------------------------
        # Training Generator! [self.g_loss]
        g_optim = tf.train.AdamOptimizer(config.learning_rate_g, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        # Use a g_loss_sum to summarize three cases
        # TODO: Why is treating g_loss and g_grads differently
        self.g_loss_sum = scalar_summary("g_loss_" + self.flags.g_objective, self.g_loss)
        self.g_grads = tf.gradients(self.g_loss, self.g_vars)[0]

        self.g_grads_sum = tf.histogram_summary("g_grad_sum", self.g_grads)
        

        # Initialize all variables before session runs.
        # Consider TF version compatibility using exception
        try:
            tf.initialize_all_variables().run()
        except:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

        # Merge generator and discriminator summaries separately
        self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum,
                                    self.g_loss_sum, self.g_sums, self.g_grads_sum])
        # TODO: There are two self.d_sum?
        self.d_sum = merge_summary([self.real_accu_sum, self.fake_accu_sum, self.z_sum, self.d_sum,
                                    self.d_loss_real_sum, self.d_loss_sum, self.d_real_sums,
                                    self.d_fake_sums, self.d_grads_sum])

        # Define a SummaryWriter instance with log_dir to write summaries into event file
        self.writer = SummaryWriter("./" + self.flags.log_dir, self.sess.graph)

        # Input sample noise z is uniform [-1,1] with a batch_size=sample_size
        # sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        # sample_z = np.random.uniform(-1, 1, [config.sample_size, self.z_dim]).astype(np.float32)
        sample_z = np.random.normal(0, 1, [self.sample_size, self.z_dim]).astype(np.float32)

        # TODO: Why is just fetching the first sample batch of images (or labels when dealing with mnist)
        if config.dataset == 'mnist':
            sample_images = data_X[0:self.sample_size]
            sample_labels = data_y[0:self.sample_size]
        elif config.dataset == 'celebA':
            data = glob(os.path.join("./data", config.dataset, "*.jpg"))
            sample_files = data[0:self.sample_size]
            sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_height, is_grayscale = self.is_grayscale) for sample_file in sample_files]
            if (self.is_grayscale):
               sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
               sample_images = np.array(sample).astype(np.float32)
        else: 
            # sample_files = data[0:self.sample_size]
            # sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_height, is_grayscale = self.is_grayscale) for sample_file in sample_files]
            # if (self.is_grayscale):
            #    sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            # else:
            #    sample_images = np.array(sample).astype(np.float32)
            sample_images = self.data_X[0:self.sample_size]

        # Initialize a counter to act as a global step
        counter = 0

        # Set starting time
        start_time = time.time()

        # Restore the variables from checkpoint file so not need to train variables from initial random values
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        print "Start Training..."
        # for-loop: Each iteration is one epoch
        for epoch in xrange(config.epoch):
            if config.dataset == 'mnist':
               batch_idxs = min(len(data_X), config.train_size) // config.batch_size
            elif config.dataset == "celebA":
               data = glob(os.path.join("./data", config.dataset, "*.jpg"))
               batch_idxs = min(len(data), config.train_size) // config.batch_size
            else:
            # Calculate the number of batches, where train_size is the maximum tolerable batch number
                batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size

            # iterate training data from the first batch (batch_images, batch_labels) to the last one
            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
                elif config.dataset == 'celebA':
                    batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_height, is_grayscale = self.is_grayscale) for batch_file in batch_files]
                    if (self.is_grayscale):
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)


                else:
                    # batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                    # batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_height, is_grayscale = self.is_grayscale) for batch_file in batch_files]
                    # if (self.is_grayscale):
                    #    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    # else:
                    #    batch_images = np.array(batch).astype(np.float32)

                    batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]

                # save mean of all images used for training
                # TODO: But this instance attribute is not used
                self.cum_image += np.mean(batch_images, axis=0, keepdims=True)

                # Increase num_images_used by 1 per batch
                self.num_images_used += 1

                # Input a nose z in one batch size with uniform [-1,1] or [0,1] distribution
                # TODO: Which one do you prefer to use?
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)
                batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # -----------------------------------------------------------------
                # Update generator and discriminator w.r.t. d_optim and g_optim(_*)
                # y stands for labels of some kind of labelled input images (e.g. mnist 0-9)

                # gather summary before training
                if counter == 0:
                    summary_str_d, summary_str_g = self.sess.run(
                        [self.d_sum, self.g_sum], feed_dict={self.images: batch_images, self.z: batch_z}
                    )
                    self.writer.add_summary(summary_str_d, -1)
                    self.writer.add_summary(summary_str_g, -1)

                # Update D network
                _, summary_str = self.sess.run(
                    [d_optim, self.d_sum], feed_dict={self.images: batch_images, self.z: batch_z}
                )
                self.writer.add_summary(summary_str, counter)

                for _ in range(self.flags.g_update):
                    # Update G network
                    _, summary_str = self.sess.run(
                        [g_optim, self.g_sum], feed_dict={self.z: batch_z, self.images: batch_images}
                    )
                    self.writer.add_summary(summary_str, counter)


                # Print loss etc Every 10 batch!
                if np.mod(counter, 10) == 0:
                    # Print d_fake, samples, d_loss and g_loss with a batch of sample_z,
                    # sample_images and sample labels.
                    # TODO: Is self.relu_state an empty list all the time?
                    result = list(self.sess.run(
                        [self.D_, self.sampler, self.d_loss, self.g_loss] + self.relu_state,
                        feed_dict={self.z: sample_z, self.images: sample_images}
                    ))
                    d_fake = result[0]
                    samples = result[1]
                    d_loss = result[2]
                    g_loss = result[3]
                    relu_state = result[4:]

                    # Print d_loss with fake and real images, and g_loss with vanilla, -logD and reverse KL
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.images: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z, self.images: batch_images})

                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss_%s: %.8f"
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake + errD_real, self.flags.g_objective ,errG))

                    # Write the average noise from (random) samples into event file
                    summary_str = self.sess.run(self.avg_noise_sum,
                                            feed_dict={self.avg_noise: avg_noise(samples)})
                    self.writer.add_summary(summary_str, counter)

                    # Write the average noise from training images into event file as well
                    summary_str = self.sess.run(self.train_avg_noise_sum,
                                            feed_dict={self.train_avg_noise: avg_noise(batch_images)})
                    self.writer.add_summary(summary_str, counter)

                # Every config.visualize_interval=5, save images to config.sample_dir path or analyze_gmm
                if np.mod(counter, config.visualize_interval) == 0:
                    if config.dataset == 'mnist':
                        save_images(samples, [8, 8],
                                    './{}/train_{:06d}.png'.format(config.sample_dir, counter))
                    elif config.dataset[0:3] == "GMM":
                        samples = samples.reshape([self.batch_size, self.flags.gmm_dim])
                        self.analyze_gmm(counter, samples, relu_state)
                    else:
                        save_images(samples, [8, 8],
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        # visualize(self.sess, self, self.flags, 5, seq=counter)
                        # mean_image_sum = image_summary(
                        #     "mean_image_sum_{:06d}".format(counter), self.mean_image)

                        # summary_str = self.sess.run(mean_image_sum,
                        #     feed_dict={ self.mean_image: self.cum_image / self.num_images_used })

                        # self.writer.add_summary(summary_str, counter)
                        # visualize batch mean
                        # mean_batch_image_sum = image_summary(
                        #     "mean_batch_image_sum_{:06d}".format(counter), self.mean_image)

                        # summary_str = self.sess.run(mean_batch_image_sum,
                        #     feed_dict={ self.mean_image: np.mean(batch_images, axis=0, keepdims=True) })

                        # self.writer.add_summary(summary_str, counter)

                # Every 500 batches, save variables to checkpoint file with checkpoint_dir path
                if np.mod(counter, 500) == 0:
                    self.save(config.checkpoint_dir, counter)

                # Every batch, counter increase by 1
                counter += 1

    def analyze_gmm(self, counter, samples, relu_state=None):
        '''
        Plot and store 2D gmm samples with estimated clusters and relu states, separately
        :param counter:
        :param samples:
        :param relu_state:
        :return:
        '''
        config = self.flags

        estimated_clusters = 0
        label = []
        # Decide by config.cluster_est whether to use gmm or jump estimate method
        if self.flags.cluster_est == "gmm":
            estimated_clusters, label, gmm = estimate_optimal_cluster_size_gmm(
                samples, clusters=range(1, self.flags.gmm_cluster + 1)
            )
            # samples, label = gmm.sample(self.batch_size)
        elif self.flags.cluster_est == "jump":
            estimated_clusters, label = estimate_optimal_cluster_size_jump(
                samples, clusters=range(1, self.flags.gmm_cluster + 1)
            )

        # Write the est_clusters into the event file
        summary_str = self.sess.run(self.est_clusters_sum,
                                    feed_dict={self.est_clusters: estimated_clusters})
        self.writer.add_summary(summary_str, counter)

        # Assign colors w.r.t. the labels
        if not np.max(label) == 0:
            colors = label / (np.max(label) + 0.0)
        else:
            colors = label

        # Plot samples in a 2D plane
        # 1) gmm dimension greater than 2, 2) gmm dimension is or less than 2
        # Use boolean-type arg "transform" to indicate whether its dimensional is greater than 2
        if self.flags.gmm_dim > 2:
            # TODO: The "axis" is not used
            axis = [-2, self.flags.gmm_scale + 2, -2, self.flags.gmm_scale + 2]
            plot_2d(samples, center=self.cluster_mean, title="Minibatch: " + str(counter), axis=None, color=colors,
                    save_path='./{}/train_{:06d}.png'.format(config.sample_dir, counter), transform=True)

        else:
            # The case when dataset is GMM_DENSE, using gmm_scale for axis
            if self.flags.dataset == "GMM_DENSE":
                plot_2d(samples, center=self.cluster_mean, title="Minibatch: " + str(counter),
                        axis=[-1, self.flags.gmm_scale + 2, -1, self.flags.gmm_scale + 2], color=colors,
                        save_path='./{}/train_{:06d}.png'.format(config.sample_dir, counter), transform=False)

            # The case when dataset is another one, not using gmm_scale for axis
            else:
                plot_2d(samples, center=self.cluster_mean, title="Minibatch: " + str(counter), axis=[-4, 4, -4, 4],
                        color=colors, save_path='./{}/train_{:06d}.png'.format(config.sample_dir, counter),
                        transform=False)

        # relu state distribution (only non-empty for GMM data)
        if relu_state and len(relu_state) > 0:

            # Convert float vector to bit vector (np.hstack used for producing a vector)
            relu_state = (np.hstack(relu_state) > 0).astype(int)

            # convert bit vector to decimal
            relu_state_decimal = []
            # TODO: What if rs is a scalar and rs.shape[-1] shows "tuple index out of range"
            for rs in relu_state:
                relu_state_decimal.append(rs.dot(1 << np.arange(rs.shape[-1] - 1, -1, -1)))

            # Assign colors w.r.t. relu_state in decimal format
            label = relu_state_decimal
            if not np.max(label) == 0:
                colors = label / (np.max(label) + 0.0)
            else:
                colors = label

            # Plot samples in a 2D plane, use colors to indicate relu state
            # The plots are stored in the relu_state_counter name
            if self.flags.gmm_dim > 2:
                # TODO: The "axis" is not used
                axis = [-2, self.flags.gmm_scale + 2, -2, self.flags.gmm_scale + 2]
                plot_2d(samples, center=self.cluster_mean, title="Minibatch: " + str(counter),
                        axis=None, color=colors,
                        save_path='./{}/relu_state_{:06d}.png'.format(config.sample_dir, counter),
                        transform=True
                        )
            else:
                if self.flags.dataset == "GMM_DENSE":
                    plot_2d(samples, center=self.cluster_mean, title="Minibatch: " + str(counter),
                            axis=[-1, self.flags.gmm_scale + 2, -1, self.flags.gmm_scale + 2], color=colors,
                            save_path='./{}/relu_state_{:06d}.png'.format(config.sample_dir, counter),
                            transform=False
                            )
                else:
                    plot_2d(samples, center=self.cluster_mean, title="Minibatch: " + str(counter),
                            axis=[-4, 4, -4, 4], color=colors,
                            save_path='./{}/relu_state_{:06d}.png'.format(config.sample_dir, counter),
                            transform=False
                            )

    def discriminator(self, image, y=None, reuse=False):
        '''
        Implement discriminator
        :param image:
        :param y:
        :param reuse:
        :return: sigmoid output and non-sigmoid output
        '''

        # "init" denotes how to initialize the weights
        init = self.flags.init_type
        with tf.variable_scope("discriminator") as scope:

            # The parameter "reuse" could control if variables will be shared
            # For real data, reuse is false; for fake data, reuse is true so that the variables created
            # in the first discriminator() method can be shared with the second discriminator() method
            if reuse:
                scope.reuse_variables()

            # Case 1: Use no labels in input images
            # TODO: Can we use a function to build the discriminator network with num_layers as an argument?
            if not self.y_dim:
                layers = []

                # A network architecture "GMM_XLARGE" (9layer-MLP)
                if self.flags.network == "GMM_XLARGE":
                    # Input a batch of images and shape them into a 2D tensor for linear layer
                    image = tf.reshape(image, [-1, self.output_width * self.output_height * self.c_dim])
                    # A linear layer with scope 'd_h[0-7]_lin'
                    h0_, h0_w, h0_b = linear(image, 128, 'd_h0_lin', init_type=init, with_w=True)
                    # nonlinear op, 'relu' or 'tanh'
                    h0 = self.nl(h0_)

                    # TODO: Can we write them by using a for-loop?
                    h1_, h1_w, h1_b = linear(h0, 128, 'd_h1_lin', init_type=init, with_w=True)
                    h1 = self.nl(h1_)
                    h2_, h2_w, h2_b = linear(h1, 128, 'd_h2_lin', init_type=init, with_w=True)
                    h2 = self.nl(h2_)
                    h3_, h3_w, h3_b = linear(h2, 128, 'd_h3_lin', init_type=init, with_w=True)
                    h3 = self.nl(h3_)
                    h4_, h4_w, h4_b = linear(h3, 128, 'd_h4_lin', init_type=init, with_w=True)
                    h4 = self.nl(h4_)
                    h5_, h5_w, h5_b = linear(h4, 128, 'd_h5_lin', init_type=init, with_w=True)
                    h5 = self.nl(h5_)
                    h6_, h6_w, h6_b = linear(h5, 128, 'd_h6_lin', init_type=init, with_w=True)
                    h6 = self.nl(h6_)
                    h7_, h7_w, h7_b = linear(h6, 128, 'd_h7_lin', init_type=init, with_w=True)
                    h7 = self.nl(h7_)

                    h8, h8_w, h8_b = linear(h7, 1, 'd_h8_lin', init_type=init, with_w=True)
                    layers = [h0, h1, h2, h3, h4, h5, h6, h7, h8]

                # A network architecture "GMM_LARGE" (5layer-MLP)
                if self.flags.network == "GMM_LARGE":
                    image = tf.reshape(image, [-1, self.output_width * self.output_height * self.c_dim])
                    h0_, h0_w, h0_b = linear(image, 128, 'd_h0_lin', init_type=init, with_w=True)
                    h0 = self.nl(h0_)
                    h1_, h1_w, h1_b = linear(h0, 128, 'd_h1_lin', init_type=init, with_w=True)
                    h1 = self.nl(h1_)

                    h2_, h2_w, h2_b = linear(h1, 128, 'd_h2_lin', init_type=init, with_w=True)
                    h2 = self.nl(h2_)
                    h3_, h3_w, h3_b = linear(h2, 128, 'd_h3_lin', init_type=init, with_w=True)
                    h3 = self.nl(h3_)

                    h4, h4_w, h4_b = linear(h3, 1, 'd_h4_lin', init_type=init, with_w=True)
                    layers = [h0, h1, h2, h3, h4]

                # A network architecture "GMM_MEDIUM" (3layer-MLP)
                elif self.flags.network == "GMM_MEDIUM":
                    image = tf.reshape(image, [-1, self.output_width * self.output_height * self.c_dim])
                    h0_, h0_w, h0_b = linear(image, 128, 'd_h0_lin', init_type=init, with_w=True)
                    h0 = self.nl(h0_)
                    h1_, h1_w, h1_b = linear(h0, 128, 'd_h1_lin', init_type=init, with_w=True)
                    h1 = self.nl(h1_)

                    h2, h2_w, h2_b = linear(h1, 1, 'd_h2_lin', init_type=init, with_w=True)
                    layers = [h0, h1, h2]

                # A network architecture "GMM_MEDIUM" (2layer-MLP)
                elif self.flags.network == "GMM_SMALL":
                    image = tf.reshape(image, [-1, self.output_width * self.output_height * self.c_dim])
                    h0_, h0_w, h0_b = linear(image, 128, 'd_h0_lin', init_type=init, with_w=True)
                    h0 = self.nl(h0_)
                    # h0 = tf.nn.tanh(h0_)

                    h1, h1_w, h1_b = linear(h0, 1, 'd_h1_lin', init_type=init, with_w=True)
                    layers = [h0, h1]

                # A network architecture "DCGAN" (5layer-CNN with batch normalization)
                elif self.flags.network == "DCGAN":
                    print "in dcgan discriminator"
                    # Use leaky relu as a nonlinear activation,
                    # and use batch normalization in each hidden layer
                    # TODO: 'h0_' is not well-defined, h0_ should be a 4D tensor, it is?
                    h0_ = conv2d(image, self.df_dim, name='d_h0_conv')
                    h0 = lrelu(h0_)

                    # The parameter self.df_dim*N indicates the output dimension of conv2d
                    h1_ = conv2d(h0, self.df_dim * 2, name='d_h1_conv')
                    h1 = lrelu(self.d_bn1(h1_))

                    h2_ = conv2d(h1, self.df_dim * 4, name='d_h2_conv')
                    h2 = lrelu(self.d_bn2(h2_))

                    h3_ = conv2d(h2, self.df_dim * 8, name='d_h3_conv')
                    h3 = lrelu(self.d_bn3(h3_))

                    h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                    print h0.get_shape().as_list()
                    print h1.get_shape().as_list()
                    print h2.get_shape().as_list()
                    print h3.get_shape().as_list()
 
                    # TODO: Why is using h0_ instead of h0?
                    layers = [h0_, h1_, h2_, h3_, h4]

                # Use 'd_[real, fake]_sums' to write snapshot value of each layer output into tf.histogram_summary
                if not reuse:
                    self.d_real_sums = []
                    for layer in layers:
                        self.d_real_sums.append(tf.histogram_summary("d_real_sums_" + layer.name, layer))
                else:
                    self.d_fake_sums = []
                    for layer in layers:
                        self.d_fake_sums.append(tf.histogram_summary("d_fake_sums_" + layer.name, layer))

                return tf.nn.sigmoid(layers[-1]), layers[-1]

            # Case 2: Use labels in input images (e.g. MNIST)
            else:
                # conv_cond_concat: Try to concatenate tensors image and label along the axis=3 dimension
                # TODO: Why is the concatenation necessary in h[0-2]?
                # TODO: A way to incorporate the label information in the input?
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                # Use conv2d and leaky_relu in first layer
                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                # Use conv2d, batch normalization and leaky_relu in second layer
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = tf.concat(1, [h1, y])

                # Use full connection in third layer
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tf.concat(1, [h2, y])

                # Use linear output to be connected to sigmoid in fourth layer
                h3 = linear(h2, 1, 'd_h3_lin')

                layers = [h0, h1, h2, h3]


                # Use 'd_[real, fake]_sums' to write snapshot value of each layer output into tf.histogram_summary
                if not reuse:
                    self.d_real_sums = []
                    for layer in layers:
                        self.d_real_sums.append(tf.histogram_summary("d_real_sums_" + layer.name, layer))
                else:
                    self.d_fake_sums = []
                    for layer in layers:
                        self.d_fake_sums.append(tf.histogram_summary("d_fake_sums_" + layer.name, layer))

                return tf.nn.sigmoid(layers[-1]), layers[-1]

    def generator(self, z, y=None):
        '''
        Implement the generator
        :param z:
        :param y:
        :return:
        '''

        # "init" denotes how to initialize the weights
        init = self.flags.init_type
        #  variable scope 'generator'
        with tf.variable_scope("generator") as scope:

            # Case 1: Use no labels to output images
            if not self.y_dim:

                # 'GMM_LARGE' network architecture (9layer-MLP)
                # TODO: Why is there not 'GMM_XLARGE' network architecture?
                if self.flags.network == "GMM_LARGE":

                    # Input is one batch of noise z
                    h0_, h0_w, h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = self.nl(h0_)

                    h1_, h1_w, h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = self.nl(h1_)

                    h2_, h2_w, h2_b = linear(h1, 128, 'g_h2_lin', init_type=init, with_w=True)
                    h2 = self.nl(h2_)

                    h3_, h3_w, h3_b = linear(h2, 128, 'g_h3_lin', init_type=init, with_w=True)
                    h3 = self.nl(h3_)

                    h4_, h4_w, h4_b = linear(h3, 128, 'g_h4_lin', init_type=init, with_w=True)
                    h4 = self.nl(h4_)

                    h5_, h5_w, h5_b = linear(h4, 128, 'g_h5_lin', init_type=init, with_w=True)
                    h5 = self.nl(h5_)

                    h6_, h6_w, h6_b = linear(h5, 128, 'g_h6_lin', init_type=init, with_w=True)
                    h6 = self.nl(h6_)

                    h7_, h7_w, h7_b = linear(h6, 128, 'g_h7_lin', init_type=init, with_w=True)
                    h7 = self.nl(h7_)

                    # Output dimension is 'batch_size*1*gmm_dim*1'
                    h8, h8_w, h8_b = linear(h7, self.flags.gmm_dim, 'g_h8_lin', init_type=init, with_w=True)
                    h8 = tf.reshape(h8, [-1, self.output_height, self.output_width, self.c_dim])

                    layers = [h0, h1, h2, h3, h4, h5, h6, h7, h8]

                    # Use 'g_sums' to write snapshot value of each layer output into tf.histogram_summary
                    self.g_sums = []
                    for layer in layers:
                        self.g_sums.append(tf.histogram_summary("g_sum_" + layer.name, layer))

                    # return one batch of generated images
                    return layers[-1]

                # 'GMM_MEDIUM' network architecture (5layer-MLP)
                elif self.flags.network == "GMM_MEDIUM":
                    h0_, h0_w, h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = self.nl(h0_)

                    h1_, h1_w, h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = self.nl(h1_)

                    h2_, h2_w, h2_b = linear(h1, 128, 'g_h2_lin', init_type=init, with_w=True)
                    h2 = self.nl(h2_)

                    h3_, h3_w, h3_b = linear(h2, 128, 'g_h3_lin', init_type=init, with_w=True)
                    h3 = self.nl(h3_)

                    h4, h4_w, h4_b = linear(h3, self.flags.gmm_dim, 'g_h4_lin', init_type=init, with_w=True)
                    h4 = tf.reshape(h4, [-1, self.output_height, self.output_width, self.c_dim])
                    layers = [h0, h1, h2, h3, h4]

                    # Use 'g_sums' to write snapshot value of each layer output into tf.histogram_summary
                    self.g_sums = []
                    for layer in layers:
                        self.g_sums.append(tf.histogram_summary("g_sum_" + layer.name, layer))

                    return layers[-1]

                # 'GMM_SMALL' network architecture (3layer-MLP)
                elif self.flags.network == "GMM_SMALL":
                    h0_, h0_w, h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = self.nl(h0_)

                    h1_, h1_w, h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = self.nl(h1_)

                    # TODO: Why is self.relu_state commented out? Do we only consider relu_state in GMM_SMALL?
                    # self.relu_state.append(h0_)
                    # self.relu_state.append(h1_)

                    h2, h2_w, h2_b = linear(h1, self.flags.gmm_dim, 'g_h2_lin', init_type=init, with_w=True)
                    h2 = tf.reshape(h2, [-1, self.output_height, self.output_width, self.c_dim])
                    layers = [h0, h1, h2]

                    # Use 'g_sums' to write snapshot value of each layer output into tf.histogram_summary
                    self.g_sums = []
                    for layer in layers:
                        self.g_sums.append(tf.histogram_summary("g_sum_" + layer.name, layer))

                    return layers[-1]

                # DCGAN use a deconvolutional NN (5layer-DeconvNN with batch normalization)
                elif self.flags.network == "DCGAN":
                    print "in dcgan generator"
                    s = self.output_height
                    s2, s4, s8, s16 = int(math.ceil(s / 2)), int(math.ceil(s / 4)), int(math.ceil(s / 8)), int(math.ceil(s / 16))

                    # project `z` and reshape, a linear fully-connected layer
                    self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s16 * s16, 'g_h0_lin', with_w=True)

                    # TODO: We should try to avoid using global variables in functions (e.g. self.h0)
                    self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
                    # Specify relu activation nd batch normalization here
                    h0 = tf.nn.relu(self.g_bn0(self.h0))

                    self.h1, self.h1_w, self.h1_b = deconv2d(
                        h0, [self.batch_size, s8, s8, self.gf_dim * 4], name='g_h1', with_w=True
                    )
                    h1 = tf.nn.relu(self.g_bn1(self.h1))

                    h2, self.h2_w, self.h2_b = deconv2d(
                        h1, [self.batch_size, s4, s4, self.gf_dim * 2], name='g_h2', with_w=True
                    )
                    h2 = tf.nn.relu(self.g_bn2(h2))

                    h3, self.h3_w, self.h3_b = deconv2d(
                        h2, [self.batch_size, s2, s2, self.gf_dim * 1], name='g_h3', with_w=True)

                    h3 = tf.nn.relu(self.g_bn3(h3))

                    h4, self.h4_w, self.h4_b = deconv2d(
                        h3, [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True
                    )

                    print h0.get_shape().as_list()
                    print h1.get_shape().as_list()
                    print h2.get_shape().as_list()
                    print h3.get_shape().as_list()
                    print h4.get_shape().as_list()
                    layers = [h0, h1, h2, h3, h4]

                    self.g_sums = []
                    for layer in layers:
                        self.g_sums.append(tf.histogram_summary("g_sum_" + layer.name, layer))

                    # TODO: Why is using 'tanh' in the last layer? To normalize the output?
                    return tf.nn.tanh(h4)

            # Case 2: use input labels such as [0,9] to output MNIST [4layer-DeconvNN with batch norm]
            else:
                s = self.output_height
                s2, s4 = int(s / 2), int(s / 4)

                # Try to concatenate input noise z and label y or yb along the axis=1 dimension
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                # project a concatenated 'z' and reshape, a linear fully-connected layer with batch norm and relu
                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat(1, [h0, y])

                # Still using linear fully-connected with batch norm and relu
                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s4 * s4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                # Deconv layer
                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                # Deconv layer
                # TODO: Why is using sigmoid? To set image point value between [0,1]?
                # TODO: Why is not using histogram_summary in this case (MNIST)?
                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        '''
        Implement the sampler which is a generator with input 'sample_z'
        :param z:
        :param y:
        :return:
        '''

        init = self.flags.init_type
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                if self.flags.network == "GMM_LARGE":

                    h0_, h0_w, h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = self.nl(h0_)

                    h1_, h1_w, h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = self.nl(h1_)

                    h2_, h2_w, h2_b = linear(h1, 128, 'g_h2_lin', init_type=init, with_w=True)
                    h2 = self.nl(h2_)
                    h3_, h3_w, h3_b = linear(h2, 128, 'g_h3_lin', init_type=init, with_w=True)
                    h3 = self.nl(h3_)
                    h4_, h4_w, h4_b = linear(h3, 128, 'g_h4_lin', init_type=init, with_w=True)
                    h4 = self.nl(h4_)
                    h5_, h5_w, h5_b = linear(h4, 128, 'g_h5_lin', init_type=init, with_w=True)
                    h5 = self.nl(h5_)
                    h6_, h6_w, h6_b = linear(h5, 128, 'g_h6_lin', init_type=init, with_w=True)
                    h6 = self.nl(h6_)
                    h7_, h7_w, h7_b = linear(h6, 128, 'g_h7_lin', init_type=init, with_w=True)
                    h7 = self.nl(h7_)

                    h8, h8_w, h8_b = linear(h7, self.flags.gmm_dim, 'g_h8_lin', init_type=init, with_w=True)
                    h8 = tf.reshape(h8, [-1, self.output_height, self.output_width, self.c_dim])

                    return h8

                elif self.flags.network == "GMM_MEDIUM":
                    h0_, h0_w, h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = self.nl(h0_)

                    h1_, h1_w, h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = self.nl(h1_)

                    h2_, h2_w, h2_b = linear(h1, 128, 'g_h2_lin', init_type=init, with_w=True)
                    h2 = self.nl(h2_)

                    h3_, h3_w, h3_b = linear(h2, 128, 'g_h3_lin', init_type=init, with_w=True)
                    h3 = self.nl(h3_)

                    h4, h4_w, h4_b = linear(h3, self.flags.gmm_dim, 'g_h4_lin', init_type=init, with_w=True)
                    h4 = tf.reshape(h4, [-1, self.output_height, self.output_width, self.c_dim])
                    layers = [h0, h1, h2, h3, h4]

                    return h4

                elif self.flags.network == "GMM_SMALL":
                    h0_, h0_w, h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = self.nl(h0_)

                    h1_, h1_w, h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = self.nl(h1_)

                    h2, h2_w, h2_b = linear(h1, self.flags.gmm_dim, 'g_h2_lin', init_type=init, with_w=True)
                    h2 = tf.reshape(h2, [-1, self.output_height, self.output_width, self.c_dim])

                    return h2

                elif self.flags.network == "DCGAN":
                    print "in dcgan sampler"
                    s = self.output_height
                    s2, s4, s8, s16 = int(math.ceil(s / 2)), int(math.ceil(s / 4)), int(math.ceil(s / 8)), int(math.ceil(s / 16))

                    # project `z` and reshape
                    h0 = tf.reshape(linear(z, self.gf_dim * 8 * s16 * s16, 'g_h0_lin'),
                                    [-1, s16, s16, self.gf_dim * 8])
                    h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                    # TODO: Why is still using 'self.batch_size' instead of 'self.sample_size'?
                    h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim * 4], name='g_h1')
                    h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                    h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim * 2], name='g_h2')
                    h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                    h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim * 1], name='g_h3')
                    h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                    h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')

                    return tf.nn.tanh(h4)

            else:
                s = self.output_height
                s2, s4 = int(s / 2), int(s / 4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s4 * s4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(
                    self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def load_mnist(self):
        '''
        Loading MNIST dataset with normalization and shuffling (self.data_X, self.data_y)
        :return:
        '''

        # data_dir = './data/mnist'
        data_dir = os.path.join("./data", self.dataset_name)

        # Open file and return a stream
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        # Construct an array from data in a text or binary file
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        # TODO: Why is starting from the index 16 or 8
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000,)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000,)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        # Concatenate training data and test data
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)

        # Randomly shuffle the X and y.
        # Note that X and y should use the same seed, otherwise the labels will be wrong
        seed = 547
        # Seed the generator before using a randomState method
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        # From int (or np.unit8) to one-hot label, and normalize the mnist data [0,1]
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec

    def save(self, checkpoint_dir, step):
        '''
        Save all variables into checkpoint file w.r.t. dataset_name, batch_size and output_height
        :param checkpoint_dir:
        :param step:
        :return:
        '''

        # TODO: Why is using a model_name here?
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        '''
        Load all variables from checkpoint file
        :param checkpoint_dir:
        :return:
        '''

        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # TODO: Why not use 'saver.restore(self.sess, ckpt.model_checkpoint_path)' directly?
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
