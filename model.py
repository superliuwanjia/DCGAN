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
                 batch_size=64, sample_size = 64, output_height=64,
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

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.process_input()
        self.build_model()

    def process_input(self):
        if not os.path.exists(self.flags.sample_dir):
            os.mkdir(self.flags.sample_dir)
        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
        elif self.dataset_name == 'GMM':
            data,label, mean = generate_gmm_data(dim=self.flags.gmm_dim, num_cluster=self.flags.gmm_cluster, var=self.flags.gmm_var, scale=self.flags.gmm_scale)
            self.data_X = data.reshape([data.shape[0], 1, self.flags.gmm_dim, 1])
            self.output_height = 1
            self.cluster_mean = mean
            self.output_width = self.flags.gmm_dim
            plot_2d(data[0:1000,:], save_path=self.flags.sample_dir+"/dataset.png", axis=None,transform=False)
        elif self.dataset_name == "GMM_CIRCLE":
            data,label, mean = generate_gmm_circle_data(num_cluster=self.flags.gmm_cluster, var=self.flags.gmm_var, scale=self.flags.gmm_scale)
            self.data_X = data.reshape([data.shape[0], 1, self.flags.gmm_dim, 1])
            self.output_height = 1
            self.cluster_mean = mean
            self.output_width = 2
            plot_2d(data[0:1000,:], save_path=self.flags.sample_dir+"/dataset.png", axis=None,transform=False)
        elif self.dataset_name == "GMM_DENSE":
            data,label, mean = generate_gmm_dense_data(dim=self.flags.gmm_dim, num_cluster=self.flags.gmm_cluster, var=self.flags.gmm_var, scale=self.flags.gmm_scale)
            self.data_X = data.reshape([data.shape[0], 1, self.flags.gmm_dim, 1])
            self.output_height = 1
            self.cluster_mean = mean
            self.output_width = 2
 
            plot_2d(data[0:1000,:], save_path=self.flags.sample_dir+"/dataset.png", transform=False, axis=[-2, self.flags.gmm_scale+2,-2,self.flags.gmm_scale+2])
        else:
            data = glob(os.path.join("./data", self.dataset_name, "*.jpg"))
            batch_files = data[0:len(data)-self.num_test_images]

            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_height, is_grayscale = self.is_grayscale) for batch_file in batch_files]
            if (self.is_grayscale):
                self.data_X = np.array(batch).astype(np.float32)[:, :, :, None]
            else:
                self.data_X = np.array(batch).astype(np.float32)
            # print(np.max(self.data_X))
            # print(np.min(self.data_X)) 
            # save part of the dataset as test images
            #batch_files_test = data[len(data) - self.num_test_images:]
            #batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_height, is_grayscale = self.is_grayscale) for batch_file in batch_files_test]
            #if (self.is_grayscale):
            #    self.data_X_test = np.array(batch).astype(np.float32)[:, :, :, None]
            #else:
            #    self.data_X_test = np.array(batch).astype(np.float32)
 
        # Intensity normalized training images
        self.data_X_normalized = self.data_X / np.linalg.norm(self.data_X.reshape([self.data_X.shape[0], np.prod(self.data_X.shape[1:4])]), axis=1).reshape([self.data_X.shape[0],1,1,1])
    
    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_height, self.output_width, self.c_dim],
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_height, self.output_width, self.c_dim],
                                        name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.z_sum = histogram_summary("z", self.z)

        # noise of samples generated, approximated as I - medianfilt(I)   
        self.avg_noise = tf.placeholder(tf.float32, None, name='avg_noise')
        
        self.avg_noise_sum = scalar_summary("avg_noise_sum", self.avg_noise)

        # noise of training images used, base line
        self.train_avg_noise = tf.placeholder(
                tf.float32, None, name='train_avg_noise')
        
        self.train_avg_noise_sum = scalar_summary(
                "train_avg_noise_sum", self.train_avg_noise)

        self.est_clusters = tf.placeholder(
                tf.float32, None, name='estimated_cluster_count')
        self.est_clusters_sum = scalar_summary(
                "estimated_clusters_count_summary", self.est_clusters)

        # average image of all images used during training 
        self.cum_image = np.zeros([1, self.output_height, self.output_width, self.c_dim]) 
        self.num_images_used = 0
        self.mean_image = tf.placeholder(tf.float32, [1, self.output_height, self.output_width, self.c_dim], name='mean_image')

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits  = self.discriminator(self.images, self.y, reuse=False)

            self.sampler = self.sampler(self.z, self.y)
            self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(self.images)

            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        
        self.real_accu = tf_accuracy(self.D, 1, self.batch_size)        
        self.fake_accu = tf_accuracy(self.D_, 0, self.batch_size)        
        
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        self.real_accu_sum = scalar_summary("real_accu_sum", self.real_accu)
        self.fake_accu_sum = scalar_summary("fake_accu_sum", self.fake_accu)
        
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
 
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                                                    
        self.g_loss_heruistic = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.g_loss = -self.d_loss
        self.g_loss_llr = -tf.reduce_mean(tf.log(self.D_/(1 - self.D_)))

      
 
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        if config.dataset == 'mnist':
             data_X, data_y = self.load_mnist()
        else:
             data = glob(os.path.join("./data", config.dataset, "*.jpg"))
        #np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.learning_rate_d, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        self.d_grads = tf.gradients(self.d_loss, self.d_vars)[0]
        self.d_grads_sum = tf.histogram_summary("d_grad_sum", self.d_grads)
        if self.flags.g_heruistic == 0:
            g_optim = tf.train.AdamOptimizer(config.learning_rate_g, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
            self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
            self.g_grads = tf.gradients(self.g_loss, self.g_vars)[0]
        elif self.flags.g_heruistic == 1:
            g_optim_heruistic = tf.train.AdamOptimizer(config.learning_rate_g, beta1=config.beta1) \
                          .minimize(self.g_loss_heruistic, var_list=self.g_vars)
            self.g_grads = tf.gradients(self.g_loss_heruistic, self.g_vars)[0]
            self.g_loss_sum = scalar_summary("g_loss_heruistic", self.g_loss_heruistic)
        else:
            g_optim_llr = tf.train.AdamOptimizer(config.learning_rate_g, beta1=config.beta1) \
                          .minimize(self.g_loss_llr, var_list=self.g_vars)
            self.g_grads = tf.gradients(self.g_loss_llr, self.g_vars)[0]
            self.g_loss_sum = scalar_summary("g_loss_llr", self.g_loss_llr)
        self.g_grads_sum = tf.histogram_summary("g_grad_sum", self.g_grads)
        #d_optim = self.d_optim
        #g_optim = self.g_optim
        try:
            tf.initialize_all_variables().run()
        except:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum, self.g_sums, self.g_grads_sum])
        self.d_sum = merge_summary([self.real_accu_sum, self.fake_accu_sum, self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum, self.d_real_sums, self.d_fake_sums, self.d_grads_sum])
        self.writer = SummaryWriter("./"+self.flags.log_dir, self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        
        if config.dataset == 'mnist':
            sample_images = data_X[0:self.sample_size]
            sample_labels = data_y[0:self.sample_size]
        else:
            # sample_files = data[0:self.sample_size]
            # sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_height, is_grayscale = self.is_grayscale) for sample_file in sample_files]
            #if (self.is_grayscale):
            #    sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            #else:
            #    sample_images = np.array(sample).astype(np.float32)
            sample_images = self.data_X[0:self.sample_size]
        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            #if config.dataset == 'mnist':
            #    batch_idxs = min(len(data_X), config.train_size) // config.batch_size
            #else:            
            #    data = glob(os.path.join("./data", config.dataset, "*.jpg"))
            #    batch_idxs = min(len(data_X), config.train_size) // config.batch_size

            batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                else:
                    #batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                    #batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_height, is_grayscale = self.is_grayscale) for batch_file in batch_files]
                    #if (self.is_grayscale):
                    #    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    #else:
                    #    batch_images = np.array(batch).astype(np.float32)

                    batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                # save mean of all images used for training
                self.cum_image = self.cum_image + np.mean(batch_images, axis=0, keepdims=True)
                self.num_images_used += 1
        
 
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)
                batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)
                

                if self.y_dim:
                    # Update D network
                    
                    # print extra info before training
                    if counter == 0:
                        summary_str_d, summary_str_g = self.sess.run([self.d_sum, self.g_sum],
                            feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels })
                        self.writer.add_summary(summary_str_d, -1)
                        self.writer.add_summary(summary_str_g, -1)

                   
                    summary_str, _ = self.sess.run([self.d_sum, d_optim],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    if config.g_heruistic == 1:
                        for _ in range(self.flags.g_update):
                            # Update G network
                            _, summary_str = self.sess.run([g_optim_heruistic, self.g_sum],
                                feed_dict={ self.z: batch_z, self.y:batch_labels })
                            self.writer.add_summary(summary_str, counter)
 
                    elif config.g_heruistic == 0:
                        for _ in range(self.flags.g_update):
                            # Update G network
                            _, summary_str = self.sess.run([g_optim, self.g_sum],
                                feed_dict={ self.z: batch_z, self.y:batch_labels, self.images:batch_images })
                            self.writer.add_summary(summary_str, counter)
                    else:
                        for _ in range(self.flags.g_update):
                            # Update G network
                            _, summary_str = self.sess.run([g_optim_llr, self.g_sum],
                                feed_dict={ self.z: batch_z, self.y:batch_labels })
                            self.writer.add_summary(summary_str, counter)
         

                else:

                    if counter == 0:
                        summary_str_d, summary_str_g = self.sess.run([self.d_sum, self.g_sum],
                            feed_dict={ self.images: batch_images, self.z: batch_z})
                        self.writer.add_summary(summary_str_d, -1)
                        self.writer.add_summary(summary_str_g, -1)


                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    if config.g_heruistic == 1:
                        for _ in range(self.flags.g_update):
                            # Update G network
                            _, summary_str = self.sess.run([g_optim_heruistic, self.g_sum],
                                feed_dict={ self.z: batch_z })
                            self.writer.add_summary(summary_str, counter)

 
                    elif config.g_heruistic == 0:
                        for _ in range(self.flags.g_update):
                            # Update G network
                            _, summary_str = self.sess.run([g_optim, self.g_sum],
                                feed_dict={ self.z: batch_z, self.images: batch_images })
                            self.writer.add_summary(summary_str, counter)

                    else:
                        for _ in range(self.flags.g_update):
                             # Update G network
                            _, summary_str = self.sess.run([g_optim_llr, self.g_sum],
                                feed_dict={ self.z: batch_z })
                            self.writer.add_summary(summary_str, counter)

               
                # print loss etc
                if self.y_dim:
                    d_fake, samples, d_loss, g_loss = self.sess.run(
                        [self.D_, self.sampler, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.y:sample_labels}
                    )
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels})
                    errD_real = self.d_loss_real.eval({self.images: batch_images, self.y:batch_labels})
                    errG = self.g_loss.eval({self.z: batch_z, self.y:batch_labels, self.images: batch_images})
                    errG_heruistic = self.g_loss_heruistic.eval({self.z: batch_z, self.y:batch_labels})
                    errG_llr = self.g_loss_llr.eval({self.z: batch_z, self.y:batch_labels})
 
                else:
                    
                    d_fake, samples, d_loss, g_loss, g_loss_heruistic, g_loss_llr = self.sess.run(
                            [self.D_, self.sampler, self.d_loss, self.g_loss, self.g_loss_heruistic, self.g_loss_llr],
                            feed_dict={self.z: sample_z, self.images: sample_images}
                    )
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.images: batch_images})
                    errG = self.g_loss.eval({self.z: batch_z, self.images: batch_images})
                    errG_heruistic = self.g_loss_heruistic.eval({self.z: batch_z})
                    errG_llr = self.g_loss_llr.eval({self.z: batch_z})
                
                if config.g_heruistic == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))
                elif config.g_heruistic == 1:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss_heu: %.8f" \
                        % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG_heruistic))
                else:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss_llr: %.8f" \
                        % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG_llr))
                



                summary_str = self.sess.run(self.avg_noise_sum,
                    feed_dict={ self.avg_noise: avg_noise(samples) })
                self.writer.add_summary(summary_str, counter)
 
                # noise for training samples used
                summary_str = self.sess.run(self.train_avg_noise_sum,
                    feed_dict={ self.train_avg_noise: avg_noise(batch_images) })
                self.writer.add_summary(summary_str, counter)
         
                if np.mod(counter, config.visualize_interval) == 0:
                    if config.dataset == 'mnist':
                        save_images(samples, [8, 8],
                                    './{}/train_{:06d}.png'.format(config.sample_dir, counter))
                    elif config.dataset[0:3] == "GMM":
                        samples = samples.reshape([self.batch_size, self.flags.gmm_dim])

                        if self.flags.cluster_est == "gmm":
                            estimated_clusters, label, gmm = estimate_optimal_cluster_size_gmm(samples, clusters=range(1,self.flags.gmm_cluster+1))
                            # samples, label = gmm.sample(self.batch_size)
                        elif self.flags.cluster_est == "jump":
                            estimated_clusters, label = estimate_optimal_cluster_size_jump(samples, clusters=range(1,self.flags.gmm_cluster+1))

                        summary_str = self.sess.run(self.est_clusters_sum,
                            feed_dict={self.est_clusters:estimated_clusters})
                        self.writer.add_summary(summary_str, counter)

                        if not np.max(label) == 0:
                            colors = label / (np.max(label) + 0.0)
                        else:
                            colors = label
                        # self.cluster_mean=None
                        if self.flags.gmm_dim > 2:

                            axis=[-2, self.flags.gmm_scale+2,-2,self.flags.gmm_scale+2]
                            plot_2d(samples, center=self.cluster_mean, title="Minibatch: " + str(counter), axis=None, color=colors,save_path='./{}/train_{:06d}.png'.format(config.sample_dir, counter), transform=True)  
                        else:    
                            if self.flags.dataset == "GMM_DENSE":
                                plot_2d(samples, center=self.cluster_mean, title="Minibatch: " + str(counter), axis=[-1,self.flags.gmm_scale+2,-1,self.flags.gmm_scale+2], color=colors,save_path='./{}/train_{:06d}.png'.format(config.sample_dir, counter), transform=False)  
                            else:
                                plot_2d(samples, center=self.cluster_mean, title="Minibatch: " + str(counter), axis=[-4,4,-4,4], color=colors,save_path='./{}/train_{:06d}.png'.format(config.sample_dir, counter), transform=False)  
                    else:
                        save_images(samples, [8, 8],
                                        './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    #if config.g_heruistic == 0:
                    #    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    #elif config.g_heruistic == 1:
                    #    print("[Sample] d_loss: %.8f, g_loss_heruistic: %.8f" % (d_loss, g_loss_heruistic)) 
                    #else:
                    #    print("[Sample] d_loss: %.8f, g_loss_llr: %.8f" % (d_loss, g_loss_llr)) 

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
                if np.mod(counter, 500) == 0:
                    self.save(config.checkpoint_dir, counter)

 

                counter += 1

    def discriminator(self, image, y=None, reuse=False):
        init = self.flags.init_type
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                if self.flags.network == "GMM_LARGE":
                    image = tf.reshape(image, [-1, self.output_width * self.output_height * self.c_dim])
                    h0_, h0_w, h0_b = linear(image,128,'d_h0_lin',init_type=init, with_w=True)
                    h0 = tf.nn.tanh(h0_)
                    h1_, h1_w, h1_b = linear(h0,128,'d_h1_lin', init_type=init, with_w=True)
                    h1 = tf.nn.tanh(h1_)
 
                    h2_, h2_w, h2_b = linear(h1,128,'d_h2_lin', init_type=init, with_w=True)
                    h2 = tf.nn.tanh(h2_)
                    h3_, h3_w, h3_b = linear(h2,128,'d_h3_lin', init_type=init, with_w=True)
                    h3 = tf.nn.tanh(h3_)
 
                    h4, h4_w, h4_b = linear(h3,1,'d_h4_lin', init_type=init, with_w=True)
                    layers = [h0,h1, h2, h3, h4]
                
                elif self.flags.network == "GMM_MEDIUM":
                    image = tf.reshape(image, [-1, self.output_width * self.output_height * self.c_dim])
                    h0_, h0_w, h0_b = linear(image,128,'d_h0_lin', init_type=init, with_w=True)
                    h0 = tf.nn.tanh(h0_)
                    h1_, h1_w, h1_b = linear(h0,128,'d_h1_lin', init_type=init, with_w=True)
                    h1 = tf.nn.tanh(h1_)
 
                    h2, h2_w, h2_b = linear(h1,1,'d_h2_lin', init_type=init, with_w=True)
                    layers = [h0, h1, h2]
                elif self.flags.network == "GMM_MEDIUM_RELU":
                    image = tf.reshape(image, [-1, self.output_width * self.output_height * self.c_dim])
                    h0_, h0_w, h0_b = linear(image,1,'d_h0_lin', init_type=init, with_w=True)
                    h0 = lrelu(h0_)
                    layers = [h0]
 
                elif self.flags.network == "DCGAN": 
                    h0 = lrelu(h0_)
            
                    h1_ = conv2d(h0, self.df_dim*2, name='d_h1_conv')
                    h1 = lrelu(self.d_bn1(h1_))

                    h2_ = conv2d(h1, self.df_dim*4, name='d_h2_conv')
                    h2 = lrelu(self.d_bn2(h2_))
                    
                    h3_ = conv2d(h2, self.df_dim*8, name='d_h3_conv')
                    h3 = lrelu(self.d_bn3(h3_))

                    h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                    layers = [h0_, h1_, h2_, h3_, h4] 
 
                if not reuse:
                    self.d_real_sums=[]
                    for layer in layers:
                        self.d_real_sums.append(tf.histogram_summary("d_real_sums_"+layer.name, layer))
                else:
                    self.d_fake_sums=[]
                    for layer in layers:
                        self.d_fake_sums.append(tf.histogram_summary("d_fake_sums_"+layer.name, layer))
                        

                return tf.nn.sigmoid(layers[-1]), layers[-1]
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])            
                h1 = tf.concat(1, [h1, y])
                
                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = tf.concat(1, [h2, y])
 
                h3 = linear(h2, 1, 'd_h3_lin')
                
                layers = [h0,h1,h2, h3] 
 
                if not reuse:
                    self.d_real_sums=[]
                    for layer in layers:
                        self.d_real_sums.append(tf.histogram_summary("d_real_sums_"+layer.name, layer))
                else:
                    self.d_fake_sums=[]
                    for layer in layers:
                        self.d_fake_sums.append(tf.histogram_summary("d_fake_sums_"+layer.name, layer))
 
                return tf.nn.sigmoid(layers[-1]), layers[-1]

    def generator(self, z, y=None):
        init = self.flags.init_type
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                if self.flags.network == "GMM_LARGE":
                    h0_, h0_w,h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = tf.nn.tanh(h0_)
         
                    h1_, h1_w,h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = tf.nn.tanh(h1_)
                    
                    h2_, h2_w,h2_b = linear(h1, 128, 'g_h2_lin', init_type=init, with_w=True)
                    h2 = tf.nn.tanh(h2_)
                    h3_, h3_w,h3_b = linear(h2, 128, 'g_h3_lin', init_type=init, with_w=True)
                    h3 = tf.nn.tanh(h3_)
 
                    h4, h4_w, h4_b = linear(h3,self.flags.gmm_dim, 'g_h4_lin', init_type=init, with_w=True) 
                    h4 = tf.reshape(h4, [-1, self.output_height, self.output_width, self.c_dim])
                    layers = [h0, h1, h2, h3, h4]
               
                    self.g_sums=[]
                    for layer in layers:
                        self.g_sums.append(tf.histogram_summary("g_sum_"+layer.name, layer))

                    return h4
                elif self.flags.network == "GMM_MEDIUM":
                    h0_, h0_w,h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = tf.nn.tanh(h0_)
         
                    h1_, h1_w,h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = tf.nn.tanh(h1_)
                    
 
                    h2, h2_w, h2_b = linear(h1,self.flags.gmm_dim, 'g_h2_lin', init_type=init, with_w=True) 
                    h2 = tf.reshape(h2, [-1, self.output_height, self.output_width, self.c_dim])
                    layers = [h0, h1, h2]
               
                    self.g_sums=[]
                    for layer in layers:
                        self.g_sums.append(tf.histogram_summary("g_sum_"+layer.name, layer))


                    return h2
                elif self.flags.network == "GMM_MEDIUM_RELU":
                    h0_, h0_w,h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = lrelu(h0_, leak=0)
         
                    h1_, h1_w,h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = lrelu(h1_,leak=0)
                    
 
                    h2, h2_w, h2_b = linear(h1,self.flags.gmm_dim, 'g_h2_lin', init_type=init, with_w=True) 
                    h2 = tf.reshape(h2, [-1, self.output_height, self.output_width, self.c_dim])
                    layers = [h0, h1, h2]
               
                    self.g_sums=[]
                    for layer in layers:
                        self.g_sums.append(tf.histogram_summary("g_sum_"+layer.name, layer))


                    return h2
 
                elif self.flags.network == "DCGAN": 
                    s = self.output_height
                    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

                    # project `z` and reshape
                    self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

                    self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
                    h0 = tf.nn.relu(self.g_bn0(self.h0))

                    self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                        [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True)
                    h1 = tf.nn.relu(self.g_bn1(self.h1))

                    h2, self.h2_w, self.h2_b = deconv2d(h1,
                        [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True)
                    h2 = tf.nn.relu(self.g_bn2(h2))

                    h3, self.h3_w, self.h3_b = deconv2d(h2,
                        [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True)
                    h3 = tf.nn.relu(self.g_bn3(h3))

                    h4, self.h4_w, self.h4_b = deconv2d(h3,
                        [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)
                    
                    layers = [h0, h1, h2, h3, h4]
               
                    self.g_sums=[]
                    for layer in layers:
                        self.g_sums.append(tf.histogram_summary("g_sum_"+layer.name, layer))

                    return tf.nn.tanh(h4)
            else:
                s = self.output_height
                s2, s4 = int(s/2), int(s/4) 

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        init = self.flags.init_type
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                if self.flags.network == "GMM_LARGE":

                    h0_, h0_w,h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = tf.nn.tanh(h0_)
         
                    h1_, h1_w,h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = tf.nn.tanh(h1_)
                    
                    h2_, h2_w,h2_b = linear(h1, 128, 'g_h2_lin', init_type=init, with_w=True)
                    h2 = tf.nn.tanh(h2_)
                    h3_, h3_w,h3_b = linear(h2, 128, 'g_h3_lin', init_type=init, with_w=True)
                    h3 = tf.nn.tanh(h3_)
 
                    h4, h4_w, h4_b = linear(h3,self.flags.gmm_dim, 'g_h4_lin', init_type=init, with_w=True) 
                    h4 = tf.reshape(h4, [-1, self.output_height, self.output_width, self.c_dim])
 
                    return h4
                elif self.flags.network == "GMM_MEDIUM":
                    h0_, h0_w,h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = tf.nn.tanh(h0_)
         
                    h1_, h1_w,h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = tf.nn.tanh(h1_)
                    
 
                    h2, h2_w, h2_b = linear(h1,self.flags.gmm_dim, 'g_h2_lin', init_type=init, with_w=True) 
                    h2 = tf.reshape(h2, [-1, self.output_height, self.output_width, self.c_dim])
 
                    return h2
                elif self.flags.network == "GMM_MEDIUM_RELU":
                    h0_, h0_w,h0_b = linear(z, 128, 'g_h0_lin', init_type=init, with_w=True)
                    h0 = lrelu(h0_)
         
                    h1_, h1_w,h1_b = linear(h0, 128, 'g_h1_lin', init_type=init, with_w=True)
                    h1 = lrelu(h1_)
                    
 
                    h2, h2_w, h2_b = linear(h1,self.flags.gmm_dim, 'g_h2_lin', init_type=init, with_w=True) 
                    h2 = tf.reshape(h2, [-1, self.output_height, self.output_width, self.c_dim])
 
                    return h2
 
                elif self.flags.network == "DCGAN": 
                
                    s = self.output_height
                    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

                    # project `z` and reshape
                    h0 = tf.reshape(linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin'),
                                    [-1, s16, s16, self.gf_dim * 8])
                    h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                    h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
                    h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                    h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
                    h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                    h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
                    h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                    h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s = self.output_height
                s2, s4 = int(s/2), int(s/4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)
        
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0
        
        return X/255.,y_vec
            
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
