from __future__ import print_function

"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import math
import json
import random
import pprint
import scipy.misc
import scipy.signal
import scipy.cluster.vq
import scipy.stats
import sklearn.manifold
import sklearn.mixture
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from time import gmtime, strftime
import tensorflow as tf

matplotlib.use('Qt4agg')
#ds = tf.contrib.distributions
#slim = tf.contrib.slim
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def switch_state_distribution(switch_state):
    switch_state = [int(state) for state in switch_state]
    return [i for i, state in enumerate(switch_state) if state]


def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def match_images(image, references):
    assert np.shape(image) == np.shape(references)[1:4]
    matchness = [np.sum(ref * image) for ref in references]
    am = np.argsort(-1 * np.array(matchness))
    return am


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def avg_noise(images):
    filtered_images = scipy.signal.medfilt(images, kernel_size=[1, 3, 3, 1])
    avg_noise = np.mean(np.sum(np.abs(filtered_images - images), axis=(1, 2, 3)))
    return avg_noise


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option, seq=None):
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
    elif option == 2:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in [random.randint(0, 99) for _ in xrange(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            # z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            try:
                make_gif(samples, './samples/test_gif_%s.gif' % (idx))
            except:
                save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    elif option == 3:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1. / config.batch_size)

        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
                         for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
    elif option == 5:
        assert not dcgan.data_X is None
        assert not seq is None
        z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
        if config.dataset == "mnist":
            y = np.random.choice(10, config.batch_size)
            y_one_hot = np.zeros((config.batch_size, 10))
            y_one_hot[np.arange(config.batch_size), y] = 1
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
        else:
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

        samples_normalized = samples / np.linalg.norm(samples.reshape([samples.shape[0], np.prod(samples.shape[1:4])]),
                                                      axis=1).reshape([samples.shape[0], 1, 1, 1])
        matched_ref_index = [match_images(sample, dcgan.data_X)[0:4] for sample in samples_normalized]
        matched = [dcgan.data_X[matched_index] for matched_index in matched_ref_index]
        all_images = reduce((lambda x, y: (np.concatenate((x, y), 0))),
                            (np.concatenate((samples[[i]], matched[i]), 0) for i in range(len(samples))),
                            np.array([]).reshape(0, samples.shape[1], samples.shape[2], samples.shape[3]))
        # save_images(samples, [8, 8], './samples/close_match_%s.png' % (str(seq)))
        save_images(all_images, [8, 8 * 5], './samples/close_match_%s.png' % (str(seq).zfill(5)))


def generate_gmm_data_random_orthant(num_data=50000, dim=2, num_cluster=None, scale=2, var=0.02):
    if num_cluster == None:
        num_cluster = np.power(2, dim)
    assert num_cluster <= np.power(2, dim)

    num_clusters = range(num_cluster)
    # random.shuffle(num_clusters)

    means = []
    for cluster in num_clusters:
        s = list(bin(cluster)[2:].zfill(dim))
        t = np.sign([int(c) - 0.5 for c in s])
        # print s
        # print t
        cluster_mean = scale * t
        means.append(cluster_mean)

    print("Cluster means: ", means)
    std = np.array([var] * num_cluster).transpose()
    weights = np.array([1. / num_cluster] * num_cluster).transpose()

    data = np.zeros([num_data, dim], dtype=np.float32)
    clusters = np.zeros([num_data, ], dtype=np.float32)
    for i in range(data.shape[0]):
        cluster = np.random.choice(range(num_cluster), p=weights)
        sample = np.random.multivariate_normal(mean=means[cluster].flatten(),
                                               cov=np.identity(dim) * std[cluster])
        data[i] = sample.transpose()
        clusters[i] = cluster
    return np.clip(data, -scale * 2, scale * 2), clusters


def generate_gmm_data(num_data=50000, dim=2, num_cluster=None, scale=2, var=0.02):
    # The number of clusters will be less than 2^(dim-1)
    if num_cluster == None:
        num_cluster = np.power(2, dim - 1)
    assert num_cluster <= np.power(2, dim - 1)

    num_clusters = range(num_cluster)
    # random.shuffle(num_clusters)

    means = []
    for cluster in num_clusters:
        # Return a list of strs - A regular binary representation of integer 'cluster',
        # for example, 0 -> ['0']
        s = list(bin(cluster)[2:].zfill(dim - 1))

        # Return an array of signs - Change 0's to -1's in s, for example, 0 -> t=[-1]
        t = np.sign([int(c) - 0.5 for c in s])
        # print s
        # print t

        # Return an array, for example, 0 -> cluster_mean=[2, -2]
        cluster_mean = np.hstack((np.array([scale]), scale * t))
        means.append(cluster_mean)

    # For example, dim=2: means=[[2, -2],[2, 2]]
    means = np.concatenate(means, axis=0)
    means = means.reshape([num_cluster, dim])

    print("Cluster means: ", means)
    # An arrary that stores variances with a length num_cluster
    std = np.array([var] * num_cluster).transpose()

    # weights denotes the prior probability that any point comes from each cluster
    weights = np.array([1. / num_cluster] * num_cluster).transpose()

    # Generate GMM data!
    data = np.zeros([num_data, dim], dtype=np.float32)
    clusters = np.zeros([num_data, ], dtype=np.float32)
    for i in range(data.shape[0]):
        cluster = np.random.choice(range(num_cluster), p=weights)
        sample = np.random.multivariate_normal(mean=means[cluster].flatten(),
                                               cov=np.identity(dim) * std[cluster])
        data[i] = sample.transpose()
        clusters[i] = cluster

    # clip the data into the range [-6, 6]
    return np.clip(data, -3 * scale, 3 * scale), clusters, means


def generate_gmm_grid_data(num_data=50000, dim=2, num_cluster=None, scale=2, var=0.02):
    if num_cluster == None:
        num_cluster = np.power(2, dim - 1)
    # assert num_cluster <= np.power(2, dim-1)

    num_clusters = range(num_cluster)
    # random.shuffle(num_clusters)

    means = []

    for cluster in num_clusters:

        cluster_mean = np.array([])
        for i in range(dim):
            cluster_mean = np.hstack((cluster_mean, np.array([1 + cluster % np.power(2, i + 1)])))
        means.append(cluster_mean * scale)

    means = np.concatenate(means, axis=0)
    means = means.reshape([num_cluster, dim])
    print("Cluster means: ", means)
    std = np.array([var] * num_cluster).transpose()
    weights = np.array([1. / num_cluster] * num_cluster).transpose()

    data = np.zeros([num_data, dim], dtype=np.float32)
    clusters = np.zeros([num_data, ], dtype=np.float32)
    for i in range(data.shape[0]):
        cluster = np.random.choice(range(num_cluster), p=weights)
        sample = np.random.multivariate_normal(mean=means[cluster].flatten(),
                                               cov=np.identity(dim) * std[cluster])
        data[i] = sample.transpose()
        clusters[i] = cluster
    return np.clip(data, -3 * scale, 3 * scale), clusters, means


def generate_gmm_dense_data(num_data=50000, dim=2, num_cluster=None, means=None, scale=2, var=0.02):
    np.random.seed(0) 
    if num_cluster == None and means == None:
        num_cluster = np.power(2, dim-1)
    elif not means == None:
        num_cluster = len(means)
    #assert num_cluster <= np.power(2, dim-1)
    
    num_clusters =range(num_cluster)
    #random.shuffle(num_clusters)

    if not means:
        means = []
    
        for cluster in num_clusters:
            cluster_mean = np.random.uniform(0, scale, size=(1, dim))        
            means.append(cluster_mean)

    means = np.concatenate(means, axis=0)
    means = means.reshape([num_cluster, dim])
    print("Cluster means: ", means)
    std = np.array([var] * num_cluster).transpose()
    weights = np.array([1. / num_cluster] * num_cluster).transpose()

    data = np.zeros([num_data, dim], dtype=np.float32)
    clusters = np.zeros([num_data, ], dtype=np.float32)
    for i in range(data.shape[0]):
        cluster = np.random.choice(range(num_cluster), p=weights)
        sample = np.random.multivariate_normal(mean=means[cluster].flatten(),
                                               cov=np.identity(dim) * std[cluster])
        data[i] = sample.transpose()
        clusters[i] = cluster
    return np.clip(data, -3 * scale, 3 * scale), clusters, means


def generate_gmm_circle_data(num_data=50000, dim=2, num_cluster=8, scale=2, var=0.02):
    if num_cluster == None:
        num_cluster = 10
    means_x = np.array([scale * np.cos(i * 2 * np.pi / num_cluster) for i in range(num_cluster)])
    means_y = np.array([scale * np.sin(i * 2 * np.pi / num_cluster) for i in range(num_cluster)])
    means = np.vstack((means_x, means_y)).transpose()
    # print means
    std = np.array([var] * num_cluster).transpose()
    weights = np.array([1. / num_cluster] * num_cluster).transpose()

    data = np.zeros([num_data, 2], dtype=np.float32)
    clusters = np.zeros([num_data, ], dtype=np.float32)
    for i in range(data.shape[0]):
        cluster = np.random.choice(range(num_cluster), p=weights)
        sample = np.random.multivariate_normal(mean=means[cluster].flatten(),
                                               cov=np.identity(2) * std[cluster])
        data[i] = sample.transpose()
        clusters[i] = cluster
    return np.clip(data, -3 * scale, 3 * scale), clusters, means


def estimate_optimal_cluster_size_gmm(data, clusters=range(1, 10), run=10):
    out = []
    predict = []
    gmms = []
    # calculate transformed distortion
    for c in clusters:
        gmm = sklearn.mixture.GaussianMixture(n_components=c, init_params='kmeans', n_init=run, covariance_type='full')
        # gmm = sklearn.mixture.VBGMM(n_components=c)
        gmm.fit(data)
        gmms.append(gmm)
        out.append(-gmm.bic(data))
        predict.append(gmm.predict(data))
        # find optimal cluster count
    # print out
    i = np.argmax(out)
    print(out)
    return clusters[i], predict[i], gmms[i]


def estimate_optimal_cluster_size_jump(data, clusters=range(1, 10), run=5):
    p = data.shape[1]
    y = int(p / 2)
    out = []
    for _ in range(run):
        d_t = [0]
        # calculate transformed distortion
        for c in clusters:
            centroid, label = scipy.cluster.vq.kmeans2(data, c, iter=5)
            t = np.array([data[i] - centroid[label[i]] for i in range(data.shape[0])])
            d = np.mean(np.array([np.inner(i, i) for i in t]) / p)
            d_t.append(np.power(d, -p))

        # find optimal cluster count
        j = [d_t[k] - d_t[k - 1] for k in range(1, len(d_t))]
        out.append(clusters[np.argmax(j)])
    return scipy.stats.mode(out)[0][0]


def sample_mog(batch_size, n_mixture=8, std=0.01, radius=1.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample_n(batch_size)

def stack_square_matrix(matrices):
    rows = []
    side_sqrt = int(np.sqrt(len(matrices)))
    for i in range(side_sqrt):
        rows.append(np.hstack(matrices[i*side_sqrt:(i+1)*side_sqrt]))
    return np.vstack(rows)

def visualize_matrix(path, matrices):
    if not os.path.exists(path):
        os.mkdir(path)

    square_matrix_ints =[]
    # each eigen vector in a specific cluster
    for matrix in matrices:
        side = int(np.sqrt(np.prod(matrix.shape)))
        matrix = matrix.reshape([side, side]) 
        # make plot dynamic range to int8
        value_range = np.max(matrix) - np.min(matrix)
        square_matrix_int = ((np.min(matrix) + matrix)
               *255/value_range).astype('uint8')
        square_matrix_ints.append(square_matrix_int)
    img = stack_square_matrix(square_matrix_ints)
    #img = PIL.Image.fromarray(img)
    scipy.misc.imsave(os.path.join(path, "matrix.png"), img) 


def visualize_cov(path, cov):
    if not os.path.exists(path):
        os.mkdir(path)

    for i, matrix in enumerate(cov):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        eigen_dict = [ (eigenvalues[i], eigenvectors[:,i]) for i in range(len(eigenvalues))]
        sorted_eigen_pair = sorted(eigen_dict, cmp=lambda x, y: cmp(y[0], x[0]))

        square_eigen_vector_ints =[]
        # each eigen vector in a specific cluster
        for (eigenvalue, eigenvector) in sorted_eigen_pair:
            side = int(np.sqrt(len(sorted_eigen_pair)))
            squared_eigen_vector = eigenvector.reshape([side, side])
            # make plot dynamic range to int8
            value_range = np.max(eigenvector) - np.min(eigenvector)
            square_eigen_vector_int = ((np.min(eigenvector) + squared_eigen_vector)
                   *255/value_range).astype('uint8')
            square_eigen_vector_ints.append(square_eigen_vector_int)
        img = stack_square_matrix(square_eigen_vector_ints)
        #img = PIL.Image.fromarray(img)
        scipy.misc.imsave(os.path.join(path, "cov_"+str(i)+".png"), img) 

def plot_2d(data, center=None, title=None, color=None, save_path=None, axis=[-2, 2, -2, 2], transform=False):
    """
    data is a N * M matrix, where N is number of data, M is number of features (2)
    """
    data_size = data.shape[0]
    if not (center == None):
        center_size = center.shape[0]
        data = np.vstack((data, center))

    if transform:
        isomap = sklearn.manifold.Isomap(n_jobs=-1)
        data = isomap.fit_transform(data)

    plt.close('all')
    plt.figure()
    if not (center == None):
        center = data[data_size:data_size + center_size, :]
        data = data[0:data_size, :]
    if not (color == None):
        # color = color.reshape([color.shape[0], 1])
        plt.scatter(data[:, 0], data[:, 1], c=color, marker="+")
    else:
        plt.scatter(data[:, 0], data[:, 1], marker="+")
    if not (color == None):
        plt.scatter(center[:, 0], center[:, 1], marker=(5, 0), c="k")
    if not (axis == None):
        plt.axis(axis)
    plt.grid(True)
    if not (title == None):
        plt.title(title)
    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path)
