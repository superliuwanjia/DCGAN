import numpy as np
import sklearn.mixture
import sklearn.decomposition
import utils
import PIL.Image
import os

class MiscModels(object):
    def __init__(self):
        matrix_to_visualize=[]

    def _train(self, data):
        return None

    def train(self, data):
        if type(data) is str:
            data = np.load(data)["X_train"]
        return self._train(data)


class PCA(MiscModels):
    def __init__(self, num_components=10): 
        super(PCA,self).__init__()
        self.num_components = num_components 

    def _train(self, data):
        pca = sklearn.decomposition.PCA(n_components=self.num_components)
        pca.fit(data)
        return [i for i in pca.components_]

class SC(MiscModels):
    def __init__(self, sparsity=0.1, num_components=10): 
        super(SC,self).__init__()
        self.sparsity = sparsity
        self.num_components = num_components

    def _train(self, data):
        dictionary_learning = sklearn.decomposition.DictionaryLearning( \
            n_components=self.num_components, transform_n_nonzero_coefs=self.sparsity,
            alpha=self.sparsity)
        dictionary_learning.fit(data)
        return [i for i in dictionary_learning.components_]
 
class ICA(MiscModels):
    def __init__(self, num_components=10): 
        super(ICA,self).__init__()
        self.num_components = num_components 

    def _train(self, data):
        ica = sklearn.decomposition.FastICA(n_components=self.num_components)
        ica.fit(data)
        return [i for i in ica.components_]
        

class GMM(MiscModels):
    def __init__(self, num_components=10): 
        super(GMM,self).__init__()
        self.num_components = num_components 

    def _train(self, data):
        gmm = sklearn.mixture.GaussianMixture(n_components=self.num_components, covariance_type="full")
        gmm.fit(data)
        return [i for i in gmm.covariances_]
 
    
 
