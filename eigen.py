import os
import numpy as np
from PIL import Image
import skimage.transform
import matplotlib.pyplot as plt

import bunch

IMAGE_DIR = '/Users/bugra/Dropbox/Github/eigfacepy/example'
DEFAULT_SIZE = 10000  # 100x100


class EigenFace(object):

    def __init__(self, image_path=IMAGE_DIR, default_size=DEFAULT_SIZE):
        self.image_dictionary = {}
        image_names = [image for image in os.listdir(image_path) if not image.startswith('.')]
        for image_name in image_names:
            image = np.asarray(Image.open(os.path.join(image_path, image_name)))
            dimensions = image.shape[0] * image.shape[1]
            downsample_factor = dimensions // default_size
            img = skimage.transform.pyramid_reduce(image, downscale=downsample_factor)
            if len(image.shape) > 2:
                self.image_dictionary[image_name] = skimage.color.rgb2gray(img)
            else:
                self.image_dictionary[image_name] = img
        self.vector_matrix = self.get_vector_representation()
        self.pca = self.get_pca()

    def get_vector_representation(self):
        for ii, (_, image) in enumerate(self.image_dictionary.iteritems()):
            if ii == 0:
                vector_2d = self.image_dictionary[self.image_dictionary.keys()[0]].flatten()
            else:
                vector = image.flatten()
                vector_2d = np.concatenate((vector_2d.T, vector.T), axis=0)
        vector_2d = np.reshape(vector_2d, (len(self.image_dictionary), vector.size))
        return vector_2d

    def get_pca(self):
        mean_vector = self.vector_matrix.mean(axis=0)
        for ii in range(self.vector_matrix.shape[0]):
            self.vector_matrix[ii] -= mean_vector
        u, s, eigen_vector = np.linalg.svd(np.dot(self.vector_matrix.T, self.vector_matrix))
        standard_deviation = s**2/float(len(s))
        variance_proportion = standard_deviation / np.sum(standard_deviation)
        pca = bunch.Bunch()
        pca.s = s
        pca.eigen_vector = eigen_vector[:self.vector_matrix.shape[0]]
        pca.variance_proportion = variance_proportion
        pca.mean_vector = mean_vector
        return pca

    def plot_image_dictionary(self):
        dictionary = self.image_dictionary
        num_row_x = num_row_y = int(np.floor(np.sqrt(len(dictionary)-1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii, (name, v) in enumerate(dictionary.iteritems()):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(v, cmap=plt.cm.gray)
            axarr[div, rem].set_title('{}'.format(name.split(".")[-1]).capitalize())
            axarr[div, rem].axis('off')
            if ii == len(dictionary) - 1:
                for jj in range(ii, num_row_x*num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')

    def plot_eigen_vector(self, n_eigen=0):
        plt.imshow(np.reshape(self.pca.eigen_vector[n_eigen], self.get_image_shape()), cmap=plt.cm.gray)

    def plot_eigen_vectors(self):
        number = self.pca.eigen_vector.shape[0]
        num_row_x = num_row_y = int(np.floor(np.sqrt(number-1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii in range(number):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(np.reshape(self.pca.eigen_vector[ii], self.get_image_shape()), cmap=plt.cm.gray)
            axarr[div, rem].axis('off')
            if ii == number - 1:
                for jj in range(ii, num_row_x*num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')

    def get_average_weight_matrix(self):
        return np.reshape(self.pca.mean_vector, self.get_image_shape())

    def plot_mean_vector(self):
        plt.imshow(self.get_average_weight_matrix(), cmap=plt.cm.gray)

    def get_image_shape(self):
        return self.image_dictionary[self.image_dictionary.keys()[0]].shape

    def plot_pca_components_proportions(self):
        plt.scatter(range(self.pca.variance_proportion.size), self.pca.variance_proportion)

    def get_eigen_value_distribution(self):
        return np.cumsum(self.pca.s) / np.sum(self.pca.s)

    def get_number_of_components_to_preserve_variance(self, variance=.95):
        for ii, eigen_value_cumsum in enumerate(self.get_eigen_value_distribution()):
            if eigen_value_cumsum > variance:
                return ii

    def plot_eigen_value_distribution(self, interval):
        plt.scatter(interval, self.get_eigen_value_distribution()[interval])
