import os
from PIL import Image
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np

import bunch

IMAGE_DIR = '/Users/bugra/Dropbox/Github/eigfacepy/example'
DEFAULT_SIZE = 10000  # 100x100
# http://www.clear.rice.edu/comp130/12spring/pca/pca_docs.shtml
# http://www.cs.princeton.edu/~cdecoro/eigenfaces/
# http://blog.nextgenetics.net/?e=42


class EigenFace(object):

    def __init__(self, image_path=IMAGE_DIR):
        self.image_dictionary = {}
        image_names = [image for image in os.listdir(image_path) if not image.startswith('.')]
        for image_name in image_names:
            image = np.asarray(Image.open(os.path.join(image_path, image_name)))
            dimensions = image.shape[0] * image.shape[1]
            downsample_factor = dimensions // DEFAULT_SIZE
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
                vector_2d -= vector_2d.mean()
            else:
                vector = image.flatten()
                vector -= vector.mean()
                vector_2d = np.concatenate((vector_2d.T, vector.T), axis=0)
        vector_2d = np.reshape(vector_2d, (len(self.image_dictionary), vector.size))
        return vector_2d

    def get_pca(self):
        u, s, weight_matrix = np.linalg.svd(self.vector_matrix)
        projected_matrix = np.dot(weight_matrix, self.vector_matrix.T).T
        standard_deviation = s**2/float(len(s))
        variance_proportion = standard_deviation / np.sum(standard_deviation)
        pca = bunch.Bunch()
        pca.u = u
        pca.s = s
        pca.weight_matrix = weight_matrix
        pca.projected_matrix = projected_matrix
        pca.standard_deviation = standard_deviation
        pca.variance_proportion = variance_proportion
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

    def plot_weight_matrix(self, number=11):
        v = self.pca.weight_matrix[:]
        num_row_x = num_row_y = int(np.floor(np.sqrt(number-1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii in range(number):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(np.reshape(v[ii], self.get_image_shape()), cmap=plt.cm.gray)
            axarr[div, rem].axis('off')
            if ii == number - 1:
                for jj in range(ii, num_row_x*num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')

    def get_average_weight_matrix(self, number=11):
        v = self.pca.weight_matrix[:]
        for ii in range(number):
            if ii == 0:
                temp = v[ii]
            else:
                temp += v[ii]
        temp /= float(number)
        return np.reshape(temp, self.get_image_shape())

    def get_image_shape(self):
        return self.image_dictionary[self.image_dictionary.keys()[0]].shape

    def plot_u_matrix(self):
        plt.imshow(self.pca.u, cmap=plt.cm.gray)

    def plot_standard_variation(self):
        plt.scatter(range(self.pca.standard_deviation.size), self.pca.standard_deviation)

    def plot_pca_components_proportions(self):
        plt.scatter(range(self.pca.variance_proportion.size), self.pca.variance_proportion)

    @staticmethod
    def imagesc(data_2d):
        plt.imshow(data_2d, extent=[0, 1, 0, 1], cmap=plt.cm.gray)
        plt.show()
