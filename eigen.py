import skimage
import matplotlib.pyplot as plt
import numpy as np

HEINLEIN_PATH = 'heinlein.png'


class EigenFace(object):

    def __init__(self, image_path=HEINLEIN_PATH):
        self.image = skimage.io.imread(image_path)
        if len(self.image.shape) > 2:
            self.image = skimage.color.rgb2gray(self.image)
        self.eigen_face = get_eigen_face(self.image)

    def show_image(self):
        plt.imshow(self.image, cmap=plt.cm.gray)
        plt.show()

    def get_eigen_face(self):
        vector = self.image.flatten()
        vector -= flattened_array.mean(axis=0)
        covariance_matrix = np.dot(vector, vector.T)
        eigen_values, eigen_vectors = np.linalg.svd(covariance_matrix)

    def get_image_size(self):
        return self.image.shape


