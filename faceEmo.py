
import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.utils import serial
from pylearn2.utils.mnist_ubyte import read_mnist_images
from pylearn2.utils.mnist_ubyte import read_mnist_labels
import os

class FaceEmo(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, center = False,
            one_hot = False, binarize = False,
            axes=['b', 0, 1, 'c'],
            preprocessor = None,
            fit_preprocessor = False,
            fit_test_preprocessor = False):

        self.args = locals()

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        if control.get_load_data():
            path = os.environ['PYLEARN2_DATA_PATH'] + '/faceEmo/'

            if which_set == 'train':
                X = np.load(path + 'train_X.npy')
                y = np.load(path + 'train_y.npy')
            else:
#                 import pdb
#                 pdb.set_trace()
                assert which_set == 'test'
                X = np.load(path + 'test_X.npy')
                y = np.load(path + 'test_y.npy')
            
            if binarize:
                X = (X > 0.5).astype('float32')

            self.one_hot = one_hot
            if one_hot:
                one_hot = np.zeros((y.shape[0],3),dtype='float32')
                for i in xrange(y.shape[0]):
                    one_hot[i,y[i]] = 1.
                y = one_hot

            if center:
                X -= X.mean(axis=0)

            super(FaceEmo,self).__init__(X = X, y = y)

        if which_set == 'test':
            assert fit_test_preprocessor is None or (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)
