
import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets.preprocessing import ZCA
import os

class FaceEmo(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, which_set, center = False,
            one_hot = False, binarize = False,
            axes=['b', 0, 1, 'c'],
            preprocessor = ZCA(),
            fit_preprocessor = False,
            fit_test_preprocessor = False):

        self.args = locals()
        print "==========IIII LOVVEEEE YOOOOOUU========"
        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        if control.get_load_data():
            path = os.environ['PYLEARN2_DATA_PATH'] + '/faceEmo/'

            if which_set == 'train':
                X = np.load(path + 'train_X.npy').astype('float32')
                y = np.load(path + 'train_y.npy').astype('float32')
            else:
#                 import pdb
#                 pdb.set_trace()
                assert which_set == 'test'
                X = np.load(path + 'test_X.npy').astype('float32')
                y = np.load(path + 'test_y.npy').astype('float32')
            
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
            preprocessor.fit(self.X)
            preprocessor.apply(self, fit_preprocessor)
