import numpy as np
import cPickle
import theano.tensor as T
from theano import function, config
import argparse
from pylearn2.datasets.preprocessing import ZCA
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

parser = argparse.ArgumentParser()
parser.add_argument('--folder')
parser.add_argument('-o')

args = parser.parse_args()
model_path = '/data/lisa/exp/wuzhen/conv2d/' + args.folder + '/convolutional_network_best.pkl'

import os
#X = np.load(os.environ['PYLEARN2_DATA_PATH'] + '/faceEmo/test_X.npy')
#y = np.load(os.environ['PYLEARN2_DATA_PATH'] + '/faceEmo/test_y.npy')
#print 'X.shape before', X.shape
X = np.load('test_input.npy')
X = X.astype('float32')
test_set = DenseDesignMatrix(X)

preproc = ZCA()
preproc.fit(test_set.X)
preproc.apply(test_set)
X = test_set.X

X = X.reshape(X.shape[0], 48, 48, 1).astype('float32')
#print 'X.shape after', X.shape
#X = np.swapaxes(X, 1, 2)
#print 'X.shape after swap', X.shape


f = open(model_path, 'rb')
mlp = cPickle.load(f)

#X_theano = mlp.get_input_space().make_batch_theano()
X_theano = T.tensor4()
y_theano = mlp.fprop(X_theano)

func = function(inputs=[X_theano], outputs=y_theano)

#outputs = np.array([])
batch_size = mlp.batch_size

inputs = X[:batch_size]
n_elements = X.shape[0]

n_batches = n_elements / batch_size

output = func(inputs)
outputs = output
print 'starting fprop'

for i in xrange(2, n_batches+1):
    inputs = X[(i-1)*batch_size : i*batch_size]
    output = func(inputs)
    output_hat = np.argmax(output, axis=1)
#    cls = np.equal(output_hat.astype('int32'),y[(i-1)*batch_size : i*batch_size].astype('int32'))
 #   yes = cls.astype('int32').sum()
  #  print yes
    outputs = np.concatenate((outputs, output), axis=0)
    print outputs.shape
    

#np.squeeze(outputs)
#print outputs
'''
print 'writing singular vector'
singular = []
index = 0
count = 0

#y_hat = np.argmax(outputs, axis=1)




#print y_hat.shape
#print y.shape


misclass = np.not_equal(y_hat.astype('int32'), y.astype('int32'))
print misclass
for e in misclass:
    if e == True:
        count += 1

print count



for ele in outputs:
    #import pdb
from theano import function
    #pdb.set_trace()
    
    i = np.argmax(ele)
    singular.append(i)
    print index+1, i, y[index] 
    if int(i) == int(y[index]):
        count += 1
        print 'yoho'
    index += 1

print 'accuracy', count

    #import pdb
    #pdb.set_trace()

#print len(singular)
#print singular[:100]

'''
singular = np.argmax(outputs, axis=1)
print 'writing to csv'
import csv
output_file = args.o
with open(output_file, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Prediction'])
    for i in xrange(1, len(singular)+1):
        writer.writerow([i, singular[i-1]])

print 'writing done', output_file

#print output

#output = func(X)
#print output.shape
#print output
