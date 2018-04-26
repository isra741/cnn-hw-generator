caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import os

from caffe import layers as L, params as P


def lenet(lmdb, batch_size, phase, conv_feat, conv_kernel, pool_kernel, pool_stride, fc_neurons):
	n = caffe.NetSpec()
	if phase == 0 or phase == 1: 
		n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb, ntop=2)
	else:
		n.data = L.Input(shape=[dict(dim=[1, 1, 32, 32])])       
	n.conv1 = L.Convolution(n.data, kernel_size=conv_kernel[0], num_output=conv_feat[0], \
weight_filler=dict(type='xavier'))
	n.pool1 = L.Pooling(n.conv1, kernel_size=pool_kernel[0], stride=pool_stride[0], pool=P.Pooling.MAX)
	n.conv2 = L.Convolution(n.pool1, kernel_size=conv_kernel[1], num_output=conv_feat[1], \
weight_filler=dict(type='xavier'))
	n.pool2 = L.Pooling(n.conv2, kernel_size=pool_kernel[1], stride=pool_stride[1], pool=P.Pooling.MAX)
	n.fc1 =   L.InnerProduct(n.pool2, num_output=fc_neurons[0], weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.fc1, in_place=True)
	n.fc2 = L.InnerProduct(n.relu1, num_output=fc_neurons[1], weight_filler=dict(type='xavier'))
	n.relu2 = L.ReLU(n.fc2, in_place=True)
	n.score = L.InnerProduct(n.relu2, num_output=fc_neurons[2], weight_filler=dict(type='xavier'))
	if phase == 0 or phase == 1:     
		n.loss =  L.SoftmaxWithLoss(n.score, n.label)
	else:
		n.prob =  L.Softmax(n.score)        
	if phase == 1:
		n.accuracy =  L.Accuracy(n.score, n.label)
    
	return n.to_proto()

def create_proto(file_name, conv_feat, conv_kernel, pool_kernel, pool_stride, fc_neurons):
    with open('models/' + file_name + 'train.prototxt', 'w+') as f:
        f.write(str(lenet('data/prohi_train_lmdb', 64, 0, conv_feat, conv_kernel, pool_kernel, pool_stride, fc_neurons)))
    os.chmod('models/' + file_name + 'train.prototxt', 0o777)
    
    with open('models/' + file_name + 'test.prototxt', 'w+') as f:
        f.write(str(lenet('data/prohi_test_lmdb', 100, 1, conv_feat, conv_kernel, pool_kernel, pool_stride, fc_neurons)))
    os.chmod('models/' + file_name + 'test.prototxt', 0o777)

    with open('models/' + file_name + 'deploy.prototxt', 'w+') as f:
        f.write(str(lenet(' ', 0, 2, conv_feat, conv_kernel, pool_kernel, pool_stride, fc_neurons)))
    os.chmod('models/' + file_name + 'deploy.prototxt', 0o777)
    return






