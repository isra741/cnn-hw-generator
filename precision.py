caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import math as mt
import matplotlib.image as mpimg
import create_cnn_types as cct
import numpy as np
import change_prec as cp
import os

debug = 0

conv = 0
pool = 0
fc  = 0
conv_kernel = 0 
conv_stride = 0 
conv_feat = 0 
conv_kernel = 0
pool_stride = 0
pool_kernel = 0  
fc_neurons = 0 
feat_layers = 0
fc_layers = 0

max_conv = []
max_fc = []

min_conv = []
min_fc = []

def gen_model_params(file_name):
    global conv
    global pool
    global fc 
    global conv_kernel 
    global conv_stride 
    global conv_feat 
    global conv_kernel
    global pool_stride
    global pool_kernel  
    global fc_neurons 
    global feat_layers
    global fc_layers

    global max_conv
    global max_fc

    global min_conv
    global min_fc

    conv, pool, fc, conv_kernel, conv_stride, conv_feat, conv_kernel, \
    pool_stride, pool_kernel,  fc_neurons, feat_layers, fc_layers  = cct.cnn_model_params(file_name)
    max_conv = [[] for i in range(conv)]
    max_fc = [[] for i in range(fc)]

    min_conv = [[] for i in range(conv)]
    min_fc = [[] for i in range(fc)]

    return

caffe.set_device(0)
caffe.set_mode_gpu()

def prec_num(min_val, max_val):
    if max_val > abs(min_val):
        n = mt.ceil(mt.log(max_val, 2)) 
        if n - mt.log(max_val, 2) == 0:
            n += 1
            print(n)
    else:
        n = mt.ceil(mt.log(abs(min_val), 2))  
    return n
   
def classify(file_name, net):
    #max and min arrays
    global fc
    global conv
    global max_conv
    global max_fc
    
    global min_conv
    global min_fc
    global nof
    
    global debug

    im = mpimg.imread(file_name, 0)
    im = caffe.io.resize(im, [32, 32])
    im = np.array([[im]])
    net.blobs['data'].data[...] = (im*256)//1

    if debug == 1:
        net.forward(start='data', end='conv1')
        #Convolution outputs integer precision change
        for l in range(conv):
            la = l + 1
            cp.change_blob(net, 'conv' + str(la), nof)
            if l == conv - 1:
                net.forward(start='pool' + str(la), end='fc1')    
            else:
                net.forward(start='pool' + str(la), end='conv' + str(la + 1)) 
                      
        #Fully-connected outputs integer precision change	
        for l in range(fc):
            min_fc[l].append(np.amin(net.blobs[fc_layers[l]].data))
            max_fc[l].append(np.amax(net.blobs[fc_layers[l]].data))
            cp.change_blob(net, fc_layers[l], nof)
        
            if  l == fc - 1:
                net.forward(start='prob')
            else:
                net.forward(start=fc_layers[l + 1], end=fc_layers[l + 1])

        for l in range(conv):
            la = l + 1
            min_conv[l].append(np.amin(net.blobs['conv' + str(la)].data))
            max_conv[l].append(np.amax(net.blobs['conv' + str(la)].data))
    else:
        net.forward()
    return net.blobs['prob'].data.argmax()

def test_net(data_root, net):
	count = 0.0
	correct = 0
	class_id =[]
	file_name = []
	f = open(data_root + 'test.txt', 'r')
	
	for line in f:
		aux=0
		class_id = []
		file_name = []
		for l in line:
			if l != ' ' and aux == 0:
				file_name.append(l)
			elif l ==' ':
				aux = 1
			elif aux == 1:
				class_id.append(l)
		class_id  = ''.join(class_id)
		class_len = len(class_id)
		class_id = class_id[0:class_len - 1]
		file_name = ''.join(file_name)
		count = count + 1
		label = classify(data_root + file_name, net)
	        print(count, label)
		if str(label) == class_id: 
			correct = correct + 1
	f.close()

	return correct/count	


def precision_change(file_name, weights, arg_nof, ncw, nfcw):
    global debug
    global nof
    
    nof = arg_nof

    model_path = os.path.join('models', file_name + 'deploy.prototxt')
    weights_path = os.path.join('snaps', file_name)
    weights_path = os.path.join(weights_path, weights)
        
    net = caffe.Net(model_path, 0, weights=weights_path)
	
    data_root = 'prohibitory/clahe/GTSRB_test/'
    debug = 0
    best_acc = test_net(data_root, net)

    f = open('logs/' + file_name + 'precision.txt', 'w')
    f.write('Accuracy using float point precision: {0}%\n'.format(best_acc))

    cp.gen_model_params(file_name)
    cp.change_weight(net, ncw, nfcw)

    #Flag to change the layer output frational precision
    debug = 1

    reduced_acc = test_net(data_root, net)
    f.write('Accuracy using layer output and weight precision reduction: {0}%\n'.format(reduced_acc))
    f.write('Layer output fractional precision: {0}\n'.format(nof))
    f.write('Convolutional weight fractional precision: {0}\n'.format(ncw))
    f.write('Fully-connected weight fractional precision: {0}\n'.format(nfcw))

    if debug == 1:
        for l in range(conv):
            min_conv[l] = np.amin(min_conv[l])
            max_conv[l] = np.amax(max_conv[l])
        for l in range(fc):
            min_fc[l] = np.amin(min_fc[l])
            max_fc[l] = np.amax(max_fc[l])
        
	    '''    
        acc = 0
        while True:	
        	cp.change_weight(net, ncw, nfcw)
    
        	acc = test_net(data_root, net)
        	nfcw =- 1
        	ncw =- 1
        	print(acc, nfcw, ncw)
        	if acc < best_acc:
        		break
    	
        print(nfcw + 1, ncw + 1)
	    '''           
    
        #Conv outputs integer fixedpoint bit-width
        nc = []
        #Conv weights fractional fixedpoint bit-width
        ncw_l = []
    
        for l in range(conv):
            nc.append(int(prec_num(min_conv[l], max_conv[l])))
            ncw_l.append(ncw)
    
        #FC outputs integer fixedpoint bit-width
        nfc = []
        #FC weights fractional fixedpoint bit-width
        nfcw_l = []
    
        for l in range(fc):
            nfc.append(int(prec_num(min_fc[l], max_fc[l])))
            nfcw_l.append(nfcw)
    
        cct.write_cnn_types(net, nc, ncw_l, nfc, nfcw_l, file_name)
        
        for l in range(conv):
            f.write('Convolutional {0} output integer-part: {1}\n'.format(l + 1, nc[l]))
    
        for l in range(fc):
            f.write('Fully-connected {0} output integer-part: {1}\n'.format(l + 1, nfc[l]))
    f.close()
    return


