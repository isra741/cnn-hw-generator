#Module to change the parameter precisions
import numpy as np
import math as mt
import create_cnn_types as cct

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
   
    conv, pool, fc, conv_kernel, conv_stride, conv_feat, conv_kernel, \
    pool_stride, pool_kernel,  fc_neurons, feat_layers, fc_layers  = cct.cnn_model_params(file_name)

    return

#Function to change the precision of a float signal (num) defined by fixed-point fractional-part (n)
def pfix(num, n):
	if n != 0:
    		num = int(num * 2**n)
    		num = float(num)/(2**n)
	else:
		num = int(num)
    	return num


def getPrec(array):
    max_pos = np.amax(array)
    max_neg = np.amax(array*-1)

    max_out = np.amax([max_pos, max_neg])
    n = mt.ceil(mt.log(max_out, 2))

    if max_neg == max_out and max_neg % 2 == 0:
        n = mt.log2(max_out)
    elif max_pos == max_out and max_pos %2 == 0:
        n = mt.log2(max_out) + 1
    return int(n)


def change_blob(net, blob, nb):
    dim = np.zeros(conv)

    batch = 1
    
    if 'conv' in blob or 'pool' in blob:
        dim = len(net.blobs[blob].data[0][0])
    
    #conv

    if 'conv' in blob or 'pool' in blob:
        l = int(blob[4]) - 1
        for b in range(batch):
            for n in range(int(conv_feat[l])):
                for x in range(dim):
                    for y in range(dim):
                        num = net.blobs[blob].data[b][n][x][y]
                        net.blobs[blob].data[b][n][x][y] = pfix(num, nb)

    
    if 'fc' in blob or 'score' in blob:
    	#print("oi")
    #fc
        if 'score' in blob:
            l = fc - 1
        else:
            l = int(blob[2]) - 1
            
        for b in range(batch):
            for n in range(int(fc_neurons[l])):
                num = net.blobs[blob].data[b][n]
                if num < 0:
		    if 'score' in blob:
         		net.blobs[blob].data[b][n] = pfix(num, nb)
		    else:
			net.blobs[blob].data[b][n] = 0
                else:
                    net.blobs[blob].data[b][n] = pfix(num, nb)
    return
 
               

def change_weight(net, nc, nfc):

    prev = []
    for l in range(conv):
        if l == 0:
            prev.append(1)
        else:
            prev.append(conv_feat[l - 1])

    n_input = []
    for l in range(fc):
        if l == 0:
            n_input.append(conv_feat[conv - 1]*((net.blobs['pool' + str(pool)].data[0][0][0].size)**2))
        else:
            n_input.append(fc_neurons[l  - 1])

    #Convolutions
    for c in range(conv):
        #bias
        for n in range(int(conv_feat[c])):
            net.params['conv' + str(c + 1)][1].data[n] = pfix(net.params['conv' + str(c + 1)][1].data[n], nc*2)
        #weights
        for n in range(int(conv_feat[c])):
            for f in range(int(prev[c])):
                for x in range(int(conv_kernel[c])):
                    for y in range(int(conv_kernel[c])):
                        net.params['conv' + str(c + 1)][0].data[n][f][x][y] = pfix(net.params['conv' + str(c + 1)][0].data[n][f][x][y], nc)

    #Fully connected
    for f in range(fc):
        #bias
        for j in range(int(fc_neurons[f])):
            if f == fc - 1:
                net.params['score'][1].data[j] = pfix(net.params['score'][1].data[j], nfc*2)
            else:
                net.params['fc'+ str(f + 1)][1].data[j] = pfix(net.params['fc'+ str(f + 1)][1].data[j], nfc*2)
                
        #weights
        for j in range(int(fc_neurons[f])):
            for i in range(int(n_input[f])):
                if f == fc - 1:
                    net.params['score'][0].data[j][i] = pfix(net.params['score'][0].data[j][i], nfc)
                else:
                    net.params['fc'+ str(f + 1)][0].data[j][i] = pfix(net.params['fc'+ str(f + 1)][0].data[j][i], nfc)
    return 
