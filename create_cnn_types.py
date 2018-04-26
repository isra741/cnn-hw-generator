caffe_root = '../'   
import sys
import numpy as np
import math as mt
import os
sys.path.insert(0, caffe_root + 'python')
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def cnn_model_params(file_name):
    net_model = os.path.join('models', file_name + 'deploy.prototxt')
    parsible_net = caffe_pb2.NetParameter()
    
    text_format.Merge(open(net_model).read(), parsible_net)
    #print(parsible_net.layer)
    
    conv = 0
    conv_feat = []
    conv_kernel = []
    conv_stride = []
    
    pool = 0
    pool_kernel = []
    pool_stride = []
    
    fc = 0
    fc_neurons = []
    
    feat_layers = []
    fc_layers = []
    
    for layer in parsible_net.layer:
        if layer.type == 'Convolution':
            conv += 1
            conv_kernel.append(layer.convolution_param.kernel_size[0])
            conv_stride.append(layer.convolution_param.stride[0] \
                               if len(layer.convolution_param.stride) else 1)
            conv_feat.append(layer.convolution_param.num_output)
        elif layer.type == 'Pooling':
            pool += 1
            pool_kernel.append(layer.pooling_param.kernel_size)
            pool_stride.append(layer.pooling_param.stride)
        elif layer.type == 'InnerProduct':
            fc += 1
            fc_neurons.append(layer.inner_product_param.num_output)
            fc_layers.append(str(layer.name))
            
        if layer.type == 'Convolution' or layer.type == 'Pooling':
            feat_layers.append(str(layer.name))
    return conv, pool, fc, conv_kernel, conv_stride, conv_feat, conv_kernel, \
pool_stride, pool_kernel, fc_neurons, feat_layers, fc_layers 

def write_cnn_types(net, nc, ncw, nfc, nfcw, file_name):
    conv, pool, fc, conv_kernel, conv_stride, conv_feat, conv_kernel, \
    pool_stride, pool_kernel,  fc_neurons, feat_layers, fc_layers = cnn_model_params(file_name)
    
    
    layers = []
    lays = []
    
    #Layer names abbreviation        
    for l in range(conv):
        la = l + 1
        layers.append('c' + str(la))
        layers.append('p' + str(la))
    
    for l in range(fc):
        la = l  + 1
        layers.append('fc' + str(la))
    
    #Convolutional and Pooling layers abbreviation    
    for l in range(conv):
        la = l + 1
        lays.append('c' + str(la))
        lays.append('p' + str(la))
        
    
    #Memory values
    width = 32
    nbytes = 4
    
    #Last pooling layer image dimensions
    dim = width
    p_dim = 0
    for i in range(conv):
        dim /= conv_stride[i]
        rate = conv_kernel[i] / conv_stride[i]
        if conv_kernel[i] > conv_stride[i]:
            if conv_kernel[i] % conv_stride[i] == 0:
                dim -= rate - 1
            else:
                dim -= rate
        
        dim /= pool_stride[i]
        rate = pool_kernel[i] / pool_stride[i]
        if pool_kernel[i] > pool_stride[i]:
            if pool_kernel[i] % pool_stride[i] == 0:
                dim -= rate - 1
            else:
                dim -= rate
    
    p_dim = dim
    
    #Convolutional and Pooling offsets definition
    dim = 2
    conv_step = np.zeros((conv, dim))
    pool_step = np.zeros((pool, dim))
    conv_offset = np.zeros((conv, dim))
    pool_offset  = np.zeros((pool, dim))
    
    n_offset = [[[]for d in range(dim)] for l in range(len(lays))]
    
    for l in range(conv):
        if l == 0:
            conv_step[l][0] = conv_stride[l]*nbytes*width
            conv_step[l][1] = conv_stride[l]*nbytes
            
            for d in range(dim):
                conv_offset[l][d] = (pool_kernel[l] - 1)*conv_step[l][d]
                
                pool_step[l][d] = pool_stride[l]*conv_step[l][d]
                pool_offset[l][d] = (conv_kernel[l + 1] - 1)*pool_step[l][d]
        else:
            for d in range(dim):
                conv_step[l][d] = conv_stride[l]*pool_step[l - 1][d]
                conv_offset[l][d] = (pool_kernel[l] - 1)*conv_step[l][d] 
                
                pool_step[l][d] = pool_stride[l]*conv_step[l][d]
                if l == (conv - 1):
                    pool_offset[l][d] = (p_dim - 1)*pool_step[l][d]
                else:
                    pool_offset[l][d] = (conv_kernel[l + 1] - 1)*pool_step[l][d]
                    
    fc_inputs = []
    for l in range(fc):
        if l == 0:
            fc_inputs.append(conv_feat[conv - 1]*p_dim**2)
        else:
            fc_inputs.append(fc_neurons[l - 1])
        
    #Convolutional and Pooling offsets bit width definition
    for l in range(len(feat_layers)):
        for d in range(dim):
            if 'conv' in feat_layers[l]:
                n_offset[l][d] = int(mt.ceil(mt.log(conv_offset[l/2][d], 2)))
            elif 'pool' in feat_layers[l]:
                n_offset[l][d] = int(mt.ceil(mt.log(pool_offset[(l - 1)/2][d], 2)))
                
    
    file_path = os.path.join('cnn_vhdl_generator', 'cnn_types.py')
    fi = open(file_path, 'w')

    fi.write('import numpy as np\n\n') 
    
    fi.write('layers = {0}\n'.format(layers))
    fi.write('feat_layers = {0}\n'.format(feat_layers))
    fi.write('lays = {0}\n\n'.format(lays))
    
    fi.write('#MEMORY\n')
    fi.write('width = {0}\n'.format(32))
    fi.write('start_bytes = {0}\n'.format(1))
    fi.write('nbytes = {0}\n'.format(4))
    fi.write('pixel = {0}\n\n'.format(8))
    
    fi.write('#CONVOLUTION\n')  
    fi.write('conv = {0}\n'.format(conv))
    fi.write('conv_feat = {0}\n'.format(conv_feat))   
    fi.write('conv_kernel = {0}\n'.format(conv_kernel))  
    fi.write('conv_stride = {0}\n\n'.format(conv_stride))  
    
    fi.write('#POOLING\n')
    fi.write('pool = {0}\n'.format(pool))  
    fi.write('pool_kernel = {0}\n'.format(pool_kernel))  
    fi.write('pool_stride = {0}\n'.format(pool_stride))
    fi.write('p{0}_dim = {1}\n\n'.format(pool, p_dim)) 
    
    fi.write('#FULLY-CONNECTED\n')
    fi.write('fc = {0}\n'.format(fc))
    fi.write('fc_neurons = {0}\n'.format(fc_neurons))
    fi.write('fc_inputs = {0}\n\n'.format(fc_inputs))
    
    fi.write('#COUNTER OFFSET BIT-WIDTH\n')
    fi.write('n_offset = {0}\n\n'.format(n_offset))
    
    fi.write('#Conv layer output integer-part\n') 
    fi.write('ncoi = {0}\n'.format(nc))
    fi.write('#Conv weights fractional-part\n')
    fi.write('ncwf = {0}\n\n'.format(ncw))
    
    fi.write('#FC layer output integer-part\n')
    fi.write('nfoi = {0}\n'.format(nfc))
    fi.write('#FC weight fractional-part\n')
    fi.write('nfwf = {0}\n\n'.format(nfcw))       
    
    
    #Convolutional params
    fi.write('conv_bias = []\n')
    fi.write('conv_weight = []\n')
    prev_feat = []
    for l in range(conv):
        if l== 0:
            prev_feat.append(1)
            fi.write('conv_weight.append(np.zeros(({0}, {1}, {2}, {3})))\n'.format(conv_feat[l],\
                     1, conv_kernel[l], conv_kernel[l])) 
        else:
            prev_feat.append(conv_feat[l - 1])
            fi.write('conv_weight.append(np.zeros(({0}, {1}, {2}, {3})))\n'.format(conv_feat[l], \
                     conv_feat[l - 1], conv_kernel[l], conv_kernel[l])) 
        fi.write('conv_bias.append(np.zeros(({0})))\n'.format(conv_feat[l])) 
    
    
    for l in range(conv):
        la = l + 1
        for n in range(conv_feat[l]):
            #Bias
            ncb = ncw[l]*2
            d = int((net.params['conv' + str(la)][1].data[n])*2**ncb)
            fi.write('conv_bias[{0}][{1}]= {2}\n'.format(l, n, d))
            #Weights
            for f in range(prev_feat[l]):
                for x in range(conv_kernel[l]):
                    for y in range(conv_kernel[l]):
                        d = int((net.params['conv' + str(la)][0].data[n][f][x][y])*2**ncw[l])
                        fi.write('conv_weight[{0}][{1}][{2}][{3}][{4}]= {5}\n'.format(l, n, f, x, y, d))
                      
    #Fully-connected params
    fi.write('fc_bias = []\n') 
    fi.write('fc_weight = []\n')
    for l in range(fc):
        if l == 0:
            fi.write('fc_weight.append(np.zeros(({0}, {1})))\n'.format(fc_neurons[l], conv_feat[conv - 1]*p_dim**2))
        else:
            fi.write('fc_weight.append(np.zeros(({0}, {1})))\n'.format(fc_neurons[l], fc_neurons[l - 1]))
        fi.write('fc_bias.append(np.zeros(({0})))\n'.format(fc_neurons[l]))
    
    for l in range(fc):
        la = l + 1
        for n in range(fc_neurons[l]):
            #Bias
            nfcb = nfcw[l]*2
            if  l == fc - 1:
                d = int((net.params['score'][1].data[n])*2**nfcb)
            else:
                d = int((net.params['fc' + str(la)][1].data[n])*2**nfcb)
            fi.write('fc_bias[{0}][{1}] = {2}\n'.format(l, n, d)) 
            #Weights
            for i in range(fc_inputs[l]):
                if  l == fc - 1:
                    d = int((net.params['score'][0].data[n][i])*2**nfcw[l])
                else:
                    d = int((net.params['fc' + str(la)][0].data[n][i])*2**nfcw[l])
                fi.write('fc_weight[{0}][{1}][{2}] = {3}\n'.format(l, n, i, d))        
    fi.close()
    return

