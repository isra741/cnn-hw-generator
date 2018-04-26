import create_net as cn
import train_net as tn
import precision as prec
import sys
sys.path.insert(0, 'cnn_vhdl_generator')
import gen_vhdl as gn


def gen_file_name(conv_layers, conv_feat, conv_kernel, pool_kernel, pool_stride, fc_layers, fc_neurons):
    file_name = ''
    for l in range(conv_layers):
        file_name +=  'cf' + str(conv_feat[l]) + '_'
        file_name +=  'ck' + str(conv_kernel[l]) + '_'

    file_name +=  'pk' + str(pool_kernel[l]) + '_'
    file_name +=  'ps' + str(pool_stride[l]) + '_'

    for l in range(fc_layers):
        file_name +=  'fc' + str(fc_neurons[l]) + '_'
    return file_name

def gen_default_model():
    #CNN model parameters
    conv_layers = 2
    conv_feat = [16, 32]
    conv_kernel = [5, 5]
    conv_stride = 1
    pool_kernel = [2, 2]
    pool_stride = [2, 2]
    fc_layers = 3
    fc_neurons = [800, 256, 12]
    file_name = []

    file_name = gen_file_name(conv_layers, conv_feat, conv_kernel, pool_kernel, pool_stride, fc_layers, fc_neurons)
    #cn.create_proto(file_name, conv_feat, conv_kernel, pool_kernel, pool_stride, fc_neurons)
    #tn.create_solver(file_name)
    #tn.run_trainning(file_name)
    #prec.gen_model_params(file_name)
    #prec.precision_change(file_name, 0, 8, 8)
    gn.gen_vhdl(file_name)
    
    return

def gen_best_models():
    #CNN model parameters
    trained_model = ['cf16_ck5_cf32_ck5_pk2_ps2_fc800_fc256_fc12_', 'lenet_iter_10000.caffemodel']
    prec.gen_model_params(trained_model[0])
    prec.precision_change(trained_model[0], trained_model[1], 0, 8, 8)
    gn.gen_vhdl(trained_model[0])
    return

def train_many_models():
    conv_layers = 2
    start_conv_feat = 16
    end_conv_feat = 32
    step_conv_feat = 16

    start_conv_kernel = 5
    end_conv_kernel = 7
    step_conv_kernel = 2

    start_pool_kernel = 2
    end_pool_kernel = 3
    step_pool_kernel = 1

    start_pool_stride = 1
    end_pool_stride = 2
    step_pool_stride = 1

    fc_layers = 3
    first_fc = [800, 400, 200]
    second_fc = [256, 100, 50]

    for cf in range(start_conv_feat, end_conv_feat + 1, step_conv_feat):
        for ck in range(start_conv_kernel, end_conv_kernel + 1, step_conv_kernel):
            for ck2 in range(ck - 2, ck + 1, step_conv_kernel):
                for pk in range(start_pool_kernel, end_pool_kernel + 1, step_pool_kernel):
                    for ps in range(start_pool_stride, end_pool_stride + 1, step_pool_stride): 
                        for f in range(3):
                            conv_feat  = [cf, cf*2]
                            conv_kernel  = [ck, ck2]
                            pool_kernel = [pk, pk]
                            pool_stride = [ps, ps]
                            fc_neurons = [first_fc[f], second_fc[f], 12]
                            file_name = gen_file_name(conv_layers, conv_feat, conv_kernel, pool_kernel, pool_stride, fc_layers, fc_neurons)
                            cn.create_proto(file_name,conv_feat,conv_kernel,pool_kernel, pool_stride,fc_neurons)
                            tn.create_solver(file_name)
                            tn.run_trainning(file_name)
    return

gen_best_models()



