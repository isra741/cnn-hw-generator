import cnn_types as ct
import numpy as np

def gen_package():
    dim = ['x', 'y']
    offset = ['step', 'limit']
    
    fi = open(ct.vhdl_path + '\cnn_package.vhd', 'w')

    fi.write('library IEEE;\n')
    fi.write('use IEEE.STD_LOGIC_1164.ALL;\n')
    fi.write('use IEEE.NUMERIC_STD.ALL;\n\n')

    fi.write('package cnn_package is\n')
    fi.write('--memory constants\n')
    fi.write('constant width : integer :={0};\n'.format(ct.width))
    fi.write('constant nbytes: integer:= {0};\n'.format(ct.nbytes))
    fi.write('constant start_bytes: integer :={0};\n'.format(ct.start_bytes))
    fi.write('-------------------------------------\n')
    fi.write('--CNN TOPOLOGY DEFINITION\n')
    fi.write('-------------------------------------\n')
    
    for l in range(ct.conv):
        la = l + 1
        fi.write('--Conv{0} constants\n'.format(la))
        fi.write('constant c{0}_kernel: integer := {1};\n'.format(la, ct.conv_kernel[l]))
        fi.write('constant c{0}_stride: integer := {1};\n'.format(la, ct.conv_stride[l]))
        fi.write('constant c{0}_feat_maps: integer := {1};\n\n'.format(la, ct.conv_feat[l]))

    for l in range(ct.pool):
        la = l + 1
        fi.write('--Pool{0} constants\n'.format(la))
        fi.write('constant p{0}_kernel: integer := {1};\n'.format(la, ct.pool_kernel[l]))
        fi.write('constant p{0}_stride: integer := {1};\n\n'.format(la, ct.pool_stride[l]))

    fi.write('constant p{0}_dim: integer := {1};\n'.format(ct.pool, ct.p2_dim))
    fi.write('constant p{0}_window: integer := p{0}_dim*p{0}_dim;\n\n'.format(ct.pool))

    for l in range(ct.fc):
        la = l + 1
        fi.write('--FC{0} constants\n'.format(la))
        fi.write('constant fc{0}_neurons: integer := {1};\n'.format(la, ct.fc_neurons[l]))
        if l == 0:
            fi.write('constant fc{0}_inputs: integer := p{1}_window*c{1}_feat_maps;\n'.format(la, ct.pool))    
        else:
            fi.write('constant fc{0}_inputs: integer := fc{1}_neurons;\n'.format(la, l))
        fi.write('constant fc{0}_dsps: integer:= {1};\n\n'.format(la, ct.fc_neurons[l]))

    fi.write('-------------------------------------\n')
    fi.write('--CNN MEMORY OFFSETS AND LIMITS\n')
    fi.write('-------------------------------------\n')

    for l in range(len(ct.lays)):
        fi.write("--{0}".format(ct.feat_layers[l]).title() +' offsets\n')
        for o in range(len(offset)):
            for d in range(len(dim)):
                if l == 0:
                    if offset[o] == 'step':
                        if dim[d] == 'x':
                            fi.write('constant {0}_step_{1}: integer := {0}_stride*nbytes*width;\n'.format(ct.lays[l], dim[d], ct.lays[l]))
                        else:
                            fi.write('constant {0}_step_y: integer := {0}_stride*nbytes;\n'.format(ct.lays[l], dim[d], ct.lays[l]))
                    else:
                        fi.write('constant {0}_limit_{1}: integer := ({2}_kernel - 1)*{0}_step_{1};\n'.format(ct.lays[l], dim[d], ct.lays[l + 1]))
                else:
                    if offset[o] == 'step':
                        fi.write('constant {0}_step_{1}: integer := {0}_stride*{2}_step_{1};\n'.format(ct.lays[l], dim[d], ct.lays[l - 1]))
                    else:
                        if l == len(ct.lays) - 1:
                            fi.write('constant {0}_limit_{1}: integer := (p{2}_dim - 1)*{0}_step_{1};\n'.format(ct.lays[l], dim[d], ct.pool))
                        else:
                            fi.write('constant {0}_limit_{1}: integer := ({2}_kernel - 1)*{0}_step_{1};\n'.format(ct.lays[l], dim[d], ct.lays[l + 1]))
                if dim[d] == 'y':
                    fi.write('\n')   

    #Conv parameters precision
    fi.write('-------------------------------------\n')
    fi.write('--CONV PARAMETERS PRECISION  = (1s |8 frac)\n')
    fi.write('-------------------------------------\n')
    fi.write('--CW - CONV WEIGHT  = (1s |{0} frac)\n'.format(ct.ncwf[0]))
    fi.write('subtype CONV_WEIGHT is SIGNED({0} downto 0);\n'.format(ct.ncwf[0]))
    fi.write('subtype CONV_BIAS is SIGNED({0} downto 0);\n'.format(ct.ncwf[0]*2 ))

    #Conv signal types declaration
    for l in range(ct.conv):
        la = l + 1
        fi.write('-------------------------------------\n')
        fi.write('--CONV {0} TYPE DECLARATIONS\n'.format(la))
        fi.write('-------------------------------------\n')
        fi.write('--Paramaters\n')
        fi.write('type CONV{0}_KERNEL is array (0 to (c{1}_kernel*c{2}_kernel - 1)) of CONV_WEIGHT ;\n'.format(la, la, la))
        if l == 0:
            fi.write('type CONV{0}_WEIGHT_LAYER is array (0 to (c{1}_feat_maps - 1)) of CONV{2}_KERNEL;\n'.format(la, la, la))
        else:
            fi.write('type CONV{0}_WEIGHT_NEURON is array (0 to (c{1}_feat_maps - 1)) of CONV{2}_KERNEL;\n'.format(la, la - 1, la))
            fi.write('type CONV{0}_WEIGHT_LAYER is array (0 to (c{1}_feat_maps - 1)) of CONV{2}_WEIGHT_NEURON;\n'.format(la, la, la))
        fi.write('type CONV{0}_BIAS_LAYER is array (0 to (c{1}_feat_maps - 1)) of CONV_BIAS;\n'.format(la, la))
        fi.write('--Outputs\n')
        
        fi.write('subtype CONV{0}_OUT is SIGNED ({1} downto 0);\n'.format(la, ct.ncoi[l]))
        fi.write('type CONV{0}_WEIGHT_MUX is array (0 to (c{1}_feat_maps - 1)) of CONV_WEIGHT;\n'.format(la, la))
        fi.write('subtype CONV{0}_MACC_OUT is SIGNED({1} downto 0);\n'.format(la, ct.ncoi[l] + ct.ncwf[l] + ct.ncwf[l]))      
        fi.write('type CONV{0}_MACC_OUT_LAYER is array (0 to (c{1}_feat_maps - 1)) of CONV{2}_MACC_OUT;\n'.format(la, la, la))

    #Pool signal types declarations
    for l in range(ct.pool):
        la = l + 1
        fi.write('------------------------------------\n')
        fi.write('--POOL {0} TYPE DECLARATIONS\n'.format(la))
        fi.write('-------------------------------------\n')
        fi.write('type POOL{0}_KERNEL is array(0 to (p{1}_kernel*p{2}_kernel - 1)) of CONV{3}_OUT;\n'.format(la, la, la, la))
        fi.write('type POOL{0}_PIXEL_LAYER is array(0 to (c{1}_feat_maps - 1)) of CONV{2}_OUT;\n'.format(la, la, la))
        fi.write('type POOL{0}_KERNEL_LAYER is array(0  to (c{1}_feat_maps - 1)) of POOL{2}_KERNEL;\n'.format(la, la, la))
        if l == (ct.pool - 1):
            fi.write('type POOL{0}_IMG_FC{1} is array(0 to (p{2}_window - 1)) of POOL{3}_PIXEL_LAYER;\n'.format(la, 1, ct.pool, la))
        else:
            fi.write('type POOL{0}_KERNEL_CONV{1} is array(0 to (c{2}_kernel*c{3}_kernel - 1)) of POOL{4}_PIXEL_LAYER;\n'.format(la, ct.conv, la, la, la))

    #FC parameters precision
    fi.write('-------------------------------------\n')
    fi.write('--FC PARAMETERS PRECISION  = (1s |8 frac)\n')
    fi.write('-------------------------------------\n')
    fi.write('--FC_WEIGHT  - FULLY-CONNECTED WEIGHT  = (1s |{0} frac)\n'.format(ct.nfwf[0]))
    fi.write('subtype FC_WEIGHT is SIGNED({0} downto 0);\n'.format(ct.nfwf[0]))
    fi.write('subtype FC_BIAS is SIGNED({0} downto 0);\n'.format(ct.nfwf[0]*2 ))
    for l in range(ct.fc):
        la = l + 1
        fi.write('-------------------------------------\n')
        fi.write('--FC {0} TYPE DECLARATIONS\n'.format(la))
        fi.write('-------------------------------------\n')
        fi.write('--Parameters\n')
        fi.write('type FC{0}_WEIGHT_NEURON is array (0 to (fc{1}_inputs - 1)) of FC_WEIGHT;\n'.format(la, la))
        fi.write('type FC{0}_WEIGHT_LAYER is array (0 to (fc{1}_neurons - 1)) of FC{2}_WEIGHT_NEURON;\n'.format(la, la, la))
        fi.write('type FC{0}_BIAS_LAYER is array (0 to (fc{1}_neurons - 1)) of FC_BIAS;\n'.format(la, la))
        fi.write('type FC{0}_WEIGHT_MUX is array (0 to (fc{1}_neurons - 1)) of FC_WEIGHT;\n'.format(la, la))
        fi.write('--Outputs\n')
        fi.write('subtype FC{0}_RELU is SIGNED({1} downto 0);\n'.format(la, ct.nfoi[l]))
        fi.write('type FC{0}_RELU_LAYER is array (0 to (fc{1}_neurons - 1)) of FC{2}_RELU;\n'.format(la, la, la))
        fi.write('subtype FC{0}_MACC_OUT is SIGNED({1} downto 0);\n'.format(la, ct.nfoi[l] + ct.nfwf[l] + ct.nfwf[l]))
        fi.write('type FC{0}_MACC_OUT_LAYER is array (0 to (fc{1}_dsps - 1)) of FC{2}_MACC_OUT;\n'.format(la, la, la))

    fi.write('--------------------------------------\n')
    fi.write('--CONV WEIGHT CONSTANTS\n')
    fi.write('--------------------------------------\n')


    fi.write('end cnn_package;\n')
    fi.close()
    return

