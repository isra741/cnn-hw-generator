import cnn_package as cp
import top_module as tm
import main_control as mc
import conv_control as cc
import pool_control as pc
import fc_control as fc
import max_pool as mp


cp.gen_package()
tm.gen_top_module()
mc.gen_main_control()
cc.gen_conv_control()
pc.gen_pool_control()
fc.gen_fc_control()
mp.gen_max_pool()
