caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import os, errno
from caffe import layers as L, params as P
import matplotlib.pyplot as plt
import numpy as np
from caffe.proto import caffe_pb2

s = caffe_pb2.SolverParameter()
s.random_seed = 0xCAFFE

niter = 80000
test_interval = 500
test_iter = 60

def create_solver(file_name):
    train_net_path = 'models/' + file_name + 'train.prototxt'
    test_net_path = 'models/' + file_name + 'test.prototxt'

    s.train_net = train_net_path
    if len(s.test_net) > 0:
        s.test_net.pop()
    s.test_net.append(test_net_path)
    s.test_interval = test_interval
    if len(s.test_iter) > 0:
        s.test_iter.pop()
    s.test_iter.append(test_iter)

    s.max_iter = niter

    s.type = "AdaGrad"
    s.base_lr = 0.01
    #s.momentum = 0.01
    s.weight_decay = 0.0005

    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75
    s.display = 500
    s.snapshot = 5000

    try:
        os.makedirs('snaps/' + file_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    s.snapshot_prefix = 'snaps/' + file_name + '/lenet'

    s.solver_mode = caffe_pb2.SolverParameter.GPU

    with open('ts_auto_solver.prototxt', 'w') as f:
        f.write(str(s))
    return

def run_trainning(file_name):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    ### load the solver and create train and test nets
    solver = caffe.get_solver('ts_auto_solver.prototxt')

    #losses will also be stored in the log
    train_loss = np.zeros(niter)
    test_acc = np.zeros(int(np.ceil(niter / test_interval)))

    for i in range(niter):
        solver.step(1)

        train_loss[i] = solver.net.blobs['loss'].data

        if i % test_interval == 0:
            print 'Iteration', i, 'testing...'
            test_acc[i // test_interval] = solver.test_nets[0].blobs['accuracy'].data
        
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.arange(niter), train_loss)
    ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')

    f = open('logs/' + file_name + 'log.txt', 'w')
    f.write('Accuracy\n')
    for i in range(len(test_acc)):
        f.write('{0} \n'.format(test_acc[i]))
    f.close()

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test accuracy')
    ax2.set_title(s.type + ' Prohibitory Accuracy: {:.2f}'.format(test_acc[-1]))
    plt.savefig('plots/' + file_name + '.png')
    return

