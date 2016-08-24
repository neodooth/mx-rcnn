import mxnet as mx
import rpn.proposal
import rpn.proposal_target
from config import config


use_global_stats = True
fix_gamma = False

def get_shared_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=True)
    bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1 , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    conv1_relu = mx.symbol.Activation(name='conv1_relu', data=bn_conv1 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu , pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
    res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1 , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1 , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1 , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=bn2a_branch2a , act_type='relu')
    res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=bn2a_branch2b , act_type='relu')
    res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2a = mx.symbol.ElementWiseSum(name='res2a', *[bn2a_branch1,bn2a_branch2c] , num_args=2)
    res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a , act_type='relu')
    res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=bn2b_branch2a , act_type='relu')
    res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=bn2b_branch2b , act_type='relu')
    res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2b = mx.symbol.ElementWiseSum(name='res2b', *[res2a_relu,bn2b_branch2c] , num_args=2)
    res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b , act_type='relu')
    res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu , num_filter=64, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=bn2c_branch2a , act_type='relu')
    res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=bn2c_branch2b , act_type='relu')
    res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res2c = mx.symbol.ElementWiseSum(name='res2c', *[res2b_relu,bn2c_branch2c] , num_args=2)
    res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c , act_type='relu')
    res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1 , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=bn3a_branch2a , act_type='relu')
    res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=bn3a_branch2b , act_type='relu')
    res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3a = mx.symbol.ElementWiseSum(name='res3a', *[bn3a_branch1,bn3a_branch2c] , num_args=2)
    res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a , act_type='relu')
    res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=bn3b1_branch2a , act_type='relu')
    res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=bn3b1_branch2b , act_type='relu')
    res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3b1 = mx.symbol.ElementWiseSum(name='res3b1', *[res3a_relu,bn3b1_branch2c] , num_args=2)
    res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1 , act_type='relu')
    res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=bn3b2_branch2a , act_type='relu')
    res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=bn3b2_branch2b , act_type='relu')
    res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3b2 = mx.symbol.ElementWiseSum(name='res3b2', *[res3b1_relu,bn3b2_branch2c] , num_args=2)
    res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2 , act_type='relu')
    res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu , num_filter=128, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=bn3b3_branch2a , act_type='relu')
    res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=bn3b3_branch2b , act_type='relu')
    res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res3b3 = mx.symbol.ElementWiseSum(name='res3b3', *[res3b2_relu,bn3b3_branch2c] , num_args=2)
    res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3 , act_type='relu')
    res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1 , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(2,2), no_bias=True)
    bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=bn4a_branch2a , act_type='relu')
    res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=bn4a_branch2b , act_type='relu')
    res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4a = mx.symbol.ElementWiseSum(name='res4a', *[bn4a_branch1,bn4a_branch2c] , num_args=2)
    res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a , act_type='relu')
    res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=bn4b1_branch2a , act_type='relu')
    res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=bn4b1_branch2b , act_type='relu')
    res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b1 = mx.symbol.ElementWiseSum(name='res4b1', *[res4a_relu,bn4b1_branch2c] , num_args=2)
    res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1 , act_type='relu')
    res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=bn4b2_branch2a , act_type='relu')
    res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=bn4b2_branch2b , act_type='relu')
    res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b2 = mx.symbol.ElementWiseSum(name='res4b2', *[res4b1_relu,bn4b2_branch2c] , num_args=2)
    res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2 , act_type='relu')
    res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=bn4b3_branch2a , act_type='relu')
    res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=bn4b3_branch2b , act_type='relu')
    res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b3 = mx.symbol.ElementWiseSum(name='res4b3', *[res4b2_relu,bn4b3_branch2c] , num_args=2)
    res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3 , act_type='relu')
    res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=bn4b4_branch2a , act_type='relu')
    res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=bn4b4_branch2b , act_type='relu')
    res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b4 = mx.symbol.ElementWiseSum(name='res4b4', *[res4b3_relu,bn4b4_branch2c] , num_args=2)
    res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4 , act_type='relu')
    res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=bn4b5_branch2a , act_type='relu')
    res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=bn4b5_branch2b , act_type='relu')
    res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b5 = mx.symbol.ElementWiseSum(name='res4b5', *[res4b4_relu,bn4b5_branch2c] , num_args=2)
    res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5 , act_type='relu')
    res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=bn4b6_branch2a , act_type='relu')
    res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=bn4b6_branch2b , act_type='relu')
    res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b6 = mx.symbol.ElementWiseSum(name='res4b6', *[res4b5_relu,bn4b6_branch2c] , num_args=2)
    res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6 , act_type='relu')
    res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=bn4b7_branch2a , act_type='relu')
    res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=bn4b7_branch2b , act_type='relu')
    res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b7 = mx.symbol.ElementWiseSum(name='res4b7', *[res4b6_relu,bn4b7_branch2c] , num_args=2)
    res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7 , act_type='relu')
    res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=bn4b8_branch2a , act_type='relu')
    res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=bn4b8_branch2b , act_type='relu')
    res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b8 = mx.symbol.ElementWiseSum(name='res4b8', *[res4b7_relu,bn4b8_branch2c] , num_args=2)
    res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8 , act_type='relu')
    res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=bn4b9_branch2a , act_type='relu')
    res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=bn4b9_branch2b , act_type='relu')
    res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b9 = mx.symbol.ElementWiseSum(name='res4b9', *[res4b8_relu,bn4b9_branch2c] , num_args=2)
    res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9 , act_type='relu')
    res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=bn4b10_branch2a , act_type='relu')
    res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=bn4b10_branch2b , act_type='relu')
    res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b10 = mx.symbol.ElementWiseSum(name='res4b10', *[res4b9_relu,bn4b10_branch2c] , num_args=2)
    res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10 , act_type='relu')
    res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=bn4b11_branch2a , act_type='relu')
    res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=bn4b11_branch2b , act_type='relu')
    res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b11 = mx.symbol.ElementWiseSum(name='res4b11', *[res4b10_relu,bn4b11_branch2c] , num_args=2)
    res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11 , act_type='relu')
    res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=bn4b12_branch2a , act_type='relu')
    res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=bn4b12_branch2b , act_type='relu')
    res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b12 = mx.symbol.ElementWiseSum(name='res4b12', *[res4b11_relu,bn4b12_branch2c] , num_args=2)
    res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12 , act_type='relu')
    res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=bn4b13_branch2a , act_type='relu')
    res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=bn4b13_branch2b , act_type='relu')
    res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b13 = mx.symbol.ElementWiseSum(name='res4b13', *[res4b12_relu,bn4b13_branch2c] , num_args=2)
    res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13 , act_type='relu')
    res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=bn4b14_branch2a , act_type='relu')
    res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=bn4b14_branch2b , act_type='relu')
    res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b14 = mx.symbol.ElementWiseSum(name='res4b14', *[res4b13_relu,bn4b14_branch2c] , num_args=2)
    res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14 , act_type='relu')
    res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=bn4b15_branch2a , act_type='relu')
    res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=bn4b15_branch2b , act_type='relu')
    res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b15 = mx.symbol.ElementWiseSum(name='res4b15', *[res4b14_relu,bn4b15_branch2c] , num_args=2)
    res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15 , act_type='relu')
    res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=bn4b16_branch2a , act_type='relu')
    res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=bn4b16_branch2b , act_type='relu')
    res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b16 = mx.symbol.ElementWiseSum(name='res4b16', *[res4b15_relu,bn4b16_branch2c] , num_args=2)
    res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16 , act_type='relu')
    res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=bn4b17_branch2a , act_type='relu')
    res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=bn4b17_branch2b , act_type='relu')
    res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b17 = mx.symbol.ElementWiseSum(name='res4b17', *[res4b16_relu,bn4b17_branch2c] , num_args=2)
    res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17 , act_type='relu')
    res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=bn4b18_branch2a , act_type='relu')
    res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=bn4b18_branch2b , act_type='relu')
    res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b18 = mx.symbol.ElementWiseSum(name='res4b18', *[res4b17_relu,bn4b18_branch2c] , num_args=2)
    res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18 , act_type='relu')
    res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=bn4b19_branch2a , act_type='relu')
    res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=bn4b19_branch2b , act_type='relu')
    res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b19 = mx.symbol.ElementWiseSum(name='res4b19', *[res4b18_relu,bn4b19_branch2c] , num_args=2)
    res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19 , act_type='relu')
    res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=bn4b20_branch2a , act_type='relu')
    res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=bn4b20_branch2b , act_type='relu')
    res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b20 = mx.symbol.ElementWiseSum(name='res4b20', *[res4b19_relu,bn4b20_branch2c] , num_args=2)
    res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20 , act_type='relu')
    res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=bn4b21_branch2a , act_type='relu')
    res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=bn4b21_branch2b , act_type='relu')
    res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b21 = mx.symbol.ElementWiseSum(name='res4b21', *[res4b20_relu,bn4b21_branch2c] , num_args=2)
    res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21 , act_type='relu')
    res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu , num_filter=256, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=bn4b22_branch2a , act_type='relu')
    res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=bn4b22_branch2b , act_type='relu')
    res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res4b22 = mx.symbol.ElementWiseSum(name='res4b22', *[res4b21_relu,bn4b22_branch2c] , num_args=2)
    res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22 , act_type='relu')

    return res4b22_relu


def get_unshared_part(data):
    res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=data , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1 , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=data , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=bn5a_branch2a , act_type='relu')
    res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=bn5a_branch2b , act_type='relu')
    res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5a = mx.symbol.ElementWiseSum(name='res5a', *[bn5a_branch1,bn5a_branch2c] , num_args=2)
    res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a , act_type='relu')
    res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=bn5b_branch2a , act_type='relu')
    res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=bn5b_branch2b , act_type='relu')
    res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5b = mx.symbol.ElementWiseSum(name='res5b', *[res5a_relu,bn5b_branch2c] , num_args=2)
    res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b , act_type='relu')
    res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=bn5c_branch2a , act_type='relu')
    res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=True)
    bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=bn5c_branch2b , act_type='relu')
    res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu , num_filter=2048, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=True)
    bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c , use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    res5c = mx.symbol.ElementWiseSum(name='res5c', *[res5b_relu,bn5c_branch2c] , num_args=2)
    res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c , act_type='relu')
    pool5 = mx.symbol.Pooling(name='pool5', data=res5c_relu , pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
    return pool5


def get_resnet_rcnn(num_classes=201):
    """
    Fast R-CNN with VGG 16 conv layers
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    rois = mx.symbol.Variable(name='rois')
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_inside_weight = mx.symbol.Variable(name='bbox_inside_weight')
    bbox_outside_weight = mx.symbol.Variable(name='bbox_outside_weight')

    # reshape input
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
    label = mx.symbol.Reshape(data=label, shape=(-1, ), name='label_reshape')
    bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_classes), name='bbox_target_reshape')
    bbox_inside_weight = mx.symbol.Reshape(data=bbox_inside_weight, shape=(-1, 4 * num_classes), name='bbox_inside_weight_reshape')
    bbox_outside_weight = mx.symbol.Reshape(data=bbox_outside_weight, shape=(-1, 4 * num_classes), name='bbox_outside_weight_reshape')

    # shared convolutional layers
    shared_conv = get_shared_conv(data)

    # Fast R-CNN
    roipool = mx.symbol.ROIPooling(
        name='roi_pool5', data=shared_conv, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625) # 1/16
    unshared = get_unshared_part(roipool)
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=unshared, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=unshared, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_outside_weight * \
                 mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0,
                                     data=bbox_inside_weight * (bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')

    # group output
    group = mx.symbol.Group([cls_prob, bbox_loss])
    return group


def get_resnet_rcnn_test(num_classes=201):
    """
    Fast R-CNN Network with VGG
    :param num_classes: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    rois = mx.symbol.Variable(name='rois')

    # reshape rois
    rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

    # shared convolutional layer
    shared_conv = get_shared_conv(data)

    # Fast R-CNN
    roipool = mx.symbol.ROIPooling(
        name='roi_pool5', data=shared_conv, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
    unshared = get_unshared_part(roipool)
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=unshared, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=unshared, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([cls_prob, bbox_pred])
    return group


def get_resnet_rpn(num_classes=201, num_anchors=9):
    """
    Region Proposal Network with VGG
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_inside_weight = mx.symbol.Variable(name='bbox_inside_weight')
    bbox_outside_weight = mx.symbol.Variable(name='bbox_outside_weight')

    # shared convolutional layers
    shared_conv = get_shared_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=shared_conv, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1), name="rpn_cls_score_reshape")

    # classification
    cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=label, multi_output=True,
                                       normalization='valid', use_ignore=True, ignore_label=-1, name="cls_prob")
    # bounding box regression
    bbox_loss_ = bbox_outside_weight * \
                 mx.symbol.smooth_l1(name='bbox_loss_', scalar=3.0,
                                     data=bbox_inside_weight * (rpn_bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.)

    # debug layers
    sym_group = []
    # sym = shared_conv.get_internals()
    # blob_names = sym.list_outputs()
    # for i in range(len(blob_names)):
    #     if blob_names[i].startswith('bn') or blob_names[i].startswith('conv') or blob_names[i].startswith('res2a'):
    #         x = sym[i]
    #         x = mx.symbol.BlockGrad(x, name=blob_names[i])
    #         sym_group.append(x)

    # group output
    group = mx.symbol.Group([cls_prob, bbox_loss] + sym_group)
    return group


def get_resnet_rpn_test(num_classes=201, num_anchors=9):
    """
    Region Proposal Network with VGG
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    shared_conv = get_shared_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=shared_conv, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    group = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2), output_score=True)
    # rois = group[0]
    # score = group[1]

    return group


def get_resnet_joint(num_classes=201, num_anchors=9, is_train=False):
    """
    Faster R-CNN
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    label = mx.symbol.Variable(name='label')
    bbox_target = mx.symbol.Variable(name='bbox_target')
    bbox_inside_weight = mx.symbol.Variable(name='bbox_inside_weight')
    bbox_outside_weight = mx.symbol.Variable(name='bbox_outside_weight')

    # shared convolutional layers
    shared_conv = get_shared_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=shared_conv, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    # rpn_cls_score = mx.symbol.BlockGrad(rpn_cls_score)
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
    # rpn_bbox_pred = mx.symbol.BlockGrad(rpn_bbox_pred)

    rpn_losses = []
    rcnn_losses = []

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # rpn_cls_score_reshape_for_rpn = mx.symbol.Reshape(
    #     data=rpn_cls_score, shape=(0, 2, -1), name="rpn_cls_score_reshape_for_rpn")
    rpn_cls_score_reshape_for_rpn = rpn_cls_score_reshape

    # classification
    rpn_cls_prob_output = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape_for_rpn, label=label, multi_output=True,
                                       normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob_output")
    # # bounding box regression
    rpn_bbox_loss_ = bbox_outside_weight * \
                 mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                     data=bbox_inside_weight * (rpn_bbox_pred - bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_)

    rpn_losses = [rpn_cls_prob_output, rpn_bbox_loss]

    # ROI Proposal
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    # rpn_cls_prob_reshape = mx.symbol.BlockGrad(rpn_cls_prob_reshape)
    rpn_roi = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rpn_rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2), is_train=is_train)  # TODO(be careful of cls_prob)
    # rpn_roi = mx.symbol.BlockGrad(rpn_roi)
    rois = mx.symbol.Custom(
        rpn_roi=rpn_roi, gt_boxes=gt_boxes, name='rois', op_type='proposal_target',
        num_classes=num_classes, is_train=is_train)

    # Fast R-CNN
    roipool = mx.symbol.ROIPooling(
        name='roi_pool5', data=shared_conv, rois=rois[0], pooled_size=(7, 7), spatial_scale=0.0625)
    unshared = get_unshared_part(roipool)
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=unshared, num_hidden=num_classes)
    # cls_score = mx.symbol.BlockGrad(cls_score)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=rois[1])
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=unshared, num_hidden=num_classes * 4)
    # bbox_pred = mx.symbol.BlockGrad(bbox_pred)
    bbox_loss_ = rois[4] * \
                 mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0,
                                     data=rois[3] * (bbox_pred - rois[2]))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.IMS_PER_BATCH, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.IMS_PER_BATCH, -1, 4 * num_classes), name='bbox_loss_reshape')

    rcnn_losses = [cls_prob, bbox_loss]

    # group output
    if is_train:
        group = mx.symbol.Group([rois[1]] + rpn_losses + rcnn_losses)  # rois[1] is used for evaluation
    else:
        group = mx.symbol.Group(rpn_losses + rcnn_losses)
    return group


def get_resnet_test(num_classes=201, num_anchors=9):
    """
    Faster R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    shared_conv = get_shared_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=shared_conv, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    rois = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2))

    # Fast R-CNN
    roipool = mx.symbol.ROIPooling(
        name='roi_pool5', data=shared_conv, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
    unshared = get_unshared_part(roipool)
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=unshared, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=unshared, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group
