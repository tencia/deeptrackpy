import sys
import os
import numpy as np
from datetime import datetime

from PIL import Image
from images2gif import writeGif

import lasagne as nn
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, Conv2DLayer, ReshapeLayer, NonlinearityLayer
from lasagne.nonlinearities import sigmoid

import utils as u
import concat_recurrent_layer as crl

def weighted_bce(predictions, targets, eps=1e-10):
    targets, weights = targets[:,:,:1], targets[:,:,1:]
    buf = T.log(predictions + eps) * weights
    ret = - T.sum(targets * buf, axis=[-1,-2])
    buf = T.log(-1 * predictions + 1 + eps) * weights
    ret = (ret - T.sum(buf, axis=[-1,-2]) + T.sum(targets*buf, axis=[-1,-2]))\
            / (targets.shape[-1]*targets.shape[-2])
    return ret

# architecture from https://github.com/pondruska/DeepTracking
# slightly different from the one in the paper
def rnn_orig(input_var, seq_len, sz=51):
    def add_shapes(sh1, sh2, axis=2):
        if isinstance(sh2, tuple):
            return sh1[:axis] + (sh1[axis]+sh2[axis],) + sh1[axis+1:]
        else:
            return sh1[:axis] + (sh1[axis]+sh2,) + sh1[axis+1:]
    ret = {}
    ret['input'] = in_layer = InputLayer((None,seq_len,2,sz,sz), input_var)
    ret['in_to_hid'] = in_to_hid = Conv2DLayer(InputLayer((None,2,sz,sz)), 16, 7, pad=3,
            nonlinearity=sigmoid)
    ret['post_concat'] = post_concat = Conv2DLayer(InputLayer(
        add_shapes(in_to_hid.output_shape, 32, 1)), 32, 7, pad=3, nonlinearity=sigmoid)
    ret['hid_to_hid'] = hid_to_hid = NonlinearityLayer(InputLayer(post_concat.output_shape),
            nonlinearity=None)
    ret['rec'] = f = crl.ConcatRecurrentLayer(in_layer, in_to_hid, hid_to_hid, post_concat)
    ret['rec_resh'] = f = ReshapeLayer(f, (-1,[2],[3],[4]))
    ret['y_pre'] = f = Conv2DLayer(f, 1,7,pad=3,nonlinearity=sigmoid)
    ret['output'] = f = ReshapeLayer(f, (-1,seq_len,[1],[2],[3]))
    return ret, nn.layers.get_output(ret['output']), nn.layers.get_output(ret['output'],
            deterministic=True)

def main(
        num_epochs=10,
        seq_len = 100,
        batch_size=2
    ):

    dtensor5 = T.TensorType('float32', (False,)*5)
    input_var = dtensor5('input')
    target_var = dtensor5('target')
    net,output, output_det = rnn_orig(input_var, seq_len=seq_len)
    params = nn.layers.get_all_params(net['output'])
    pred = theano.function([input_var], output)

    lr = theano.shared(nn.utils.floatX(1e-3))
    loss = weighted_bce(output, target_var).mean()
    loss_det = weighted_bce(output_det, target_var).mean()
    updates = nn.updates.adam(loss, params, learning_rate = lr)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    test_fn = theano.function([input_var, target_var], loss_det)

    data = u.DataH5PyStreamer('data/deeptrack.hdf5', folds=(10,0), batch_size=batch_size*seq_len)

    def transform_data(imb):
        tgt = imb[0].reshape((-1,seq_len) + imb[0].shape[1:]).astype(theano.config.floatX)
        inp = np.copy(tgt)
        for i in xrange(inp.shape[1]):
            if i % 20 >= 10:
                inp[:,i] = 0
        return inp,tgt
    if not os.path.exists('params'):
        os.makedirs('params')
    u.train_with_hdf5(data, num_epochs, train_fn, test_fn,
                      tr_transform = transform_data,
                      te_transform = transform_data,
                      train_shuffle = False,
                      max_per_epoch=-1,
                      use_tqdm=True,
                      #grad_clip=10,
                      last_layer = net['output'],
                      save_params_to='params/{}_params.npz'.format(datetime.strftime(datetime.now(), "%Y%m%d%H%M%S"))
                     )

if __name__ == '__main__':
    # make all arguments of main(...) command line arguments (with type inferred from
    # the default value) - this doesn't work on bools so those are strings when
    # passed into main.
    import argparse, inspect
    parser = argparse.ArgumentParser(description='Command line options')
    ma = inspect.getargspec(main)
    for arg_name,arg_type in zip(ma.args[-len(ma.defaults):],[type(de) for de in ma.defaults]):
        parser.add_argument('--{}'.format(arg_name), type=arg_type, dest=arg_name)
    args = parser.parse_args(sys.argv[1:])
    main(**{k:v for (k,v) in vars(args).items() if v is not None})

