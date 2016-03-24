import numpy as np
import theano
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init

from lasagne.layers import MergeLayer, Layer
from lasagne.layers import helper

__all__ = [
    "ConcatRecurrentLayer",
]

# adapted from Lasagne's CustomRecurrentLayer implementation:
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/recurrent.py

class ConcatRecurrentLayer(MergeLayer):
    def __init__(self, incoming, input_to_hidden, hidden_to_hidden, post_concat,
                 hid_init=init.Constant(0.), learn_init=False,
                 grad_clipping=0, 
                 **kwargs):

        incomings = [incoming]
        self.hid_init_incoming_index = -1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1

        super(ConcatRecurrentLayer, self).__init__(incomings, **kwargs)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.post_concat = post_concat
        self.learn_init = learn_init
        self.grad_clipping = grad_clipping

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1,) + hidden_to_hidden.output_shape[1:],
                name="hid_init", trainable=learn_init, regularizable=False)

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(ConcatRecurrentLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.input_to_hidden, **tags)
        params += helper.get_all_params(self.hidden_to_hidden, **tags)
        return params

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return ((input_shape[0], input_shape[1]) + self.post_concat.output_shape[1:])

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        hid_init = None
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, *range(2, input.ndim))
        seq_len, num_batch = input.shape[0], input.shape[1]

        # precompute inputs before scanning
        trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
        input = T.reshape(input, (seq_len*num_batch,) + trailing_dims)
        input = helper.get_output(
            self.input_to_hidden, input, **kwargs)

        # Reshape back to (seq_len, batch_size, trailing dimensions...)
        trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
        input = T.reshape(input, (seq_len, num_batch) + trailing_dims)

        # pass params to step
        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        non_seqs += helper.get_all_params(self.post_concat)

        # Create single recurrent computation step function
        def step(input_n, hid_previous, *args):
            # Compute the hidden-to-hidden activation
            hid_pre = helper.get_output(
                self.hidden_to_hidden, hid_previous, **kwargs)
            hid_pre = T.concatenate([hid_pre, input_n], axis=1)
            hid_pre = helper.get_output(self.post_concat, hid_pre, **kwargs)
            if self.grad_clipping:
                hid_pre = theano.gradient.grad_clip(
                    hid_pre, -self.grad_clipping, self.grad_clipping)
            return hid_pre

        sequences = input
        step_fun = step

        if not isinstance(self.hid_init, Layer):
            # repeats self.hid_init num_batch times in first dimension
            dot_dims = (list(range(1, self.hid_init.ndim - 1)) +
                        [0, self.hid_init.ndim - 1])
            hid_init = T.dot(T.ones((num_batch, 1)),
                             self.hid_init.dimshuffle(dot_dims))

        hid_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            go_backwards=False,
            outputs_info=[hid_init],
            non_sequences=non_seqs,
            truncate_gradient=-1,
            strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hid_out.dimshuffle(1, 0, *range(2, hid_out.ndim))

        return hid_out
