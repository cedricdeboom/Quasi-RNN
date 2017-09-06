import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import nonlinearities
from lasagne import init


__all__ = [
    "QRNNLayer"
]

class QRNNLayer(lasagne.layers.Layer):
    def __init__(self, incoming, seq_len, original_features, num_units, filter_width, pooling='f', **kwargs):
        assert pooling in ['f', 'fo', 'ifo']
        self.pooling = pooling
        
        self.incoming = incoming
        self.seq_len = seq_len
        self.original_features = original_features
        self.num_units = num_units
        self.filter_width = filter_width
        self.internal_seq_len = seq_len + filter_width - 1
        super(QRNNLayer, self).__init__(incoming, **kwargs)
        
        self.Z_W = self.add_param(init.GlorotUniform(),
                                  (self.num_units, 1, self.filter_width, self.original_features), name="Z_W")
        self.F_W = self.add_param(init.GlorotUniform(),
                                  (self.num_units, 1, self.filter_width, self.original_features), name="F_W")
        
        self.hid_init = self.add_param(init.Constant(0.), (1, self.num_units), name="hid_init",
                                        trainable=False, regularizable=False)
        
        if self.pooling == 'fo' or self.pooling == 'ifo':
            self.O_W = self.add_param(init.GlorotUniform(),
                                  (self.num_units, 1, self.filter_width, self.original_features), name="O_W")
        if self.pooling == 'ifo':
            self.I_W = self.add_param(init.GlorotUniform(),
                                  (self.num_units, 1, self.filter_width, self.original_features), name="I_W")
        
    def get_output_for(self, inputs, **kwargs):
        num_batch, _, _ = inputs.shape
        
        #add padded zeros in front of sequence
        padded_input = T.concatenate([T.zeros((num_batch, self.filter_width - 1, self.original_features)), inputs], axis=1)
        
        #reshape input to include 1 filter dimension
        rs = padded_input.dimshuffle([0, 'x', 1, 2])
        
        #apply convolutions for all "gates" (output = (n_batch, n_filters, n_time_steps, 1))
        Z = nonlinearities.tanh(T.nnet.conv2d(rs, self.Z_W,
                                              input_shape=(None, 1, self.internal_seq_len, self.original_features),
                                              filter_shape=(self.num_units, 1, self.filter_width, self.original_features)))
        F = nonlinearities.sigmoid(T.nnet.conv2d(rs, self.F_W,
                                              input_shape=(None, 1, self.internal_seq_len, self.original_features),
                                              filter_shape=(self.num_units, 1, self.filter_width, self.original_features)))
        
        if self.pooling == 'fo' or self.pooling == 'ifo':
            O = nonlinearities.sigmoid(T.nnet.conv2d(rs, self.O_W,
                                              input_shape=(None, 1, self.internal_seq_len, self.original_features),
                                              filter_shape=(self.num_units, 1, self.filter_width, self.original_features)))
        if self.pooling == 'ifo':
            I = nonlinearities.sigmoid(T.nnet.conv2d(rs, self.I_W,
                                              input_shape=(None, 1, self.internal_seq_len, self.original_features),
                                              filter_shape=(self.num_units, 1, self.filter_width, self.original_features)))
        
        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        Z = Z.flatten(ndim=3)
        Z = Z.dimshuffle([2, 0, 1])
        F = F.flatten(ndim=3)
        F = F.dimshuffle([2, 0, 1])
        if self.pooling == 'fo' or self.pooling == 'ifo':
            O = O.flatten(ndim=3)
            O = O.dimshuffle([2, 0, 1])
        if self.pooling == 'ifo':
            I = I.flatten(ndim=3)
            I = I.dimshuffle([2, 0, 1])
        
        # Dot against a 1s vector to repeat to shape (num_batch, num_units)
        ones = T.ones((num_batch, 1))
        hid_init = T.dot(ones, self.hid_init)
        
        # Create single recurrent computation step function
        # input_n is the n'th vector of the input: (n_batch, n_features)
        def step_f(forget_n, z_n, hid_previous, *args):
            return forget_n * hid_previous + (1.0 - forget_n) * z_n
        def step_fo(forget_n, z_n, o_n, hid_previous, cell_previous, *args):
            cell_current = forget_n * cell_previous + (1.0 - forget_n) * z_n
            hid_current = o_n * cell_current
            return [hid_current, cell_current]
        def step_ifo(forget_n, z_n, o_n, i_n, hid_previous, cell_previous, *args):
            cell_current = forget_n * cell_previous + i_n * z_n
            hid_current = o_n * cell_current
            return [hid_current, cell_current]
        
        if self.pooling == 'f':
            step = step_f
            sequences = [F, Z]
            outputs_info = [hid_init]
        if self.pooling == 'fo':
            step = step_fo
            sequences = [F, Z, O]
            # Note that, below, we use hid_init as the initial /cell/ state!
            # That way we only need to declare one set of weights
            outputs_info = [T.zeros((num_batch, self.num_units)), hid_init]
        if self.pooling == 'ifo':
            step = step_ifo
            sequences = [F, Z, O, I]
            outputs_info = [T.zeros((num_batch, self.num_units)), hid_init]
        
        outputs = theano.scan(
                fn=step,
                sequences=sequences,
                outputs_info=outputs_info,
                strict=True)[0]
        
        hid_out = outputs
        if self.pooling == 'fo' or self.pooling == 'ifo':
            hid_out = outputs[0]
        
        # Shuffle back to (n_batch, n_time_steps, n_features)
        hid_out = hid_out.dimshuffle([1, 0, 2])
        return hid_out

    def get_output_shape_for(self, input_shape):
        # Keep in mind that applying convolutions reduces the sequences length by (filter_width - 1)
        return input_shape[0], input_shape[1], self.num_units