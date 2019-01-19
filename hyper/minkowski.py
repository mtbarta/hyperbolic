import tensorflow as tf
from tensorflow.python.framework import function
import numpy as np
from hyper.util import tf_hyp_non_lin

EPS = 1e-15
MAX_TANH_ARG = 15.0

# Why isn't this in numpy?
def acosh(x):
    return np.log(x + np.sqrt(x**2-1))

# Real x, not vector!
def tf_cosh(x):
    return tf.cosh(x)
#    return tf.cosh(tf.minimum(tf.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))

# Real x, not vector!
def tf_sinh(x):
    return tf.sinh(x)
#    return tf.sinh(tf.minimum(tf.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))

@function.Defun(tf.float64, tf.float64)
def norm_grad(x, dy):
    return dy*(x/(tf.norm(x, axis = 1, keepdims=True)+1.0e-8))

@function.Defun(tf.float64, grad_func=norm_grad)
def tf_norm(x):
    res = tf.norm(x, axis = 1, keepdims=True)
    return res

def tf_dot(x, y):
    # return tf.reduce_sum(tf.multiply(x, y), axis=1, keepdims=True)
    return tf.matmul(x, y) + EPS

# Hyperbolic distance
# def dist(u,v):
#     z  = 2 * np.linalg.norm(u-v)**2
#     uu = 1. + z/((1-np.linalg.norm(u)**2)*(1-np.linalg.norm(v)**2))
#     return acosh(uu)


# def tf_mink_dot_matrix(a, b, dim=1):
#     rank = a.get_shape()[dim] - 1

#     # flip_velocity = np.asarray([1 for _ in range(rank)] + [-1])
#     # flip_velocity = tf.constant(flip_velocity, dtype=tf.float64)
#     # flip_velocity = a * flip_velocity + EPS
#     # res = tf_dot(flip_velocity, tf.transpose(b)) + EPS
#     a_euc = a[:, :rank]
#     b_euc = tf.transpose(b[:, :rank])
#     # print('a_euc', a_euc.get_shape())
#     # print('b_euc', b_euc.get_shape())

#     timelike_a = a[:,rank][:, tf.newaxis]
#     timelike_b = tf.transpose(b[:, rank][:, tf.newaxis])

#     euc_dps = tf_dot(a_euc, b_euc)
#     # print('euc_dps', euc_dps.get_shape())
#     timelike = tf_dot(timelike_a, timelike_b)
#     # print('timelike', timelike.get_shape())

#     # res = tf.stack([euc_dps, timelike], axis=-1)
#     # res = tf.reshape(res, [])
#     # print('res', res.get_shape())
#     return tf.subtract(euc_dps, timelike)

def tf_mink_dot_matrix(a, b, dim=1):
    one = tf.reduce_sum(tf.matmul(a[:,:-1], tf.transpose(b[:,:-1])))
    two = tf.multiply(tf.cast(2., tf.float64), tf.matmul(a[:,-1][:, tf.newaxis], tf.transpose(b[:,-1][:, tf.newaxis])))

    return tf.subtract(one, two)

# def tf_hyper_add(u, v):
#     numer = tf.add(u, v)
#     denom = 

def tf_logarithm(base, other):
    """
    Return the logarithm of `other` in the tangent space of `base`.
    """
    mdp = tf_mink_dot_matrix(base, other)
    dist = tf.acosh(-mdp)
    proj = other + (mdp * base)
    norm = tf.sqrt(tf_mink_dot_matrix(proj, proj)) 
    proj *= dist / norm
    return proj

def tf_geodesic_parallel_transport(base, tangent):
    """
    Parallel transport `tangent`, a tangent vector at point `base`, along the
    geodesic in the direction `direction` (another tangent vector at point
    `base`, not necessarily unit length)
    """
    direction = tf_logarithm(base, tangent)
    norm_direction = tf.sqrt(tf_mink_dot_matrix(direction, direction))
    unit_direction = direction / norm_direction
    parallel_component = tf_mink_dot_matrix(tangent, unit_direction)
    unit_direction_transported = tf_sinh(norm_direction) * base + tf_cosh(norm_direction) * unit_direction
    return parallel_component * unit_direction_transported + tangent - parallel_component * unit_direction 


def tf_exp_map_x(x, v, c):
    """https://research.fb.com/wp-content/uploads/2018/07/Learning-Continuous-Hierarchies-in-the-Lorentz-Model-of-Hyperbolic-Geometry.pdf?

    from tangent space to hyperboloid/lorentz
    """
    v = v + EPS # Perturbe v to avoid dealing with v = 0
    norm_v = tf_norm(v)
    second_term = tf_cosh(np.sqrt(c) * norm_v) * x + tf_sinh(np.sqrt(c) * norm_v) * (v / norm_v)
    # second_term = (tf_tanh(np.sqrt(c) * tf_lambda_x(x, c) * norm_v / 2) / (np.sqrt(c) * norm_v)) * v
    return second_term

class LorentzRNN(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 num_units,
                 inputs_geom,
                 bias_geom,
                 c_val,
                 non_lin,
                 fix_biases,
                 fix_matrices,
                 matrices_init_eye,
                 dtype,
                 bias=True,
                 layer=0):
        self._num_units = num_units
        self.c_val = c_val
        self.built = False
        self.__dtype = dtype
        self.non_lin = non_lin
        self.bias = bias
        self.layer=layer
        assert self.non_lin in ['id', 'relu', 'tanh', 'sigmoid']

        self.bias_geom = bias_geom
        self.inputs_geom = inputs_geom
        assert self.inputs_geom in ['eucl', 'hyp']
        assert self.bias_geom in ['eucl', 'hyp']

        self.fix_biases = fix_biases
        self.fix_matrices = fix_matrices
        if matrices_init_eye or self.fix_matrices:
            self.matrix_initializer = tf.initializers.identity()
        else:
            self.matrix_initializer = tf.truncated_normal_initializer(0.0001, .00001)
            # self.matrix_initializer = tf.contrib.layers.xavier_initializer()

        self.eucl_vars = []
        self.hyp_vars = []

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    # Performs the hyperbolic version of the operation Wh + Ux + b.
    def one_rnn_transform(self, W, h, U, x, b):
        W_x_h = tf_mink_dot_matrix(h, W) #becomes (10, 800)
        # W_x_h = tf_geodesic_parallel_transport(h, W)
        
        # W_x_h = tf.Print(W_x_h, [W_x_h], 'W_x_h')
        U_x_h = tf_mink_dot_matrix(x, tf.transpose(U)) + EPS #becomes (?, 50)
        # U_x_h = tf_geodesic_parallel_transport(x, tf.transpose(U))
        # U_x_h = tf.Print(U_x_h, [U_x_h], 'U_x_h')
        result = W_x_h + U_x_h + b
        # result = tf.Print(result, [result], 'result')

        return result


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if not self.built:
                inputs_shape = inputs.get_shape()
                print('Init RNN cell')
                if inputs_shape[1].value is None:
                    raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                     % inputs_shape)
                input_depth = inputs_shape[1].value

                self.W = tf.get_variable(
                    'W'+str(self.layer), dtype= self.__dtype,
                    shape=[self._num_units, self._num_units],
                    trainable=(not self.fix_matrices),
                    initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    # print('appending W matrix to hyp_vars')
                    self.eucl_vars.append(self.W)

                self.U = tf.get_variable(
                    'U'+str(self.layer), dtype= self.__dtype,
                    shape=[input_depth, self._num_units],
                    trainable=(not self.fix_matrices),
                    initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    # print("appending U matrix to hyp_vars")
                    self.eucl_vars.append(self.U)

                self.b = tf.get_variable(
                    'b'+str(self.layer), dtype= self.__dtype,
                    shape=[1, self._num_units],
                    trainable=(not self.fix_biases),
                    initializer=tf.constant_initializer(0.0))

                if not self.fix_biases:
                    if self.bias_geom == 'hyp':
                        self.hyp_vars.append(self.b)
                    else:
                        self.eucl_vars.append(self.b)

                self.built = True

            new_h = self.one_rnn_transform(self.W, state, self.U, inputs, self.b)
            new_h = tf_hyp_non_lin(new_h, non_lin=self.non_lin, hyp_output=True, c=self.c_val)

        return new_h, new_h