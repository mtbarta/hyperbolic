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
    return tf.matmul(x, y)

# Hyperbolic distance
# def dist(u,v):
#     z  = 2 * np.linalg.norm(u-v)**2
#     uu = 1. + z/((1-np.linalg.norm(u)**2)*(1-np.linalg.norm(v)**2))
#     return acosh(uu)

def mink_dot_matrix(a, b, dim=1):
    rank = a.shape[dim] - 1
    euc_dps = a[:, :rank].dot(b[:,:rank].T)
    timelike = a[:,rank][:,np.newaxis].dot(b[:,rank][:,np.newaxis].T)
    return euc_dps - timelike

def tf_mink_dot_matrix(a, b, dim=1):
    # print('mink_dot a shape', a.get_shape())
    rank = tf.shape(a)[dim] - 1
    
    a_euc = a[:, :rank]
    b_euc = tf.transpose(b[:, :rank])
    # print('a_euc', a_euc.get_shape())
    # print('b_euc', b_euc.get_shape())

    timelike_a = a[:,rank][:, tf.newaxis]
    timelike_b = tf.transpose(b[:, rank][:, tf.newaxis])

    euc_dps = tf_dot(a_euc, b_euc)
    # print('euc_dps', euc_dps.get_shape())
    timelike = tf_dot(timelike_a, timelike_b)
    # print('timelike', timelike.get_shape())

    return tf.subtract(euc_dps, timelike)

# def tf_hyper_add(u, v):
#     numer = tf.add(u, v)
#     denom = 

    

def tf_exp_map_x(x, v, c):
    """https://research.fb.com/wp-content/uploads/2018/07/Learning-Continuous-Hierarchies-in-the-Lorentz-Model-of-Hyperbolic-Geometry.pdf?

    from tangent space to hyperboloid/lorentz
    """
    v = v + EPS # Perturbe v to avoid dealing with v = 0
    norm_v = tf_norm(v)
    second_term = tf_cosh(np.sqrt(c) * norm_v) * x + tf_sinh(np.sqrt(c) * norm_v) * (v / norm_v)
    # second_term = (tf_tanh(np.sqrt(c) * tf_lambda_x(x, c) * norm_v / 2) / (np.sqrt(c) * norm_v)) * v
    return second_term

# def tf_log_map_x(x, y, c):
#     diff = tf_mob_add(-x, y, c) + EPS
#     norm_diff = tf_norm(diff)
#     lam = tf_lambda_x(x, c)
#     return ( ( (2. / np.sqrt(c)) / lam) * tf_atanh(np.sqrt(c) * norm_diff) / norm_diff) * diff

# def tf_mob_scalar_mul(r, v, c):
#     v = v + EPS
#     norm_v = tf_norm(v)
#     nomin = tf_tanh(r * tf_atanh(np.sqrt(c) * norm_v))
#     result= nomin / (np.sqrt(c) * norm_v) * v
#     return tf_project_hyp_vecs(result, c)

def tf_einstein_addition(u, v, c=1.0):
    # proper velocity space model and proper velocity addition
    # https://en.wikipedia.org/wiki/Gyrovector_space
    print("einstein_addition")
    print("u", u.get_shape())
    print("v", v.get_shape())

    s = tf.add(u, v)
    bta = beta_factor(u, c)
    print('beta factor u', bta.get_shape())
    inner =  bta / (1.0 + bta)
    # inner should be size of beta_u -- (10, 800)
    # inner = tf.divide(beta_factor(u, c), tf.add(tf.constant(1, dtype=tf.float64), beta_factor(u, c)))
    inner2 = tf.matmul(u,v) # should be (10, 10).
    inner3 = tf.divide(tf.subtract(1.0, beta_factor(v, c)), beta_factor(v, c))
    # inner3 should be same size as beta_v -- (10, 800)

    print("inner", inner.get_shape())
    print("inner2", inner2.get_shape())
    total_inner = tf.add(tf_mink_dot_matrix(inner, inner2), tf.transpose(inner3))
    # this becomes a matmul btwn (10, 800) and (10, 10)
    print('total_inner', total_inner.get_shape())
    print("inner2", inner2.get_shape())
    t = tf_mink_dot_matrix(total_inner, u)
    # total inner should be (10, 800)
    print("ein_add_s", s.get_shape())
    print("ein_add_t", t.get_shape())
    return tf.add(s, t)

def beta_factor(vec, c):
    denom = np.sqrt(tf.add(tf.constant(1, dtype=tf.float64), tf.pow(tf_norm(vec), 2)))
    return tf.divide(tf.constant(1, dtype=tf.float64), denom)

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
                 bias=True):
        self._num_units = num_units
        self.c_val = c_val
        self.built = False
        self.__dtype = dtype
        self.non_lin = non_lin
        self.bias = bias
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
            self.matrix_initializer = tf.contrib.layers.xavier_initializer()

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
        # hyp_x = x
        # if self.inputs_geom == 'eucl':
        #     hyp_x = util.tf_exp_map_x(x, self.c_val)

        hyp_b = b
        # if self.bias_geom == 'eucl':
        #     hyp_b = util.tf_exp_map_x(b, self.c_val)

        # W_otimes_h = util.tf_mob_mat_mul(W, h, self.c_val)
        # U_otimes_x = util.tf_mob_mat_mul(U, hyp_x, self.c_val)
        # Wh_plus_Ux = util.tf_mob_add(W_otimes_h, U_otimes_x, self.c_val)
        # result = util.tf_mob_add(Wh_plus_Ux, hyp_b, self.c_val)
        # print('h', h.get_shape())
        # print('W', W.get_shape())
        # print('x', x.get_shape()) # (?, 50)
        # print('U', U.get_shape()) # (50, 800)

        W_x_h = tf_mink_dot_matrix(h, W) #becomes (10, 800)
        U_x_h = tf_mink_dot_matrix(x, tf.transpose(U)) #becomes (?, 50)

        # print('W_x_h', W_x_h.get_shape())
        # print("U_x_h", U_x_h.get_shape())
        result =  W_x_h + U_x_h
        # result = tf_einstein_addition(W_x_h, U_x_h, self.c_val)
        # if self.bias:
        #     # result is probably not commutative, this math doesn't hold
        #     result = result + (hyp_b / tf_)

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
                    'W', dtype= self.__dtype,
                    shape=[self._num_units, self._num_units],
                    trainable=(not self.fix_matrices),
                    initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    print('appending W matrix to hyp_vars')
                    self.hyp_vars.append(self.W)

                self.U = tf.get_variable(
                    'U', dtype= self.__dtype,
                    shape=[input_depth, self._num_units],
                    trainable=(not self.fix_matrices),
                    initializer=self.matrix_initializer)
                if not self.fix_matrices:
                    print("appending U matrix to hyp_vars")
                    self.hyp_vars.append(self.U)

                self.b = tf.get_variable(
                    'b', dtype= self.__dtype,
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