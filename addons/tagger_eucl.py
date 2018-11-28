import math
import json
from baseline.model import Tagger, create_tagger_model, load_tagger_model
import os
import tensorflow as tf
import numpy as np
import pickle
import time
import random
from random import shuffle
import math
import argparse
from datetime import datetime
from baseline.tf.tfy import *
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers import fully_connected, xavier_initializer
from baseline.model import Tagger, create_tagger_model, load_tagger_model
from hyper import util
from hyper import rnn_impl
from hyper import crf
now = datetime.now()

class HyperbolicRNNModel(Tagger):
    def __init__(self):
        super().__init__()
        # self.word_to_id = word_to_id
        # self.id_to_word = id_to_word

        # self.construct_placeholders()
        # self.construct_execution_graph()
        self.burn_in_factor = 1.0

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        state = {"mxlen": self.mxlen, "maxw": self.maxw, "crf": self.crf, "proj": self.proj, "crf_mask": self.crf_mask, 'span_type': self.span_type}
        with open(basename + '.state', 'w') as f:
            json.dump(state, f)

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)
            
        if len(self.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(self.word_vocab, f)

        # with open(basename + '-char.vocab', 'w') as f:
        #     json.dump(self.char_vocab, f)

    def make_input(self, batch_dict, do_dropout=False):
        x = batch_dict['x']
        y = batch_dict.get('y', None)
        xch = batch_dict['xch']
        lengths = batch_dict['lengths']

        pkeep = 1.0-self.pdrop_value if do_dropout else 1.0

        if do_dropout and self.pdropin_value > 0.0:
            UNK = self.word_vocab['<UNK>']
            PAD = self.word_vocab['<PAD>']
            drop_indices = np.where((np.random.random(x.shape) < self.pdropin_value) & (x != PAD))
            x[drop_indices[0], drop_indices[1]] = UNK
        feed_dict = {self.x: x, self.xch: xch, self.lengths: lengths, self.pkeep: pkeep}
        if y is not None:
            feed_dict[self.y] = y
        return feed_dict

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    def predict_text(self, text):
        summary, loss, argmax_idx = \
          sess.run([self.summary_merged, self.loss, self.argmax_idx], feed_dict={
              self.word_ids_1: batch_word_ids_1,
              self.num_words_1: batch_num_words_1,
              self.word_ids_2: batch_word_ids_2,
              self.num_words_2: batch_num_words_2,
              self.label_placeholder: batch_label,
              self.dropout_placeholder: 1.0
          })
        self.test_summary_writer.add_summary(summary, summary_i)

    @staticmethod
    def load(basename, **kwargs):
        basename = unzip_model(basename)
        model = RNNTaggerModel()
        model.sess = kwargs.get('sess', tf.Session())
        checkpoint_name = kwargs.get('checkpoint_name', basename)
        checkpoint_name = checkpoint_name or basename
        with open(basename + '.state') as f:
            state = json.load(f)
            model.mxlen = state.get('mxlen', 100)
            model.maxw = state.get('maxw', 100)
            model.crf = bool(state.get('crf', False))
            model.crf_mask = bool(state.get('crf_mask', False))
            model.span_type = state.get('span_type')
            model.proj = bool(state.get('proj', False))

        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            model.sess.graph.as_default()
            tf.import_graph_def(gd, name='')

            model.sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name})
            model.x = tf.get_default_graph().get_tensor_by_name('x:0')
            model.xch = tf.get_default_graph().get_tensor_by_name('xch:0')
            model.y = tf.get_default_graph().get_tensor_by_name('y:0')
            model.lengths = tf.get_default_graph().get_tensor_by_name('lengths:0')
            model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            model.loss = tf.get_default_graph().get_tensor_by_name('Loss/loss:0')
            # model.all_optimizer_var_updates_op = tf.get_default_graph().get_tensor_by_name("Loss_1/loss:0")
            model.best = tf.get_default_graph().get_tensor_by_name('output/ArgMax:0')
            model.probs = tf.get_default_graph().get_tensor_by_name('output/Reshape_1:0')  # TODO: rename
            try:
                model.A = tf.get_default_graph().get_tensor_by_name('Loss/transitions:0')
                #print('Found transition matrix in graph, setting crf=True')
                if not model.crf:
                    print('Warning: meta-data says no CRF but model contains transition matrix!')
                    model.crf = True
            except:
                if model.crf is True:
                    print('Warning: meta-data says there is a CRF but not transition matrix found!')
                model.A = None
                model.crf = False

        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        model.word_vocab = {}
        if os.path.exists(basename + '-word.vocab'):
            with open(basename + '-word.vocab', 'r') as f:
                model.word_vocab = json.load(f)

        with open(basename + '-char.vocab', 'r') as f:
            model.char_vocab = json.load(f)

        model.saver = tf.train.Saver(saver_def=saver_def)
        return model

    def save_using(self, saver):
        self.saver = saver

    def _compute_word_level_loss(self, mask):

        nc = len(self.labels)
        # Cross entropy loss
        # cross_entropy = tf.one_hot(self.y, nc, axis=-1) * tf.log(tf.nn.softmax(self.probs))
        # # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.probs, labels=tf.one_hot(self.y, nc, axis=-1))
        # cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        # cross_entropy *= mask
        # cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        # all_loss = tf.reduce_mean(cross_entropy, name="loss")
        # return all_loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.probs, labels=self.y)
        losses = tf.boolean_mask(losses, mask)
        return tf.reduce_mean(losses)

    def _compute_sentence_level_loss(self):

        if self.crf_mask:
            assert self.span_type is not None, "To mask transitions you need to provide a tagging span_type, choices are `IOB`, `BIO` (or `IOB2`), and `IOBES`"
            A = tf.get_variable(
                "transitions_raw",
                shape=(len(self.labels), len(self.labels)),
                dtype=tf.float64,
                trainable=True
            )

            self.mask = crf_mask(self.labels, self.span_type, self.labels['<GO>'], self.labels['<EOS>'], self.labels.get('<PAD>'))
            self.inv_mask = tf.cast(tf.equal(self.mask, 0), tf.float64) * tf.constant(-1e4, dtype=tf.float64)

            self.A = tf.add(tf.multiply(A, self.mask), self.inv_mask, name="transitions")
            ll, self.A = crf.crf_log_likelihood(self.probs, self.y, self.lengths, self.A)
        else:
            ll, self.A = crf.crf_log_likelihood(self.probs, self.y, self.lengths)
        return tf.reduce_mean(-ll)

    def create_loss(self):

        with tf.variable_scope("Loss"):
            gold = tf.cast(self.y, tf.float64)
            mask = tf.sign(gold)

            if self.crf is True:
                print('crf=True, creating SLL')
                all_loss = self._compute_sentence_level_loss()
            else:
                print('crf=False, creating WLL')
                all_loss = self._compute_word_level_loss(mask)

        return all_loss

    def get_vocab(self, vocab_type='word'):
        return self.word_vocab if vocab_type == 'word' else self.char_vocab

    def get_labels(self):
        return self.labels

    def predict(self, batch_dict):

        feed_dict = self.make_input(batch_dict)
        lengths = batch_dict['lengths']
        # We can probably conditionally add the loss here
        preds = []
        if self.crf is True:

            probv, tranv = self.sess.run([self.probs, self.A], feed_dict=feed_dict)
            batch_sz, _, label_sz = probv.shape
            start = np.full((batch_sz, 1, label_sz), -1e4)
            start[:, 0, self.labels['<GO>']] = 0
            probv = np.concatenate([start, probv], 1)

            for pij, sl in zip(probv, lengths):
                unary = pij[:sl + 1]
                viterbi, _ = tf.contrib.crf.viterbi_decode(unary, tranv)
                viterbi = viterbi[1:]
                preds.append(viterbi)
        else:
            # Get batch (B, T)
            bestv = self.sess.run(self.best, feed_dict=feed_dict)
            # Each sentence, probv
            for pij, sl in zip(bestv, lengths):
                unary = pij[:sl]
                preds.append(unary)

        return preds

    @staticmethod
    def create(labels, embeddings, **kwargs):

        word_vec = embeddings['word']
        char_vec = embeddings['char']
        model = HyperbolicRNNModel()
        model.sess = kwargs.get('sess', tf.Session())

        model.mxlen = kwargs.get('maxs', 100)
        model.maxw = kwargs.get('maxw', 100)

        hsz = int(kwargs['hsz'])
        pdrop = kwargs.get('dropout', 0.5)
        pdrop_in = kwargs.get('dropin', 0.0)
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = kwargs.get('layers', 1)
        model.labels = labels
        model.crf = bool(kwargs.get('crf', False))
        model.crf_mask = bool(kwargs.get('crf_mask', False))
        model.span_type = kwargs.get('span_type')
        model.proj = bool(kwargs.get('proj', False))
        model.feed_input = bool(kwargs.get('feed_input', False))
        model.activation_type = kwargs.get('activation', 'tanh')

        char_dsz = char_vec.dsz
        nc = len(labels)
        model.x = kwargs.get('x', tf.placeholder(tf.int32, [None, model.mxlen], name="x"))
        model.xch = kwargs.get('xch', tf.placeholder(tf.int32, [None, model.mxlen, model.maxw], name="xch"))
        model.y = kwargs.get('y', tf.placeholder(tf.int64, [None, model.mxlen], name="y"))
        model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
        model.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float64, name="pkeep"))
        model.pdrop_value = pdrop
        model.pdropin_value = pdrop_in
        model.word_vocab = {}

        inputs_geom = kwargs.get("inputs_geom", "hyp")
        bias_geom = kwargs.get("bias_geom", "hyp")
        ffnn_geom = kwargs.get("ffnn_geom", "hyp")
        sent_geom = kwargs.get("sent_geom", "hyp")
        mlr_geom = kwargs.get("mlr_geom", "hyp")
        c_val = kwargs.get("c_val", 1.0)
        cell_non_lin = kwargs.get("cell_non_lin", "id") #"id/relu/tanh/sigmoid."
        ffnn_non_lin = kwargs.get("ffnn_non_lin", "id")
        cell_type = kwargs.get("cell_type", 'rnn')
        lr_words = kwargs.get("lr_words", 0.01)
        lr_ffnn = kwargs.get("lr_ffnn", 0.01)
        optimizer = kwargs.get("optimizer", "rsgd")
        eucl_clip = kwargs.get("eucl_clip", 1.0)
        hyp_clip = kwargs.get("hyp_clip", 1.0)
        before_mlr_dim = kwargs.get("before_mlr_dim", nc)
        batch_sz = 10

        print("C_val:", c_val)

        eucl_vars = []
        hyp_vars = []

        if word_vec is not None:
            model.word_vocab = word_vec.vocab

        # model.char_vocab = char_vec.vocab
        seed = np.random.randint(10e8)
        if word_vec is not None:
            # word_embeddings = embed(model.x, len(word_vec.vocab), word_vec.dsz,
            #                         initializer=tf.constant_initializer(word_vec.weights, dtype=tf.float32, verify_shape=True))
            with tf.variable_scope("LUT"):
                W = tf.get_variable("W",
                                    dtype=tf.float64,
                                    initializer=tf.constant_initializer(word_vec.weights, dtype=tf.float64, verify_shape=True),
                                    shape=[len(word_vec.vocab), word_vec.dsz], trainable=True)
                # e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_vec.dsz]))
                # with tf.control_dependencies([W]):
                word_embeddings = tf.nn.embedding_lookup(W, model.x)

        # Wch = tf.Variable(tf.constant(char_vec.weights, dtype=tf.float32), name="Wch")
        # ce0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))

        # word_char, _ = pool_chars(model.xch, Wch, ce0, char_dsz, **kwargs)
        # joint = word_char if word_vec is None else tf.concat(values=[word_embeddings, word_char], axis=2)   
        # word_embeddings = tf.Print(word_embeddings, [word_embeddings], message="embeddings")

        embedseq = word_embeddings

        # embedseq = tf.nn.dropout(word_embeddings, model.pkeep)
        # if (mlr_geom == 'hyp'):
        #     embedseq = util.tf_exp_map_zero(embedseq, c_val)
        
        if cell_type == 'rnn' and sent_geom == 'eucl':
            cell_class = lambda h_dim: tf.contrib.rnn.BasicRNNCell(h_dim)
        if cell_type == 'lstm' and sent_geom == 'eucl':
            cell_class = lambda h_dim: tf.contrib.rnn.BasicLSTMCell(h_dim)
        if cell_type == 'rnn' and sent_geom == 'hyp':
            cell_class = lambda h_dim: rnn_impl.HypRNN(num_units=h_dim,
                                                       inputs_geom=inputs_geom,
                                                       bias_geom=bias_geom,
                                                       c_val=c_val,
                                                       non_lin=cell_non_lin,
                                                       fix_biases=False,
                                                       fix_matrices=False,
                                                       matrices_init_eye=False,
                                                       dtype=tf.float64)
        elif cell_type == 'gru' and sent_geom == 'hyp':
            cell_class = lambda h_dim: rnn_impl.HypGRU(num_units=h_dim,
                                                       inputs_geom=inputs_geom,
                                                       bias_geom=bias_geom,
                                                       c_val=c_val,
                                                       non_lin=cell_non_lin,
                                                       fix_biases=False,
                                                       fix_matrices=False,
                                                       matrices_init_eye=False,
                                                       dtype=tf.float64)
        
        if rnntype == 'rnn':
            cell = cell_class(hsz)
            initial_state = cell.zero_state(batch_sz, tf.float64)
            
            eucl_vars += cell.eucl_vars
            if sent_geom == 'hyp':
                
                hyp_vars += cell.hyp_vars

            # rnnout = tf.contrib.rnn.DropoutWrapper(cell)
            rnnout, state = tf.nn.dynamic_rnn(cell,
                                              embedseq, \
                                              sequence_length=model.lengths,
                                              initial_state=initial_state,
                                              dtype=tf.float64)
        elif rnntype == 'bi':
            cell_1 = cell_class(hsz)
            cell_2 = cell_class(hsz)

            init_fw = cell_1.zero_state(batch_sz, tf.float64)
            init_bw = cell_2.zero_state(batch_sz, tf.float64)

            eucl_vars += cell_1.eucl_vars + cell_2.eucl_vars
            if sent_geom == 'hyp':
                hyp_vars += cell_1.hyp_vars + cell_2.hyp_vars

            rnnout, state = tf.nn.bidirectional_dynamic_rnn(cell_1, 
                                                            cell_2, 
                                                            embedseq,
                                                            initial_state_fw=init_fw,
                                                            initial_state_bw=init_bw,
                                                            sequence_length=model.lengths,
                                                            dtype=tf.float64)
            rnnout = tf.concat(axis=2, values=rnnout)
        else:
            cell = cell_class(hsz)

            eucl_vars += cell.eucl_vars
            if sent_geom == 'hyp':
                hyp_vars += cell.hyp_vars

                
            # rnnout = tf.contrib.rnn.DropoutWrapper(cell)
            rnnout, state = tf.nn.dynamic_rnn(cell, embedseq, sequence_length=model.lengths, dtype=tf.float64)
        # rnnout = tf.Print(rnnout, [rnnout], message="rnnout")

        tf.summary.histogram('RNN/rnnout', rnnout)

        # # Converts seq to tensor, back to (B,T,W)
        hout = rnnout.get_shape()[-1]
        print(rnnout.get_shape())
        # # Flatten from [B x T x H] - > [BT x H]
        rnnout_bt_x_h = tf.reshape(rnnout, [-1, hout])


        ################## first feed forward layer ###################

        # Define variables for the first feed-forward layer: W1 * s1 + W2 * s2 + b + bd * d(s1,s2)
        W_ff_s1 = tf.get_variable('W_ff_s1',
                                  dtype=tf.float64,
                                  shape=[hout, before_mlr_dim],  # 400, 20 -- 20 number of classes
                                  initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float64))

        tf.summary.histogram("W_ff_s1", W_ff_s1)

        b_ff = tf.get_variable('b_ff',
                               dtype=tf.float64,
                               shape=[1, before_mlr_dim],
                               initializer=tf.constant_initializer(0.0))

        eucl_vars += [W_ff_s1]

        if ffnn_geom == 'eucl':
            eucl_vars += [b_ff]
        else:
            hyp_vars += [b_ff]

        if ffnn_geom == 'eucl':
            output_ffnn = tf.matmul(rnnout_bt_x_h, W_ff_s1) + b_ff
            output_ffnn = util.tf_eucl_non_lin(output_ffnn, non_lin=ffnn_non_lin)
        else:
            ffnn_s1 = util.tf_mob_mat_mul(W_ff_s1, rnnout_bt_x_h, c_val)
            tf.summary.histogram("ffnn_s1", ffnn_s1)
            output_ffnn = util.tf_mob_add(ffnn_s1, b_ff, c_val)
            output_ffnn = util.tf_hyp_non_lin(output_ffnn,
                                              non_lin=ffnn_non_lin,
                                              hyp_output = True, #(mlr_geom == 'hyp'),
                                              c=c_val)
            
        
        tf.summary.histogram("output_ffnn", output_ffnn)
        # output_ffnn = tf.Print(output_ffnn, [output_ffnn], message="output_ffnn")

        # Mobius dropout
        # if dropout < 1.0:
        #     # If we are here, then output_ffnn should be Euclidean.
        #     output_ffnn = tf.nn.dropout(output_ffnn, keep_prob=model.pkeep)
        #     if (mlr_geom == 'hyp'):
        #         output_ffnn = util.tf_exp_map_zero(output_ffnn, c_val)
        
        # ################## MLR ###################
        # # output_ffnn is batch_size x before_mlr_dim

        A_mlr = []
        P_mlr = []
        logits_list = []
        dtype=tf.float64

        print('output shape', output_ffnn.get_shape())

        for cl in range(nc):
            with tf.variable_scope('mlp'):
                A_mlr.append(tf.get_variable('A_mlr' + str(cl),
                                            dtype=dtype,
                                            shape=[1, before_mlr_dim],
                                            initializer=tf.contrib.layers.xavier_initializer()))
                eucl_vars += [A_mlr[cl]]

                P_mlr.append(tf.get_variable('P_mlr' + str(cl),
                                            dtype=dtype,
                                            shape=[1, before_mlr_dim],
                                            initializer=tf.constant_initializer(0.0)))

                if mlr_geom == 'eucl':
                    eucl_vars += [P_mlr[cl]]
                    logits_list.append(tf.reshape(util.tf_dot(-P_mlr[cl] + output_ffnn, A_mlr[cl]), [-1]))

                elif mlr_geom == 'hyp':
                    hyp_vars += [P_mlr[cl]]
                    minus_p_plus_x = util.tf_mob_add(-P_mlr[cl], output_ffnn, c_val)
                    norm_a = util.tf_norm(A_mlr[cl])
                    lambda_px = util.tf_lambda_x(minus_p_plus_x, c_val)
                    # blow-- P+X == [10, 20] tensor. A_mlr is also [10,20]. px_dot_a is [10, 1]
                    px_dot_a = util.tf_dot(minus_p_plus_x, tf.nn.l2_normalize(A_mlr[cl]))
                    logit = 2. / np.sqrt(c_val) * norm_a * tf.asinh(np.sqrt(c_val) * px_dot_a * lambda_px)
                    # logit = tf.reshape(logit, [-1])

                    logits_list.append(logit)

        probs = tf.stack(logits_list, axis=1)
        print("probs shape", probs.get_shape())
        model.probs = tf.reshape(probs, [-1, model.mxlen, nc])
        print("reshaped probs", model.probs.get_shape())
        tf.summary.histogram("probs", model.probs)

        model.best = tf.argmax(model.probs, 2)


        model.loss = model.create_loss()

        # model.best = tf.argmax(model.probs, axis=1, output_type=tf.int32)
        #     ######################################## OPTIMIZATION ######################################
        all_updates_ops = []
        model.step = tf.train.get_or_create_global_step()

    #     ###### Update Euclidean parameters using Adam.
        optimizer_euclidean_params = tf.train.AdamOptimizer(learning_rate=1e-3)
        eucl_grads = optimizer_euclidean_params.compute_gradients(model.loss, eucl_vars)
        capped_eucl_gvs = [(tf.clip_by_norm(grad, eucl_clip), var) for grad, var in eucl_grads]  ###### Clip gradients
        all_updates_ops.append(optimizer_euclidean_params.apply_gradients(capped_eucl_gvs))


        ###### Update Hyperbolic parameters, i.e. word embeddings and some biases in our case.
        def rsgd(v, riemannian_g, learning_rate):
            if optimizer == 'rsgd':
                return util.tf_exp_map_x(v, -model.burn_in_factor * learning_rate * riemannian_g, c=c_val)
            else:
                # Use approximate RSGD based on a simple retraction.
                updated_v = v - model.burn_in_factor * learning_rate * riemannian_g
                # Projection op after SGD update. Need to make sure embeddings are inside the unit ball.
                return util.tf_project_hyp_vecs(updated_v, c_val)


        if inputs_geom == 'hyp':
            grads_and_indices_hyp_words = tf.gradients(model.loss, W)
            grads_hyp_words = grads_and_indices_hyp_words[0].values
            # grads_hyp_words = tf.Print(grads_hyp_words, [grads_hyp_words], message="grads_hyp_words")
            
            repeating_indices = grads_and_indices_hyp_words[0].indices


            unique_indices, idx_in_repeating_indices = tf.unique(repeating_indices)
            # unique_indices = tf.Print(unique_indices, [unique_indices], message="unique_indices")
            # idx_in_repeating_indices = tf.Print(idx_in_repeating_indices, [idx_in_repeating_indices], message="idx_in_repeating_indices")
            
            agg_gradients = tf.unsorted_segment_sum(grads_hyp_words,
                                                    idx_in_repeating_indices,
                                                    tf.shape(unique_indices)[0])

            agg_gradients = tf.clip_by_norm(agg_gradients, hyp_clip) ######## Clip gradients
            # agg_gradients = tf.Print(agg_gradients, [agg_gradients], message="agg_gradients")

            unique_word_emb = tf.nn.embedding_lookup(W, unique_indices)  # no repetitions here
            # unique_word_emb = tf.Print(unique_word_emb, [unique_word_emb], message="unique_word_emb")

            riemannian_rescaling_factor = util.riemannian_gradient_c(unique_word_emb, c=c_val)
            # riemannian_rescaling_factor = tf.Print(riemannian_rescaling_factor, [riemannian_rescaling_factor], message="rescl factor")
            rescaled_gradient = riemannian_rescaling_factor * agg_gradients
            # rescaled_gradient = tf.Print(rescaled_gradient, [rescaled_gradient], message="rescl gradient")
            all_updates_ops.append(tf.scatter_update(W,
                                                    unique_indices,
                                                    rsgd(unique_word_emb, rescaled_gradient, lr_words))) # Updated rarely

        if len(hyp_vars) > 0:
            hyp_grads = tf.gradients(model.loss, hyp_vars)
            capped_hyp_grads = [tf.clip_by_norm(grad, hyp_clip) for grad in hyp_grads]  ###### Clip gradients


            for i in range(len(hyp_vars)):
                riemannian_rescaling_factor = util.riemannian_gradient_c(hyp_vars[i], c=c_val)
                rescaled_gradient = riemannian_rescaling_factor * capped_hyp_grads[i]
                all_updates_ops.append(tf.assign(hyp_vars[i], rsgd(hyp_vars[i], rescaled_gradient, lr_ffnn)))  # Updated frequently

        model.all_optimizer_var_updates_op = tf.group(*all_updates_ops)
        print("all ops: ", model.all_optimizer_var_updates_op)

        model.summary_merged = tf.summary.merge_all()

        model.test_summary_writer = tf.summary.FileWriter('./runs/hyper/' + str(os.getpid()))


        return model

def create_model(labels, embeddings, **kwargs):
    return HyperbolicRNNModel.create(labels, embeddings, **kwargs)


def load_model(modelname, **kwargs):
    return HyperbolicRNNModel.load(modelname, **kwargs)
