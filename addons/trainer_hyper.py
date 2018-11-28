from baseline.tf.tagger.train import TaggerTrainerTf, TaggerEvaluatorTf
import tensorflow as tf
import numpy as np
from baseline.utils import to_spans, f_score, listify, revlut, get_model_file
from baseline.progress import create_progress_bar
from baseline.reporting import basic_reporting
from baseline.tf.tfy import optimizer
from baseline.train import EpochReportingTrainer, create_trainer
import os
from baseline.utils import zip_model
import time
import numpy as np
from baseline.utils import create_user_trainer, export

class HyperbolicTrainer(EpochReportingTrainer):
    def __init__(self, model, **kwargs):
        super(HyperbolicTrainer, self).__init__()
        self.model = model
        # self.loss = self.model.create_loss()
        span_type = kwargs.get('span_type', 'iob')
        verbose = kwargs.get('verbose', False)
        self.evaluator = TaggerEvaluatorTf(model, span_type, verbose)
        # self.loss = model.create_loss()
        # self.global_step = tf.train.create_global_step()
        # self.global_step, self.train_op = optimizer(self.loss, **kwargs)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-tagger-%d/lm" % os.getpid())

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-tagger-%d" % os.getpid())
        print("Reloading " + latest)
        self.model.saver.restore(self.model.sess, latest)

    def train(self, ts, reporting_fns, debug=False):
        start_time = time.time()
        metrics = self._train(ts, debug)
        duration = time.time() - start_time
        print('Training time (%.3f sec)' % duration)
        self.train_epochs += 1

        for reporting in reporting_fns:
            reporting(metrics, self.train_epochs * len(ts), 'Train')
        return metrics

    def test(self, vs, reporting_fns, phase='Valid', **kwargs):
        start_time = time.time()
        metrics = self._test(vs, **kwargs)
        duration = time.time() - start_time
        print('%s time (%.3f sec)' % (phase, duration))
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        for reporting in reporting_fns:
            msg = reporting(metrics, epochs, phase)
        return metrics

    def _train(self, ts, debug=False):
        total_loss = 0
        steps = len(ts)

        metrics = {}
        pg = create_progress_bar(steps)
        for batch_dict in ts:
            feed_dict = self.model.make_input(batch_dict, do_dropout=True)
            preds, lossv, _, summs = self.model.sess.run([self.model.probs, self.model.loss, self.model.all_optimizer_var_updates_op, self.model.summary_merged], feed_dict=feed_dict)
            # print(preds)
            total_loss += lossv
            self.model.test_summary_writer.add_summary(summs)
            pg.update()
        pg.done()
        metrics['avg_loss'] = float(total_loss)/steps
        return metrics

    def _test(self, ts):
        return self.evaluator.test(ts)


def create_trainer(model, **kwargs):
    return HyperbolicTrainer(model, **kwargs)