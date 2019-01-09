
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import time
import os

from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras.legacy import interfaces


def smooth_l1_loss(y_true, y_pred):
    """Compute L1-smooth loss.

    # Arguments
        y_true: Ground truth bounding boxes,
            tensor of shape (?, num_boxes, 4).
        y_pred: Predicted bounding boxes,
            tensor of shape (?, num_boxes, 4).

    # Returns
        l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).

    # References
        https://arxiv.org/abs/1504.08083
    """
    abs_loss = tf.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred)**2
    loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    return tf.reduce_sum(loss, axis=-1)

def softmax_loss(y_true, y_pred):
    """Compute cross entropy loss aka softmax loss.

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).

    # Returns
        softmax_loss: Softmax loss, tensor of shape (?, num_boxes).
    """
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.-eps)
    loss = - y_true * tf.log(y_pred)
    return tf.reduce_sum(loss, axis=-1)

def cross_entropy_loss(y_true, y_pred):
    """Compute binary cross entropy loss.
    """
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.-eps)
    loss = - y_true*tf.log(y_pred) - (1-y_true)*tf.log(1-y_pred)
    return tf.reduce_sum(loss, axis=-1)

def focal_loss(y_true, y_pred, gamma=2., alpha=1.):
    """Compute binary focal loss.
    
    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
    
    # Returns
        focal_loss: Focal loss, tensor of shape (?, num_boxes).

    # References
        https://arxiv.org/abs/1708.02002
    """
    #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.-eps)
    #loss = - K.pow(1-y_pred, gamma) * y_true*tf.log(y_pred) - K.pow(y_pred, gamma) * (1-y_true)*tf.log(1-y_pred)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = - K.pow(1.-pt, gamma) * K.log(pt)
    loss = alpha * loss
    return tf.reduce_sum(loss, axis=-1)


class LearningRateDecay(Callback):
    def __init__(self, methode='linear', base_lr=1e-3, n_desired=40000, desired=0.1, bias=0.0, minimum=0.1):
        super(LearningRateDecay, self).__init__()
        self.methode = methode
        self.base_lr = base_lr
        self.n_desired = n_desired
        self.desired = desired
        self.bias = bias
        self.minimum = minimum
        
        #TODO: better naming

    def compute_learning_rate(self, n, methode):
        n_desired = self.n_desired
        desired = self.desired
        base_lr = self.base_lr
        bias = self.bias
        
        offset = base_lr * desired * bias
        base_lr = base_lr - offset
        
        desired = desired / (1-desired*bias) * (1-bias)
        
        if methode == 'default':
            k = (1 - desired) / n_desired
            lr = np.maximum( -k * n + 1, 0)
        elif methode == 'linear':
            k = (1 / desired - 1) / n_desired
            lr = 1 / (1 + k * n)
        elif methode == 'quadratic':
            k = (np.sqrt(1/desired)-1) / n_desired
            lr = 1 / (1 + k * n)**2
        elif methode == 'exponential':
            k = -1 * np.log(desired) / n_desired
            lr = np.exp(-k*n)
        
        lr = base_lr * lr + offset
        lr = np.maximum(lr, self.base_lr * self.minimum)
        return lr
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        steps_per_epoch = self.params['steps']
        iteration = self.epoch * steps_per_epoch + batch
        
        lr = self.compute_learning_rate(iteration, self.methode)
        K.set_value(self.model.optimizer.lr, lr)

    def plot_learning_rates(self):
        n = np.linspace(0, self.n_desired*2, 101)
        plt.figure(figsize=[16, 6])
        plt.plot([n[0], n[-1]], [self.base_lr*self.desired*self.bias]*2, 'k')
        for m in ['default', 'linear', 'quadratic', 'exponential']:
            plt.plot(n, self.compute_learning_rate(n, m))
        plt.legend(['bias', '$-kn+1$', '$1/(1+kn)$', '$1/(1+kn)^2$', '$e^{-kn}$'])
        plt.grid()
        plt.xlim(0, n[-1])
        plt.ylim(0, None)
        plt.show()


class ModelSnapshot(Callback):
    """Save the model weights after an interval of iterations."""
    
    def __init__(self, logdir, interval=10000, verbose=1):
        super(ModelSnapshot, self).__init__()
        self.logdir = logdir
        self.interval = interval
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
    
    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        # steps/batches/iterations
        steps_per_epoch = self.params['steps']
        self.iteration = self.epoch * steps_per_epoch + batch + 1
        
    def on_batch_end(self, batch, logs=None):
        if self.iteration % self.interval == 0:
            filepath = os.path.join(self.logdir, 'weights.%06i.h5' % (self.iteration))
            if self.verbose > 0:
                print('\nSaving model %s' % (filepath))
            self.model.save_weights(filepath, overwrite=True)


class Logger(Callback):
    
    def __init__(self, logdir):
        super(Logger, self).__init__()
        self.logdir = logdir
    
    def save_history(self):
        df = pd.DataFrame.from_dict(self.model.history.history)
        df.to_csv(os.path.join(self.logdir, 'history.csv'), index=False)
    
    def append_log(self, logs):
        data = {k:[float(logs[k])] for k in self.model.metrics_names}
        data['iteration'] = [self.iteration]
        data['epoch'] = [self.epoch]
        data['batch'] = [self.batch]
        data['time'] = [time.time() - self.start_time]
        #data['lr'] = [float(K.get_value(self.model.optimizer.lr))]
        df = pd.DataFrame.from_dict(data)
        with open(os.path.join(self.logdir, 'log.csv'), 'a') as f:
            df.to_csv(f, header=f.tell()==0, index=False)
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.save_history()

    def on_batch_begin(self, batch, logs=None):
        self.batch = batch
        # steps/batches/iterations
        steps_per_epoch = self.params['steps']
        self.iteration = self.epoch * steps_per_epoch + batch
        
    def on_batch_end(self, batch, logs=None):
        self.append_log(logs)
    
    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        self.save_history()


def plot_log(log_dir, names=None, limits=None, window_length=250, log_dir_compare=None):
    
    # TODO: differnet batch sizes lead to different epoch length
    
    if limits is None:
        limits = slice(None)
    elif type(limits) in [list, tuple]:
        limits = slice(*limits)
    
    print(log_dir)
    d = pd.read_csv(os.path.join('.', 'checkpoints', log_dir, 'log.csv'))
    d = d[limits]
    iteration = np.array(d['iteration'])
    epoch = np.array(d['epoch'])
    idx = np.argwhere(np.diff(epoch))[:,0] + 1
    
    if log_dir_compare is not None:
        print(log_dir_compare)
        d2 = pd.read_csv(os.path.join('.', 'checkpoints', log_dir_compare, 'log.csv'))
        d2 = d2[limits]
        iteration2 = np.array(d2['iteration'])
        
    if names is None:
        names = set(d.keys())
    else:
        names = set(names)
        names.intersection_update(set(d.keys()))
    if log_dir_compare is not None:
        names.intersection_update(set(d2.keys()))
    names.difference_update({'epoch', 'batch', 'iteration', 'time'})
    print(names)
    
    if 'time' in d.keys() and len(idx) > 1:
        t = np.array(d['time'])
        print('time per epoch %3.1f h' % ((t[idx[1]]-t[idx[0]])/3600))
    
    # reduce epoch ticks
    max_ticks = 20
    n = len(idx)
    if n > 1:
        n = round(n,-1*int(np.floor(np.log10(n))))
        while n >= max_ticks:
            if n/2 < max_ticks:
                n /= 2
            else:
                if n/5 < max_ticks:
                    n /= 5
                else:
                    n /= 10
        idx_step = int(np.ceil(len(idx)/n))
        epoch_step = epoch[idx[idx_step]] - epoch[idx[0]]
        for first_idx in range(len(idx)):
            if epoch[idx[first_idx]] % epoch_step == 0:
                break
        idx_red = [idx[i] for i in range(first_idx, len(idx), idx_step)]
    else:
        idx_red = idx
    
    if window_length is not None:
        #w = np.ones(window_length) # moving average
        w = np.hanning(window_length) # hanning window
        wh = int(window_length/2)
    
    for k in names:
        plt.figure(figsize=(16, 8))
        plt.plot(iteration, d[k], zorder=0)
        plt.title(k, y=1.05)
        
        # filter signal
        if window_length and len(iteration) > window_length:
            x = iteration[wh-1:-wh]
            y = np.convolve(w/w.sum(), d[k], mode='valid')
            plt.plot(x, y)
        
        # second log
        if log_dir_compare is not None:
            plt.plot(iteration2, d2[k], zorder=0)
            
            if window_length and len(iteration2) > window_length:
                x = iteration2[wh-1:-wh]
                y = np.convolve(w/w.sum(), d2[k], mode='valid')
                plt.plot(x, y)
            xmin = min(iteration[0], iteration2[0])
            xmax = max(iteration[-1], iteration2[-1])
        else:
            xmin = iteration[0]
            xmax = iteration[-1]
        
        ax1 = plt.gca()
        ax1.set_xlim(xmin, xmax)
        ax1.yaxis.grid(True)
        #ax1.set_xlabel('iteration')
        #ax1.set_yscale('linear')
        ax1.get_yaxis().get_major_formatter().set_useOffset(False)
        
        ax2 = ax1.twiny()
        ax2.xaxis.grid(True)
        ax2.set_xticks(iteration[idx_red])
        ax2.set_xticklabels(epoch[idx_red])
        ax2.set_xlim(xmin, xmax)
        #ax2.set_xlabel('epoch')
        #ax2.set_yscale('linear')
        ax2.get_yaxis().get_major_formatter().set_useOffset(False)
        
        k_end = k.split('_')[-1]
        if k_end in ['loss']:
            ymin = 0
            ymax = min(np.max(d[k][np.isfinite(d[k])]), np.mean(d[k][np.isfinite(d[k])])*8)
            ax1.set_ylim(ymin, ymax)
        if k_end in ['precision', 'recall', 'fmeasure', 'accuracy']:
            ax1.set_ylim(0, 1)
        
        plt.show()


class AdamAccumulate(Optimizer):
    """Adam optimizer with accumulated gradients for having a virtual batch size larger 
    than the physical batch size.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        accum_iters: Number of batches between parameter update.
        
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
    """
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, accum_iters=10, **kwargs):
        super(AdamAccumulate, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.accum_iters = K.variable(accum_iters)
    
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = ms + vs
        
        flag = K.equal(t % self.accum_iters, 0)
        flag = K.cast(flag, dtype='float32')
        
        for p, g, m, v, gg in zip(params, grads, ms, vs, gs):

            gg_t = (1 - flag) * (gg + g)
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + flag * g) / self.accum_iters
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square((gg + flag * g) / self.accum_iters)
            p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append((m, flag * m_t + (1 - flag) * m))
            self.updates.append((v, flag * v_t + (1 - flag) * v))
            self.updates.append((gg, gg_t))
            
            # apply constraints.
            new_p = p_t
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates
            
    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


