
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import json
import time

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
    l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    return tf.reduce_sum(l1_loss, -1)

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
    y_pred = K.clip(y_pred, eps, 1. - eps)
    softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
    return softmax_loss

def focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    """Compute focal loss.
    
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
    y_pred = K.clip(y_pred, eps, 1. - eps)
    
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_loss = -tf.reduce_sum(alpha * K.pow(1. - pt, gamma) * K.log(pt), axis=-1)
    return focal_loss


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
            filepath = self.logdir + '/weights.%06i.h5' % (self.iteration)
            if self.verbose > 0:
                print('\nSaving model %s' % (filepath))
            self.model.save_weights(filepath, overwrite=True)


class Logger(Callback):
    
    def __init__(self, logdir):
        super(Logger, self).__init__()
        self.logdir = logdir
    
    def save_history(self):
        with open(self.logdir+'/history.json','w') as f:
            json.dump(self.model.history.history, f)
        f.close()
    
    def on_train_begin(self, logs=None):
        self.json_log = open(self.logdir+'/log.json', mode='wt', buffering=1)
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
        data = {k:float(logs[k]) for k in self.model.metrics_names}
        data['iteration'] = self.iteration
        data['epoch'] = self.epoch
        data['batch'] = self.batch
        data['time'] = time.time() - self.start_time
        data['lr'] = float(K.get_value(self.model.optimizer.lr))
        self.json_log.write(json.dumps(data) + '\n')
    
    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        self.json_log.close()
        self.save_history()


def plot_log(log_file, names=None, limits=None, window_length=250, log_file_compare=None):
    
    # TODO: print name of both files, length, compare history
    
    def load_log(log_file):
        with open(log_file,'r') as f:
            data = f.readlines()
        keys = json.loads(data[0]).keys()
        d = {k:[] for k in keys}
        for i, line in enumerate(data):
            if not limits == None and (i < limits[0] or i > limits[1]):
                continue
            dat = json.loads(line)
            for k in keys:
                d[k].append(dat[k])
        d = {k:np.array(d[k]) for k in keys}
        return d
    
    d = load_log(log_file)
    print(log_file)
    
    if log_file_compare is not None:
        d2 = load_log(log_file_compare)
        print(log_file_compare)
    
    if names is None:
        names = [k for k in d.keys() if k not in ['epoch', 'batch', 'iteration']]
    else:
        names = [k for k in names if k in d.keys()]
    print(names)

    iteration = d['iteration']
    epoch = d['epoch']
    idx = []
    for i in range(1,len(epoch)):
        if epoch[i] != epoch[i-1]:
            idx.append(i)
    
    if 'time' in d.keys() and len(idx) > 1:
        print('time per epoch %3.1f h' % ((d['time'][idx[1]]-d['time'][idx[0]])/3600))
    
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
        if k in ['epoch', 'batch', 'iteration', 'time']:
            continue
        plt.figure(figsize=(16, 8))
        plt.plot(iteration, d[k], zorder=0)
        plt.title(k, y=1.05)
        
        # filter signal
        if window_length and len(iteration) > window_length:
            x = iteration[wh-1:-wh]
            y = np.convolve(w/w.sum(), d[k], mode='valid')
            plt.plot(x, y)
        
        # second log
        if log_file_compare is not None and k in d2.keys():
            plt.plot(d2['iteration'], d2[k], zorder=0)
            
            if window_length and len(d2['iteration']) > window_length:
                x = d2['iteration'][wh-1:-wh]
                y = np.convolve(w/w.sum(), d2[k], mode='valid')
                plt.plot(x, y)
            xmin = min(d['iteration'][0], d2['iteration'][0])
            xmax = max(d['iteration'][-1], d2['iteration'][-1])
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


