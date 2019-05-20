"""Some utils related to Keras models."""

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import h5py
import os


def load_weights(model, filepath, layer_names=None):
    """Loads layer weights from a HDF5 save file.
     
    # Arguments
        model: Keras model
        filepath: Path to HDF5 file
        layer_names: List of strings, names of the layers for which the 
            weights should be loaded. List of tuples 
            (name_in_file, name_in_model), if the names in the file differ 
            from those in model.
    """
    filepath = os.path.expanduser(filepath)
    f = h5py.File(filepath, 'r')
    if layer_names == None:
        layer_names = [s.decode() for s in f.attrs['layer_names']]
    for name in layer_names:
        if type(name) in [tuple, list]:
            layer_name = name[1]
            name = name[0]
        else:
            layer_name = name
        g = f[name]
        weights = [np.array(g[wn]) for wn in g.attrs['weight_names']]
        try:
            layer = model.get_layer(layer_name)
            #assert layer is not None
        except:
            print('layer missing %s' % (layer_name))
            print('    file  %s' % ([w.shape for w in weights]))
            continue
        try:
            #print('load %s' % (layer_name))
            layer.set_weights(weights)
        except Exception as e:
            print('something went wrong %s' % (layer_name))
            print('    model %s' % ([w.shape.as_list() for w in layer.weights]))
            print('    file  %s' % ([w.shape for w in weights]))
            print(e)
    f.close()


def calc_memory_usage(model, batch_size=1):
    """Compute the memory usage of a keras modell.
    
    # Arguments
        model: Keras model.
        batch_size: Batch size used for training.
    
    source: https://stackoverflow.com/a/46216013/445710
    """

    shapes_mem_count = 0
    for l in model.layers:
        shapes_mem_count += np.sum([np.sum([np.prod(s[1:]) for s in n.output_shapes]) for n in l._inbound_nodes])
        
    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    
    # each shape unit occupies 4 bytes in memory
    total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    
    for s in ['Byte', 'KB', 'MB', 'GB', 'TB']:
        if total_memory > 1024:
            total_memory /= 1024
        else:
            break
    print('model memory usage %8.2f %s' % (total_memory, s))


def count_parameters(model):
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    
    print('trainable     {:>16,d}'.format(trainable_count))
    print('non-trainable {:>16,d}'.format(non_trainable_count))


def plot_parameter_statistic(model, layer_types=['Dense', 'Conv2D'], trainable=True, non_trainable=True, outputs=False):
    layer_types = [l.__name__ if type(l) == type else l for l in layer_types]
    
    def get_layers_recursion(model):
        layers = []
        for l in model.layers:
            if l.__class__.__name__ is 'Model':
                child_layers = get_layers_recursion(l)
            else:
                child_layers = [l]
            for cl in child_layers:
                if cl not in layers:
                    layers.append(cl)
        return layers
    
    layers = get_layers_recursion(model)
    
    layers = [l for l in layers if l.__class__.__name__ in layer_types]
    names = [l.name for l in layers]
    y = range(len(names))
    
    plt.figure(figsize=[12,max(len(y)//4,1)])
    
    offset = np.zeros(len(layers), dtype=int)
    legend = []
    if trainable:
        counts_trainable = [np.sum([K.count_params(p) for p in set(l.trainable_weights)]) for l in layers]
        plt.barh(y, counts_trainable, align='center', color='#1f77b4')
        offset += np.array(counts_trainable, dtype=int)
        legend.append('trainable')
    if non_trainable:
        counts_non_trainable = [np.sum([K.count_params(p) for p in set(l.non_trainable_weights)]) for l in layers]
        plt.barh(y, counts_non_trainable, align='center', color='#ff7f0e',  left=offset)
        offset += np.array(counts_non_trainable, dtype=int)
        legend.append('non-trainable')
    if outputs:
        counts_outputs = [np.sum([np.sum([np.prod(s[1:]) for s in n.output_shapes]) for n in l._inbound_nodes]) for l in layers]
        plt.barh(y, counts_outputs, align='center', color='#2ca02c', left=offset)
        offset += np.array(counts_outputs, dtype=int)
        legend.append('outputs')
        
    plt.yticks(y, names)
    plt.ylim(y[0]-1, y[-1]+1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.legend(legend)
    plt.show()


def calc_receptive_field(model, layer_name, verbose=False):
    """Calculate the receptive field related to a certain layer.
    
    # Arguments
        model: Keras model.
        layer_name: Name of the layer.
    
    # Return
        rf: Receptive field (w, h).
        es: Effictive stides in the input image.
        offset: Center of the receptive field associated with the first unit (x, y).
    """
    # TODO...
    
    fstr = '%-20s %-16s %-10s %-10s %-10s %-16s %-10s %-16s'
    if verbose:
        print(fstr % ('name', 'type', 'kernel', 'stride', 'dilation', 'receptive field', 'offset', 'effective stride'))
    l = model.get_layer(layer_name)
    rf = np.ones(2)
    es = np.ones(2)
    offset = np.zeros(2)
    
    while True:
        layer_type = l.__class__.__name__
        k, s, d = (1,1), (1,1), (1,1)
        p = 'same'
        if layer_type in ['Conv2D']:
            k = l.kernel_size
            d = l.dilation_rate
            s = l.strides
            p = l.padding
        elif layer_type in ['MaxPooling2D', 'AveragePooling2D']:
            k = l.pool_size
            s = l.strides
            p = l.padding
        elif layer_type in ['ZeroPadding2D']:
            p = l.padding
        elif layer_type in ['InputLayer', 'Activation', 'BatchNormalization']:
            pass
        else:
            print('unknown layer type %s %s' % (l.name, layer_type))
            
        k = np.array(k)
        s = np.array(s)
        d = np.array(d)
        
        ek = k + (k-1)*(d-1) # effective kernel size
        rf = rf * s + (ek-s)
        es = es * s
        
        if p == 'valid':
            offset += ek/2
            print(ek/2, offset)
        if type(p) == tuple:
            offset -= [p[0][0], p[1][0]]
            print([p[0][0], p[1][0]], offset)
        
        rf = rf.astype(int)
        es = es.astype(int)
        #offset = offset.astype(int)
        if verbose:
            print(fstr % (l.name, l.__class__.__name__, k, s, d, rf, offset, es))
        
        if layer_type == 'InputLayer':
            break
        
        input_name = l.input.name.split('/')[0]
        input_name = input_name.split(':')[0]
        l = model.get_layer(input_name)
    
    return rf, es, offset
