"""Tools for converting model parameter from Caffe to Keras."""

import numpy as np
import os
import sys
import shutil
import h5py
import collections
import pickle


def dump_weights(model_proto, model_weights, weight_output, shape_output=None, caffe_home='~/caffe'):
    """Helper function to dump caffe model weithts in keras tf format
    
    # Arguments
        model_proto: path to the caffe model .prototxt file
        model_weights: path to the caffe model .caffemodel file
        weight_output: path to HDF5 output file
        shape_output: path to pickle output file
        
    # Notes
        caffe requres to run the function in python 2.x
    """
    
    def expand(path):
        return os.path.abspath(os.path.expanduser(path))
    
    caffe_home = expand(caffe_home)
    model_proto = expand(model_proto)
    model_weights = expand(model_weights)
    #print(caffe_home + '\n' + model_proto + '\n' + model_weights + '\n' + weight_output  + '\n' + shape_output )
    
    # import caffe
    sys.path.insert(0, os.path.join(caffe_home, 'python'))
    import caffe
    
    # create model
    caffe.set_mode_cpu()
    net = caffe.Net(model_proto, model_weights, caffe.TEST)
    
    if os.path.exists(weight_output):
        os.remove(weight_output)
    
    f = h5py.File(weight_output, 'w')
    
    # process the layers
    layer_names = list(net._layer_names)
    weights_shape = {}
    for name in net.params:
        layer =  net.layers[layer_names.index(name)]
        blobs = net.params[name]
        blobs_shape = [list(b.shape) for b in blobs]
        weights_shape[name] = blobs_shape
        
        print('%-25s %-20s %-3s %s' % (name, layer.type, len(blobs), blobs_shape))

        params = collections.OrderedDict()
        if layer.type == 'Convolution':
            W = blobs[0].data
            W = W.transpose(2,3,1,0)
            params[name+'_W_1:0'] = W
            if len(blobs) > 1:
                b = blobs[1].data
                params[name+'_b_1:0'] = b
        elif layer.type == 'Normalize':
            gamma = blobs[0].data
            params[name+'_gamma_1:0'] = gamma
        elif layer.type == 'BatchNorm':
            size = blobs[0].shape[0]
            running_mean = blobs[0].data
            running_std = blobs[1].data
            gamma = np.empty(size)
            gamma.fill(blobs[2].data[0])
            beta = np.zeros(size)
            params[name+'_gamma_1:0'] = gamma
            params[name+'_beta_1:0'] = beta
            params[name+'_running_mean_1:0'] = running_mean
            params[name+'_running_std_1:0'] = running_std
        elif layer.type == 'Scale':
            gamma = blobs[0].data
            beta = blobs[1].data
            params[name+'_gamma_1:0'] = gamma
            params[name+'_beta_1:0'] = beta
        elif layer.type == 'InnerProduct':
            W = blobs[0].data
            W = W.T
            b = blobs[1].data
            params[name+'_W_1:0'] = W
            params[name+'_b_1:0'] = b
        else:
            if len(blobs) > 0:
                print('UNRECOGNISED BLOBS')
        
        # create group and add parameters
        g = f.create_group(name)
        for weight_name, value in params.items():
            param_dset = g.create_dataset(weight_name, value.shape, dtype=value.dtype)
            if not value.shape:
                # scalar
                param_dset[()] = value
            else:
                param_dset[:] = value
        g.attrs['weight_names'] = list(params.keys())
    
    f.attrs['layer_names'] = layer_names

    f.flush()
    f.close()
    
    # output model shape
    if shape_output is not None:
        output_shape = {}
        for layer_name, blob in net.blobs.iteritems():
            #print('%-40s %s' %(layer_name, str(blob.data.shape)))
            output_shape[layer_name] = blob.data.shape
            
        shape = {}
        shape['output_shape'] = output_shape
        shape['weights_shape'] = weights_shape
        
        shape_output = expand(shape_output)
        if os.path.exists(shape_output):
            os.remove(shape_output)
        
        with open(shape_output , 'wb') as f:
            pickle.dump(shape, f, protocol=pickle.HIGHEST_PROTOCOL)


def add_missing_layers(model, input_file_name, output_file_name):
    """Helper function to add the missing keras layers in a HDF5 file
    
    # Arguments
        model: keras model
        input_file_name: path to input HDF5 file
        output_file_name: path to output HDF5 file
    """

    shutil.copy(input_file_name, output_file_name)

    f = h5py.File(output_file_name, 'r+')

    # add missing layers
    layer_names_model = [layer.name for layer in model.layers]
    layer_names_new = []
    for name in layer_names_model:
        if not name in f.keys():
            print('add %s' % name)
            g = f.create_group(name)
            g.attrs['weight_names'] = []
        layer_names_new.append(name)

    print('update layer_names')
    f.attrs['layer_names'] = [s.encode('ascii') for s in layer_names_new]

    f.flush()
    f.close()


def compare_output_shape(model, shape_file):
    """Compares the output shape of the layers in caffe and keras model
    
    # Arguments
        model: keras model
        shape_file: path to pickle file dumped by 'dump_weights'
    """
    with open(shape_file, 'rb') as f:
        shape = pickle.load(f)
        #print('%-30s %-20s %-20s' % ('', 'caffe shape', 'keras shape'))
        for layer in model.layers:
            if layer.name in shape['output_shape']:
                shape_caffe = list(shape['output_shape'][layer.name][1:])
                # TODO: depends on layer type
                if len(shape_caffe) == 3:
                    shape_caffe_mod = [shape_caffe[1], shape_caffe[2], shape_caffe[0]]
                else:    
                    shape_caffe_mod = list(shape_caffe)
                shape_keras = list(layer.output_shape[1:])
                mismatch = 'mismatch' if (shape_caffe_mod != shape_keras) else ''
                print('%-30s %-20s %-20s %s' % (layer.name, shape_caffe, shape_keras, mismatch))
                #print('%-30s \n%-20s \n%-20s' % (layer.name, shape_caffe, shape_keras))


def compare_weights_shape(model, shape_file):
    """Compares the parameter shape of the layers in caffe and keras model
    
    # Arguments
        model: keras model
        shape_file: path to pickle file dumped by 'dump_weights'
    """
    with open(shape_file, 'rb') as f:
        shape = pickle.load(f)
        #print('%-30s %-20s %-20s' % ('', 'caffe shape', 'keras shape'))
        for layer in model.layers:
            if layer.name in shape['weights_shape']:
                shape_caffe = shape['weights_shape'][layer.name]
                # TODO: depends on layer type
                shape_caffe_mod = [ [s[2],s[3],s[1],s[0]] if len(s) == 4 else s for s in shape_caffe]
                shape_keras = [w.shape.as_list() for w in layer.weights]
                mismatch = 'mismatch' if not all([shape_caffe_mod[i] == shape_keras[i] for i in range(len(shape_keras))]) else ''
                print('%-30s %-40s %-40s %s' % (layer.name, shape_caffe, shape_keras, mismatch))
                #print('%-30s \n%-40s \n%-40s' % (layer.name, shape_caffe, shape_keras))


if __name__ == '__main__':
    model_proto = './resnet152/ResNet-152-deploy.prototxt'
    model_weights = './resnet152/ResNet-152-model.caffemodel'
    weights_output = 'resnet152_weights.hdf5'
    shape_output = 'resnet152_shape.pkl'

    dump_weights(model_proto, model_weights, weights_output, shape_output=shape_output)
