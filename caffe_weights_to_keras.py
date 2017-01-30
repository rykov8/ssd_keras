import caffe
caffe.set_mode_cpu()
model_def = 'SSD/models/deploy.prototxt'
model_weights = 'SSD/models/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

from ssd import SSD300

input_shape=(300, 300, 3)
ssd300 = SSD300(input_shape)

for layer_name in net.params:
    if len(net.params[layer_name]) == 2:
        # transpose is correct for TF format
        W = net.params[layer_name][0].data.transpose(2, 3, 1, 0)
        b = net.params[layer_name][1].data
        ssd300.get_layer(layer_name).set_weights([W, b])
    elif len(net.params[layer_name]) == 1:
        b = net.params[layer_name][0].data
        ssd300.get_layer(layer_name).set_weights([b])
        
ssd300.save_weights('weights_300x300.hdf5')