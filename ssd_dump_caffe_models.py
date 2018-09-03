from utils.caffe2keras import dump_weights

# Note: caffe requires python 2.x and 'ln -s ~/caffe_forks/ssd/data'

caffe_home = '~/caffe_forks/ssd'

model_proto = caffe_home + '/models/VGGNet/VOC0712/SSD_300x300_ft/deploy.prototxt'
model_weights = caffe_home + '/models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel'
weight_output = './models/ssd300_voc_weights.hdf5'
shape_output = './models/ssd300_voc_shape.pkl'
dump_weights(model_proto, model_weights, weight_output, shape_output, caffe_home=caffe_home)

model_proto = caffe_home + '/models/VGGNet/VOC0712/SSD_512x512_ft/deploy.prototxt'
model_weights = caffe_home + '/models/VGGNet/VOC0712/SSD_512x512_ft/VGG_VOC0712_SSD_512x512_ft_iter_120000.caffemodel'
weight_output = './models/ssd512_voc_weights.hdf5'
shape_output = './models/ssd512_voc_shape.pkl'
dump_weights(model_proto, model_weights, weight_output, shape_output, caffe_home=caffe_home)
