from utils.caffe2keras import dump_weights

# Note: caffe requires python 2.x and 'ln -s ~/caffe_forks/ssd/data'

caffe_home = '~/caffe_forks/ssd'

model_proto = caffe_home + '/models/VGGNet/VOC0712Plus/SSD_300x300/deploy.prototxt'
model_weights = caffe_home + '/models/VGGNet/VOC0712Plus/SSD_300x300/VGG_VOC0712Plus_SSD_300x300_iter_240000.caffemodel'
weight_output = 'ssd300_voc_weights.hdf5'
shape_output = 'ssd300_voc_shape.pkl'
dump_weights(model_proto, model_weights, weight_output, shape_output, caffe_home=caffe_home)

model_proto = caffe_home + '/models/VGGNet/VOC0712Plus/SSD_512x512/deploy.prototxt'
model_weights = caffe_home + '/models/VGGNet/VOC0712Plus/SSD_512x512/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel'
weight_output = 'ssd512_voc_weights.hdf5'
shape_output = 'ssd512_voc_shape.pkl'
dump_weights(model_proto, model_weights, weight_output, shape_output, caffe_home=caffe_home)
