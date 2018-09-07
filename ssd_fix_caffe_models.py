from keras import backend as K

from ssd_model import SSD300, SSD512
from utils.caffe2keras import add_missing_layers


model = SSD300((300, 300, 3), num_classes=21)
add_missing_layers(model, './models/ssd300_voc_weights.hdf5', './models/ssd300_voc_weights_fixed.hdf5')

K.clear_session()

model = SSD512((512, 512, 3), num_classes=21)
add_missing_layers(model, './models/ssd512_voc_weights.hdf5', './models/ssd512_voc_weights_fixed.hdf5')

K.clear_session()

model = SSD300((300, 300, 3), num_classes=81)
add_missing_layers(model, './models/ssd300_coco_weights.hdf5', './models/ssd300_coco_weights_fixed.hdf5')

K.clear_session()

model = SSD512((512, 512, 3), num_classes=81)
add_missing_layers(model, './models/ssd512_coco_weights.hdf5', './models/ssd512_coco_weights_fixed.hdf5')
