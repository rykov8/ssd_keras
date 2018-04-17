from ssd import SSD300, SSD512
from utils.caffe2keras import add_missing_layers

model = SSD300((300, 300, 3), num_classes=21)
add_missing_layers(model, 'ssd300_voc_weights.hdf5', 'ssd300_voc_weights_fixed.hdf5')

model = SSD512((512, 512, 3), num_classes=21)
add_missing_layers(model, 'ssd512_voc_weights.hdf5', 'ssd512_voc_weights_fixed.hdf5')
