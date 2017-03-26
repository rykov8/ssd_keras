from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.misc import imread
import tensorflow as tf
from keras import backend as K
import math
import time

from ssd_v2 import SSD300v2
from ssd_utils import BBoxUtility

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
K.set_session(sess)

np.set_printoptions(suppress=True)

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1

network_size = 300
input_shape=(network_size, network_size, 3)
model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

inputs = []
images = []


def get_image_from_path(img_path):
    img = image.load_img(img_path, target_size=(network_size, network_size))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())

for idx in range(1292, 1293):
    get_image_from_path('./GTAV/GD' + str(idx) + '.png')

inputs = preprocess_input(np.array(inputs))
t1 = time.time()
preds = model.predict(inputs, batch_size=1, verbose=1)
t2 = time.time()
print('elapse time {:f}   fsp {:f}'.format(t2-t1, 1/(t2-t1)))
results = bbox_util.detection_out(preds)

a = model.predict(inputs, batch_size=1)
b = bbox_util.detection_out(preds)

norm = mpl.colors.Normalize(vmin=0., vmax=5.)


def plot_activations(activations, plot_enable=True):

    num_channel = activations.shape[2]
    act_border = activations.shape[0]
    map_border_num = int(math.ceil(math.sqrt(num_channel)))
    map_border = act_border * map_border_num
    print('create act map {:d} x {:d}'.format(map_border, map_border))
    act_map = np.zeros((map_border, map_border))

    print(activations.shape)
    all_sum = 0
    for i_x in range(map_border_num):
        for i_y in range(map_border_num):
            idx = i_x * map_border_num + i_y
            if idx >= num_channel:
                break
            act = activations[:, :, idx]
            act_map[i_x*act_border:(i_x+1)*act_border, i_y*act_border:(i_y+1)*act_border] = act
            act_sum = sum(sum(act))
            all_sum += act_sum
            # print('filter-{:d}  act_sum={:f}'.format(idx, act_sum))

    print('all_sum = {:f}'.format(all_sum))
    fig_act = plt.figure()
    plt.imshow(act_map, cmap='gray')
    fig_act.show()


immediate_layer = K.function([model.input, K.learning_phase()],
                             [model.get_layer(name='pool5').output])

for i, img in enumerate(images):

    # plot activations
    layer_output = immediate_layer([inputs, 1])[0][i]
    plot_activations(layer_output)

    fig_img = plt.figure()

    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img, aspect='auto')
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

    fig_img.show()

plt.show()

