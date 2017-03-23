# coding: utf-8

from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.misc import imread
import tensorflow as tf
from keras import backend as K
import math
import time
import pickle

from ssd_v2 import SSD300v2
from ssd_conv4_3 import SSD300_conv4_3
from ssd_utils import BBoxUtility
from plot_util import plot_activations

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']

NUM_CLASSES = len(voc_classes) + 1
network_size = 1024
batch_size = 2
input_shape = (network_size, network_size, 3)

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()


def get_detections(result):
    detections = map(lambda r: {'label': r[0],
                                'conf': r[1],
                                'xmin': r[2],
                                'ymin': r[3],
                                'xmax': r[4],
                                'ymax': r[5]},
                     result)
    return detections


def get_layer_output(model, inputs, output_layer_name):
    immediate_layer = K.function([model.input, K.learning_phase()],
                                 [model.get_layer(name=output_layer_name).output])
    output = immediate_layer([inputs, 1])[0]
    return output


def get_layer_predict(model, input_layer_name, input_layer_feature):
    immediate_layer = K.function([model.get_layer(name=input_layer_name), K.learning_phase()],
                                 [model.output])
    model_predict = immediate_layer([input_layer_feature, 1])[0]
    return model_predict


def plot_detections(image_list, detection_result):
    # for each image
    for i, img in enumerate(image_list):
        detections = get_detections(detection_result[i])
        detections = list(filter(lambda x: x['conf'] > 0.6, detections))
        fig_img = plt.figure()
        plt.imshow(img, aspect='auto')
        current_axis = fig_img.gca()

        for det in detections:
            xmin = int(round(det['xmin'] * img.shape[1]))
            ymin = int(round(det['ymin'] * img.shape[0]))
            xmax = int(round(det['xmax'] * img.shape[1]))
            ymax = int(round(det['ymax'] * img.shape[0]))
            conf = det['conf']
            label = int(det['label'])
            label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(conf, label_name)
            # print(display_txt)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            color = colors[label]
            current_axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            current_axis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})
        fig_img.show()


def load_inputs():
    inputs = []
    images = []
    begin_index = 1000
    for idx in range(begin_index, begin_index + batch_size):
        image_file_name = './GTAV/GD' + str(idx) + '.png'
        img = image.load_img(image_file_name, target_size=(network_size, network_size))
        inputs.append(image.img_to_array(img))
        images.append(imread(image_file_name))
    return inputs, images


def run_network(model, inputs):
    time_begin = time.time()
    predictions = model.predict(inputs, batch_size=batch_size, verbose=1)
    time_elapsed = time.time() - time_begin
    print('elapsed time {:0.4f} sec  {:.4f} fps'.format(time_elapsed, batch_size / time_elapsed))
    return predictions


def compare_models(model1, input1, layer1, model2, input2, layer2, plot_activation_enable=False):
    layer_output1 = get_layer_output(model=model1, inputs=input1, output_layer_name=layer1)
    layer_output2 = get_layer_output(model=model2, inputs=input2, output_layer_name=layer2)
    diff = (layer_output1 - layer_output2)

    print('layer_output1 sum =', sum(layer_output1[0].ravel()))
    print('layer_output2 sum =', sum(layer_output2[0].ravel()))
    print('diff min={:f} max={:f} sum={:f}'.format(
        min(np.absolute(diff).ravel()),
        max(np.absolute(diff).ravel()),
        sum(np.absolute(diff).ravel())))
    eq = np.array_equal(layer_output1, layer_output2)
    if eq:
        print('equal')
    else:
        print('not equal')

    if plot_activation_enable:
        plot_activations(layer_output1[0])
        plot_activations(layer_output2[0])


use_feature_flow = True
use_dump_file = False
plot_activation_enable = False


def feature_flow():
    bbox_util = BBoxUtility(NUM_CLASSES)
    raw_inputs, images = load_inputs()
    inputs = preprocess_input(np.array(raw_inputs))

    dump_activation_layer = 'conv4_2'
    compare_layer_name = 'predictions'
    print('dump_activation_layer', dump_activation_layer)
    print('target_layer_name', compare_layer_name)

    # normal SSD network
    model1 = SSD300v2(input_shape, num_classes=NUM_CLASSES)
    model1.load_weights('weights_SSD300.hdf5', by_name=True)
    predictions = run_network(model1, inputs)
    results = bbox_util.detection_out(predictions)
    plot_detections(images, results)

    # get dump layer's output (as input for flow network)
    layer_dump = get_layer_output(model=model1, inputs=inputs, output_layer_name=dump_activation_layer)
    print('layer_dump.shape = ', layer_dump.shape)
    if use_dump_file:
        with open(dump_activation_layer + '.pickle', 'wb') as handle:
            # print('dump_layer_output sum', sum(layer_dump[0].ravel()))
            pickle.dump(layer_dump, handle)

    # flow net
    model2 = SSD300_conv4_3((128, 128, 512), num_classes=NUM_CLASSES)
    model2.load_weights('weights_SSD300.hdf5', by_name=True)
    if use_dump_file:
        filter_activation_input = pickle.load(open(dump_activation_layer + '.pickle', 'rb'))
    else:
        filter_activation_input = layer_dump
    predictions = run_network(model2, filter_activation_input)
    results = bbox_util.detection_out(predictions)
    plot_detections(images, results)

    # get specific layer's output (for debugging)
    compare_models(model1, inputs, compare_layer_name,
                   model2, layer_dump, compare_layer_name,
                   False)

    plt.show()


if __name__ == '__main__':
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)
    K.set_session(sess)
    feature_flow()
