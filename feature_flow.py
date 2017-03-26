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
from subprocess import check_output
import time
import pickle
import cv2
from plot_util import *

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
        detections = list(filter(lambda x: x['conf'] > 0.8, detections))
        fig_img = plt.figure()
        plt.imshow(img)

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


def load_inputs(file_list):
    inputs = []
    images = []
    for file in file_list:
        img = image.load_img(file, target_size=(network_size, network_size))
        inputs.append(image.img_to_array(img))
        images.append(imread(file))
    return inputs, images


def run_network(model, inputs):
    time_begin = time.time()
    predictions = model.predict(inputs, batch_size=batch_size, verbose=1)
    time_elapsed = time.time() - time_begin
    print('elapsed time {:0.4f} sec  {:.4f} fps'.format(time_elapsed, batch_size / time_elapsed))
    return predictions


def compare_model_layer(model1, input1, layer1, model2, input2, layer2, plot_activation_enable=False):
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


def shift_filter(feature, flow):
    # feature shape = (None, 128, 128, 512)
    shifted_feature = list()
    for feat in feature:
        print(feat.shape)
        for i in range(feat.shape[-1]):
            act2d = feat[..., i]
            act2d = act2d[:, :, np.newaxis]
            res = warp_flow(act2d, flow)
            shifted_feature.append(res)

            if False:
                print('act2d', act2d.shape, sum(act2d.ravel()))
                print('flow', flow.shape, sum(flow.ravel()))
                plt.figure(11)
                plt.imshow(act2d[:, :, 0], cmap='gray')
                plt.figure(12)
                plt.imshow(flow[..., 0], cmap='gray')
                plt.figure(13)
                plt.imshow(flow[..., 1], cmap='gray')
                plt.figure(14)
                plt.imshow(res, cmap='gray')
                plt.show()
                pass

    return np.asarray([shifted_feature]).swapaxes(1, 2).swapaxes(2, 3)


def compute_flow(image_path1, image_path2):
    flow_cmd = './run_flow.sh ' + image_path1 + ' ' + image_path2
    check_output([flow_cmd], shell=True)
    flow = np.load('./flow.npy')
    flow = flow.transpose(1, 2, 0)
    # flow.shape should be (height, width, 2)
    return flow


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_map = flow.copy()
    flow_map[:, :, 0] += np.arange(w)
    flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)
    return res


use_feature_flow = True
use_dump_file = False
plot_activation_enable = False
image_files = ['/home/cory/cedl/vid/videos/vid04/0270.jpg', '/home/cory/cedl/vid/videos/vid04/0299.jpg']

# image_files = ['/home/cory/ssd_keras/GTAV/GD1015.png', '/home/cory/ssd_keras/GTAV/GD1020.png']
# '/home/cory/ssd_keras/GTAV/GD21.png'
# '/home/cory/cedl/vid/videos/vid04/1000.jpg'


def feature_flow():
    bbox_util = BBoxUtility(NUM_CLASSES)
    raw_inputs, images = load_inputs(image_files)
    inputs = preprocess_input(np.array(raw_inputs))

    dump_activation_layer = 'conv4_2'
    compare_layer_name = 'conv6_2'
    print('dump_activation_layer', dump_activation_layer)
    print('target_layer_name', compare_layer_name)

    # normal SSD network
    model1 = SSD300v2(input_shape, num_classes=NUM_CLASSES)
    model1.load_weights('weights_SSD300.hdf5', by_name=True)
    predictions = run_network(model1, inputs)
    results = bbox_util.detection_out(predictions)
    plot_detections(images, results)

    # get dump layer's output (as input for flow network)
    input_img2 = inputs[1:2, :, :, :]
    layer_dump = get_layer_output(model=model1, inputs=input_img2, output_layer_name=dump_activation_layer)
    print('layer_dump.shape = ', layer_dump.shape)

    # flow (raw rgb)
    flow_rgb = compute_flow(image_files[1], image_files[0])

    print('flow.shape', flow_rgb.shape)
    fig = plt.figure()
    fig.canvas.set_window_title('flow_rgb')
    plt.imshow(cv2.cvtColor(draw_hsv(flow_rgb), cv2.COLOR_BGR2RGB))

    # flow (re-sized for feature map)
    flow_feature = get_flow_for_filter(flow_rgb)

    fig = plt.figure()
    fig.canvas.set_window_title('flow_feature_y')
    plt.imshow(flow_feature[:, :, 0], cmap='gray')
    fig = plt.figure()
    fig.canvas.set_window_title('flow_feature_x')
    plt.imshow(flow_feature[:, :, 1], cmap='gray')

    # warp image by flow_rgb
    iimg1 = cv2.imread(image_files[0])
    img_warp = warp_flow(iimg1, flow_rgb)

    fig = plt.figure()
    fig.canvas.set_window_title('img_warp')
    plt.imshow(cv2.cvtColor(img_warp, cv2.COLOR_BGR2RGB))

    # shift feature
    shifted_feature = shift_filter(layer_dump, flow_feature)

    # flow net
    model2 = SSD300_conv4_3((128, 128, 512), num_classes=NUM_CLASSES)
    model2.load_weights('weights_SSD300.hdf5', by_name=True)
    predictions = run_network(model2, shifted_feature)
    results = bbox_util.detection_out(predictions)
    plot_detections(images[1:2], results)

    # get specific layer's output and compare them (for debugging)
    compare_model_layer(model1, input_img2, compare_layer_name,
                        model2, shifted_feature, compare_layer_name,
                        True)

    sess.close()
    cv2.waitKey(0)
    plt.show()


def get_flow_for_filter(flow):
    filter_map_width = 128
    flow_ratio_y = flow.shape[0] / filter_map_width
    flow_ratio_x = flow.shape[1] / filter_map_width
    flow_small = np.asarray([cv2.resize(flow[:, :, 0] / flow_ratio_y, (filter_map_width, filter_map_width)),
                             cv2.resize(flow[:, :, 1] / flow_ratio_x, (filter_map_width, filter_map_width))])
    flow_small = flow_small.transpose([1, 2, 0])
    print('flow_small.shape', flow_small.shape)
    return flow_small


if __name__ == '__main__':
    config = tf.ConfigProto(
        device_count={'GPU': 1}
    )
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    K.set_session(sess)
    feature_flow()
