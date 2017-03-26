import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def imshow_fig(img, title='', **kwargs):
    h = img.shape[0]
    w = img.shape[1]
    dpi = 96
    fig = plt.figure(figsize=(w/dpi, h/dpi))
    fig.add_axes([0., 0., 1., 1.])
    fig.canvas.set_window_title(title)
    plt.imshow(img, **kwargs)
    plt.axis('off')
    return fig


def plot_feature_map(activations, title=''):

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
    fig = imshow_fig(act_map, title, cmap='gray')
    fig.show()


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr