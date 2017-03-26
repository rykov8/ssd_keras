import matplotlib.pyplot as plt
import cv2
import numpy as np
from subprocess import check_output


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