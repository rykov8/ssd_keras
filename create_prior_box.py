import pickle
import numpy as np


box_configs_ssd300 = [
    {'layer_width': 38, 'layer_height': 38, 'num_prior': 3, 'min_size':  30.0,
     'max_size': None, 'aspect_ratios': [1.0, 2.0, 1/2.0]},
    {'layer_width': 19, 'layer_height': 19, 'num_prior': 6, 'min_size':  60.0,
     'max_size': 114.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 114.0,
     'max_size': 168.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width':  5, 'layer_height':  5, 'num_prior': 6, 'min_size': 168.0,
     'max_size': 222.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width':  3, 'layer_height':  3, 'num_prior': 6, 'min_size': 222.0,
     'max_size': 276.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
    {'layer_width':  1, 'layer_height':  1, 'num_prior': 6, 'min_size': 276.0,
     'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
]

box_configs_ssd512 = [
    {'layer_width': 64, 'layer_height': 64, 'num_prior': 4, 'min_size': 35.84, 
     'max_size': 76.8, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0]},
    {'layer_width': 32, 'layer_height': 32, 'num_prior': 6, 'min_size': 76.8, 
     'max_size': 153.6, 'aspect_ratios':  [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]}, 
    {'layer_width': 16, 'layer_height': 16, 'num_prior': 6, 'min_size': 153.6, 
     'max_size': 230.4, 'aspect_ratios':  [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]}, 
    {'layer_width': 8, 'layer_height': 8, 'num_prior': 6, 'min_size': 230.4, 
     'max_size': 307.2, 'aspect_ratios':  [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]}, 
    {'layer_width': 4, 'layer_height': 4, 'num_prior': 6, 'min_size': 307.2, 
     'max_size': 384.0, 'aspect_ratios':  [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]}, 
    {'layer_width': 2, 'layer_height': 2, 'num_prior': 4, 'min_size': 384.0, 
     'max_size': 460.8, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0]}, 
    {'layer_width': 1, 'layer_height': 1, 'num_prior': 4, 'min_size': 460.8, 
     'max_size': 537.6, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0]}    
]


def create_prior_box(box_configs, img_width, img_height):
    variance = [0.1, 0.1, 0.2, 0.2]
    boxes_paras = []
    
    for layer_config in box_configs:
        layer_width, layer_height = layer_config["layer_width"], layer_config["layer_height"]
        num_priors = layer_config["num_prior"]
        aspect_ratios = layer_config["aspect_ratios"]
        min_size = layer_config["min_size"]
        max_size = layer_config["max_size"]

        step_x = float(img_width) / float(layer_width)
        step_y = float(img_height) / float(layer_height)

        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)

        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        assert(num_priors == len(aspect_ratios))
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))

        box_widths = []
        box_heights = []
        for ar in aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(min_size)
                box_heights.append(min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(min_size * max_size))
                box_heights.append(np.sqrt(min_size * max_size))
            elif ar != 1:
                box_widths.append(min_size * np.sqrt(ar))
                box_heights.append(min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        # Normalize to 0-1
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        # clip to 0-1
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        piror_variances = np.tile(variance, (len(prior_boxes),1))
        boxes_para = np.concatenate((prior_boxes, piror_variances), axis=1)
        boxes_paras.append(boxes_para)

    return np.concatenate(boxes_paras, axis=0)


if __name__ == "__main__":
    
    boxes_paras = create_prior_box(box_configs_ssd300, 300, 300)
    with open('prior_boxes_ssd300.pkl', 'wb') as f:
        pickle.dump(boxes_paras.astype('float32'), f, protocol=2)

    boxes_paras = create_prior_box(box_configs_ssd512, 512, 512)
    with open('prior_boxes_ssd512.pkl', 'wb') as f:
        pickle.dump(boxes_paras.astype('float32'), f, protocol=2)

