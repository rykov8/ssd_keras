"""Tools for vizualisation of convolutional neural network filter in keras models."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools


def reduce_ticks(ticks, max_ticks=25):
    n = len(ticks)
    n = round(n,-1*floor(log10(n)))
    n_tmp = n
    while n >= max_ticks:
        if n/2 < max_ticks:
            n = n/2
        else:
            if n/5 < max_ticks:
                n = n/5
            else:
                n = n/10
    return [ticks[i] for i in range(0,len(ticks),floor(n_tmp/n))]


def mosaic(images, grid_size, border=1):
    """Creates a mosaic of images.
    
    # Arguments
        images: Array of shape (number_of_images, image_height, image_width)
        grid_size: (number_of_rows, number_of_columns)
        border: Border width in pixels if type is int
                Total border width in percent if type is str
    
    # Return
        Image data
    """
    nrows, ncols = grid_size
    nimgs = images.shape[0]
    if nimgs > nrows * ncols:
        nimgs = nrows * ncols
    height, width = image_size = images.shape[1:]
    if type(border) == str and border[-1] == '%':
        border_percent = float(border[:-1])
        border = int(max(border_percent*width/(100 - border_percent),1))
        
    data_size = (nrows*height+(nrows+1)*border, ncols*width+(ncols+1)*border) 
    data = np.ones(data_size, dtype=np.float32) * images.max()
    for i in range(nimgs):
        irow, icol = i // nrows, i % nrows
        x = border*(irow+1) + height*irow
        y = border*(icol+1) + width*icol
        data[x:x+height, y:y+width] = images[i]
    return data


def plot_activation(model, input_image, layer_name):
    """Plots a mosaic of feature activation.
    
    # Arguments
        model: Keras model
        input_image: Test image which is feed into the network
        layer_name: Layer name of feature map
    """
    from keras import backend as K
    from IPython.display import display
    
    f = K.function(model.inputs, [model.get_layer(layer_name).output])
    output = f([[input_image]])
    output = np.moveaxis(output[0][0], [0,1,2], [1,2,0])
    print('%-20s input_shape: %-16s output_shape: %-16s' % (layer_name, str(input_image.shape), str(output.shape)))
    
    num_y = num_x = int(np.ceil(np.sqrt(output.shape[0])))
    data = mosaic(output, (num_x, num_y), '5%')
    
    #plt.figure(figsize=(12, 12))
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=0.1, pad=0.05)
    im = ax.imshow(data, vmin=data.min(), vmax=data.max(), interpolation='nearest', cmap=cm.binary)
    plt.colorbar(im, cax=cax)
    
    display(plt.gcf())
    plt.close()


def plot_priorboxes(prior_boxes, image, number_of_boxes_per_location, location_indices=None):
    """Visualizes the output of a PriorBox layer.
    
    # Arguments
        ...
    """
    
    colors='rgbcmy'*3
    
    nboxes = len(prior_boxes)
    nratios = number_of_boxes_per_location
    nlocs = nboxes/nratios

    if location_indices == None:
        n = 4
        location_indices = np.unique(np.linspace(0,nlocs-1, int((n**2-1+nlocs)**0.5/n), dtype=int))

    plt.imshow(image)
    fig = plt.gcf()
    fig.set_figheight(12)
    fig.set_figwidth(12)
    ax = plt.gca()
    
    x_all, y_all = [], []
    img_height, img_width = image.shape[:2]
    for j, box in enumerate(prior_boxes):
        xmin, ymin, xmax, ymax = box[:4] * np.array([img_width, img_height, img_width, img_height])
        x_all.append((xmin+xmax)/2.)
        y_all.append((ymin+ymax)/2.)
        
        if not j // nratios in location_indices:
            continue
        
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=colors[j%nratios], linewidth=2))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.plot(x_all, y_all, 'r.')
    plt.show()


def to_rec(box, image_size):
    """Find minimum rectangle around some points and scale it to desired image
    size.
    
    # Arguments
        box: Box or points [x1, y1, x2, y2, ...] with values between 0 and 1.
        image_size: Size of output image.
    # Return
        xy_rec: Corner coordinates of rectangle, array of shape (4, 2).
    """
    img_height, img_width = image_size
    xmin = np.min(box[0::2]) * img_width
    xmax = np.max(box[0::2]) * img_width
    ymin = np.min(box[1::2]) * img_height
    ymax = np.max(box[1::2]) * img_height
    xy_rec = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return xy_rec


def plot_gt(image, gt_data, classes=None, show_labels=True):
    if classes is not None:
        colors = plt.cm.hsv(np.linspace(0, 1, len(classes)+1)).tolist()
    img_size = image.shape[:2]
    
    plt.imshow(image / 255.)
    ax = plt.gca()
    
    for d in gt_data:
        box = d[:4]
        class_one_hot = d[4:]
        label = np.nonzero(class_one_hot)[0][0]+1
        color = 'r' if classes == None else colors[label]
        xy_rec = to_rec(box, img_size)
        ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=2))
        
        if show_labels:
            label_name = label if classes == None else classes[label-1]
            xmin, ymin = xy_rec[0]
            ax.text(xmin, ymin, label_name, bbox={'facecolor':color, 'alpha':0.5})
    
    plt.show()
    return


def plot_results(image, results, classes=None, confidence_threshold=0.6, show_labels=True, gt_data=None):
    top_indices = [i for i, conf in enumerate(results[:, 1]) if conf >= confidence_threshold]
    if classes is not None:
        colors = plt.cm.hsv(np.linspace(0, 1, len(classes)+1)).tolist()
    img_size = image.shape[:2]
    
    plt.imshow(image / 255.)
    ax = plt.gca()
    
    # draw ground truth
    if gt_data is not None:
        for box in gt_data:
            label = np.nonzero(box[4:])[0][0]+1
            color = 'g' if classes == None else colors[label]
            xy_rec = to_rec(box[:4], img_size)
            ax.add_patch(plt.Polygon(xy_rec, fill=True, color=color, linewidth=1, alpha=0.3))
    
    # draw prediction
    for r in results[top_indices]:
        label = int(r[0])
        confidence = r[1]
        color = 'r' if classes == None else colors[label]
        xy_rec = to_rec(r[2:], img_size)
        ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=2))
        
        if show_labels:
            label_name = label if classes == None else classes[label-1]
            xmin, ymin = xy_rec[0]
            display_txt = '%0.2f, %s' % (confidence, label_name)        
            ax.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

    plt.show()
    return #results[top_indices]



def plot_box(box, box_format='xywh', color='r', linewidth=1):
    if box_format == 'xywh': # opencv
        xmin, ymin, w, h = box
        xmax, ymax = xmin + w, ymin + h
    elif box_format == 'xyxy':
        xmin, ymin, xmax, ymax = box
    elif box_format == 'percent':
        im = plt.gci()
        img_h, img_w = im.get_size()
        xmin, ymin, xmax, ymax = box * [img_h, img_w, img_h, img_w]
    if box_format == 'polygon':
        xy_rec = np.reshape(box, (-1, 2))
    else:
        xy_rec = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    ax = plt.gca()
    ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=linewidth))

