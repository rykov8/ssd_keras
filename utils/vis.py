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


def to_rec(box, image_size):
    """Finds minimum rectangle around some points and scales it to desired 
    image size.
    
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


def plot_box(box, box_format='xywh', color='r', linewidth=1, normalized=False, vertices=False):
    if box_format == 'xywh': # opencv
        xmin, ymin, w, h = box
        xmax, ymax = xmin + w, ymin + h
    elif box_format == 'xyxy':
        xmin, ymin, xmax, ymax = box
    if box_format == 'polygon':
        xy_rec = np.reshape(box, (-1, 2))
    else:
        xy_rec = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    if normalized:
        im = plt.gci()
        xy_rec = xy_rec * np.tile(im.get_size(), (4,1))
    ax = plt.gca()
    ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=linewidth))
    if vertices:
        c = 'rgby'
        for i in range(4):
            plt.plot(xy_rec[i,0],xy_rec[i,1], c[i], marker='o', markersize=4)


def escape_latex(s):
    new_s = []
    for c in s:
        if c in '#$%&_{}':
            new_s.extend('\\'+c)
        elif c == '\\':
            new_s.extend('\\textbackslash{}')
        elif c == '^':
            new_s.extend('\\textasciicircum{}')
        elif c == '~':
            new_s.extend('\\textasciitilde{}')
        else:
            new_s.append(c)
    return ''.join(new_s)
    
    # from pgf.py
    # TeX defines a set of special characters, such as:
    # # $ % & ~ _ ^ \ { }
    # Generally, these characters must be escaped correctly. For convenience,
    # some characters (_,^,%) are automatically escaped outside of math
    # environments.
