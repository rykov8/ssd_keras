"""Some utils for data augmentation."""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os

eps = 1e-10

class BaseGTUtility(object):
    """Base class for handling datasets.
    
    Derived classes should implement the following attributes and call the init methode:
        gt_path         str
        image_path      str
        classes         list of str, first class is normaly 'Background'
        image_names     list of str
        data            list of array (boxes, n * xy + one_hot_class)
    optional attributes are:
        text            list of list of str
    """
    def __init__(self):
        self.gt_path = ''
        self.image_path = ''
        self.classes = []
        self.image_names = []
        self.data = []
        
    def init(self):
        self.num_classes = len(self.classes)
        self.classes_lower = [s.lower() for s in self.classes]
        self.colors = plt.cm.hsv(np.linspace(0, 1, len(self.classes)+1)).tolist()

        # statistics
        stats = np.zeros(self.num_classes)
        num_without_annotation = 0
        for i in range(len(self.data)):
            #stats += np.sum(self.data[i][:,-self.num_classes:], axis=0)
            if len(self.data[i]) == 0:
                num_without_annotation += 1
            else:
                unique, counts = np.unique(self.data[i][:,-1].astype(np.int16), return_counts=True)
                stats[unique] += counts
        self.stats = stats
        self.num_without_annotation = num_without_annotation
        
        self.num_samples = len(self.image_names)
        self.num_images = len(self.data)
        self.num_objects = sum(self.stats)
    
    def __str__(self):
        if not hasattr(self, 'stats'):
            self.init()
        
        s = ''
        for i in range(self.num_classes):
            s += '%-16s %8i\n' % (self.classes[i], self.stats[i])
        s += '\n'
        s += '%-16s %8i\n' % ('images', self.num_images)
        s += '%-16s %8i\n' % ('objects', self.num_objects)
        s += '%-16s %8.2f\n' % ('per image', self.num_objects/self.num_images)
        s += '%-16s %8i\n' % ('no annotation', self.num_without_annotation)
        return s
    
    def plot_gt(self, boxes, show_labels=True):
        # if parameter is sample index
        if type(boxes) in [int]:
            boxes = self.data[boxes]
        
        ax = plt.gca()
        im = plt.gci()
        w, h = im.get_size()
        
        for box in boxes:
            class_idx = int(box[-1])
            color = self.colors[class_idx]
            is_polygon = len(box)-1 > 4
            if is_polygon:
                xy = box[:-1].reshape((-1,2))
            else:
                xmin, ymin, xmax, ymax = box[:4]
                xy = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            xy = xy * [h, w]
            ax.add_patch(plt.Polygon(xy, fill=False, edgecolor=color, linewidth=1))
            if show_labels:
                label_name = self.classes[class_idx]
                if is_polygon:
                    angle = np.arctan((xy[1,0]-xy[0,0])/(xy[1,1]-xy[0,1]+eps))
                    if angle < 0:
                        angle += np.pi
                    angle = angle/np.pi*180-90
                else:
                    angle = 0                
                ax.text(xy[0,0], xy[0,1], label_name, bbox={'facecolor':color, 'alpha':0.5}, rotation=angle)
    
    def plot_input(self, input_img):
        img = np.copy(input_img)
        mean = np.array([104,117,123])
        img += mean[np.newaxis, np.newaxis, :]
        img = img[:, :, (2,1,0)]
        img /= 255
        plt.imshow(img)
    
    def sample(self, idx=None, preserve_aspect_ratio=False, aspect_ratio=1.0):
        '''Draw a random sample form the dataset.
        '''
        if idx is None:
            idx = np.random.randint(0, len(self.image_names))
        file_path = os.path.join(self.image_path, self.image_names[idx])
        img = cv2.imread(file_path)
        if preserve_aspect_ratio:
            img = pad_image(img, aspect_ratio)
        img = img[:, :, (2,1,0)]
        img = img / 255.
        return idx, img, self.data[idx]
    
    def sample_random_batch(self, batch_size=32, input_size=(512,512), seed=1337, preserve_aspect_ratio=False):
        '''Draws a batch of random samples from the dataset.
        
        # Arguments
            batch_size: The batch size.
            input_size: Tuple with height and width of model input.
            seed: Seed for drawing the sample indices.
            preserve_aspect_ratio: Boolean flag for padding the images with 
                random pixels and preserving the aspect ratio of the content.

        # Return
            idxs: List of the sample idices in the dataset.
            inputs: List of preprocessed input images (BGR).
            images: List of normalized images for vizualization (RGB).
            data: List of Ground Truth data, arrays with bounding boxes and class label.
        '''
        h, w = input_size
        aspect_ratio = w/h
        if seed is not None:
            np.random.seed(seed)
        idxs = np.random.randint(0, self.num_samples, batch_size)
        
        inputs = []
        images = []
        data = []
        for i in idxs:
            img_path = os.path.join(self.image_path, self.image_names[i])
            img = cv2.imread(img_path)
            
            if preserve_aspect_ratio:
                img, gt = pad_image(img, aspect_ratio, self.data[i])
            else:
                gt = self.data[i]
            
            inputs.append(preprocess(img, input_size))
            img = cv2.resize(img, (w,h), cv2.INTER_LINEAR).astype('float32') # should we do resizing
            img = img[:, :, (2,1,0)] # BGR to RGB
            img /= 255
            images.append(img)
            data.append(gt)
        inputs = np.asarray(inputs)

        return idxs, inputs, images, data
    
    def sample_batch(self, batch_size, batch_index, input_size=(512,512), preserve_aspect_ratio=False):
        h, w = input_size
        aspect_ratio = w/h
        idxs = np.arange(min(batch_size*batch_index, self.num_samples), 
                         min(batch_size*(batch_index+1), self.num_samples))
        
        if len(idxs) == 0:
            print('WARNING: empty batch')
        
        inputs = []
        data = []
        for i in idxs:
            img_path = os.path.join(self.image_path, self.image_names[i])
            img = cv2.imread(img_path)
            if preserve_aspect_ratio:
                img = pad_image(img, aspect_ratio)
            inputs.append(preprocess(img, input_size))
            data.append(self.data[i])
        inputs = np.asarray(inputs)
        
        return inputs, data
    
    def subset(self, start_idx=0, end_idx=-1):
        gtu = BaseGTUtility()
        gtu.gt_path = self.gt_path
        gtu.image_path = self.image_path
        gtu.classes = self.classes
        
        gtu.image_names = self.image_names[start_idx:end_idx]
        gtu.data = self.data[start_idx:end_idx]
        if hasattr(gtu, 'text'):
            gtu.text = self.text[start_idx:end_idx]
        
        gtu.init()
        
        return gtu
    
    def split(self, split=0.8):
        gtu1 = BaseGTUtility()
        gtu1.gt_path = self.gt_path
        gtu1.image_path = self.image_path
        gtu1.classes = self.classes
        
        gtu2 = BaseGTUtility()
        gtu2.gt_path = self.gt_path
        gtu2.image_path = self.image_path
        gtu2.classes = self.classes
        
        n = int(round(split * len(self.image_names)))
        gtu1.image_names = self.image_names[:n]
        gtu2.image_names = self.image_names[n:]
        gtu1.data = self.data[:n]
        gtu2.data = self.data[n:]
        if hasattr(self, 'text'):
            gtu1.text = self.text[:n]
            gtu2.text = self.text[n:]
        
        gtu1.init()
        gtu2.init()
        return gtu1, gtu2

    def merge(self, gtu2):
        gtu1 = self
        
        if len(set(gtu1.classes)^set(gtu2.classes)) > 0:
            raise Exception('Classes are different')
        gtu = BaseGTUtility()
        gtu.classes = gtu1.classes
        
        # if image_path are the same
        if gtu1.image_path == gtu2.image_path:
            gtu.image_path = gtu1.image_path
            gtu.image_names = gtu1.image_names + gtu2.image_names
        else:
            s1 = gtu1.image_path.split(os.path.sep)
            s2 = gtu2.image_path.split(os.path.sep)
            lmin = min(len(s1),len(s1))
            for i in range(lmin):
                if s1[i] != s2[i]:
                    break
                if i == lmin-1:
                    i = lmin
            prefix1 = os.path.join('', *s1[i:])
            prefix2 = os.path.join('', *s2[i:])
            
            gtu.image_path = os.path.join('', *s1[:i])
            gtu.image_names = \
                    [os.path.join(prefix1, n) for n in gtu1.image_names] + \
                    [os.path.join(prefix2, n) for n in gtu2.image_names]
            
        gtu.data = gtu1.data + gtu2.data
        if hasattr(gtu1, 'text') and hasattr(gtu2, 'text'):
            gtu.text = gtu1.text + gtu2.text
        
        gtu.init()
        return gtu
    
    def convert(self, new_classes, conversion_map=None):
        classes_lower = [s.lower() for s in self.classes]
        new_classes_lower = [s.lower() for s in new_classes]
        
        # do renaming if conversion map is provided
        if conversion_map is not None:
            for i in range(len(conversion_map)):
                m_old = conversion_map[i][0].lower()
                m_new = conversion_map[i][1].lower()
                if m_old in classes_lower:
                    idx_old = classes_lower.index(m_old)
                    classes_lower[idx_old] = m_new
        
        old_to_new = []
        for i in range(len(classes_lower)):
            if classes_lower[i] in new_classes_lower:
                old_to_new.append(new_classes_lower.index(classes_lower[i]))
            else:
                old_to_new.append(None)
        
        gtu = BaseGTUtility()
        gtu.gt_path = self.gt_path
        gtu.image_path = self.image_path
        gtu.classes = new_classes
        
        num_old_classes = len(self.classes)
        num_new_classes = len(new_classes)
        
        gtu.image_names = []
        gtu.data = []
        if hasattr(self, 'text'):
            gtu.text = []
        
        for i in range(len(self.image_names)):
            boxes = []
            for j in range(len(self.data[i])):
                #old_class_idx = np.argmax(self.data[i][j,-num_old_classes:])
                old_class_idx = int(self.data[i][j,-1])
                new_class_idx = old_to_new[old_class_idx]
                if new_class_idx is not None:
                    #class_one_hot = [0] * num_new_classes
                    #class_one_hot[new_class_idx] = 1
                    box = self.data[i][j,:-1]
                    box = list(box) + [new_class_idx]
                    boxes.append(box)
            if len(boxes) > 0:
                boxes = np.asarray(boxes)
                gtu.data.append(boxes)
                gtu.image_names.append(self.image_names[i])
                if hasattr(self, 'text'):
                    gtu.text.append(self.text[i])
        
        gtu.init()
        return gtu


class InputGenerator(object):
    """Model input generator for data augmentation."""
    # TODO
    # flag to protect bounding boxes from cropping?
    # flag for preserving aspect ratio or not
    # padding to preserve aspect ratio? crop_area_range=[0.75, 1.25]
    
    def __init__(self, gt_util, prior_util, batch_size, input_size,
                augmentation=False,
                saturation_var=0.5,
                brightness_var=0.5,
                contrast_var=0.5,
                lighting_std=0.5,
                hflip_prob=0.5,
                vflip_prob=0.0,
                do_crop=True,
                add_noise=True,
                crop_area_range=[0.75, 1.0],
                aspect_ratio_range=[4./3., 3./4.]):
        
        self.__dict__.update(locals())
        
        self.num_batches = gt_util.num_samples // batch_size
        
        self.color_jitter = []
        if saturation_var:
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.color_jitter.append(self.contrast)
    
    def __str__(self):
        f = '%-20s %s\n'
        s = ''
        s += f % ('input_size', self.input_size)
        s += f % ('batch_size', self.batch_size)
        s += f % ('num_samples', self.gt_util.num_samples)
        s += f % ('num_batches', self.num_batches)
        return s
    
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    
    def noise(self, img):
        img_size = img.shape[:2]
        scale = np.random.randint(16)
        noise = np.array(np.random.exponential(scale, img_size), dtype=np.int) * np.random.randint(-1,2, size=img_size)
        #noise = np.array(np.random.normal(0, scale, img_size), dtype=np.int)
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
        img = img + noise
        return np.clip(img, 0, 255)
    
    def horizontal_flip(self, img, y, hflip_prob):
        if np.random.random() < hflip_prob:
            img = img[:, ::-1]
            num_coords = y.shape[1] - 1
            if num_coords == 8: # polygon case
                y[:,[0,2,4,6]] = 1 - y[:,[2,0,6,4]]
                y[:,[1,3,5,7]] = y[:,[3,1,7,5]]
            else:
                y[:,[0,2]] = 1 - y[:,[2,0]]    
        return img, y
    
    def vertical_flip(self, img, y, vflip_prob):
        if np.random.random() < vflip_prob:
            img = img[::-1]
            num_coords = y.shape[1] - 1
            if num_coords == 8: # polynom case
                y[:,[0,2,4,6]] = y[:,[6,4,2,0]]
                y[:,[1,3,5,7]] = 1 - y[:,[7,5,3,1]]
            else:
                y[:,[1,3]] = 1 - y[:,[3,1]]
        return img, y
    
    def random_sized_crop(self, img, target):
        img_h, img_w = img.shape[:2]
        
        # make sure that we can preserve the aspect ratio
        ratio_range = self.aspect_ratio_range
        random_ratio = ratio_range[0] + np.random.random() * (ratio_range[1] - ratio_range[0])
        # a = w/h, w_i-w >= 0, h_i-h >= 0 leads to LP: max. h s.t. h <= w_i/a, h <= h_i
        max_h = min(img_w/random_ratio, img_h)
        max_w = max_h * random_ratio
        
        # scale the area
        crop_range = self.crop_area_range
        random_scale = crop_range[0] + np.random.random() * (crop_range[1] - crop_range[0])
        target_area = random_scale * max_w * max_h
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        x = np.random.random() * (img_w - w)
        y = np.random.random() * (img_h - h)
        
        w_rel = w / img_w
        h_rel = h / img_h
        x_rel = x / img_w
        y_rel = y / img_h
        
        w, h, x, y = int(w), int(h), int(x), int(y)
        
        # crop image and transform boxes
        new_img = img[y:y+h, x:x+w]
        num_coords = target.shape[1] - 1
        new_target = []
        if num_coords == 8: # polynom case
            for box in target:
                new_box = np.copy(box)
                new_box[0:8:2] -= x_rel
                new_box[0:8:2] /= w_rel
                new_box[1:8:2] -= y_rel
                new_box[1:8:2] /= h_rel
                
                if (new_box[0] < 1 and new_box[6] < 1 and new_box[2] > 0 and new_box[4] > 0 and 
                    new_box[1] < 1 and new_box[3] < 1 and new_box[5] > 0 and new_box[7] > 0):
                    new_target.append(new_box)
            new_target = np.asarray(new_target)
        else:
            for box in target:
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                box_w = xmax - xmin
                box_h = ymax - ymin

                # add only boxes if width and height is larger then 10% of original width and height
                a = 0.1
                if (xmax-0 > a*box_w and 1-xmin > a*box_w and
                    ymax-0 > a*box_h and 1-ymin > a*box_h):
                    new_box = np.copy(box)
                    new_box[:4] = np.clip([xmin, ymin, xmax, ymax], 0, 1)
                    new_target.append(new_box)
            new_target = np.asarray(new_target).reshape(-1, target.shape[1])
        return new_img, new_target
    
    def generate(self, debug=False, encode=True):
        h, w = self.input_size
        mean = np.array([104,117,123])
        gt_util = self.gt_util
        batch_size = self.batch_size
        num_batches = self.num_batches
        
        inputs, targets = [], []
        
        while True:
            idxs = np.arange(gt_util.num_samples)
            np.random.shuffle(idxs)
            idxs = idxs[:num_batches*batch_size]
            for j, i in enumerate(idxs):
                img_name = gt_util.image_names[i]
                img_path = os.path.join(gt_util.image_path, img_name)
                img = cv2.imread(img_path)
                y = np.copy(gt_util.data[i])
                
                if debug:
                    raw_img = img.astype(np.float32)
                    raw_y = np.copy(y)
                
                if self.augmentation:
                    if self.do_crop:
                        for _ in range(10): # tries to crop without losing ground truth
                            img_tmp, y_tmp = self.random_sized_crop(img, y)
                            if len(y_tmp) > 0:
                                break
                        if len(y_tmp) > 0:
                            img = img_tmp
                            y = y_tmp
                        
                    img = cv2.resize(img, (w,h), cv2.INTER_LINEAR)
                    img = img.astype(np.float32)
                    
                    random.shuffle(self.color_jitter)
                    for jitter in self.color_jitter: # saturation, brightness, contrast
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y, self.hflip_prob)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y, self.vflip_prob)
                    if self.add_noise:
                        img = self.noise(img)
                else:
                    img = cv2.resize(img, (w,h), cv2.INTER_LINEAR)
                    img = img.astype(np.float32)
                
                if debug:
                    plt.figure(figsize=(12,6))
                    # origal gt image
                    plt.subplot(121)
                    dbg_img = np.copy(raw_img)
                    dbg_img /= 256
                    dbg_img = dbg_img[:,:,(2,1,0)]
                    plt.imshow(dbg_img)
                    gt_util.plot_gt(raw_y)
                    # network input image
                    plt.subplot(122)
                    dbg_img = np.copy(img)
                    dbg_img /= 256
                    dbg_img = dbg_img[:,:,(2,1,0)]
                    plt.imshow(dbg_img)
                    gt_util.plot_gt(y)
                    plt.show()
                    
                img -= mean[np.newaxis, np.newaxis, :]
                #img = img / 25.6
                
                inputs.append(img)
                targets.append(y)
                
                #if len(targets) == batch_size or j == len(idxs)-1: # last batch in epoch can be smaller then batch_size
                if len(targets) == batch_size:
                    if encode:
                        targets = [self.prior_util.encode(y) for y in targets]
                        targets = np.array(targets, dtype=np.float32)
                    tmp_inputs = np.array(inputs, dtype=np.float32)
                    tmp_targets = targets
                    inputs, targets = [], []
                    yield tmp_inputs, tmp_targets
                elif j == len(idxs)-1:
                    # forgett last batch
                    inputs, targets = [], []
                    break
                    
            print('NEW epoch')
        print('EXIT generator')


def pad_image(img, aspect_ratio, gt_data=None):
    """Padds an image with random pixels to get one with specific 
    aspect ratio while avoiding distortion of the image content.
    """
    src_h, src_w, src_c = img.shape
    if src_h * aspect_ratio > src_w:
        new_w = int(src_h * aspect_ratio)
        new_img = np.random.rand(src_h, new_w, src_c) * 255
        padding = int((new_w-src_w)/2)
        new_img[:,padding:padding+src_w,:] = img
        if gt_data is not None:
            new_gt_data = np.copy(gt_data)
            new_gt_data[:,0:-1:2] = new_gt_data[:,0:-1:2] * src_w/new_w + padding/new_w
            return new_img, new_gt_data
        else:
            return new_img
    else:
        new_h = int(src_w / aspect_ratio)
        new_img = np.random.rand(new_h, src_w, src_c) * 255
        padding = int((new_h-src_h)/2)
        new_img[padding:padding+src_h,:,:] = img
        if gt_data is not None:
            new_gt_data = np.copy(gt_data)
            new_gt_data[:,1:-1:2] = new_gt_data[:,1:-1:2] * src_h/new_h + padding/new_h
            return new_img, new_gt_data
        else:
            return new_img


def preprocess(img, size):
    """Precprocess an image for ImageNet models.
    
    # Arguments
        img: Input Image
        size: Target image size (height, width).
    
    # Return
        Resized and mean subtracted BGR image, if input was also BGR.
    """
    h, w = size
    img = np.copy(img)
    img = cv2.resize(img, (w,h), cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    mean = np.array([104,117,123])
    img -= mean[np.newaxis, np.newaxis, :]
    return img




def preprocess_image(file_name, size=(300,300), lib='skimage'):
    """Preprocess a given image for models trained on ImageNet.
        Does the following steps: load, resize, subtract mean

    # Arguments
        file_name: Path to source image file.
        size: Target image size (height, width).
        lib: Name of the library being usesd
            'pil', 'scipy', 'skimage', 'opencv'

    # Returns
        BGR image as numpy array
    """
    if lib == 'pil':
        import PIL
        img = PIL.Image.open(file_name)
        img = img.resize(size, PIL.Image.BILINEAR)
        img = np.array(img, dtype='float32')
        img = img[:,:,(2,1,0)] # RGB to BGR
    elif lib == 'scipy':
        import scipy.misc
        img = scipy.misc.imread(file_name).astype('float32')
        img = scipy.misc.imresize(img, size).astype('float32')
        img = img[:,:,(2,1,0)] # RGB to BGR
    elif lib == 'opencv':
        import cv2
        h, w = size
        img = cv2.imread(file_name)
        img = cv2.resize(img, (w,h), cv2.INTER_LINEAR).astype('float32')
        img = img.astype('float32')
    else:
        # same as in caffe implementation
        import skimage.io
        import skimage.transform
        img = skimage.io.imread(file_name)
        img = img.astype('float32')
        img = img/255.
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)
        img = skimage.transform.resize(img, size, order=1) # interpolation order, default is linear
        img = img * (img_max - img_min) + img_min
        img = img.astype('float32')
        img *= 255 # the reference model operates on images in [0,255] range instead of [0,1]
        img = img[:,:,(2,1,0)] # RGB to BGR
    
    #from IPython.display import display
    #plt.imshow(img[:, :, (2,1,0)]/255.)
    #display(plt.gcf())
    #plt.close()
    
    #mean = np.array([103.939, 116.779, 123.68])
    mean = np.array([104,117,123])
    img -= mean[np.newaxis, np.newaxis, :] # subtract mean
    
    #print((img.shape, img.max(), input_img.min(), img[1,150,150]))
    
    return img

    # %%timeit
    # pil:     3.57 ms ± 17 µs
    # scipy:   5.33 ms ± 15.2 µs
    # opencv:  2.24 ms ± 11.2 µs
    # skimage: 8.81 ms ± 25.3 µs
    # per loop (mean ± std. dev. of 7 runs, 100 loops each)
    
