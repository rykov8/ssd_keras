
import numpy as np
import os
import cv2

from sl_utils import polygon_to_rbox


def crop_words(img, boxes, height, width=None, grayscale=True):
    """
    
    # Note
        make sure that the vertices of all boxes are inside the image
    """
    
    #plt.figure(figsize=[12]*2)
    #plt.imshow(img[:, :, (2,1,0)]/255)
    #self.plot_gt(i, show_labels=False)
    #plt.show()

    words = []
    for j in range(len(boxes)):
        h, w = img.shape[:2]
        if boxes.shape[1] == 4:
            # box case
            box = np.round(boxes[j] * [w, h, w, h]).astype(np.int32)
            xmin, ymin, xmax, ymax = box
            word_w, word_h = xmax - xmin, ymax - ymin
            word_ar = word_w / word_h
            word_h = int(height)
            word_w = int(round(height * word_ar))

            word = img[ymin:ymax,xmin:xmax,:]
            word = cv2.resize(word, (word_w, word_h), interpolation=cv2.INTER_CUBIC)
        else:
            # polygon case
            box = np.reshape(boxes[j], (-1,2))
            rbox = polygon_to_rbox(box)
            word_w, word_h = rbox[2]*w, rbox[3]*h
            word_ar = word_w / word_h
            word_h = int(height)
            word_w = int(round(height * word_ar))

            src = np.asarray(box*[w,h], np.float32)
            dst = np.array([
                [0, 0],
                [word_w, 0],
                [word_w, word_h],
                [0, word_h]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)

            word = cv2.warpPerspective(img, M, (word_w, word_h), flags=cv2.INTER_CUBIC)
        
        if grayscale:
            word = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
            word = cv2.normalize(word, word, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            word = word[:,:,None]

        word = word.astype(np.float32)

        if width is not None:
            tmp_word = word[:,:width,:]
            word = np.ones([height, width, tmp_word.shape[2]])
            word[:,slice(0, tmp_word.shape[1]), :] = tmp_word
        
        words.append(word)
    return words


class InputGenerator(object):
    """Model input generator for cropping bounding boxes."""
    def __init__(self, gt_util, batch_size, alphabet, input_size=(255,32),
                grayscale=True, max_string_len=30):
        
        self.__dict__.update(locals())
    
    def generate(self, train=True):
        gt_util = self.gt_util
            
        alphabet = self.alphabet
        batch_size = self.batch_size
        width, height = self.input_size
        max_string_len = self.max_string_len
        
        inputs = []
        targets = []
        
        i = gt_util.num_samples
        while True:
            while len(targets) < batch_size:
                if i == gt_util.num_samples:
                    idxs = np.arange(gt_util.num_samples)
                    np.random.seed(1337)
                    np.random.shuffle(idxs)
                    i = 0
                    print('NEW epoch')
                idx = idxs[i]
                i += 1
                
                self.idx = idx
                
                img_name = gt_util.image_names[idx]
                img_path = os.path.join(gt_util.image_path, img_name)
                img = cv2.imread(img_path)
                #mean = np.array([104,117,123])
                #img -= mean[np.newaxis, np.newaxis, :]
                boxes = np.copy(gt_util.data[idx][:,:-1])
                texts = np.copy(gt_util.text[idx])
                
                # drop boxes with vertices outside the image
                mask = np.array([not (np.any(b < 0.) or np.any(b > 1.)) for b in boxes])
                boxes = boxes[mask]
                texts = texts[mask]
                
                if len(boxes) == 0: continue
                
                try:
                    words = crop_words(img, boxes, height, width, self.grayscale)
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    print(img_path)
                    continue
                
                # drop words with width > height here
                mask = np.array([w.shape[1] > w.shape[0] for w in words])
                words = np.asarray(words)[mask]
                texts = texts[mask]
                
                inputs.extend(words)
                targets.extend(texts)
            
            # shuffle batch
            idxs_batch = np.arange(len(inputs))
            np.random.shuffle(idxs_batch)
            inputs = [inputs[j] for j in idxs_batch]
            targets = [targets[j] for j in idxs_batch]
                
            #yield inputs[:batch_size], targets[:batch_size]
            
            source_str = np.array(targets[:batch_size])
            
            images = np.ones([batch_size, width, height, 1])
            labels = -np.ones([batch_size, max_string_len])
            input_length = np.zeros([batch_size, 1])
            label_length = np.zeros([batch_size, 1])
            for j in range(batch_size):
                images[j] = inputs[j].transpose(1,0,2)
                input_length[j,0] = max_string_len
                label_length[j,0] = len(source_str[j])
                for k, c in enumerate(source_str[j][:max_string_len]):
                    if not c in alphabet or c == '_':
                        #print('bad char', c)
                        labels[j][k] = alphabet.index(' ')
                    else:
                        labels[j][k] = alphabet.index(c)
            
            inputs_dict = {
                'image_input': images,
                'label_input': labels,
                'input_length': input_length, # used by ctc
                'label_length': label_length, # used by ctc
                'source_str': source_str, # used for visualization only
            }
            outputs_dict = {'ctc': np.zeros([batch_size])}  # dummy
            yield inputs_dict, outputs_dict
            
            inputs = inputs[batch_size:]
            targets = targets[batch_size:]
