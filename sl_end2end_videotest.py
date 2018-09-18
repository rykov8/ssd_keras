""" A class for testing a SegLink model on a video file or webcam """

import numpy as np
import cv2
import tensorflow as tf
from timeit import default_timer as timer

from sl_model import DSODSL512, SL512
from sl_utils import PriorUtil
from ssd_data import preprocess
from sl_utils import rbox_to_polygon

from crnn_model import CRNN
from crnn_utils import alphabet87 as alphabet
from crnn_data import crop_words
from crnn_utils import decode


if __name__ == '__main__':
    
    Model = DSODSL512
    input_shape = (512,512,3)
    weights_path = './checkpoints/201711132011_dsodsl512_synthtext/weights.001.h5'
    segment_threshold = 0.55
    link_threshold = 0.40
    
    sl_graph = tf.Graph()
    with sl_graph.as_default():
        sl_session = tf.Session()
        with sl_session.as_default():
            sl_model = Model(input_shape)
            prior_util = PriorUtil(sl_model)
            sl_model.load_weights(weights_path, by_name=True)
    
    #input_width = 256
    input_width = 384
    input_height = 32
    weights_path = './checkpoints/201806190711_crnn_gru_synthtext/weights.300000.h5'
    
    crnn_graph = tf.Graph()
    with crnn_graph.as_default():
        crnn_session = tf.Session()
        with crnn_session.as_default():
            crnn_model = CRNN((input_width, input_height, 1), len(alphabet), prediction_only=True, gru=True)
            crnn_model.load_weights(weights_path, by_name=True)
    
    # To test on webcam 0, /dev/video0
    video_path = 0
    start_frame = 0
    record = True
    record_file_name = 'sl_end2end_record.avi'
    try:
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
        
        vid_w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid_h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # skip frames until reaching start_frame
        if start_frame > 0:
            vid.set(cv2.CAP_PROP_POS_MSEC, start_frame)
            
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        
        input_size = input_shape[:2]
        
        record_buffer = []
        record_timestamps = []
        init_time = timer()
        
        while True:
            retval, img = vid.read()
            if not retval:
                print("Done!")
                break
                
            # model to predict 
            x = np.array([preprocess(img, input_size)])
            with sl_graph.as_default():
                with sl_session.as_default():
                    y = sl_model.predict(x)
            
            result = prior_util.decode(y[0], segment_threshold, link_threshold)
            
            img1 = np.copy(img)
            img2 = np.zeros_like(img)
            
            # calculate fps
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            
            rboxes = result[:,:5]
            
            if len(rboxes) > 0:
                bh = rboxes[:,3]
                rboxes[:,2] += bh * 0.1
                rboxes[:,3] += bh * 0.2
                
                boxes = np.asarray([rbox_to_polygon(r) for r in rboxes])
                boxes = np.flip(boxes, axis=1) # TODO: fix order of points, why?
                boxes = np.reshape(boxes, (-1, 8))
                
                boxes_mask = np.array([not (np.any(b < 0-10) or np.any(b > 512+10)) for b in boxes]) # box inside image
                
                boxes = boxes[boxes_mask]
                rboxes = rboxes[boxes_mask]
                
                if len(boxes) == 0:
                    boxes = np.empty((0,8))
                
                #for b in boxes:
                #    xy = b.reshape((-1,1,2)) / input_size * [vid_w, vid_h]
                #    xy = np.round(xy)
                #    xy = xy.astype(np.int32)
                #    cv2.polylines(img1, [xy], True, (0,0,255))
                
                boxes = np.clip(boxes/512, 0, 1)
                
                words = crop_words(img, boxes, input_height, width=input_width, grayscale=True)
                words = np.asarray([w.transpose(1,0,2) for w in words])
                
                if len(words) > 0:
                    with crnn_graph.as_default():
                        with crnn_session.as_default():
                            res_crnn = crnn_model.predict(words)
                
                xy = rboxes[:,:2]
                xy[:,0] = xy[:,0] - rboxes[:,2] / 2
                xy = xy / input_size * [vid_w, vid_h]
                
                for i in range(len(words)):
                    idxs = np.argmax(res_crnn[i], axis=1)
                    confs = res_crnn[i][range(len(idxs)),idxs]
                    non_blank_mask = idxs != len(alphabet)-1
                    
                    if np.any(non_blank_mask):
                        mean_conf = np.mean(confs[non_blank_mask])
                        chars = [alphabet[c] for c in idxs]
                        res_str = decode(chars)
                        
                        # filter based on recognition threshold
                        #if mean_conf > 0.7-0.4*np.exp(-0.1*np.sum(non_blank_mask)):
                        if mean_conf > 0.6:
                            b = boxes[i].reshape((-1,1,2)) * [vid_w, vid_h]
                            b = np.asarray(np.round(b), dtype=np.int32)
                            cv2.polylines(img1, [b], True, (0,0,255))
                            
                            #cv2.imwrite('croped_word_%03i.png' % (i), words[i])
                            cv2.putText(img2, res_str, tuple(xy[i].astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                        else:
                            #print('drop %5.3f %s' % (mean_conf, res_str))
                            pass
            
            # draw fps
            cv2.rectangle(img1, (0,0), (50, 17), (255,255,255), -1)
            cv2.putText(img1, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            
            img = np.concatenate((img1, img2), axis=1)
            
            cv2.imshow("SegLink detection", img)
            
            if record:
                record_buffer.append(img)
                record_timestamps.append(timer()-init_time)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if record:
                    print('write viedo file: %s' % (record_file_name))
                    fourcc = cv2.VideoWriter_fourcc(*'HFYU') # losseless
                    output_size = (record_buffer[0].shape[1], record_buffer[0].shape[0])
                    output_framerate = 20.0
                    out = cv2.VideoWriter(record_file_name, fourcc, output_framerate, output_size)
                    #for i in range(len(record_buffer)):
                    #    out.write(record_buffer[i])
                    i = -1
                    for t in np.arange(0, record_timestamps[-1], 1/output_framerate):
                        if i == -1 and t > record_timestamps[0]:
                            i += 1
                        elif t > record_timestamps[i]:
                            i += 1
                        
                        if i == -1:
                            output_img = np.zeros_like(record_buffer[0])
                        else:
                            output_img = record_buffer[i]
                        out.write(output_img)
                    out.release()
                break
            
    except KeyboardInterrupt:
        pass

# ffmpeg -y -i sl_end2end_record.avi -c:v libx264 -b:v 2400k -preset slow -movflags +faststart -pix_fmt yuv420p sl_end2end_record.mp4
