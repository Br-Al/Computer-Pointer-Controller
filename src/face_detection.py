from generic import Generic
import cv2
import numpy as np
class FaceDetector(Generic):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.4):
        '''
        TODO: Use this to set your instance variables.
        '''
        Generic.__init__(self, model_name, device, extensions, threshold)


    def preprocess_output(self, image, outputs, initial_w, initial_h):
    
        boxes = outputs[0,0]
        for box in boxes:
            if box[2] > self.threshold:
                x_min = int(box[3]*initial_w)
                y_min = int(box[4]*initial_h)
                x_max = int(box[5]*initial_w)
                y_max = int(box[6]*initial_h)
                image = image[y_min:y_max, x_min:x_max]
                
                break 
        return image
