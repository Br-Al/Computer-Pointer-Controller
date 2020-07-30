from generic import Generic
class LandmarksDetector(Generic):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        Generic.__init__(self, model_name, device, extensions)

    
    def preprocess_output(self, image, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        
        h = image.shape[0]
        w = image.shape[1]
        padding = int((h*w)/2)
        coords = []
        for output in outputs:
            
            output = output.reshape(-1,2)
            coords.append(output)
            
        left_eye_coords = coords[0][0]
        rigth_eye_coords = coords[0][1]
        left_eye_x = int(left_eye_coords[0]*w)
        left_eye_y = int(left_eye_coords[1]*h)
        rigth_eye_x = int(rigth_eye_coords[0]*w)
        rigth_eye_y = int(rigth_eye_coords[1]*h)
        left_eye = image[left_eye_y-padding:left_eye_y+padding, left_eye_x-padding:left_eye_x+padding]
        rigth_eye = image[rigth_eye_y-padding:rigth_eye_y+padding, rigth_eye_x-padding:rigth_eye_x+padding]
       
        
        real_coords = [left_eye_x, left_eye_y, rigth_eye_x, rigth_eye_y]
        return left_eye, rigth_eye, real_coords

