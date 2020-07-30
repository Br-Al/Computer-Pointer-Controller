from generic import Generic
class GazeEstimator(Generic):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        Generic.__init__(self, model_name, device, extensions)
        self.input_shape =  self.model.input_info['left_eye_image'].input_data.shape 

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        
        left_eye = self.preprocess_input(left_eye_image)
        right_eye = self.preprocess_input(right_eye_image)
        input_dict = {'left_eye_image':left_eye, 'right_eye_image':right_eye, 'head_pose_angles':head_pose_angles}
        outputs = self.exec_network.infer(input_dict)[self.output_name]
        
        return outputs

    def preprocess_output(self, outputs):
        return outputs[0]
    
