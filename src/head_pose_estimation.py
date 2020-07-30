from generic import Generic
class HeadPoseEstimator(Generic):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        Generic.__init__(self, model_name, device, extensions)

   

    def predict(self, image):
        
        input_img = self.preprocess_input(image)
        input_dict = {self.input_name:input_img}
        outputs = self.exec_network.infer(input_dict)
        return outputs


    def preprocess_output(self, outputs):
        yaw = outputs['angle_y_fc'][0][0]
        pitch =outputs['angle_p_fc'][0][0]
        roll  = outputs['angle_r_fc'][0][0]
        return [yaw, pitch, roll]
