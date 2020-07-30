from openvino.inference_engine import IECore
import cv2
import numpy as np
import logging as log

class Generic:
    '''
    Class for all the Models.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.4):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions = extensions
        self.threshold = threshold

        try:
            self.core = IECore()
            self.model=self.core.read_network(model=self.model_structure, weights=self.model_weights)
            self.check_model()
            self.load_model()
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        key = list(self.model.input_info.keys())[0]
        self.input_name= self.model.input_info[key].name
        self.input_shape=self.model.input_info[self.input_name].input_data.shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(network = self.model, device_name = self.device)
        return self.exec_network

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img = self.preprocess_input(image)
        input_dict = {self.input_name:input_img}
        outputs = self.exec_network.infer(input_dict)[self.output_name]
        return outputs

    def check_model(self):
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.info("Unsupported layers found for the given device ({}):\n {}".format(self.device, unsupported_layers))
            if 'CPU' in self.device and self.extensions:
                self.core.add_extension(self.extensions, self.device)


    def preprocess_input(self, image):
        n, c, h, w = self.input_shape
        image=cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
        image=np.moveaxis(image, -1, 0)
        
        return image
