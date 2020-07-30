from face_detection import FaceDetector
from facial_landmarks_detection import LandmarksDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimator
from input_feeder import InputFeeder
from mouse_controller import MouseController
from draw import Drawer


from argparse import ArgumentParser
import cv2
import numpy as np
import math
import time
import logging as log

# Default Models, by defauld all models use the FP32 precission

FACE_DETECTION_MODEL = 'models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
LANDMARKS_MODEL = 'models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
HEAD_POSE_MODEL = 'models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'
GAZE_MODEL = 'models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'

# Devices 
DEVICES = ['CPU', 'GPU', 'FPGA', 'VPU']

def build_argparser():

    parser = ArgumentParser()

    parser.add_argument('-i', '--input', default=0, help="Path to the input video")
    parser.add_argument('-pt', '--threshold', default=0.4, help="Probability threshold for detections filtering")
    #models
    parser.add_argument('-m_fd', '--model_fd', default=FACE_DETECTION_MODEL, required=False, help='Face detection model name path')
    parser.add_argument('-m_ld', '--model_ld', default=LANDMARKS_MODEL, required=False, help='Landmarks detection model name path')
    parser.add_argument('-m_hpe', '--model_hpe', default=HEAD_POSE_MODEL, required=False, help='Head pose estimation model name path')
    parser.add_argument('-m_ge', '--model_ge', default=GAZE_MODEL, required=False, help='Gaze estimation model name path')


    #devices choice one of the listed devices
    parser.add_argument('-d_fd', '--device_fd', choices=DEVICES, default='CPU', help='Face detection device')
    parser.add_argument('-d_ld', '--device_ld', choices=DEVICES, default='CPU', help='Landmarks detection device')
    parser.add_argument('-d_hpe', '--device_hpe', choices=DEVICES, default='CPU', help='Head pose estimation device')
    parser.add_argument('-d_ge', '--device_ge', choices=DEVICES, default='CPU', help='Gaze estimation device')

    #extensions (openvino 2020 adds extensions automatically)
    parser.add_argument('-e_fd', '--ext_fd', help='Face detection model extension')
    parser.add_argument('-e_ld', '--ext_ld', help='Landmarks detection model extension')
    parser.add_argument('-e_hpe', '--ext_hpe', help='Head pose estimation model extension')
    parser.add_argument('-e_ge', '--ext_ge', help='Gaze estimation model extension')

    return parser
    
def main(args):

    fd_infer_time, ld_infer_time, hpe_infer_time, ge_infer_time = 0 ,0 ,0 ,0

    start = time.time()
    face_detector = FaceDetector(args.model_fd, args.device_fd, args.ext_fd)
    fd_load_time = time.time() - start 

    start = time.time()
    landmarks_detector = LandmarksDetector(args.model_ld, args.device_ld, args.ext_ld)
    ld_load_time = time.time() - start 

    start = time.time()
    head_pose_estimator = HeadPoseEstimator(args.model_hpe, args.device_hpe, args.ext_hpe)
    hpe_load_time = time.time() - start 

    start = time.time()
    gaze_estimator = GazeEstimator(args.model_ge, args.device_ge, args.ext_ge)
    ge_load_time = time.time() - start 
    log.info("Models Loading...")
    log.info("Face detection load time       :{:.4f}ms".format(fd_load_time))    
    log.info("Landmarks estimation load time :{:.4f}ms".format(ld_load_time))     
    log.info("Head pose estimation load time :{:.4f}ms".format(hpe_load_time))     
    log.info("Gaze estimation load time      :{:.4f}ms".format(ge_load_time))  
    log.info('All Models loaded')
    mouse_controller = MouseController('high', 'fast')


    if args.input == 0:
        input_feeder = InputFeeder('cam', args.input)
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        input_feeder = InputFeeder('image', args.input)
    else:
        input_feeder = InputFeeder('video', args.input)
    
    input_feeder.load_data()
    init_w  = input_feeder.init_w
    init_h =  input_feeder.init_h
    

    counter = 0

    for flag, frame in input_feeder.next_batch():
        
        if not flag:
            break

        counter +=1

        key = cv2.waitKey(60)
        try:
            start = time.time()
            outputs = face_detector.predict(frame)
            
            face = face_detector.preprocess_output(frame, outputs, init_w, init_h)
            
            fd_infer_time += time.time() - start

            start = time.time()
            outputs = landmarks_detector.predict(face)
            
            left_eye, right_eye, real_landmraks = landmarks_detector.preprocess_output(face, outputs)
           
            ld_infer_time += time.time() - start

            start = time.time()

            outputs = head_pose_estimator.predict(face)
            head_pose_angles = head_pose_estimator.preprocess_output(outputs)
            
            hpe_infer_time += time.time() - start

            
            start = time.time()
            
            outputs = gaze_estimator.predict(left_eye, right_eye, head_pose_angles)
            
            gaze = gaze_estimator.preprocess_output(outputs)
            
            ge_infer_time += time.time() - start
  

            log.info("Face detection time       :{:.4f}ms".format(fd_infer_time/counter))    
            log.info("Landmarks estimation time :{:.4f}ms".format(ld_infer_time/counter))     
            log.info("Head pose estimation time :{:.4f}ms".format(hpe_infer_time/counter))     
            log.info("Gaze estimation time      :{:.4f}ms".format(ge_infer_time/counter))     

            if args.input != 0:
                drawer = Drawer(face, real_landmraks, head_pose_angles, gaze)
                drawer.draw_landmarks(20)
                drawer.draw_head_pose()
                drawer.draw_gazes()
                drawer.show()
            roll_cos = math.cos(head_pose_angles[2] *  math.pi/180)

            roll_sin = math.sin(head_pose_angles[2] *  math.pi/180)

            mouse_x = gaze[0] * roll_cos + gaze[0] * roll_sin
            mouse_y = gaze[1] * roll_cos + gaze[1] * roll_sin

            mouse_controller.move(mouse_x, mouse_y)

        except Exception as e:
            log.error(e)
        finally:
            if key == 27:
                break

    input_feeder.close()


if __name__ == '__main__':

   
    args = build_argparser().parse_args()

    main(args)
