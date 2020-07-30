import cv2
import numpy as np
import math

class Drawer:

    '''
        Class for visualzing models outputs of the models.
    '''
    def __init__(self, face, eyes_landmarks, head_pose_angles, gaze):
        self.face = face
        self.eyes_landmarks = eyes_landmarks
        self.head_pose_angles = head_pose_angles
        self.gaze = gaze
        self.lx, self.ly, self.rx, self.ry = self.eyes_landmarks

    def draw_landmarks(self, padding):

        

        cv2.rectangle(self.face, (self.lx-padding,self.ly-padding),(self.lx+padding,self.ly+padding),(0,255,0),1)
        cv2.rectangle(self.face, (self.rx-padding,self.ry-padding),(self.rx+padding,self.ry+padding),(0,255,0),1)
    

    def draw_gazes(self):
        
        arrow = 0.6 * self.face.shape[1]
        gaze_arrow = np.array([self.gaze[0], - self.gaze[1]], dtype=np.float64) * arrow
        
        gx, gy = gaze_arrow.astype(int)

        cv2.arrowedLine(self.face, (self.lx, self.ly), (self.lx + gx, self.ly + gy), (0, 255,255), 1)
        cv2.arrowedLine(self.face, (self.rx, self.ry), (self.rx + gx, self.ry + gy), (0,255,255), 1)


    def draw_head_pose(self):
        

        sin_yaw = math.sin(self.head_pose_angles[0] * math.pi / 180)
        sin_pitch = math.sin(self.head_pose_angles[1] * math.pi / 180 )
        sin_roll = math.sin(self.head_pose_angles[2])

        cos_yaw = math.cos(self.head_pose_angles[0] * math.pi / 180)
        cos_pitch = math.cos(self.head_pose_angles[1] * math.pi / 180 )
        cos_roll = math.cos(self.head_pose_angles[2] * math.pi / 180 )


        axis_length = 0.5 * self.face.shape[1]

        

        center = (int((self.lx + self.rx)/2),  int((self.ly + self.ry)/2))
      
        cv2.arrowedLine(self.face, center, (int(center[0] + axis_length * (cos_roll * cos_yaw + sin_yaw * sin_pitch * sin_roll)),\
             int(center[1] + axis_length * cos_pitch * sin_roll)), (0,0,255), 1)

        cv2.arrowedLine(self.face, center, (int(center[0] + axis_length * (cos_roll * sin_yaw * sin_pitch + cos_yaw * sin_roll)),\
             int(center[1] - axis_length * cos_pitch * cos_roll)), (0,255,0), 1)

        cv2.arrowedLine(self.face, center, (int(center[0] + axis_length * sin_yaw * cos_pitch),\
             int(center[1] + axis_length * sin_pitch)), (255,0,0), 1)

        cv2.putText(self.face, "yaw={:.3f}, pitch={:.3f}, roll={:.3f}"\
            .format(self.head_pose_angles[0], self.head_pose_angles[1], self.head_pose_angles[2]), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
        


    def show(self):
        cv2.imshow('face', self.face)