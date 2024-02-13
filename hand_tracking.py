import cv2
import mediapipe as mp 
import numpy as np
import time 


#TYPE
confidence = float 
webcam_image = np.ndarray
rgb_tuple = tuple[int, int, int]


#CLASS

class Detect:
    def __init__(self, 
                 mode: bool = False, 
                 number_hands: int = 2, 
                 model_complexity: int = 1, 
                 min_detection_confidance: confidence = 0.5, 
                 min_tracking_confidance: confidence = 0.5 ):
        

        #Parameters to initialize Hands 
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detection_confidance
        self.tracking_con = min_tracking_confidance

        
        #Initialize Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, 
                                         self.max_num_hands,
                                         self.complexity,
                                         self.detection_con,
                                         self.tracking_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self,
                   img: webcam_image,
                   draw_hands: bool = True):
        
        #Color Correction - BGR to RGB
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Collect and Analyze Hands process results 
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks and draw_hands:
            for hand in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)

        return img
    
    def find_position(self, 
                      img:webcam_image,
                      hand_number: int = 0):
        self.required_landmark_list = []

        if self.results.multi_hand_landmarks:
            height, width, _ = img.shape
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                center_x, center_y = int(lm.x*width),int(lm.y*height)
                
                self.required_landmark_list.append([id, center_x, center_y])


               
        return self.required_landmark_list

            


#CLASS TEST

if __name__ == '__main__':

    #Class
    Detect = Detect()

    #Image Capture
    capture = cv2.VideoCapture(0)

    #FrameRate Capture 
    previour_time = 0
    current_time = 0


    while True:

      #Frame Capture
      _, img = capture.read()

      #calculate FPS
      current_time = time.time()
      fps = 1/(current_time - previour_time)
      previour_time = current_time


      #Manipulate frame
      cv2.putText(img,
                  str(int(fps)),
                  (10, 70), 
                  cv2.FONT_HERSHEY_DUPLEX, 
                  2, 
                  (255, 255, 255),
                  3)
      img = Detect.find_hands(img)
      landmark_list = Detect.find_position(img)
      if landmark_list:
          print(landmark_list[8])

      #Show Frame
      cv2.imshow('Hand Tracking Webcam', img)

      #Quit
      if cv2.waitKey(20) & 0xFF == ord('q'):
          break
