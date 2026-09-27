#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from collections import Counter

class FaceRec:
    def __init__(self):
        # 初始化人脸识别模型与分类器
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('/home/zjw1/nav_ws/src/face_detection/trainer/trainer.yml')
        self.cascadePath = "/home/zjw1/nav_ws/src/face_detection/trainer/haarcascade_frontalface_alt.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 名称映射字典（新增部分）
        self.names = ['None', 'zjw', 'zjw', 'lhb', 'wl', 'ljq']
        self.name_dict = {  # 新增中文字典
            'zjw': '张佳伟',
            'hl': '黄来',
            'lhb': '李华宝',
            'wl': '王乐',
            'ljq': '刘家杞',
            'unknown': '未知'
        }
        
        self.current_id = "None"
        self.final_name = "unknown"
        self.detected_names = []

        # ROS初始化
        rospy.init_node('face_recognition_node')
        self.tts_pub = rospy.Publisher('/tts_text', String, queue_size=10)
        self.image_pub = rospy.Publisher('/face_recognition', Image, queue_size=10)
        rospy.Subscriber('/iat_text', String, self.iat_callback)
        rospy.Subscriber('/zed/zed_node/left/image_rect_color', Image, self.image_callback)
        self.bridge = CvBridge()
        
        rospy.Timer(rospy.Duration(2), self.aggregate_names)

    def iat_callback(self, msg):
        if "前面是谁" in msg.data or "前面是谁1" in msg.data:
            # 使用字典转换最终名称（修改部分）
            chinese_name = self.name_dict.get(self.final_name, '未知')
            response = "前面是 " + chinese_name
            self.tts_pub.publish(response)
    
    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        img = frame.copy()
        self.face_recon(img)
        
        cv2.imshow("face_recognition", img)
        cv2.waitKey(1)
    
    def face_recon(self, img):
        minW = 30
        minH = 30
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(minW, minH))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 99:
                self.current_id = self.names[id]
                confidence_text = "  {0}%".format(round(160 - confidence))
            else:
                self.current_id = "unknown"
                confidence_text = "  {0}%".format(round(160 - confidence))
            
            if self.current_id not in ["unknown", "None"]:
                self.detected_names.append(self.current_id)
            
            # 保持原始ID显示（维持原状）
            cv2.putText(img, str(self.current_id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)
        
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            self.image_pub.publish(img_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
    
    def aggregate_names(self, event):
        if self.detected_names:
            count = Counter(self.detected_names)
            most_common_name, _ = count.most_common(1)[0]
            self.final_name = most_common_name
        else:
            self.final_name = "unknown"
        self.detected_names = []

if __name__ == '__main__':
    face_rec = FaceRec()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()