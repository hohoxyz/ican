#!/usr/bin/env python3
import fastdeploy as fd
import cv2
import os
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

def ocr_detect(im):
    det_model_file = os.path.join("ch_PP-OCRv3_det_infer", "/home/zjw1/nav_ws/src/blind_vision/ocr/ch_PP-OCRv3_det_infer/inference.pdmodel")
    det_params_file = os.path.join("ch_PP-OCRv3_det_infer", "/home/zjw1/nav_ws/src/blind_vision/ocr/ch_PP-OCRv3_det_infer/inference.pdiparams")
    
    cls_model_file = os.path.join("ch_ppocr_mobile_v2.0_cls_infer", "/home/zjw1/nav_ws/src/blind_vision/ocr/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel")
    cls_params_file = os.path.join("ch_ppocr_mobile_v2.0_cls_infer", "/home/zjw1/nav_ws/src/blind_vision/ocr/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams")
    
    rec_model_file = os.path.join("ch_PP-OCRv3_rec_infer", "/home/zjw1/nav_ws/src/blind_vision/ocr/ch_PP-OCRv3_rec_infer/inference.pdmodel")
    rec_params_file = os.path.join("ch_PP-OCRv3_rec_infer", "/home/zjw1/nav_ws/src/blind_vision/ocr/ch_PP-OCRv3_rec_infer/inference.pdiparams")
    
    rec_label_file = "/home/zjw1/nav_ws/src/blind_vision/ocr/ppocr_keys_v1.txt"

    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    det_option.use_ort_backend()
    cls_option.use_ort_backend()
    rec_option.use_ort_backend()

    det_model = fd.vision.ocr.DBDetector(det_model_file, det_params_file, runtime_option=det_option)
    cls_model = fd.vision.ocr.Classifier(cls_model_file, cls_params_file, runtime_option=cls_option)
    rec_model = fd.vision.ocr.Recognizer(rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

    # Parameters settings for pre and post processing of Det/Cls/Rec Models.
    det_model.preprocessor.max_side_len = 1280
    det_model.postprocessor.det_db_thresh = 0.3
    det_model.postprocessor.det_db_box_thresh = 0.6
    det_model.postprocessor.det_db_unclip_ratio = 1.5
    det_model.postprocessor.det_db_score_mode = "slow"
    cls_model.postprocessor.cls_thresh = 0.8

    ppocr_v3 = fd.vision.ocr.PPOCRv3(det_model=det_model, cls_model=cls_model, rec_model=rec_model)

    ppocr_v3.cls_batch_size = 1
    ppocr_v3.rec_batch_size = 12
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)  # 转回三通道用于OCR
    result = ppocr_v3.predict(enhanced)
    
    return result

class OCRNode:
    def __init__(self):
        rospy.init_node('ocr_node')
        self.frame_count = 0  # 初始化计数器        
        self.ocr_pub = rospy.Publisher('/ocr_result', String, queue_size=10)
        self.vis_pub = rospy.Publisher('/vis_im_result', Image, queue_size=10)  # 新增的发布者
        self.bridge = CvBridge()  # 初始化CvBridge
        rospy.Subscriber('/zed/zed_node/left/image_rect_color', Image, self.image_callback)


    def image_callback(self, msg):
        # 处理第一帧
        if self.frame_count == 0:
            try:
                img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if img_data.shape[2] == 4:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)

                result = ocr_detect(img_data)
                rospy.loginfo("OCR Result: %s", result)
                self.ocr_pub.publish(str(result))

                vis_im = fd.vision.vis_ppocr(img_data, result)
                #cv2.imshow('character_recognition', vis_im)
                # 将可视化结果发布到/vis_im_result话题
                vis_im_msg = self.bridge.cv2_to_imgmsg(vis_im, encoding="bgr8")  # 转换为ROS图像消息
                self.vis_pub.publish(vis_im_msg)


                # 设置计数器为5，等待下次处理
                self.frame_count = 10  
            except Exception as e:
                rospy.logerr("Error processing image: %s", e)
        else:
            # 计数器递减
            self.frame_count -= 1

if __name__ == '__main__':
    try:
        node = OCRNode()
        rospy.spin()  # 保持节点运行，等待回调
    except rospy.ROSInterruptException:
        pass
