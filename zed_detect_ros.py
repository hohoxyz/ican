#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import threading

class ObjectDetectionNode:
    def __init__(self):
        rospy.init_node('object_detection_node')
        self.bridge = CvBridge()
        self.yolo_model = YOLO('/home/zjw1/nav_ws/src/object_detection/scripts/yolo11n.pt')  

        self.image_sub = rospy.Subscriber('/zed/zed_node/left/image_rect_color', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/zed/zed_node/depth/depth_registered', Image, self.depth_callback)
        self.command_sub = rospy.Subscriber('/iat_text', String, self.command_callback)

        self.tts_pub = rospy.Publisher('/tts_text', String, queue_size=1)
        self.detection_pub = rospy.Publisher('/detection_result', Image, queue_size=1)

        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_depth = None
        self.latest_objects = {}  # 保存目标和距离信息

        self.category_dict = {
            'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorbike': '摩托车',
            'aeroplane': '飞机', 'bus': '公交车', 'train': '火车', 'truck': '卡车',
            'boat': '船', 'traffic light': '红绿灯', 'fire hydrant': '消防栓',
            'stop sign': '停止标志', 'parking meter': '停车收费表', 'bench': '长凳',
            'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马', 'sheep': '羊',
            'cow': '牛', 'elephant': '象', 'bear': '熊', 'zebra': '斑马',
            'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞',
            'handbag': '手提包', 'tie': '领带', 'suitcase': '手提箱', 'frisbee': '飞盘',
            'skis': '滑雪板', 'snowboard': '单板滑雪', 'sports ball': '运动球',
            'kite': '风筝', 'baseball bat': '棒球棒', 'baseball glove': '棒球手套',
            'skateboard': '滑板', 'surfboard': '冲浪板', 'tennis racket': '网球拍',
            'bottle': '瓶子', 'wine glass': '红酒杯', 'cup': '杯子', 'fork': '叉子',
            'knife': '刀', 'spoon': '勺', 'bowl': '碗', 'banana': '香蕉',
            'apple': '苹果', 'sandwich': '三明治', 'orange': '橙子',
            'broccoli': '西兰花', 'carrot': '胡萝卜', 'hot dog': '热狗',
            'pizza': '比萨', 'donut': '甜甜圈', 'cake': '蛋糕', 'chair': '椅子',
            'sofa': '长椅', 'pottedplant': '盆栽', 'bed': '床', 
            'dining table': '餐桌', 'toilet': '马桶', 'tvmonitor': '电视',
            'laptop': '笔记本电脑', 'mouse': '鼠标', 'remote': '遥控器',
            'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉',
            'oven': '烤箱', 'toaster': '烤面包机', 'sink': '洗碗槽',
            'refrigerator': '冰箱', 'book': '书', 'clock': '时钟',
            'vase': '花瓶', 'scissors': '剪刀', 'teddy bear': '泰迪熊',
            'hair drier': '吹风机', 'toothbrush': '牙刷'
        }

    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.perform_detection()  # 持续检测
        detection_img = self.bridge.cv2_to_imgmsg(self.latest_image, 'bgr8')
        self.detection_pub.publish(detection_img)

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def command_callback(self, msg):
        if msg.data.strip() in ["前面有什么1"]:
            with self.lock:
                if self.latest_objects:
                    parts = []
                    for label_en, distance in self.latest_objects.items():
                        label_cn = self.category_dict.get(label_en, label_en)
                        parts.append(f"{label_cn}距离{distance:.2f}米")
                    result_str = "前面有" + "，".join(parts)
                else:
                    result_str = "前方未检测到目标"
            self.tts_pub.publish(String(data=result_str))

    def perform_detection(self):
        if self.latest_image is None or self.latest_depth is None:
            return

        results = self.yolo_model(self.latest_image, verbose=False)
        latest_objects = {}

        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = self.yolo_model.names[class_id]

                center_x = np.clip((x1 + x2) // 2, 0, self.latest_depth.shape[1] - 1)
                center_y = np.clip((y1 + y2) // 2, 0, self.latest_depth.shape[0] - 1)
                depth = self.latest_depth[center_y, center_x]

                if not np.isnan(depth) and depth > 0:
                    # 更新最后一次的目标信息
                    if label not in latest_objects or depth < latest_objects[label]:
                        latest_objects[label] = depth

                    # 绘图显示
                    cv2.rectangle(self.latest_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_cn = self.category_dict.get(label, label)
                    cv2.putText(self.latest_image, f'{label} {depth:.2f}m',
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        with self.lock:
            self.latest_objects = latest_objects

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = ObjectDetectionNode()
    node.run()
