#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import subprocess
import os
import signal
import time

class VoiceCommandProcessor:
    def __init__(self):
        rospy.init_node('voice_command_processor')
        self.process = None
        self.tts_pub = rospy.Publisher('/tts_text', String, queue_size=10)
        self.iat_pub = rospy.Publisher('/iat_text', String, queue_size=10)
        rospy.Subscriber('/iat_text', String, self.iat_callback)
        rospy.loginfo("Voice Command Processor Initialized")

    def iat_callback(self, msg):
        command = msg.data.strip()
        rospy.loginfo(f"Received command: {command}")

        if command == "前面是谁":
            self.handle_command(
                launch_cmd=['rosrun', 'face_detection', 'recongnition_ros.py'],
                initial_delay=3,
                repeat_command="前面是谁1",
                repeat_delay=4
            )
        elif command == "这是什么药":
            self.handle_command(
                launch_cmd=['roslaunch', 'ocr_recognition', 'ocr_launch.launch'],
                initial_delay=5,
                repeat_command="前面什么文字1",
                repeat_delay=4
            )
        elif command == "前面有什么":
            self.handle_command(
                launch_cmd=['rosrun', 'object_detection', 'zed_detect_ros.py'],
                initial_delay=7,
                repeat_command="前面有什么1",
                repeat_delay=4                
            )
        elif command == "打开导航":
            self.handle_command(
                launch_cmd=['roslaunch', 'navigation', 'nav_module.launch'],
                initial_delay=7,
                repeat_command="打开导航1",
                repeat_delay=4,
                auto_stop=False  # 不自动停止
            )

    def handle_command(self, launch_cmd, initial_delay, repeat_command, repeat_delay, auto_stop=True):
        if self.process:
            self.stop_process()

        self.tts_pub.publish("正在启动")
        rospy.loginfo(f"Starting process: {' '.join(launch_cmd)}")
        self.process = subprocess.Popen(launch_cmd)

        rospy.sleep(initial_delay)
        self.iat_pub.publish(repeat_command)
        rospy.loginfo(f"Re-published command: {repeat_command}")

        rospy.sleep(repeat_delay)
        if auto_stop:
            self.stop_process()
        else:
            rospy.loginfo("Process left running (auto_stop=False)")

    def stop_process(self):
        if self.process:
            rospy.loginfo(f"Stopping process with PID: {self.process.pid}")
            os.kill(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rospy.logwarn(f"Process PID {self.process.pid} did not terminate gracefully; killing it.")
                os.kill(self.process.pid, signal.SIGKILL)
                self.process.wait()
            self.process = None
            rospy.loginfo("Process stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_process()

if __name__ == "__main__":
    try:
        with VoiceCommandProcessor() as processor:
            rospy.spin()
    except rospy.ROSInterruptException:
        pass

