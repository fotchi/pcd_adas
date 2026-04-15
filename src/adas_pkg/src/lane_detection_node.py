#!/usr/bin/env python3
"""
Lane Detection Node
Subscribe: /carla/camera/seg
Publish:   /adas/lane_status, /adas/lane_image
"""
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String


@dataclass
class LaneConfig:
    ROI_TOP_RATIO: float = 0.5
    CANNY_LOW: int = 50
    CANNY_HIGH: int = 150
    HOUGH_THRESHOLD: int = 30
    MIN_LINE_LENGTH: int = 30
    MAX_LINE_GAP: int = 15
    SLOPE_THRESHOLD: float = 0.3
    MORPH_KERNEL: int = 5
    BLUR_KERNEL: int = 5
    STATUS_WINDOW: int = 5
    STATUS_MIN_COUNT: int = 3
    CAMERA_TOPIC: str = '/carla/camera/rgb'
    STATUS_PUB_TOPIC: str = '/adas/lane_status'
    IMAGE_PUB_TOPIC: str = '/adas/lane_image'
    QUEUE_SIZE: int = 1


class LaneDetectionNode:
    def __init__(self):
        rospy.init_node('lane_detection_node')
        self.bridge = CvBridge()
        self.config = self._load_config()
        self.status_history = deque(maxlen=self.config.STATUS_WINDOW)

        self.pub_status = rospy.Publisher(
            self.config.STATUS_PUB_TOPIC,
            String,
            queue_size=self.config.QUEUE_SIZE
        )
        self.pub_image = rospy.Publisher(
            self.config.IMAGE_PUB_TOPIC,
            Image,
            queue_size=self.config.QUEUE_SIZE
        )

        rospy.Subscriber(self.config.CAMERA_TOPIC, Image, self.callback)
        rospy.loginfo("Lane Detection Node demarre")

    @staticmethod
    def _load_config() -> LaneConfig:
        config = LaneConfig()
        config.ROI_TOP_RATIO = rospy.get_param('~roi_top_ratio', config.ROI_TOP_RATIO)
        config.CANNY_LOW = rospy.get_param('~canny_low', config.CANNY_LOW)
        config.CANNY_HIGH = rospy.get_param('~canny_high', config.CANNY_HIGH)
        config.HOUGH_THRESHOLD = rospy.get_param('~hough_threshold', config.HOUGH_THRESHOLD)
        config.MIN_LINE_LENGTH = rospy.get_param('~min_line_length', config.MIN_LINE_LENGTH)
        config.MAX_LINE_GAP = rospy.get_param('~max_line_gap', config.MAX_LINE_GAP)
        config.SLOPE_THRESHOLD = rospy.get_param('~slope_threshold', config.SLOPE_THRESHOLD)
        config.MORPH_KERNEL = rospy.get_param('~morph_kernel', config.MORPH_KERNEL)
        config.BLUR_KERNEL = rospy.get_param('~blur_kernel', config.BLUR_KERNEL)
        config.STATUS_WINDOW = rospy.get_param('~status_window', config.STATUS_WINDOW)
        config.STATUS_MIN_COUNT = rospy.get_param('~status_min_count', config.STATUS_MIN_COUNT)
        config.ROI_TOP_RATIO = max(0.3, min(0.9, float(config.ROI_TOP_RATIO)))
        config.SLOPE_THRESHOLD = max(0.1, float(config.SLOPE_THRESHOLD))
        return config

    @staticmethod
    def _odd_kernel(value: int, minimum: int = 3) -> int:
        k = max(minimum, int(value))
        return k if k % 2 == 1 else k + 1

    def _stabilize_status(self, status: str) -> str:
        self.status_history.append(status)
        if len(self.status_history) < self.config.STATUS_MIN_COUNT:
            return status
        counter = Counter(self.status_history)
        return counter.most_common(1)[0][0]

    def callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {e}")
            return

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # Marquages blancs : luminosité élevée
        white_mask = cv2.inRange(hls, (0, 180, 0), (255, 255, 255))

        # Marquages jaunes : teinte ~15-35, saturation et luminosité suffisantes
        yellow_mask = cv2.inRange(hls, (15, 80, 80), (35, 255, 255))

        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

        k = self._odd_kernel(self.config.MORPH_KERNEL)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)

        blur_k = self._odd_kernel(self.config.BLUR_KERNEL)
        lane_mask = cv2.GaussianBlur(lane_mask, (blur_k, blur_k), 0)

        if np.count_nonzero(lane_mask) < 50:
            status = self._stabilize_status('NO_LANE')
            self.pub_status.publish(String(data=status))
            out = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            try:
                self.pub_image.publish(self.bridge.cv2_to_imgmsg(out, encoding='bgr8'))
            except CvBridgeError:
                pass
            return

        h, w = lane_mask.shape
        roi_start = int(h * self.config.ROI_TOP_RATIO)
        roi = lane_mask[roi_start:, :]

        edges = cv2.Canny(roi, self.config.CANNY_LOW, self.config.CANNY_HIGH)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.config.HOUGH_THRESHOLD,
            minLineLength=self.config.MIN_LINE_LENGTH,
            maxLineGap=self.config.MAX_LINE_GAP
        )

        out = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)

        left_ok = False
        right_ok = False
        mid_x = w / 2.0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                y1 += roi_start
                y2 += roi_start

                if (x2 - x1) == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)

                if slope < -self.config.SLOPE_THRESHOLD and x1 < mid_x and x2 < mid_x:
                    cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    left_ok = True
                elif slope > self.config.SLOPE_THRESHOLD and x1 > mid_x and x2 > mid_x:
                    cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    right_ok = True

        if left_ok and right_ok:
            status = 'OK'
        elif not left_ok and not right_ok:
            status = 'NO_LANE'
        elif not left_ok:
            status = 'DRIFT_LEFT'
        else:
            status = 'DRIFT_RIGHT'

        stable_status = self._stabilize_status(status)

        self.pub_status.publish(String(data=stable_status))
        try:
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(out, encoding='bgr8'))
        except CvBridgeError:
            pass

        if stable_status != 'OK':
            rospy.logwarn_throttle(2, f"Lane: {stable_status}")


if __name__ == '__main__':
    LaneDetectionNode()
    rospy.spin()
