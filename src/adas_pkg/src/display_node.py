#!/usr/bin/env python3
"""
Display Node — 2 screens
Screen 1: raw camera + HUD (speed, radar, GPS, decision)
Screen 2: detection + lane overlays
"""
from dataclasses import dataclass
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import pygame
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String, NavSatFix


@dataclass
class DisplayConfig:
    WIDTH: int = 800
    HEIGHT: int = 450
    HUD_W: int = 320
    HUD_H: int = 140
    FPS_TARGET: int = 30
    SHOW_PERCEPTION_WINDOW: bool = True

    CAMERA_TOPIC: str = '/carla/camera/rgb'
    DETECTION_IMAGE_TOPIC: str = '/adas/detection_image'
    LANE_IMAGE_TOPIC: str = '/adas/lane_image'
    SPEED_TOPIC: str = '/carla/speed'
    RADAR_TOPIC: str = '/carla/radar/front'
    DECISION_TOPIC: str = '/adas/decision'
    LANE_STATUS_TOPIC: str = '/adas/lane_status'
    GNSS_TOPIC: str = '/carla/gnss'


class DisplayNode:
    def __init__(self):
        rospy.init_node('display_node')
        self.config = self._load_config()
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.frame_rgb: Optional[np.ndarray] = None
        self.frame_detection: Optional[np.ndarray] = None
        self.frame_lane: Optional[np.ndarray] = None

        self.speed: float = 0.0
        self.radar: Optional[float] = None
        self.decision: str = 'GO'
        self.lane_st: str = 'OK'
        self.gps: Tuple[float, float] = (0.0, 0.0)

        rospy.Subscriber(self.config.CAMERA_TOPIC, Image, self.cb_rgb)
        rospy.Subscriber(self.config.DETECTION_IMAGE_TOPIC, Image, self.cb_detection)
        rospy.Subscriber(self.config.LANE_IMAGE_TOPIC, Image, self.cb_lane)
        rospy.Subscriber(self.config.SPEED_TOPIC, Float32, self.cb_speed)
        rospy.Subscriber(self.config.RADAR_TOPIC, Float32, self.cb_radar)
        rospy.Subscriber(self.config.DECISION_TOPIC, String, self.cb_decision)
        rospy.Subscriber(self.config.LANE_STATUS_TOPIC, String, self.cb_lane_st)
        rospy.Subscriber(self.config.GNSS_TOPIC, NavSatFix, self.cb_gnss)

        rospy.loginfo("Display Node started")

    @staticmethod
    def _load_config() -> DisplayConfig:
        config = DisplayConfig()
        config.WIDTH = rospy.get_param('~width', config.WIDTH)
        config.HEIGHT = rospy.get_param('~height', config.HEIGHT)
        config.HUD_W = rospy.get_param('~hud_w', config.HUD_W)
        config.HUD_H = rospy.get_param('~hud_h', config.HUD_H)
        config.FPS_TARGET = rospy.get_param('~fps_target', config.FPS_TARGET)
        config.SHOW_PERCEPTION_WINDOW = rospy.get_param(
            '~show_perception_window',
            config.SHOW_PERCEPTION_WINDOW
        )
        return config

    def cb_rgb(self, msg):
        self._update_frame('frame_rgb', msg)

    def cb_detection(self, msg):
        self._update_frame('frame_detection', msg)

    def cb_lane(self, msg):
        self._update_frame('frame_lane', msg)

    def cb_speed(self, msg):
        self.speed = msg.data

    def cb_radar(self, msg):
        self.radar = msg.data

    def cb_decision(self, msg):
        self.decision = msg.data

    def cb_lane_st(self, msg):
        self.lane_st = msg.data

    def cb_gnss(self, msg):
        self.gps = (msg.latitude, msg.longitude)

    def _update_frame(self, attr, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                setattr(self, attr, img)
        except CvBridgeError:
            pass

    def _decision_color(self) -> Tuple[int, int, int]:
        d = self.decision
        if d in ('BRAKE', 'STOP'):
            return (255, 50, 50)
        if d in ('SLOW',):
            return (255, 200, 0)
        if d == 'GO':
            return (50, 255, 50)
        return (200, 200, 200)

    def _cv2_to_surface(self, img, size):
        if img is None:
            s = pygame.Surface(size)
            s.fill((30, 30, 30))
            return s
        resized = cv2.resize(img, size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def run(self):
        pygame.init()
        w, h = self.config.WIDTH, self.config.HEIGHT

        screen1 = pygame.display.set_mode((w, h), flags=0)
        pygame.display.set_caption('ADAS - Screen 1 - Driver View')
        font_lg = pygame.font.SysFont('monospace', 20, bold=True)
        font_sm = pygame.font.SysFont('monospace', 16)

        clock = pygame.time.Clock()
        running = True

        if self.config.SHOW_PERCEPTION_WINDOW:
            cv2.namedWindow('ADAS - Screen 2 - Perception', cv2.WINDOW_NORMAL)

        rospy.loginfo("Display started - press Q to quit")

        while running and not rospy.is_shutdown():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            with self.lock:
                rgb = self.frame_rgb
                det = self.frame_detection
                lane = self.frame_lane

            screen1.blit(self._cv2_to_surface(rgb, (w, h)), (0, 0))

            hud = pygame.Surface((self.config.HUD_W, self.config.HUD_H), pygame.SRCALPHA)
            hud.fill((0, 0, 0, 160))
            screen1.blit(hud, (5, 5))

            screen1.blit(
                font_lg.render(f"Speed: {self.speed:.1f} km/h", True, (255, 255, 255)),
                (12, 10)
            )

            if self.radar is not None:
                if self.radar < 5.0:
                    rc = (255, 50, 50)
                elif self.radar < 12.0:
                    rc = (255, 200, 0)
                else:
                    rc = (50, 255, 50)
                radar_text = f"Radar: {self.radar:.1f}m"
            else:
                rc = (50, 255, 50)
                radar_text = "Radar: clear"

            screen1.blit(font_sm.render(radar_text, True, rc), (12, 38))

            screen1.blit(
                font_sm.render(f"GPS: {self.gps[0]:.4f}, {self.gps[1]:.4f}", True, (200, 200, 200)),
                (12, 62)
            )

            lc = (50, 255, 50) if self.lane_st == 'OK' else (255, 200, 0)
            screen1.blit(font_sm.render(f"Lane: {self.lane_st}", True, lc), (12, 86))

            dc = self._decision_color()
            screen1.blit(font_lg.render(f"Action: {self.decision}", True, dc), (12, 110))

            fps = clock.get_fps()
            screen1.blit(font_sm.render(f"FPS: {fps:.1f}", True, (180, 180, 180)), (12, 130))

            pygame.display.flip()

            if self.config.SHOW_PERCEPTION_WINDOW and (det is not None or lane is not None):
                h2, w2 = h, w
                screen2 = np.zeros((h2, w2, 3), dtype=np.uint8)

                if det is not None:
                    left = cv2.resize(det, (w2 // 2, h2))
                    screen2[:, :w2 // 2] = left

                if lane is not None:
                    right = cv2.resize(lane, (w2 // 2, h2))
                    screen2[:, w2 // 2:] = right

                cv2.putText(screen2, "Object detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(screen2, "Lane detection", (w2 // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.line(screen2, (w2 // 2, 0), (w2 // 2, h2), (255, 255, 255), 1)

                cv2.imshow('ADAS - Screen 2 - Perception', screen2)
                cv2.waitKey(1)

            clock.tick(self.config.FPS_TARGET)

        pygame.quit()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    node = DisplayNode()
    node.run()
