#!/usr/bin/env python3
"""
Radar Bridge Node
Convertit le topic radar CARLA (PointCloud2) en distance minimale (Float32).
Subscribe: /carla/ego_vehicle/radar_front  (sensor_msgs/PointCloud2)
Publish:   /carla/radar/front              (std_msgs/Float32, distance en mètres)
"""
import struct

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32

# Champs CARLA radar PointCloud2: x, y, z, Velocity, Azimuth, Altitude, depth
# On calcule la distance 3D depuis x, y, z
FIELD_X = 0
FLOAT_SIZE = 4
POINT_STEP = 7 * FLOAT_SIZE  # 7 champs float32


def extract_min_distance(msg: PointCloud2) -> float:
    """Retourne la distance minimale (m) parmi tous les points radar."""
    if msg.width == 0 or msg.row_step == 0:
        return float('inf')

    step = msg.point_step
    data = msg.data
    min_dist = float('inf')

    try:
        for i in range(msg.width * msg.height):
            offset = i * step
            x, y, z = struct.unpack_from('fff', data, offset)
            dist = (x * x + y * y + z * z) ** 0.5
            if dist < min_dist:
                min_dist = dist
    except struct.error:
        pass

    return min_dist


class RadarBridgeNode:
    def __init__(self):
        rospy.init_node('radar_bridge_node')
        self.pub = rospy.Publisher('/carla/radar/front', Float32, queue_size=1)
        rospy.Subscriber(
            '/carla/ego_vehicle/radar_front',
            PointCloud2,
            self.callback,
            queue_size=1
        )
        rospy.loginfo("Radar Bridge Node demarre")

    def callback(self, msg: PointCloud2):
        dist = extract_min_distance(msg)
        if dist != float('inf'):
            self.pub.publish(Float32(data=dist))


if __name__ == '__main__':
    RadarBridgeNode()
    rospy.spin()
