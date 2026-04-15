#!/usr/bin/env python3
"""
Decision Node — liaison lane_viewer → commandes voiture
Subscribe: /lane/status, /lane/deviation, /carla/radar/front, /carla/speed
Publish:   /adas/control, /adas/decision
"""
import rospy
from std_msgs.msg import Float32, String

# ─── Seuils ──────────────────────────────────────────────────────────────────
BRAKE_DIST  = 5.0    # m  → BRAKE
SLOW_DIST   = 12.0   # m  → SLOW
DEV_THRESH  = 0.04   # déviation > 4 % largeur → corriger
LANE_TIMEOUT = 0.6   # s  → considéré perdu si pas de message récent


class DecisionNode:
    def __init__(self):
        rospy.init_node('decision_node')

        # état capteurs
        self.lane_status   = 'NO LANE'
        self.lane_dev      = 0.0
        self.radar_dist    = None
        self.speed         = 0.0
        self.last_lane_t   = rospy.Time(0)
        self.last_radar_t  = rospy.Time(0)

        # subscribers
        rospy.Subscriber('/lane/status',      String,  self._cb_lane_st)
        rospy.Subscriber('/lane/deviation',   Float32, self._cb_lane_dev)
        rospy.Subscriber('/carla/radar/front',Float32, self._cb_radar)
        rospy.Subscriber('/carla/speed',      Float32, self._cb_speed)

        # publishers
        self.pub_ctrl = rospy.Publisher('/adas/control',  String, queue_size=1)
        self.pub_dec  = rospy.Publisher('/adas/decision', String, queue_size=1)

        rospy.Timer(rospy.Duration(0.1), self._decide)
        rospy.loginfo("Decision Node demarre")

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _cb_lane_st(self, msg):
        self.lane_status = msg.data
        self.last_lane_t = rospy.Time.now()

    def _cb_lane_dev(self, msg):
        self.lane_dev = msg.data

    def _cb_radar(self, msg):
        self.radar_dist   = msg.data
        self.last_radar_t = rospy.Time.now()

    def _cb_speed(self, msg):
        self.speed = msg.data

    # ── Logique de décision ───────────────────────────────────────────────────
    def _decide(self, event):
        now          = rospy.Time.now()
        lane_fresh   = (now - self.last_lane_t).to_sec()  < LANE_TIMEOUT
        radar_fresh  = (now - self.last_radar_t).to_sec() < 0.5

        cmd    = 'GO'
        reason = 'Voie libre'

        # 1 — Priorité sécurité : radar
        if radar_fresh and self.radar_dist is not None:
            if self.radar_dist < BRAKE_DIST:
                cmd    = 'BRAKE'
                reason = f'Obstacle {self.radar_dist:.1f}m'
            elif self.radar_dist < SLOW_DIST:
                cmd    = 'SLOW'
                reason = f'Obstacle proche {self.radar_dist:.1f}m'

        # 2 — Maintien de voie (seulement si pas déjà en freinage)
        if cmd == 'GO' and lane_fresh:
            status = self.lane_status

            if status == 'DRIFT LEFT' or self.lane_dev > DEV_THRESH:
                cmd    = 'STEER_RIGHT'
                reason = f'Dérive gauche ({self.lane_dev:+.3f})'

            elif status == 'DRIFT RIGHT' or self.lane_dev < -DEV_THRESH:
                cmd    = 'STEER_LEFT'
                reason = f'Dérive droite ({self.lane_dev:+.3f})'

            elif status == 'NO LANE':
                cmd    = 'SLOW'
                reason = 'Voie perdue — prudence'

            else:  # CENTER OK
                cmd    = 'GO'
                reason = 'Centre OK'

        self.pub_ctrl.publish(String(data=cmd))
        self.pub_dec.publish(String(data=reason))
        rospy.loginfo_throttle(2, f'[DECISION] {cmd} | {reason}')


if __name__ == '__main__':
    DecisionNode()
    rospy.spin()
