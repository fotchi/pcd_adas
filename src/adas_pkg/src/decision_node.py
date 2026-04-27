#!/usr/bin/env python3
"""
decision_node.py — Logique de décision ADAS
Souscrit : /lane/status       (String)
           /lane/deviation    (Float32)
           /carla/radar/front (Float32)
           /carla/speed       (Float32)
Publie   : /adas/control      (String)  → GO | SLOW | BRAKE | STEER_LEFT | STEER_RIGHT
           /adas/decision     (String)  → raison humaine
"""

import rospy
from std_msgs.msg import Float32, String

# ─── Seuils de base ───────────────────────────────────────────────────────────
BRAKE_DIST_BASE = 5.0    # m — freinage d'urgence (à vitesse faible)
SLOW_DIST_BASE  = 12.0   # m — décélération préventive
DEV_THRESH      = 0.04   # déviation > 4 % de la largeur → correction
DEV_THRESH_EXIT = 0.02   # hystérésis pour éviter les bascules
DEV_TARGET      = -0.050 # cible latérale: légèrement à droite du centre de voie (HLS détecte le vrai centre)
DEV_GAIN_RIGHT  = 1.60   # amplification déviation côté droite (virage plus facile)
DEV_GAIN_LEFT   = 1.15   # amplification déviation côté gauche (reste plus prudente)
ERR_RIGHT_ON    = 0.008  # déclenchement STEER_RIGHT (encore plus précoce)
ERR_LEFT_ON     = 0.090  # déclenchement STEER_LEFT (plus strict pour éviter sens inverse)
ERR_EXIT        = 0.018  # sortie de correction autour de la cible
STEER_HOLD_TICKS = 7     # maintien un peu plus long de la correction en virage
LEFT_CONFIRM_TICKS = 8   # confirmations DRIFT_LEFT nécessaires avant STEER_LEFT (résiste aux intersections)
LEFT_CONFIRM_DEV   = -0.14  # dev mini pour valider un vrai drift gauche
CURVE_SLOW_ERR     = 0.18   # si erreur latérale forte, ralentir dans le virage
LANE_TIMEOUT    = 0.6    # s — voie considérée perdue si aucun message récent
RADAR_TIMEOUT   = 0.5    # s — radar considéré invalide si trop vieux
DET_TIMEOUT     = 1.0    # s — détection objet considérée périmée
STOP_HOLD_TICKS = 20     # ticks d'arrêt complet après stop sign (~2 s)
TURN_INTENT_TICKS = 40  # ticks de mémorisation du panneau directionnel (~4 s)
DEV_SMOOTH      = 0.30   # lissage léger — la déviation est déjà lissée dans lane_viewer
DEV_FLIP_THRESH = 0.10   # inversion de signe au-delà de ce seuil → protection sens inverse
DEV_FLIP_GUARD  = 8      # ticks de ralentissement après détection d'inversion suspecte
SPEED_LIMIT_HOLD_TICKS = 90   # maintien limite vitesse (~9 s à 10 Hz)
SPEED_LIMIT_BAND_KMH   = 0.8  # marge autour de la cible


class DecisionNode:

    def __init__(self):
        rospy.init_node('decision_node')

        # ── État capteurs ─────────────────────────────────────────────────────
        self.lane_status  = 'NO LANE'
        self.lane_dev     = 0.0
        self.lane_dev_f   = 0.0
        self.radar_dist   = None
        self.speed        = 0.0           # km/h
        self.last_lane_t  = rospy.Time(0)
        self.last_radar_t = rospy.Time(0)
        self._steer_state = 0             # -1 left, 0 none, +1 right
        self._hold_ticks  = 0
        self._left_confirm = 0
        self._dev_sign_prev    = 0
        self._flip_guard_ticks = 0

        # ── État détection objets ─────────────────────────────────────────────
        self.det_light_color  = None    # 'red' | 'yellow' | 'green' | None
        self.det_stop_sign    = False
        self.det_speed_limit  = False
        self.det_speed_kmh    = None
        self.det_turn_right   = False
        self.det_turn_left    = False
        self.last_det_t       = rospy.Time(0)
        self._stop_hold_ticks = 0       # ticks d'arrêt complet restants
        self._turn_intent     = None    # 'right' | 'left' | None — mémorisé après le panneau
        self._turn_intent_ticks = 0     # décompte avant oubli de l'intention
        self._speed_limit_ticks = 0
        self._speed_limit_target_kmh = None

        # ── Subscribers ───────────────────────────────────────────────────────
        rospy.Subscriber('/lane/status',       String,  self._cb_lane_st)
        rospy.Subscriber('/lane/deviation',    Float32, self._cb_lane_dev)
        rospy.Subscriber('/carla/radar/front', Float32, self._cb_radar)
        rospy.Subscriber('/carla/speed',       Float32, self._cb_speed)
        rospy.Subscriber('/adas/detection',    String,  self._cb_detection)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_ctrl = rospy.Publisher('/adas/control',  String, queue_size=1)
        self.pub_dec  = rospy.Publisher('/adas/decision', String, queue_size=1)

        rospy.Timer(rospy.Duration(0.1), self._decide)
        rospy.loginfo("Decision Node démarré")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_lane_st(self, msg):
        self.lane_status = msg.data
        self.last_lane_t = rospy.Time.now()

    def _cb_lane_dev(self, msg):
        self.lane_dev = msg.data
        self.lane_dev_f = DEV_SMOOTH * self.lane_dev_f + (1.0 - DEV_SMOOTH) * float(msg.data)

    def _cb_radar(self, msg):
        self.radar_dist   = msg.data
        self.last_radar_t = rospy.Time.now()

    def _cb_speed(self, msg):
        self.speed = msg.data   # km/h

    def _cb_detection(self, msg):
        """Parse: label:detail:conf:x1:y1:x2:y2 | ..."""
        if msg.data == 'none':
            return

        found_any = False
        det_light_color = None
        det_stop_sign   = False
        det_speed_limit = False
        det_speed_kmh   = None
        det_turn_right  = False
        det_turn_left   = False

        prio_map = {'red': 3, 'rouge': 3, 'yellow': 2, 'jaune': 2, 'green': 1, 'vert': 1}

        for token in msg.data.split('|'):
            parts = token.split(':')
            if len(parts) < 2:
                continue
            label, detail = parts[0].strip(), parts[1].strip()
            found_any = True

            # ── Feux : classes génériques (couleur dans detail) ──────────────
            if label == 'feu':
                color = detail  # 'red' | 'yellow' | 'green' | 'unknown'
                if color in prio_map:
                    if (det_light_color is None or
                            prio_map.get(color, 0) > prio_map.get(det_light_color, 0)):
                        det_light_color = color

            # ── Feux : classes natives du modèle (rouge/vert/jaune) ──────────
            elif label == 'feu_rouge':
                det_light_color = 'red'
            elif label == 'feu_vert':
                if det_light_color not in ('red',):  # rouge reste prioritaire
                    det_light_color = 'green'
            elif label == 'feu_jaune':
                if det_light_color not in ('red',):
                    det_light_color = 'yellow'

            # ── Panneaux ─────────────────────────────────────────────────────
            elif label == 'stop':
                det_stop_sign = True
            elif label == 'vitesse':
                det_speed_limit = True
                try:
                    v = float(detail.replace(',', '.'))
                    if v > 0:
                        det_speed_kmh = v
                except Exception:
                    det_speed_kmh = None

            # ── Panneaux directionnels : mémoriser l'intention de virage ─────
            elif label == 'droite':
                det_turn_right     = True
                self._turn_intent      = 'right'
                self._turn_intent_ticks = TURN_INTENT_TICKS
            elif label == 'gauche':
                det_turn_left      = True
                self._turn_intent      = 'left'
                self._turn_intent_ticks = TURN_INTENT_TICKS

        if not found_any:
            return

        self.last_det_t      = rospy.Time.now()
        self.det_light_color = det_light_color
        self.det_stop_sign   = det_stop_sign
        self.det_speed_limit = det_speed_limit
        self.det_speed_kmh   = det_speed_kmh
        self.det_turn_right  = det_turn_right
        self.det_turn_left   = det_turn_left

    # ── Logique de décision ───────────────────────────────────────────────────

    def _decide(self, _event):  # timer callback — event non utilisé
        now         = rospy.Time.now()
        lane_fresh  = (now - self.last_lane_t).to_sec()  < LANE_TIMEOUT
        radar_fresh = (now - self.last_radar_t).to_sec() < RADAR_TIMEOUT

        # Distances adaptatives : plus la vitesse est élevée, plus on freine tôt
        # Règle : 0.5 s de marge à n'importe quelle vitesse (v en m/s = km/h / 3.6)
        spd_ms      = self.speed / 3.6
        brake_dist  = max(BRAKE_DIST_BASE, spd_ms * 0.5)
        slow_dist   = max(SLOW_DIST_BASE,  spd_ms * 1.5)

        cmd    = 'GO'
        reason = 'Voie libre'
        det_fresh = (now - self.last_det_t).to_sec() < DET_TIMEOUT

        # ── 0. Signalisation routière (priorité absolue) ──────────────────────
        if det_fresh:
            # Stop sign : arrêt complet puis redémarrage
            if self.det_stop_sign and self._stop_hold_ticks == 0:
                self._stop_hold_ticks = STOP_HOLD_TICKS

            if self._stop_hold_ticks > 0:
                self._stop_hold_ticks -= 1
                cmd    = 'BRAKE'
                reason = f'STOP sign — arrêt ({self._stop_hold_ticks} ticks restants)'

            # Feu rouge : freinage complet
            elif self.det_light_color == 'red':
                cmd    = 'BRAKE'
                reason = 'Feu ROUGE — freinage'

            # Feu orange : ralenti
            elif self.det_light_color == 'yellow':
                cmd    = 'SLOW'
                reason = 'Feu ORANGE — ralenti'

            # Limitation vitesse : ralenti préventif
            elif self.det_speed_limit and cmd == 'GO':
                target = self.det_speed_kmh if self.det_speed_kmh is not None else 10.0
                self._speed_limit_target_kmh = max(5.0, min(90.0, target))
                self._speed_limit_ticks = SPEED_LIMIT_HOLD_TICKS

        # Maintien de la limitation vitesse même si la détection devient intermittente.
        if self._speed_limit_ticks > 0 and self._speed_limit_target_kmh is not None and cmd != 'BRAKE':
            self._speed_limit_ticks -= 1
            target = self._speed_limit_target_kmh
            if self.speed > target + 4.0:
                cmd = 'BRAKE'
                reason = f'Limite {target:.0f} km/h — freinage (v={self.speed:.1f})'
            elif self.speed > target + SPEED_LIMIT_BAND_KMH:
                cmd = 'LIMIT_10' if target <= 10.5 else 'SLOW'
                reason = f'Limite {target:.0f} km/h — réduction vitesse (v={self.speed:.1f})'
            elif self.speed < target - SPEED_LIMIT_BAND_KMH:
                cmd = 'GO'
                reason = f'Limite {target:.0f} km/h — accélération contrôlée (v={self.speed:.1f})'
            else:
                cmd = 'LIMIT_10' if target <= 10.5 else 'SLOW'
                reason = f'Limite {target:.0f} km/h — maintien (v={self.speed:.1f})'
        elif self._speed_limit_ticks == 0:
            self._speed_limit_target_kmh = None

        # ── 1. Priorité sécurité : radar ─────────────────────────────────────
        if radar_fresh and self.radar_dist is not None:
            if self.radar_dist < brake_dist:
                cmd    = 'BRAKE'
                reason = f'Obstacle {self.radar_dist:.1f} m — FREINAGE'
            elif self.radar_dist < slow_dist:
                cmd    = 'SLOW'
                reason = f'Obstacle proche {self.radar_dist:.1f} m — RALENTI'

        # ── 1b. Protection sens inverse — inversion brutale de déviation ────────
        # Si la déviation change de signe brusquement (ex. voie adverse détectée
        # en virage), on ralentit quelques ticks sans corriger le cap.
        if lane_fresh and self._flip_guard_ticks == 0:
            dev_sign = (1 if self.lane_dev_f > DEV_FLIP_THRESH
                        else -1 if self.lane_dev_f < -DEV_FLIP_THRESH else 0)
            if (self._dev_sign_prev != 0
                    and dev_sign != 0
                    and dev_sign != self._dev_sign_prev):
                self._flip_guard_ticks = DEV_FLIP_GUARD
                self._steer_state = 0
                self._hold_ticks  = 0
            if dev_sign != 0:
                self._dev_sign_prev = dev_sign

        if self._flip_guard_ticks > 0:
            self._flip_guard_ticks -= 1
            if cmd == 'GO':
                cmd    = 'SLOW'
                reason = 'Inversion déviation suspecte — maintien sens voie, ralenti'

        # ── 2. Maintien de voie (si pas de freinage en cours) ────────────────
        if cmd == 'GO' and lane_fresh:
            status = self.lane_status
            dev = self.lane_dev_f
            err = dev - DEV_TARGET
            err_eff = err * (DEV_GAIN_RIGHT if err >= 0.0 else DEV_GAIN_LEFT)

            if status == 'NO_LANE':
                # Décompter l'intention de virage à chaque tick
                if self._turn_intent_ticks > 0:
                    self._turn_intent_ticks -= 1
                    if self._turn_intent == 'right':
                        self._steer_state = 1
                        cmd    = 'STEER_RIGHT'
                        reason = f'NO_LANE + panneau DROITE → virage droite'
                    elif self._turn_intent == 'left':
                        self._steer_state = -1
                        cmd    = 'STEER_LEFT'
                        reason = f'NO_LANE + panneau GAUCHE → virage gauche'
                    else:
                        cmd    = 'SLOW'
                        reason = 'Voie perdue — SLOW + cap maintenu'
                else:
                    self._turn_intent = None
                    cmd    = 'SLOW'
                    reason = 'Voie perdue — SLOW + cap maintenu (re-acquisition en cours)'
            else:
                # Contrôle asymétrique autour d'une cible à droite.
                if dev < LEFT_CONFIRM_DEV and err_eff < -ERR_LEFT_ON:
                    self._left_confirm = min(self._left_confirm + 1, LEFT_CONFIRM_TICKS + 2)
                else:
                    self._left_confirm = max(0, self._left_confirm - 1)

                # Pilotage uniquement basé sur l'erreur à la cible décalée à droite.
                if err_eff > ERR_RIGHT_ON:
                    self._steer_state = 1
                    self._hold_ticks = STEER_HOLD_TICKS + 2
                    self._left_confirm = 0
                elif self._left_confirm >= LEFT_CONFIRM_TICKS:
                    self._steer_state = -1
                    self._hold_ticks = STEER_HOLD_TICKS
                elif abs(err_eff) < ERR_EXIT:
                    self._steer_state = 0
                    self._hold_ticks = 0
                    self._left_confirm = 0
                elif self._steer_state != 0 and self._hold_ticks > 0:
                    self._hold_ticks -= 1

                if self._steer_state > 0:
                    cmd    = 'STEER_RIGHT'
                    reason = f'Offset droite dev={dev:+.3f} cible={DEV_TARGET:+.3f} → correction droite'
                elif self._steer_state < 0:
                    cmd    = 'STEER_LEFT'
                    reason = f'Offset droite dev={dev:+.3f} cible={DEV_TARGET:+.3f} → correction gauche'

                else:   # CENTER OK
                    cmd    = 'GO'
                    reason = f'Offset droite OK — {self.speed:.1f} km/h'

                # En virage prononcé, privilégier la stabilité en réduisant la vitesse.
                if abs(err_eff) > CURVE_SLOW_ERR and cmd != 'BRAKE':
                    cmd = 'SLOW'
                    reason = f'Virage serré err={err_eff:+.3f} — ralenti stabilité'

        self.pub_ctrl.publish(String(data=cmd))
        self.pub_dec.publish(String(data=reason))
        rospy.loginfo_throttle(2, f'[DECISION] {cmd:12s} | {reason}')


if __name__ == '__main__':
    DecisionNode()
    rospy.spin()
