#!/usr/bin/env python3
"""
decision_node.py — Logique de décision ADAS
Souscrit : /lane/status       (String)
           /lane/deviation    (Float32)
           /carla/radar/front (Float32)
           /carla/speed       (Float32)
           /adas/detection    (String)
Publie   : /adas/control      (String)  → GO|SLOW|BRAKE|LIMIT_10|LIMIT_20|STEER_*|TURN_RIGHT
           /adas/decision     (String)  → raison lisible

Format /adas/detection : label:detail:conf:x1:y1:x2:y2  (pipes pour plusieurs)
  Exemples :
    stop:stop:0.99:200:30:440:240
    vitesse:20:0.95:200:30:440:240
    vitesse:10:0.95:200:30:440:240
    feu_rouge:red:0.97:280:60:400:200
    feu_vert:green:0.97:280:60:400:200
    droite:right:0.95:200:40:440:220
    gauche:left:0.95:200:40:440:220
"""

import rospy
from collections import deque
from std_msgs.msg import Float32, String

# ─── Seuils de base ───────────────────────────────────────────────────────────
BRAKE_DIST_BASE = 5.0      # m — freinage d'urgence
SLOW_DIST_BASE  = 12.0     # m — ralentissement préventif

DEV_TARGET      = -0.050   # cible: légèrement à droite du centre
DEV_GAIN_RIGHT  = 1.60
DEV_GAIN_LEFT   = 1.15
ERR_RIGHT_ON    = 0.008
ERR_LEFT_ON     = 0.090
ERR_EXIT        = 0.018
STEER_HOLD_TICKS   = 7
LEFT_CONFIRM_TICKS = 8
LEFT_CONFIRM_DEV   = -0.14
CURVE_SLOW_ERR     = 0.18

LANE_TIMEOUT    = 0.6
RADAR_TIMEOUT   = 0.5
DET_TIMEOUT     = 1.2      # +0.2 s de tolérance vs version précédente

STOP_HOLD_TICKS      = 50   # safety max (5 s) — rarement atteint
STOP_SPEED_KMH       = 2.0  # vitesse "arrêt complet" en km/h
STOP_REST_TICKS      = 10   # ticks à l'arrêt avant relâche (1 s)
TURN_INTENT_TICKS    = 200  # ~20 s — fenêtre pour atteindre l'intersection
TURN_NO_LANE_CONFIRM     = 20   # attendre 2 s après le 1er NO_LANE avant de déclencher
TURN_SIGN_WINDOW         = 12   # fenêtre glissante de détection panneau (1.2 s)
TURN_SIGN_CONFIRM        = 3    # détections requises dans la fenêtre (tolère l'intermittence)
TURN_MIN_APPROACH_TICKS  = 8    # délai (0.8 s) après armement — juste le temps de quitter le panneau
# ── Machine d'état virage 4 phases ───────────────────────────────────────────
TURN_BRAKE_TICKS     = 60   # phase 1a : freinage (6 s max) — sort dès v < TURN_SPEED_KMH
TURN_STOP_TICKS      = 8    # phase 1b : tenu à l'arrêt (0.8 s) — confirmation arrêt complet
TURN_RIGHT_TICKS     = 80   # phase 2  : ticks de rotation 90° (8 s max, sortie anticipée si voie détectée)
TURN_SEARCH_TICKS    = 20   # phase 3  : marche avant 2 s puis reprise
TURN_SPEED_KMH       = 1.5  # vitesse quasi-nulle pour garantir rayon serré
DEV_SMOOTH           = 0.30
DEV_FLIP_THRESH      = 0.10
DEV_FLIP_GUARD       = 8

# Nombre de ticks de maintien de la limitation vitesse
# 90 ticks × 0.1 s = 9 s de maintien même si la détection devient intermittente
SPEED_LIMIT_HOLD_TICKS = 90
SPEED_LIMIT_BAND_KMH   = 1.5   # marge autour de la cible (±1.5 km/h = neutre)


class DecisionNode:

    def __init__(self):
        rospy.init_node('decision_node')

        # ── État capteurs ─────────────────────────────────────────────────────
        self.lane_status  = 'NO LANE'
        self.lane_dev     = 0.0
        self.lane_dev_f   = 0.0
        self.radar_dist   = None
        self.speed        = 0.0
        self.last_lane_t  = rospy.Time(0)
        self.last_radar_t = rospy.Time(0)

        # ── État maintien de voie ─────────────────────────────────────────────
        self._steer_state     = 0
        self._hold_ticks      = 0
        self._left_confirm    = 0
        self._dev_sign_prev   = 0
        self._flip_guard_ticks = 0

        # ── État détection objets ─────────────────────────────────────────────
        self.det_light_color  = None
        self.det_stop_sign    = False
        self.det_speed_limit  = False
        self.det_speed_kmh    = None
        self.det_turn_right   = False
        self.det_turn_left    = False
        self.last_det_t       = rospy.Time(0)

        # ── État interne commandes ────────────────────────────────────────────
        self._stop_hold_ticks       = 0
        self._stop_rest_ticks       = 0   # ticks consécutifs à vitesse ≈ 0
        self._turn_intent           = None
        self._turn_intent_ticks     = 0
        self._turn_pending          = False
        self._turn_no_lane_ticks    = 0
        self._turn_approach_ticks   = 0   # délai d'approche après armement (bloque NO_LANE prématuré)
        self._turn_sign_count       = 0   # conservé pour compatibilité
        self._turn_sign_hist        = deque(maxlen=TURN_SIGN_WINDOW)   # fenêtre glissante
        # Machine d'état virage : None | 'brake' | 'exec' | 'search'
        self._turn_state            = None
        self._turn_state_ticks      = 0
        self._turn_stop_hold        = 0   # ticks consécutifs à l'arrêt complet
        self._speed_limit_ticks     = 0
        self._speed_limit_target    = None   # km/h

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
        rospy.loginfo('Decision Node démarré')

    # ─── Callbacks ────────────────────────────────────────────────────────────

    def _cb_lane_st(self, msg):
        self.lane_status = msg.data
        self.last_lane_t = rospy.Time.now()

    def _cb_lane_dev(self, msg):
        self.lane_dev   = msg.data
        self.lane_dev_f = DEV_SMOOTH * self.lane_dev_f + (1.0 - DEV_SMOOTH) * float(msg.data)

    def _cb_radar(self, msg):
        self.radar_dist   = msg.data
        self.last_radar_t = rospy.Time.now()

    def _cb_speed(self, msg):
        self.speed = msg.data

    def _cb_detection(self, msg):
        """
        Parse le message de détection.
        Format : label:detail:conf:x1:y1:x2:y2 | ...
        Labels gérés :
          stop, vitesse, feu_rouge, feu_vert, feu_jaune, feu,
          droite, gauche
        """
        if msg.data == 'none':
            return

        prio = {'red': 3, 'rouge': 3, 'yellow': 2, 'jaune': 2, 'green': 1, 'vert': 1}

        det_light  = None
        det_stop   = False
        det_speed  = False
        det_kmh    = None
        det_right  = False
        det_left   = False
        found      = False

        for token in msg.data.split('|'):
            parts = token.strip().split(':')
            if len(parts) < 2:
                continue
            label  = parts[0].strip().lower()
            detail = parts[1].strip().lower()
            found  = True

            # Feux
            if label == 'feu':
                if detail in prio and (det_light is None or
                        prio[detail] > prio.get(det_light, 0)):
                    det_light = detail

            elif label == 'feu_rouge':
                det_light = 'red'

            elif label == 'feu_vert':
                if det_light != 'red':
                    det_light = 'green'

            elif label == 'feu_jaune':
                if det_light not in ('red',):
                    det_light = 'yellow'

            # Stop
            elif label == 'stop':
                det_stop = True

            # Vitesse (detail = valeur numérique en km/h)
            elif label == 'vitesse':
                det_speed = True
                try:
                    v = float(detail.replace(',', '.'))
                    if v > 0:
                        det_kmh = v
                except Exception:
                    pass

            # Panneaux directionnels
            elif label == 'droite':
                det_right = True
                self._turn_intent       = 'right'
                self._turn_intent_ticks = TURN_INTENT_TICKS
                # L'armement de _turn_pending se fait dans _decide
                # après TURN_SIGN_CONFIRM ticks consécutifs de détection (anti faux positifs)

            elif label == 'gauche':
                det_left = True
                self._turn_intent       = 'left'
                self._turn_intent_ticks = TURN_INTENT_TICKS

        if not found:
            return

        self.last_det_t      = rospy.Time.now()
        self.det_light_color = det_light
        self.det_stop_sign   = det_stop
        self.det_speed_limit = det_speed
        self.det_speed_kmh   = det_kmh
        self.det_turn_right  = det_right
        self.det_turn_left   = det_left

    # ─── Logique de décision (timer 10 Hz) ───────────────────────────────────

    def _decide(self, _):
        now = rospy.Time.now()

        lane_fresh  = (now - self.last_lane_t).to_sec()  < LANE_TIMEOUT
        radar_fresh = (now - self.last_radar_t).to_sec() < RADAR_TIMEOUT
        det_fresh   = (now - self.last_det_t).to_sec()   < DET_TIMEOUT

        spd_ms     = self.speed / 3.6
        brake_dist = max(BRAKE_DIST_BASE, spd_ms * 0.5)
        slow_dist  = max(SLOW_DIST_BASE,  spd_ms * 1.5)

        cmd    = 'GO'
        reason = 'Voie libre'

        if self._turn_intent_ticks > 0:
            self._turn_intent_ticks -= 1
            if self._turn_intent_ticks == 0:
                self._turn_intent = None
                self._turn_pending = False
                self._turn_no_lane_ticks = 0
                self._turn_sign_count = 0
                self._turn_sign_hist.clear()
                if self._turn_state is None:   # ne pas interrompre un virage en cours
                    self._turn_state = None
                    self._turn_state_ticks = 0

        # ════════════════════════════════════════════════════════════════════
        #  PRIORITÉ 0 — Signalisation routière
        # ════════════════════════════════════════════════════════════════════

        # ── Stop sign : armement (une seule fois par panneau) ────────────────
        # L'armement exige det_fresh ; la séquence de freinage tourne ensuite
        # de façon autonome jusqu'à l'arrêt effectif, indépendamment de det_fresh.
        if det_fresh and self.det_stop_sign and self._stop_hold_ticks == 0:
            self._stop_hold_ticks = STOP_HOLD_TICKS
            self._stop_rest_ticks = 0
            rospy.loginfo('[DECISION] STOP armé')

        if self._stop_hold_ticks > 0:
            self._stop_hold_ticks -= 1

            if self.speed <= STOP_SPEED_KMH:
                # Phase 2 : véhicule arrêté → compteur de repos
                self._stop_rest_ticks += 1
            else:
                # Phase 1 : encore en mouvement → freinage
                self._stop_rest_ticks = 0

            if self._stop_rest_ticks >= STOP_REST_TICKS:
                # Arrêt complet tenu 1 s → séquence terminée, relâche
                self._stop_hold_ticks = 0
                self._stop_rest_ticks = 0
                rospy.loginfo('[DECISION] STOP terminé — reprise')
            else:
                phase = ('arrêt complet' if self.speed <= STOP_SPEED_KMH
                         else 'freinage')
                cmd    = 'BRAKE'
                reason = (f'STOP sign — {phase} '
                          f'({self._stop_rest_ticks}/{STOP_REST_TICKS} repos, '
                          f'v={self.speed:.1f})')

        if det_fresh and self._stop_hold_ticks == 0:
            # ── Feu rouge ─────────────────────────────────────────────────────
            if self.det_light_color == 'red':
                cmd    = 'BRAKE'
                reason = 'Feu ROUGE — freinage prioritaire'

           

            # ── Feu vert : autoriser GO et décision ──────────────────────────
            elif self.det_light_color == 'green':
                cmd    = 'GO'
                reason = 'Feu VERT — passage autorisé'

            # ── Limitation vitesse : armer le régulateur ──────────────────────
            elif self.det_speed_limit:
                target = self.det_speed_kmh if self.det_speed_kmh is not None else 20.0
                # Limite stricte: ne jamais depasser 15 km/h quand un panneau est detecte
                target = min(target, 15.0)
                target = max(5.0, min(90.0, target))
                # N'armer que si la nouvelle limite est différente ou si le régulateur est inactif
                if self._speed_limit_ticks == 0 or abs(target - (self._speed_limit_target or 0)) > 1.0:
                    self._speed_limit_target = target
                    self._speed_limit_ticks  = SPEED_LIMIT_HOLD_TICKS
                    rospy.loginfo(f'[DET] Limite armée : {target:.0f} km/h')

        # ── Maintien de la limitation vitesse (survit à l'intermittence) ─────
        if self._speed_limit_ticks > 0 and self._speed_limit_target is not None:
            self._speed_limit_ticks -= 1
            target = self._speed_limit_target

            if cmd not in ('BRAKE', 'SLOW'):   # ne pas écraser une décision sécurité
                # Freinage plus agressif pour limite 10 km/h
                if target <= 10.5:
                    # Limite 10 : freinage strict
                    if self.speed > target + 6.0:
                        # Loin au-dessus → freinage fort
                        cmd    = 'BRAKE'
                        reason = f'Limite {target:.0f} km/h — FREINAGE FORT (v={self.speed:.1f})'
                    elif self.speed > target + SPEED_LIMIT_BAND_KMH:
                        # Au-dessus → régulateur stricte
                        cmd    = 'LIMIT_10'
                        reason = f'Limite {target:.0f} km/h — réduction stricte (v={self.speed:.1f})'
                    elif self.speed < target - SPEED_LIMIT_BAND_KMH:
                        # En dessous → accélération libre (GO)
                        cmd    = 'GO'
                        reason = f'Limite {target:.0f} km/h — sous limite, GO (v={self.speed:.1f})'
                    else:
                        # Dans la bande → maintien régulateur
                        cmd    = 'LIMIT_10'
                        reason = f'Limite {target:.0f} km/h — maintien stricte (v={self.speed:.1f})'
                else:
                    # Limite 20+ : logique standard
                    if self.speed > target + 4.0:
                        cmd    = 'BRAKE'
                        reason = f'Limite {target:.0f} km/h — freinage (v={self.speed:.1f})'
                    elif self.speed > target + SPEED_LIMIT_BAND_KMH:
                        cmd    = 'LIMIT_20'
                        reason = f'Limite {target:.0f} km/h — réduction (v={self.speed:.1f})'
                    elif self.speed < target - SPEED_LIMIT_BAND_KMH:
                        cmd    = 'GO'
                        reason = f'Limite {target:.0f} km/h — sous limite, GO (v={self.speed:.1f})'
                    else:
                        cmd    = 'LIMIT_20'
                        reason = f'Limite {target:.0f} km/h — maintien (v={self.speed:.1f})'

        elif self._speed_limit_ticks == 0:
            self._speed_limit_target = None   # expiration → oublier la limite

        # ════════════════════════════════════════════════════════════════════
        #  PRIORITÉ 1 — Sécurité radar (si pas de feu rouge)
        # ════════════════════════════════════════════════════════════════════
        if radar_fresh and self.radar_dist is not None and cmd != 'BRAKE':
            if self.radar_dist < brake_dist:
                cmd    = 'BRAKE'
                reason = f'Obstacle {self.radar_dist:.1f} m — FREINAGE'
            elif self.radar_dist < slow_dist and cmd not in ('BRAKE',):
                cmd    = 'SLOW'
                reason = f'Obstacle proche {self.radar_dist:.1f} m — RALENTI'

        # ════════════════════════════════════════════════════════════════════
        #  PRIORITÉ 1b — Protection sens inverse (inversion déviation)
        # ════════════════════════════════════════════════════════════════════
        if lane_fresh and self._flip_guard_ticks == 0:
            sign = (1 if self.lane_dev_f >  DEV_FLIP_THRESH else
                   -1 if self.lane_dev_f < -DEV_FLIP_THRESH else 0)
            if self._dev_sign_prev != 0 and sign != 0 and sign != self._dev_sign_prev:
                self._flip_guard_ticks = DEV_FLIP_GUARD
                self._steer_state = 0
                self._hold_ticks  = 0
            if sign != 0:
                self._dev_sign_prev = sign

        if self._flip_guard_ticks > 0:
            self._flip_guard_ticks -= 1
            if cmd == 'GO':
                cmd    = 'SLOW'
                reason = 'Inversion déviation — ralenti, maintien sens voie'

        # ════════════════════════════════════════════════════════════════════
        #  Confirmation panneau DROITE — anti faux positifs
        # ════════════════════════════════════════════════════════════════════
        # Fenêtre glissante — tolère la détection YOLO intermittente
        on_lane = self.lane_status not in ('NO_LANE', 'NO LANE')
        self._turn_sign_hist.append(
            1 if (det_fresh and self.det_turn_right and on_lane) else 0)
        sign_score = sum(self._turn_sign_hist)

        if (sign_score >= TURN_SIGN_CONFIRM
                and not self._turn_pending
                and self._turn_state is None):
            self._turn_pending        = True
            self._turn_no_lane_ticks  = 0
            self._turn_approach_ticks = TURN_MIN_APPROACH_TICKS
            rospy.loginfo(f'[DECISION] Panneau DROITE confirmé '
                          f'(score {sign_score}/{TURN_SIGN_WINDOW}) → virage armé')

        # ════════════════════════════════════════════════════════════════════
        #  PRIORITÉ 2 — Maintien de voie
        # ════════════════════════════════════════════════════════════════════
        turn_active = False
        if lane_fresh:

            # ── Phase 1 : freinage + arrêt complet tenu ──────────────────────────
            if self._turn_state == 'brake':
                self._turn_state_ticks -= 1
                cmd    = 'BRAKE'
                turn_active = True

                if self.speed <= TURN_SPEED_KMH:
                    self._turn_stop_hold += 1
                else:
                    self._turn_stop_hold = 0   # relâchement → recommencer le comptage

                if self._turn_stop_hold >= TURN_STOP_TICKS:
                    # Arrêt complet confirmé → lancer la rotation
                    self._turn_state       = 'exec'
                    self._turn_state_ticks = TURN_RIGHT_TICKS
                    self._turn_stop_hold   = 0
                    reason = f'Arrêt complet — rotation démarrée'
                    rospy.loginfo(f'[DECISION] Arrêt confirmé (v={self.speed:.1f}) → rotation')
                elif self._turn_state_ticks <= 0:
                    # Safety timeout
                    self._turn_state       = 'exec'
                    self._turn_state_ticks = TURN_RIGHT_TICKS
                    self._turn_stop_hold   = 0
                    reason = f'Pré-virage — timeout, rotation forcée (v={self.speed:.1f})'
                    rospy.logwarn(f'[DECISION] Timeout freinage (v={self.speed:.1f}) → rotation forcée')
                else:
                    reason = (f'Arrêt complet — freinage ({self._turn_stop_hold}/{TURN_STOP_TICKS} ticks)'
                              if self.speed <= TURN_SPEED_KMH
                              else f'Pré-virage — freinage (v={self.speed:.1f} km/h)')

            # ── Phase 2 : rotation 90° — toujours TURN_RIGHT, le contrôleur gère la vitesse
            elif self._turn_state == 'exec':
                turn_active = True
                self._turn_state_ticks -= 1
                cmd    = 'TURN_RIGHT'
                reason = f'Virage droite 90° ({self._turn_state_ticks} ticks restants)'
                # Sortie anticipée si la voie est retrouvée (virage terminé)
                # Guard : au moins 20 ticks pour éviter sortie prématurée en début de virage
                lane_found = lane_fresh and self.lane_status not in ('NO_LANE', 'NO LANE')
                if self._turn_state_ticks <= 0 or (
                        lane_found and self._turn_state_ticks < TURN_RIGHT_TICKS - 25):
                    self._turn_state       = 'search'
                    self._turn_state_ticks = TURN_SEARCH_TICKS
                    rospy.loginfo('[DECISION] Rotation terminée → recherche nouvelle voie')

            # ── Phase 3 : avance lente — sort dès que la voie est détectée ────
            elif self._turn_state == 'search':
                self._turn_state_ticks -= 1
                turn_active = True
                if cmd not in ('BRAKE',):
                    cmd    = 'SLOW'
                    reason = f'Post-virage — recherche voie ({self._turn_state_ticks} ticks)'
                # Sortie anticipée si le lane viewer a retrouvé une voie
                lane_found = lane_fresh and self.lane_status not in ('NO_LANE', 'NO LANE')
                if self._turn_state_ticks <= 0 or (lane_found and self._turn_state_ticks < TURN_SEARCH_TICKS - 5):
                    self._turn_state        = None
                    self._turn_state_ticks  = 0
                    self._turn_intent       = None
                    self._turn_intent_ticks = 0
                    self._turn_pending      = False
                    self._turn_sign_hist.clear()
                    rospy.loginfo(f'[DECISION] Voie {"détectée" if lane_found else "timeout"} → reprise normale')

            # ── Attente intersection (turn pending, pas encore en NO_LANE) ────
            elif self._turn_pending:
                # Décompter le délai d'approche — bloque NO_LANE sur la route actuelle
                if self._turn_approach_ticks > 0:
                    self._turn_approach_ticks -= 1
                    self._turn_no_lane_ticks = 0   # réinitialiser pendant l'approche
                    if cmd == 'GO':
                        cmd    = 'SLOW'
                        reason = f'Panneau DROITE — approche ({self._turn_approach_ticks} ticks)'
                else:
                    # Dès le 1er NO_LANE détecté → armer le délai 1 s (ne pas remettre à 0)
                    if self.lane_status in ('NO_LANE', 'NO LANE') or self._turn_no_lane_ticks > 0:
                        self._turn_no_lane_ticks += 1

                    if self._turn_no_lane_ticks >= TURN_NO_LANE_CONFIRM:
                        # Intersection confirmée → démarrer phase 1
                        self._turn_state         = 'brake'
                        self._turn_state_ticks   = TURN_BRAKE_TICKS
                        self._turn_pending        = False
                        self._turn_no_lane_ticks  = 0
                        cmd    = 'BRAKE'
                        reason = 'Intersection détectée — pré-freinage virage'
                        turn_active = True
                        rospy.loginfo('[DECISION] Intersection détectée → séquence virage')
                    elif cmd == 'GO':
                        cmd    = 'SLOW'
                        reason = f'Panneau DROITE — attente intersection ({self._turn_no_lane_ticks}/{TURN_NO_LANE_CONFIRM})'

        if not turn_active and cmd in ('GO', 'SLOW') and lane_fresh:
            status  = self.lane_status
            dev     = self.lane_dev_f
            err     = dev - DEV_TARGET
            err_eff = err * (DEV_GAIN_RIGHT if err >= 0.0 else DEV_GAIN_LEFT)

            if status == 'NO_LANE':
                if self._turn_intent_ticks > 0 and self._turn_intent == 'left':
                    cmd = 'STEER_LEFT'
                    reason = 'NO_LANE + panneau GAUCHE → virage gauche'
                else:
                    cmd    = 'SLOW'
                    reason = 'Voie perdue — SLOW (re-acquisition)'

            else:
                # Compteur de confirmation pour STEER_LEFT (évite faux positifs)
                if dev < LEFT_CONFIRM_DEV and err_eff < -ERR_LEFT_ON:
                    self._left_confirm = min(self._left_confirm + 1, LEFT_CONFIRM_TICKS + 2)
                else:
                    self._left_confirm = max(0, self._left_confirm - 1)

                if err_eff > ERR_RIGHT_ON:
                    self._steer_state = 1
                    self._hold_ticks  = STEER_HOLD_TICKS + 2
                    self._left_confirm = 0
                elif self._left_confirm >= LEFT_CONFIRM_TICKS:
                    self._steer_state = -1
                    self._hold_ticks  = STEER_HOLD_TICKS
                elif abs(err_eff) < ERR_EXIT:
                    self._steer_state = 0
                    self._hold_ticks  = 0
                    self._left_confirm = 0
                elif self._steer_state != 0 and self._hold_ticks > 0:
                    self._hold_ticks -= 1

                if self._steer_state > 0:
                    cmd    = 'STEER_RIGHT'
                    reason = f'dev={dev:+.3f} → correction droite'
                elif self._steer_state < 0:
                    cmd    = 'STEER_LEFT'
                    reason = f'dev={dev:+.3f} → correction gauche'
                else:
                    cmd    = 'GO'
                    reason = f'Centre OK — {self.speed:.1f} km/h'

                if abs(err_eff) > CURVE_SLOW_ERR and cmd != 'BRAKE':
                    cmd    = 'SLOW'
                    reason = f'Virage serré err={err_eff:+.3f} — ralenti'

        # ════════════════════════════════════════════════════════════════════
        #  Publication
        # ════════════════════════════════════════════════════════════════════
        self.pub_ctrl.publish(String(data=cmd))
        self.pub_dec.publish(String(data=reason))
        rospy.loginfo_throttle(2, f'[DECISION] {cmd:12s} | {reason}')


if __name__ == '__main__':
    DecisionNode()
    rospy.spin()