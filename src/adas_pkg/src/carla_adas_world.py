#!/usr/bin/env python3
"""
carla_adas_world.py  —  Pont CARLA ↔ ROS

Séquence :
  Init  5s  — attente caméra + nœuds ROS
  Dep   4s  — départ libre
  S1   10s  — STOP sign        → stop.png
  S2   20s  — Feu ROUGE/VERT   → light_red.png / light_green.png  (PNG seul, pas de forçage CARLA)
  S3   12s  — Limite 10 km/h   → speed_10.png
  S4   12s  — Piéton traverse devant
  S5   12s  — Voiture immobile devant
  S6   10s  — Virage DROITE à intersection réelle → turn_right.png
"""

import math
import random
import threading
import time

import carla
import cv2
import numpy as np
import pygame
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

# ─── Paramètres PD ───────────────────────────────────────────────────────────
THROTTLE_GO          = 0.52
THROTTLE_SLW         = 0.36
THROTTLE_TURN        = 0.30
THROTTLE_TURN_CREEP  = 0.22   # throttle pendant virage serré
TURN_CREEP_SPEED_KMH = 5.0    # vitesse cible pendant la rotation (km/h)
K_P                  = 0.78
K_D                  = 0.90
MAX_STEER            = 0.48
STEER_TURN_LOCK      = 1.0    # verrouillage volant MAX pour rayon de virage minimal
STEER_RATE           = 0.11
DEV_SMOOTH           = 0.25
CMD_STEER_BIAS_RIGHT = 0.11
CMD_STEER_BIAS_LEFT  = 0.04
DEV_DEADBAND         = 0.012
NO_LANE_RIGHT_BIAS   = 0.03

SIGN_DIR  = '/home/fedi/adas_signs'
SIGN_FILES = {
    'stop':         'stop.png',
    'speed_10':     'speed_10.png',
    'speed_20':     'speed_20.png',
    'speed_30':     'speed_30.png',
    'speed_50':     'speed_50.png',
    'speed_60':     'speed_60.png',
    'speed_90':     'speed_90.png',
    'light_red':    'light_red.png',
    'light_yellow': 'light_yellow.png',
    'light_green':  'light_green.png',
    'turn_right':   'turn_right.png',
    'turn_left':    'turn_left.png',
}
SIGN_SIZE = (120, 120)


def _safe_destroy(actor):
    if actor is None:
        return
    try:
        if actor.is_alive:
            actor.destroy()
    except Exception:
        pass


def _load_png(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 2:
        bgr, alpha = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), None
    elif raw.shape[2] == 4:
        bgr   = raw[:, :, :3].copy()
        alpha = raw[:, :, 3].astype(np.float32) / 255.0
    else:
        bgr, alpha = raw.copy(), None
    bgr = cv2.resize(bgr, SIGN_SIZE, interpolation=cv2.INTER_AREA)
    if alpha is not None:
        alpha = cv2.resize(alpha, SIGN_SIZE, interpolation=cv2.INTER_AREA)
    return {'bgr': bgr, 'alpha': alpha}


def _paste_sign(frame, sign, px, py):
    fh, fw = frame.shape[:2]
    sh, sw = sign['bgr'].shape[:2]
    x0, y0 = max(px, 0), max(py, 0)
    x1, y1 = min(px + sw, fw), min(py + sh, fh)
    if x1 <= x0 or y1 <= y0:
        return frame
    sx0, sy0 = x0 - px, y0 - py
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    roi  = frame[y0:y1, x0:x1].astype(np.float32)
    crop = sign['bgr'][sy0:sy1, sx0:sx1].astype(np.float32)
    if sign['alpha'] is not None:
        a   = sign['alpha'][sy0:sy1, sx0:sx1, None]
        out = crop * a + roi * (1.0 - a)
    else:
        out = crop * 0.90 + roi * 0.10
    frame[y0:y1, x0:x1] = out.clip(0, 255).astype(np.uint8)
    return frame


class CarlaADASWorld:

    def __init__(self):
        rospy.init_node('carla_adas_world', anonymous=False)

        self._lock        = threading.Lock()
        self._cmd         = 'WAIT'
        self._deviation   = 0.0
        self._display_bgr = None
        self._dev_filt    = 0.0
        self._dev_prev    = 0.0
        self._steer_prev  = 0.0
        self._lane_ready  = False
        self._lane_status = 'NO_LANE'

        self._sign_key    = None
        self._sign_frames = 0
        self._hud_text    = ''
        self._hud_frames  = 0

        self._signs  = {}
        self._bridge = CvBridge()
        self.actors  = []
        self.client = self.world = self.vehicle = self.tm = None
        self._pub_seg = None
        self._force_detection = bool(rospy.get_param('~force_detection', False))

        self._pub_img   = rospy.Publisher('/carla/camera/rgb',  Image,   queue_size=1)
        self._pub_radar = rospy.Publisher('/carla/radar/front', Float32, queue_size=1)
        self._pub_speed = rospy.Publisher('/carla/speed',       Float32, queue_size=1)
        self._pub_det_str = rospy.Publisher('/adas/detection',  String,  queue_size=1)

        rospy.Subscriber('/adas/control',   String,  self._cb_ctrl,        queue_size=1)
        rospy.Subscriber('/lane/deviation', Float32, self._cb_dev,         queue_size=1)
        rospy.Subscriber('/lane/status',    String,  self._cb_lane_status, queue_size=1)

        self._load_signs()

    # ── Chargement PNG ────────────────────────────────────────────────────────

    def _load_signs(self):
        ok = 0
        for key, fname in SIGN_FILES.items():
            entry = _load_png(f'{SIGN_DIR}/{fname}')
            if entry is None:
                rospy.logwarn(f'[SIGNS] introuvable : {SIGN_DIR}/{fname}')
            else:
                self._signs[key] = entry
                ok += 1
        rospy.loginfo(f'[SIGNS] {ok}/{len(SIGN_FILES)} panneaux chargés')

    # ── Overlay helpers ───────────────────────────────────────────────────────

    def _show_sign(self, key, duration_s):
        if key not in self._signs:
            rospy.logwarn(f'[SIGNS] clé inconnue : {key}')
            return
        with self._lock:
            self._sign_key    = key
            self._sign_frames = max(1, int(duration_s * 30))
        rospy.loginfo(f'[SIGNS] → {key}  ({duration_s:.1f} s)')

    def _clear_sign(self):
        with self._lock:
            self._sign_key    = None
            self._sign_frames = 0

    def _set_hud(self, text, duration_s=7.0):
        rospy.loginfo(f'\n{"="*55}\n  {text}\n{"="*55}')
        with self._lock:
            self._hud_text   = text
            self._hud_frames = max(1, int(duration_s * 30))

    # ── Callbacks ROS ─────────────────────────────────────────────────────────

    def _cb_ctrl(self, msg):
        with self._lock:
            self._cmd = msg.data

    def _cb_dev(self, msg):
        with self._lock:
            self._deviation  = msg.data
            self._lane_ready = True

    def _cb_lane_status(self, msg):
        with self._lock:
            self._lane_status = msg.data

    # ── Publication détection ─────────────────────────────────────────────────

    def _pub_det(self, tokens, duration_s, period=0.08):
        # Désactivé: on garde l'affichage PNG, mais aucune publication de détection.
        return

    # ── Connexion CARLA ───────────────────────────────────────────────────────

    def connect(self):
        TARGET = 'Town10HD_Opt'
        rospy.loginfo('Connexion CARLA …')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(60.0)
        try:
            self.world = self.client.get_world()
            if self.world.get_map().name.split('/')[-1] != TARGET:
                rospy.loginfo(f'Chargement {TARGET} …')
                self.world = self.client.load_world(TARGET)
            else:
                rospy.loginfo(f'{TARGET} déjà chargée')
        except Exception:
            self.world = self.client.load_world(TARGET)
        s = self.world.get_settings()
        s.synchronous_mode = False
        self.world.apply_settings(s)
        rospy.loginfo('Connecté — mode asynchrone')

    def setup_tm(self):
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(False)
        self.tm.set_global_distance_to_leading_vehicle(3.0)
        self.tm.set_random_device_seed(42)

    # ── Spawn ego ─────────────────────────────────────────────────────────────

    def spawn_ego(self):
        bp_lib = self.world.get_blueprint_library()
        vbp    = bp_lib.find('vehicle.tesla.model3')
        vbp.set_attribute('color', '255,50,50')
        wmap = self.world.get_map()
        sps  = wmap.get_spawn_points()
        best_sp, best_score = None, 999.0
        for sp in sps:
            wp = wmap.get_waypoint(sp.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            if wp is None: continue
            rw = wp.get_right_lane()
            if not rw or rw.lane_type != carla.LaneType.Driving: continue
            cur, yaws = wp, []
            for _ in range(15):
                nxt = cur.next(2.0)
                if not nxt: break
                cur = nxt[0]; yaws.append(cur.transform.rotation.yaw)
            if len(yaws) < 5: continue
            score = max(yaws) - min(yaws)
            if score < best_score:
                best_score, best_sp = score, sp
        sp = best_sp or sps[0]
        self.vehicle = self.world.try_spawn_actor(vbp, sp)
        if self.vehicle is None:
            rospy.logfatal('Spawn ego échoué !'); return
        self.vehicle.set_autopilot(False)
        self.actors.append(self.vehicle)
        rospy.loginfo(f'Ego spawné (courbure {best_score:.1f}°)')

    # ── Capteurs ──────────────────────────────────────────────────────────────

    def attach_camera(self):
        bp_lib = self.world.get_blueprint_library()
        cb = bp_lib.find('sensor.camera.rgb')
        cb.set_attribute('image_size_x', '640')
        cb.set_attribute('image_size_y', '360')
        cb.set_attribute('fov', '90')
        cb.set_attribute('sensor_tick', '0.033')
        cb.set_attribute('motion_blur_intensity', '0.0')
        cam = self.world.spawn_actor(cb,
              carla.Transform(carla.Location(x=1.5, z=1.7)),
              attach_to=self.vehicle)
        cam.listen(self._camera_cb)
        self.actors.append(cam)
        rospy.loginfo('Caméra RGB 640×360 attachée')

    def _camera_cb(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgr = np.ascontiguousarray(
              arr.reshape((image.height, image.width, 4))[:, :, :3])
        with self._lock:
            key    = self._sign_key
            frames = self._sign_frames
        if key and frames > 0:
            sign = self._signs.get(key)
            if sign is not None:
                bgr = _paste_sign(bgr, sign,
                                  image.width - SIGN_SIZE[0] - 15, 15)
            with self._lock:
                self._sign_frames = max(0, self._sign_frames - 1)
                if self._sign_frames == 0:
                    self._sign_key = None
        with self._lock:
            self._display_bgr = bgr.copy()
        try:
            msg = self._bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            msg.header.stamp = rospy.Time.now()
            self._pub_img.publish(msg)
        except Exception:
            pass

    def attach_semantic_camera(self):
        bp_lib = self.world.get_blueprint_library()
        sb = bp_lib.find('sensor.camera.semantic_segmentation')
        sb.set_attribute('image_size_x', '640')
        sb.set_attribute('image_size_y', '360')
        sb.set_attribute('fov', '90')
        sb.set_attribute('sensor_tick', '0.1')
        self._pub_seg = rospy.Publisher('/carla/camera/seg', Image, queue_size=1)
        seg = self.world.spawn_actor(sb,
              carla.Transform(carla.Location(x=1.5, z=1.7)),
              attach_to=self.vehicle)
        seg.listen(self._seg_cb)
        self.actors.append(seg)
        rospy.loginfo('Caméra sémantique attachée')

    def _seg_cb(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        seg = np.ascontiguousarray(
              arr.reshape((image.height, image.width, 4))[:, :, :3])
        try:
            msg = self._bridge.cv2_to_imgmsg(seg, encoding='bgr8')
            msg.header.stamp = rospy.Time.now()
            if self._pub_seg: self._pub_seg.publish(msg)
        except Exception:
            pass

    def attach_radar(self):
        bp_lib = self.world.get_blueprint_library()
        rb = bp_lib.find('sensor.other.radar')
        rb.set_attribute('horizontal_fov', '60')
        rb.set_attribute('vertical_fov',   '20')
        rb.set_attribute('range',          '50')
        rb.set_attribute('points_per_second', '20000')
        rb.set_attribute('sensor_tick',    '0.05')
        rad = self.world.spawn_actor(rb,
              carla.Transform(carla.Location(x=2.0, z=1.0)),
              attach_to=self.vehicle)
        rad.listen(self._radar_cb)
        self.actors.append(rad)
        rospy.loginfo('Radar frontal attaché')

    def _radar_cb(self, data):
        mn = min((pt.depth for pt in data), default=float('inf'))
        if mn != float('inf'):
            self._pub_radar.publish(Float32(data=mn))

    def spawn_npc(self, count=4):
        bp_lib = self.world.get_blueprint_library()
        sps    = self.world.get_map().get_spawn_points()
        ego    = self.vehicle.get_location()
        far    = [s for s in sps if s.location.distance(ego) > 60]
        random.shuffle(far)
        n = 0
        for sp in far[:count * 3]:
            npc = self.world.try_spawn_actor(
                      random.choice(bp_lib.filter('vehicle.*')), sp)
            if npc:
                npc.set_autopilot(True)
                self.actors.append(npc)
                n += 1
            if n >= count: break
        rospy.loginfo(f'{n} NPC spawné(s)')

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def _wp_ahead(self, n_steps, step_m=2.0):
        """Waypoint n_steps × step_m mètres devant l'ego sur la même voie."""
        try:
            wp = self.world.get_map().get_waypoint(
                     self.vehicle.get_transform().location,
                     project_to_road=True, lane_type=carla.LaneType.Driving)
            for _ in range(n_steps):
                nxt = wp.next(step_m)
                if nxt: wp = nxt[0]
                else:   break
            return wp
        except Exception:
            return None

    def _find_right_turn_intersection(self, search_m=120):
        """
        Parcourt le graphe waypoint jusqu'à search_m m devant l'ego et
        retourne le premier waypoint où un virage à droite est possible
        (voie droite disponible ET la route tourne à droite).
        Retourne (wp_intersection, distance) ou (None, 0).
        """
        try:
            wmap = self.world.get_map()
            ego_loc = self.vehicle.get_transform().location
            wp = wmap.get_waypoint(ego_loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            dist = 0.0
            prev_yaw = wp.transform.rotation.yaw
            for _ in range(int(search_m / 2)):
                nxt_list = wp.next(2.0)
                if not nxt_list:
                    break
                wp   = nxt_list[0]
                dist += 2.0
                rw   = wp.get_right_lane()
                # Critère : voie droite de type Driving disponible
                if rw and rw.lane_type == carla.LaneType.Driving:
                    return wp, dist
            return None, 0
        except Exception:
            return None, 0

    # ─────────────────────────────────────────────────────────────────────────
    #  SCÉNARIOS
    # ─────────────────────────────────────────────────────────────────────────

    # ── S1 : STOP ─────────────────────────────────────────────────────────────
    def _s1_stop(self, dur=10.0):
        self._set_hud('[S1] Panneau STOP — freinage !', dur)
        self._show_sign('stop', dur)
        self._pub_det(['stop:stop:0.99:200:30:440:240'], dur)
        time.sleep(dur)
        self._clear_sign()

    # ── S2 : Feu ROUGE → VERT  (PNG seul, pas de forçage CARLA) ─────────────
    def _s2_traffic_light(self, red_dur=12.0, green_dur=8.0):
        """
        Affiche light_red.png pendant red_dur s, puis light_green.png pendant green_dur s.
        Le feu CARLA N'EST PAS forcé — object_detection_node doit détecter le vrai feu.
        Le PNG sert d'indication visuelle pour la démo / rapport.
        """
        # ── ROUGE ──────────────────────────────────────────────────────────
        self._set_hud('[S2] Feu ROUGE — attente détection …', red_dur)
        self._show_sign('light_red', red_dur)
        # On publie quand même le token pour que decision_node réagisse
        # si object_detection ne détecte pas encore
        self._pub_det(['feu_rouge:red:0.98:260:50:400:210'], red_dur)
        time.sleep(red_dur)
        self._clear_sign()

        # ── VERT ───────────────────────────────────────────────────────────
        self._set_hud('[S2] Feu VERT — reprendre !', green_dur)
        self._show_sign('light_green', green_dur)
        self._pub_det(['feu_vert:green:0.98:260:50:400:210'], green_dur)
        time.sleep(green_dur)
        self._clear_sign()

    # ── S3 : Limite 30 km/h ───────────────────────────────────────────────────
    def _s3_limit10(self, dur=12.0):
        self._set_hud('[S3] Limite 30 km/h — ralentir !', dur)
        self._show_sign('speed_30', dur)
        self._pub_det(['vitesse:30:0.97:200:30:440:240'], dur)
        time.sleep(dur)
        self._clear_sign()

    # ── S4 : Piéton traverse devant ───────────────────────────────────────────
    def _s4_pedestrian(self, dur=12.0):
        """
        Spawn un piéton 18 m devant l'ego, côté droit.
        Il traverse lentement → radar le détecte → decision_node → BRAKE.
        """
        self._set_hud('[S4] PIÉTON traverse — STOP !', dur)
        actors_local = []
        try:
            bp_lib    = self.world.get_blueprint_library()
            walker_bp = random.choice(list(bp_lib.filter('walker.pedestrian.*')))
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            t     = self.vehicle.get_transform()
            fwd   = t.get_forward_vector()
            right = t.get_right_vector()

            # Spawn 18 m devant, 3.5 m à droite (trottoir)
            loc = carla.Location(
                x = t.location.x + fwd.x * 18 + right.x * 3.5,
                y = t.location.y + fwd.y * 18 + right.y * 3.5,
                z = t.location.z + 0.5)

            walker = self.world.try_spawn_actor(walker_bp, carla.Transform(loc))
            if not walker:
                loc.x += right.x * 0.8; loc.y += right.y * 0.8
                walker = self.world.try_spawn_actor(walker_bp, carla.Transform(loc))
            if not walker:
                rospy.logwarn('Spawn piéton échoué')
                time.sleep(dur); return

            actors_local.append(walker)
            self.actors.append(walker)
            rospy.loginfo(f'      → Piéton spawné ({loc.x:.1f},{loc.y:.1f})')

            ctrl_bp = bp_lib.find('controller.ai.walker')
            ctrl    = self.world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
            actors_local.append(ctrl)
            self.actors.append(ctrl)
            self.world.wait_for_tick()
            ctrl.start()

            # Traverse de droite → gauche (coupe la voie de l'ego)
            dest = carla.Location(
                x = loc.x - right.x * 9.0,
                y = loc.y - right.y * 9.0,
                z = loc.z)
            ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.0)   # lent → l'ego a le temps de freiner

            time.sleep(dur)

        except Exception as e:
            rospy.logwarn(f'_s4_pedestrian: {e}')
            time.sleep(dur)
        finally:
            for a in actors_local:
                _safe_destroy(a)
                if a in self.actors: self.actors.remove(a)

    # ── S5 : Voiture immobile devant ──────────────────────────────────────────
    def _s5_car_ahead(self, dur=12.0):
        """
        Spawn une voiture arrêtée ~16 m devant l'ego sur la même voie.
        Le radar la détecte → decision_node → BRAKE/SLOW.
        """
        self._set_hud('[S5] Voiture devant — FREINER !', dur)
        car = None
        try:
            wp = self._wp_ahead(8)   # 8 × 2 m = 16 m
            if wp is None:
                rospy.logwarn('_s5: pas de waypoint'); time.sleep(dur); return

            tf = carla.Transform(
                carla.Location(x=wp.transform.location.x,
                               y=wp.transform.location.y,
                               z=wp.transform.location.z + 0.3),
                wp.transform.rotation)

            bp_lib  = self.world.get_blueprint_library()
            car_bps = [b for b in bp_lib.filter('vehicle.*')
                       if int(b.get_attribute('number_of_wheels')) == 4]
            car = self.world.try_spawn_actor(random.choice(car_bps), tf)

            if car:
                car.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                self.actors.append(car)
                dist = self.vehicle.get_location().distance(tf.location)
                rospy.loginfo(f'      → Voiture à {dist:.0f} m (même voie, freinée)')
            else:
                rospy.logwarn('      Spawn voiture échoué')

            time.sleep(dur)

        except Exception as e:
            rospy.logwarn(f'_s5_car_ahead: {e}')
        finally:
            _safe_destroy(car)
            if car and car in self.actors: self.actors.remove(car)

    # ── S6 : Virage DROITE à intersection réelle ──────────────────────────────
    def _s6_turn_right(self, dur=10.0):
        """
        Affiche turn_right.png — object_detection_node détecte le panneau naturellement
        et decision_node exécute le virage.
        """
        self._set_hud('[S6] Virage DROITE — intersection réelle', dur)
        self._show_sign('turn_right', dur)
        try:
            time.sleep(dur)
        finally:
            self._clear_sign()


    # ─────────────────────────────────────────────────────────────────────────
    #  SÉQUENCE PRINCIPALE
    # ─────────────────────────────────────────────────────────────────────────

    def start_scenarios(self):
        def _run():
            rospy.loginfo('\n' + '='*55)
            rospy.loginfo('  STOP → Feu R/V → Lim10 → Piéton → Voiture → Virage droite')
            rospy.loginfo('='*55 + '\n')

            # Attendre caméra + nœuds ROS
            self._set_hud('Initialisation …', 5.0)
            time.sleep(5)

            self._set_hud('Départ — accélération libre', 4.0)
            time.sleep(4)

            # S1 — STOP (10 s)
            self._s1_stop(dur=5.0)
            time.sleep(5)

            # S2 — Feu ROUGE → VERT (12 s + 8 s)
            self._s2_traffic_light(red_dur=9.0, green_dur=3.0)
            time.sleep(6)

            # S3 — Limite 10 km/h (12 s)
            self._s3_limit10(dur=9.0)
            time.sleep(3)

            # S4 — Piéton traverse (12 s)
            self._s4_pedestrian(dur=8.0)
            time.sleep(3)

            # S5 — Voiture devant (12 s)
            self._s5_car_ahead(dur=6.0)
            time.sleep(3)

            # S6 — Virage droite à intersection réelle (10 s)
            self._s6_turn_right(dur=7.0)
            time.sleep(10)

            self._set_hud('Simulation terminée — tous les scénarios OK !', 10.0)
            rospy.loginfo('\nTous les scénarios terminés.')

        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  CONTRÔLE VÉHICULE — PD controller
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_control(self):
        with self._lock:
            cmd         = self._cmd
            dev         = self._deviation
            ready       = self._lane_ready
            lane_status = self._lane_status

        if not ready:
            ctrl = carla.VehicleControl()
            ctrl.throttle = 0.0; ctrl.brake = 0.4
            self.vehicle.apply_control(ctrl)
            v   = self.vehicle.get_velocity()
            spd = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6
            self._pub_speed.publish(Float32(data=spd))
            return spd, 'WAIT'

        self._dev_filt = DEV_SMOOTH * self._dev_filt + (1.0 - DEV_SMOOTH) * float(dev)
        dev_eff = self._dev_filt
        if abs(dev_eff) < DEV_DEADBAND: dev_eff = 0.0

        d_dev          = dev_eff - self._dev_prev
        self._dev_prev = dev_eff
        steer_target   = float(np.clip(K_P * dev_eff + K_D * d_dev, -MAX_STEER, MAX_STEER))

        if cmd == 'STEER_LEFT':    steer_target -= CMD_STEER_BIAS_LEFT
        elif cmd == 'STEER_RIGHT': steer_target += CMD_STEER_BIAS_RIGHT
        if lane_status == 'NO_LANE':
            steer_target = max(steer_target, NO_LANE_RIGHT_BIAS)

        steer_target = float(np.clip(steer_target, -MAX_STEER, MAX_STEER))
        steer_cmd = float(np.clip(steer_target,
                                  self._steer_prev - STEER_RATE,
                                  self._steer_prev + STEER_RATE))
        self._steer_prev = steer_cmd

        ctrl = carla.VehicleControl()

        v_now   = self.vehicle.get_velocity()
        spd_now = math.sqrt(v_now.x**2 + v_now.y**2 + v_now.z**2) * 3.6

        if cmd == 'TURN_RIGHT':
            # Volant MAX instantané (bypass rate-limiter) — arc de virage minimal
            ctrl.steer = STEER_TURN_LOCK
            self._steer_prev = STEER_TURN_LOCK
            if spd_now > TURN_CREEP_SPEED_KMH:
                # Trop rapide → freine mais GARDE le volant bloqué à droite
                ctrl.throttle = 0.0
                ctrl.brake    = 1.0
            else:
                # Vitesse correcte → avance en tournant
                ctrl.throttle = THROTTLE_TURN_CREEP
                ctrl.brake    = 0.0
            self.vehicle.apply_control(ctrl)
            v   = self.vehicle.get_velocity()
            spd = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6
            self._pub_speed.publish(Float32(data=spd))
            return spd, cmd

        ctrl.steer = steer_cmd
        if cmd == 'BRAKE':
            ctrl.throttle = 0.0; ctrl.brake = 1.0
        elif cmd == 'SLOW':
            ctrl.throttle = THROTTLE_SLW; ctrl.brake = 0.0
        elif cmd == 'LIMIT_10':
            err = 10.0 - spd_now
            if   err >  1.0: ctrl.throttle = float(np.clip(0.15 + 0.03*err,  0.0, 0.38)); ctrl.brake = 0.0
            elif err < -1.0: ctrl.throttle = 0.0; ctrl.brake = float(np.clip(0.08 + 0.04*(-err), 0.0, 0.50))
            else:             ctrl.throttle = 0.12; ctrl.brake = 0.0
        elif cmd == 'LIMIT_30':
            err = 30.0 - spd_now
            if   err >  1.0: ctrl.throttle = float(np.clip(0.20 + 0.025*err, 0.0, 0.42)); ctrl.brake = 0.0
            elif err < -1.0: ctrl.throttle = 0.0; ctrl.brake = float(np.clip(0.06 + 0.03*(-err), 0.0, 0.45))
            else:             ctrl.throttle = 0.18; ctrl.brake = 0.0
        elif cmd in ('STEER_LEFT', 'STEER_RIGHT'):
            ctrl.throttle = THROTTLE_TURN; ctrl.brake = 0.0
        else:   # GO
            sp = 1.0 - 0.45 * min(abs(steer_cmd) / max(MAX_STEER, 1e-6), 1.0)
            ctrl.throttle = THROTTLE_GO * sp; ctrl.brake = 0.0

        self.vehicle.apply_control(ctrl)
        v   = self.vehicle.get_velocity()
        spd = math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6
        self._pub_speed.publish(Float32(data=spd))
        return spd, cmd

    # ── Spectateur ────────────────────────────────────────────────────────────

    def _update_spectator(self):
        spec = self.world.get_spectator()
        t    = self.vehicle.get_transform()
        fwd  = t.get_forward_vector()
        spec.set_transform(carla.Transform(
            carla.Location(x=t.location.x - fwd.x*9,
                           y=t.location.y - fwd.y*9,
                           z=t.location.z + 6),
            carla.Rotation(pitch=-18, yaw=t.rotation.yaw)))

    # ─────────────────────────────────────────────────────────────────────────
    #  BOUCLE PRINCIPALE pygame
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        pygame.init()
        display  = pygame.display.set_mode((640, 360))
        pygame.display.set_caption('CARLA ADAS')
        font_big = pygame.font.SysFont('monospace', 20, bold=True)
        font_sm  = pygame.font.SysFont('monospace', 15)
        clock    = pygame.time.Clock()

        CMD_COLORS = {
            'BRAKE':       (255,  60,  60),
            'SLOW':        (255, 200,   0),
            'LIMIT_10':    (255, 130,   0),
            'LIMIT_20':    (255, 170,   0),
            'STEER_LEFT':  ( 80, 180, 255),
            'STEER_RIGHT': ( 80, 180, 255),
            'TURN_RIGHT':  ( 80, 180, 255),
            'GO':          ( 80, 220,  80),
            'WAIT':        (160, 160, 160),
        }

        rospy.loginfo('Fenêtre pygame ouverte — Q pour quitter')

        while not rospy.is_shutdown():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cleanup(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    self.cleanup(); return

            spd, cmd = self._apply_control()
            self._update_spectator()

            with self._lock:
                bgr      = self._display_bgr.copy() if self._display_bgr is not None else None
                hud_text = self._hud_text
                hud_fr   = self._hud_frames

            if bgr is not None:
                surf = pygame.surfarray.make_surface(bgr[:, :, ::-1].swapaxes(0, 1))
                display.blit(surf, (0, 0))
            else:
                display.fill((20, 20, 20))
                display.blit(font_sm.render('En attente caméra …',
                             True, (80, 200, 80)), (80, 165))

            # HUD haut
            top = pygame.Surface((640, 55), pygame.SRCALPHA)
            top.fill((10, 10, 10, 190))
            display.blit(top, (0, 0))
            col = CMD_COLORS.get(cmd, (200, 200, 200))
            display.blit(font_big.render(f'Vitesse : {spd:5.1f} km/h', True, (255,255,255)), (10,  8))
            display.blit(font_sm.render( f'Cmd     : {cmd}',           True, col),            (10, 35))

            # HUD bas scénario
            if hud_fr > 0 and hud_text:
                bot = pygame.Surface((640, 42), pygame.SRCALPHA)
                bot.fill((20, 20, 100, 220))
                display.blit(bot, (0, 308))
                display.blit(font_big.render(hud_text[:62], True, (255,255,255)), (8, 316))
                with self._lock:
                    self._hud_frames = max(0, self._hud_frames - 1)

            pygame.display.flip()
            clock.tick(30)

        self.cleanup()

    # ── Nettoyage ─────────────────────────────────────────────────────────────

    def cleanup(self):
        rospy.loginfo('Nettoyage …')
        for a in self.actors:
            _safe_destroy(a)
        pygame.quit()
        rospy.loginfo('Terminé.')


# ─────────────────────────────────────────────────────────────────────────────

def main():
    node = CarlaADASWorld()
    rospy.on_shutdown(node.cleanup)
    node.connect()
    node.setup_tm()
    node.spawn_ego()
    time.sleep(1.0)
    node.attach_camera()
    node.attach_semantic_camera()
    node.attach_radar()
    node.spawn_npc(4)
    node.start_scenarios()
    node.run()


if __name__ == '__main__':
    main()