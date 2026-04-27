#!/usr/bin/env python3
"""
object_detection_node.py — Détection temps réel (bestobject.pt uniquement)

Classes actives (bestobject.pt) :
  Véhicules : person(0) car(1) truck(2) bus(3) motorcycle(4)
  Feux      : red light(5) green light(6)
  Panneaux  : stop sign(7) speed limit 20-120 (10-18)

Seuils séparés par catégorie pour compenser le domain gap CARLA.

Publie :
  /adas/detection        (String)  — pipe-separated: class:detail:conf:x1:y1:x2:y2
  /adas/detection_image  (Image)   — frame annotée
Souscrit :
  /carla/camera/rgb      (Image)
"""

import os
import threading
import time
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SRC_DIR, 'bestobject.pt')

INFER_SIZE = 640

# Seuils par catégorie
CONF_VEHICLES = 0.05   # un peu plus sensible pour récupérer car/pieton
CONF_LIGHTS   = 0.25   # plus sensible pour récupérer les feux CARLA
CONF_SIGNS    = 0.06   # panneaux stop/vitesse
CONF_SPEED_SIGNS = 0.30  # plus strict pour limiter les faux panneaux vitesse

LIGHT_MIN_AREA = 140   # px² — plus permissif pour feux lointains
LIGHT_MAX_KEEP = 3     # max N feux par frame
SIGN_MIN_AREA  = 450   # px² — filtre des petits faux positifs
SIGN_MIN_AR    = 0.65  # ratio w/h minimal attendu
SIGN_MAX_AR    = 1.35  # ratio w/h maximal attendu

DEDUP_IOU_THRESH = 0.55   # suppression de doublons bbox
MAX_VEH_KEEP     = 20     # max véhicules publiés par frame
MAX_SIGN_KEEP    = 6      # max panneaux publiés par frame
STICKY_HOLD_SEC  = 1.20   # maintien un peu plus long pour feux/panneaux
LOG_DET_EVERY_SEC = 1.0   # cadence d'affichage terminal
LOG_NONE          = True  # afficher aussi 'none' pour diagnostiquer le flux
USE_CLASS_FILTER  = True  # filtrer sur classes utiles du modèle

# IDs par catégorie
VEHICLE_IDS = {0, 1, 2, 3, 4}   # person car truck bus motorcycle
LIGHT_IDS   = {5, 6}             # red light, green light
SIGN_IDS    = {7, 10, 11, 12, 13, 14, 15, 16, 17, 18}  # stop + speed limits

ALL_ACTIVE  = list(VEHICLE_IDS | LIGHT_IDS | SIGN_IDS)

LABEL_MAP = {
    'person':        'pieton',
    'pedestrian':    'pieton',
    'walker':        'pieton',
    'car':           'voiture',
    'vehicle':       'voiture',
    'truck':         'camion',
    'bus':           'bus',
    'motorcycle':    'moto',
    'motorbike':     'moto',
    'red light':     'feu_rouge',
    'green light':   'feu_vert',
    'stop sign':     'stop',
    'turn right':    'droite',
    'right turn':    'droite',
    'mandatory right': 'droite',
    'turn left':     'gauche',
    'left turn':     'gauche',
    'mandatory left': 'gauche',
    'speed limit 20':  'vitesse',
    'speed limit 30':  'vitesse',
    'speed limit 40':  'vitesse',
    'speed limit 50':  'vitesse',
    'speed limit 60':  'vitesse',
    'speed limit 70':  'vitesse',
    'speed limit 80':  'vitesse',
    'speed limit 100': 'vitesse',
    'speed limit 120': 'vitesse',
}

BOX_COLORS = {
    'pieton':    (0,   0,   255),
    'voiture':   (255, 140, 0),
    'camion':    (255, 60,  0),
    'bus':       (255, 100, 0),
    'moto':      (255, 100, 0),
    'feu_rouge': (0,   0,   255),
    'feu_vert':  (0,   255, 0),
    'stop':      (0,   0,   200),
    'vitesse':   (200, 0,   200),
    'default':   (128, 128, 128),
}


class ObjectDetectionNode:

    def __init__(self):
        rospy.init_node('object_detection_node', anonymous=False)
        self._bridge = CvBridge()
        self._lock   = threading.Lock()
        self._frame  = None
        self._model  = None
        self._torch  = None
        self._names  = {}
        self._vehicle_ids = set(VEHICLE_IDS)
        self._light_ids   = set(LIGHT_IDS)
        self._sign_ids    = set(SIGN_IDS)
        self._all_active  = list(ALL_ACTIVE)
        self._sticky_until = {}
        self._sticky_det   = {}

        # Paramètres runtime (tuning sans modifier le code)
        self._model_path         = rospy.get_param('~model_path', MODEL_PATH)
        self._infer_size         = int(rospy.get_param('~infer_size', INFER_SIZE))
        self._conf_vehicles      = float(rospy.get_param('~conf_vehicles', CONF_VEHICLES))
        self._conf_lights        = float(rospy.get_param('~conf_lights', CONF_LIGHTS))
        self._conf_signs         = float(rospy.get_param('~conf_signs', CONF_SIGNS))
        self._conf_speed_signs   = float(rospy.get_param('~conf_speed_signs', CONF_SPEED_SIGNS))
        self._light_min_area     = int(rospy.get_param('~light_min_area', LIGHT_MIN_AREA))
        self._light_max_keep     = int(rospy.get_param('~light_max_keep', LIGHT_MAX_KEEP))
        self._sign_min_area      = int(rospy.get_param('~sign_min_area', SIGN_MIN_AREA))
        self._sign_min_ar        = float(rospy.get_param('~sign_min_ar', SIGN_MIN_AR))
        self._sign_max_ar        = float(rospy.get_param('~sign_max_ar', SIGN_MAX_AR))
        self._dedup_iou_thresh   = float(rospy.get_param('~dedup_iou_thresh', DEDUP_IOU_THRESH))
        self._max_veh_keep       = int(rospy.get_param('~max_vehicle_keep', MAX_VEH_KEEP))
        self._max_sign_keep      = int(rospy.get_param('~max_sign_keep', MAX_SIGN_KEEP))
        self._sticky_hold_sec    = float(rospy.get_param('~sticky_hold_sec', STICKY_HOLD_SEC))
        self._use_class_filter   = bool(rospy.get_param('~use_class_filter', USE_CLASS_FILTER))
        self._enable_visual      = bool(rospy.get_param('~enable_visualization', True))
        self._publish_image      = bool(rospy.get_param('~publish_image', True))
        self._loop_hz            = float(rospy.get_param('~loop_hz', 60.0))
        self._log_det_every_sec  = float(rospy.get_param('~log_det_every_sec', LOG_DET_EVERY_SEC))
        self._log_none           = bool(rospy.get_param('~log_none', LOG_NONE))

        self._model_path = self._resolve_model_path(self._model_path)

        # Frame d'attente réutilisée pour éviter des allocations inutiles
        self._wait_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(self._wait_frame,
                    'En attente /carla/camera/rgb ...',
                    (30, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (80, 200, 80), 2, cv2.LINE_AA)

        self._load_model()
        self._pub_det = rospy.Publisher('/adas/detection',       String, queue_size=1)
        self._pub_img = rospy.Publisher('/adas/detection_image', Image, queue_size=1) if self._publish_image else None
        rospy.Subscriber('/carla/camera/rgb', Image, self._cb_frame,
                         queue_size=1, buff_size=52428800)
        rospy.loginfo("Object Detection Node prêt")

    def _resolve_model_path(self, candidate_path):
        """Privilégie bestobject.pt et fournit un fallback robuste."""
        requested = str(candidate_path).strip() if candidate_path is not None else ''
        candidates = []
        if requested:
            candidates.append(requested)
        if MODEL_PATH not in candidates:
            candidates.append(MODEL_PATH)

        for p in candidates:
            if os.path.exists(p):
                if os.path.basename(p).lower() != 'bestobject.pt':
                    rospy.logwarn(f"[OBJ] Modèle non standard sélectionné: {p} (recommandé: bestobject.pt)")
                return p

        # Conserver la demande utilisateur pour un message d'erreur explicite ensuite.
        return requested if requested else MODEL_PATH

    def _load_model(self):
        if not os.path.exists(self._model_path):
            rospy.logerr(f"[OBJ] Modèle introuvable: {self._model_path}")
            return
        try:
            import torch
            self._torch = torch
        except ImportError:
            rospy.logerr("PyTorch non installé")
            return
        try:
            from ultralytics import YOLO
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                torch.backends.cudnn.benchmark = True
            self._model = YOLO(self._model_path)
            self._model.to(device)
            try:
                self._model.fuse()
            except Exception:
                # Le fuse n'est pas disponible/utile pour tous les backends
                pass
            self._names = self._model.names
            self._configure_class_groups()
            rospy.loginfo(f"[OBJ] Modèle chargé ({device}) : {self._model_path}")
        except Exception as e:
            rospy.logerr(f"[OBJ] Chargement échoué : {e}")

    def _configure_class_groups(self):
        # Adapte les groupes à la nomenclature réelle du modèle pour éviter
        # les faux filtres quand les IDs changent entre checkpoints.
        if isinstance(self._names, dict):
            id_to_name = {int(k): str(v).lower() for k, v in self._names.items()}
        elif isinstance(self._names, (list, tuple)):
            id_to_name = {i: str(v).lower() for i, v in enumerate(self._names)}
        else:
            id_to_name = {}

        if not id_to_name:
            self._vehicle_ids = set(VEHICLE_IDS)
            self._light_ids = set(LIGHT_IDS)
            self._sign_ids = set(SIGN_IDS)
            self._all_active = list(self._vehicle_ids | self._light_ids | self._sign_ids)
            rospy.logwarn("[OBJ] Noms de classes indisponibles, fallback IDs par défaut")
            return

        available = set(id_to_name.keys())
        default_vehicle = set(VEHICLE_IDS) & available
        default_light = set(LIGHT_IDS) & available
        default_sign = set(SIGN_IDS) & available

        detected_vehicle = set()
        detected_light = set()
        detected_sign = set()

        for cls_id, n in id_to_name.items():
            if any(k in n for k in ('person', 'pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'motorbike')):
                detected_vehicle.add(cls_id)
            if ('red' in n and 'light' in n) or ('green' in n and 'light' in n):
                detected_light.add(cls_id)
            if (('stop' in n and 'sign' in n)
                    or ('speed' in n and 'limit' in n)
                    or ('turn' in n and ('left' in n or 'right' in n))
                    or ('mandatory' in n and ('left' in n or 'right' in n))):
                detected_sign.add(cls_id)

        self._vehicle_ids = detected_vehicle if detected_vehicle else default_vehicle
        self._light_ids = detected_light if detected_light else default_light
        self._sign_ids = detected_sign if detected_sign else default_sign

        # Si vraiment rien n'est trouvé, ne pas bloquer l'inférence par un filtre vide
        self._all_active = sorted(self._vehicle_ids | self._light_ids | self._sign_ids)
        rospy.loginfo(f"[OBJ] IDs classes -> veh:{sorted(self._vehicle_ids)} light:{sorted(self._light_ids)} sign:{sorted(self._sign_ids)}")

    def _category_from_name(self, name):
        n = str(name).lower().strip()
        if any(k in n for k in ('person', 'pedestrian', 'walker', 'car', 'vehicle', 'truck', 'bus', 'motorcycle', 'motorbike')):
            return 'vehicle'
        if ('red' in n and 'light' in n) or ('green' in n and 'light' in n):
            return 'light'
        if (('stop' in n and 'sign' in n)
                or ('speed' in n and 'limit' in n)
                or ('turn' in n and ('left' in n or 'right' in n))
                or ('mandatory' in n and ('left' in n or 'right' in n))):
            return 'sign'
        return None

    def _class_name(self, cls_id):
        # Certains modèles renvoient names en list, dict[int->str] ou dict[str->str]
        if isinstance(self._names, dict):
            if cls_id in self._names:
                return str(self._names[cls_id])
            key = str(cls_id)
            if key in self._names:
                return str(self._names[key])
            return str(cls_id)
        if isinstance(self._names, (list, tuple)) and 0 <= cls_id < len(self._names):
            return str(self._names[cls_id])
        return str(cls_id)

    def _category(self, cls_id, name):
        by_name = self._category_from_name(name)
        if by_name is not None:
            return by_name
        # Fallback via IDs détectés au chargement du modèle
        if cls_id in self._vehicle_ids:
            return 'vehicle'
        if cls_id in self._light_ids:
            return 'light'
        if cls_id in self._sign_ids:
            return 'sign'
        return None

    def _cb_frame(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._lock:
                self._frame = frame
        except Exception as e:
            rospy.logwarn_throttle(5, f"[OBJ] Erreur réception image: {e}")

    def run(self):
        if self._enable_visual:
            cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Object Detection', 860, 490)

        rate = rospy.Rate(self._loop_hz if self._loop_hz > 0 else 60.0)
        prev_t = time.time()

        while not rospy.is_shutdown():
            with self._lock:
                frame = self._frame.copy() if self._frame is not None else None

            if frame is not None and self._model is not None:
                now    = time.time()
                fps    = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                annotated, dets = self._process(frame, draw=self._enable_visual)
                if self._enable_visual and annotated is not None:
                    cv2.rectangle(annotated, (0, 0), (frame.shape[1], 26), (10, 10, 10), -1)
                    cv2.putText(annotated, f'FPS:{fps:.0f}  Objets:{len(dets)}',
                                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.imshow('Object Detection', annotated)
            else:
                if self._enable_visual:
                    wait = self._wait_frame.copy()
                    if self._model is None:
                        cv2.rectangle(wait, (0, 0), (640, 26), (10, 10, 10), -1)
                        cv2.putText(wait, 'Chargement modele ...',
                                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.imshow('Object Detection', wait)

            if self._enable_visual:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            rate.sleep()

        if self._enable_visual:
            cv2.destroyAllWindows()

    def _process(self, frame, draw=False):
        dets      = self._run_inference(frame)
        
        # ── Détection feu par couleur en backup ────────────────────────────
        # Si aucun feu détecté par le modèle, chercher par couleur
        has_light = any(d['label'] in ('feu_rouge', 'feu_vert') for d in dets)
        if not has_light:
            color_light = self._detect_light_by_color(frame)
            if color_light is not None:
                dets.append(color_light)
        
        det_str   = self._build_det_str(dets)
        need_draw = draw or (self._pub_img is not None)
        annotated = self._annotate(frame, dets) if need_draw else None
        try:
            self._pub_det.publish(String(data=det_str))
            if self._pub_img is not None and annotated is not None:
                self._pub_img.publish(self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(5, f"[OBJ] Erreur publication: {e}")
        summary = self._detection_summary(dets)
        if det_str != 'none':
            rospy.loginfo_throttle(max(0.2, self._log_det_every_sec), f'[OBJ] {summary} | {det_str}')
        elif self._log_none:
            rospy.loginfo_throttle(max(0.2, self._log_det_every_sec), f'[OBJ] {summary} | none')
        return annotated, dets

    def _detect_light_by_color(self, frame):
        """
        Détecte les feux tricolores par couleur en backup du modèle.
        Cherche dans le tiers supérieur de l'image (zone des feux).
        Retourne un détail si rouge/vert trouvé, sinon None.
        """
        try:
            h, w = frame.shape[:2]
            roi_h = max(1, h // 3)  # Région supérieure
            roi = frame[0:roi_h, :, :]
            
            # Convertir en HSV pour meilleure détection couleur
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Plages de teinte pour rouge et vert en HSV
            # Rouge : 0-10 et 170-180 (circulaire)
            # Vert : 35-85
            
            red_low1 = np.array([0, 100, 100])
            red_high1 = np.array([10, 255, 255])
            red_low2 = np.array([170, 100, 100])
            red_high2 = np.array([180, 255, 255])
            
            green_low = np.array([35, 100, 100])
            green_high = np.array([85, 255, 255])
            
            mask_red = cv2.inRange(hsv, red_low1, red_high1) | cv2.inRange(hsv, red_low2, red_high2)
            mask_green = cv2.inRange(hsv, green_low, green_high)
            
            # Compter pixels pour déterminer couleur dominante
            red_cnt = cv2.countNonZero(mask_red)
            green_cnt = cv2.countNonZero(mask_green)
            
            # Seuil minimum de pixels (au moins 200 pixels détectés)
            threshold = 200
            
            if red_cnt > threshold and red_cnt > green_cnt:
                # Feu rouge détecté
                return {
                    'label': 'feu_rouge',
                    'detail': 'rouge',
                    'conf': 0.65,
                    'x1': w // 4,
                    'y1': 10,
                    'x2': 3 * w // 4,
                    'y2': roi_h - 10
                }
            elif green_cnt > threshold and green_cnt > red_cnt:
                # Feu vert détecté
                return {
                    'label': 'feu_vert',
                    'detail': 'vert',
                    'conf': 0.65,
                    'x1': w // 4,
                    'y1': 10,
                    'x2': 3 * w // 4,
                    'y2': roi_h - 10
                }
        except Exception as e:
            rospy.logwarn_throttle(5, f'[OBJ] Erreur détection couleur: {e}')
        
        return None

    def _detection_summary(self, dets):
        if not dets:
            return 'lights=0 signs=0 vehicles=0'
        n_l = sum(1 for d in dets if d['label'] in ('feu_rouge', 'feu_vert'))
        n_s = sum(1 for d in dets if d['label'] in ('stop', 'vitesse', 'droite', 'gauche'))
        n_v = max(0, len(dets) - n_l - n_s)
        return f'lights={n_l} signs={n_s} vehicles={n_v}'

    def _run_inference(self, frame):
        dets = []
        try:
            # Inférence avec le seuil le plus bas pour tout récupérer
            infer_conf = min(self._conf_vehicles, self._conf_signs)
            classes_arg = self._all_active if (self._use_class_filter and self._all_active) else None
            if self._torch is not None:
                with self._torch.inference_mode():
                    results = self._model(frame, imgsz=self._infer_size,
                                          conf=infer_conf,
                                          classes=classes_arg, verbose=False)
            else:
                results = self._model(frame, imgsz=self._infer_size,
                                      conf=infer_conf,
                                      classes=classes_arg, verbose=False)

            light_candidates = []
            vehicle_candidates = []
            sign_candidates = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    name   = self._class_name(cls_id)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    category = self._category(cls_id, name)

                    if category == 'vehicle':
                        if conf >= self._conf_vehicles:
                            det = self._make_det(frame, name, conf, x1, y1, x2, y2)
                            if det is not None:
                                vehicle_candidates.append(det)

                    elif category == 'light':
                        det = self._make_det(frame, name, conf, x1, y1, x2, y2)
                        if det is not None:
                            area = (det['x2'] - det['x1']) * (det['y2'] - det['y1'])
                            if conf >= self._conf_lights and area >= self._light_min_area:
                                light_candidates.append(det)

                    elif category == 'sign':
                        if conf >= self._conf_signs:
                            det = self._make_det(frame, name, conf, x1, y1, x2, y2)
                            if det is not None:
                                w = det['x2'] - det['x1']
                                h = det['y2'] - det['y1']
                                area = w * h
                                ar = (float(w) / float(h)) if h > 0 else 0.0

                                # Filtres géométriques anti faux panneaux
                                if area < self._sign_min_area:
                                    continue
                                if ar < self._sign_min_ar or ar > self._sign_max_ar:
                                    continue

                                # Les panneaux vitesse sont souvent les plus bruités -> seuil dédié
                                if det['label'] == 'vitesse' and conf < self._conf_speed_signs:
                                    continue
                                sign_candidates.append(det)

            # Garder seulement les N feux les plus confiants
            light_candidates.sort(key=lambda d: d['conf'], reverse=True)
            vehicle_candidates = self._deduplicate(vehicle_candidates)
            sign_candidates = self._deduplicate(sign_candidates)
            light_candidates = self._deduplicate(light_candidates)

            vehicle_candidates.sort(key=lambda d: d['conf'], reverse=True)
            sign_candidates.sort(key=lambda d: d['conf'], reverse=True)

            dets.extend(vehicle_candidates[:max(self._max_veh_keep, 0)])
            dets.extend(sign_candidates[:max(self._max_sign_keep, 0)])
            dets.extend(light_candidates[:max(self._light_max_keep, 0)])

            dets = self._apply_sticky(dets)

        except Exception as e:
            rospy.logwarn_throttle(5, f'[OBJ] Inference error: {e}')
        return dets

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a['x1'], a['y1'], a['x2'], a['y2']
        bx1, by1, bx2, by2 = b['x1'], b['y1'], b['x2'], b['y2']
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        return float(inter) / float(max(1, union))

    def _deduplicate(self, dets):
        """Supprime les boites quasi identiques par label en gardant la plus confiante."""
        if not dets:
            return dets
        kept = []
        for d in sorted(dets, key=lambda x: x['conf'], reverse=True):
            duplicate = False
            for k in kept:
                if d['label'] != k['label']:
                    continue
                if self._iou(d, k) >= self._dedup_iou_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(d)
        return kept

    def _apply_sticky(self, dets):
        """Maintien court des détections critiques pour lisser le flicker."""
        now = time.time()
        sticky_labels = {'feu_rouge', 'feu_vert', 'stop', 'droite', 'gauche', 'vitesse'}

        live_labels = set()
        for d in dets:
            label = d['label']
            if label in sticky_labels:
                live_labels.add(label)
                self._sticky_det[label] = d
                self._sticky_until[label] = now + max(0.0, self._sticky_hold_sec)

        for label in list(self._sticky_until.keys()):
            if self._sticky_until.get(label, 0.0) < now:
                self._sticky_until.pop(label, None)
                self._sticky_det.pop(label, None)

        for label, expiry in self._sticky_until.items():
            if label in live_labels:
                continue
            if expiry >= now and label in self._sticky_det:
                cached = dict(self._sticky_det[label])
                cached['conf'] = min(float(cached.get('conf', 0.0)), 0.99)
                dets.append(cached)

        return dets

    def _make_det(self, frame, name, conf, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None

        raw_label = LABEL_MAP.get(name.lower(), name.lower())
        detail    = 'none'
        if raw_label == 'feu_rouge':
            detail = 'rouge'
        elif raw_label == 'feu_vert':
            detail = 'vert'
        elif raw_label == 'stop':
            detail = 'stop'
        elif raw_label == 'vitesse':
            # Conserver la valeur du panneau (ex: speed limit 50 -> detail=50)
            parts = name.lower().split()
            if parts and parts[-1].isdigit():
                detail = parts[-1]
        elif raw_label == 'droite':
            detail = 'droite'
        elif raw_label == 'gauche':
            detail = 'gauche'
        return {'label': raw_label, 'detail': detail,
                'conf': conf, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    def _build_det_str(self, dets):
        if not dets:
            return 'none'
        return '|'.join(
            f"{d['label']}:{d['detail']}:{d['conf']:.2f}"
            f":{d['x1']}:{d['y1']}:{d['x2']}:{d['y2']}"
            for d in dets)

    def _annotate(self, frame, dets):
        out = frame.copy()
        for d in dets:
            label  = d['label']
            detail = d['detail']
            conf   = d['conf']
            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            color = BOX_COLORS.get(label, BOX_COLORS['default'])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if detail not in ('none', ''):
                txt = f'{label.upper()} [{detail}] {conf:.0%}'
            else:
                txt = f'{label.upper()} {conf:.0%}'
            font      = cv2.FONT_HERSHEY_SIMPLEX
            scale     = 0.50
            thickness = 1
            tw, th    = cv2.getTextSize(txt, font, scale, thickness)[0]
            ty        = max(y1 - 4, th + 2)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(out, txt, (x1 + 2, ty - 2), font, scale,
                        (255, 255, 255), thickness)
        return out


if __name__ == '__main__':
    try:
        node = ObjectDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
