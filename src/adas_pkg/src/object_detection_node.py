#!/usr/bin/env python3
"""
object_detection_node.py — Détection temps réel double modèle

  best-11.pt  → feux (red_light, green_light) + stop_sign + speed_limit + turn_right
  yolov8n.pt  → piétons (person) + voitures (car) — COCO pré-entraîné

Publie :
  /adas/detection        (String)  — pipe-separated: class:detail:conf:x1:y1:x2:y2
  /adas/detection_image  (Image)   — frame annotée
Souscrit :
  /carla/camera/rgb      (Image)
"""

import os
import threading
import time
from collections import deque
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SRC_DIR, 'best-11.pt')
MODEL_COCO = os.path.join(_SRC_DIR, 'yolov8n.pt')

INFER_SIZE      = 896
INFER_SIZE_COCO = 640

# Seuils best-11 (feux + stop)
CONF_LIGHTS   = 0.62   # strict — élimine les détections ambiguës
CONF_SIGNS    = 0.04   # stop uniquement

# Seuils COCO (piétons + voitures)
CONF_COCO     = 0.40
COCO_PERSON   = 0
COCO_CAR      = 2

LIGHT_MIN_AREA = 600   # px² — rejette les feux trop petits/lointains
LIGHT_MAX_KEEP = 1     # 1 seul feu publié par frame (le plus confiant)
LIGHT_CONF_MARGIN = 0.22  # écart minimum rouge/vert pour trancher sans ambiguïté
SIGN_MIN_AREA  = 250   # px² — accepte les panneaux plus petits
SIGN_MIN_AR    = 0.45  # ratio w/h minimal attendu
SIGN_MAX_AR    = 1.65  # ratio w/h maximal attendu

DEDUP_IOU_THRESH = 0.55   # suppression de doublons bbox
MAX_VEH_KEEP     = 30     # max véhicules publiés par frame
MAX_SIGN_KEEP    = 6      # max panneaux publiés par frame
STICKY_HOLD_SEC  = 1.20   # maintien un peu plus long pour feux/panneaux
STICKY_CONF_DECAY = 0.82  # décroissance confiance sticky par tick (évite faux persistants)
LOG_DET_EVERY_SEC = 1.0   # cadence d'affichage terminal
LOG_NONE          = True  # afficher aussi 'none' pour diagnostiquer le flux
USE_CLASS_FILTER  = True  # filtrer sur classes utiles du modèle

# Vote temporel feux — élimine les faux positifs par consensus strict
LIGHT_VOTE_WINDOW  = 8    # fenêtre glissante (frames)
LIGHT_VOTE_THRESH  = 6    # 6/8 frames = 75% de consensus requis
LIGHT_MIN_REL_AREA = 8e-4 # 0.08 % surface frame — filtre relatif plus strict
LIGHT_COLOR_BACKUP = False # backup HSV désactivé — source de faux positifs

# IDs best-11 actifs (vehicles retirés → gérés par COCO)
VEHICLE_IDS = set()          # non utilisé pour best-11
LIGHT_IDS   = {2, 3}        # red_light, green_light
SIGN_IDS    = {4, 5, 6, 7}  # stop_sign, speed_limit_10, speed_limit_30, turn_right

ALL_ACTIVE  = list(LIGHT_IDS | SIGN_IDS)

LABEL_MAP = {
    'person':         'pieton',
    'vehicle':        'voiture',
    'red_light':      'feu_rouge',
    'green_light':    'feu_vert',
    'stop_sign':      'stop',
    'speed_limit_10': 'vitesse',
    'speed_limit_30': 'vitesse',
    'turn_right':     'droite',
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
    'droite':    (200, 200, 0),
    'default':   (128, 128, 128),
}


class ObjectDetectionNode:

    def __init__(self):
        rospy.init_node('object_detection_node', anonymous=False)
        self._bridge     = CvBridge()
        self._lock       = threading.Lock()
        self._frame      = None
        self._model      = None   # best-11 : feux + stop
        self._model_coco = None   # yolov8n : piétons + voitures
        self._torch      = None
        self._names      = {}
        self._vehicle_ids = set()          # vide — best-11 ne détecte plus les véhicules
        self._light_ids   = set(LIGHT_IDS)
        self._sign_ids    = set(SIGN_IDS)
        self._all_active  = list(ALL_ACTIVE)
        self._sticky_until = {}
        self._sticky_det   = {}
        # Vote temporel feux : deque de labels (str|None) sur la fenêtre glissante
        self._light_history: deque = deque(maxlen=LIGHT_VOTE_WINDOW)

        # Paramètres runtime (tuning sans modifier le code)
        self._model_path         = rospy.get_param('~model_path', MODEL_PATH)
        self._model_coco_path    = rospy.get_param('~model_coco_path', MODEL_COCO)
        self._infer_size         = int(rospy.get_param('~infer_size', INFER_SIZE))
        self._conf_lights        = float(rospy.get_param('~conf_lights', CONF_LIGHTS))
        self._conf_signs         = float(rospy.get_param('~conf_signs', CONF_SIGNS))
        self._conf_coco          = float(rospy.get_param('~conf_coco',   CONF_COCO))
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
        self._use_color_backup   = bool(rospy.get_param('~use_color_backup', LIGHT_COLOR_BACKUP))
        self._light_min_rel_area = float(rospy.get_param('~light_min_rel_area', LIGHT_MIN_REL_AREA))
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
        rospy.loginfo("Object Detection Node prêt — best-11 (feux+stop) + YOLOv8n (piétons+voitures)")

    def _resolve_model_path(self, candidate_path):
        """Privilégie best-11.pt et fournit un fallback robuste."""
        requested = str(candidate_path).strip() if candidate_path is not None else ''
        candidates = []
        if requested:
            candidates.append(requested)
        if MODEL_PATH not in candidates:
            candidates.append(MODEL_PATH)

        for p in candidates:
            if os.path.exists(p):
                if os.path.basename(p).lower() != 'best-11.pt':
                    rospy.logwarn(f"[OBJ] Modèle non standard sélectionné: {p} (recommandé: best-11.pt)")
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

            # ── best-11 : feux + stop ──────────────────────────────────────
            self._model = YOLO(self._model_path)
            self._model.to(device)
            try:
                self._model.fuse()
            except Exception:
                pass
            self._names = self._model.names
            self._configure_class_groups()
            # Forcer : best-11 ne détecte plus les véhicules (gérés par COCO)
            self._vehicle_ids = set()
            self._all_active  = sorted(self._light_ids | self._sign_ids)
            rospy.loginfo(f"[OBJ] best-11 chargé ({device}) — feux + stop + speed_limit + turn_right")

            # ── yolov8n COCO : piétons + voitures ─────────────────────────
            try:
                if not os.path.exists(self._model_coco_path):
                    rospy.logerr(f"[OBJ] Modèle COCO introuvable: {self._model_coco_path}")
                else:
                    self._model_coco = YOLO(self._model_coco_path)
                    self._model_coco.to(device)
                    rospy.loginfo(f"[OBJ] YOLOv8n COCO chargé ({device}) — piétons + voitures")
            except Exception as e:
                rospy.logwarn(f"[OBJ] YOLOv8n non chargé : {e}")

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
            if any(k in n for k in ('person', 'vehicle')):
                detected_vehicle.add(cls_id)
            if 'red_light' in n or 'green_light' in n or (('red' in n or 'green' in n) and 'light' in n):
                detected_light.add(cls_id)
            if (('stop' in n and 'sign' in n)
                    or ('speed' in n and 'limit' in n)
                    or ('turn' in n and 'right' in n)):
                detected_sign.add(cls_id)

        self._vehicle_ids = detected_vehicle if detected_vehicle else default_vehicle
        self._light_ids = detected_light if detected_light else default_light
        self._sign_ids = detected_sign if detected_sign else default_sign

        # Si vraiment rien n'est trouvé, ne pas bloquer l'inférence par un filtre vide
        self._all_active = sorted(self._vehicle_ids | self._light_ids | self._sign_ids)
        rospy.loginfo(f"[OBJ] IDs classes -> veh:{sorted(self._vehicle_ids)} light:{sorted(self._light_ids)} sign:{sorted(self._sign_ids)}")

    def _category_from_name(self, name):
        n = str(name).lower().strip()
        if any(k in n for k in ('person', 'vehicle')):
            return 'vehicle'
        if 'red_light' in n or 'green_light' in n or (('red' in n or 'green' in n) and 'light' in n):
            return 'light'
        if (('stop' in n and 'sign' in n)
                or ('speed' in n and 'limit' in n)
                or ('turn' in n and 'right' in n)):
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

            if frame is not None and (self._model is not None or self._model_coco is not None):
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
        Backup HSV : ne se déclenche que si un blob compact et très lumineux
        est trouvé dans le tiers supérieur de l'image.

        Les feux tricolores dans CARLA sont de petits disques très saturés et
        très brillants (S>180, V>200). Les bâtiments/voitures rouges ont une
        saturation/luminosité bien plus faible → ils ne passent pas ce filtre.

        Contraintes blob (anti-faux-positifs) :
          - aire entre BLOB_MIN_PX et BLOB_MAX_PX (pas trop petit, pas trop grand)
          - circularité  > BLOB_MIN_CIRC  (forme ronde, pas un rectangle)
          - ratio hauteur/largeur proche de 1 (carré/cercle attendu)
        """
        # Seuils blob — feux CARLA sont de petits disques lumineux
        BLOB_MIN_PX   = 40     # px² minimum par blob
        BLOB_MAX_PX   = 1800   # px² maximum (exclut grandes surfaces rouges)
        BLOB_MIN_CIRC = 0.35   # circularité minimale (4π·A / P²)

        try:
            h, w = frame.shape[:2]
            roi_h = max(1, h // 3)
            roi   = frame[0:roi_h, :, :]
            hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # HSV très strict : saturation ≥ 180 et valeur ≥ 200
            # Élimine bâtiments, voitures, tout ce qui n'est pas un feu allumé
            red1_lo = np.array([0,   180, 200])
            red1_hi = np.array([10,  255, 255])
            red2_lo = np.array([170, 180, 200])
            red2_hi = np.array([180, 255, 255])
            grn_lo  = np.array([45,  180, 200])
            grn_hi  = np.array([80,  255, 255])

            mask_r = cv2.inRange(hsv, red1_lo, red1_hi) | cv2.inRange(hsv, red2_lo, red2_hi)
            mask_g = cv2.inRange(hsv, grn_lo,  grn_hi)

            # Nettoyage morphologique : supprime le bruit pixel isolé
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k)
            mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, k)

            def _best_blob(mask):
                """Retourne (score, cx, cy, bx1, by1, bx2, by2) du meilleur blob, ou None."""
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best = None
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if not (BLOB_MIN_PX <= area <= BLOB_MAX_PX):
                        continue
                    perim = cv2.arcLength(cnt, True)
                    if perim < 1:
                        continue
                    circ = (4.0 * np.pi * area) / (perim * perim)
                    if circ < BLOB_MIN_CIRC:
                        continue
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    ratio = float(bw) / float(bh) if bh > 0 else 0
                    if not (0.4 <= ratio <= 2.5):
                        continue
                    score = area * circ
                    if best is None or score > best[0]:
                        best = (score, x + bw // 2, y + bh // 2,
                                x, y, x + bw, y + bh)
                return best

            blob_r = _best_blob(mask_r)
            blob_g = _best_blob(mask_g)

            if blob_r is None and blob_g is None:
                return None

            # Choisir la couleur avec le meilleur score blob
            if blob_r and (blob_g is None or blob_r[0] >= blob_g[0]):
                _, _, _, bx1, by1, bx2, by2 = blob_r
                return {'label': 'feu_rouge', 'detail': 'rouge', 'conf': 0.55,
                        'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2}
            elif blob_g:
                _, _, _, bx1, by1, bx2, by2 = blob_g
                return {'label': 'feu_vert', 'detail': 'vert', 'conf': 0.55,
                        'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2}

        except Exception as e:
            rospy.logwarn_throttle(5, f'[OBJ] Erreur détection couleur: {e}')

        return None

    def _verify_light_color(self, frame, det):
        """Corrige la couleur d'un feu en analysant les pixels HSV à l'intérieur
        de sa bounding box.  Le modèle YOLO peut confondre rouge/vert ; l'analyse
        directe des pixels de la bbox est une vérité terrain fiable.

        Retourne 'feu_rouge', 'feu_vert', ou None si trop ambigu pour conclure.
        """
        MIN_PIX   = 8    # pixels colorés minimum pour décider
        RATIO_MIN = 2.0  # couleur dominante doit être 2× plus présente

        try:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return det['label']

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Seuils HSV stricts — uniquement les pixels très lumineux/saturés
            # (le bulbe allumé du feu, pas le boîtier noir)
            r1 = cv2.inRange(hsv, np.array([0,   160, 160]), np.array([10,  255, 255]))
            r2 = cv2.inRange(hsv, np.array([170, 160, 160]), np.array([180, 255, 255]))
            g  = cv2.inRange(hsv, np.array([45,  160, 160]), np.array([80,  255, 255]))

            red_px   = int(cv2.countNonZero(r1)) + int(cv2.countNonZero(r2))
            green_px = int(cv2.countNonZero(g))

            if red_px < MIN_PIX and green_px < MIN_PIX:
                return det['label']   # bbox trop sombre → garder prédiction modèle

            if red_px >= green_px * RATIO_MIN:
                return 'feu_rouge'
            if green_px >= red_px * RATIO_MIN:
                return 'feu_vert'

            # Ambiguïté — garder prédiction modèle
            return det['label']

        except Exception:
            return det['label']

    def _vote_light(self, current_label):
        """Vote temporel sur la couleur du feu pour supprimer le flicker frame-à-frame.

        Pousse le label courant (ou None) dans la fenêtre glissante et retourne
        la couleur majoritaire seulement si elle atteint LIGHT_VOTE_THRESH votes.
        """
        self._light_history.append(current_label)
        votes: dict = {}
        for lbl in self._light_history:
            if lbl is not None:
                votes[lbl] = votes.get(lbl, 0) + 1
        if not votes:
            return None
        best, count = max(votes.items(), key=lambda x: x[1])
        return best if count >= LIGHT_VOTE_THRESH else None

    def _detection_summary(self, dets):
        if not dets:
            return 'lights=0 signs=0 vehicles=0'
        n_l = sum(1 for d in dets if d['label'] in ('feu_rouge', 'feu_vert'))
        n_s = sum(1 for d in dets if d['label'] in ('stop', 'vitesse', 'droite'))
        n_v = max(0, len(dets) - n_l - n_s)
        return f'lights={n_l} signs={n_s} vehicles={n_v}'

    def _run_inference(self, frame):
        dets = []
        try:
            fh, fw = frame.shape[:2]
            frame_area = max(1, fh * fw)

            # ── best-11 : feux + stop ──────────────────────────────────────
            if self._model is not None:
                classes_arg = self._all_active if (self._use_class_filter and self._all_active) else None
                if self._torch is not None:
                    with self._torch.inference_mode():
                        results = self._model(frame, imgsz=self._infer_size,
                                              conf=self._conf_signs,
                                              classes=classes_arg, verbose=False)
                else:
                    results = self._model(frame, imgsz=self._infer_size,
                                          conf=self._conf_signs,
                                          classes=classes_arg, verbose=False)

                light_candidates = []
                sign_candidates  = []

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf   = float(box.conf[0])
                        name   = self._class_name(cls_id)
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        category = self._category(cls_id, name)

                        if category == 'light':
                            det = self._make_det(frame, name, conf, x1, y1, x2, y2)
                            if det is not None:
                                area     = (det['x2'] - det['x1']) * (det['y2'] - det['y1'])
                                rel_area = area / frame_area
                                if (conf >= self._conf_lights
                                        and area >= self._light_min_area
                                        and rel_area >= self._light_min_rel_area):
                                    verified = self._verify_light_color(frame, det)
                                    if verified != det['label']:
                                        det = dict(det)
                                        det['label']  = verified
                                        det['detail'] = 'rouge' if verified == 'feu_rouge' else 'vert'
                                    light_candidates.append(det)

                        elif category == 'sign':
                            if conf >= self._conf_signs:
                                det = self._make_det(frame, name, conf, x1, y1, x2, y2)
                                if det is not None:
                                    w  = det['x2'] - det['x1']
                                    h  = det['y2'] - det['y1']
                                    ar = (float(w) / float(h)) if h > 0 else 0.0
                                    if w * h >= self._sign_min_area and self._sign_min_ar <= ar <= self._sign_max_ar:
                                        sign_candidates.append(det)

                sign_candidates  = self._deduplicate(sign_candidates)
                light_candidates = self._deduplicate(light_candidates)
                sign_candidates.sort(key=lambda d: d['conf'], reverse=True)
                light_candidates.sort(key=lambda d: d['conf'], reverse=True)

                # Arbitrage rouge / vert
                if light_candidates:
                    red   = [d for d in light_candidates if d['label'] == 'feu_rouge']
                    green = [d for d in light_candidates if d['label'] == 'feu_vert']
                    top_r, top_g = (red[0] if red else None), (green[0] if green else None)
                    if top_r and top_g:
                        light_candidates = ([top_r] if top_r['conf'] - top_g['conf'] >= LIGHT_CONF_MARGIN
                                            else [top_g] if top_g['conf'] - top_r['conf'] >= LIGHT_CONF_MARGIN
                                            else [])
                    else:
                        light_candidates = [top_r or top_g]

                if not light_candidates and self._use_color_backup:
                    backup = self._detect_light_by_color(frame)
                    if backup:
                        light_candidates = [backup]

                # Vote temporel feux
                if light_candidates:
                    voted = self._vote_light(light_candidates[0]['label'])
                    light_candidates = ([d for d in light_candidates if d['label'] == voted]
                                        if voted else [])
                else:
                    self._vote_light(None)

                dets.extend(sign_candidates[:max(self._max_sign_keep, 0)])
                dets.extend(light_candidates[:max(self._light_max_keep, 0)])

            # ── yolov8n COCO : piétons + voitures ─────────────────────────
            if self._model_coco is not None:
                try:
                    if self._torch is not None:
                        with self._torch.inference_mode():
                            coco_results = self._model_coco(
                                frame, imgsz=INFER_SIZE_COCO,
                                conf=self._conf_coco,
                                classes=[COCO_PERSON, COCO_CAR],
                                verbose=False)
                    else:
                        coco_results = self._model_coco(
                            frame, imgsz=INFER_SIZE_COCO,
                            conf=self._conf_coco,
                            classes=[COCO_PERSON, COCO_CAR],
                            verbose=False)

                    coco_label = {COCO_PERSON: ('pieton', 'pieton'),
                                  COCO_CAR:    ('voiture', 'voiture')}
                    veh_candidates = []
                    for r in coco_results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf   = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            label, detail = coco_label.get(cls_id, ('unknown', 'unknown'))
                            h, w = frame.shape[:2]
                            x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
                            y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
                            if x2 > x1 and y2 > y1:
                                veh_candidates.append({'label': label, 'detail': detail,
                                                       'conf': conf,
                                                       'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                    veh_candidates = self._deduplicate(veh_candidates)
                    veh_candidates.sort(
                        key=lambda d: (d['x2'] - d['x1']) * (d['y2'] - d['y1']) * d['conf'],
                        reverse=True)
                    dets.extend(veh_candidates[:max(self._max_veh_keep, 0)])
                except Exception as e:
                    rospy.logwarn_throttle(5, f'[OBJ] COCO inference error: {e}')

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
        sticky_labels = {'feu_rouge', 'feu_vert', 'stop', 'droite', 'vitesse'}

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
                # Décroissance : la confiance diminue à chaque tick sans re-détection
                decayed = float(cached.get('conf', 0.0)) * STICKY_CONF_DECAY
                if decayed < 0.10:
                    continue  # trop faible : ne pas republier
                cached['conf'] = decayed
                self._sticky_det[label]['conf'] = decayed  # propager pour le prochain tick
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
            # Conserver la valeur du panneau (ex: speed_limit_30 -> detail=30)
            parts = name.lower().replace('-', '_').split('_')
            if parts and parts[-1].isdigit():
                detail = parts[-1]
        elif raw_label == 'droite':
            detail = 'droite'
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

            is_vehicle = label in ('pieton', 'voiture')
            box_th = 3 if is_vehicle else 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.62 if is_vehicle else 0.50
            thickness = 2 if is_vehicle else 1

            # Double-border box for better contrast
            outer = (10, 10, 10) if is_vehicle else (230, 230, 230)
            cv2.rectangle(out, (x1, y1), (x2, y2), outer, box_th + 1)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, box_th)

            label_disp = label.replace('_', ' ')
            if is_vehicle:
                txt = f'{label_disp} {conf:.0%}'
            elif detail not in ('none', '') and detail != label:
                txt = f'{label_disp} {detail} {conf:.0%}'
            else:
                txt = f'{label_disp} {conf:.0%}'

            tw, th = cv2.getTextSize(txt, font, scale, thickness)[0]
            pad_x = 8 if is_vehicle else 6
            pad_y = 6 if is_vehicle else 4
            h, w = out.shape[:2]

            if is_vehicle:
                ty = max(y1 - 6, th + pad_y)
                x0 = max(0, x1 - 2)
                y0 = max(0, ty - th - pad_y)
            else:
                y0 = min(h - 1, y2 + 4)
                ty = min(h - 2, y0 + th + pad_y)
                x0 = max(0, x1 - 2)
                if y0 + th + pad_y >= h:
                    ty = max(y1 - 6, th + pad_y)
                    y0 = max(0, ty - th - pad_y)

            x1b = min(w - 1, x1 + tw + pad_x)
            y1b = min(h - 1, y0 + th + pad_y)

            shadow = 2 if is_vehicle else 1
            cv2.rectangle(out, (x0 + shadow, y0 + shadow), (x1b + shadow, y1b + shadow), (0, 0, 0), -1)
            overlay = out.copy()
            cv2.rectangle(overlay, (x0, y0), (x1b, y1b), color, -1)
            alpha = 0.85 if is_vehicle else 0.78
            cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
            cv2.putText(out, txt, (x1 + 2, y0 + th + pad_y - 2), font, scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
        return out


if __name__ == '__main__':
    try:
        node = ObjectDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
