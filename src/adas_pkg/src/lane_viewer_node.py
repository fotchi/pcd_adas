#!/usr/bin/env python3
"""
lane_viewer_node.py — Détection de voies CARLA (vision classique HLS, sans modèle CNN)
Approche : seuillage couleur HLS — jaune (ligne centrale) + blanc (bord droit)
Publie  : /lane/deviation  (Float32)
          /lane/status     (String)    CENTER_OK | DRIFT_LEFT | DRIFT_RIGHT | NO_LANE
          /lane/image      (Image)
Souscrit: /carla/camera/rgb  (Image)
          /adas/control      (String)

Convention déviation :
  dev > 0  →  centre voie à DROITE de l'image  →  voiture a dérivé à GAUCHE  →  STEER_RIGHT
  dev < 0  →  centre voie à GAUCHE de l'image  →  voiture a dérivé à DROITE  →  STEER_LEFT
"""

import threading, time
import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

# ── Paramètres ────────────────────────────────────────────────────────────────
N_STRIPS      = 20     # bandes verticales sliding-window
WIN_MARGIN    = 70     # demi-largeur fenêtre de recherche (px)
MIN_HIST      = 1.5    # somme minimum histogramme (masque float32)
MIN_WIN       = 0.5    # somme minimum fenêtre
MIN_PTS       = 3      # points minimum pour valider une voie
DEV_THRESH    = 0.04   # déviation > 4 % → label DRIFT
DEV_SMOOTH    = 0.40   # EMA — lissage temporel
HOLD_FRAMES   = 10     # frames de maintien si perte de détection
BASE_SEARCH_W = 0.13   # demi-largeur recherche autour position précédente (ratio image) — serré pour éviter lignes transversales
MAX_DEV_JUMP  = 0.065  # saut max déviation par frame — anti-inversion sens inverse
BASE_EMA      = 0.12   # inertie forte sur les bases — résiste aux lignes d'intersection
POLY_SMOOTH   = 0.15   # lissage EMA des coefficients polynomiaux (zone verte)
MAX_BASE_JUMP = 0.10   # saut max absolu d'une base par frame (ratio image) — bloque lignes transversales

# Seuils couleur HLS pour CARLA (environnement synthétique)
#  Jaune : ligne centrale — TOUJOURS à gauche de notre voie
HLS_YELLOW_LO = np.array([15,  60,  80], dtype=np.uint8)
HLS_YELLOW_HI = np.array([40, 220, 255], dtype=np.uint8)
#  Blanc : lignes de bord de voie (droit)
HLS_WHITE_LO  = np.array([ 0, 175,   0], dtype=np.uint8)
HLS_WHITE_HI  = np.array([180, 255,  80], dtype=np.uint8)


class LaneDetectionNode:

    def __init__(self):
        rospy.init_node('lane_detection_node', anonymous=True)

        self._bridge     = CvBridge()
        self._lock       = threading.Lock()
        self._frame      = None
        self._last_cmd   = 'WAIT'
        self._dev_prev   = 0.0
        self._lw_px      = None   # largeur voie estimée (px)
        self._prev_lx    = None   # base jaune lissée (EMA)
        self._prev_rx    = None   # base blanche lissée (EMA)
        self._no_det_cnt = 0
        self._coef_l     = None   # coefficients polynomiaux gauche lissés (EMA)
        self._coef_r     = None   # coefficients polynomiaux droite lissés (EMA)

        self._pub_dev = rospy.Publisher('/lane/deviation', Float32, queue_size=1)
        self._pub_st  = rospy.Publisher('/lane/status',    String,  queue_size=1)
        self._pub_img = rospy.Publisher('/lane/image',     Image,   queue_size=1)

        rospy.Subscriber('/carla/camera/rgb', Image,  self._cb_frame, queue_size=1)
        rospy.Subscriber('/adas/control',     String, self._cb_cmd,   queue_size=1)
        rospy.loginfo("Lane Detection Node prêt — vision classique HLS (sans modèle CNN)")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_frame(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._lock:
                self._frame = frame
        except Exception:
            pass

    def _cb_cmd(self, msg):
        with self._lock:
            self._last_cmd = msg.data

    # ── Détection couleur ─────────────────────────────────────────────────────

    def _cv_masks(self, frame):
        """Seuillage HLS → masques jaune, blanc, combiné (float32 0-1)."""
        h, w = frame.shape[:2]

        # Amélioration contraste légère pour normaliser la luminosité CARLA
        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

        hls = cv.cvtColor(enhanced, cv.COLOR_BGR2HLS)
        yellow = cv.inRange(hls, HLS_YELLOW_LO, HLS_YELLOW_HI)
        white  = cv.inRange(hls, HLS_WHITE_LO,  HLS_WHITE_HI)

        # ROI trapézoïdal — élimine le ciel et les bords latéraux
        roi = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[
            (int(w * 0.05), h - 1),
            (int(w * 0.95), h - 1),
            (int(w * 0.68), int(h * 0.48)),
            (int(w * 0.32), int(h * 0.48)),
        ]], dtype=np.int32)
        cv.fillPoly(roi, pts, 255)
        yellow = cv.bitwise_and(yellow, roi)
        white  = cv.bitwise_and(white,  roi)

        # Morphologie pour fermer les lacunes dans les lignes
        k = cv.getStructuringElement(cv.MORPH_RECT, (7, 3))
        yellow   = cv.morphologyEx(yellow,   cv.MORPH_CLOSE, k)
        white    = cv.morphologyEx(white,    cv.MORPH_CLOSE, k)
        combined = cv.bitwise_or(yellow, white)

        to_f = lambda m: cv.GaussianBlur(m.astype(np.float32) / 255.0, (5, 5), 0)
        return to_f(yellow), to_f(white), to_f(combined)

    # ── Bases (bas de l'image) ────────────────────────────────────────────────

    def _find_bases(self, yellow_f, white_f):
        """
        Base gauche  → pic du masque JAUNE  (ligne centrale — ancre sûre)
        Base droite  → pic du masque BLANC  (bord droit de voie)
        La séparation jaune/blanc garantit qu'on ne confond jamais les deux côtés,
        même en virage.
        """
        h, w = yellow_f.shape
        mid  = w // 2

        hist_y = yellow_f[int(h * 0.65):, :].sum(axis=0)
        hist_w = white_f[int(h * 0.65):, :].sum(axis=0)
        hist_c = hist_y + hist_w

        def _peak(hist, x1, x2):
            seg = hist[x1:x2]
            if seg.max() < MIN_HIST:
                return None
            pk = int(np.argmax(seg))
            lo, hi = max(0, pk - 40), min(len(seg), pk + 41)
            s = seg[lo:hi]
            if s.sum() < MIN_HIST:
                return None
            return float(np.average(np.arange(lo, hi), weights=s)) + x1

        def _peak_near(hist, cx_prev):
            sw = int(w * BASE_SEARCH_W)
            return _peak(hist, max(0, int(cx_prev - sw)), min(w, int(cx_prev + sw)))

        # Base jaune : chercher dans toute l'image (peut dériver en virage)
        if self._prev_lx is not None:
            lb = _peak_near(hist_y, self._prev_lx) or _peak(hist_y, 0, w)
        else:
            lb = _peak(hist_y, 0, mid)   # init : supposer côté gauche

        # Base blanche : chercher à droite de la base jaune
        right_start = max(mid // 2, (int(lb) + 30) if lb is not None else mid)
        if self._prev_rx is not None:
            rb = _peak_near(hist_w, self._prev_rx) or _peak(hist_w, right_start, w)
        else:
            rb = _peak(hist_w, right_start, w)

        # Fallback sur le masque combiné si un côté reste absent
        if lb is None and self._prev_lx is not None:
            lb = _peak_near(hist_c, self._prev_lx) or self._prev_lx
        if rb is None and self._prev_rx is not None:
            rb = _peak_near(hist_c, self._prev_rx) or self._prev_rx

        # Sanity check : si les bases se croisent, tout rejeter
        if lb is not None and rb is not None and lb >= rb - 30:
            lb, rb = None, None

        return lb, rb

    # ── Sliding window ────────────────────────────────────────────────────────

    def _sliding_window(self, prob_l, prob_r, lb, rb):
        """
        Remonte de bas en haut par bandes.
        prob_l (yellow) → points gauche   prob_r (white) → points droite
        Masques séparés : la gauche ne peut pas suivre des pixels blancs.
        """
        h, w  = prob_l.shape
        sh    = max(1, h // N_STRIPS)
        lpts, rpts = [], []
        lx, rx = lb, rb

        for i in range(N_STRIPS):
            y1 = max(0, h - (i + 1) * sh)
            y2 = h - i * sh
            yc = (y1 + y2) // 2

            def _cx(base, row, pts_list):
                if base is None:
                    return base
                x1 = max(0,   int(base - WIN_MARGIN))
                x2 = min(w,   int(base + WIN_MARGIN))
                win = row[y1:y2, x1:x2]
                if win.sum() < MIN_WIN:
                    return base
                r_idx, c_idx = np.where(win > 0.25)
                if c_idx.size > 0:
                    cx = int(x1 + np.average(c_idx, weights=win[r_idx, c_idx]))
                else:
                    r_s, c_s = np.where(win > 0.05)
                    cx = int(x1 + np.average(c_s, weights=win[r_s, c_s])) if c_s.size else int((x1 + x2) // 2)
                pts_list.append((cx, yc))
                return float(cx)

            lx = _cx(lx, prob_l, lpts)
            rx = _cx(rx, prob_r, rpts)

        if lpts:
            raw = float(lpts[0][0])
            self._prev_lx = (BASE_EMA * raw + (1 - BASE_EMA) * self._prev_lx
                             if self._prev_lx is not None else raw)
        if rpts:
            raw = float(rpts[0][0])
            self._prev_rx = (BASE_EMA * raw + (1 - BASE_EMA) * self._prev_rx
                             if self._prev_rx is not None else raw)

        return lpts, rpts

    # ── Déviation ─────────────────────────────────────────────────────────────

    def _deviation(self, lpts, rpts, w):
        mid  = w / 2.0
        prev = self._dev_prev

        def _base_x(pts):
            if not pts:
                return None
            return float(np.mean([p[0] for p in pts[:min(5, len(pts))]]))

        xl, xr = _base_x(lpts), _base_x(rpts)
        min_w, max_w = 0.18 * w, 0.80 * w
        valid, dev_raw = False, prev

        if xl is not None and xr is not None \
                and len(lpts) >= MIN_PTS and len(rpts) >= MIN_PTS:
            lw = abs(xr - xl)
            if min_w <= lw <= max_w:
                self._lw_px = lw
                dev_raw = ((xl + xr) / 2.0 - mid) / w
                valid   = True
        elif xl is not None and self._lw_px and len(lpts) >= MIN_PTS:
            dev_raw = (xl + self._lw_px / 2.0 - mid) / w
            valid   = True
        elif xr is not None and self._lw_px and len(rpts) >= MIN_PTS:
            dev_raw = (xr - self._lw_px / 2.0 - mid) / w
            valid   = True

        if valid:
            self._no_det_cnt = 0
        else:
            self._no_det_cnt += 1
            if self._no_det_cnt <= HOLD_FRAMES:
                return prev
            dev_raw = 0.0

        dev = DEV_SMOOTH * prev + (1.0 - DEV_SMOOTH) * dev_raw
        if abs(dev - prev) > MAX_DEV_JUMP:
            dev = prev + np.sign(dev - prev) * MAX_DEV_JUMP
        self._dev_prev = dev
        return dev

    # ── Rendu ─────────────────────────────────────────────────────────────────

    def _draw(self, frame, lpts, rpts, dev, fps, cmd, yellow_f, white_f):
        h, w = frame.shape[:2]
        has_l = len(lpts) >= MIN_PTS
        has_r = len(rpts) >= MIN_PTS

        if not has_l and not has_r:
            label, col = 'NO_LANE',     (50,  50, 255)
        elif dev >  DEV_THRESH:
            label, col = 'DRIFT_RIGHT', (0,  160, 255)
        elif dev < -DEV_THRESH:
            label, col = 'DRIFT_LEFT',  (0,  160, 255)
        else:
            label, col = 'CENTER_OK',   (50, 220,  50)

        # Superposition masques couleur détectés (debug visuel)
        ov = frame.copy()
        ov[yellow_f > 0.15] = [0, 220, 220]    # jaune → cyan
        ov[white_f  > 0.15] = [200, 200, 200]  # blanc → gris clair
        cv.addWeighted(ov, 0.20, frame, 0.80, 0, frame)

        if has_l and has_r:
            try:
                ys_pts = [p[1] for p in lpts]
                c_l = np.polyfit(ys_pts, [p[0] for p in lpts], 2)
                c_r = np.polyfit([p[1] for p in rpts], [p[0] for p in rpts], 2)
                # EMA sur les coefficients → zone verte stable
                if self._coef_l is None:
                    self._coef_l, self._coef_r = c_l, c_r
                else:
                    self._coef_l = POLY_SMOOTH * c_l + (1 - POLY_SMOOTH) * self._coef_l
                    self._coef_r = POLY_SMOOTH * c_r + (1 - POLY_SMOOTH) * self._coef_r
                ys   = np.linspace(int(h * 0.48), h - 1, 60).astype(int)
                lx_a = np.clip(np.polyval(self._coef_l, ys).astype(int), 0, w - 1)
                rx_a = np.clip(np.polyval(self._coef_r, ys).astype(int), 0, w - 1)
                zone = np.concatenate([
                    np.stack([lx_a, ys], axis=1),
                    np.stack([rx_a, ys], axis=1)[::-1],
                ])
                fill = frame.copy()
                cv.fillPoly(fill, [zone.reshape(-1, 1, 2)], (0, 200, 0))
                cv.addWeighted(fill, 0.30, frame, 0.70, 0, frame)
            except Exception:
                self._coef_l = self._coef_r = None
        else:
            for pts, c in ((lpts, (0, 220, 220)), (rpts, (200, 200, 200))):
                if len(pts) >= 2:
                    for j in range(len(pts) - 1):
                        cv.line(frame, pts[j], pts[j + 1], c, 3, cv.LINE_AA)

        cv.rectangle(frame, (0, 0), (w, 64), (10, 10, 10), -1)
        cv.putText(frame, f'LANE: {label}',
                   (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.80, col, 2, cv.LINE_AA)
        cv.putText(frame, f'DEV : {dev:+.3f}',
                   (10, 52), cv.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv.LINE_AA)
        cmd_col = {'BRAKE': (50, 50, 255), 'SLOW': (0, 200, 255),
                   'STEER_LEFT': (80, 200, 255), 'STEER_RIGHT': (80, 200, 255),
                   'GO': (50, 220, 50)}.get(cmd, (160, 160, 160))
        cv.putText(frame, f'CMD : {cmd}',
                   (w - 220, 26), cv.FONT_HERSHEY_SIMPLEX, 0.72, cmd_col, 2, cv.LINE_AA)
        cv.putText(frame, f'{fps:.0f} fps',
                   (w - 85, 52), cv.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv.LINE_AA)

        cx  = w // 2
        off = int(dev * w * 0.6)
        cv.rectangle(frame, (cx - 80, 58), (cx + 80, 63), (40, 40, 40), -1)
        if off:
            cv.rectangle(frame, (min(cx, cx + off), 58),
                                 (max(cx, cx + off), 63), col, -1)
        cv.line(frame, (cx, 55), (cx, 66), (120, 120, 120), 1)

        return frame, label

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def _process(self, frame, fps, cmd):
        yellow_f, white_f, combined_f = self._cv_masks(frame)
        lb, rb     = self._find_bases(yellow_f, white_f)
        lpts, rpts = self._sliding_window(yellow_f, white_f, lb, rb)
        dev        = self._deviation(lpts, rpts, frame.shape[1])
        out        = frame.copy()
        out, label = self._draw(out, lpts, rpts, dev, fps, cmd, yellow_f, white_f)
        return out, dev, label

    # ── Boucle ────────────────────────────────────────────────────────────────

    def run(self):
        cv.namedWindow('Lane Detection', cv.WINDOW_NORMAL)
        cv.resizeWindow('Lane Detection', 860, 520)
        prev_t = time.time()

        while not rospy.is_shutdown():
            with self._lock:
                frame = self._frame.copy() if self._frame is not None else None
                cmd   = self._last_cmd

            if frame is not None:
                now    = time.time()
                fps    = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now

                result, dev, label = self._process(frame, fps, cmd)

                try:
                    self._pub_dev.publish(Float32(data=float(dev)))
                    self._pub_st.publish(String(data=label))
                    msg = self._bridge.cv2_to_imgmsg(result, encoding='bgr8')
                    msg.header.stamp = rospy.Time.now()
                    self._pub_img.publish(msg)
                except Exception:
                    pass

                cv.imshow('Lane Detection', result)
                rospy.loginfo_throttle(3, f'[LANE] {label:12s} dev={dev:+.3f}  fps={fps:.0f}')
            else:
                wait = np.zeros((360, 640, 3), np.uint8)
                cv.putText(wait, 'En attente de /carla/camera/rgb ...',
                           (40, 170), cv.FONT_HERSHEY_SIMPLEX,
                           0.75, (80, 200, 80), 2, cv.LINE_AA)
                cv.imshow('Lane Detection', wait)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cv.destroyAllWindows()


if __name__ == '__main__':
    LaneDetectionNode().run()
