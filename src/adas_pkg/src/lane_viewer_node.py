#!/usr/bin/env python3
"""
lane_detection_node.py — CARLA ADAS
Zone verte transparente uniquement — pas de lignes laser
Topics : /lane/deviation  /lane/status  /lane/image
"""
import cv2 as cv
import numpy as np
import threading, time
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

# ─── Config ──────────────────────────────────────────────────────────────────
SMOOTH       = 0.60   # réactif virages
DEV_THRESH   = 0.05
HOUGH_THRESH = 25
MIN_LINE_LEN = 30
MAX_LINE_GAP = 150

# ─── Singletons ──────────────────────────────────────────────────────────────
_clahe     = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
_roi_cache = {}
_ema       = {'left': None, 'right': None}
_missing   = {'left': 0,    'right': 0}

# ─── ROI resserré (fix zone verte trop large) ────────────────────────────────
def roi_mask(h, w):
    if (h, w) not in _roi_cache:
        pts  = np.array([[(int(w*.15), h),
                           (int(w*.85), h),
                           (int(w*.58), int(h*.58)),
                           (int(w*.42), int(h*.58))]], np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv.fillPoly(mask, pts, 255)
        _roi_cache[(h, w)] = mask
    return _roi_cache[(h, w)]

# ─── Prétraitement ───────────────────────────────────────────────────────────
def preprocess(frame):
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    hls[:, :, 1] = _clahe.apply(hls[:, :, 1])
    return cv.cvtColor(hls, cv.COLOR_HLS2BGR)

# ─── Masque couleur + Sobel + Canny ─────────────────────────────────────────
def edge_mask(frame):
    hls   = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    color = cv.bitwise_or(
        cv.inRange(hls, (  0, 150,   0), (255, 255,  60)),
        cv.inRange(hls, ( 15,  60,  80), ( 45, 255, 255)))
    gray  = cv.GaussianBlur(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), (5, 5), 0)
    mag   = np.sqrt(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)**2 +
                    cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)**2)
    mag   = np.uint8(255 * mag / (mag.max() + 1e-6))
    _, grad = cv.threshold(mag, 25, 255, cv.THRESH_BINARY)
    return cv.Canny(cv.bitwise_or(color, grad), 30, 100)

# ─── Fit lignes + EMA + reset virage ────────────────────────────────────────
def fit_lines(h, w, lines):
    mid   = w / 2.0
    sides = {'left': [], 'right': []}

    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1: continue
        length = np.hypot(x2-x1, y2-y1)
        if length < 20: continue
        m, b = np.polyfit((x1, x2), (y1, y2), 1)

        # Filtre pente strict
        if not (0.5 < abs(m) < 2.5): continue

        # Filtre position strict gauche/droite
        if   m < 0 and max(x1,x2) < mid * 1.05: sides['left'].append(([m,b], length))
        elif m > 0 and min(x1,x2) > mid * 0.95: sides['right'].append(([m,b], length))

    coords = {}
    for side, cands in sides.items():
        if cands:
            # Rejet outliers : garde pentes proches de la médiane
            slopes = [p[0] for p,_ in cands]
            med    = np.median(slopes)
            cands  = [(p,l) for (p,l) in cands if abs(p[0]-med) < 0.4]

        if cands:
            P   = np.array([p for p,_ in cands])
            W   = np.array([l for _,l in cands]); W /= W.sum()
            cur = (P * W[:,None]).sum(axis=0)
            _ema[side]     = SMOOTH * _ema[side] + (1-SMOOTH)*cur \
                             if _ema[side] is not None else cur
            _missing[side] = 0
        else:
            _missing[side] += 1
            # Reset EMA après 8 frames sans détection (virage serré)
            if _missing[side] > 8:
                _ema[side]     = None
                _missing[side] = 0

        if _ema[side] is not None:
            m, b     = _ema[side]
            y1c, y2c = h, int(h * 0.56)
            coords[side] = (int((y1c-b)/m), y1c, int((y2c-b)/m), y2c)

    # Déviation pondérée bas 70% + haut 30%
    dev = 0.0
    if len(coords) == 2:
        x_bas  = (coords['left'][0] + coords['right'][0]) / 2.0
        x_haut = (coords['left'][2] + coords['right'][2]) / 2.0
        dev    = (x_bas*0.7 + x_haut*0.3 - mid) / w
    return coords, dev

# ─── Rendu : zone verte transparente uniquement ──────────────────────────────
def render(frame, lanes):
    if len(lanes) == 2:
        xl1,yl1,xl2,yl2 = lanes['left']
        xr1,yr1,xr2,yr2 = lanes['right']
        overlay = frame.copy()
        cv.fillPoly(overlay,
                    [np.array([[xl1,yl1],[xr1,yr1],[xr2,yr2],[xl2,yl2]], np.int32)],
                    (0, 200, 0))
        cv.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    return frame

# ─── HUD ─────────────────────────────────────────────────────────────────────
def draw_hud(frame, lanes, dev, fps):
    w = frame.shape[1]
    if   len(lanes) == 0:      label, col = 'NO LANE',     (50,  50, 255)
    elif 'left'  not in lanes: label, col = 'DRIFT LEFT',  (0,  160, 255)
    elif 'right' not in lanes: label, col = 'DRIFT RIGHT', (0,  160, 255)
    elif dev >  DEV_THRESH:    label, col = 'DRIFT LEFT',  (0,  160, 255)
    elif dev < -DEV_THRESH:    label, col = 'DRIFT RIGHT', (0,  160, 255)
    else:                      label, col = 'CENTER OK',   (50, 220,  50)

    cv.rectangle(frame, (0,0), (w,48), (10,10,10), -1)
    cv.putText(frame, f'Lane: {label}', (12,33),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv.LINE_AA)
    cv.putText(frame, f'{fps:.0f} FPS', (w-110,33),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (140,140,140), 1, cv.LINE_AA)
    cx  = w // 2
    off = int(dev * w * 0.8)
    cv.rectangle(frame, (cx-60,43), (cx+60,47), (50,50,50), -1)
    if off: cv.rectangle(frame, (cx,43), (cx+off,47), col, -1)
    cv.line(frame, (cx,41), (cx,49), (120,120,120), 1)
    return frame, label

# ─── Pipeline ────────────────────────────────────────────────────────────────
def process(frame, fps):
    h, w  = frame.shape[:2]
    edges = cv.bitwise_and(edge_mask(preprocess(frame)), roi_mask(h, w))
    lines = cv.HoughLinesP(edges, 1, np.pi/180,
                            HOUGH_THRESH, None, MIN_LINE_LEN, MAX_LINE_GAP)
    lanes, dev = fit_lines(h, w, lines) if lines is not None else ({}, 0.0)
    out        = render(frame.copy(), lanes)
    out, label = draw_hud(out, lanes, dev, fps)
    return out, dev, label

# ─── ROS ─────────────────────────────────────────────────────────────────────
rospy.init_node('lane_detection_node', anonymous=True)
bridge = CvBridge()
_lock  = threading.Lock()
_frame = None

pub_img = rospy.Publisher('/lane/image',     Image,   queue_size=1)
pub_dev = rospy.Publisher('/lane/deviation', Float32, queue_size=1)
pub_st  = rospy.Publisher('/lane/status',    String,  queue_size=1)

def cb(msg):
    global _frame
    try:
        f = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with _lock: _frame = f
    except Exception: pass

rospy.Subscriber('/carla/camera/rgb', Image, cb, queue_size=1)

cv.namedWindow('Lane Detection', cv.WINDOW_NORMAL)
cv.resizeWindow('Lane Detection', 860, 520)
print("✅ Lane Detection Node prêt — Q pour quitter")

prev_t = time.time()

while not rospy.is_shutdown():
    with _lock:
        frame = _frame.copy() if _frame is not None else None
    if frame is not None:
        now = time.time()
        fps, prev_t = 1.0 / max(now - prev_t, 1e-6), now
        result, dev, label = process(frame, fps)
        try:
            pub_dev.publish(Float32(data=float(dev)))
            pub_st.publish(String(data=label))
            m = bridge.cv2_to_imgmsg(result, encoding='bgr8')
            m.header.stamp = rospy.Time.now()
            pub_img.publish(m)
        except Exception: pass
        cv.imshow('Lane Detection', result)
    else:
        rospy.sleep(0.01)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
print("✅ Lane Detection Node arrêté")