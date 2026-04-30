import cv2
import math
import numpy as np
import mediapipe as mp
from collections import defaultdict

from .config import CFG

# ------------ MediaPipe (module-level singletons) ------------
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

mp_pose = mp.solutions.pose.Pose(
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ------------ Landmark indices ------------
L_EYE = dict(h=(33, 133),   v1=(159, 145), v2=(158, 153))
R_EYE = dict(h=(362, 263),  v1=(386, 374), v2=(385, 380))
MOUTH = dict(h=(61, 291),   v1=(13, 14),   v2=(81, 311))

MODEL_3D   = np.array([[0.,0.,0.],[-30.,-30.,-30.],[30.,-30.,-30.],
                        [-40.,30.,-30.],[40.,30.,-30.],[0.,60.,-10.]], np.float32)
MODEL_IDXS = [1, 33, 263, 61, 291, 199]

IRIS_L = [468, 469, 470, 471]
IRIS_R = [473, 474, 475, 476]

NOSE = 0; L_SH = 11; R_SH = 12; L_EL = 13; R_EL = 14; L_WR = 15; R_WR = 16
EYE_L = 2; EYE_R = 5

# ------------ Geometry helpers ------------
def _dist(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))
def _center(box):  x1,y1,x2,y2=box; return ((x1+x2)/2.0, (y1+y2)/2.0)

def _iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = max(0, ax2-ax1) * max(0, ay2-ay1)
    ub = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (ua + ub - inter + 1e-6)

# ------------ Facial metrics ------------
def eye_EAR(land):
    def ear(idx):
        h = _dist(land[idx['h'][0]], land[idx['h'][1]])
        v = (_dist(land[idx['v1'][0]], land[idx['v1'][1]]) +
             _dist(land[idx['v2'][0]], land[idx['v2'][1]]))
        return v / (2.0*h + 1e-6)
    return (ear(L_EYE) + ear(R_EYE)) / 2.0

def mouth_MAR(land):
    h = _dist(land[MOUTH['h'][0]], land[MOUTH['h'][1]])
    v = (_dist(land[MOUTH['v1'][0]], land[MOUTH['v1'][1]]) +
         _dist(land[MOUTH['v2'][0]], land[MOUTH['v2'][1]]))
    return v / (2.0*h + 1e-6)

def head_pose(land, W, H):
    pts_2d = np.array([land[i] for i in MODEL_IDXS], np.float32)
    pts_2d[:, 0] *= W; pts_2d[:, 1] *= H
    focal = float(W)
    cam = np.array([[focal,0,W/2],[0,focal,H/2],[0,0,1]], np.float32)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts_2d, cam, np.zeros((4,1), np.float32),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy    = math.sqrt(float(R[0,0]**2 + R[1,0]**2))
    yaw   = math.degrees(math.atan2(float(R[2,0]), sy))
    pitch = math.degrees(math.atan2(float(-R[2,1]), float(R[2,2])))
    roll  = math.degrees(math.atan2(float(-R[1,0]), float(R[0,0])))
    return float(yaw), float(pitch), float(roll)

# ------------ Landmark extraction ------------
def landmarks_from_fullframe(frame_bgr, W, H):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)
    out = []
    if res.multi_face_landmarks:
        for lm in res.multi_face_landmarks:
            out.append([(p.x, p.y) for p in lm.landmark])
    return out

def lm_to_boxes(landmarks_list, W, H):
    boxes = []
    for pts in landmarks_list:
        xs = [p[0]*W for p in pts]; ys = [p[1]*H for p in pts]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        boxes.append((max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)))
    return boxes

def match_landmarks_to_boxes(landmarks_list, W, H, boxes_xyxy):
    matched = [None] * len(boxes_xyxy)
    if not landmarks_list: return matched
    lm_boxes = []
    for pts in landmarks_list:
        xs = [p[0]*W for p in pts]; ys = [p[1]*H for p in pts]
        lm_boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    for bi, (x1,y1,x2,y2) in enumerate(boxes_xyxy):
        best, best_pts = 0.0, None
        for (bx1,by1,bx2,by2), pts in zip(lm_boxes, landmarks_list):
            iou = _iou((x1,y1,x2,y2), (bx1,by1,bx2,by2))
            if iou > best:
                best, best_pts = iou, pts
        matched[bi] = best_pts
    return matched

# ------------ ROI refinement ------------
def refine_landmarks_roi(frame_bgr, box_xyxy, W, H,
                         margin=CFG.roi_margin, max_side=CFG.roi_max_side,
                         iou_pick=CFG.roi_iou_pick):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h   = (x2-x1), (y2-y1)
    w2, h2 = int(w*(1+margin)), int(h*(1+margin))
    rx1 = max(0, int(cx-w2/2)); ry1 = max(0, int(cy-h2/2))
    rx2 = min(W, int(cx+w2/2)); ry2 = min(H, int(cy+h2/2))
    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0: return None
    scale = max_side / max(1, max(roi.shape[0], roi.shape[1]))
    roi_up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else roi
    if scale <= 1.0: scale = 1.0
    rgb = cv2.cvtColor(roi_up, cv2.COLOR_BGR2RGB)
    res = mp_face_mesh.process(rgb)
    if not res.multi_face_landmarks: return None
    best_pts, best_iou = None, -1.0
    for lm in res.multi_face_landmarks:
        pts = []; xs = []; ys = []
        for p in lm.landmark:
            xf = rx1 + (p.x * roi_up.shape[1]) / scale
            yf = ry1 + (p.y * roi_up.shape[0]) / scale
            pts.append((xf/W, yf/H)); xs.append(xf); ys.append(yf)
        bx1,by1,bx2,by2 = int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))
        iou = _iou((x1,y1,x2,y2),(bx1,by1,bx2,by2)) if iou_pick else 1.0
        if iou > best_iou:
            best_iou, best_pts = iou, pts
    return best_pts

# ------------ Pose detection ------------
def detect_pose(frame_bgr, W, H):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = mp_pose.process(rgb)
    if not res.pose_landmarks: return None
    lm = res.pose_landmarks.landmark
    def ok(i): return (lm[i].visibility or 0.0) >= CFG.raise_vis_thr
    xs = [lm[i].x*W for i in [NOSE,L_SH,R_SH,L_EL,R_EL,L_WR,R_WR,EYE_L,EYE_R]]
    ys = [lm[i].y*H for i in [NOSE,L_SH,R_SH,L_EL,R_EL,L_WR,R_WR,EYE_L,EYE_R]]
    box = (int(max(0,min(xs))), int(max(0,min(ys))), int(min(W-1,max(xs))), int(min(H-1,max(ys))))
    return {"lm": lm, "box": box, "ok": ok}

def match_pose_to_faces(pose, face_boxes):
    if pose is None: return [None] * len(face_boxes)
    pose_box = pose["box"]
    best_iou, best_idx = 0.0, -1
    for i, fbox in enumerate(face_boxes):
        iou = _iou(pose_box, fbox)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    res = [None] * len(face_boxes)
    if best_idx >= 0:
        res[best_idx] = pose
    return res

# ------------ Hand raise ------------
def _angle(a, b, c):
    v1 = np.array(a) - np.array(b); v2 = np.array(c) - np.array(b)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0))))

def hand_raise_score(pose, W, H):
    if pose is None: return 0.0
    lm = pose["lm"]; ok = pose["ok"]
    def pt(i): return np.array([lm[i].x*W, lm[i].y*H], float)
    head_y = (min(lm[EYE_L].y, lm[EYE_R].y) * H
               if (lm[EYE_L].visibility > 0 and lm[EYE_R].visibility > 0)
               else lm[NOSE].y * H)
    def side(wi, ei, si):
        if not (ok(wi) and ok(ei) and ok(si)): return 0.0
        Wp, Ep, Sp = pt(wi), pt(ei), pt(si)
        denom   = max(10.0, abs(Sp[1] - head_y))
        v1      = float(np.clip((Sp[1]-Wp[1])/denom, 0.0, 1.0))
        v2      = 1.0 if Ep[1] <= Sp[1] else 0.0
        ang     = _angle(Sp, Ep, Wp)
        straight = float(np.clip((ang-120.0)/60.0, 0.0, 1.0))
        return 0.55*v1 + 0.20*v2 + 0.25*straight
    return float(max(side(L_WR,L_EL,L_SH), side(R_WR,R_EL,R_SH)))

# ------------ EMA smoother ------------
class MetricState:
    def __init__(self, alpha=0.3):
        self.talk_ema  = defaultdict(float)
        self.focus_ema = defaultdict(float)
        self.eng_ema   = defaultdict(float)
        self.alpha = alpha

    def smooth(self, tid, talk, focus, eng):
        a = self.alpha
        self.talk_ema[tid]  = (1-a)*self.talk_ema[tid]  + a*talk
        self.focus_ema[tid] = (1-a)*self.focus_ema[tid] + a*focus
        self.eng_ema[tid]   = (1-a)*self.eng_ema[tid]   + a*eng
        return self.talk_ema[tid], self.focus_ema[tid], self.eng_ema[tid]

# ------------ Ensemble NMS merge ------------
def nms_merge(detections: list, iou_thr: float = 0.45) -> list:
    """
    Merge detections from multiple models.
    Groups by class name, then greedily keeps highest-confidence boxes
    that don't overlap (IoU > iou_thr) with already-kept ones.
    """
    if not detections:
        return []
    by_class: dict = {}
    for d in detections:
        by_class.setdefault(d['name'], []).append(d)
    result = []
    for dets in by_class.values():
        dets = sorted(dets, key=lambda d: d['conf'], reverse=True)
        kept = []
        for d in dets:
            if all(_iou(d['box'], k['box']) < iou_thr for k in kept):
                kept.append(d)
        result.extend(kept)
    return result

# ------------ Name canonicalization ------------
def canon(name: str) -> str:
    n = str(name).strip().lower()
    return CFG.name_alias.get(n, n)

# ------------ Drawing helpers ------------
def draw_text_with_bg(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.55,
                      txt_color=(0,0,0), bg_color=(0,255,255), thickness=2, pad=4):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+pad), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, txt_color, thickness, cv2.LINE_AA)

def draw_face_overlay(frame, box, tid, talk, focus, eng,
                      eye_look=None, eye_obj=None, raise_score=None):
    x1, y1, x2, y2 = map(int, box)
    color = (0,255,0) if eng >= 0.6 else (0,165,255) if eng >= 0.35 else (0,0,255)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    y_text = max(22, y1 - 8)
    txt = f"ID {tid}  Eng {eng:.2f}  Focus {focus:.2f}  Talk {talk:.2f}"
    if raise_score is not None and raise_score >= CFG.raise_show_thr:
        txt += f"  Raise {raise_score:.2f}"
    if eye_look is not None and eye_obj is not None and eye_look >= CFG.eye_overlay_thr:
        txt += f"  Looking@{eye_obj}"
    draw_text_with_bg(frame, txt, (x1, y_text))

def draw_objects(frame, objects):
    for o in objects:
        x1, y1, x2, y2 = map(int, o["box"])
        cname = canon(o.get('name', 'unknown'))
        if cname in CFG.object_weights:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2)
            draw_text_with_bg(frame, f"{cname} {o.get('conf',1.0):.2f}", (x1, max(24,y1-10)))

# ------------ Iris / EyeLook helpers ------------
def _px(pt, W, H): return np.array([pt[0]*W, pt[1]*H], float)

def _eye_center_width(land, eye_idx, W, H):
    p1 = _px(land[eye_idx['h'][0]], W, H)
    p2 = _px(land[eye_idx['h'][1]], W, H)
    C  = (p1+p2)*0.5
    w  = np.linalg.norm(p1-p2)
    return C, max(1.0, w)

def _iris_centers_px(land, W, H):
    Li = np.mean([[land[i][0]*W, land[i][1]*H] for i in IRIS_L], axis=0)
    Ri = np.mean([[land[i][0]*W, land[i][1]*H] for i in IRIS_R], axis=0)
    return Li, Ri

def draw_iris_points(frame, pts, W, H, obj_center=None):
    Li, Ri = _iris_centers_px(pts, W, H)
    Lc, _  = _eye_center_width(pts, L_EYE, W, H)
    Rc, _  = _eye_center_width(pts, R_EYE, W, H)
    for P in (Li, Ri):
        cv2.circle(frame, (int(P[0]), int(P[1])), 3, (0,255,255), -1)
    if obj_center is not None:
        Ox, Oy = int(obj_center[0]), int(obj_center[1])
        cv2.line(frame, (int(Lc[0]),int(Lc[1])), (int(Li[0]),int(Li[1])), (0,255,0), 2)
        cv2.line(frame, (int(Rc[0]),int(Rc[1])), (int(Ri[0]),int(Ri[1])), (0,255,0), 2)
        cv2.line(frame, (int(Lc[0]),int(Lc[1])), (Ox,Oy), (255,0,255), 2)
        cv2.line(frame, (int(Rc[0]),int(Rc[1])), (Ox,Oy), (255,0,255), 2)
        cv2.circle(frame, (Ox,Oy), 4, (255,0,255), -1)

def eye_look_score(land, W, H, obj_box):
    xs = [p[0]*W for p in land]; ys = [p[1]*H for p in land]
    fw = max(12.0, max(xs)-min(xs)); fh = max(12.0, max(ys)-min(ys))
    face_diag = math.hypot(fw, fh)
    ox, oy = _center(obj_box); O = np.array([ox, oy], float)
    Lc, Lw = _eye_center_width(land, L_EYE, W, H)
    Rc, Rw = _eye_center_width(land, R_EYE, W, H)
    Li, Ri = _iris_centers_px(land, W, H)
    EAR    = float(np.clip(eye_EAR(land), 0, 1))
    ear_q  = max(CFG.ear_min_q,
                 float(np.clip((EAR-CFG.ear_open_ref)/CFG.ear_open_scale, 0.0, 1.0)))

    def eye_score(E, I, eye_w):
        disp = np.linalg.norm(I-E) / max(1.0, 0.5*eye_w)
        if disp < CFG.gaze_min_disp: return 0.0
        g = (I-E) / (np.linalg.norm(I-E)+1e-6)
        o_vec = O - E; o_norm = np.linalg.norm(o_vec)
        if o_norm < 1e-6: return 0.0
        o_hat = o_vec / o_norm
        cosv  = float(np.clip(np.dot(g, o_hat), -1.0, 1.0))
        if cosv <= CFG.gaze_deadzone: base = 0.0
        else: base = (cosv-CFG.gaze_deadzone) / (1.0-CFG.gaze_deadzone)
        prox   = math.exp(-o_norm / max(1.0, CFG.gaze_prox_k*face_diag))
        disp_q = float(np.clip((disp-CFG.gaze_min_disp)/0.35, 0.0, 1.0))
        return float(np.clip(base*disp_q*prox*ear_q, 0.0, 1.0))

    s = max(eye_score(Lc,Li,Lw), eye_score(Rc,Ri,Rw))
    return float(np.clip(s, 0.0, 1.0)) ** CFG.gaze_gamma
