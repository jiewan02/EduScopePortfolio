"""
EngagementEngine: stateful per-session processor.
Call process_frame(frame_bgr) → dict each frame.

Model strategy:
  face_model          — dedicated face detector (yolo12n_widerface), enables YOLO tracking
  obj_models [×2]     — ensembled COCO object detectors for cup / bottle / cell phone
  phone_models [×2]   — ensembled phone-usage detectors for using_phone / phone
"""

import math
import time
import numpy as np
import cv2
from ultralytics import YOLO

from .config import CFG
from .utils import (
    _iou, _center,
    refine_landmarks_roi, detect_pose, match_pose_to_faces,
    hand_raise_score, eye_EAR, mouth_MAR, head_pose, eye_look_score,
    landmarks_from_fullframe, lm_to_boxes, match_landmarks_to_boxes,
    MetricState, canon, nms_merge,
)
from .tracker import SimpleTracker, BotSortWrapper


def _run_model(model, frame, conf):
    """Run one YOLO model and return (xyxy, conf_arr, cls_arr)."""
    pred = model(frame, imgsz=CFG.imgsz, conf=conf, verbose=False)[0]
    if pred.boxes and len(pred.boxes) > 0:
        return (pred.boxes.xyxy.cpu().numpy().astype(int),
                pred.boxes.conf.cpu().numpy().astype(float),
                pred.boxes.cls.cpu().numpy().astype(int))
    return (np.zeros((0, 4), int), np.zeros((0,), float), np.zeros((0,), int))


class EngagementEngine:
    def __init__(self):
        print("[Engine] Loading models…")

        # Dedicated face detector
        self.face_model = YOLO(CFG.face_weights)
        self.face_id    = 0   # 'face' is always class 0 in widerface model

        # Ensembled object detectors (cup / bottle / cell phone)
        self.obj_models = [YOLO(CFG.det_weights), YOLO(CFG.det_weights2)]

        # Ensembled phone-usage detectors
        self.phone_models = [YOLO(CFG.using_phone_weights),
                             YOLO(CFG.using_phone_weights2)]

        print("[Engine] All models loaded. Face detection: YOLO (widerface)")
        self._init_state()
        print("[Engine] Ready.")

    def _init_state(self):
        # Face model always has 'face' class → can use BotSort
        if CFG.use_botsort:
            self.tracker = BotSortWrapper(frame_rate=CFG.botsort_fps)
            if not self.tracker.available():
                print("[Engine] BoT-SORT unavailable → SimpleTracker")
                self.tracker = SimpleTracker()
        else:
            self.tracker = SimpleTracker()

        self.metr          = MetricState(alpha=CFG.ema_alpha)
        self.metrics_cache = {}
        self.t0            = time.time()
        self.t_prev        = time.time()
        self.fps           = 0.0
        self.frame_idx     = 0

    def reset(self):
        self._init_state()

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> dict:
        H, W = frame.shape[:2]
        t_now  = time.time()
        dt     = max(1e-3, t_now - self.t_prev)
        self.t_prev  = t_now
        elapsed      = t_now - self.t0
        self.frame_idx += 1
        recompute = (self.frame_idx % CFG.metric_stride == 0)

        # ── A. Phone-usage ensemble ───────────────────────────────────
        raw_phone = []
        for pm in self.phone_models:
            xyxy, conf, cls = _run_model(pm, frame, CFG.conf_phone)
            for i in range(len(xyxy)):
                cname = canon(pm.names[cls[i]])
                raw_phone.append({"box": tuple(xyxy[i]),
                                  "conf": float(conf[i]), "name": cname})
        up_list = nms_merge(raw_phone, CFG.nms_iou)

        # ── B. Object detection ensemble ─────────────────────────────
        raw_obj = []
        for om in self.obj_models:
            xyxy, conf, cls = _run_model(om, frame, CFG.conf_obj)
            for i in range(len(xyxy)):
                cname = canon(om.names[cls[i]])
                if cname in CFG.object_weights:
                    raw_obj.append({"box": tuple(xyxy[i]),
                                    "conf": float(conf[i]), "name": cname})
        # Suppress 'cell phone' where 'using phone' already detected
        filtered_obj = [
            o for o in raw_obj
            if not (o["name"] == "cell phone" and
                    any(_iou(o["box"], u["box"]) >= CFG.up_suppress_iou
                        for u in up_list if u["name"] == "using phone"))
        ]
        obj_list = nms_merge(up_list + filtered_obj, CFG.nms_iou)

        # ── C. Face detection (dedicated widerface model) ─────────────
        face_xyxy, face_conf, _ = _run_model(self.face_model, frame, CFG.conf_face)

        # ── D. Tracking ───────────────────────────────────────────────
        lm_all = landmarks_from_fullframe(frame, W, H)

        if isinstance(self.tracker, BotSortWrapper) and self.tracker.available() and len(face_xyxy) > 0:
            tracks = self.tracker.update(
                face_xyxy, face_conf,
                np.zeros(len(face_xyxy), int), frame)   # class 0 = face
            ids = [-1] * len(face_xyxy)
            if tracks:
                trk_boxes = np.array([t["box"] for t in tracks], int)
                trk_ids   = [t["id"] for t in tracks]
                for di, db in enumerate(face_xyxy):
                    ious = np.array([_iou(db, tb) for tb in trk_boxes], float)
                    j    = int(np.argmax(ious))
                    if ious[j] > 0.1:
                        ids[di] = trk_ids[j]
            face_boxes = face_xyxy
        else:
            face_boxes = face_xyxy
            ids = self.tracker.update(face_boxes) if len(face_boxes) > 0 else []

        matched = match_landmarks_to_boxes(lm_all, W, H, face_boxes)

        # ── E. Pose detection ─────────────────────────────────────────
        pose        = detect_pose(frame, W, H) if recompute else None
        pose_assign = match_pose_to_faces(pose, face_boxes)

        # ── F. Per-face engagement ────────────────────────────────────
        present_names = {o["name"] for o in obj_list if o["name"] in CFG.object_weights}
        persons = []

        for i, box in enumerate(face_boxes):
            x1, y1, x2, y2 = map(int, box)
            tid = ids[i] if i < len(ids) else -1
            pts = matched[i] if i < len(matched) else None

            if recompute and CFG.roi_enable and pts is not None:
                if (x2-x1) < CFG.roi_face_min_px or (y2-y1) < CFG.roi_face_min_px:
                    pts_ref = refine_landmarks_roi(frame, (x1,y1,x2,y2), W, H)
                    if pts_ref is not None:
                        pts = pts_ref

            eye_look_show = eye_obj_show = raise_show = None

            if pts is not None:
                if recompute or tid not in self.metrics_cache:
                    EAR = float(np.clip(eye_EAR(pts), 0, 1))
                    MAR = float(np.clip(mouth_MAR(pts), 0, 1.5))
                    yaw, pitch, _ = head_pose(pts, W, H)

                    front = math.exp(-(abs(yaw)/CFG.yaw_scale + abs(pitch)/CFG.pitch_scale))
                    eyes  = float(np.clip((EAR - CFG.ear_mu)/CFG.ear_scale, 0, 1))
                    talk  = float(np.clip(0.6*np.tanh((MAR-0.3)*3), 0, 1))
                    focus = float(np.clip(0.5*front + 0.5*eyes, 0, 1))

                    prev   = self.metrics_cache.get(tid, {})
                    tdiff  = t_now - prev.get('last_t', t_now)
                    closed_dur = (prev.get('closed_dur', 0.0) + tdiff
                                  if EAR < CFG.eye_closed_ear_thr else 0.0)

                    obj_pen = best_look = best_name = None
                    best_term = 0.0
                    if present_names:
                        for o in obj_list:
                            cn = o['name']
                            if cn not in CFG.object_weights: continue
                            look = eye_look_score(pts, W, H, o['box'])
                            if best_look is None or look > best_look:
                                best_look, best_name = look, cn
                            best_term = max(best_term, CFG.object_weights[cn]*look)
                        obj_pen = CFG.w_obj * best_term if best_name else 0.0
                    else:
                        obj_pen = 0.0

                    penalty = (CFG.w_front*(1-front) + CFG.w_eyes*(1-eyes) +
                               CFG.w_talk*talk + obj_pen)
                    eng = float(np.clip(1.0 - penalty, 0.0, 1.0))
                    if closed_dur >= CFG.eye_closed_secs:
                        eng *= CFG.eye_closed_scale

                    pr          = pose_assign[i] if pose_assign[i] is not None else pose
                    raise_score = hand_raise_score(pr, W, H) if pr is not None else 0.0
                    eng         = float(min(1.0, eng + CFG.w_raise*raise_score))
                    if raise_score >= CFG.raise_show_thr:
                        raise_show = raise_score

                    talk_s, focus_s, eng_s = self.metr.smooth(tid, talk, focus, eng)
                    self.metrics_cache[tid] = dict(
                        EAR=EAR, MAR=MAR, yaw=yaw, pitch=pitch,
                        front=front, eyes=eyes, talk=talk, focus=focus, eng=eng,
                        talk_s=talk_s, focus_s=focus_s, eng_s=eng_s,
                        eyelook=best_look, eyeobj=best_name, obj_pen=obj_pen,
                        raise_score=raise_score,
                        closed_dur=closed_dur, last_t=t_now,
                    )
                else:
                    m = self.metrics_cache[tid]
                    talk_s, focus_s, eng_s = m['talk_s'], m['focus_s'], m['eng_s']
                    best_look, best_name   = m['eyelook'], m['eyeobj']
                    if best_name in present_names:
                        eye_look_show, eye_obj_show = best_look, best_name
                    if m.get('raise_score', 0.0) >= CFG.raise_show_thr:
                        raise_show = m['raise_score']
                    m['last_t'] = t_now

                m = self.metrics_cache.get(tid, {})
                persons.append(dict(
                    id=int(tid), box=[x1,y1,x2,y2],
                    engagement=round(float(m.get('eng_s', 0)), 3),
                    focus=round(float(m.get('focus_s', 0)), 3),
                    talk=round(float(m.get('talk_s', 0)), 3),
                    EAR=round(float(m.get('EAR', 0)), 3),
                    yaw=round(float(m.get('yaw', 0)), 1),
                    pitch=round(float(m.get('pitch', 0)), 1),
                    hand_raise=round(float(m.get('raise_score', 0)), 3),
                    eyes_closed=m.get('closed_dur', 0.0) >= CFG.eye_closed_secs,
                    distraction=m.get('eyeobj'),
                ))
            else:
                cx, cy = (x1+x2)/2, (y1+y2)/2
                d      = np.hypot((cx-W/2)/(W/2+1e-6), (cy-H/2)/(H/2+1e-6))
                focus  = float(np.exp(-2.5*d))
                eng    = float(np.clip(0.7*focus, 0.0, 1.0))
                talk_s, focus_s, eng_s = self.metr.smooth(tid, 0.0, focus, eng)
                persons.append(dict(
                    id=int(tid), box=[x1,y1,x2,y2],
                    engagement=round(float(eng_s),3), focus=round(float(focus_s),3),
                    talk=0.0, EAR=0.0, yaw=0.0, pitch=0.0,
                    hand_raise=0.0, eyes_closed=False, distraction=None,
                ))

        inst     = 1.0 / max(1e-6, dt)
        self.fps = inst if self.fps == 0.0 else (0.9*self.fps + 0.1*inst)

        avg_eng  = float(np.mean([p['engagement'] for p in persons])) if persons else 0.0
        objects  = [{"name": o["name"], "box": list(o["box"]), "conf": round(o["conf"], 2)}
                    for o in obj_list]
        return dict(
            persons=persons,
            objects=objects,
            frame_size=[W, H],
            fps=round(self.fps, 1),
            elapsed=round(elapsed, 2),
            frame_idx=self.frame_idx,
            avg_engagement=round(avg_eng, 3),
            num_persons=len(persons),
        )
