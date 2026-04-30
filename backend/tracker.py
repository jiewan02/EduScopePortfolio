import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import CFG

try:
    from ultralytics.trackers.bot_sort import BOTSORT
    _HAS_BOTSORT = True
except Exception:
    _HAS_BOTSORT = False


class SimpleTracker:
    def __init__(self, max_age=30, dist_thresh=120):
        self.next_id    = 0
        self.tracks     = {}
        self.max_age    = max_age
        self.dist_thresh = dist_thresh

    def update(self, boxes_xyxy):
        ids = [-1] * len(boxes_xyxy)
        if len(boxes_xyxy) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid][1] += 1
                if self.tracks[tid][1] > self.max_age:
                    del self.tracks[tid]
            return ids

        centers = np.array([[(x1+x2)/2, (y1+y2)/2] for x1,y1,x2,y2 in boxes_xyxy], np.float32)

        if len(self.tracks) == 0:
            for i in range(len(centers)):
                ids[i] = self.next_id
                self.tracks[self.next_id] = [centers[i], 0]
                self.next_id += 1
            return ids

        tids = list(self.tracks.keys())
        prev = np.array([self.tracks[t][0] for t in tids], np.float32)
        D    = np.linalg.norm(prev[:,None,:] - centers[None,:,:], axis=2)
        r, c = linear_sum_assignment(D)
        used = set()
        for ri, ci in zip(r, c):
            if D[ri, ci] <= self.dist_thresh:
                tid = tids[ri]
                ids[ci] = tid
                self.tracks[tid] = [centers[ci], 0]
                used.add(ci)

        for i in range(len(centers)):
            if i not in used:
                ids[i] = self.next_id
                self.tracks[self.next_id] = [centers[i], 0]
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in ids:
                self.tracks[tid][1] += 1
                if self.tracks[tid][1] > self.max_age:
                    del self.tracks[tid]
        return ids


class BotSortWrapper:
    def __init__(self, frame_rate=30.0):
        if not (_HAS_BOTSORT and CFG.use_botsort):
            self.trk = None
            return
        args = lambda: None
        args.track_high_thresh  = CFG.botsort_conf_thres
        args.track_low_thresh   = CFG.botsort_conf_thres
        args.new_track_thresh   = CFG.botsort_conf_thres
        args.match_thresh       = CFG.botsort_match_thres
        args.track_buffer       = CFG.botsort_track_buffer
        args.mot20              = CFG.botsort_mot20
        args.frame_rate         = frame_rate
        args.gmc_method         = 'None'
        args.proximity_thresh   = 0.5
        args.appearance_thresh  = 0.25
        args.with_reid          = False
        try:
            self.trk = BOTSORT(args, frame_rate=frame_rate)
        except TypeError:
            try:
                self.trk = BOTSORT(args)
            except Exception:
                print("BoT-SORT init failed, disabling.")
                self.trk = None

    def available(self): return self.trk is not None

    def update(self, dets_xyxy, dets_conf, dets_cls, frame_bgr):
        if self.trk is None or len(dets_xyxy) == 0:
            return []
        dets = np.concatenate([
            dets_xyxy.astype(float),
            dets_conf.reshape(-1,1).astype(float),
            dets_cls.reshape(-1,1).astype(float),
        ], axis=1)
        tracks = self.trk.update(dets, frame_bgr)
        out = []
        for t in tracks:
            x1,y1,x2,y2,tid = int(t[0]),int(t[1]),int(t[2]),int(t[3]),int(t[4])
            out.append({"box": (x1,y1,x2,y2), "id": tid})
        return out
