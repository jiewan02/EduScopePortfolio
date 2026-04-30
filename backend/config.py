import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.abspath(os.path.join(_HERE, '..', 'models'))


def _mp(name: str) -> str:
    return os.path.join(_MODELS_DIR, name)


class CFG:
    # ── Model paths ───────────────────────────────────────────────────
    face_weights = _mp('yolo12n_50epoch_widerface.pt')

    det_weights  = _mp('yolo12n_FT_openimage.pt')
    det_weights2 = _mp('yolov8n_FT_openimage.pt')

    using_phone_weights  = _mp('usingphone_last.pt')
    using_phone_weights2 = _mp('usingphone_first.pt')

    up_suppress_iou = 0.3

    imgsz = 640

    conf_face  = 0.40
    conf_obj   = 0.20
    conf_phone = 0.25

    nms_iou = 0.45

    w_front = 0.30
    w_eyes  = 0.25
    w_talk  = 0.15
    w_obj   = 0.50
    w_raise = 0.15

    raise_vis_thr  = 0.5
    raise_show_thr = 0.5

    yaw_scale   = 25.0
    pitch_scale = 20.0

    ear_mu    = 0.18
    ear_scale = 0.12

    ear_open_ref   = 0.20
    ear_open_scale = 0.10
    ear_min_q      = 0.30
    gaze_min_disp  = 0.05
    gaze_deadzone  = 0.05
    gaze_gamma     = 0.125
    gaze_prox_k    = 0.9

    eye_overlay_thr = 0.65
    metric_stride   = 3

    eye_closed_ear_thr = 0.15
    eye_closed_secs    = 2.0
    eye_closed_scale   = 0.5

    object_weights = {
        "using phone": 1.5,
        "cell phone":  0.80,
        "cup":         0.50,
        "bottle":      0.50,
    }
    name_alias = {
        "mobile phone": "cell phone",
        "cellphone":    "cell phone",
        "phone":        "using phone",
        "using_phone":  "using phone",
    }

    roi_enable      = True
    roi_face_min_px = 220
    roi_margin      = 0.25
    roi_max_side    = 640
    roi_iou_pick    = True

    use_botsort          = False
    botsort_fps          = 30.0
    botsort_conf_thres   = 0.01
    botsort_match_thres  = 0.8
    botsort_track_buffer = 30
    botsort_mot20        = False

    log_dir   = os.path.join(_HERE, '..', 'logs')
    ema_alpha = 0.3
