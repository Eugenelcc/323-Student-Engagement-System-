import argparse, csv, json, os, time, threading
from collections import deque, defaultdict
from datetime import datetime
import cv2, numpy as np, mediapipe as mp

# Optional deps (only used if you choose --backend torch)
try:
    import torch, timm
except Exception:
    torch, timm = None, None

try:
    import onnxruntime as ort
except Exception:
    ort = None

# ----------------- Labels & Mapping -----------------
EMO = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
ENG3 = ["Engaged", "Neutral", "Disengaged"]
EMO2ENG = np.array(
    [
        [0.70, 0.25, 0.05],  # surprise
        [0.10, 0.25, 0.65],  # fear
        [0.05, 0.15, 0.80],  # disgust
        [0.85, 0.10, 0.05],  # happiness
        [0.05, 0.15, 0.80],  # sadness
        [0.05, 0.15, 0.80],  # anger
        [0.35, 0.55, 0.10],  # neutral
    ],
    dtype=np.float32,
)
W_ENG10 = np.array([7, 2, 1, 9, 2, 2, 5], dtype=np.float32)
IM_MU = np.array([0.485, 0.456, 0.406], np.float32)
IM_SD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------- Utils -----------------
def preprocess(bgr, size=224):
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (size, size)).astype(np.float32) / 255.0
    x = (x - IM_MU) / IM_SD
    return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / (e.sum() + 1e-9)


class EMA:
    def __init__(self, a=0.25):
        self.a = a
        self.v = None

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        self.v = x if self.v is None else self.a * x + (1 - self.a) * self.v
        return self.v


# ----------------- Threaded webcam -----------------
class Webcam:
    def __init__(self, cam=0, width=1280, height=720, backend=None):
        if backend is None:
            self.cap = cv2.VideoCapture(cam)
        else:
            self.cap = cv2.VideoCapture(cam, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.ok, self.frame = self.cap.read()
        self.alive = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.alive:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.ok, self.frame = ok, f
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ok, (None if self.frame is None else self.frame.copy())

    def release(self):
        self.alive = False
        try:
            self.t.join(timeout=0.2)
        except Exception:
            pass
        self.cap.release()


# ----------------- Model wrappers -----------------
class OnnxFER:
    def __init__(self, path):
        assert ort is not None, "onnxruntime not installed"
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.name = f"ONNX:{os.path.basename(path)}"

    def __call__(self, chw):
        return self.sess.run(None, {self.inp: chw})[0][0]


class TorchFER:
    def __init__(self, arch, ckpt, num_classes=7):
        assert torch is not None and timm is not None, "Install torch & timm"
        arch = arch.lower()
        if arch == "resnet18":
            model = timm.create_model(
                "resnet18", pretrained=False, num_classes=num_classes
            )
        elif arch in ["deit-tiny", "deit", "vit-tiny"]:
            model = timm.create_model(
                "deit_tiny_patch16_224", pretrained=False, num_classes=num_classes
            )
        else:
            raise ValueError("arch must be resnet18 | deit-tiny (Mini-X: export ONNX)")
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
        model.eval()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.dev)
        self.name = f"TORCH:{arch}:{os.path.basename(ckpt)}"

    @torch.no_grad()
    def __call__(self, chw):
        x = torch.from_numpy(chw).to(self.dev)
        return self.model(x).cpu().numpy()[0]


# ----------------- Drawing helpers -----------------
def put(img, txt, xy, s=0.55, col=(255, 255, 255), th=1):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def draw_rounded_rect(img, x, y, w, h, color, radius=12, thickness=-1):
    """Draw a filled or stroked rounded rectangle (approximation using rects + circles)."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    if w <= 0 or h <= 0:
        return
    # center rects
    cv2.rectangle(
        img, (x + radius, y), (x + w - radius, y + h), color, thickness, cv2.LINE_AA
    )
    cv2.rectangle(
        img, (x, y + radius), (x + w, y + h - radius), color, thickness, cv2.LINE_AA
    )
    # corners
    cv2.circle(img, (x + radius, y + radius), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (x + w - radius, y + radius), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (x + radius, y + h - radius), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(
        img, (x + w - radius, y + h - radius), radius, color, thickness, cv2.LINE_AA
    )


def draw_panel(img, x0, w):
    H, _ = img.shape[:2]
    # subtle shadow
    shadow_col = (8, 10, 12)
    draw_rounded_rect(
        img, x0 + 8, 18, w - 12, H - 36, shadow_col, radius=16, thickness=-1
    )
    # panel
    draw_rounded_rect(
        img, x0 + 4, 12, w - 12, H - 36, (20, 28, 36), radius=14, thickness=-1
    )


def bar(img, x, y, w, h, p, fill=(120, 220, 255)):
    p = float(np.clip(p, 0, 1))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * p), y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)


def gauge(img, cx, cy, r, v01, col=(70, 185, 255)):
    a0, a1 = 210, -30
    # bands
    cv2.ellipse(
        img,
        (cx, cy),
        (r, r),
        0,
        a0,
        a0 + (a1 - a0) * 0.33,
        (40, 55, 80),
        8,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        img,
        (cx, cy),
        (r, r),
        0,
        a0 + (a1 - a0) * 0.33,
        a0 + (a1 - a0) * 0.75,
        (55, 75, 110),
        8,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        img,
        (cx, cy),
        (r, r),
        0,
        a0 + (a1 - a0) * 0.75,
        a1,
        (70, 100, 140),
        8,
        cv2.LINE_AA,
    )
    # active arc + needle
    s = a0 + (a1 - a0) * float(np.clip(v01, 0, 1))
    cv2.ellipse(img, (cx, cy), (r, r), 0, a0, s, col, 6, cv2.LINE_AA)
    ang = np.deg2rad(210 + 240 * float(np.clip(v01, 0, 1)))
    tip = (int(cx + (r - 8) * np.cos(ang)), int(cy + (r - 8) * np.sin(ang)))
    cv2.line(img, (cx, cy), tip, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_timeline_box(img, values01, x, y, w=296, h=90, color=(70, 185, 255)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    for t, c in [(0.25, (40, 50, 70)), (0.6, (55, 70, 95))]:
        yy = y + h - int(h * t)
        cv2.line(img, (x, yy), (x + w, yy), c, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    if values01 and len(values01) > 1:
        pts = np.array(
            [
                [x + int(w * i / (len(values01) - 1)), y + h - int(h * float(v))]
                for i, v in enumerate(values01)
            ],
            np.int32,
        )
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)


# ----------- Pills, Help, Toast -----------
def draw_pill(img, text, x, y, color=(60, 160, 255)):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad_x = 12
    pad_y = 8
    box_w = w + pad_x * 2
    box_h = h + pad_y
    # draw subtle pill shadow
    shadow = (12, 14, 16)
    draw_rounded_rect(
        img, x + 2, y - h - 6 + 2, box_w, box_h, shadow, radius=12, thickness=-1
    )
    # pill background
    draw_rounded_rect(
        img, x, y - h - 6, box_w, box_h, (35, 40, 48), radius=12, thickness=-1
    )
    # border
    cv2.rectangle(img, (x, y - h - 6), (x + box_w, y + 2), (70, 70, 80), 1, cv2.LINE_AA)
    cv2.putText(
        img, text, (x + pad_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
    )
    return x + box_w + 12


def draw_help(canvas, W, H):
    overlay = canvas.copy()
    cv2.rectangle(overlay, (W - 420, 60), (W - 20, 380), (30, 30, 30), -1)
    put(overlay, "Keys", (W - 400, 88), 0.9)
    y = 120
    for k, desc in [
        ("1-4", "Tabs: Live / Analytics / Compare / Settings"),
        ("M", "Switch model"),
        ("E", "Toggle emotions"),
        ("L", "CSV live log"),
        ("S", "Snapshot"),
        ("X", "Export CSV+PNG+JSON"),
        ("P", "Pause/Resume"),
        ("{ / }", "Detection cadence +/-"),
        ("H", "Help"),
        ("Q", "Quit"),
    ]:
        put(overlay, f"{k:<6} {desc}", (W - 400, y), 0.7, (220, 220, 220), 1)
        y += 28
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)


# ----------------- Responsive canvas helpers -----------------
DESIGN_W, DESIGN_H = 1280, 720  # logical design size


def begin_canvas():
    return np.full((DESIGN_H, DESIGN_W, 3), (16, 20, 24), np.uint8)


def fit_into(img, box_w, box_h):
    if img is None or img.size == 0:
        return np.zeros((box_h, box_w, 3), np.uint8)
    h, w = img.shape[:2]
    r = min(box_w / max(1, w), box_h / max(1, h))
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((box_h, box_w, 3), np.uint8)
    x = (box_w - nw) // 2
    y = (box_h - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def get_window_size(title, fallback=(DESIGN_W, DESIGN_H)):
    try:
        _, _, w, h = cv2.getWindowImageRect(title)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return fallback


# ----------------- Export helpers -----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def export_session(history, perf, model_name, out_dir="exports"):
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"session_{ts}")

    # CSV
    csv_path = base + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "model", "eng10", "engaged", "neutral", "disengaged", *EMO])
        for h in history:
            w.writerow(
                [
                    f"{h['t']:.3f}",
                    h["model"],
                    f"{h['eng10']:.4f}",
                    f"{h['p3'][0]:.4f}",
                    f"{h['p3'][1]:.4f}",
                    f"{h['p3'][2]:.4f}",
                    *[f"{float(v):.4f}" for v in h["p7"]],
                ]
            )
    # PNG timeline
    png_path = base + "_timeline.png"
    W, H = 900, 300
    canvas = np.full((H, W, 3), (245, 247, 250), np.uint8)
    cv2.rectangle(canvas, (60, 40), (W - 40, H - 40), (210, 210, 210), 1)
    vals = [(h["eng10"] - 1) / 9.0 for h in history]
    if len(vals) > 1:
        step = max(1, len(vals) // (W - 120))
        slim = vals[::step][: W - 120]
        pts = np.array(
            [[60 + i, H - 40 - int((H - 80) * v)] for i, v in enumerate(slim)], np.int32
        )
        cv2.polylines(canvas, [pts], False, (70, 130, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Engagement Timeline (normalized 0..1)",
        (60, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(png_path, canvas)

    # JSON meta
    json_path = base + ".json"
    meta = {
        "model": model_name,
        "frames": len(history),
        "duration_sec": history[-1]["t"] - history[0]["t"] if history else 0.0,
        "mean_fps": (
            np.mean(perf.get(model_name, {}).get("fps", [0.0])) if perf else 0.0
        ),
        "export_time": ts,
        "notes": "Auto-exported engagement session (Responsive+Threaded+Cadence)",
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"csv": csv_path, "png": png_path, "json": json_path}


def save_snapshot(frame, out_dir="exports"):
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"snapshot_{ts}.png")
    cv2.imwrite(path, frame)
    return path


def render_demo_image(out_path="exports/demo_ui.png", size=(1280, 720)):
    """Render a static demo UI frame (no camera) and save it to out_path."""
    ensure_dir(os.path.dirname(out_path) or ".")
    W, H = size
    canvas = np.full((H, W, 3), (16, 20, 24), np.uint8)

    # sample state
    current_tab = "Live"
    paused = False
    log_on = True
    fer_name = "DEMO:resnet18"
    eng10 = 7.4
    p7 = np.array([0.05, 0.03, 0.02, 0.65, 0.08, 0.04, 0.13], np.float32)
    p7 = p7 / (p7.sum() + 1e-9)
    p3 = p7 @ EMO2ENG
    timeline = [((np.sin(i / 10.0) + 1.0) / 2.0) for i in range(120)]

    # top tabs
    tabs = ["Live", "Analytics", "Compare", "Settings"]
    x = 20
    tab_y = 24
    for i, t in enumerate(tabs):
        label = f"[{i+1}] {t}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        tab_w = tw + 20
        is_active = current_tab == t
        bg = (50, 95, 170) if is_active else (28, 32, 38)
        fg = (255, 255, 255) if is_active else (180, 180, 180)
        draw_rounded_rect(canvas, x - 6, 8, tab_w + 12, 36, bg, radius=10, thickness=-1)
        put(canvas, label, (x, tab_y), 0.6, fg, 1)
        x += tab_w + 14

    # state pills
    xpill = 24
    xpill = draw_pill(
        canvas,
        "LIVE" if not paused else "PAUSED",
        xpill,
        60,
        (90, 220, 120) if not paused else (200, 180, 80),
    )
    xpill = draw_pill(
        canvas,
        f"LOG {'ON' if log_on else 'OFF'}",
        xpill,
        60,
        (120, 200, 255) if log_on else (150, 150, 150),
    )
    draw_pill(canvas, fer_name[:22], xpill, 60, (160, 200, 255))

    # left live area placeholder image
    live_w, live_h = W - 320, H
    # create a fake camera-like placeholder
    cam_sample = np.full((live_h, live_w, 3), (40, 40, 48), np.uint8)
    cv2.putText(
        cam_sample,
        "Camera preview (demo)",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    canvas[0:live_h, 0:live_w] = cam_sample

    # right panel
    panel_x = W - 320
    draw_panel(canvas, panel_x, 320)
    put(canvas, "Engagement (1-10)", (panel_x + 12, 72), 0.65, (255, 255, 255), 2)
    gauge(canvas, panel_x + 160, 146, 60, (eng10 - 1) / 9)
    put(canvas, f"{eng10:.1f} / 10", (panel_x + 120, 180), 0.8, (255, 255, 255), 2)
    bar(canvas, panel_x + 12, 200, 296, 18, (eng10 - 1) / 9, (120, 220, 255))

    put(canvas, "3-Class", (panel_x + 12, 232), 0.6)
    for i, (lab, p) in enumerate(zip(ENG3, p3)):
        yy = 258 + i * 30
        put(canvas, lab, (panel_x + 12, yy), 0.55, (220, 220, 220), 1)
        bar(canvas, panel_x + 120, yy - 12, 180, 16, p, (255, 255, 255))

    slim = timeline[:120]
    draw_timeline_box(canvas, slim, panel_x + 12, 360, 296, 90)
    put(canvas, "Timeline (demo)", (panel_x + 12, 352), 0.6)

    # show emotions
    put(canvas, "Emotions", (panel_x + 12, 472), 0.6)
    for i, (lab, pp) in enumerate(zip(EMO, p7)):
        put(canvas, lab, (panel_x + 12, 494 + i * 20), 0.5, (200, 200, 200), 1)
        bar(canvas, panel_x + 110, 484 + i * 20, 180, 14, float(pp), (255, 255, 255))

    put(
        canvas,
        "Demo: static preview (no camera)",
        (panel_x + 12, H - 30),
        0.5,
        (180, 180, 180),
        1,
    )

    cv2.imwrite(out_path, canvas)
    return out_path


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend", action="append", choices=["onnx", "torch"], required=True
    )
    ap.add_argument("--weights", action="append", required=True)
    ap.add_argument(
        "--arch", action="append", help="for torch entries: resnet18 | deit-tiny"
    )
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--ema", type=float, default=0.25)
    ap.add_argument("--csv_live", default="engagement_log.csv")
    ap.add_argument(
        "--detect_every", type=int, default=4, help="run face detection every N frames"
    )
    ap.add_argument(
        "--demo", action="store_true", help="render a demo UI image and exit"
    )
    ap.add_argument(
        "--demo-out", default="exports/demo_ui.png", help="path for demo image output"
    )
    args = ap.parse_args()

    if args.demo:
        out = render_demo_image(out_path=args.demo_out)
        print(f"Demo image written: {out}")
        return

    # Build models list
    archs = args.arch or []
    models, t_idx = [], 0
    for b, w in zip(args.backend, args.weights):
        if b == "onnx":
            models.append(OnnxFER(w))
        else:
            assert t_idx < len(archs), "Missing --arch for a torch backend"
            models.append(TorchFER(archs[t_idx], w))
            t_idx += 1
    m_idx = 0
    fer = models[m_idx]

    # Face detector, smoothers, perf stats, history
    mp_fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )
    emo_s, eng3_s, eng10_s = EMA(args.ema), EMA(args.ema), EMA(args.ema)
    timeline = deque(maxlen=1200)  # ~60s @ ~20fps
    history = []  # list of dicts: {t, model, eng10, p3, p7}
    perf = defaultdict(lambda: {"fps": [], "eng10": []})

    # Threaded camera & resizable window
    TITLE = "Student Engagement Monitor — Hybrid Tabs (Responsive+Extras)"
    cam = Webcam(args.cam, 1280, 720)
    ok0, _ = cam.read()
    if not ok0:
        print("Camera not available")
        cam.release()
        return
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)

    paused = False
    show_emo = False
    show_help_flag = False
    log_on = False
    csv_file = None
    csv_writer = None
    fps = 0.0
    t0 = time.time()
    t_start = time.time()
    current_tab = "Live"
    export_msg, export_t = "", 0

    # cadence + cache
    detect_every = max(1, int(args.detect_every))
    frame_idx = 0
    last_bb = None

    while True:
        if not paused:
            ok, frame = cam.read()
            if not ok or frame is None:
                break

        # ---------- Face detection cadence ----------
        H_cam, W_cam = frame.shape[:2]
        run_det = (frame_idx % detect_every == 0) or (last_bb is None)
        if run_det:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = mp_fd.process(rgb).detections or []
            if dets:
                last_bb = dets[0].location_data.relative_bounding_box
            else:
                last_bb = None

        # ---------- FER ----------
        p7 = np.zeros(7, np.float32)
        if last_bb is not None:
            x1, y1 = int(last_bb.xmin * W_cam), int(last_bb.ymin * H_cam)
            x2, y2 = int((last_bb.xmin + last_bb.width) * W_cam), int(
                (last_bb.ymin + last_bb.height) * H_cam
            )
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W_cam, x2), min(H_cam, y2)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                face = frame[y1:y2, x1:x2]
                logits = fer(preprocess(face, args.size))
                p7 = softmax(logits)

        # Smooth & map
        p7 = emo_s(p7)
        s = p7.sum()
        if s > 0:
            p7 = p7 / s
        p3 = p7 @ EMO2ENG
        p3 = p3 / (p3.sum() + 1e-9)
        p3 = eng3_s(p3)
        p3 = p3 / (p3.sum() + 1e-9)
        eng10 = float(np.clip(W_ENG10 @ p7, 1, 10))
        eng10 = float(eng10_s(eng10))

        # FPS & buffers
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (now - t0))
        t0 = now
        timeline.append((eng10 - 1.0) / 9.0)
        since = now - t_start
        history.append(
            {
                "t": since,
                "model": fer.name,
                "eng10": eng10,
                "p3": p3.copy(),
                "p7": p7.copy(),
            }
        )
        perf[fer.name]["fps"].append(fps)
        perf[fer.name]["eng10"].append(eng10)
        frame_idx += 1

        # ----------------- Build logical canvas -----------------
        canvas = begin_canvas()
        H, W = DESIGN_H, DESIGN_W
        panel_x = W - 320

        # Tabbar — draw rounded tabs and highlight active
        tabs = ["Live", "Analytics", "Compare", "Settings"]
        x = 20
        tab_y = 24
        for i, t in enumerate(tabs):
            label = f"[{i+1}] {t}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            tab_w = tw + 20
            is_active = current_tab == t
            bg = (50, 95, 170) if is_active else (28, 32, 38)
            fg = (255, 255, 255) if is_active else (180, 180, 180)
            # background with small padding
            draw_rounded_rect(
                canvas, x - 6, 8, tab_w + 12, 36, bg, radius=10, thickness=-1
            )
            put(canvas, label, (x, tab_y), 0.6, fg, 1)
            x += tab_w + 14

        # State pills
        xpill = 24
        xpill = draw_pill(
            canvas,
            "LIVE" if not paused else "PAUSED",
            xpill,
            60,
            (90, 220, 120) if not paused else (200, 180, 80),
        )
        xpill = draw_pill(
            canvas,
            f"LOG {'ON' if log_on else 'OFF'}",
            xpill,
            60,
            (120, 200, 255) if log_on else (150, 150, 150),
        )
        draw_pill(canvas, fer.name[:22], xpill, 60, (160, 200, 255))

        # Left live area
        live_w, live_h = W - 320, H
        live_view = fit_into(frame, live_w, live_h)
        canvas[0:live_h, 0:live_w] = live_view

        # ----------------- Tabs -----------------
        if current_tab == "Live":
            draw_panel(canvas, panel_x, 320)
            put(
                canvas,
                "Engagement (1-10)",
                (panel_x + 12, 72),
                0.65,
                (255, 255, 255),
                2,
            )
            gauge(canvas, panel_x + 160, 146, 60, (eng10 - 1) / 9)
            put(
                canvas,
                f"{eng10:.1f} / 10",
                (panel_x + 120, 180),
                0.8,
                (255, 255, 255),
                2,
            )
            bar(canvas, panel_x + 12, 200, 296, 18, (eng10 - 1) / 9, (120, 220, 255))

            put(canvas, "3-Class", (panel_x + 12, 232), 0.6)
            for i, (lab, p) in enumerate(zip(ENG3, p3)):
                yy = 258 + i * 30
                put(canvas, lab, (panel_x + 12, yy), 0.55, (220, 220, 220), 1)
                bar(canvas, panel_x + 120, yy - 12, 180, 16, p, (255, 255, 255))

            vals = list(timeline)
            take = max(1, len(vals) // 120)
            slim = vals[::take][:120] if vals else []
            draw_timeline_box(canvas, slim, panel_x + 12, 360, 296, 90)
            put(canvas, "Timeline (60s)", (panel_x + 12, 352), 0.6)

            if show_emo:
                put(canvas, "Emotions", (panel_x + 12, 472), 0.6)
                for i, (lab, pp) in enumerate(zip(EMO, p7)):
                    put(
                        canvas,
                        lab,
                        (panel_x + 12, 494 + i * 20),
                        0.5,
                        (200, 200, 200),
                        1,
                    )
                    bar(
                        canvas,
                        panel_x + 110,
                        484 + i * 20,
                        180,
                        14,
                        float(pp),
                        (255, 255, 255),
                    )

            msg = (
                "Low Engagement"
                if eng10 < 4
                else ("High Engagement" if eng10 > 7.5 else "Stable")
            )
            put(canvas, msg, (panel_x + 160, 196), 0.55, (200, 220, 220), 2)
            put(
                canvas,
                "[Q]Quit  [P]Pause  [L]Log  [E]Emo  [M]Model  [S]Snap  [X]Export  [H]Help  [{ } ] Cadence",
                (panel_x + 12, H - 12),
                0.5,
                (180, 180, 180),
                1,
            )

        elif current_tab == "Analytics":
            cv2.rectangle(canvas, (20, 60), (W - 20, H - 80), (245, 247, 250), -1)
            put(
                canvas,
                "Analytics — Session Timeline & Stats",
                (28, 52),
                0.7,
                (40, 40, 40),
                2,
            )
            vals = [(h["eng10"] - 1) / 9.0 for h in history]
            if len(vals) > 1:
                step = max(1, len(vals) // (W - 120))
                slim = vals[::step][: W - 100]
                x0, y0, w, h = 40, 90, W - 80, 260
                for t, c in [(0.25, (210, 230, 255)), (0.6, (210, 220, 240))]:
                    yy = y0 + h - int(h * t)
                    cv2.line(canvas, (x0, yy), (x0 + w, yy), c, 1, cv2.LINE_AA)
                cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 1)
                pts = np.array(
                    [[x0 + i, y0 + h - int(h * v)] for i, v in enumerate(slim)],
                    np.int32,
                )
                cv2.polylines(canvas, [pts], False, (70, 130, 255), 2, cv2.LINE_AA)
                put(
                    canvas,
                    "Engagement timeline (normalized)",
                    (x0, y0 - 10),
                    0.55,
                    (60, 60, 60),
                    1,
                )
            if history:
                mean = float(np.mean([h["eng10"] for h in history]))
                mn = float(np.min([h["eng10"] for h in history]))
                mx = float(np.max([h["eng10"] for h in history]))
                put(canvas, f"Frames: {len(history)}", (40, 380), 0.7, (50, 50, 50), 2)
                put(
                    canvas,
                    f"Mean: {mean:.2f}  Min: {mn:.2f}  Max: {mx:.2f}",
                    (40, 410),
                    0.7,
                    (50, 50, 50),
                    2,
                )
                dur = history[-1]["t"] - history[0]["t"]
                put(canvas, f"Duration: {dur:.1f}s", (40, 440), 0.7, (50, 50, 50), 2)
            put(
                canvas,
                "Press [X] to export CSV + PNG + JSON",
                (40, H - 30),
                0.65,
                (70, 70, 70),
                2,
            )

        elif current_tab == "Compare":
            put(
                canvas,
                "Model Comparison — Mean FPS and Engagement (last 60s)",
                (20, 60),
                0.75,
                (230, 230, 230),
                2,
            )
            y = 100
            for model_name, stats in perf.items():
                fps_vals = stats["fps"][-60:] if stats["fps"] else [0.0]
                eng_vals = stats["eng10"][-60:] if stats["eng10"] else [0.0]
                mfps = float(np.mean(fps_vals))
                meng = float(np.mean(eng_vals))
                put(canvas, model_name[:38], (20, y), 0.65, (255, 255, 255), 2)
                put(canvas, f"FPS {mfps:5.1f}", (420, y), 0.6, (220, 220, 220), 1)
                bar(
                    canvas, 510, y - 14, 220, 16, min(mfps / 60.0, 1.0), (120, 220, 255)
                )
                put(canvas, f"Eng {meng:4.1f}/10", (760, y), 0.6, (220, 220, 220), 1)
                bar(canvas, 880, y - 14, 220, 16, (meng - 1) / 9.0, (255, 255, 255))
                y += 36
            put(
                canvas,
                "Switch models with [M] in Live tab to collect stats.",
                (20, DESIGN_H - 30),
                0.6,
                (180, 180, 180),
                1,
            )

        elif current_tab == "Settings":
            put(
                canvas,
                "Settings — (Demo) Adjust parameters via CLI flags",
                (20, 60),
                0.75,
                (230, 230, 230),
                2,
            )
            put(
                canvas,
                f"Backend(s): {', '.join(args.backend)}",
                (20, 100),
                0.65,
                (255, 255, 255),
                2,
            )
            put(canvas, f"Model: {fer.name}", (20, 132), 0.65, (220, 220, 220), 2)
            put(canvas, f"Input size: {args.size}", (20, 164), 0.65, (220, 220, 220), 2)
            put(canvas, f"EMA alpha: {args.ema}", (20, 196), 0.65, (220, 220, 220), 2)
            put(
                canvas,
                f"Detect every: {detect_every} frame(s)   ({{}} to change)",
                (20, 228),
                0.65,
                (220, 220, 220),
                2,
            )
            put(
                canvas,
                "Tip: Tune EMO2ENG & W_ENG10 to your course context.",
                (20, 260),
                0.6,
                (200, 200, 200),
                1,
            )

        # CSV live logging
        if log_on:
            if csv_file is None:
                csv_file = open(args.csv_live, "w", newline="")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    [
                        "ts",
                        "model",
                        "fps",
                        "eng10",
                        "engaged",
                        "neutral",
                        "disengaged",
                        *EMO,
                    ]
                )
            csv_writer.writerow(
                [
                    time.time(),
                    fer.name,
                    f"{fps:.2f}",
                    f"{eng10:.3f}",
                    f"{p3[0]:.4f}",
                    f"{p3[1]:.4f}",
                    f"{p3[2]:.4f}",
                    *[f"{float(v):.4f}" for v in p7],
                ]
            )
        else:
            if csv_file is not None:
                csv_file.close()
                csv_file = None
                csv_writer = None

        # Export toast (2.5s)
        if export_msg and time.time() - export_t < 2.5:
            bar_w = 820
            x = (DESIGN_W - bar_w) // 2
            y = DESIGN_H - 36
            cv2.rectangle(canvas, (x, y - 22), (x + bar_w, y + 10), (40, 40, 40), -1)
            put(canvas, export_msg, (x + 12, y), 0.6, (255, 255, 255), 1)

        # Help overlay
        if show_help_flag:
            draw_help(canvas, DESIGN_W, DESIGN_H)

        # Scale to window
        win_w, win_h = get_window_size(TITLE, (DESIGN_W, DESIGN_H))
        display = cv2.resize(canvas, (win_w, win_h), interpolation=cv2.INTER_AREA)
        cv2.imshow(TITLE, display)

        # Keys
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q"), ord("Q")):
            break
        elif k in (ord("p"), ord("P")):
            paused = not paused
        elif k in (ord("e"), ord("E")):
            show_emo = not show_emo
        elif k in (ord("l"), ord("L")):
            log_on = not log_on
        elif k in (ord("m"), ord("M")) and len(models) > 1:
            m_idx = (m_idx + 1) % len(models)
            fer = models[m_idx]
            emo_s.v = None
            eng3_s.v = None
            eng10_s.v = None
        elif k == ord("1"):
            current_tab = "Live"
        elif k == ord("2"):
            current_tab = "Analytics"
        elif k == ord("3"):
            current_tab = "Compare"
        elif k == ord("4"):
            current_tab = "Settings"
        elif k in (ord("s"), ord("S")):
            path = save_snapshot(display)
            print("Saved snapshot:", path)
        elif k in (ord("x"), ord("X")):
            out = export_session(history, perf, fer.name)
            export_msg = f"Exported: {os.path.basename(out['csv'])}, {os.path.basename(out['png'])}, {os.path.basename(out['json'])}"
            export_t = time.time()
            print("Exported:", out)
        elif k in (ord("h"), ord("H")):
            show_help_flag = not show_help_flag
        elif k == ord("{"):
            detect_every = max(1, detect_every - 1)
        elif k == ord("}"):
            detect_every += 1

    cam.release()
    cv2.destroyAllWindows()
    if csv_file is not None:
        csv_file.close()


if __name__ == "__main__":
    main()
