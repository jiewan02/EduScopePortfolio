# EduScope – Classroom Engagement Monitor

A real-time classroom engagement monitoring system using your webcam. The app detects faces, tracks individuals, and estimates engagement levels (focus, eye openness, head pose, hand raises, phone usage) displayed on a live dashboard.

---

## Requirements

- **Python 3.10 or 3.11** (recommended; 3.12 also works)
- **A webcam** connected to your computer
- **~4 GB free disk space** (for Python packages, mainly PyTorch)
- Internet connection for the first install (to download packages)

---

## Project Structure

```
EduScopePortfolio/
├── backend/          # FastAPI server + ML inference engine
│   ├── __init__.py
│   ├── server.py
│   ├── engine.py
│   ├── config.py
│   ├── tracker.py
│   └── utils.py
├── frontend/         # Browser dashboard (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/           # Pre-trained YOLO model weights
│   ├── yolo12n_50epoch_widerface.pt
│   ├── yolo12n_FT_openimage.pt
│   ├── yolov8n_FT_openimage.pt
│   ├── usingphone_last.pt
│   └── usingphone_first.pt
├── logs/             # Session logs saved automatically at runtime
├── requirements.txt
├── run.sh            # One-command launcher (macOS/Linux)
└── README.md
```

---

## Step-by-Step Setup

### Step 1 — Check your Python version

Open a terminal and run:

**macOS / Linux:**
```bash
python3 --version
```

**Windows:**
```cmd
python --version
```

You need **Python 3.10, 3.11, or 3.12**. If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/).

---

### Step 2 — Clone the repository

```bash
git clone https://github.com/jiewan02/EduScopePortfolio.git
cd EduScopePortfolio
```

If you downloaded a ZIP file instead, unzip it and navigate into the folder:

```bash
cd /path/to/EduScopePortfolio
```

---

### Step 3 — Create a virtual environment

A virtual environment keeps the project's dependencies isolated from your system Python.

**macOS / Linux:**
```bash
python3 -m venv venv
```

**Windows:**
```cmd
python -m venv venv
```

This creates a `venv/` folder inside the project directory.

---

### Step 4 — Activate the virtual environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

After activation, your terminal prompt will show `(venv)` at the start.

> **macOS / Linux users running `./run.sh`:** The script activates the virtual environment automatically — you do not need to activate it manually before using the run script.
>
> **Windows users or anyone running uvicorn directly (Step 6):** You must activate the virtual environment each time you open a new terminal before running the app.

---

### Step 5 — Install dependencies

First, upgrade pip to avoid installation issues with complex packages:

```bash
pip install --upgrade pip
```

Then install all required packages:

```bash
pip install -r requirements.txt
```

This will install FastAPI, Uvicorn, OpenCV, MediaPipe, PyTorch, SciPy, and Ultralytics YOLO. It may take **5–10 minutes** depending on your internet speed.

> **Note for Apple Silicon Macs (M1/M2/M3/M4):** `pip install -r requirements.txt` should work directly. If you get an error installing `torch`, install it separately first, then re-run:
> ```bash
> pip install torch torchvision torchaudio
> pip install -r requirements.txt
> ```

> **Note for Windows:** If `opencv-python` fails, try:
> ```bash
> pip install opencv-python-headless
> ```

---

### Step 6 — Run the server

**macOS / Linux (using the run script):**
```bash
./run.sh
```

> The script automatically activates the `venv/` virtual environment — no manual activation needed.

**Any platform (using uvicorn directly):**

> Make sure the virtual environment is activated first (Step 4).

```bash
uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

You should see output like:
```
Starting EduScope server…
Open http://localhost:8000 in your browser.

[Engine] Loading models…
[Engine] All models loaded. Face detection: YOLO (widerface)
[Engine] Ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> **The first startup takes 30–60 seconds** while the models load into memory. Wait for `[Engine] Ready.` before opening the browser.

---

### Step 7 — Open the dashboard

Open your web browser and go to:

```
http://localhost:8000
```

---

### Step 8 — Start monitoring

1. Click the **▶ Start Camera** button in the top-right corner.
2. When your browser asks for camera permission, click **Allow**.
3. The dashboard will show a live video feed with engagement overlays.
4. Click **■ Stop** to end the session. Logs are automatically saved to the `logs/` folder.
5. Click **↺ Reset** to clear the current session's stats and chart without stopping the camera.

---

## What You See on the Dashboard

| Element | Description |
|---|---|
| Colored box around a face | Green = high engagement (≥60%), Orange = medium (35–60%), Red = low (<35%) |
| Engagement bar | Thin bar at the bottom of each face box showing engagement level |
| `P0`, `P1`, … labels | Person IDs assigned by the tracker |
| 😴 tag | Eyes closed for more than 2 seconds (drowsy) |
| ✋ tag | Hand raised |
| 📱 tag | Phone detected near the person |
| Session Stats panel | Live count of people, average engagement %, processing FPS, elapsed time |
| Engagement Timeline chart | Per-person engagement history over the session |

---

## Troubleshooting

**"Camera access denied"**
- Make sure your browser has permission to access the camera.
- In Chrome: click the camera icon in the address bar → Allow.
- In Safari: System Settings → Privacy & Security → Camera → allow your browser.

**`Permission denied` when running `./run.sh`**
- Make the script executable and try again:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

**Server starts but models fail to load / FileNotFoundError**
- Confirm all 5 `.pt` files are present in the `models/` folder.
- Run `ls models/` (or `dir models\` on Windows) and verify the filenames match exactly.

**`ModuleNotFoundError: No module named 'ultralytics'` (or any package)**
- Make sure the virtual environment is activated (you should see `(venv)` in your prompt).
- Re-run `pip install --upgrade pip && pip install -r requirements.txt`.

**Dashboard shows "Loading models…" indefinitely**
- Check the terminal for error messages. Common causes: missing model file, wrong Python version, or a package installation problem.

**Very low FPS (< 2)**
- The system is CPU-bound. Close other heavy applications.
- For significantly better performance, a machine with an NVIDIA GPU and CUDA installed is recommended.

**`Address already in use` error**
- Port 8000 is taken. Either stop the other process or run on a different port:
  ```bash
  uvicorn backend.server:app --host 0.0.0.0 --port 8001 --reload
  ```
  Then open `http://localhost:8001` instead.

---

## Stopping the Server

Press **Ctrl+C** in the terminal where the server is running.

To deactivate the virtual environment afterward:
```bash
deactivate
```

---

## Session Logs

Each monitoring session is automatically saved as a `.jsonl` file in the `logs/` folder (e.g., `logs/session_a1b2c3d4.jsonl`). Each line is a JSON record containing:

```json
{
  "ts": "2026-04-29T10:30:00.123456",
  "elapsed": 12.5,
  "avg_engagement": 0.72,
  "persons": [...]
}
```

You can view past sessions via the API at `http://localhost:8000/api/sessions`.
