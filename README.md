# AETHER — Draw on Thin Air

A browser-based rebuild of the AI Virtual Painter. Runs entirely on
hand-tracking via **MediaPipe Hands**, streamed live from your browser's
webcam through **streamlit-webrtc** — no desktop install, no `cv2.imshow`,
shareable as a single link.

![status](https://img.shields.io/badge/status-working-5EEAD4)
![python](https://img.shields.io/badge/python-3.9%2B-blue)

## Live demo

Deploy in ~2 minutes on [Streamlit Community Cloud](https://streamlit.io/cloud) (free):

1. Push this folder to a public GitHub repo.
2. Go to share.streamlit.io → **New app** → point it at `app.py`.
3. Streamlit Cloud auto-installs `requirements.txt` **and** `packages.txt`
   (the apt-level libs OpenCV/PyAV need — already included here).
4. Done. Share the `*.streamlit.app` URL.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL, click **Start** on the video widget, and allow camera
access.

## How to use it

| Gesture | Effect |
|---|---|
| ☝️ Only index finger raised | **Draw / act** with the current tool |
| ✌️ Index + middle raised | Move without drawing (repositions the shape anchor) |

Tool, color, and stroke thickness are chosen from the **sidebar** — not by
hovering over an in-frame toolbar like the original desktop version. In a
browser, a mouse is already available and far more precise for menu
selection than a dwell-based hand gesture; the hand tracking is spent
entirely on the part that's actually the point of the project: drawing in
the air.

- **Undo / Redo / Clear** — sidebar buttons, backed by a 25-state snapshot
  history of the canvas.
- **Download** — exports the current canvas as a PNG, client-side, no
  server storage.

## Architecture

```
Browser webcam
     │  (WebRTC)
     ▼
streamlit-webrtc  ── runs PainterProcessor.recv() on a dedicated media thread
     │
     ├─ MediaPipe Hands → 21 hand landmarks per frame
     ├─ Gesture state machine → draw / line / rectangle / circle / erase
     ├─ OpenCV canvas compositing (mask + camera frame)
     ▼
Streamed back to the <video> element in the browser
```

Streamlit's main thread (widgets, buttons) and the WebRTC media thread run
**concurrently**. `PainterProcessor` exposes a small, `threading.Lock`-protected
surface (`tool`, `color`, `thickness`, `pending_action`) — the UI thread
writes to it, the media thread reads it once per frame. The canvas mask and
undo history live exclusively on the media thread to avoid tearing.

## Files

```
app.py              # entire application (UI + video processor)
requirements.txt    # pip dependencies
packages.txt         # apt dependencies (OpenCV/PyAV need libgl1, ffmpeg, etc.)
.streamlit/config.toml  # dark theme + server config
```

## Known constraints

- Cold starts on free hosting tiers can take 10-30s to spin up MediaPipe.
- Webcam access requires HTTPS (Streamlit Cloud provides this automatically;
  for local dev, `localhost` is exempt from the HTTPS requirement).
- One hand tracked at a time (`max_num_hands=1`) — keeps latency low on
  free-tier CPUs. Bump it in `PainterProcessor.__init__` if you deploy on
  something beefier.
