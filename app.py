"""
AETHER — Web Edition
============================
AI hand-tracked painter, rebuilt for the browser with streamlit-webrtc.

Architecture:
  - PainterProcessor (VideoProcessorBase) owns all per-frame state (canvas
    mask, undo/redo history, current tool/color/thickness) and runs on
    streamlit-webrtc's dedicated media thread.
  - The Streamlit main thread never touches OpenCV frames directly — it only
    reads/writes small, lock-protected attributes on the processor
    (tool, color, thickness, pending_action). This is the standard safe
    pattern for streamlit-webrtc: cross-thread state must be tiny and locked.
  - Tool/color selection moves to real UI widgets (sidebar) instead of the
    original project's "hover over a toolbar" gesture — mouse input is
    available in a browser, so we spend the hand-tracking budget on the
    part that's actually the point: freehand drawing/erasing in the air.
"""

import threading
import time
from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AETHER",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FRAME_W, FRAME_H = 640, 480
MAX_HISTORY = 25

TOOLS = ["draw", "line", "rectangle", "circle", "erase"]

# BGR tuples (OpenCV order) mapped to display hex for the UI swatches
PALETTE = [
    ("Signal Red", (60, 60, 255), "#FF3C3C"),
    ("Aether Cyan", (212, 234, 94), "#5EEAD4"),
    ("Voltage Violet", (250, 139, 167), "#A78BFA"),
    ("Alert Amber", (0, 176, 255), "#FFB000"),
    ("Mint", (166, 227, 161), "#A1E3A6"),
    ("Pure White", (255, 255, 255), "#FFFFFF"),
]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ----------------------------------------------------------------------
# Theme — dark "computer-vision HUD" aesthetic
# ----------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #0B0E14;
  --surface: #10141F;
  --surface-2: #161B29;
  --border: #232A3B;
  --text: #E6E8EF;
  --muted: #7A8299;
  --cyan: #5EEAD4;
  --violet: #A78BFA;
  --amber: #FFB000;
}

html, body, [class*="css"]  {
  font-family: 'Inter', sans-serif;
  color: var(--text);
}

.stApp {
  background:
    radial-gradient(circle at 15% 0%, rgba(94,234,212,0.06), transparent 40%),
    radial-gradient(circle at 85% 10%, rgba(167,139,250,0.06), transparent 40%),
    var(--bg);
}

section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.01em; }

.aether-hero {
  display: flex; align-items: center; gap: 22px;
  padding: 22px 26px; margin-bottom: 18px;
  background: linear-gradient(135deg, rgba(94,234,212,0.06), rgba(167,139,250,0.05));
  border: 1px solid var(--border);
  border-radius: 14px;
  position: relative; overflow: hidden;
}

.aether-hero::after {
  content: "";
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(94,234,212,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(94,234,212,0.05) 1px, transparent 1px);
  background-size: 28px 28px;
  mask-image: linear-gradient(to right, black, transparent 80%);
  pointer-events: none;
}

.aether-hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.1rem; font-weight: 700; margin: 0;
  background: linear-gradient(90deg, var(--cyan), var(--violet));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

.aether-hero-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem; color: var(--muted); margin-top: 6px; letter-spacing: 0.02em;
}

.aether-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem; color: var(--cyan);
  background: rgba(94,234,212,0.08);
  border: 1px solid rgba(94,234,212,0.25);
  padding: 3px 10px; border-radius: 100px; margin-top: 10px;
}
.aether-badge .dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--cyan);
  box-shadow: 0 0 6px var(--cyan);
  animation: pulse 1.6s ease-in-out infinite;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

.hud-panel {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
  margin-bottom: 14px;
}
.hud-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin-bottom: 8px;
}

.stat-row { display: flex; gap: 10px; }
.stat-chip {
  flex: 1; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 12px; text-align: center;
}
.stat-chip .v {
  font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 600;
  color: var(--cyan);
}
.stat-chip .k {
  font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted);
  text-transform: uppercase; margin-top: 2px;
}

div[data-testid="stVerticalBlock"] div[role="radiogroup"] label {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px 10px !important;
  margin-bottom: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
}

.stButton>button {
  font-family: 'JetBrains Mono', monospace;
  background: var(--surface); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px;
  transition: all 0.15s ease;
}
.stButton>button:hover {
  border-color: var(--cyan); color: var(--cyan);
}

.footer-note {
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
  color: var(--muted); text-align: center; margin-top: 18px;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Hero header with an animated hand-landmark "constellation" signature
# ----------------------------------------------------------------------
HERO_SVG = """
<svg width="86" height="86" viewBox="0 0 86 86" style="flex-shrink:0;">
  <g stroke="#5EEAD4" stroke-width="1.4" fill="none" opacity="0.9">
    <line x1="43" y1="70" x2="30" y2="46"><animate attributeName="opacity" values="0.4;1;0.4" dur="2.4s" repeatCount="indefinite"/></line>
    <line x1="43" y1="70" x2="43" y2="40"><animate attributeName="opacity" values="1;0.4;1" dur="2.1s" repeatCount="indefinite"/></line>
    <line x1="43" y1="70" x2="56" y2="42"><animate attributeName="opacity" values="0.5;1;0.5" dur="2.7s" repeatCount="indefinite"/></line>
    <line x1="43" y1="70" x2="66" y2="50"><animate attributeName="opacity" values="0.7;0.3;0.7" dur="1.9s" repeatCount="indefinite"/></line>
    <line x1="30" y1="46" x2="26" y2="20"/>
    <line x1="43" y1="40" x2="41" y2="12"/>
    <line x1="56" y1="42" x2="58" y2="16"/>
    <line x1="66" y1="50" x2="74" y2="34"/>
  </g>
  <g fill="#A78BFA">
    <circle cx="43" cy="70" r="3.4"/>
    <circle cx="30" cy="46" r="2.4"/><circle cx="26" cy="20" r="2.4"><animate attributeName="r" values="2.4;3.4;2.4" dur="2.4s" repeatCount="indefinite"/></circle>
    <circle cx="43" cy="40" r="2.4"/><circle cx="41" cy="12" r="2.4"><animate attributeName="r" values="2.4;3.4;2.4" dur="2.1s" repeatCount="indefinite"/></circle>
    <circle cx="56" cy="42" r="2.4"/><circle cx="58" cy="16" r="2.4"><animate attributeName="r" values="2.4;3.4;2.4" dur="2.7s" repeatCount="indefinite"/></circle>
    <circle cx="66" cy="50" r="2.4"/><circle cx="74" cy="34" r="2.4"><animate attributeName="r" values="2.4;3.4;2.4" dur="1.9s" repeatCount="indefinite"/></circle>
  </g>
</svg>
"""

st.markdown(
    f"""
    <div class="aether-hero">
      {HERO_SVG}
      <div>
        <p class="aether-hero-title">AETHER</p>
        <p class="aether-hero-sub">DRAW ON THIN AIR &middot; REAL-TIME HAND-LANDMARK TRACKING &middot; MEDIAPIPE + OPENCV</p>
        <span class="aether-badge"><span class="dot"></span> LIVE MODEL RUNNING ON-DEVICE</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Painting engine
# ----------------------------------------------------------------------


class History:
    """Bounded undo/redo snapshot stack for the canvas mask."""

    def __init__(self, initial):
        self._stack = [initial.copy()]
        self._pos = 0

    def push(self, state):
        self._stack = self._stack[: self._pos + 1]
        self._stack.append(state.copy())
        if len(self._stack) > MAX_HISTORY:
            self._stack.pop(0)
        self._pos = len(self._stack) - 1

    def undo(self):
        if self._pos > 0:
            self._pos -= 1
        return self._stack[self._pos].copy()

    def redo(self):
        if self._pos < len(self._stack) - 1:
            self._pos += 1
        return self._stack[self._pos].copy()


def index_raised(yi, y9):
    return (y9 - yi) > 40


class PainterProcessor:
    """Owns canvas + hand-tracking state. Runs on the streamlit-webrtc
    media thread via recv(); the Streamlit UI thread only touches the
    small, lock-protected control attributes."""

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_hands=1,
        )
        self.draw_utils = mp.solutions.drawing_utils

        self.mask = np.ones((FRAME_H, FRAME_W, 3), dtype="uint8") * 255
        self.history = History(self.mask)

        # control state (written by UI thread under lock)
        self.tool = "draw"
        self.color = (60, 60, 255)
        self.thickness = 5
        self.pending_action = None  # "undo" | "redo" | "clear" | None

        # drawing state (owned by media thread only)
        self.prevx, self.prevy = 0, 0
        self.var_inits = False
        self.xii, self.yii = 0, 0
        self.fps_times = deque(maxlen=30)

        self.lock = threading.Lock()
        self.last_frame_out = None  # for save/snapshot

    # -- called from UI thread --
    def set_controls(self, tool=None, color=None, thickness=None):
        with self.lock:
            if tool is not None:
                self.tool = tool
            if color is not None:
                self.color = color
            if thickness is not None:
                self.thickness = thickness

    def request(self, action):
        with self.lock:
            self.pending_action = action

    def get_mask_copy(self):
        with self.lock:
            return self.mask.copy()

    # -- called from media thread --
    def recv(self, frame):
        t0 = time.time()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (FRAME_W, FRAME_H))

        with self.lock:
            tool = self.tool
            color = self.color
            thickness = self.thickness
            action = self.pending_action
            self.pending_action = None

        if action == "undo":
            self.mask = self.history.undo()
        elif action == "redo":
            self.mask = self.history.redo()
        elif action == "clear":
            self.mask = np.ones((FRAME_H, FRAME_W, 3), dtype="uint8") * 255
            self.history.push(self.mask)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                self.draw_utils.draw_landmarks(
                    img, handlms, mp.solutions.hands.HAND_CONNECTIONS,
                    self.draw_utils.DrawingSpec(color=(94, 234, 212), thickness=1, circle_radius=2),
                    self.draw_utils.DrawingSpec(color=(167, 139, 250), thickness=1),
                )

                x = int(handlms.landmark[8].x * FRAME_W)
                y = int(handlms.landmark[8].y * FRAME_H)
                xi = int(handlms.landmark[12].x * FRAME_W)
                yi = int(handlms.landmark[12].y * FRAME_H)
                y9 = int(handlms.landmark[9].y * FRAME_H)
                active = index_raised(yi, y9)

                if tool == "draw":
                    if active:
                        if self.prevx == 0 and self.prevy == 0:
                            self.prevx, self.prevy = x, y
                        cv2.line(self.mask, (self.prevx, self.prevy), (x, y), color, thickness)
                        self.prevx, self.prevy = x, y
                    else:
                        if self.prevx or self.prevy:
                            self.history.push(self.mask)
                        self.prevx, self.prevy = 0, 0

                elif tool == "erase":
                    if active:
                        cv2.circle(img, (x, y), 32, (255, 255, 255), -1)
                        cv2.circle(self.mask, (x, y), 32, (255, 255, 255), -1)
                    else:
                        self.history.push(self.mask)

                elif tool == "line":
                    if active:
                        if not self.var_inits:
                            self.xii, self.yii = x, y
                            self.var_inits = True
                        cv2.line(img, (self.xii, self.yii), (x, y), color, thickness)
                    else:
                        if self.var_inits:
                            cv2.line(self.mask, (self.xii, self.yii), (x, y), color, thickness)
                            self.history.push(self.mask)
                        self.var_inits = False

                elif tool == "rectangle":
                    if active:
                        if not self.var_inits:
                            self.xii, self.yii = x, y
                            self.var_inits = True
                        cv2.rectangle(img, (self.xii, self.yii), (x, y), color, thickness)
                    else:
                        if self.var_inits:
                            cv2.rectangle(self.mask, (self.xii, self.yii), (x, y), color, thickness)
                            self.history.push(self.mask)
                        self.var_inits = False

                elif tool == "circle":
                    if active:
                        if not self.var_inits:
                            self.xii, self.yii = x, y
                            self.var_inits = True
                        r = int(((self.xii - x) ** 2 + (self.yii - y) ** 2) ** 0.5)
                        cv2.circle(img, (self.xii, self.yii), r, color, thickness)
                    else:
                        if self.var_inits:
                            r = int(((self.xii - x) ** 2 + (self.yii - y) ** 2) ** 0.5)
                            cv2.circle(self.mask, (self.xii, self.yii), r, color, thickness)
                            self.history.push(self.mask)
                        self.var_inits = False

        # composite mask onto camera frame (white = transparent)
        gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
        inv_3ch = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        bg = cv2.bitwise_and(img, cv2.bitwise_not(inv_3ch))
        fg = cv2.bitwise_and(self.mask, inv_3ch)
        out = cv2.add(bg, fg)

        # tiny HUD burned into the corner of the stream itself
        self.fps_times.append(time.time() - t0)
        fps = 1.0 / (sum(self.fps_times) / len(self.fps_times) + 1e-6)
        cv2.putText(out, f"{tool.upper()}", (12, FRAME_H - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, f"{fps:4.1f} FPS", (12, FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        with self.lock:
            self.last_frame_out = out

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ----------------------------------------------------------------------
# Sidebar — control panel
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="hud-label">// Tool</div>', unsafe_allow_html=True)
    tool = st.radio("Tool", TOOLS, horizontal=False, label_visibility="collapsed",
                     format_func=lambda t: {
                         "draw": "✏️  Freehand Draw",
                         "line": "／  Straight Line",
                         "rectangle": "▭  Rectangle",
                         "circle": "◯  Circle",
                         "erase": "⌫  Eraser",
                     }[t])

    st.markdown('<div class="hud-label" style="margin-top:16px;">// Color</div>', unsafe_allow_html=True)
    color_names = [c[0] for c in PALETTE]
    color_choice = st.radio("Color", color_names, horizontal=True,
                             label_visibility="collapsed")
    chosen = next(c for c in PALETTE if c[0] == color_choice)
    st.markdown(
        f'<div style="width:100%;height:10px;border-radius:6px;'
        f'background:{chosen[2]};margin:-6px 0 10px 0;"></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hud-label" style="margin-top:6px;">// Stroke Thickness</div>', unsafe_allow_html=True)
    thickness = st.slider("Thickness", 1, 30, 5, label_visibility="collapsed")

    st.markdown('<div class="hud-label" style="margin-top:16px;">// History</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    do_undo = c1.button("↶ Undo", use_container_width=True)
    do_redo = c2.button("↷ Redo", use_container_width=True)
    do_clear = c3.button("✕ Clear", use_container_width=True)

    st.markdown('<div class="hud-label" style="margin-top:16px;">// How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#7A8299;line-height:1.7;">
        1. Allow camera access below<br>
        2. Raise <b style="color:#5EEAD4;">only your index finger</b> to draw<br>
        3. Raise index + middle to move without drawing<br>
        4. Pick tool / color / thickness here anytime
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Main layout — live stream + HUD stat panel
# ----------------------------------------------------------------------
col_stream, col_hud = st.columns([2.4, 1], gap="large")

with col_stream:
    ctx = webrtc_streamer(
        key="aether-painter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=PainterProcessor,
        media_stream_constraints={
            "video": {"width": FRAME_W, "height": FRAME_H},
            "audio": False,
        },
        async_processing=True,
    )

with col_hud:
    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    st.markdown('<div class="hud-label">// Session Status</div>', unsafe_allow_html=True)

    if ctx.state.playing:
        status_txt, status_color = "TRACKING", "#5EEAD4"
    else:
        status_txt, status_color = "IDLE", "#7A8299"

    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-chip"><div class="v" style="color:{status_color};">{status_txt}</div>'
        f'<div class="k">status</div></div>'
        f'<div class="stat-chip"><div class="v">{tool.upper()}</div><div class="k">tool</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-row" style="margin-top:10px;">'
        f'<div class="stat-chip"><div class="v">{thickness}px</div><div class="k">thickness</div></div>'
        f'<div class="stat-chip"><div class="v" style="color:{chosen[2]};">■</div><div class="k">{color_choice}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    st.markdown('<div class="hud-label">// Export</div>', unsafe_allow_html=True)
    if ctx.video_processor:
        snapshot = ctx.video_processor.get_mask_copy()
        ok, buf = cv2.imencode(".png", snapshot)
        if ok:
            st.download_button(
                "⬇ Download Painting (PNG)",
                data=buf.tobytes(),
                file_name=f"aether_painting_{int(time.time())}.png",
                mime="image/png",
                use_container_width=True,
            )
    else:
        st.caption("Start the stream to enable export.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    st.markdown('<div class="hud-label">// Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.76rem;color:#7A8299;line-height:1.9;">
        Browser cam <span style="color:#5EEAD4;">→</span> WebRTC<br>
        <span style="color:#5EEAD4;">→</span> MediaPipe Hands (21 landmarks)<br>
        <span style="color:#5EEAD4;">→</span> Gesture state machine<br>
        <span style="color:#5EEAD4;">→</span> OpenCV canvas compositing<br>
        <span style="color:#5EEAD4;">→</span> Streamed back to browser
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# push control state / actions into the live processor
if ctx.video_processor:
    ctx.video_processor.set_controls(tool=tool, color=chosen[1], thickness=thickness)
    if do_undo:
        ctx.video_processor.request("undo")
    if do_redo:
        ctx.video_processor.request("redo")
    if do_clear:
        ctx.video_processor.request("clear")

st.markdown(
    '<p class="footer-note">AETHER · MediaPipe + OpenCV + Streamlit-WebRTC · '
    'No server-side camera access — video never leaves your session</p>',
    unsafe_allow_html=True,
)
