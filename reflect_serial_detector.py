import cv2
import numpy as np
import time
import threading
import csv
import os
import ctypes
from collections import deque

# ================= CONFIGURATION =================
ESP32_STREAM_URL = "http://192.168.4.1:81/stream"  # ESP32 AP default stream URL
SHORT_TERM_THRESHOLD = 60.0  # <60s = Yellow Alert, >=60s = Red Critical Alert

# Native Screen Resolution Detection (Windows)
try:
    user32 = ctypes.windll.user32
    SCREEN_W = user32.GetSystemMetrics(0)
    SCREEN_H = user32.GetSystemMetrics(1)
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080

CANVAS_W = SCREEN_W
CANVAS_H = SCREEN_H

# Palette (Crisp Neomorphic Slate & Glassmorphism)
BG_CANVAS      = (244, 247, 250)
CARD_BG        = (255, 255, 255)
CARD_BORDER    = (218, 226, 235)
CARD_SHADOW    = (200, 210, 222)
TEXT_PRIMARY   = (30, 41, 59)
TEXT_MUTED     = (100, 116, 139)
ACCENT_BLUE    = (214, 116, 14)   # Cerulean
ACCENT_GREEN   = (52, 168, 83)    # Emerald Idle
ACCENT_YELLOW  = (38, 178, 245)   # Amber Warning
ACCENT_RED     = (46, 49, 235)    # Crimson Alert

# Precompute Gamma Lookup Table (Night Vision Lift)
GAMMA_VALUE = 0.50
inv_gamma = 1.0 / GAMMA_VALUE
GAMMA_LUT = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# 24 Seats Matrix
ROWS = ["A", "B", "C", "D"]
COLS = [1, 2, 3, 4, 5, 6]

# Background Stream Worker
class ESP32StreamManager:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.is_connected = False
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                self.is_connected = False
                time.sleep(1.0)
                continue

            self.is_connected = True
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.is_connected = False
                    break
                self.frame = frame
                time.sleep(0.01)
            cap.release()
            self.is_connected = False
            time.sleep(1.0)

    def get_frame(self):
        return self.is_connected, self.frame

stream_mgr = ESP32StreamManager(ESP32_STREAM_URL)

# Graphics Helpers
def draw_rounded_rect(img, pt1, pt2, color, radius=14, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, lineType=cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)

def draw_neomorphic_card(canvas, x1, y1, x2, y2, radius=14):
    draw_rounded_rect(canvas, (x1 + 2, y1 + 3), (x2 + 3, y2 + 4), CARD_SHADOW, radius, -1)
    draw_rounded_rect(canvas, (x1, y1), (x2, y2), CARD_BG, radius, -1)
    draw_rounded_rect(canvas, (x1, y1), (x2, y2), CARD_BORDER, radius, 1)

# Strict Aspect Ratio Preservation Scaler
def fit_exact_aspect_ratio(src_image, container_w, container_h):
    src_h, src_w = src_image.shape[:2]
    src_aspect = src_w / float(src_h)
    box_aspect = container_w / float(container_h)

    if box_aspect > src_aspect:
        scaled_h = container_h
        scaled_w = int(scaled_h * src_aspect)
    else:
        scaled_w = container_w
        scaled_h = int(scaled_w / src_aspect)

    resized = cv2.resize(src_image, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
    
    # Neutral matte background
    canvas_box = np.full((container_h, container_w, 3), (230, 235, 242), dtype=np.uint8)
    ox = (container_w - scaled_w) // 2
    oy = (container_h - scaled_h) // 2
    canvas_box[oy:oy + scaled_h, ox:ox + scaled_w] = resized
    
    return canvas_box, scaled_w, scaled_h, ox, oy

# Proportional Cinema Seating Matrix Map
def compute_cinema_grid(view_w, view_h, off_x=0, off_y=0):
    grid = {}
    grid_top    = off_y + int(view_h * 0.12)
    grid_bottom = off_y + int(view_h * 0.90)
    grid_left   = off_x + int(view_w * 0.05)
    grid_right  = off_x + int(view_w * 0.95)

    total_w = grid_right - grid_left
    total_h = grid_bottom - grid_top

    row_h = total_h // len(ROWS)
    aisle_w = int(total_w * 0.08)
    block_w = (total_w - aisle_w) // 6

    for r_idx, r in enumerate(ROWS):
        y1 = grid_top + r_idx * row_h + int(row_h * 0.12)
        y2 = grid_top + (r_idx + 1) * row_h - int(row_h * 0.12)

        for c_idx, c in enumerate(COLS):
            sid = f"{r}{c}"
            if c_idx < 3:
                x1 = grid_left + c_idx * block_w + 5
                x2 = grid_left + (c_idx + 1) * block_w - 5
            else:
                x1 = grid_left + c_idx * block_w + aisle_w + 5
                x2 = grid_left + (c_idx + 1) * block_w + aisle_w - 5

            grid[sid] = {
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2)
            }
    return grid

# Optical Lens Reticle Drawing Utility
def draw_optical_reticle(img, cx, cy, label, duration, is_alert):
    color = ACCENT_RED if is_alert else ACCENT_YELLOW
    
    # Outer target ring
    cv2.circle(img, (cx, cy), 16, color, 2, lineType=cv2.LINE_AA)
    # Inner pinpoint glint lock
    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    
    # Crosshair alignment notches
    cv2.line(img, (cx - 22, cy), (cx - 18, cy), color, 2, lineType=cv2.LINE_AA)
    cv2.line(img, (cx + 18, cy), (cx + 22, cy), color, 2, lineType=cv2.LINE_AA)
    cv2.line(img, (cx, cy - 22), (cx, cy - 18), color, 2, lineType=cv2.LINE_AA)
    cv2.line(img, (cx, cy + 18), (cx, cy + 22), color, 2, lineType=cv2.LINE_AA)

    # Telemetry Label
    tag = f"{label} [{duration:.0f}s]"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
    lx, ly = cx + 22, cy - 6
    draw_rounded_rect(img, (lx - 4, ly - th - 4), (lx + tw + 6, ly + 4), (20, 25, 32), radius=4, thickness=-1)
    cv2.putText(img, tag, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)

# Active State Variables
active_tracks = {}
threat_history = deque(maxlen=240)
start_time = time.time()
camera_only_mode = False
is_fullscreen = True

CSV_FILE = "reflect_piracy_audit.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Location", "Status", "Duration_Sec", "X", "Y"])

WINDOW_NAME = "R.E.F.L.E.C.T. Command Center"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def on_mouse(event, x, y, flags, param):
    global is_fullscreen
    if event == cv2.EVENT_LBUTTONDOWN:
        if param["fs_btn"][0] <= x <= param["fs_btn"][2] and param["fs_btn"][1] <= y <= param["fs_btn"][3]:
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

mouse_params = {"fs_btn": (0, 0, 0, 0)}
cv2.setMouseCallback(WINDOW_NAME, on_mouse, mouse_params)

# ================= MAIN RUN LOOP =================
while True:
    now = time.time()
    elapsed = now - start_time
    is_connected, raw_frame = stream_mgr.get_frame()

    # Dynamic Layout Calculations
    TOP_BAR_H = int(CANVAS_H * 0.08)
    MARGIN = 24
    CARD_W = (CANVAS_W - (MARGIN * 3)) // 2
    CARD_H = (CANVAS_H - TOP_BAR_H - (MARGIN * 3)) // 2

    C1_X1, C1_Y1 = MARGIN, TOP_BAR_H + MARGIN
    C1_X2, C1_Y2 = C1_X1 + CARD_W, C1_Y1 + CARD_H

    C2_X1, C2_Y1 = C1_X2 + MARGIN, C1_Y1
    C2_X2, C2_Y2 = CANVAS_W - MARGIN, C1_Y2

    C3_X1, C3_Y1 = MARGIN, C1_Y2 + MARGIN
    C3_X2, C3_Y2 = C1_X1 + CARD_W, CANVAS_H - MARGIN

    C4_X1, C4_Y1 = C2_X1, C3_Y1
    C4_X2, C4_Y2 = CANVAS_W - MARGIN, CANVAS_H - MARGIN

    CAM_BOX_W = CARD_W - 28
    CAM_BOX_H = CARD_H - 58

    detected_targets = []

    # Frame Processing & Glint Extraction
    if is_connected and raw_frame is not None:
        night_feed = cv2.LUT(raw_frame, GAMMA_LUT)

        # Dual HSV Purple/IR + Brightness Threshold
        hsv = cv2.cvtColor(night_feed, cv2.COLOR_BGR2HSV)
        mask_purple = cv2.inRange(hsv, np.array([125, 35, 80]), np.array([170, 255, 255]))
        gray = cv2.cvtColor(night_feed, cv2.COLOR_BGR2GRAY)
        _, mask_bright = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_or(mask_purple, mask_bright)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if 5 <= cv2.contourArea(cnt) <= 600:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    detected_targets.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        # Exact Native Aspect Ratio Fitting
        feed_render, cam_w, cam_h, off_x, off_y = fit_exact_aspect_ratio(night_feed, CAM_BOX_W, CAM_BOX_H)
        scale_x = cam_w / float(raw_frame.shape[1])
        scale_y = cam_h / float(raw_frame.shape[0])
    else:
        placeholder = np.full((480, 640, 3), (230, 235, 242), dtype=np.uint8)
        cv2.putText(placeholder, "ESP32-CAM STREAM OFFLINE", (160, 230), cv2.FONT_HERSHEY_DUPLEX, 0.7, (130, 145, 160), 1, lineType=cv2.LINE_AA)
        cv2.putText(placeholder, "Awaiting Wi-Fi Connection...", (210, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, lineType=cv2.LINE_AA)
        feed_render, cam_w, cam_h, off_x, off_y = fit_exact_aspect_ratio(placeholder, CAM_BOX_W, CAM_BOX_H)
        scale_x, scale_y = 1.0, 1.0

    # Map cinema seats directly over the un-stretched camera viewport
    SEATING_MAP = compute_cinema_grid(cam_w, cam_h, off_x, off_y)

    def match_seat(cx, cy):
        for sid, d in SEATING_MAP.items():
            bx1, by1, bx2, by2 = d["box"]
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                return sid, False
        min_d = float('inf')
        best = "A1"
        for sid, d in SEATING_MAP.items():
            dist = np.hypot(cx - d["center"][0], cy - d["center"][1])
            if dist < min_d:
                min_d = dist
                best = sid
        return best, True

    # Register Active Target Locations
    for tx, ty in detected_targets:
        disp_x = int(tx * scale_x) + off_x
        disp_y = int(ty * scale_y) + off_y
        seat_id, is_approx = match_seat(disp_x, disp_y)

        if seat_id not in active_tracks:
            active_tracks[seat_id] = {"first": now, "last": now, "pos": (disp_x, disp_y), "approx": is_approx, "logged": False}
        else:
            active_tracks[seat_id]["last"] = now
            active_tracks[seat_id]["pos"] = (disp_x, disp_y)
            active_tracks[seat_id]["approx"] = is_approx

    # Prune dropped targets
    for sid in list(active_tracks.keys()):
        if now - active_tracks[sid]["last"] > 1.8:
            del active_tracks[sid]

    threat_history.append((elapsed, len(active_tracks)))

    # Draw Seats on Live Feed
    for sid, d in SEATING_MAP.items():
        bx1, by1, bx2, by2 = d["box"]
        draw_rounded_rect(feed_render, (bx1, by1), (bx2, by2), (200, 210, 225), radius=6, thickness=1)
        cv2.putText(feed_render, sid, (bx1 + 3, by1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 175, 190), 1, lineType=cv2.LINE_AA)

    # Draw the Circular Optical Reticle on Detected Dots
    for sid, t in active_tracks.items():
        px, py = t["pos"]
        dur = now - t["first"]
        is_alert = dur >= SHORT_TERM_THRESHOLD
        tag_label = f"Seat {sid}" if not t["approx"] else f"~{sid}"
        draw_optical_reticle(feed_render, px, py, tag_label, dur, is_alert)

    # ================= VIEW 1: FULLSCREEN CAMERA SURVEILLANCE =================
    if camera_only_mode:
        fs_source = night_feed if is_connected and raw_frame is not None else placeholder
        # Un-stretched aspect ratio container for fullscreen
        full_cam, f_w, f_h, f_ox, f_oy = fit_exact_aspect_ratio(fs_source, CANVAS_W, CANVAS_H)
        fs_grid = compute_cinema_grid(f_w, f_h, f_ox, f_oy)

        # Draw Seats
        for sid, d in fs_grid.items():
            bx1, by1, bx2, by2 = d["box"]
            col = ACCENT_RED if (sid in active_tracks and (now - active_tracks[sid]["first"]) >= SHORT_TERM_THRESHOLD) else \
                  (ACCENT_YELLOW if sid in active_tracks else (180, 195, 210))
            draw_rounded_rect(full_cam, (bx1, by1), (bx2, by2), col, radius=8, thickness=2)
            cv2.putText(full_cam, sid, (bx1 + 10, by1 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.62, col, 1, lineType=cv2.LINE_AA)

        # Draw Central Aisle
        aisle_x = f_ox + f_w // 2
        cv2.line(full_cam, (aisle_x, f_oy + int(f_h * 0.12)), (aisle_x, f_oy + int(f_h * 0.90)), (170, 185, 205), 1, lineType=cv2.LINE_AA)
        cv2.putText(full_cam, "CENTRAL AISLE", (aisle_x - 55, f_oy + int(f_h * 0.94)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 165, 185), 1, lineType=cv2.LINE_AA)

        # Draw Reticles in Fullscreen
        for sid, t in active_tracks.items():
            norm_x = (t["pos"][0] - off_x) / float(cam_w)
            norm_y = (t["pos"][1] - off_y) / float(cam_h)
            fs_px = int(f_ox + norm_x * f_w)
            fs_py = int(f_oy + norm_y * f_h)
            dur = now - t["first"]
            is_alert = dur >= SHORT_TERM_THRESHOLD
            tag_label = f"Seat {sid}" if not t["approx"] else f"~{sid}"
            draw_optical_reticle(full_cam, fs_px, fs_py, tag_label, dur, is_alert)

        # Top Floating HUD Capsule
        draw_rounded_rect(full_cam, (25, 20), (460, 85), (255, 255, 255), 12, -1)
        draw_rounded_rect(full_cam, (25, 20), (460, 85), CARD_BORDER, 12, 1)
        cv2.putText(full_cam, "R.E.F.L.E.C.T. | AUDITORIUM SCAN", (42, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)
        cv2.putText(full_cam, "Press 'C' to restore 4-card dashboard", (42, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, lineType=cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, full_cam)

    # ================= VIEW 2: 4-CARD NEOMORPHIC DASHBOARD =================
    else:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_CANVAS, dtype=np.uint8)

        # Top Bar
        draw_rounded_rect(canvas, (MARGIN, 12), (CANVAS_W - MARGIN, TOP_BAR_H), CARD_BG, 14, -1)
        draw_rounded_rect(canvas, (MARGIN, 12), (CANVAS_W - MARGIN, TOP_BAR_H), CARD_BORDER, 14, 1)

        cv2.putText(canvas, "R.E.F.L.E.C.T.", (MARGIN + 20, 38), cv2.FONT_HERSHEY_TRIPLEX, 0.82, ACCENT_BLUE, 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "Real-Time Theater Anti-Piracy Detection System", (MARGIN + 22, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, lineType=cv2.LINE_AA)

        time_str = f"SESSION TIME: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
        cv2.putText(canvas, time_str, (CANVAS_W // 2 - 120, 44), cv2.FONT_HERSHEY_DUPLEX, 0.52, TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)

        fs_btn_x1, fs_btn_y1 = CANVAS_W - MARGIN - 360, 20
        fs_btn_x2, fs_btn_y2 = fs_btn_x1 + 140, TOP_BAR_H - 10
        mouse_params["fs_btn"] = (fs_btn_x1, fs_btn_y1, fs_btn_x2, fs_btn_y2)

        draw_rounded_rect(canvas, (fs_btn_x1, fs_btn_y1), (fs_btn_x2, fs_btn_y2), (235, 242, 250), 12, -1)
        draw_rounded_rect(canvas, (fs_btn_x1, fs_btn_y1), (fs_btn_x2, fs_btn_y2), CARD_BORDER, 12, 1)
        fs_label = "WINDOWED" if is_fullscreen else "FULLSCREEN"
        cv2.putText(canvas, fs_label, (fs_btn_x1 + 22, fs_btn_y1 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.42, ACCENT_BLUE, 1, lineType=cv2.LINE_AA)

        pill_x1, pill_y1 = CANVAS_W - MARGIN - 200, 20
        pill_x2, pill_y2 = CANVAS_W - MARGIN - 20, TOP_BAR_H - 10
        p_bg = (220, 248, 225) if is_connected else (225, 225, 250)
        p_border = (130, 215, 145) if is_connected else (140, 140, 240)
        p_text = ACCENT_GREEN if is_connected else ACCENT_RED
        s_txt = "ESP32: ONLINE" if is_connected else "ESP32: OFFLINE"

        draw_rounded_rect(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), p_bg, 14, -1)
        draw_rounded_rect(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), p_border, 14, 1)
        cv2.circle(canvas, (pill_x1 + 16, (pill_y1 + pill_y2) // 2), 5, p_text, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, s_txt, (pill_x1 + 28, (pill_y1 + pill_y2) // 2 + 5), cv2.FONT_HERSHEY_DUPLEX, 0.44, p_text, 1, lineType=cv2.LINE_AA)

        # Card 1: Camera Feed
        draw_neomorphic_card(canvas, C1_X1, C1_Y1, C1_X2, C1_Y2)
        cv2.putText(canvas, "OPTICAL INFRARED SURVEILLANCE FEED (UNSTRETCHED)", (C1_X1 + 20, C1_Y1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.48, TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "[Press 'C' for Full Feed]", (C1_X2 - 180, C1_Y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, lineType=cv2.LINE_AA)
        canvas[C1_Y1 + 45:C1_Y1 + 45 + CAM_BOX_H, C1_X1 + 14:C1_X1 + 14 + CAM_BOX_W] = feed_render

        # Card 2: Seating Matrix Status
        draw_neomorphic_card(canvas, C2_X1, C2_Y1, C2_X2, C2_Y2)
        cv2.putText(canvas, "AUDITORIUM SEATING MATRIX (REAL-TIME STATUS)", (C2_X1 + 20, C2_Y1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)

        mat_x = C2_X1 + 30
        mat_y = C2_Y1 + 55
        blk_w = int((CARD_W - 60 - (5 * 14)) / 6)
        blk_h = int((CARD_H - 120 - (3 * 12)) / 4)

        for r_idx, r in enumerate(ROWS):
            for c_idx, c in enumerate(COLS):
                sid = f"{r}{c}"
                bx = mat_x + c_idx * (blk_w + 14)
                by = mat_y + r_idx * (blk_h + 12)

                if sid in active_tracks:
                    dur = now - active_tracks[sid]["first"]
                    tile_bg = ACCENT_RED if dur >= SHORT_TERM_THRESHOLD else ACCENT_YELLOW
                    txt_sub = f"{dur:.0f}s ALERT" if dur >= SHORT_TERM_THRESHOLD else f"{dur:.0f}s"
                    txt_col = (255, 255, 255)
                else:
                    tile_bg = (240, 249, 242)
                    txt_sub = "CLEAR"
                    txt_col = ACCENT_GREEN

                draw_rounded_rect(canvas, (bx, by), (bx + blk_w, by + blk_h), tile_bg, 8, -1)
                draw_rounded_rect(canvas, (bx, by), (bx + blk_w, by + blk_h), (210, 225, 220), 8, 1)
                cv2.putText(canvas, sid, (bx + 12, by + 22), cv2.FONT_HERSHEY_DUPLEX, 0.52, txt_col if sid in active_tracks else TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)
                cv2.putText(canvas, txt_sub, (bx + 10, by + blk_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, txt_col, 1, lineType=cv2.LINE_AA)

                if sid in active_tracks and active_tracks[sid]["approx"]:
                    cv2.circle(canvas, (bx + blk_w - 10, by + 10), 4, (0, 140, 255), -1, lineType=cv2.LINE_AA)

        # Legend
        leg_y = C2_Y2 - 20
        cv2.circle(canvas, (C2_X1 + 40, leg_y), 5, ACCENT_GREEN, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "Idle", (C2_X1 + 52, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (C2_X1 + 140, leg_y), 5, ACCENT_YELLOW, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "< 60s Rec", (C2_X1 + 152, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (C2_X1 + 260, leg_y), 5, ACCENT_RED, -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "> 60s Piracy Alert", (C2_X1 + 272, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (C2_X1 + 430, leg_y), 4, (0, 140, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "Aisle Offset Lock", (C2_X1 + 442, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, lineType=cv2.LINE_AA)

        # Card 3: Timeline Graph
        draw_neomorphic_card(canvas, C3_X1, C3_Y1, C3_X2, C3_Y2)
        cv2.putText(canvas, "RECORDING THREATS vs. MOVIE TIMELINE", (C3_X1 + 20, C3_Y1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)

        gx1, gy1 = C3_X1 + 50, C3_Y1 + 50
        gx2, gy2 = C3_X2 - 30, C3_Y2 - 35
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (235, 240, 248), -1)
        cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (210, 220, 230), 1)

        for l in range(5):
            y_pos = gy2 - int((l / 4.0) * (gy2 - gy1))
            cv2.line(canvas, (gx1, y_pos), (gx2, y_pos), (220, 228, 238), 1)
            cv2.putText(canvas, str(l), (gx1 - 22, y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, lineType=cv2.LINE_AA)

        if len(threat_history) > 1:
            pts = []
            for i, (t, count) in enumerate(threat_history):
                px = int(gx1 + (i / float(len(threat_history) - 1)) * (gx2 - gx1))
                py = int(gy2 - (min(count, 4) / 4.0) * (gy2 - gy1))
                pts.append((px, py))
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i+1], ACCENT_BLUE, 2, lineType=cv2.LINE_AA)

        # Card 4: Audit Log Table
        draw_neomorphic_card(canvas, C4_X1, C4_Y1, C4_X2, C4_Y2)
        cv2.putText(canvas, "INCIDENT AUDIT & LOGGING REPORT", (C4_X1 + 20, C4_Y1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)

        th_y = C4_Y1 + 50
        draw_rounded_rect(canvas, (C4_X1 + 20, th_y), (C4_X2 - 20, th_y + 28), (242, 246, 252), 6, -1)
        cv2.putText(canvas, "LOCATION", (C4_X1 + 35, th_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "SEVERITY", (C4_X1 + 180, th_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "DURATION", (C4_X1 + 330, th_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "STATUS", (C4_X1 + 480, th_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, lineType=cv2.LINE_AA)

        row_y = th_y + 55
        if not active_tracks:
            cv2.putText(canvas, "All auditorium seats clear - no active threats recorded.", (C4_X1 + 60, C4_Y1 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 165, 180), 1, lineType=cv2.LINE_AA)
        else:
            for sid, info in sorted(active_tracks.items(), key=lambda x: x[1]["first"]):
                dur = now - info["first"]
                is_crit = dur >= SHORT_TERM_THRESHOLD
                loc = f"Seat {sid}" if not info["approx"] else f"Aisle (~{sid})"
                status = "CRITICAL ALERT" if is_crit else "MONITORING"
                color = ACCENT_RED if is_crit else ACCENT_YELLOW

                if is_crit and not info["logged"]:
                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([time.strftime("%X"), loc, status, f"{dur:.1f}", info["pos"][0], info["pos"][1]])
                    info["logged"] = True

                cv2.putText(canvas, loc, (C4_X1 + 35, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.42, TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)
                cv2.putText(canvas, "HIGH" if is_crit else "LOW", (C4_X1 + 180, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, lineType=cv2.LINE_AA)
                cv2.putText(canvas, f"{dur:.1f}s", (C4_X1 + 330, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.42, TEXT_PRIMARY, 1, lineType=cv2.LINE_AA)
                cv2.putText(canvas, status, (C4_X1 + 480, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, lineType=cv2.LINE_AA)
                cv2.line(canvas, (C4_X1 + 20, row_y + 10), (C4_X2 - 20, row_y + 10), (235, 240, 248), 1)

                row_y += 36
                if row_y > C4_Y2 - 25:
                    break

        cv2.imshow(WINDOW_NAME, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') or key == ord('C'):
        camera_only_mode = not camera_only_mode
    elif key == ord('f') or key == ord('F'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

stream_mgr.running = False
cv2.destroyAllWindows()