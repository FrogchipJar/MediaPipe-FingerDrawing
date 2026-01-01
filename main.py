import cv2
import numpy as np
import mediapipe as mp
import os
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def create_color_wheel(diameter):
    """
    Generates HSV color wheel.
    
    Concept: Creates a grid of coordinates, converts them to polar (angle/radius),
    maps angle to Hue and radius to Saturation, then converts to BGR.
    """
    x = np.linspace(-1, 1, diameter)
    y = np.linspace(-1, 1, diameter)
    X, Y = np.meshgrid(x, y)
    
    # Convert Cartesian (x,y) to Polar (radius, angle)
    rho = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    
    # Map Angle (-pi to pi) to Hue (0-179 in OpenCV)
    hue = ((phi + np.pi) / (2 * np.pi)) * 179
    # Map Radius (0 to 1) to Saturation (0-255)
    sat = np.clip(rho * 255, 0, 255)
    val = np.ones_like(hue) * 255 # Constant brightness (Value)
    
    # Merge channels and convert HSV -> BGR for OpenCV
    hsv = cv2.merge((hue.astype(np.uint8), sat.astype(np.uint8), val.astype(np.uint8)))
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Create a circular Alpha Mask (make corners transparent)
    mask = (rho <= 1).astype(np.uint8) * 255
    return cv2.merge((bgr, mask)) # Returns BGRA image

def overlay_image(background, overlay, x, y):
    """
    Overlays a transparent PNG (the wheel) onto the background frame.
    Uses Alpha Blending: Result = (Overlay * Alpha) + (Background * (1 - Alpha))
    """
    h, w = overlay.shape[:2]
    
    # Safety Check: Ensure overlay fits within frame boundaries
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background
    
    # Region of Interest (ROI) where the image will be placed
    roi = background[y:y+h, x:x+w]
    
    # Separate Color and Alpha channels
    overlay_img = overlay[:,:,:3]
    overlay_mask = overlay[:,:,3] / 255.0 # Normalize 0-255 to 0.0-1.0
    
    # Blend each color channel
    for c in range(0, 3):
        roi[:,:,c] = (overlay_mask * overlay_img[:,:,c] + 
                      (1.0 - overlay_mask) * roi[:,:,c])
    
    background[y:y+h, x:x+w] = roi
    return background

def draw_hand_on_frame(image, hand_landmarks):
    """Draws the 21-point skeleton of the hand."""
    h, w, _ = image.shape
    # Draw connections (bones)
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        p1 = hand_landmarks[p1_idx]
        p2 = hand_landmarks[p2_idx]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        cv2.line(image, (x1, y1), (x2, y2), (220, 220, 220), 2)
    
    # Draw keypoints (joints)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (50, 50, 255), -1)

# ==========================================
# 2. CONFIGURATION & SETUP
# ==========================================

# MediaPipe Setup
model_path = 'hand_landmarker.task' 
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2, # Detect both Palette (Right) and Brush (Left)
    min_hand_detection_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# 
# Manual list of how hand points connect (Wrist=0, Thumb=1-4, Index=5-8, etc.)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Generate assets once at startup
WHEEL_DIAMETER = 250 
color_wheel_img = create_color_wheel(WHEEL_DIAMETER)

# App State Variables
cap = cv2.VideoCapture(0) # Camera Index
canvas = None
recent_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # Initial history
current_color = (0, 0, 0) # Default color (Black)
is_picking_color = False 

# Brush Settings
brush_thickness = 7
min_brush_size = 2
max_brush_size = 50
eraser_thickness = 40

prev_x, prev_y = 0, 0
save_counter = 1

# ==========================================
# 3. MAIN LOOP
# ==========================================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1) # Mirror view for natural interaction
    h, w, c = frame.shape

    # Initialize canvas on first frame
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # **Double Buffering Trick**: 
    # Copy the canvas so we can draw the "Ghost Cursor" without saving it permanently.
    canvas_display = canvas.copy()

    # Convert for MediaPipe (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    
    # Run Detection
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    brush_hand = None
    palette_hand = None

    # Parse Hands
    if detection_result.hand_landmarks:
        for i, landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[i][0].category_name
            draw_hand_on_frame(frame, landmarks)
            
            # Assign Roles: Right Hand = Palette, Left Hand = Brush
            if handedness == "Right":    
                palette_hand = landmarks
            elif handedness == "Left":   
                brush_hand = landmarks

    # -------------------------------------
    # PALETTE LOGIC (RIGHT HAND)
    # -------------------------------------
    wheel_center = None
    color_zones = [] # Stores clickable buttons (x, y, color)
    slider_zone = None 

    if palette_hand:
        # Calculate Anchor Point: Center of Palm
        wrist = palette_hand[0]
        idx_mcp = palette_hand[5] # Index knuckle
        cx = int((wrist.x + idx_mcp.x) / 2 * w)
        cy = int((wrist.y + idx_mcp.y) / 2 * h)

        # UI Positioning: Offset everything 60px below palm center
        base_y_offset = 60 
        if cy + base_y_offset + 50 > h: base_y_offset = h - cy - 50 # Keep on screen

        # 1. Size Slider (Vertical Bar)
        slider_x = cx - 140
        slider_y = cy + base_y_offset - 30 
        slider_h = 100
        slider_w = 30
        
        # Draw Slider UI
        cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (255, 255, 255), 2)
        
        # Fill ratio based on current brush size
        fill_ratio = (brush_thickness - min_brush_size) / (max_brush_size - min_brush_size)
        fill_h = int(fill_ratio * slider_h)
        cv2.rectangle(frame, (slider_x, slider_y + slider_h - fill_h), (slider_x + slider_w, slider_y + slider_h), (200, 200, 200), -1)
        
        # Save slider zone for interaction
        slider_zone = {'x': slider_x, 'y': slider_y, 'w': slider_w, 'h': slider_h}

        # 2. Eraser Button
        eraser_x, eraser_y = cx - 80, cy + base_y_offset
        cv2.rectangle(frame, (eraser_x, eraser_y), (eraser_x + 40, eraser_y + 40), (0, 0, 0), -1) # Black box
        cv2.putText(frame, "E", (eraser_x + 10, eraser_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        color_zones.append({'color': (0, 0, 0), 'cx': eraser_x + 20, 'cy': eraser_y + 20})

        # 3. History Slots (Last 3 Colors)
        for i, color in enumerate(recent_colors):
            slot_x = cx - 20 + (i * 50)
            slot_y = cy + base_y_offset
            cv2.rectangle(frame, (slot_x, slot_y), (slot_x + 40, slot_y + 40), color, -1)
            cv2.rectangle(frame, (slot_x, slot_y), (slot_x + 40, slot_y + 40), (255, 255, 255), 2)
            color_zones.append({'color': color, 'cx': slot_x + 20, 'cy': slot_y + 20})

        # 4. Color Wheel (Pop-up on Open Palm)
        # Check if fingers are extended (Tip higher than PIP joint)
        fingers_open = True
        for tip, pip in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            if palette_hand[tip].y > palette_hand[pip].y: # Y increases downwards
                fingers_open = False
                break
        
        if fingers_open:
            wheel_x = cx - WHEEL_DIAMETER // 2
            wheel_y = cy - WHEEL_DIAMETER // 2 - 150 # Floating above fingers
            frame = overlay_image(frame, color_wheel_img, wheel_x, wheel_y)
            wheel_center = (wheel_x + WHEEL_DIAMETER // 2, wheel_y + WHEEL_DIAMETER // 2)

    # -------------------------------------
    # BRUSH LOGIC (LEFT HAND)
    # -------------------------------------
    if brush_hand:
        # Index Tip Coordinates
        index_tip = brush_hand[8]
        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        
        # --- Smart Gesture Check ---
        # Instead of strict Up/Down, check if finger is Extended vs Folded
        # by comparing distance from Wrist (0) to Tip (8) vs Knuckle (6)
        wrist = brush_hand[0]
        
        def is_finger_extended(tip_idx, pip_idx):
            dist_tip = (brush_hand[tip_idx].x - wrist.x)**2 + (brush_hand[tip_idx].y - wrist.y)**2
            dist_pip = (brush_hand[pip_idx].x - wrist.x)**2 + (brush_hand[pip_idx].y - wrist.y)**2
            return dist_tip > dist_pip * 1.5 # Tip must be significantly further

        index_extended = is_finger_extended(8, 6)
        middle_extended = is_finger_extended(12, 10)
        ring_extended = is_finger_extended(16, 14)
        pinky_extended = is_finger_extended(20, 18)

        # Drawing = Index is Extended, all others are Folded
        is_drawing_gesture = index_extended and (not middle_extended) and (not ring_extended) and (not pinky_extended)

        touching_ui = False
        
        # A. Check Slider Interaction
        if slider_zone:
            sx, sy, sw, sh = slider_zone['x'], slider_zone['y'], slider_zone['w'], slider_zone['h']
            if sx - 20 < ix < sx + sw + 20 and sy - 20 < iy < sy + sh + 20:
                touching_ui = True
                # Map finger Y position to Size (0.0 to 1.0)
                ratio = (iy - sy) / sh
                ratio = max(0.0, min(1.0, ratio)) 
                brush_thickness = int(min_brush_size + ratio * (max_brush_size - min_brush_size))
                # Visual feedback
                cv2.circle(frame, (ix, iy), brush_thickness // 2, (150, 150, 150), 2)

        # B. Check Wheel Interaction (Math.Hypot for circular distance)
        if wheel_center and not touching_ui:
            wcx, wcy = wheel_center
            dist = math.hypot(ix - wcx, iy - wcy)
            if dist < (WHEEL_DIAMETER / 2):
                touching_ui = True
                is_picking_color = True
                # Calculate pixel coordinate relative to wheel image
                local_x = int(ix - (wcx - WHEEL_DIAMETER // 2))
                local_y = int(iy - (wcy - WHEEL_DIAMETER // 2))
                # Clamp to image bounds
                local_x = max(0, min(local_x, WHEEL_DIAMETER - 1))
                local_y = max(0, min(local_y, WHEEL_DIAMETER - 1))
                # Sample color
                pixel = color_wheel_img[local_y, local_x]
                if pixel[3] > 0: # Ignore transparent pixels
                    current_color = (int(pixel[0]), int(pixel[1]), int(pixel[2]))

        # C. Check Button Interaction (Eraser + History)
        if not touching_ui:
            for zone in color_zones:
                dist = math.hypot(ix - zone['cx'], iy - zone['cy'])
                if dist < 25:
                    current_color = zone['color']
                    touching_ui = True
                    is_picking_color = False 

        # D. Update History (When finger leaves picker)
        if is_picking_color and not touching_ui:
            is_picking_color = False
            # Push to history if it's a new color and not the eraser
            if current_color != recent_colors[0] and current_color != (0,0,0):
                recent_colors.insert(0, current_color)
                recent_colors.pop() 

        # E. Drawing Action
        if is_drawing_gesture and not touching_ui:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = ix, iy
            
            # Use separate thickness for eraser vs brush
            thickness = eraser_thickness if current_color == (0,0,0) else brush_thickness
            cv2.line(canvas, (prev_x, prev_y), (ix, iy), current_color, thickness)
            prev_x, prev_y = ix, iy
        else:
            prev_x, prev_y = 0, 0
        
        # --- Cursor Visualization (Ghost vs Solid) ---
        cursor_radius = brush_thickness // 2
        display_color = current_color
        outline_color = (255, 255, 255)
        
        if is_drawing_gesture:
            # Drawing Mode: Solid Circle
            cv2.circle(frame, (ix, iy), cursor_radius, display_color, -1)
            cv2.circle(frame, (ix, iy), cursor_radius, outline_color, 1)
            cv2.circle(canvas_display, (ix, iy), cursor_radius, display_color, -1)
        else:
            # Hover Mode: Hollow Ring (Targeting)
            cv2.circle(frame, (ix, iy), cursor_radius, display_color, 2)
            cv2.circle(frame, (ix, iy), cursor_radius + 2, outline_color, 1)
            cv2.circle(canvas_display, (ix, iy), cursor_radius, display_color, 2)
            cv2.circle(canvas_display, (ix, iy), cursor_radius + 2, outline_color, 1)

    else:
        prev_x, prev_y = 0, 0

    # -------------------------------------
    # HUD (HEADS UP DISPLAY)
    # -------------------------------------
    # Top-right box showing Active Color & Size
    cv2.rectangle(frame, (w - 120, 20), (w - 20, 120), (50, 50, 50), -1)
    cv2.rectangle(frame, (w - 120, 20), (w - 20, 120), (255, 255, 255), 2)
    
    preview_radius = brush_thickness // 2
    px, py = w - 70, 70
    
    if current_color == (0, 0, 0):
        cv2.circle(frame, (px, py), preview_radius, (255, 255, 255), 2)
        cv2.putText(frame, "Eraser", (w - 105, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.circle(frame, (px, py), preview_radius, current_color, -1)
        cv2.circle(frame, (px, py), preview_radius, (255, 255, 255), 1)
        cv2.putText(frame, "Color", (w - 95, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Combine video feed with the display canvas
    combined = np.hstack((frame, canvas_display))
    cv2.imshow("CV Finger Paint", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    elif key == ord("s"):
        path = os.path.join(os.getcwd(), f'art_{save_counter}.png')
        cv2.imwrite(path, canvas) # Save the clean canvas (no ghost cursor)
        print(f"Saved: {path}")
        save_counter += 1

cap.release()
cv2.destroyAllWindows()