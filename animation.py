#./.venv/bin/python animation.py

import cv2
import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Generate the ground truth 3D points
CHECKERBOARD = (6, 9)
SQUARE_SIZE = 50

height = (CHECKERBOARD[1] + 2) * SQUARE_SIZE
width = (CHECKERBOARD[0] + 2) * SQUARE_SIZE
margin = 200
sim_width = width + 2 * margin
sim_height = height + 2 * margin

# Extended Dashboard Dimensions for better readability
UI_WIDTH = 600 
TOTAL_WIDTH = sim_width + UI_WIDTH

# Simulation Configuration (can be overridden by CLI)
GLOBAL_NOISE = 4.0
GLOBAL_FOCAL = 1200.0
GLOBAL_SPEED = 1.0

# Generate a randomized order for the 70 mini-map squares to appear
total_minimap_boxes = (CHECKERBOARD[0] + 1) * (CHECKERBOARD[1] + 1)
RANDOM_REVEAL_ORDER = np.random.permutation(total_minimap_boxes)

# Create the base canonical image
base_img = np.full((height, width, 3), 200, dtype=np.uint8)
cv2.rectangle(base_img, (SQUARE_SIZE, SQUARE_SIZE), (width-SQUARE_SIZE, height-SQUARE_SIZE), (255, 255, 255), -1)

for y in range(CHECKERBOARD[1] + 1):
    for x in range(CHECKERBOARD[0] + 1):
        if (x + y) % 2 == 1:
            px = x * SQUARE_SIZE + int(SQUARE_SIZE * 0.5)
            py = y * SQUARE_SIZE + int(SQUARE_SIZE * 0.5)
            cv2.rectangle(base_img, (px, py), (px + SQUARE_SIZE, py + SQUARE_SIZE), (30, 30, 30), -1)

base_img = cv2.GaussianBlur(base_img, (3, 3), 0)

# INPUT 1: The 3D Object Points (Real world)
objectp3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

def generate_and_detect_frame(i):
    """Worker function optimized for Apple Silicon Multi-Core"""
    # 60 FPS Realistic Handheld 3D Motion (Pitch, Yaw, Roll)
    cam_mtx = np.array([[GLOBAL_FOCAL, 0, sim_width/2], [0, GLOBAL_FOCAL, sim_height/2], [0, 0, 1]], dtype=np.float32)
    dist_coefs = np.zeros(5, dtype=np.float32)
    
    pts3d = np.float32([
        [-width/2, -height/2, 0],
        [width/2, -height/2, 0],
        [-width/2, height/2, 0],
        [width/2, height/2, 0]
    ])
    # Apply Speed multiplier to timeline
    i_mod = i * GLOBAL_SPEED
    
    rvec = np.array([
        0.5 * np.sin(i_mod * 0.01),   
        0.5 * np.cos(i_mod * 0.0125),   
        0.2 * np.sin(i_mod * 0.0075)    
    ], dtype=np.float32)
    
    # 60 FPS 3D Translation (X drift, Y drift, Z zoom)
    tvec = np.array([
        100 * np.sin(i_mod * 0.015),   
        50 * np.cos(i_mod * 0.0175),    
        1400 + 400 * np.sin(i_mod * 0.005) 
    ], dtype=np.float32)
    
    pts2, _ = cv2.projectPoints(pts3d, rvec, tvec, cam_mtx, dist_coefs)
    pts2 = pts2.reshape(4, 2)
    
    pts1 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    frame = cv2.warpPerspective(base_img, M, (sim_width, sim_height), borderValue=(50,50,50))
    
    noise = np.random.normal(0, GLOBAL_NOISE, frame.shape) 
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags=cv2.CALIB_CB_FAST_CHECK)
    
    corners2 = None
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
    return i, frame, ret, corners2

def draw_cyber_ui(display_frame):
    """Draws the structural lines for the dashboard"""
    cv2.rectangle(display_frame, (sim_width, 0), (TOTAL_WIDTH, sim_height), (20, 20, 25), -1)
    cv2.line(display_frame, (sim_width, 0), (sim_width, sim_height), (100, 255, 100), 2)
    cv2.putText(display_frame, "SYSTEM INPUT DATA", (sim_width + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
    cv2.putText(display_frame, f"CONFIG: NOISE={GLOBAL_NOISE} | FOCAL={GLOBAL_FOCAL} | SPEED={GLOBAL_SPEED}x", (sim_width + 30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(display_frame, "MAPPING 3D WORLD TO 2D PIXELS", (sim_width + 30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.line(display_frame, (sim_width + 30, 115), (TOTAL_WIDTH - 30, 115), (100, 255, 100), 1)

def draw_digital_twin(canvas, snapshots_collected, target_snapshots, offset_x, offset_y):
    """Draws the 3D ground truth model tied to ACTUAL data collection in random order"""
    map_size = 200
    cv2.rectangle(canvas, (offset_x-15, offset_y-30), (offset_x + map_size + 15, offset_y + map_size + 15), (25, 25, 30), -1)
    cv2.putText(canvas, "INTERNAL 2D PLANE (Z=0)", (offset_x, offset_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cols = CHECKERBOARD[0] + 1
    rows = CHECKERBOARD[1] + 1
    
    # Calculate how many boxes to show based on REAL collected data
    progress_ratio = min(snapshots_collected / target_snapshots, 1.0)
    boxes_to_reveal = int(progress_ratio * total_minimap_boxes)
    
    box_w = map_size // cols
    box_h = map_size // rows
    
    # Draw the randomly revealed boxes
    for i in range(boxes_to_reveal):
        # Pick the random box index from our global list
        actual_box_index = RANDOM_REVEAL_ORDER[i]
        
        # Convert 1D index back to 2D grid coordinates
        bx = actual_box_index % cols
        by = actual_box_index // cols
        
        px = offset_x + bx * box_w
        py = offset_y + by * box_h
        
        is_black = (bx + by) % 2 == 1
        color = (50, 50, 50) if is_black else (200, 200, 200)
        
        # Add a green flash to the most recently spawned boxes to look like active scanning
        if i >= boxes_to_reveal - 3 and progress_ratio < 1.0:
            color = (0, 255, 0)
            
        cv2.rectangle(canvas, (px, py), (px + box_w, py + box_h), color, -1)
        
        # Render internal nodes (the actual 3D tracking points)
        if bx < cols - 1 and by < rows - 1:
            cv2.circle(canvas, (px + box_w, py + box_h), 2, (0, 255, 0), -1)

def create_calibration_animation():
    threedpoints = []
    twodpoints = []
    center_trail = []
    live_data_log = []

    m4_cores = os.cpu_count() or 8
    print(f"Unleashing Apple Processor: Pre-rendering across {m4_cores} CPU cores...")
    num_frames = 600 
    target_snapshots = num_frames // 20
    
    frames_data = [None] * num_frames
    with ThreadPoolExecutor(max_workers=m4_cores) as executor:
        futures = {executor.submit(generate_and_detect_frame, i): i for i in range(num_frames)}
        completed = 0
        for future in as_completed(futures):
            i = futures[future]
            frames_data[i] = future.result()
            completed += 1
            progress = int((completed / num_frames) * 40)
            sys.stdout.write(f"\rGenerating Frames: [{'=' * progress}{' ' * (40 - progress)}] {completed}/{num_frames}")
            sys.stdout.flush()
    print()
    
    print("Starting animation playback...")
    
    window_name = "M4 Calibration Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(1)
    
    for (i, frame, ret, corners2) in frames_data:
        canvas = np.zeros((sim_height, TOTAL_WIDTH, 3), dtype=np.uint8)
        canvas[:, :sim_width] = frame
        
        draw_cyber_ui(canvas)
        
        # Draw Camera Viewfinder
        cv2.rectangle(canvas, (20, 20), (sim_width-20, sim_height-20), (255, 255, 255), 2)
        cv2.line(canvas, (sim_width//2 - 20, sim_height//2), (sim_width//2 + 20, sim_height//2), (0, 255, 0), 1)
        cv2.line(canvas, (sim_width//2, sim_height//2 - 20), (sim_width//2, sim_height//2 + 20), (0, 255, 0), 1)

        if ret:
            cv2.drawChessboardCorners(canvas[:, :sim_width], CHECKERBOARD, corners2, ret)
            
            center_pt = tuple(map(int, corners2[27][0])) 
            center_trail.append(center_pt)
            if len(center_trail) > 1:
                for t in range(1, len(center_trail)):
                    cv2.line(canvas, center_trail[t-1], center_trail[t], (255, 0, 255), 2)
            
            if i % 20 == 0:
                threedpoints.append(objectp3d)
                twodpoints.append(corners2)
                
                rand_idx = np.random.randint(0, len(objectp3d))
                pt_3d = objectp3d[rand_idx]
                pt_2d = corners2[rand_idx][0]
                
                log_str = f"3D [{pt_3d[0]:3.0f}, {pt_3d[1]:3.0f}, 0]  --->  2D [{pt_2d[0]:5.1f}, {pt_2d[1]:5.1f}]"
                live_data_log.insert(0, log_str)
                if len(live_data_log) > 12:
                    live_data_log.pop()
                    
        # Render Telemetry Log
        y_offset = 150
        for idx, log_text in enumerate(live_data_log):
            intensity = max(255 - (idx * 20), 80)
            if idx == 0:
                cv2.putText(canvas, log_text, (sim_width + 20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)
            else:
                cv2.putText(canvas, log_text, (sim_width + 20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, intensity, intensity), 1)
            y_offset += 35
            
        # REAL-TIME UPDATE: Pass len(threedpoints) instead of timeline ratio
        draw_digital_twin(canvas, len(threedpoints), target_snapshots, sim_width + 360, 180)
            
        cv2.imshow(window_name, canvas)
        
        # 16ms delay = 60 FPS
        if cv2.waitKey(16) & 0xFF == 27:
            return 

    # ==========================================
    # PRESENTER PAUSE (Wait for 'E' Key)
    # ==========================================
    # Force the Digital Twin to 100% complete at the pause screen
    draw_digital_twin(canvas, target_snapshots, target_snapshots, sim_width + 360, 180)
    
    cv2.putText(canvas, "DATA COLLECTION COMPLETE", (sim_width + 30, sim_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    cv2.putText(canvas, "Press 'E' to Extract Matrix...", (sim_width + 30, sim_height - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 2)
    cv2.imshow(window_name, canvas)
    
    print("\nWaiting for Presenter... Press 'E' to continue or 'ESC' to exit.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('e') or key == ord('E'):
            break
        elif key == 27: 
            cv2.destroyAllWindows()
            return

    # ==========================================
    # PHASE 2: OUTPUT ANALYSIS DASHBOARD
    # ==========================================
    if len(threedpoints) > 0:
        canvas.fill(15)
        cv2.putText(canvas, "PROCESSING MATRICES...", (TOTAL_WIDTH//2 - 200, sim_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(800)
            
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
            threedpoints, twodpoints, (sim_width, sim_height), None, None, flags=flags)
            
        canvas.fill(20)
        
        cv2.putText(canvas, "THE INPUTS", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.line(canvas, (50, 80), (450, 80), (100, 100, 100), 2)
        cv2.putText(canvas, f"Total Grid Snapshots: {len(threedpoints)}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(canvas, f"Total 3D Points: {len(threedpoints) * 54}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(canvas, f"Total 2D Pixels Found: {len(twodpoints) * 54}", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
        # Draw the final 3D Plane Model into the Input column
        draw_digital_twin(canvas, target_snapshots, target_snapshots, 100, 320)
        
        right_col = 550
        cv2.putText(canvas, "THE OUTPUTS", (right_col, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)
        cv2.line(canvas, (right_col, 80), (TOTAL_WIDTH - 50, 80), (100, 255, 100), 2)
        
        cv2.putText(canvas, "Intrinsic Camera Matrix (K):", (right_col, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        for r in range(3):
            row_str = f"[{matrix[r][0]:8.2f},  {matrix[r][1]:8.2f},  {matrix[r][2]:8.2f}]"
            cv2.putText(canvas, row_str, (right_col + 20, 190 + r * 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(window_name, canvas)
            cv2.waitKey(300)
            
        cv2.putText(canvas, "Distortion Coefficients:", (right_col, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        dist_str = f"k1: {distortion[0][0]:.3f} | k2: {distortion[0][1]:.3f} | p1: {distortion[0][2]:.3f}"
        cv2.putText(canvas, dist_str, (right_col + 20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(400)

        mean_error = 0
        for i in range(len(threedpoints)):
            imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
            error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        total_error = mean_error / len(threedpoints)

        cv2.putText(canvas, "System Accuracy (Reprojection Error):", (right_col, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(canvas, f"{total_error:.4f} Pixels Off", (right_col + 20, 560), cv2.FONT_HERSHEY_DUPLEX, 1.2, (100, 255, 100), 2)

        cv2.putText(canvas, "Press any key to finish.", (50, sim_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="M4 Camera Calibration Simulation")
    parser.add_argument("--noise", type=float, default=4.0, help="Camera sensor noise level (default: 4.0)")
    parser.add_argument("--focal", type=float, default=1200.0, help="Simulated intrinsic focal length (default: 1200.0)")
    parser.add_argument("--speed", type=float, default=1.0, help="Camera movement speed multiplier (default: 1.0)")
    args = parser.parse_args()
    
    # Overwrite configuration with user inputs
    GLOBAL_NOISE = args.noise
    GLOBAL_FOCAL = args.focal
    GLOBAL_SPEED = args.speed

    create_calibration_animation()