import cv2
import numpy as np
from tqdm import tqdm
import os

input_folder = "input_videos"
output_folder = "output_videos"

os.makedirs(output_folder, exist_ok=True)
video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]

for video_file in video_files:
    input_path = os.path.join(input_folder, video_file)

    print(f"\n🎬 Processing {video_file} ...")

    # Lặp lại 2 lần cho mỗi video
    for pass_num in range(1, 3):
        print(f"\n=== 🔁 PASS {pass_num}/2 ===")

        output_path = os.path.join(
            output_folder, f"cropped_pass{pass_num}_{video_file}"
        )

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Cannot open {video_file}, skipping...")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        analyze_frames = min(total_frames, int(fps * 5))  # 5s đầu để phân tích

        print(f"▶️ Step 1: Detecting motion from first {analyze_frames} frames...")

        # --- STEP 1: Motion detection heatmap ---
        motion_map = np.zeros((h, w), np.float32)
        ret, prev_frame = cap.read()
        if not ret:
            print("❌ Cannot read first frame, skipping...")
            continue
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        for _ in tqdm(range(1, analyze_frames), desc="Building motion map"):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            _, thresh = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
            motion_map += thresh
            prev_gray = gray

        motion_map = cv2.GaussianBlur(motion_map, (25, 25), 0)
        motion_map = (motion_map - motion_map.min()) / (motion_map.max() - motion_map.min() + 1e-6)

        # Xác định vùng có chuyển động mạnh
        motion_mask = (motion_map > 0.2).astype(np.uint8)
        ys, xs = np.where(motion_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            print("⚠️ No significant motion detected, using full frame.")
            x_left, x_right, y_top, y_bottom = 0, w, 0, h
        else:
            y_top, y_bottom = np.min(ys), np.max(ys)
            x_left, x_right = np.min(xs), np.max(xs)

        # Thêm margin nhẹ để tránh crop quá sát
        margin_x = int((x_right - x_left) * 0.05)
        margin_y = int((y_bottom - y_top) * 0.05)
        x_left = max(0, x_left - margin_x)
        x_right = min(w, x_right + margin_x)
        y_top = max(0, y_top - margin_y)
        y_bottom = min(h, y_bottom + margin_y)

        print(f"📦 Motion region: x={x_left}, y={y_top}, w={x_right-x_left}, h={y_bottom-y_top}")

        # --- STEP 2: Brightness & sharpness refinement ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        variance_map = np.zeros((h, w), np.float32)
        brightness_map = np.zeros((h, w), np.float32)

        print("🧠 Refining crop with brightness and sharpness...")

        for i in tqdm(range(analyze_frames), desc="Analyzing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            variance_map += lap ** 2
            brightness_map += gray

        variance_map /= max(1, analyze_frames)
        brightness_map /= max(1, analyze_frames)

        # Focus only on motion region
        v_crop = variance_map[y_top:y_bottom, x_left:x_right]
        b_crop = brightness_map[y_top:y_bottom, x_left:x_right]

        dark_mask = (b_crop < 20).astype(np.uint8)
        dark_rows = dark_mask.mean(axis=1)
        dark_cols = dark_mask.mean(axis=0)

        def find_non_black_range(profile, limit=0.90):
            mask = profile < limit
            if not mask.any():
                return 0, len(profile)
            start = np.argmax(mask)
            end = len(profile) - np.argmax(np.flip(mask))
            if start >= end:
                return 0, len(profile)
            return start, end

        y_inner_top, y_inner_bottom = find_non_black_range(dark_rows)
        x_inner_left, x_inner_right = find_non_black_range(dark_cols)

        # Adjust to global coords
        y_top += y_inner_top
        y_bottom = y_top + (y_inner_bottom - y_inner_top)
        x_left += x_inner_left
        x_right = x_left + (x_inner_right - x_inner_left)

        # --- STEP 3: Sharpness fine-tuning ---
        v_profile_y = variance_map.mean(axis=1)
        v_profile_x = variance_map.mean(axis=0)
        v_profile_y = cv2.GaussianBlur(v_profile_y.reshape(-1, 1), (15, 1), 0).flatten()
        v_profile_x = cv2.GaussianBlur(v_profile_x.reshape(-1, 1), (15, 1), 0).flatten()

        v_profile_y = (v_profile_y - v_profile_y.min()) / (v_profile_y.max() - v_profile_y.min() + 1e-6)
        v_profile_x = (v_profile_x - v_profile_x.min()) / (v_profile_x.max() - v_profile_x.min() + 1e-6)

        th = 0.15
        y_top_fine = np.argmax(v_profile_y > th)
        y_bottom_fine = h - np.argmax(np.flip(v_profile_y) > th)
        x_left_fine = np.argmax(v_profile_x > th)
        x_right_fine = w - np.argmax(np.flip(v_profile_x) > th)

        # Kết hợp tinh chỉnh nhẹ
        y_top = max(y_top, y_top_fine)
        y_bottom = min(y_bottom, y_bottom_fine)
        x_left = max(x_left, x_left_fine)
        x_right = min(x_right, x_right_fine)

        w_crop = x_right - x_left
        h_crop = y_bottom - y_top

        if w_crop <= 0 or h_crop <= 0:
            print("❌ Invalid crop range, skipping...")
            cap.release()
            continue

        print(f"✅ Final crop area: x={x_left}, y={y_top}, w={w_crop}, h={h_crop}")

        # --- STEP 4: Crop and save ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w_crop, h_crop))

        for i in tqdm(range(total_frames), desc=f"Saving cropped video (pass {pass_num})"):
            ret, frame = cap.read()
            if not ret:
                break
            cropped = frame[y_top:y_bottom, x_left:x_right]
            out.write(cropped)

        cap.release()
        out.release()
        print(f"💾 Saved cropped video: {output_path}")
