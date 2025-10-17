import cv2
import numpy as np
import os
from tqdm import tqdm

# === Thư mục nguồn & đích ===
input_folder = "input_videos"
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)


def detect_real_content_region(cap, sample_frames=50):
    """Kết hợp motion + sharpness + adaptive trim để xác định vùng nội dung thật"""
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, total // sample_frames)

        ret, prev = cap.read()
        if not ret:
            return 0, w

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        motion_map = np.zeros((h, w), np.float32)
        sharp_map = np.zeros((h, w), np.float32)

        for i in range(1, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            motion_map += diff.astype(np.float32)
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            sharp_map += np.abs(lap)
            prev_gray = gray

        motion_profile = motion_map.mean(axis=0)
        sharp_profile = sharp_map.mean(axis=0)
        motion_profile = (motion_profile - motion_profile.min()) / (motion_profile.max() - motion_profile.min() + 1e-6)
        sharp_profile = (sharp_profile - sharp_profile.min()) / (sharp_profile.max() - sharp_profile.min() + 1e-6)

        combined = motion_profile * sharp_profile
        combined = cv2.GaussianBlur(combined.reshape(-1, 1), (51, 1), 0).flatten()

        th = combined.max() * 0.35
        mask = combined > th
        xs = np.where(mask)[0]
        if len(xs) == 0:
            return 0, w

        left, right = np.min(xs), np.max(xs)
        margin = int((right - left) * 0.03)
        left = max(0, left - margin)
        right = min(w, right + margin)

        grad = np.gradient(sharp_profile)
        grad = np.abs(grad) / (grad.max() + 1e-6)
        window = 40
        left_flat = np.mean(sharp_profile[:window]) < 0.2 and np.mean(grad[:window]) < 0.05
        right_flat = np.mean(sharp_profile[-window:]) < 0.2 and np.mean(grad[-window:]) < 0.05
        if left_flat:
            left += int(window * 0.8)
        if right_flat:
            right -= int(window * 0.8)
        left = max(0, left)
        right = min(w, right)

        return left, right
    except Exception as e:
        print(f"❌ Error in detect_real_content_region: {e}")
        return 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


def crop_video(input_path, output_path):
    """Cắt video theo vùng nội dung thực tế."""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Cannot open {input_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n🎬 Processing {os.path.basename(input_path)} ...")

        x1, x2 = detect_real_content_region(cap)
        crop_w = x2 - x1
        if crop_w <= 0:
            print(f"❌ Invalid crop width for {input_path}")
            cap.release()
            return False

        print(f"📦 Final content region: x={x1}, width={crop_w}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, h))

        for _ in tqdm(range(total), desc="Cropping video"):
            ret, frame = cap.read()
            if not ret:
                break
            cropped = frame[:, x1:x2]
            out.write(cropped)

        cap.release()
        out.release()
        print(f"✅ Saved: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Failed to crop {input_path}: {e}")
        return False


# === MAIN RUN ===
for f in os.listdir(input_folder):
    if not f.lower().endswith(".mp4"):
        continue
    inp = os.path.join(input_folder, f)
    out = os.path.join(output_folder, f"cropped_{f}")
    success = crop_video(inp, out)
    if not success:
        print(f"⚠️ Video not output: {inp}")
