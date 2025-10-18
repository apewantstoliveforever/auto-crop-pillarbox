import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
from moviepy.editor import VideoFileClip

# === CONFIG ===
MIN_WIDTH = 64  # fallback if crop region too small

# === INPUT FOLDER ===
input_folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
input_folder = os.path.abspath(input_folder)
if not os.path.exists(input_folder):
    print(f"❌ Input folder does not exist: {input_folder}")
    sys.exit(1)

print(f"📂 Input folder: {input_folder}")

# === DETECT CONTENT REGION ===
def detect_content_region(cap, sample_frames=50):
    def compute_crop(threshold):
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, total // sample_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev = cap.read()
        if not ret:
            return 0, w, w, 1.0  # (left, right, crop_width, dark_ratio)

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        motion_map = np.zeros((h, w), np.float32)
        sharp_map = np.zeros((h, w), np.float32)
        brightness_map = np.zeros((h, w), np.float32)

        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_map += np.abs(gray - prev_gray)
            sharp_map += np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            brightness_map += gray
            prev_gray = gray

        motion_profile = motion_map.mean(axis=0)
        sharp_profile = sharp_map.mean(axis=0)
        brightness_profile = brightness_map.mean(axis=0)

        # normalize
        motion_profile = (motion_profile - motion_profile.min()) / (motion_profile.max() - motion_profile.min() + 1e-6)
        sharp_profile = (sharp_profile - sharp_profile.min()) / (sharp_profile.max() - sharp_profile.min() + 1e-6)

        combined = motion_profile * sharp_profile
        xs = np.where(combined > combined.max() * threshold)[0]
        if len(xs) == 0:
            return 0, w, w, 1.0

        left, right = np.min(xs), np.max(xs)

        # refine by brightness
        bright_mask = brightness_profile[left:right] > 20
        bright_cols = np.where(bright_mask)[0]
        if len(bright_cols) > 0:
            left += np.min(bright_cols)
            right = left + len(bright_cols)

        # measure dark ratio
        crop_brightness = brightness_profile[left:right]
        dark_ratio = np.mean(crop_brightness < 20) if len(crop_brightness) > 0 else 1.0

        # sanity
        if (right - left) < MIN_WIDTH:
            xs = np.where(motion_profile > motion_profile.max() * threshold)[0]
            if len(xs) == 0:
                left, right = 0, w
            else:
                left, right = np.min(xs), np.max(xs)

        left = max(0, left + 2)
        right = min(w, right - 2)
        if left >= right:
            left, right = 0, w

        removed_ratio = (w - (right - left)) / w
        print(f"   └─ Pre-crop check (threshold={threshold:.2f}): removed={removed_ratio:.2%}")
        return left, right, right - left, dark_ratio

    # --- First pass (threshold = 0.05)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    x1, x2, crop_w, dark_ratio = compute_crop(0.05)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ratio = crop_w / w
    removed = 1 - ratio
    print(f"🧪 Threshold 0.05: keep={ratio:.2f}, removed={removed:.2f}, dark={dark_ratio:.2f}")

    # Nếu threshold 0.05 mà phần bị loại bỏ nhiều -> thử lại với 0.2
    if removed > 0.1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        x1b, x2b, crop_wb, dark_ratio2 = compute_crop(0.2)
        ratio2 = crop_wb / w
        removed2 = 1 - ratio2
        print(f"🔁 Threshold 0.2: keep={ratio2:.2f}, removed={removed2:.2f}, dark={dark_ratio2:.2f}")

        # Nếu threshold 0.2 vẫn sáng, không quá hẹp, và không bị cắt quá 75% -> dùng 0.2
        if dark_ratio2 < 0.5 and crop_wb > MIN_WIDTH and removed2 < 0.75:
            print("✅ Using threshold 0.2 (cleaner crop)")
            return x1b, x2b
        else:
            print("⚙️  Threshold 0.2 too aggressive, reverting to 0.05")

    print("✅ Using threshold 0.05 (initial crop)")
    if crop_w <= 0:
        x1, x2 = 0, w
    return x1, x2


# === CROP VIDEO TO TEMP FILE ===
def crop_video_to_temp(input_path, temp_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n🎬 Processing {os.path.basename(input_path)} ...")
    x1, x2 = detect_content_region(cap)
    crop_w = x2 - x1
    print(f"📦 Final content region: x={x1}, width={crop_w}")

    if crop_w <= 0:
        print("⚠️  Invalid crop width, skipping video")
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (crop_w, h))

    for _ in tqdm(range(total), desc="Cropping video"):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame[:, x1:x2])

    cap.release()
    out.release()
    return True


# === MERGE AUDIO USING MOVIEPY ===
def merge_audio_with_moviepy(temp_video_path, original_path, final_output_path):
    print("🎧 Merging audio using MoviePy ...")
    try:
        orig = VideoFileClip(original_path)
        crop = VideoFileClip(temp_video_path)
        final = crop.set_audio(orig.audio)
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        final.write_videofile(final_output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        crop.close()
        orig.close()
        final.close()
        os.remove(temp_video_path)
        print(f"✅ Saved final video: {final_output_path}")
    except Exception as e:
        print(f"❌ MoviePy merge failed: {e}")


# === MAIN RUN (recursive) ===
for root, dirs, files in os.walk(input_folder):
    # Skip 'cropped' folders
    if os.path.basename(root).lower() == "cropped":
        continue

    if not files:
        continue

    for f in files:
        if f.lower().endswith(".mp4"):
            full_path = os.path.join(root, f)
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"❌ Cannot open {full_path}")
                continue

            x1, x2 = detect_content_region(cap)
            crop_w = x2 - x1
            cap.release()

            if crop_w <= 0:
                print(f"⚠️  No valid crop region, skipping {full_path}")
                continue

            cropped_folder = os.path.join(root, "cropped")
            os.makedirs(cropped_folder, exist_ok=True)

            temp = os.path.join(cropped_folder, "temp_crop.mp4")
            final_output_path = os.path.join(cropped_folder, f)

            try:
                success = crop_video_to_temp(full_path, temp)
                if success:
                    merge_audio_with_moviepy(temp, full_path, final_output_path)
                else:
                    print(f"❌ Skipped {full_path}")
            except Exception as e:
                print(f"❌ Failed {full_path}: {e}")
