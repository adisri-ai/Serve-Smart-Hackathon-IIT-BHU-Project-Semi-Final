# ==================================================================================
# FINAL GRANDMASTER PIPELINE: ROBUST, RESUMABLE, & MEMORY-SAFE
# ==================================================================================

# --- USER CONFIGURATION ---
# UPDATE THIS PATH to exactly where your zip file is in your Google Drive
ZIP_PATH_IN_DRIVE = '/content/drive/MyDrive/military_object_dataset.zip'
# PROJECT NAME (Folder in Drive where models will be saved)
PROJECT_PATH = '/content/drive/MyDrive/Hackathon_Project/Runs'
# ---------------------------

import os
import sys
import shutil
import subprocess
import glob
import gc
import yaml
import torch
import numpy as np
from google.colab import files

print("🚀 STARTING FINAL PIPELINE...")

# ------------------------------------------------------------------
# [1/8] GPU CHECK & LIBRARY INSTALL (SELF-HEALING)
# ------------------------------------------------------------------
print("\n[1/8] Checking Environment...")
if not torch.cuda.is_available():
    print("⛔ WARNING: No GPU detected. Training will be extremely slow.")
else:
    print(f"   ✅ GPU Detected: {torch.cuda.get_device_name(0)}")

try:
    import ultralytics
    from ensemble_boxes import weighted_boxes_fusion
    print("   ✅ Libraries ready.")
except ImportError:
    print("   ⚠️ Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "ensemble-boxes", "-q"])
    import ultralytics
    from ensemble_boxes import weighted_boxes_fusion
    print("   ✅ Installation complete.")

from ultralytics import YOLO
from tqdm.notebook import tqdm

# ------------------------------------------------------------------
# [2/8] DATASET SETUP (RESTORES IF DELETED)
# ------------------------------------------------------------------
print("\n[2/8] Verifying Dataset...")
if not os.path.exists('/content/drive'):
    from google.colab import drive
    drive.mount('/content/drive')

dataset_dir = '/content/military_dataset'
# Check if dataset seems valid (has train images)
if os.path.exists(f"{dataset_dir}/military_object_dataset/train/images") or os.path.exists(f"{dataset_dir}/train/images"):
    print("   ✅ Dataset found locally.")
else:
    print(f"   🔄 Restoring dataset from {ZIP_PATH_IN_DRIVE}...")
    if not os.path.exists(ZIP_PATH_IN_DRIVE):
        raise FileNotFoundError(f"❌ Zip file not found at {ZIP_PATH_IN_DRIVE}")
    
    # Unzip
    shutil.unpack_archive(ZIP_PATH_IN_DRIVE, dataset_dir)
    print("   ✅ Dataset Unzipped.")

# Find true root path
true_root = None
for root, dirs, files_ in os.walk(dataset_dir):
    if 'train' in dirs and 'val' in dirs:
        true_root = root
        break
if not true_root: raise ValueError("❌ Could not find 'train' folder in unzipped data!")
print(f"   Dataset Root: {true_root}")

# ------------------------------------------------------------------
# [3/8] SMART "FLOOR 500" BALANCING
# ------------------------------------------------------------------
print("\n[3/8] Applying 'Floor 500' Class Balancing...")
train_img_dir = os.path.join(true_root, 'train/images')
train_lbl_dir = os.path.join(true_root, 'train/labels')

# 1. Scan current counts
class_counts = {}
all_files = [f for f in os.listdir(train_lbl_dir) if f.endswith('.txt') and "_copy_" not in f]

for lbl_file in all_files:
    with open(os.path.join(train_lbl_dir, lbl_file), 'r') as f:
        lines = f.readlines()
    seen = set()
    for line in lines:
        try:
            c = int(line.split()[0])
            seen.add(c)
        except: continue
    for c in seen:
        class_counts[c] = class_counts.get(c, 0) + 1

# 2. Boost tiny classes to ~500, leave big classes alone
files_to_clone = {k: [] for k in class_counts.keys()}
# Map files to classes
for lbl_file in all_files:
    with open(os.path.join(train_lbl_dir, lbl_file), 'r') as f:
        lines = f.readlines()
    seen = set()
    for line in lines:
        try:
            c = int(line.split()[0])
            seen.add(c)
        except: continue
    for c in seen: files_to_clone[c].append(lbl_file)

cloned_total = 0
for cls_id, count in class_counts.items():
    if count < 500:
        multiplier = int(500 / count)
        if multiplier > 1:
            print(f"   -> Boosting Class {cls_id} ({count} images) by {multiplier}x...")
            file_list = files_to_clone[cls_id]
            
            for lbl_file in file_list:
                base_name = os.path.splitext(lbl_file)[0]
                src_img = os.path.join(train_img_dir, base_name + ".jpg")
                if not os.path.exists(src_img): src_img = src_img.replace(".jpg", ".png")
                if not os.path.exists(src_img): continue
                
                src_lbl = os.path.join(train_lbl_dir, lbl_file)
                for i in range(multiplier - 1):
                    new_name = f"{base_name}_copy_{cls_id}_{i}"
                    # Check exist to prevent double-cloning on resume
                    if not os.path.exists(os.path.join(train_lbl_dir, new_name + ".txt")):
                        shutil.copy(src_lbl, os.path.join(train_lbl_dir, new_name + ".txt"))
                        shutil.copy(src_img, os.path.join(train_img_dir, new_name + os.path.splitext(src_img)[1]))
                        cloned_total += 1
print(f"   ✅ Balanced! Created {cloned_total} new synthetic samples.")

# ------------------------------------------------------------------
# [4/8] CONFIGURATION
# ------------------------------------------------------------------
print("\n[4/8] Generating Data Config...")
data_yaml = {
    'path': true_root, 'train': 'train/images', 'val': 'val/images', 'test': 'test/images',
    'nc': 12,
    'names': ['camouflage_soldier', 'weapon', 'military_tank', 'military_truck', 
              'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle', 
              'military_artillery', 'trench', 'military_aircraft', 'military_warship']
}
with open('/content/data.yaml', 'w') as f: yaml.dump(data_yaml, f)

# ------------------------------------------------------------------
# [5/8] TRAIN MODEL A (NANO - 50 EPOCHS - RESUMABLE)
# ------------------------------------------------------------------
print("\n[5/8] Checking Model A (Nano)...")
run_name_n = 'military_nano_final'
last_pt_n = f'{PROJECT_PATH}/{run_name_n}/weights/last.pt'
best_pt_n = f'{PROJECT_PATH}/{run_name_n}/weights/best.pt'

if os.path.exists(last_pt_n):
    print("   ⚠️ Interrupted run detected! Resuming Model A...")
    model_n = YOLO(last_pt_n)
    model_n.train(resume=True)
elif os.path.exists(best_pt_n):
    print("   ✅ Model A already finished. Skipping.")
else:
    print("   🆕 Starting Model A from scratch...")
    model_n = YOLO('yolov8n.pt')
    model_n.train(data='/content/data.yaml', epochs=50, patience=15, imgsz=640, batch=16,
                  project=PROJECT_PATH, name=run_name_n, exist_ok=True,
                  box=7.5, hsv_h=0.015, degrees=10.0, mosaic=1.0, verbose=True)

# ------------------------------------------------------------------
# [6/8] TRAIN MODEL B (SMALL - 50 EPOCHS - RESUMABLE)
# ------------------------------------------------------------------
print("\n[6/8] Checking Model B (Small)...")
run_name_s = 'military_small_final'
last_pt_s = f'{PROJECT_PATH}/{run_name_s}/weights/last.pt'
best_pt_s = f'{PROJECT_PATH}/{run_name_s}/weights/best.pt'

if os.path.exists(last_pt_s):
    print("   ⚠️ Interrupted run detected! Resuming Model B...")
    model_s = YOLO(last_pt_s)
    model_s.train(resume=True)
elif os.path.exists(best_pt_s):
    print("   ✅ Model B already finished. Skipping.")
else:
    print("   🆕 Starting Model B from scratch...")
    model_s = YOLO('yolov8s.pt')
    model_s.train(data='/content/data.yaml', epochs=50, patience=15, imgsz=640, batch=16,
                  project=PROJECT_PATH, name=run_name_s, exist_ok=True,
                  box=7.5, hsv_h=0.015, degrees=10.0, mosaic=1.0, verbose=True)

# ------------------------------------------------------------------
# [7/8] MEMORY-SAFE ENSEMBLE INFERENCE
# ------------------------------------------------------------------
print("\n[7/8] Running Inference...")
output_dir = '/content/submission_final_safe'
if os.path.exists(output_dir): shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Resolve Weights (Use Last if Best doesn't exist for some reason)
w_n = best_pt_n if os.path.exists(best_pt_n) else last_pt_n
w_s = best_pt_s if os.path.exists(best_pt_s) else last_pt_s

if not os.path.exists(w_n) or not os.path.exists(w_s):
    print("❌ CRITICAL: Weights missing. Training likely failed.")
else:
    print(f"   Using Weights:\n   - {w_n}\n   - {w_s}")
    model_n = YOLO(w_n)
    model_s = YOLO(w_s)

    test_dir = os.path.join(true_root, 'test/images')
    image_files = sorted(glob.glob(os.path.join(test_dir, '*')))
    BATCH_SIZE = 50
    batches = [image_files[i:i+BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE)]
    
    print(f"   Processing {len(image_files)} images in {len(batches)} batches...")
    weights = [1, 2, 3] # Nano, NanoTTA, Small
    
    for batch in tqdm(batches, desc="Inference"):
        try:
            # 1. Nano Standard
            res_n1 = model_n.predict(batch, imgsz=640, conf=0.15, augment=False, verbose=False)
            # 2. Nano TTA
            res_n2 = model_n.predict(batch, imgsz=800, conf=0.15, augment=True, verbose=False)
            # 3. Small Standard
            res_s1 = model_s.predict(batch, imgsz=640, conf=0.15, augment=False, verbose=False)

            # Fuse Results
            for i, r_base in enumerate(res_n1):
                boxes_list, scores_list, labels_list = [], [], []
                
                def extract(res):
                    if len(res.boxes):
                        boxes_list.append(res.boxes.xyxyn.cpu().numpy().tolist())
                        scores_list.append(res.boxes.conf.cpu().numpy().tolist())
                        labels_list.append(res.boxes.cls.cpu().numpy().tolist())
                    else: boxes_list.append([]); scores_list.append([]); labels_list.append([])
                
                extract(res_n1[i])
                extract(res_n2[i])
                extract(res_s1[i])

                # WBF
                boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=0.60, skip_box_thr=0.001)

                # Write File
                fname = os.path.basename(r_base.path).replace(os.path.splitext(r_base.path)[1], ".txt")
                with open(os.path.join(output_dir, fname), 'w') as f:
                    for b, s, l in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = b
                        xc, yc, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
                        # Clamp
                        xc, yc = max(0, min(1, xc)), max(0, min(1, yc))
                        w, h = max(0, min(1, w)), max(0, min(1, h))
                        f.write(f"{int(l)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {s:.6f}\n")
            
            # Flush Memory
            del res_n1, res_n2, res_s1
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"⚠️ Batch Error: {e}")

    # ------------------------------------------------------------------
    # [8/8] DOWNLOAD
    # ------------------------------------------------------------------
    print("\n[8/8] Zipping & Downloading...")
    shutil.make_archive('/content/submission_final', 'zip', output_dir)
    try:
        files.download('/content/submission_final.zip')
        print("✅ Auto-download triggered.")
    except:
        print("⚠️ Auto-download failed. Please download 'submission_final.zip' manually.")