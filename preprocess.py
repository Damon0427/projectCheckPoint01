"""
Extract face crops from FaceForensics++ videos and save as .npy batches into SAVE PATH.

"""

import os
import gc
import argparse
import glob

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

SAVE_PATH = "./data/"
NUM_FRAMES = 16
# After downloading the FaceForensics++ dataset, set these paths to the respective video directories:
ORIGINAL_DIR = ""
DEEPFAKE_DIR = ""
TARGET_SIZE = (224, 224)
BLAZE_PATH = "./MediaPipeModel/blaze_face_short_range.tflite"  



## Face detection using MediaPipe's BlazeFace model
def build_face_detector(tflite_path: str):
    BaseOptions        = mp.tasks.BaseOptions
    FaceDetector       = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=tflite_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return FaceDetector.create_from_options(options)


# Frame extraction and face cropping for a single video
def process_video_to_faces(video_path, num_frames, target_size, face_detector):
    """
    Sample `num_frames` frames from a video, detect & crop the face in each,
    and return an array of shape (num_frames, H, W, 3) normalised to [0, 1].
    Returns None if fewer than num_frames faces were detected.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    # Equidistant sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    face_sequence = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results   = face_detector.detect(mp_image)

        if results.detections:
            bbox = results.detections[0].bounding_box
            ih, iw, _ = frame.shape

            # 3 % margin around the bounding box
            margin_w = int(bbox.width  * 0.03)
            margin_h = int(bbox.height * 0.03)

            x = int(max(0, bbox.origin_x - margin_w // 2))
            y = int(max(0, bbox.origin_y - margin_h // 2))
            w = int(min(iw - x, bbox.width  + margin_w))
            h = int(min(ih - y, bbox.height + margin_h))

            face_crop = frame_rgb[y:y+h, x:x+w]
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, target_size)
                face_sequence.append(face_crop.astype(np.float32) / 255.0)

    cap.release()

    return np.array(face_sequence) if len(face_sequence) == num_frames else None


# Batch processing of videos in a directory
def run_batch_processing(source_dir, label_name, label_value,
                         num_frames, target_size, face_detector,
                         save_dir, batch_size=50):
    if not os.path.exists(source_dir):
        print(f"[ERROR] Directory not found: {source_dir}")
        return

    video_files = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]
    current_data, current_labels = [], []
    batch_idx = 1

    print(f"Processing {label_name} videos — total: {len(video_files)}")

    for i, filename in enumerate(tqdm(video_files)):
        video_path = os.path.join(source_dir, filename)
        faces = process_video_to_faces(video_path, num_frames, target_size, face_detector)

        if faces is not None:
            current_data.append(faces)
            current_labels.append(label_value)

        # Flush batch to disk
        if len(current_data) == batch_size or i == len(video_files) - 1:
            if len(current_data) > 0:
                np.save(f"{save_dir}/{label_name}_data_b{batch_idx}.npy",  np.array(current_data))
                np.save(f"{save_dir}/{label_name}_label_b{batch_idx}.npy", np.array(current_labels))
                batch_idx += 1
                del current_data, current_labels
                gc.collect()
                current_data, current_labels = [], []

    print(f"Done — saved to: {save_dir}")



# Main execution for processing both real and fake videos into .npy batches
if __name__ == "__main__":
    os.makedirs(SAVE_PATH, exist_ok=True)

    if not ORIGINAL_DIR:
        raise ValueError("ORIGINAL_DIR is empty. Please set the path to the real videos directory.")

    if not DEEPFAKE_DIR:
        raise ValueError("DEEPFAKE_DIR is empty. Please set the path to the fake videos directory.")
    
    face_detector = build_face_detector(BLAZE_PATH)

    # Real videos -> label 0
    run_batch_processing(
        ORIGINAL_DIR,
        "real",
        0,
        NUM_FRAMES,
        TARGET_SIZE,
        face_detector,
        SAVE_PATH,
        batch_size=50,
    )

    # Fake videos -> label 1
    run_batch_processing(
        DEEPFAKE_DIR,
        "fake",
        1,
        NUM_FRAMES,
        TARGET_SIZE,
        face_detector,
        SAVE_PATH,
        batch_size=50,
    )