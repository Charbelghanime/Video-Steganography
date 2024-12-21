import cv2
import os
import numpy as np
from scipy.signal import welch
import pywt
from skimage.feature import hog
import librosa

# Utility functions
def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frames = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(output_dir, f"frames_{video_name}")
    os.makedirs(frame_dir, exist_ok=True)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frame_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        count += 1

    cap.release()
    print(f"[INFO] Frames extracted and saved to {frame_dir}")
    return frames

def extract_audio(video_path):
    audio_path = f"{os.path.splitext(video_path)[0]}.wav"
    os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y")
    audio, sr = librosa.load(audio_path, sr=None)
    print(f"[INFO] Audio extracted and saved as {audio_path}")
    return audio

def calculate_sift_descriptors(frame_path):
    frame = cv2.imread(frame_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
    if descriptors is not None:
        hash_sift = np.sum(descriptors) % 256  # Example hash computation from descriptors
    else:
        hash_sift = 0
    print("[INFO] Hash sequences from SIFT features calculated")
    return hash_sift

def calculate_ste_hash(audio_signal, frame_length=2048):
    energy = [np.sum(audio_signal[i:i + frame_length] ** 2) for i in range(0, len(audio_signal), frame_length)]
    hash_ste = [1 if e > np.mean(energy) else 0 for e in energy]
    print("[INFO] Hash sequences from short-term energy features calculated")
    return hash_ste

def calculate_dwt_hash(audio_signal):
    coeffs = pywt.dwt(audio_signal, 'db1')
    # Coefficients from DWT are returned as a tuple: (approximation, detail)
    approximation, detail = coeffs
    
    # Concatenate both coefficients (optional, depending on use case)
    combined_coeffs = np.concatenate((approximation, detail))
    
    # Generate hash based on the combined coefficients
    hash_dwt = [1 if x > np.mean(combined_coeffs) else 0 for x in combined_coeffs]
    
    print("[INFO] Hash sequences from DWT coefficients calculated using PyWavelets")
    return hash_dwt

def update_retrieval_database(R, video_id, feature_id, position_id, hash_sequence):
    if hash_sequence not in R:
        R[hash_sequence] = []
    R[hash_sequence].append((video_id, feature_id, position_id))

def video_retrieval_database_construction(videos_dir):
    retrieval_database = {}
    carrier_videos = []

    video_files = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir) if f.endswith(".mp4")]
    output_dir = "Processed"
    os.makedirs(output_dir, exist_ok=True)

    for i, video_path in enumerate(video_files):
        video_id = i + 1

        # Step 3: Extract frame images
        frames = extract_frames(video_path, output_dir)

        # Step 4-6: Process each frame
        for j, frame_path in enumerate(frames):
            hash_sift = calculate_sift_descriptors(frame_path)
            update_retrieval_database(retrieval_database, video_id, 'SIFT', j, hash_sift)
        print("[INFO] Hash sequences from SIFT features calculated")

        # Step 8: Extract audio
        audio = extract_audio(video_path)

        # Step 9-11: Short-term energy hash
        hash_ste = calculate_ste_hash(audio)
        for j, h in enumerate(hash_ste):
            update_retrieval_database(retrieval_database, video_id, 'STE', j, h)

        # Step 13-15: DWT hash
        hash_dwt = calculate_dwt_hash(audio)
        for j, h in enumerate(hash_dwt):
            update_retrieval_database(retrieval_database, video_id, 'DWT', j, h)

        # Step 17: Append to carrier videos
        carrier_videos.append(video_path)
        print(f"[INFO] Video {video_path} processed")

    return retrieval_database, carrier_videos

# Example usage
videos_dir = "Videos"
retrieval_database, carrier_videos = video_retrieval_database_construction(videos_dir)

print("Retrieval Database:", retrieval_database)
print("Carrier Videos:", carrier_videos)
