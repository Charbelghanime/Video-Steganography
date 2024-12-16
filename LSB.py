from stegano import lsb
from os.path import isfile, join, basename, splitext
import time
import cv2
import numpy as np
import math
import os
import shutil
from subprocess import call, STDOUT

def get_output_paths(video):
    video_name = splitext(basename(video))[0]
    frames_folder = f"frames-{video_name}"
    audio_file = f"audio-{video_name}.mp3"
    return frames_folder, audio_file

def frame_extraction(video):
    frames_folder, _ = get_output_paths(video)
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    print(f"[INFO] Directory '{frames_folder}' is created to store frames.")
    vidcap = cv2.VideoCapture(video)
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(frames_folder, "{:d}.png".format(count)), image)
        count += 1

def extract_audio(video):
    _, audio_file = get_output_paths(video)
    print(f"[INFO] Extracting audio to '{audio_file}'")
    try:
        call(["ffmpeg", "-i", video, "-q:a", "0", "-map", "a", audio_file, "-y"], stdout=open(os.devnull, "w"), stderr=STDOUT)
        print("[INFO] Audio extracted successfully")
    except Exception as e:
        print(f"[ERROR] Failed to extract audio: {e}")
    return audio_file

def stitch_frames_and_audio(video):
    frames_folder, audio_file = get_output_paths(video)
    embedded_video = f"Embedded_Video_with_Audio-{splitext(basename(video))[0]}.mp4"
    print("[INFO] Stitching frames into video")
    try:
        call(["ffmpeg", "-i", os.path.join(frames_folder, "%d.png"), "-vcodec", "png", "temp_video.mp4", "-y"], stdout=open(os.devnull, "w"), stderr=STDOUT)
        print("[INFO] Frames stitched successfully into 'temp_video.mp4'")
    except Exception as e:
        print(f"[ERROR] Failed to stitch frames: {e}")
        return

    print("[INFO] Adding audio to the video")
    try:
        call(["ffmpeg", "-i", "temp_video.mp4", "-i", audio_file, "-codec", "copy", embedded_video, "-y"], stdout=open(os.devnull, "w"), stderr=STDOUT)
        print(f"[INFO] Audio added successfully. Video saved as '{embedded_video}'")
    except Exception as e:
        print(f"[ERROR] Failed to add audio: {e}")

    os.remove("temp_video.mp4")  # Clean up temporary stitched video

def split_string(split_str,count=10):
    per_c=math.ceil(len(split_str)/count)
    c_cout=0
    out_str=''
    split_list=[]
    for s in split_str:
        out_str+=s
        c_cout+=1
        if c_cout == per_c:
            split_list.append(out_str) # The message is divided into substrings
            out_str=''
            c_cout=0
    if c_cout!=0:
        split_list.append(out_str)
    return split_list

def embed_string(input_string,video):
    frames_folder, _ = get_output_paths(video)
    frame_extraction(video)  # Extract frames from video
    extract_audio(video)  # Extract audio from video
    split_string_list=split_string(input_string)   # Acquire the splitted string from the message.
    for i in range(0,len(split_string_list)):
        f_name = os.path.join(frames_folder, "{}.png".format(i))                  
        if not os.path.exists(f_name):
            print(f"[ERROR] Frame not found: {f_name}")
            continue
        secret_enc = lsb.hide(f_name, split_string_list[i])
        secret_enc.save(f_name)
        print(f"[INFO] Frame {f_name} holds '{lsb.reveal(f_name)}'")

    stitch_frames_and_audio(video)


def extract_string(video):
    frames_folder, _ = get_output_paths(video)
    frame_extraction(video)  # Extract each frame
    secret = []
    for i in range(len(os.listdir(frames_folder))):
        f_name = os.path.join(frames_folder, "{}.png".format(i))
        secret_dec = lsb.reveal(f_name)
        if secret_dec is None:
            break
        secret.append(secret_dec)
    print("".join(secret))
    print("[INFO] Secret message retrieved successfully.")

def clean_temp(video):
    frames_folder, audio_file = get_output_paths(video)
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
        print(f"[INFO] '{frames_folder}' cleaned up.")
    if os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"[INFO] '{audio_file}' cleaned up.")

if __name__ == "__main__":    
    while True:
        print("1.Hide a message in video\n2.Reveal the secret from the video\n")
        print("Any other value to exit\n")
        choice = input("Enter your choice :")
        if choice == '1':
            video_file=input("Enter the name of video file with extension:")
            secret_message = input("Enter the secret message: ")
            embed_string(secret_message, video_file)
        elif choice == '2':
            video_file = input("Enter the name of video file with extension: ")
            extract_string(video_file)
        else:
            break
