import cv2  # For video and image processing
import os  # For file and directory operations
import numpy as np  # For numerical operations
import pywt  # For computing Discrete Wavelet Transform (DWT)
import librosa  # For audio processing and feature extraction
import time # For timing the process
import sqlite3  # For SQLite database operations
import json  # For saving and loading progress
import signal  # For handling keyboard interrupts

# Utility functions
def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return None

def reset_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

def extract_frames(video_path, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = [] # List to store the paths of extracted frames
    # Get the name of the video file without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Create a directory to save the extracted frames
    frame_dir = os.path.join(output_dir, f"frames_{video_name}")
    os.makedirs(frame_dir, exist_ok=True)

    count = 0  # Frame counter
    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read() # Read the next frame
        if not ret: # If no more frames, break the loop
            break 

        # Save the current frame as an image file
        frame_path = os.path.join(frame_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path) # Add frame path to the list
        count += 1 # Increment frame counter

    cap.release() # Release the video capture object
    print(f"[INFO] Frames extracted and saved to {frame_dir}")
    return frames, frame_dir

def extract_audio(video_path):
    # Define the output audio file path
    audio_path = f"{os.path.splitext(video_path)[0]}.wav"
    if not os.path.exists(audio_path):
        # Use FFmpeg to extract audio from the video
        os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y")
        print(f"[INFO] Audio extracted and saved as {audio_path}")
    else:
        print(f"[INFO] Audio already exists at {audio_path}")
    # Load the audio using librosa
    audio, sr = librosa.load(audio_path, sr=None)
    return audio, audio_path 

def delete_files(file_paths):
    # Delete files from the provided list
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[INFO] Deleted file: {file_path}")

def delete_directory(directory_path):
    # Delete the directory and its contents
    if os.path.exists(directory_path):
        for root, dirs, files in os.walk(directory_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(directory_path)
        print(f"[INFO] Deleted directory: {directory_path}")

def create_database(db_path):
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RetrievalDatabase (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ByteSequence TEXT NOT NULL,
                VideoID INTEGER NOT NULL,
                FeatureID TEXT NOT NULL,
                PositionID INTEGER NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        print(f"[INFO] SQLite database created at {db_path}")
    else:
        print(f"[INFO] Database already exists at {db_path}")

def insert_into_database(db_path, byte_sequence, video_id, feature_id, position_id, unique_hash_count):
    """
    Inserts a unique entry into the RetrievalDatabase. If an entry with the same
    ByteSequence, VideoID, FeatureID, and PositionID already exists, it skips insertion.
    Also keep track of the variable related to the unique hash sequences generated in the datbase. 
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the byte sequence is unique
    cursor.execute("""
        SELECT COUNT(*) FROM RetrievalDatabase WHERE ByteSequence = ?
    """, (byte_sequence,))
    if cursor.fetchone()[0] == 0:
        unique_hash_count[0] += 1

    # Check for duplicate entry
    cursor.execute("""
        SELECT COUNT(*) FROM RetrievalDatabase 
        WHERE ByteSequence = ? AND VideoID = ? AND FeatureID = ? AND PositionID = ?
    """, (byte_sequence, video_id, feature_id, position_id))

    if cursor.fetchone()[0] == 0:  # Proceed if no duplicate entry exists
        cursor.execute("""
            INSERT INTO RetrievalDatabase (ByteSequence, VideoID, FeatureID, PositionID)
            VALUES (?, ?, ?, ?)
        """, (byte_sequence, video_id, feature_id, position_id))
        conn.commit()
        conn.close()

def check_database_integrity(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM RetrievalDatabase WHERE ByteSequence IS NULL OR ByteSequence = ''")
    null_count = cursor.fetchone()[0]
    conn.close()
    return null_count == 0
    
def generate_sift_hash(frame_path):
    """
    Generate SIFT-based hash for a frame based on the paper's implementation.
    """
    # Step 1: Process the frame image
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    resized_frame = cv2.resize(frame, (512, 512))  # Uniform size to 512x512
    block_size = 512 // 3  # Divide into 3x3 blocks

    # Step 2: Divide the frame into 3x3 blocks and count SIFT feature points
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(resized_frame, None)

    # Initialize a list to count SIFT points for each block
    sift_counts = [0] * 9

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        block_x = x // block_size
        block_y = y // block_size
        block_index = block_y * 3 + block_x  # Map 2D block to 1D index
        sift_counts[block_index] += 1

    # Step 3: Generate the hash sequence based on comparisons
    hash_sequence = []
    for i in range(len(sift_counts) - 1):  # Compare adjacent blocks
        if sift_counts[i] < sift_counts[i + 1]:
            hash_sequence.append('1')
        else:
            hash_sequence.append('0')

    # Ensure the hash sequence is exactly 8 bits
    if len(hash_sequence) > 8:
        hash_sequence = hash_sequence[:8]
    elif len(hash_sequence) < 8:
        hash_sequence += ['0'] * (8 - len(hash_sequence))  # Pad with zeros to 8 bits

    # Convert hash sequence to a string
    hash_sequence_str = ''.join(hash_sequence)

    # Step 4: Generate the inverted hash sequence
    inverted_sequence = ''.join('1' if bit == '0' else '0' for bit in hash_sequence_str)

    # Return the sequences as a list of 8-bit entries
    return [hash_sequence_str, inverted_sequence]


def calculate_short_term_energy(audio_signal, frame_length=200, frame_shift=80):
    # Divide the audio signal into frames and calculate short-term energy
    num_frames = (len(audio_signal) - frame_length) // frame_shift + 1
    energy_frames = []
    for i in range(num_frames):
        frame = audio_signal[i * frame_shift: i * frame_shift + frame_length]
        energy = np.sum(frame ** 2)
        energy_frames.append(energy)
    return energy_frames

def calculate_segmented_energy(energy_frames, L0=180):
    if len(energy_frames) == 0:
        raise ValueError("Energy frames are empty. Cannot calculate segmented energy.")
    
    h0 = len(energy_frames) // L0 if len(energy_frames) >= L0 else len(energy_frames)
    if h0 < 8:
        raise ValueError(f"Insufficient segments for hash generation. h0={h0}")
    
    segmented_energy = []
    for i in range(0, len(energy_frames), h0):
        segment = energy_frames[i:i + h0]
        segmented_energy.append(np.sum(segment))
    
    return segmented_energy

def generate_ste_hash(segmented_energy):
    num_hashes = len(segmented_energy) // 8  # Number of 8-bit hashes
    hash_sequences = []
    inverted_sequences = []
    for i in range(num_hashes):
        segment = segmented_energy[i * 8:(i + 1) * 8]
        threshold_k = np.mean(segment)
        hash_sequence = ''.join(['1' if e >= threshold_k else '0' for e in segment])
        hash_sequences.append(hash_sequence)
        # Generate inverted hash sequence
        inverted_sequence = ''.join(format(int(hash_sequence, 2) ^ 0xFF, '08b'))
        inverted_sequences.append(inverted_sequence)

    # Combine the original and inverted sequences
    final_sequences = hash_sequences + inverted_sequences

    return final_sequences

def calculate_dwt_hash(audio_signal, total_values=2750):
    """
    Perform DWT-based hash sequence generation based on the paper's implementation.
    """
    # Step 1: Perform DWT three times to get low-frequency components
    coeffs = audio_signal
    for _ in range(3):
        coeffs, _ = pywt.dwt(coeffs, 'db1')
    low_freq_components = np.abs(coeffs)  # Take absolute values as stated in the paper

    # Step 2: Process coefficients to calculate Zc(j) based on Equation (7)
    num_values = len(low_freq_components)
    h1 = num_values // total_values  # Equation (9)
    if h1 < 2:
        raise ValueError("Not enough coefficients to generate Zc sequence.")

    zc_values = [
        np.sum(low_freq_components[(j - 1) * total_values : j * total_values])
        for j in range(1, h1 + 1)
    ]

    # Step 3: Generate the bit sequence H(j) based on Equation (8)
    bit_sequence = []
    for j in range(len(zc_values) - 1):
        if zc_values[j] > zc_values[j + 1]:
            bit_sequence.append('1')
        else:
            bit_sequence.append('0')

    # Step 4: Convert the bit sequence into bytes
    byte_sequence = []
    for i in range(0, len(bit_sequence), 8):
        byte = bit_sequence[i:i + 8]
        if len(byte) < 8:  # Pad with zeros if necessary
            byte += ['0'] * (8 - len(byte))
        byte_sequence.append(''.join(byte))

    # Step 5: Invert the byte sequence for B'(2)
    inverted_byte_sequence = [format(int(byte, 2) ^ 0xFF, '08b') for byte in byte_sequence]

    # Step 6: Repeat for 2 × h sequences
    h = len(byte_sequence) // 8  # Equation (9)
    final_sequences = []
    for _ in range(2 * h):
        final_sequences.extend(byte_sequence)
        final_sequences.extend(inverted_byte_sequence)

    return final_sequences


def update_retrieval_database(db_path, video_id, feature_id, position_id, hash_sequence, unique_hash_count):
    byte_sequence = ''.join(map(str, hash_sequence))  # Convert hash sequence to string
    insert_into_database(db_path, byte_sequence, video_id, feature_id, position_id, unique_hash_count)

def get_row_count(db_path):
    """Get the current number of rows in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM RetrievalDatabase")
    row_count = cursor.fetchone()[0]
    conn.close()
    return row_count

def video_retrieval_database_construction(videos_dir, db_path, PROGRESS_FILE):
    start_time = time.time() #Start Time
    unique_hash_count = [0]  # Use a mutable list to track the number of unique hash sequences
    round = 0  # Round counter to keep track of how many rounds the process has gone through
    no_change_counter = 0  # Counter to track consecutive rounds with no new entries  
    max_rounds_without_change = 5  # Maximum allowed rounds without database entry changes  
    create_database(db_path)

    # Load progress if it exists
    if os.path.exists(PROGRESS_FILE):
        progress = load_progress()
    else:
        progress = {"video_index": 0, "frame_index": 0, "feature_stage": "SIFT"}
    video_index = progress["video_index"]
    frame_index = progress["frame_index"]
    feature_stage = progress["feature_stage"]

    carrier_videos = []  # To track processed videos
    # List all video files in the specified directory
    video_files = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir) if f.endswith(".mp4")]
    output_dir = "Processed" # Directory to store processed files
    os.makedirs(output_dir, exist_ok=True)

    def handle_interrupt(signal, frame):
        print("\n[INFO] KeyboardInterrupt detected. Saving progress ....")
        save_progress({"video_index": video_index, "frame_index": frame_index, "feature_stage": feature_stage})
        delete_directory("Processed")
        exit(1)

    signal.signal(signal.SIGINT, handle_interrupt)

    previous_row_count = get_row_count(db_path)  # Get the initial row count
    while unique_hash_count[0] < 256:  # Check if unique hash sequences generated are less than 256

        round_start_time = time.time() # Time the round

        terminate = False # Flag to terminate the while loop
        round += 1
        print(f"[INFO] Starting round {round} of processing...")
        # Process each video file
        for i, video_path in enumerate(video_files):
            if unique_hash_count[0] >= 256:
                print("[INFO] All 256 unique hash sequences have been generated. Terminating early.")
                terminate = True
                break #Break out of the for loop

            video_id = i + 1 # Assign a unique ID to the video

            # Step 3: Extract frame images
            frames, frame_dir = extract_frames(video_path, output_dir)

            if feature_stage == "SIFT":
                
                # Time the process of hash generation based on SIFT 
                sift_start_time = time.time()

                # Step 4-6: Process each frame to generate SIFT hash
                print("[INFO] Hash sequence generation from SIFT features has started")
                for j, frame_path in enumerate(frames[frame_index:], start=frame_index):  # Start from frame_index
                    hash_sift_list = generate_sift_hash(frame_path)  # Returns [hash_sequence, inverted_sequence]
            
                    # Insert the original hash sequence into the database with FeatureID=0 and append '0'
                    original_feature_id = "00"  # Direct mapping
                    update_retrieval_database(db_path, video_id, original_feature_id, j, hash_sift_list[0], unique_hash_count)

                    # Insert the inverted hash sequence into the database with FeatureID=0 and append '1'
                    inverted_feature_id = "01"  # Indirect mapping
                    update_retrieval_database(db_path, video_id, inverted_feature_id, j, hash_sift_list[1], unique_hash_count)
                    # Save progress
                    save_progress({"video_index": i, "frame_index": j + 1, "feature_stage": "SIFT"})
                    frame_index = j + 1 # Save frame index locally in order to save progress upon keyboard interrupt
                sift_end_time = time.time()
                print("[INFO] Hash sequence generation from SIFT features is complete")
                print(f"[INFO] SIFT hash sequence generation took and mapping {sift_end_time - sift_start_time:.2f} seconds")

                # Save progress
                save_progress({"video_index": i, "frame_index": 0, "feature_stage": "STE"})
                # Move to next stage
                feature_stage = "STE"  

            # Step 8: Extract audio
            audio, audio_path = extract_audio(video_path)

            
            if feature_stage == "STE":

                # Time the process of hash generation based on STE 
                ste_start_time = time.time()

                # Step 9-11: Short-term energy hash
                print("[INFO] Hash sequence generation from STE features has started")
                energy_frames = calculate_short_term_energy(audio)
                segmented_energy = calculate_segmented_energy(energy_frames)
                hash_ste_list = generate_ste_hash(segmented_energy)
                print(f"[DEBUG] Generated STE hash sequences: {hash_ste_list}")  # Debug to verify the hash
                for j in range(0, len(hash_ste_list), 2):
                    # Direct STE hash sequence
                    direct_feature_id = "10"  # FeatureID for direct mapping
                    update_retrieval_database(db_path, video_id, direct_feature_id, j // 2, hash_ste_list[j], unique_hash_count)

                    # Inverted STE hash sequence
                    inverted_feature_id = "11"  # FeatureID for inverted mapping
                    update_retrieval_database(db_path, video_id, inverted_feature_id, j // 2, hash_ste_list[j + 1], unique_hash_count)

                    # Save progress
                    save_progress({"video_index": i, "frame_index": j + 1, "feature_stage": "STE"})
                ste_end_time = time.time()
                print("[INFO] Hash sequence generation from STE features is complete")
                print(f"[INFO] STE hash sequence generation and mapping took {ste_end_time - ste_start_time:.2f} seconds")
                # Save progress
                save_progress({"video_index": i, "frame_index": 0, "feature_stage": "DWT"})
                # Move to next stage
                feature_stage = "DWT" 

            if feature_stage == "DWT":

                # Time the process of hash generation based on DWT coefficients
                dwt_start_time = time.time()

                # Step 13-15: DWT hash
                print("[INFO] Hash sequence generation from DWT features has started")
                dwt_hash_sequences = calculate_dwt_hash(audio)
                print(f"[DEBUG] Generated DWT hash sequences: {dwt_hash_sequences}")  # Debug to verify the hash
                for j in range(0, len(dwt_hash_sequences), 2):
                    # Direct STE hash sequence
                    direct_feature_id = "20"  # FeatureID for direct mapping
                    update_retrieval_database(db_path, video_id, direct_feature_id, j // 2, dwt_hash_sequences[j], unique_hash_count)

                    # Inverted STE hash sequence
                    inverted_feature_id = "21"  # FeatureID for inverted mapping
                    update_retrieval_database(db_path, video_id, inverted_feature_id, j // 2, dwt_hash_sequences[j + 1], unique_hash_count)

                    # Save progress
                    save_progress({"video_index": i, "frame_index": j + 1, "feature_stage": "DWT"})
                dwt_end_time = time.time()
                print("[INFO] Hash sequence generation from DWT features is complete")
                print(f"[INFO] DWT hash sequence generation and mapping took {dwt_end_time - dwt_start_time:.2f} seconds")
                # Save progress
                save_progress({"video_index": i + 1, "frame_index": 0, "feature_stage": "SIFT"})
                # Move to next video
                feature_stage = "SIFT"

            # Step 17: Append to carrier videos
            if (video_path) not in carrier_videos: # Check if the video is already in the carrier videos folder
                carrier_videos.append(video_path)
                print(f"[INFO] Video {video_path} processed")

        # Check for changes in database row count
        current_row_count = get_row_count(db_path)
        if current_row_count == previous_row_count:
            no_change_counter += 1
            print(f"[INFO] No new entries detected in round {round}. Counter: {no_change_counter}/{max_rounds_without_change}")
        else:
            no_change_counter = 0  # Reset the counter if new rows are added
        previous_row_count = current_row_count

        # Terminate if no new entries after maximum allowed rounds
        if no_change_counter >= max_rounds_without_change:
            print("[INFO] No new entries detected for 5 consecutive rounds. Terminating to save time.")
            break

        round_end_time = time.time()
        print(f"[INFO] Completed round {round} in {round_end_time - round_start_time:.2f} seconds")
        
        if terminate:
            break  # Break out of the while loop

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"[INFO] Retrieval database construction completed in {total_duration:.2f} seconds")
    
    # Step 18-20: Check the retrieval database and finalize
    if check_database_integrity(db_path):
        print("[INFO] Retrieval database successfully constructed and verified to be non-empty.")
    else:
        print("[WARNING] Retrieval database contains empty or null entries.")

    delete_directory("Processed")
    
    # Step 21: Return the constructed retrieval database and carrier videos
    return db_path, carrier_videos

if __name__ == "__main__": # Execute only when called directly
    # Example usage
    # Progress file path
    PROGRESS_FILE = "progress.json"
    videos_dir = "Videos" # Directory containing input videos
    db_path = "retrieval_database.sqlite"
    db_path, carrier_videos = video_retrieval_database_construction(videos_dir, db_path, PROGRESS_FILE)
    print(f"[INFO] Database stored at {db_path}")
    print(f"[INFO] Carrier Videos: {carrier_videos}")
