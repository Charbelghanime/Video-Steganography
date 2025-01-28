import cv2  # For video and image processing
import os  # For file and directory operations
import numpy as np  # For numerical operations
import pywt  # For computing Discrete Wavelet Transform (DWT)
import librosa  # For audio processing and feature extraction
import time  # For timing the process
import sqlite3  # For SQLite database operations
import json  # For saving and loading progress
import signal  # For handling keyboard interrupts


class VideoProcessor:
    def __init__(self, videos_dir, db_path, progress_file):
        self.videos_dir = videos_dir  # Directory containing input videos
        self.db_path = db_path  # Path to the SQLite database
        self.progress_file = progress_file  # Path to the progress file
        self.unique_hash_count = [0]  # Use a mutable list to track the number of unique hash sequences
        self.sift = cv2.SIFT_create()
        

    def save_progress(self, progress):
        """Save progress to a JSON file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)

    def load_progress(self):
        """Load progress from a JSON file if it exists."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return None

    def reset_progress(self):
        """Reset progress by deleting the progress file."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Convert to grayscale once
        cap.release()
        return frames

    def extract_audio(self, video_path):
        audio, sr = librosa.load(video_path, sr=None)
        return audio, sr

    def create_database(self):
        """Create the SQLite database if it doesn't exist."""
        if not os.path.exists(self.db_path):
            print(f"[INFO] Creating new database at {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA synchronous = OFF")
            cursor.execute("PRAGMA journal_mode = MEMORY")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS RetrievalDatabase (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ByteSequence TEXT NOT NULL,
                    VideoID INTEGER NOT NULL,
                    FeatureID TEXT NOT NULL,
                    PositionID INTEGER NOT NULL,
                    UNIQUE(ByteSequence, VideoID, FeatureID, PositionID)
                )
            """)
            conn.commit()
            conn.close()
            print(f"[INFO] SQLite database and table created at {self.db_path}")
        else:
            print(f"[INFO] Database already exists at {self.db_path}")
            # Verify if the table exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='RetrievalDatabase'")
            table_exists = cursor.fetchone()
            conn.close()
            if not table_exists:
                print("[WARNING] Table 'RetrievalDatabase' does not exist. Deleting and recreating the database.")
                os.remove(self.db_path)
                self.create_database()  # Recursively recreate the database

    def insert_batch_into_database(self, batch):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        cursor.executemany("""
            INSERT OR IGNORE INTO RetrievalDatabase 
            (ByteSequence, VideoID, FeatureID, PositionID)
            VALUES (?, ?, ?, ?)
        """, batch)
        conn.commit()
        conn.close()

    def check_database_integrity(self):
        """Check if the database contains any empty or null entries."""
        conn = sqlite3.connect(self.db_path)  # Connect to the database
        cursor = conn.cursor()
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        cursor.execute("SELECT COUNT(*) FROM RetrievalDatabase WHERE ByteSequence IS NULL OR ByteSequence = ''")
        null_count = cursor.fetchone()[0]  # Count the number of null or empty entries
        conn.close()  # Close the connection
        return null_count == 0  # Return True if no null or empty entries exist

    def generate_sift_hash(self, frame):
        """
        Generate SIFT-based hash for a frame based on the paper's implementation.
        """
        # Step 1: Process the frame image
        resized_frame = cv2.resize(frame, (512, 512))  # Uniform size to 512x512
        block_size = 512 // 3  # Divide into 3x3 blocks

        # Step 2: Divide the frame into 3x3 blocks and count SIFT feature points
        keypoints, _ = self.sift.detectAndCompute(resized_frame, None)  # Detect keypoints

        # Initialize a list to count SIFT points for each block
        sift_counts = [0] * 9

        for kp in keypoints:  # Count SIFT points in each block
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

    def calculate_short_term_energy(self, audio_signal, frame_length=200, frame_shift=80):
        """Divide the audio signal into frames and calculate short-term energy."""
        num_frames = (len(audio_signal) - frame_length) // frame_shift + 1
        indices = np.arange(frame_length)[None, :] + frame_shift * np.arange(num_frames)[:, None]
        frames = audio_signal[indices]
        return np.sum(frames ** 2, axis=1)

    def calculate_segmented_energy(self, energy_frames, L0=180):
        """Calculate segmented energy from the energy frames."""
        if len(energy_frames) == 0:
            raise ValueError("Energy frames are empty. Cannot calculate segmented energy.")

        h0 = len(energy_frames) // L0 if len(energy_frames) >= L0 else len(energy_frames)
        if h0 < 8:
            raise ValueError(f"Insufficient segments for hash generation. h0={h0}")

        segmented_energy = []
        for i in range(0, len(energy_frames), h0):
            segment = energy_frames[i:i + h0]
            segmented_energy.append(np.sum(segment))  # Sum the energy in each segment

        return segmented_energy

    def generate_ste_hash(self, segmented_energy):
        """Generate hash sequences from segmented energy."""
        num_hashes = len(segmented_energy) // 8  # Number of 8-bit hashes
        hash_sequences = []
        inverted_sequences = []
        for i in range(num_hashes):
            segment = segmented_energy[i * 8:(i + 1) * 8]
            threshold_k = np.mean(segment)  # Calculate the threshold
            hash_sequence = ''.join(['1' if e >= threshold_k else '0' for e in segment])  # Generate hash sequence
            hash_sequences.append(hash_sequence)
            # Generate inverted hash sequence
            inverted_sequence = ''.join(format(int(hash_sequence, 2) ^ 0xFF, '08b'))
            inverted_sequences.append(inverted_sequence)

        # Combine the original and inverted sequences
        final_sequences = hash_sequences + inverted_sequences

        return final_sequences

    def calculate_dwt_hash(self, audio_signal, total_values=2750):
        """
        Perform DWT-based hash sequence generation based on the paper's implementation.
        """
        # Step 1: Perform DWT three times to get low-frequency components
        coeffs = audio_signal
        for _ in range(3):
            coeffs, _ = pywt.dwt(coeffs, 'db1')  # Perform DWT
        low_freq_components = np.abs(coeffs)  # Take absolute values as stated in the paper

        # Step 2: Process coefficients to calculate Zc(j) based on Equation (7)
        num_values = len(low_freq_components)
        h1 = num_values // total_values  # Equation (9)
        if h1 < 2:
            raise ValueError("Not enough coefficients to generate Zc sequence.")

        zc_values = [
            np.sum(low_freq_components[(j - 1) * total_values: j * total_values])
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

    def get_row_count(self):
        """Get the current number of rows in the database."""
        conn = sqlite3.connect(self.db_path)  # Connect to the database
        cursor = conn.cursor()
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        cursor.execute("SELECT COUNT(*) FROM RetrievalDatabase")  # Count rows
        row_count = cursor.fetchone()[0]
        conn.close()  # Close the connection
        return row_count
    
    def count_unique_hash_sequences(self):
        """
        Count the number of unique hash sequences in the database.

        Returns:
            int: Number of unique hash sequences.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # SQL query to count distinct ByteSequence
        cursor.execute("SELECT COUNT(DISTINCT ByteSequence) FROM RetrievalDatabase")
        unique_count = cursor.fetchone()[0]
        
        conn.close()
        return unique_count
    
    def video_retrieval_database_construction(self):
        """Construct the retrieval database by processing videos."""
        start_time = time.time()  # Start time
        round = 0  # Round counter
        no_change_counter = 0  # Counter to track consecutive rounds with no new entries
        max_rounds_without_change = 2  # Maximum allowed rounds without database entry changes
        self.create_database()  # Create the database if it doesn't exist
        batch = []
        BATCH_SIZE = 1000  # Adjust based on memory constraints

        progress = self.load_progress()
        if progress:
            video_index = progress.get("video_index", 0)
            frame_index = progress.get("frame_index", 0)
            feature_stage = progress.get("feature_stage", "SIFT")
            audio_progress = progress.get("audio_progress", {})
        else:
            video_index = 0
            frame_index = 0
            feature_stage = "SIFT"
            audio_progress = {}

        carrier_videos = []  # To track processed videos
        video_files = [os.path.join(self.videos_dir, f) for f in os.listdir(self.videos_dir) if f.endswith(".mp4")]  # List all video files

        def handle_interrupt(signal, frame):
            print("\n[INFO] KeyboardInterrupt detected. Saving progress ....")
            self.save_progress({
                "video_index": video_index,
                "frame_index": frame_index,
                "feature_stage": feature_stage,
                "audio_progress": audio_progress
            })
            exit(1)

        signal.signal(signal.SIGINT, handle_interrupt)  # Set up the interrupt handler

        previous_row_count = self.get_row_count()  # Get the initial row count
        
        while True:  # Check if unique hash sequences generated are less than 256
            round_start_time = time.time()  # Time the round
            terminate = False  # Flag to terminate the while loop
            round += 1
            print(f"[INFO] Starting round {round} of processing...")

            # Process each video file
            for i, video_path in enumerate(video_files[video_index:], start=video_index):
                unique_hash_count = self.count_unique_hash_sequences()
                if unique_hash_count >= 256:  # Check if all 256 unique hash sequences have been generated
                    print("[INFO] All 256 unique hash sequences have been generated. Terminating early.")
                    terminate = True
                    break  # Break out of the for loop

                video_id = i + 1  # Assign a unique ID to the video

                # Step 3: Extract frame images
                frames = self.extract_frames(video_path)
                audio, _ = self.extract_audio(video_path)

                if feature_stage == "SIFT":
                    # Time the process of hash generation based on SIFT
                    sift_start_time = time.time()

                    # Step 4-6: Process each frame to generate SIFT hash
                    print("[INFO] Hash sequence generation from SIFT features for Video " + video_id + "has started")
                    for j, frame_path in enumerate(frames[frame_index:], start=frame_index):  # Start from frame_index
                        hash_sift_list = self.generate_sift_hash(frame_path)  # Returns [hash_sequence, inverted_sequence]

                        batch.append((hash_sift_list[0], video_id, "00", j))
                        batch.append((hash_sift_list[1], video_id, "01", j))
                        if len(batch) >= BATCH_SIZE:
                            self.insert_batch_into_database(batch)
                            batch = []

                        # Save progress
                        self.save_progress({
                            "video_index": i,
                            "frame_index": j + 1,
                            "feature_stage": "SIFT",
                            "audio_progress": audio_progress
                        })
                        frame_index = j + 1  # Save frame index locally in order to save progress upon keyboard interrupt

                    sift_end_time = time.time()
                    print(f"[INFO] SIFT hash sequence generation took and mapping {sift_end_time - sift_start_time:.2f} seconds")
                    print("[INFO] Hash sequence generation from SIFT features from Video " + video_id + "is complete")
                    # Save progress
                    feature_stage = "STE"
                    frame_index = 0
                    self.save_progress({
                        "video_index": i,
                        "frame_index": 0,
                        "feature_stage": "STE",
                        "audio_progress": audio_progress
                    })

                if feature_stage == "STE":
                    if f"video_{video_id}_ste" not in audio_progress:
                    # Time the process of hash generation based on STE
                        ste_start_time = time.time()

                        # Step 9-11: Short-term energy hash
                        print("[INFO] Hash sequence generation from STE features for Video " + video_id + "has started")
                        audio_progress[f"video_{video_id}_ste"] = "started"
                        self.save_progress({
                            "video_index": i,
                            "frame_index": 0,
                            "feature_stage": "STE",
                            "audio_progress": audio_progress
                        })

                        energy_frames = self.calculate_short_term_energy(audio)
                        segmented_energy = self.calculate_segmented_energy(energy_frames)
                        hash_ste_list = self.generate_ste_hash(segmented_energy)
                        print(f"[DEBUG] Generated STE hash sequences: {hash_ste_list}")  # Debug to verify the hash
                        for j in range(0, len(hash_ste_list), 2):
                            # Direct STE hash sequence
                            batch.append((hash_ste_list[j], video_id, "10", j // 2))
                            batch.append((hash_ste_list[j + 1], video_id, "11", j // 2))
                            if len(batch) >= BATCH_SIZE:
                                self.insert_batch_into_database(batch)
                                batch = []

                        # Save progress
                        # Mark STE hash generation as complete
                        audio_progress[f"video_{video_id}_ste"] = "complete"
                        self.save_progress({
                            "video_index": i,
                            "frame_index": 0,
                            "feature_stage": "DWT",
                            "audio_progress": audio_progress
                        })

                        ste_end_time = time.time()
                        print(f"[INFO] STE hash sequence generation and mapping took {ste_end_time - ste_start_time:.2f} seconds")
                    print("[INFO] Hash sequence generation from STE features for Video " + video_id + "is complete")
                    
                    # Move to next stage
                    feature_stage = "DWT"

                if feature_stage == "DWT":
                    if f"video_{video_id}_dwt" not in audio_progress:
                        # Time the process of hash generation based on DWT coefficients
                        dwt_start_time = time.time()

                        audio_progress[f"video_{video_id}_dwt"] = "started"
                        self.save_progress({
                            "video_index": i,
                            "frame_index": 0,
                            "feature_stage": "DWT",
                            "audio_progress": audio_progress
                        })

                        # Step 13-15: DWT hash
                        print("[INFO] Hash sequence generation from DWT features for Video " + video_id + "has started")
                        dwt_hash_sequences = self.calculate_dwt_hash(audio)
                        print(f"[DEBUG] Generated DWT hash sequences: {dwt_hash_sequences}")  # Debug to verify the hash
                        for j in range(0, len(dwt_hash_sequences), 2):
                            # Direct STE hash sequence
                            batch.append((dwt_hash_sequences[j], video_id, "20", j // 2))
                            batch.append((dwt_hash_sequences[j + 1], video_id, "21", j // 2))
                            if len(batch) >= BATCH_SIZE:
                                self.insert_batch_into_database(batch)
                                batch = []
                        # Save progress
                        audio_progress[f"video_{video_id}_dwt"] = "complete"
                        self.save_progress({
                            "video_index": i + 1,
                            "frame_index": 0,
                            "feature_stage": "SIFT",
                            "audio_progress": audio_progress
                        })

                        dwt_end_time = time.time()
                        print(f"[INFO] DWT hash sequence generation and mapping took {dwt_end_time - dwt_start_time:.2f} seconds")

                    print("[INFO] Hash sequence generation from DWT features for Video " + video_id +  "is complete")
                    
                    # Move to next video
                    feature_stage = "SIFT"
                    video_index += 1

                # Step 17: Append to carrier videos
                if video_path not in carrier_videos:  # Check if the video is already in the carrier videos folder
                    carrier_videos.append(video_path)
                    print(f"[INFO] Video {video_path} processed")

            if batch:  # Insert remaining entries
                self.insert_batch_into_database(batch)
                
            # Check for changes in database row count
            current_row_count = self.get_row_count()
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
        if self.check_database_integrity():
            print("[INFO] Retrieval database successfully constructed and verified to be non-empty.")
        else:
            print("[WARNING] Retrieval database contains empty or null entries.")

        # Step 21: Return the constructed retrieval database and carrier videos
        return self.db_path, carrier_videos


if __name__ == "__main__":
    # Example usage
    PROGRESS_FILE = "progress.json"  # Progress file path
    videos_dir = "Videos"  # Directory containing input videos
    db_path = "retrieval_database.sqlite"  # Path to the SQLite database
    processor = VideoProcessor(videos_dir, db_path, PROGRESS_FILE)  # Create a VideoProcessor object
    db_path, carrier_videos = processor.video_retrieval_database_construction()  # Construct the database
    print(f"[INFO] Database stored at {db_path}")
    print(f"[INFO] Carrier Videos: {carrier_videos}")
