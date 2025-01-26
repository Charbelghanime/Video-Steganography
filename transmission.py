import sqlite3
import os
import random
import json
import subprocess
from establishment_of_db import VideoProcessor  # Import the VideoProcessor class


class SecretTransmitter:
    def __init__(self, db_path):
        self.db_path = db_path  # Path to the retrieval database

    def text_to_binary(self, text):
        """Convert a string to its binary representation."""
        return ''.join(format(ord(char), '08b') for char in text)

    def read_binary_file(self, file_path):
        """Read the content of a binary file and convert it to a binary string."""
        with open(file_path, 'rb') as file:
            binary_data = file.read()
            return ''.join(format(byte, '08b') for byte in binary_data)

    def get_file_extension(self, file_path):
        """Extract the file extension from the file path."""
        return os.path.splitext(file_path)[1]  # Get the file extension (e.g., '.txt', '.jpg')

    def segment_and_pad(self, secret_info):
        """
        Segment the secret information into 8-bit chunks (bytes) and pad if necessary.
        """
        byte_sequence = []
        length = len(secret_info)
        mod_value = length % 8

        if mod_value != 0:
            # Pad the sequence with zeros to make its length a multiple of 8
            padding = 8 - mod_value
            secret_info += '0' * padding
            byte_sequence = [secret_info[i:i + 8] for i in range(0, len(secret_info), 8)]
            # Add an additional byte to indicate the number of zeros padded
            byte_sequence.append(format(padding, '08b'))
        else:
            # Segment into 8-bit chunks and add auxiliary byte 00000000
            byte_sequence = [secret_info[i:i + 8] for i in range(0, len(secret_info), 8)]
            byte_sequence.append('00000000')
            print("[INFO] The message was not padded. Auxiliary information '00000000' is added.")

        return byte_sequence

    def retrieve_from_database(self, byte_sequence):
        """
        Retrieve the corresponding retrieval information for each byte from the database.
        """
        retrieval_info = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for byte in byte_sequence:
            cursor.execute("SELECT VideoID, FeatureID, PositionID FROM RetrievalDatabase WHERE ByteSequence = ?", (byte,))
            results = cursor.fetchall()
            if results:
                retrieval_info.append(random.choice(results))
            else:
                print(f"[WARNING] No match found for byte sequence: {byte}")

        conn.close()
        return retrieval_info

    def transmit_secret_info(self, secret_info, file_path=None):
        """
        Transmit the secret information using the constructed retrieval database and carrier videos.
        """
        file_extension = None
        if file_path:
            # If a file is provided, read its binary content and store the file extension
            file_extension = self.get_file_extension(file_path)  # Get the file extension (e.g., '.txt')
            file_content = self.read_binary_file(file_path)  # Read file content as binary
            secret_info = file_content  # Use only the file content (extension is stored separately)

        # Step 1: Segment and pad the secret information
        byte_sequence = self.segment_and_pad(secret_info)
        print(f"[INFO] Byte sequence after segmentation and padding: {byte_sequence}")

        # Step 2: Retrieve information from the database
        retrieval_info = self.retrieve_from_database(byte_sequence)
        print(f"[INFO] Retrieved retrieval information: {retrieval_info}")

        # Step 3: Save retrieval information and file extension to a file
        with open("retrieval_info.json", "w") as f:
            json.dump({
                "retrieval_info": retrieval_info,
                "file_extension": file_extension,  # Store the file extension explicitly
                "is_file": file_path is not None  # Indicate whether the secret is a file or a message
            }, f)
        print("[INFO] Retrieval information and metadata saved to retrieval_info.json")

        # Step 4: Simulate sending the retrieval information and carrier videos
        print(f"[INFO] Transmitting retrieval information and carrier videos...")

        return retrieval_info


if __name__ == "__main__":
    # Example usage
    videos_dir = "Videos"  # Directory containing carrier videos
    db_path = "retrieval_database.sqlite"  # Path to the retrieval database
    PROGRESS_FILE = "progress.json"  # Path to the progress file

    # Check if the database exists or needs to be constructed
    if os.path.exists(db_path):
        print(f"[INFO] Retrieval database already exists at: {db_path}")
    else:
        # Use the VideoProcessor class to construct the database
        processor = VideoProcessor(videos_dir, db_path, PROGRESS_FILE)
        db_path, carrier_videos = processor.video_retrieval_database_construction()

    # Create a SecretTransmitter object
    transmitter = SecretTransmitter(db_path)

    # Prompt user for input type
    while True:
        try:
            option = int(input("Choose an option:\n1. Message\n2. File\nEnter your choice (1 or 2): ").strip())
            if option == 1:
                secret_info = input("Enter the secret message: ").strip()
                binary_secret_info = transmitter.text_to_binary(secret_info)
                retrieval_info = transmitter.transmit_secret_info(binary_secret_info)
                break
            elif option == 2:
                file_path = input("Enter the path to the file: ").strip()
                if os.path.exists(file_path):
                    retrieval_info = transmitter.transmit_secret_info(None, file_path)
                    break
                else:
                    print("[ERROR] File not found. Please try again.")
            else:
                print("[ERROR] Invalid choice. Please choose either 1 or 2.")
        except ValueError:
            print("[ERROR] Please enter a valid number.")

    print("[INFO] Transmission complete.")
