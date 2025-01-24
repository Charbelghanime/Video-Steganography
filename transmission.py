import math
import sqlite3
import os
from establishment_of_db import video_retrieval_database_construction
import random

def text_to_binary(text):
    """
    Convert a string to its binary representation.
    """
    return ''.join(format(ord(char), '08b') for char in text)

def read_text_file(file_path):
    """
    Read the content of a text file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def segment_and_pad(secret_info):
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

def retrieve_from_database(db_path, byte_sequence):
    """
    Retrieve the corresponding retrieval information for each byte from the database.
    """
    retrieval_info = []

    conn = sqlite3.connect(db_path)
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

def transmit_secret_info(db_path, secret_info):
    """
    Transmit the secret information using the constructed retrieval database and carrier videos.
    """
    # Step 1: Segment and pad the secret information
    byte_sequence = segment_and_pad(secret_info)
    print(f"[INFO] Byte sequence after segmentation and padding: {byte_sequence}")

    # Step 2: Retrieve information from the database
    retrieval_info = retrieve_from_database(db_path, byte_sequence)
    print(f"[INFO] Retrieved retrieval information: {retrieval_info}")

    # Step 3: Send the retrieval information and carrier videos (here just simulate sending)
    print(f"[INFO] Transmitting retrieval information and carrier videos...")

    return retrieval_info

if __name__ == "__main__":
    # Example usage
    videos_dir = "Videos"  # Directory containing carrier videos
    db_path = "retrieval_database.sqlite"  # Path to the retrieval database
    PROGRESS_FILE = "progress.json" # Path to the progress file

    # Check if the database exists or needs to be constructed
    if os.path.exists(db_path):
        print(f"[INFO] Retrieval database already exists at: {db_path}")
    else:
        db_path, carrier_videos = video_retrieval_database_construction(videos_dir, db_path, PROGRESS_FILE)

    # Prompt user for input type
    while True:
        try:
            option = int(input("Choose an option:\n1. Message\n2. Message in a text file\nEnter your choice (1 or 2): ").strip())
            if option == 1:
                secret_info = input("Enter the secret message: ").strip()
                binary_secret_info = text_to_binary(secret_info)
                break
            elif option == 2:
                file_path = input("Enter the path to the text file: ").strip()
                if os.path.exists(file_path):
                    secret_info = read_text_file(file_path)
                    binary_secret_info = text_to_binary(secret_info)
                    break
                else:
                    print("[ERROR] File not found. Please try again.")
            else:
                print("[ERROR] Invalid choice. Please choose either 1 or 2.")
        except ValueError:
            print("[ERROR] Please enter a valid number.")

    # Transmit the secret information
    retrieval_info = transmit_secret_info(db_path, binary_secret_info)
    print("[INFO] Transmission complete.")
