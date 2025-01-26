import sqlite3
import json
import os


class SecretExtractor:
    def __init__(self, db_path):
        """
        Initialize the SecretExtractor with the path to the retrieval database.

        Args:
            db_path (str): Path to the retrieval database.
        """
        self.db_path = db_path

    def map_retrieval_info_to_bytes(self, retrieval_info):
        """
        Map retrieval information to the corresponding byte sequences.

        Args:
            retrieval_info (list): List of tuples containing retrieval information (VideoID, FeatureID, PositionID).

        Returns:
            list: List of byte sequences corresponding to the retrieval information.
        """
        byte_sequence = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for info in retrieval_info:
            video_id, feature_id, position_id = info
            cursor.execute(
                "SELECT ByteSequence FROM RetrievalDatabase WHERE VideoID = ? AND FeatureID = ? AND PositionID = ?",
                (video_id, feature_id, position_id)
            )
            result = cursor.fetchone()
            if result:
                byte_sequence.append(result[0])
            else:
                print(f"[WARNING] No byte sequence found for retrieval info: {info}")

        conn.close()
        return byte_sequence

    def remove_padding(self, byte_sequence):
        """
        Remove the padding from the byte sequence to recover the original secret information.

        Args:
            byte_sequence (list): List of binary strings (8-bit) representing the byte sequence.

        Returns:
            str: Original secret information without padding.
        """
        if not byte_sequence:
            print("[ERROR] Byte sequence is empty.")
            return ""

        # Check the last byte to determine padding
        last_byte = byte_sequence[-1]
        if last_byte == '00000000':
            # No padding
            byte_sequence = byte_sequence[:-1]
        else:
            # Padding exists, remove the padded zeros
            padding_length = int(last_byte, 2)
            byte_sequence = byte_sequence[:-1]  # Remove the padding indicator byte
            if padding_length > 0:
                byte_sequence[-1] = byte_sequence[-1][:-padding_length]

        # Combine the byte sequence into the original secret information
        secret_info = ''.join(byte_sequence)
        return secret_info

    def binary_to_text(self, binary_data):
        """
        Convert binary string data to a readable text message.

        Args:
            binary_data (str): Binary string representing the secret information.

        Returns:
            str: Decoded text message.
        """
        chars = [chr(int(binary_data[i:i + 8], 2)) for i in range(0, len(binary_data), 8)]
        return ''.join(chars)

    def binary_to_file(self, binary_data, file_path):
        """
        Convert binary string data to a file.

        Args:
            binary_data (str): Binary string representing the file content.
            file_path (str): Path to save the file.
        """
        with open(file_path, 'wb') as file:
            file.write(bytes(int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8)))

    def extract_secret_info(self, retrieval_info, file_extension, is_file):
        """
        Extract the secret information from the retrieval information and database.

        Args:
            retrieval_info (list): List of tuples containing retrieval information (VideoID, FeatureID, PositionID).
            file_extension (str): The file extension to use for the recovered file (if applicable).
            is_file (bool): Whether the secret is a file or a message.

        Returns:
            tuple: (message, file_extension, file_path) if a file is extracted, else (message, None, None)
        """
        # Step 1: Map retrieval info to byte sequence
        byte_sequence = self.map_retrieval_info_to_bytes(retrieval_info)
        print(f"[INFO] Byte sequence retrieved: {byte_sequence}")

        # Step 2: Remove padding to recover secret information
        binary_data = self.remove_padding(byte_sequence)
        print(f"[INFO] Extracted secret information (binary): {binary_data}")

        if is_file:
            # Step 3: Save the file content
            file_path = os.path.join(os.getcwd(), f"recovered{file_extension}")  # Save in current directory

            if file_extension in ['.txt', '.md', '.py']:  # Text-based files
                file_content_text = self.binary_to_text(binary_data)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(file_content_text)
            else:  # Binary files
                self.binary_to_file(binary_data, file_path)

            print(f"[INFO] File recovered and saved as: {file_path}")
            return None, file_extension, file_path
        else:
            # Step 3: Convert binary data to text (message)
            message = self.binary_to_text(binary_data)
            print(f"[INFO] Extracted message: {message}")
            return message, None, None


if __name__ == "__main__":
    # Example usage
    db_path = "retrieval_database.sqlite"  # Path to the retrieval database

    # Load retrieval information from the file
    try:
        with open("retrieval_info.json", "r") as f:
            retrieval_data = json.load(f)
            retrieval_info = retrieval_data["retrieval_info"]
            file_extension = retrieval_data["file_extension"]
            is_file = retrieval_data["is_file"]
        print(f"[INFO] Retrieval information loaded: {retrieval_info}")
        print(f"[INFO] File extension: {file_extension}")
        print(f"[INFO] Is file: {is_file}")
    except FileNotFoundError:
        print("[ERROR] Retrieval information file not found. Run transmission.py first.")
        exit(1)

    # Create a SecretExtractor object
    extractor = SecretExtractor(db_path)

    # Extract the secret information
    message, file_extension, file_path = extractor.extract_secret_info(retrieval_info, file_extension, is_file)
    if message:
        print("[INFO] Final message:", message)
    else:
        print(f"[INFO] File recovered with extension: {file_extension}, saved at: {file_path}")
