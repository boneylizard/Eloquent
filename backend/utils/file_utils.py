import os
import logging
import shutil
import uuid
import aiofiles
from typing import Optional
from fastapi import UploadFile

# Configuration
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "data/uploads")

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save an uploaded file to disk and return the file path
    """
    try:
        # Create a unique filename to prevent collisions
        file_extension = os.path.splitext(upload_file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Write the file to disk
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
        
        logging.info(f"File saved to {file_path}")
        return file_path
    
    except Exception as e:
        logging.error(f"Error saving uploaded file: {str(e)}")
        raise e

def get_file_info(file_path: str) -> dict:
    """
    Get basic information about a file
    """
    try:
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "size": os.path.getsize(file_path),
            "extension": os.path.splitext(file_path)[1].lower()
        }
    except Exception as e:
        logging.error(f"Error getting file info: {str(e)}")
        raise e

def delete_file(file_path: str) -> bool:
    """
    Delete a file from disk
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"File deleted: {file_path}")
            return True
        else:
            logging.warning(f"File not found for deletion: {file_path}")
            return False
    except Exception as e:
        logging.error(f"Error deleting file: {str(e)}")
        return False

def copy_file(source_path: str, dest_path: str) -> Optional[str]:
    """
    Copy a file to a new location
    """
    try:
        if os.path.exists(source_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(source_path, dest_path)
            logging.info(f"File copied from {source_path} to {dest_path}")
            return dest_path
        else:
            logging.warning(f"Source file not found for copying: {source_path}")
            return None
    except Exception as e:
        logging.error(f"Error copying file: {str(e)}")
        return None

def list_files(directory: str = UPLOAD_DIR, filter_ext: list = None) -> list:
    """
    List files in a directory, optionally filtered by extension
    """
    try:
        files = []
        if not os.path.exists(directory):
            return files
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                if filter_ext is None or any(filename.endswith(ext) for ext in filter_ext):
                    files.append(get_file_info(file_path))
        
        return files
    except Exception as e:
        logging.error(f"Error listing files: {str(e)}")
        return []