'''
For downloading images from the Google Drive. Credit to Peter Sushko (https://github.com/peter-sushko/photobench/tree/main/data/downloading_scripts)
'''

import os
import requests

def download_image_from_gdrive(url: str, image_id: str, download_folder: str, verbose: bool = True):
    """
    Downloads an image from a Google Drive URL.

    Parameters:
    - url (str): The Google Drive URL of the image.
    - image_id (str): The ID to use for the saved image file.
    - download_folder (str): The folder to save the downloaded image.
    - verbose (bool): Whether to print status messages.
    """
    status = 'Not Attempted'

    try:
        # Extract file ID from the Google Drive URL
        file_id = url.split('/d/')[1].split('/')[0]
    except IndexError:
        status = 'Failure: Invalid Google Drive URL'
        if verbose:
            print(image_id, status)
        return status

    # Create a direct download URL
    direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        response = requests.get(direct_download_url)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                file_extension = content_type.split('/')[1]
                filename = f"{image_id}.{file_extension}"
                filepath = os.path.join(download_folder, filename)

                with open(filepath, 'wb') as out_file:
                    out_file.write(response.content)

                status = "success"
            else:
                status = 'Failure: Not an image'
        else:
            status = f"Error {response.status_code}"
    except Exception as e:
        status = 'Failure: ' + str(e)

    if verbose:
        print(image_id, status)
    return status
