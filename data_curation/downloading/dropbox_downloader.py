'''
Dropbox Image Downloader

This module provides a function to download images from Dropbox given a URL. It extracts the direct download link 
from the Dropbox page and saves the image to the specified folder with the specified image ID.

Functions:
- extract_download_link(url): Extracts the direct download link from the Dropbox page.
- get_file_extension(content_type): Determines the file extension based on the content type.
- download_image_from_dropbox(url, image_id, download_folder, verbose=True): Downloads the image from Dropbox 
  and saves it to the specified folder with the specified image ID.
'''

import requests
from bs4 import BeautifulSoup
import os

def extract_download_link(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        meta_tag = soup.find('meta', property='og:image')
        if meta_tag:
            return meta_tag['content']
        # If above fails try to find image link in the content
        img_tag = soup.find('img')
        if img_tag and 'src' in img_tag.attrs:
            return img_tag['src']
    return None

def get_file_extension(content_type):
    if content_type == 'image/jpeg':
        return 'jpg'
    elif content_type == 'image/png':
        return 'png'
    else:
        return 'jpeg'

def download_image_from_dropbox(url, image_id, download_folder, verbose=True):
    status = 'Not Attempted'
    dl_url = extract_download_link(url)
    
    if dl_url:
        if '/static/metaserver/static/images/' in dl_url:
            status = "Skipped: folder icon"
            if verbose:
                print(image_id, status)
            return status
        
        try:
            response = requests.get(dl_url)
            content_type = response.headers.get('content-type', '')
            file_extension = get_file_extension(content_type)
            
            if response.status_code == 200 and file_extension:
                filename = f"{image_id}.{file_extension}"
                full_save_path = os.path.join(download_folder, filename)
                with open(full_save_path, 'wb') as file:
                    file.write(response.content)
                status = "success"
            else:
                status = f"Failed. Status code: {response.status_code}"
        except Exception as e:
            status = 'Failure: ' + str(e)
    else:
        status = "Failed to extract download link."
    
    if verbose:
        print(image_id, status)
    return status
