'''
For downloading any generic url including Reddit. Credit to Peter Sushko (https://github.com/peter-sushko/photobench/tree/main/data/downloading_scripts)
'''

import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def download_any_image(url: str, image_id: str, download_folder: str, verbose: bool = True):
    '''
    For downloading imgur URLs that are classified as images.
    '''
    headers = {'User-Agent': 'Mozilla/5.0'}
    retry_strategy = Retry(
        total=2,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry
        backoff_factor=2  # Wait 2, 4, 8 seconds between retries
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    status = 'Not Attempted'

    try:
        response = http.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            filename = f"{image_id}.jpg"  # Adjust if necessary.
            filepath = os.path.join(download_folder, filename)
            with open(filepath, 'wb') as out_file:
                out_file.write(response.content)
            status = 'success'
        else:
            status = f"Error {response.status_code}"
    except Exception as e:
        status = 'Failure: ' + str(e)

    if url == 'unrecovered':
        status = 'URL wasnt recovered'
    if url == '':
        status = 'empty URL'
    if verbose:
        print(image_id, status)
    return status
