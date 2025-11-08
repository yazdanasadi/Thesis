import os
from urllib.parse import urlsplit

import requests


def download_url(url, root, filename=None, chunk_size=1024 * 1024):
	"""
	Minimal file downloader that mirrors torchvision.datasets.utils.download_url
	in spirit without requiring torchvision/Pillow. Provides just enough
	functionality for dataset bootstrap scripts.
	"""
	os.makedirs(root, exist_ok=True)

	if filename is None:
		filename = os.path.basename(urlsplit(url).path) or "downloaded_file"

	filepath = os.path.join(root, filename)
	if os.path.exists(filepath):
		return filepath

	print(f"Downloading {url} -> {filepath}")
	with requests.get(url, stream=True) as response:
		response.raise_for_status()
		with open(filepath, "wb") as fout:
			for chunk in response.iter_content(chunk_size=chunk_size):
				if chunk:
					fout.write(chunk)

	return filepath
