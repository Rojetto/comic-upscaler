# Comic Upscaler

A script to upscale comic pages inside CBZ and CBR archives using waifu2x.

## Requirements
- Python 3.11

## Installation
Install dependencies using:
```sh
pip install -r requirements.txt -c constraints.txt
```

## Usage
To upscale a single comic archive or all archives in a directory:
```sh
python upscale_comics.py /path/to/comic.cbz
```
Replace `/path/to/comic.cbz` with a directory path to process all CBZ/CBR files in that directory.