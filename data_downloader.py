# data_downloader.py
# Download celebrity images using Bing Image Downloader

from bing_image_downloader import downloader

downloader.download(
    query="Virat Kohli",
    limit=100,
    output_dir="data",        # IMPORTANT: use 'data' (same as feature_extractor)
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
    verbose=True
)
