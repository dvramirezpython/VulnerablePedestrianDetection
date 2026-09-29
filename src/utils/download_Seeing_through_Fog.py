# Download files from URLs and save it to a local path
import os
import json
import requests

def download_files(json_path, output_dir="downloads"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Validate structure
    # if not isinstance(data, list) or not data or "urls" not in data:
    #     raise ValueError("Expected JSON structure: {'urls': [ { 'key': ..., 'url': ... }, ... ] }")

    urls_list = data["urls"]

    for item in urls_list:
        key = item.get("key")
        url = item.get("url")

        if not key or not url:
            print(f"Skipping invalid entry: {item}")
            continue

        try:
            print(f"⬇️ Downloading {key} from {url} ...")
            response = requests.get(url, stream=True, timeout=15)
            response.raise_for_status()

            # Guess extension from URL
            # ext = os.path.splitext(url)[-1]
            ext = key.split('/')[-1]
            # filename = f"{key}{ext}"
            # filepath = os.path.join(output_dir, filename)
            # filename = f"{os.path.basename(key)}{ext}"   # just the file part
            filename = f'{os.path.basename(key).split('.')[0]}{ext}'   # just the file part
            subdir = os.path.join(output_dir, os.path.dirname(key))
            # Create subdirectories if needed
            os.makedirs(subdir, exist_ok=True)
            filepath = os.path.join(subdir, filename)

            # Save file
            with open(filepath, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f_out.write(chunk)

            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Failed to download {key} from {url}: {e}")

if __name__ == "__main__":
    download_files("Seeing_through_Fog.json", "/home/dvramirez/Descargas")
