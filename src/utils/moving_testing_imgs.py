'''
Move testing images from the current directory to a subdirectory.
Testing image filenames are listed in a txt file.
'''
import os
import shutil
import sys
def move_testing_images(txt_file, source_dir, dest_dir):
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Read the list of testing image filenames from the txt file
    with open(txt_file, 'r') as f:
        image_filenames = f.read().splitlines()

    # Move each image to the destination directory
    for filename in image_filenames:
        filename = f'{filename.replace(',', '_')}.tiff'
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            print(f'Moved: {source_path} -> {dest_path}')
        else:
            print(f'File not found: {source_path}')

txt_file = 'dataset/SeeingThroughFog/test_clear_night.txt'
source_dir = 'dataset/SeeingThroughFog/imags'
dest_dir = 'dataset/SeeingThroughFog/test_night_imags'

move_testing_images(txt_file, source_dir, dest_dir)