# Import Image from wand.image module

from PIL import Image as PILImage
from wand.image import COLORSPACE_TYPES
from wand.image import Image
import os
import io


def apply_gamma_filter(image_path, output_path, gamma_value=3.7):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with Image(filename=image_path) as img:
        # Apply gamma filter
        img.gamma(gamma_value)
        # Increase brightness
        img.modulate(brightness=195)
        img.colorspace = 'rgb'
        img.depth = 8
        # Convert wand image to PIL image
        blob = img.make_blob(format='PNG')
        pil_img = PILImage.open(io.BytesIO(blob)).convert('L')  # Ensure grayscale
        # Convert grayscale to RGB
        rgb_image = PILImage.merge('RGB', (pil_img, pil_img, pil_img))
        # Save the modified image
        rgb_image.save(output_path)
        print(f"Saved: {output_path}")


dest_dir = 'dataset/SeeingThroughFog/test_night_imags_gamma'
os.makedirs(dest_dir, exist_ok=True)
source_dir = 'dataset/SeeingThroughFog/test_night_imags'
for filename in os.listdir(source_dir):
    if filename.endswith('.tiff'):
        image_path = os.path.join(source_dir, filename)
        output_path = os.path.join(dest_dir, f'{os.path.splitext(filename)[0]}_gamma.png')
        apply_gamma_filter(image_path, output_path)

# Read image using Image function
# with Image(filename ="dataset/SeeingThroughFog/test_night_imags/2018-02-04_00-00-32_00000.tiff") as img:
#     img.gamma(3.7)
#     img.save(filename ="dataset/SeeingThroughFog/test_night_imags/2018-02-04_00-00-32_00000_gamma.tiff")