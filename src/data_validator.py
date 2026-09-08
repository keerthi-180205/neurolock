import os
from PIL import Image
import logging
# import pytorch

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_validator')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
logger.setLevel('DEBUG')

file_path = os.path.join(log_dir, 'data_validator.log')
file_handler = logging.FileHandler(file_path)
logger.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# validates the data by checking the dicrectory path and the image extension
def validate_dataset(image_path):

    for folder in os.listdir(image_path):
        if os.path.isdir(os.path.join(image_path, folder)):
            images = os.listdir(os.path.join(image_path, folder))

            for image in images:
                full_image_path = os.path.join(image_path, folder, image)

                if not os.path.isfile(full_image_path):
                    logger.warning(
                        "Skipping non-file: %s",
                        full_image_path
                    )
                    continue

                _, extension = os.path.splitext(image)

                if extension.lower() not in [".jpg", ".jpeg", ".png"]:
                    logger.warning(
                    "Unsupported image format: %s",
                    full_image_path
                )
                    continue

                try:
                    img = Image.open(full_image_path)

                    logger.debug(
                        "Valid image: %s | Size: %s | Mode: %s | Format: %s",
                        full_image_path,
                        img.size,
                        img.mode,
                        img.format
                    )

                except Exception as e:
                    logger.error(
                        "Corrupted/unreadable image: %s | Error: %s",
                        full_image_path,
                        e
                    )

def main():
    image_path = "/home/keerthi-180205/MY-PROJECTS/neurolock/data/raw"
    validate_dataset(image_path)
    

if __name__ == "__main__":
    main()