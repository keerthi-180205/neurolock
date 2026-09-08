import os
import random
import shutil
import logging
import torch

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_splitter')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
logger.setLevel('DEBUG')

file_path = os.path.join(log_dir, 'data_splitter.log')
file_handler = logging.FileHandler(file_path)
logger.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# get the identties of each sample in the raw dataset
def get_identities(image_path):

    try:
        identities = []
        for folder in os.listdir(image_path):
            if os.path.isdir(os.path.join(image_path, folder)):
                identities.append(folder)
        logger.debug("Identities fetched successfully")
        return identities
    except FileNotFoundError:
        logger.error("Path doesn't exist")
        return None
    except Exception as e:
        logger.error("Unknown error : %s", e)
        return None

# split the data into train, val and test and store it in a processed folder
def split_data(identities, image_path, processed_path):
    for identity in identities:
        identity_path = os.path.join(image_path, identity)
        images = os.listdir(identity_path)

        random.shuffle(images)
        train_images = images[:24]
        val_images = images[24:27]
        test_images = images[27:]

        train_path = os.path.join(processed_path, "train", identity)
        val_path = os.path.join(processed_path, "val", identity)
        test_path = os.path.join(processed_path, "test", identity)

        os.makedirs(train_path,exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for image in train_images:
            source = os.path.join(identity_path, image)
            destination = os.path.join(train_path, image)
            shutil.copy2(source, destination)

        for image in val_images:
            source = os.path.join(identity_path, image)
            destination = os.path.join(val_path, image)
            shutil.copy2(source, destination)

        for image in test_images:
            source = os.path.join(identity_path, image)
            destination = os.path.join(test_path, image)
            shutil.copy2(source, destination)

def main():
    image_path = "/home/keerthi-180205/MY-PROJECTS/neurolock/data/raw"
    processed_path = "/home/keerthi-180205/MY-PROJECTS/neurolock/data/processed"
    identities = get_identities(image_path)
    split_data(identities, image_path, processed_path)
                

if __name__ == "__main__":
    main()

