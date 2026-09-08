from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


# Converting the image labels into cnn understandabl format
class FaceDataset(Dataset):

    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = []

        for identity in sorted(os.listdir(self.root_dir)):

            identity_path = os.path.join(self.root_dir, identity)

            if os.path.isdir(identity_path):

                label = self.class_to_idx[identity]

                for image in os.listdir(identity_path):

                    image_path = os.path.join(identity_path, image)

                    self.samples.append([image_path, label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path, label = sample

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


train_dir = "data/processed/train"

identities = sorted(os.listdir(train_dir))

class_to_idx = {
    identity: index
    for index, identity in enumerate(identities)
    if os.path.isdir(os.path.join(train_dir, identity))
}


# transfomer 

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(size=(224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = FaceDataset(
    "data/processed/train",
    class_to_idx,
    train_transform
)

val_dataset = FaceDataset(
    "data/processed/val",
    class_to_idx,
    val_transform
)

test_dataset = FaceDataset(
    "data/processed/test",
    class_to_idx,
    val_transform
)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


