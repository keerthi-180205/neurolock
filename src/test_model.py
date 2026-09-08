from datasets import train_dataset, class_to_idx
from model import FaceCNN

num_classes = len(class_to_idx)

myModel = FaceCNN(num_classes)

image, label = train_dataset[5]

print(image.shape)
print(label)

image = image.unsqueeze(0)

print(image.shape)

output = myModel(image)

print(output.shape)