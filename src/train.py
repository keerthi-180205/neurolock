import torch
import torch.nn as nn

from datasets import train_loader, val_loader, class_to_idx
from model import FaceCNN

num_classes = len(class_to_idx)

model = FaceCNN(num_classes)

# loss function
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

for epoch in range(epochs):

    total_epoch_loss = 0

    for batch_features, batch_labels in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_features)

        loss = criterion(outputs, batch_labels)

        loss.backward()

        optimizer.step()

        total_epoch_loss = total_epoch_loss + loss.item()

    avg_loss = total_epoch_loss/len(train_loader)
    print(f'Epoch: {epoch + 1} , Loss: {avg_loss}')


    #validaion loop
    model.eval()

    total_val_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():

        for batch_features, batch_labels in val_loader:

            outputs = model(batch_features)

            loss = criterion(outputs, batch_labels)

            total_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += batch_labels.shape[0]

            correct += (predicted == batch_labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)

    val_accuracy = correct / total


    print(
        f"Epoch: {epoch + 1}, "
        f"Train Loss: {avg_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"Val Accuracy: {val_accuracy:.4f}"
    )