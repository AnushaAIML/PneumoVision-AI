import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from src.dataset import XrayDataset
from src.model import get_model


def train(model, dataloader, epochs=3):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nUsing device: {device}\n")

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch_idx, (images, labels) in enumerate(dataloader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            # Show progress every 10 batches
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        print(
            f"\nEpoch {epoch+1}/{epochs} Completed "
            f"Total Loss: {total_loss:.4f}\n"
        )

    # Save trained model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["NORMAL", "PNEUMONIA"]
        },
        "model/model.pt"
    )

    print("\n✅ Model saved successfully!")


if __name__ == "__main__":

    dataset = XrayDataset(
        root_dir="dataset/train"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,   # smaller batch for CPU stability
        shuffle=True,
        num_workers=0
    )

    model = get_model()

    train(
        model,
        dataloader,
        epochs=3
    )