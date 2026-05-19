import torch
import cv2
import numpy as np

from src.model import get_model
from src.gradcam import GradCAM


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load trained checkpoint
checkpoint = torch.load(
    "model/model.pt",
    map_location=device
)

# Build model
model = get_model()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()

# GradCAM target layer for ResNet18
target_layer = model.layer4[-1]

gradcam = GradCAM(
    model,
    target_layer
)


def preprocess(image):

    # Handle grayscale images
    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    img = cv2.resize(image, (224, 224))

    img = img / 255.0

    img = img.transpose(2, 0, 1)

    img = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0)

    return img.to(device)


def predict(image):

    # Keep original image for overlay
    original = cv2.resize(image, (224, 224))

    input_tensor = preprocess(image)

    outputs = model(input_tensor)

    probs = torch.softmax(outputs, dim=1)

    conf, pred = torch.max(probs, 1)

    # Generate GradCAM heatmap
    cam = gradcam.generate(
        input_tensor,
        pred.item()
    )

    # Convert heatmap to color
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    # Overlay heatmap
    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    return (
        pred.item(),
        conf.item(),
        overlay
    )