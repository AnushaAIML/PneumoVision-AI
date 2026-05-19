import torch
import cv2
import numpy as np


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):

        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def generate(self, input_image, class_idx):

        self.model.zero_grad()

        output = self.model(input_image)

        loss = output[:, class_idx]

        loss.backward()

        gradients = self.gradients[0].cpu().detach().numpy()
        activations = self.activations[0].cpu().detach().numpy()

        # Average gradients spatially
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(
            activations.shape[1:],
            dtype=np.float32
        )

        # Weighted combination
        for i, w in enumerate(weights):

            cam += w * activations[i]

        # ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        cam = cam - np.min(cam)

        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        # Resize
        cam = cv2.resize(cam, (224, 224))

        return cam