from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms  # type: ignore

# Гнучкий імпорт Grad-CAM (Part 4)
try:
    from pytorch_grad_cam import GradCAM  # type: ignore
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore
except ImportError:
    GradCAM = None


def predict_image(
    model: torch.nn.Module, img: Image.Image
) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
    """Обробка зображення та прогноз. Повертає (idx, confidence, tensor, all_probs)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        probs = F.softmax(output, dim=1)[0]  # Весь вектор ймовірностей
        conf, pred = torch.max(probs, 0)

    return int(pred.item()), float(conf.item()), img_t, probs


def run_gradcam(
    model: torch.nn.Module, img_tensor: torch.Tensor
) -> Optional[np.ndarray]:
    """Візуалізація Grad-CAM (Part 4)."""
    if GradCAM is None:
        return None

    # Шукаємо останній шар згортки для підсвічування
    target_layers = [
        module for module in model.modules() if isinstance(module, torch.nn.Conv2d)
    ][-1:]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor)[0, :]

    # Де-нормалізація для коректного відображення фону
    rgb_img = img_tensor[0].numpy().transpose(1, 2, 0)
    rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array(
        [0.485, 0.456, 0.406]
    )
    rgb_img = np.clip(rgb_img, 0, 1)

    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
