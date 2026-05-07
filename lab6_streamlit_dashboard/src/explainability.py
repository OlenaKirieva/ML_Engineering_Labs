import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lime import lime_image # type: ignore
from skimage.segmentation import mark_boundaries # type: ignore
from src.inference import get_transform

from pytorch_grad_cam import GradCAM # type: ignore
from pytorch_grad_cam.utils.image import show_cam_on_image # type: ignore
import cv2 # type: ignore

def run_gradcam(model, img_tensor, original_size):
    """Покращений Grad-CAM із вибором оптимального шару для деталізації."""

    # ЛОГІКА ВИБОРУ ШАРУ:
    # Замість самого останнього (4-го) блоку, беремо 3-й.
    # Там роздільна здатність 8x8 замість 4x4 - це дасть більше деталей!
    if hasattr(model, 'block3'):
        target_layers = [model.block3[-3]] # Останній Conv2d у третьому блоці
    elif hasattr(model, 'conv2'):
        target_layers = [model.conv2]
    else:
        # Універсальний пошук передостаннього шару згортки
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        target_layers = [conv_layers[-2]] if len(conv_layers) > 1 else [conv_layers[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Генеруємо карту
    grayscale_cam = cam(input_tensor=img_tensor)[0, :]
    
    # Підготовка фону
    rgb_img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
    rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    rgb_img = np.clip(rgb_img, 0, 1)
    
    # Накладання
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Повертаємо результат з гарним ресайзом
    return cv2.resize(cam_image, original_size, interpolation=cv2.INTER_CUBIC)

def run_lime(model, img_pil):
    """
    Покращений LIME: сегментує високу роздільну здатність, 
    але прогнозує через 32x32 (Part 4 Explainability).
    """
    # 1. Залишаємо картинку у гарній якості для сегментації (наприклад 224x224)
    display_size = (224, 224)
    img_hd = img_pil.resize(display_size)
    img_array = np.array(img_hd)

    explainer = lime_image.LimeImageExplainer()

    # 2. Функція прогнозу, яка вміє стискати HD-сегменти до 32x32 для моделі
    def predict_fn(images):
        model.eval()
        transform = get_transform() # наш стандартний трансформ з Resize((32, 32))
        
        # Перетворюємо масиви сегментів назад у тензори 32x32
        batch = torch.stack([transform(Image.fromarray(i)) for i in images])
        
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # 3. Генеруємо пояснення
    # Збільшуємо num_samples для кращої точності на великому зображенні
    explanation = explainer.explain_instance(
        img_array, 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=200 
    )
    
    # 4. Створюємо маску (як у документації)
    # Поєднуємо оригінальне фото з підсвічуванням важливих зон
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False # Показуємо всю картинку, а не тільки блоки
    )
    
    # ПЕРЕТВОРЕННЯ: з float (0-1) у uint8 (0-255)
    res_img = (res_img * 255).astype(np.uint8)
    
    # Повертаємо чіткий результат
    return cv2.resize(res_img, (256, 256), interpolation=cv2.INTER_NEAREST)