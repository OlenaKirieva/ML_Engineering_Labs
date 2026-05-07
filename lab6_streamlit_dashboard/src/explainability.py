import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image  # type: ignore
from PIL import Image
from skimage.segmentation import mark_boundaries  # type: ignore

from src.inference import get_transform


def run_gradcam(model, img_tensor, original_size):
    """Покращений Grad-CAM із вибором оптимального шару для деталізації."""
    import cv2  # type: ignore
    from pytorch_grad_cam import GradCAM  # type: ignore
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore

    # ЛОГІКА ВИБОРУ ШАРУ:
    # Замість самого останнього (4-го) блоку, беремо 3-й.
    # Там роздільна здатність 8x8 замість 4x4 - це дасть більше деталей!
    if hasattr(model, "block3"):
        target_layers = [model.block3[-3]]  # Останній Conv2d у третьому блоці
    elif hasattr(model, "conv2"):
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
    rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array(
        [0.485, 0.456, 0.406]
    )
    rgb_img = np.clip(rgb_img, 0, 1)

    # Накладання
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Повертаємо результат з гарним ресайзом
    return cv2.resize(cam_image, original_size, interpolation=cv2.INTER_CUBIC)


# def run_gradcam(model, img_tensor, original_size):
#     from pytorch_grad_cam import GradCAM # type: ignore
#     from pytorch_grad_cam.utils.image import show_cam_on_image # type: ignore
#     import cv2 # type: ignore

#     target_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)][-1:]
#     cam = GradCAM(model=model, target_layers=target_layers)
#     grayscale_cam = cam(input_tensor=img_tensor)[0, :]

#     rgb_img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
#     rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
#     rgb_img = np.clip(rgb_img, 0, 1)

#     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#     return cv2.resize(cam_image, original_size)


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
        transform = get_transform()  # наш стандартний трансформ з Resize((32, 32))

        # Перетворюємо масиви сегментів назад у тензори 32x32
        batch = torch.stack([transform(Image.fromarray(i)) for i in images])

        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # 3. Генеруємо пояснення
    # Збільшуємо num_samples для кращої точності на великому зображенні
    explanation = explainer.explain_instance(
        img_array, predict_fn, top_labels=1, hide_color=0, num_samples=200
    )

    # 4. Створюємо маску (як у документації)
    # Поєднуємо оригінальне фото з підсвічуванням важливих зон
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False,  # Показуємо всю картинку, а не тільки блоки
    )

    # Малюємо чіткі межі зон, на які дивилася модель
    res_img = mark_boundaries(temp / 255.0, mask, color=(0, 1, 0), mode="outer")

    return res_img


# def run_lime(model, img_pil):
#     """Покращений LIME для низької роздільної здатності."""
#     from lime import lime_image
#     from skimage.segmentation import mark_boundaries
#     import numpy as np
#     import cv2

#     # 1. Готуємо зображення
#     img_32 = img_pil.resize((32, 32))
#     img_array = np.array(img_32)

#     explainer = lime_image.LimeImageExplainer()

#     def predict_fn(images):
#         from src.inference import get_transform
#         model.eval()
#         batch = torch.stack([get_transform()(Image.fromarray(i)) for i in images])
#         with torch.no_grad():
#             probs = torch.nn.functional.softmax(model(batch), dim=1)
#         return probs.cpu().numpy()

#     # 2. Генеруємо пояснення з фіксованою сегментацією
#     explanation = explainer.explain_instance(
#         img_array,
#         predict_fn,
#         top_labels=1,
#         hide_color=0,
#         num_samples=250, # Трохи більше для точності
#         # ВАЖЛИВО: сегментуємо картинку на 15 чітких блоків
#         segmentation_fn=lambda x: cv2.pyrMeanShiftFiltering(x, 10, 10).sum(axis=2) % 15
#     )

#     # 3. Виділяємо зони, які ПІДТВЕРДЖУЮТЬ прогноз
#     # hide_rest=True зробить все неважливе сірим
#     temp, mask = explanation.get_image_and_mask(
#         explanation.top_labels[0],
#         positive_only=True,
#         num_features=3,
#         hide_rest=True
#     )

#     # Малюємо межі
#     res_img = mark_boundaries(temp / 255.0, mask, color=(1, 1, 0), mode='thick')

#     # Повертаємо чіткий (не розмитий) результат
#     return cv2.resize(res_img, (256, 256), interpolation=cv2.INTER_NEAREST)

# def run_lime(model, img_pil):
#     """Оптимізований LIME для CIFAR-10."""
#     from lime import lime_image
#     from skimage.segmentation import mark_boundaries
#     import numpy as np

#     # КРОК 1: Примусовий ресайз до 32x32 (щоб LIME працював миттєво)
#     img_32 = img_pil.resize((32, 32))
#     img_array = np.array(img_32)

#     explainer = lime_image.LimeImageExplainer()

#     def predict_fn(images):
#         from src.inference import get_transform
#         model.eval()
#         # Тут images - це масиви numpy 32x32
#         batch = torch.stack([get_transform()(Image.fromarray(i)) for i in images])
#         with torch.no_grad():
#             probs = torch.nn.functional.softmax(model(batch), dim=1)
#         return probs.cpu().numpy()

#     # КРОК 2: Налаштування сегментації під 32x32
#     explanation = explainer.explain_instance(
#         img_array,
#         predict_fn,
#         top_labels=1,
#         hide_color=0,
#         num_samples=100, # Достатньо для такої малої картинки
#         batch_size=10
#     )

#     # КРОК 3: Візуалізація позитивних зон (що вплинуло на прогноз)
#     temp, mask = explanation.get_image_and_mask(
#         explanation.top_labels[0],
#         positive_only=True,
#         num_features=3, # Виділяємо 3 найважливіші зони
#         hide_rest=False,
#         min_weight=0.05
#     )

#     # Повертаємо результат, розтягнутий назад для краси (наприклад до 200x200)
#     import cv2
#     res_img = mark_boundaries(temp / 255.0, mask, color=(1, 1, 0), mode='thick')
#     return cv2.resize(res_img, (200, 200), interpolation=cv2.INTER_NEAREST)

# def run_lime(model, img_pil):
#     """Оптимізований LIME для швидкості та видимості."""
#     from lime import lime_image
#     from skimage.segmentation import mark_boundaries
#     from src.inference import get_transform
#     import numpy as np

#     explainer = lime_image.LimeImageExplainer()

#     def predict_fn(images):
#         model.eval()
#         batch = torch.stack([get_transform()(Image.fromarray(i)) for i in images])
#         with torch.no_grad():
#             probs = torch.nn.functional.softmax(model(batch), dim=1)
#         return probs.cpu().numpy()

#     explanation = explainer.explain_instance(
#         np.array(img_pil),
#         predict_fn,
#         top_labels=1,
#         hide_color=0,
#         num_samples=100 # Зменшено для швидкості
#     )

#     # positive_only=False дозволить бачити і те, що заважало прогнозу
#     temp, mask = explanation.get_image_and_mask(
#         explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
#     )

#     # Робимо межі жовтими та товстими
#     return mark_boundaries(temp / 255.0, mask, color=(1, 1, 0), mode='thick')

# def run_lime(model, img_pil):
#     """Пояснення за допомогою LIME."""
#     explainer = lime_image.LimeImageExplainer()
#     from src.inference import get_transform

#     # Внутрішня функція прогнозу для LIME
#     def predict_fn(images):
#         model.eval()
#         # Перетворюємо масиви numpy назад у PIL, потім у тензори
#         batch = torch.stack([get_transform()(Image.fromarray(i)) for i in images])
#         with torch.no_grad():
#             logits = model(batch)
#             probs = F.softmax(logits, dim=1)
#         return probs.cpu().numpy()

#     explanation = explainer.explain_instance(
#         np.array(img_pil),
#         predict_fn,
#         top_labels=1,
#         hide_color=0,
#         num_samples=50 # Прискорено для CPU
#     )

#     temp, mask = explanation.get_image_and_mask(
#         explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
#     )
#     return mark_boundaries(temp / 255.0, mask)

# import torch
# import numpy as np
# from lime import lime_image # type: ignore
# from skimage.segmentation import mark_boundaries # type: ignore
# from src.inference import get_transform

# def run_gradcam(model, img_tensor, target_layers):
#     # Код, який ми вже писали раніше
#     from pytorch_grad_cam import GradCAM # type: ignore
#     from pytorch_grad_cam.utils.image import show_cam_on_image # type: ignore

#     cam = GradCAM(model=model, target_layers=target_layers)
#     grayscale_cam = cam(input_tensor=img_tensor)[0, :]

#     rgb_img = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
#     rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
#     rgb_img = np.clip(rgb_img, 0, 1)

#     return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# def run_lime(model, img_pil):
#     """Пояснення за допомогою LIME (Part 4 [Task] Explainability)."""
#     explainer = lime_image.LimeImageExplainer()

#     # Функція-класифікатор для LIME
#     def batch_predict(images):
#         model.eval()
#         transform = get_transform()
#         batch = torch.stack([transform(img) for img in images])
#         logits = model(batch)
#         probs = torch.nn.functional.softmax(logits, dim=1)
#         return probs.detach().cpu().numpy()

#     # Генеруємо пояснення
#     explanation = explainer.explain_instance(
#         np.array(img_pil),
#         batch_predict,
#         top_labels=1,
#         hide_color=0,
#         num_samples=100 # Для швидкості
#     )

#     temp, mask = explanation.get_image_and_mask(
#         explanation.top_labels[0],
#         positive_only=True,
#         num_features=5,
#         hide_rest=False
#     )

#     return mark_boundaries(temp / 255.0, mask)

# def run_lime(model, img_pil, predict_fn):
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(
#         np.array(img_pil),
#         predict_fn,
#         top_labels=1,
#         hide_color=0,
#         num_samples= 50 #100
#     )
#     temp, mask = explanation.get_image_and_mask(
#         explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
#     )
#     return mark_boundaries(temp / 255.0, mask)
