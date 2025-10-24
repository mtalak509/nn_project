import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import requests
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import os

# настройки страницы
st.set_page_config(
    page_title="Intel Image Classification",
    page_icon="🌄",
    layout="wide"
)

# классы модели
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# путь к модели
MODEL_PATH = 'models/model_1_params/intel_image_classifier_resnet50.pth'

# загрузка модели
@st.cache_resource
def load_model():
    """Загружает обученную модель"""
    try:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 6)

        # загружаем веса модели по правильному пути
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        st.info(f"Проверьте путь: {MODEL_PATH}")
        return None, None

# трансформации
def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# функция предсказания
def predict_image(model, image, transforms):
    """Предсказывает класс для изображения с измерением времени"""
    start_time = time.time()

    image_tensor = transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(outputs[0]).item()
        confidence = probabilities[predicted_class].item()

    inference_time = time.time() - start_time
    return predicted_class, confidence, probabilities, inference_time

# загрузка изображения по URL
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"Ошибка загрузки изображения: {e}")
        return None

# главная страница
def show_main_page():
    st.title("Intel Image Classification")
    st.markdown("---")

    # загружаем модель и данные
    model, checkpoint = load_model()

    if model is None:
        st.error("Модель не загружена. Убедитесь, что файл модели существует.")
        return

    # информация о процессе обучения
    st.header("Процесс обучения модели")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Архитектура модели")
        st.info("""
        - **Базовая модель:** ResNet50 (предобученная на ImageNet)
        - **Transfer Learning:** Заморожены все слои кроме последнего
        - **Классификатор:** Полносвязный слой с 6 выходами
        - **Входной размер:** 224×224×3
        - **Выходные классы:** 6
        """)

    with col2:
        st.subheader("Параметры обучения")
        st.info("""
        - **Эпохи:** 5
        - **Оптимизатор:** Adam (lr=0.001)
        - **Функция потерь:** CrossEntropyLoss
        - **Batch Size:** 32
        - **Время обучения:** ~10 минут на GPU
        - **Метрика:** Accuracy
        """)

    st.markdown("---")

    # графики обучения
    st.subheader("Графики обучения")

    # данные из чекпоинта
    if checkpoint and 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
        class_names = checkpoint.get('class_names', CLASS_NAMES)
    else:
        # примерные данные
        train_losses = [0.3730, 0.3720, 0.3713, 0.3727, 0.3674]
        val_losses = [0.3152, 0.2588, 0.2512, 0.2736, 0.2830]
        train_accuracies = [86.47, 86.60, 86.70, 86.55, 86.87]
        val_accuracies = [89.07, 91.27, 91.07, 90.43, 89.37]
        class_names = CLASS_NAMES

    # создаем графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # график Loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'o-', linewidth=2, markersize=8, label='Train Loss')
    ax1.plot(epochs, val_losses, 's-', linewidth=2, markersize=8, label='Validation Loss')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # график Accuracy
    ax2.plot(epochs, train_accuracies, 'o-', linewidth=2, markersize=8, label='Train Accuracy')
    ax2.plot(epochs, val_accuracies, 's-', linewidth=2, markersize=8, label='Validation Accuracy')
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)

    # метрики обучения
    st.subheader("Метрики обучения")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Лучшая точность", f"{max(val_accuracies):.2f}%")
    with col2:
        st.metric("Финальная точность", f"{val_accuracies[-1]:.2f}%")
    with col3:
        st.metric("Финальный Loss", f"{val_losses[-1]:.4f}")
    with col4:
        st.metric("Время обучения", "10 минут")

    st.markdown("---")

    # # F1 Score и Confusion Matrix
    # st.subheader("🎯 Дополнительные метрики")

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.subheader("F1 Score по классам")
    #     # Примерные значения F1 Score (можно заменить реальными)
    #     f1_scores = {
    #         'buildings': 0.87,
    #         'forest': 0.92,
    #         'glacier': 0.89,
    #         'mountain': 0.85,
    #         'sea': 0.91,
    #         'street': 0.88
    #     }

    #     for class_name, score in f1_scores.items():
    #         st.progress(score, text=f"{class_name}: {score:.3f}")

    # with col2:
    #     st.subheader("Confusion Matrix")
    #     # Примерная confusion matrix
    #     confusion_matrix = np.array([
    #         [420, 15, 8, 12, 5, 10],
    #         [10, 450, 5, 8, 12, 5],
    #         [8, 6, 430, 15, 20, 6],
    #         [15, 10, 12, 400, 8, 15],
    #         [5, 15, 18, 6, 435, 6],
    #         [12, 8, 5, 18, 7, 430]
    #     ])

    #     fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    #     sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
    #                xticklabels=class_names, yticklabels=class_names)
    #     ax_cm.set_title('Confusion Matrix', fontweight='bold')
    #     ax_cm.set_xlabel('Predicted')
    #     ax_cm.set_ylabel('Actual')
    #     st.pyplot(fig_cm)

    # st.markdown("---")

    # # Пример изображения из датасета
    # st.subheader("Пример изображения из датасета")

    # # Создаем пример изображения
    # fig_example, ax_example = plt.subplots(figsize=(8, 6))
    # demo_image = np.random.rand(150, 150, 3)
    # ax_example.imshow(demo_image)
    # ax_example.set_title('Пример: mountain', fontweight='bold')
    # ax_example.axis('off')
    # st.pyplot(fig_example)

    # st.info("Это пример изображения из тестового набора данных")

# Страница классификации
def show_classification_page():
    st.title("Классификация изображений")
    st.markdown("---")

    # Загружаем модель
    model, checkpoint = load_model()

    if model is None:
        st.error("Модель не загружена")
        return

    transforms = get_transforms()
    class_names = checkpoint.get('class_names', CLASS_NAMES) if checkpoint else CLASS_NAMES

    # Способы загрузки изображения
    option = st.radio("Выберите способ загрузки:",
                     ["Загрузить файл", "Ввести URL"])

    image = None

    if option == "Загрузить файл":
        uploaded_file = st.file_uploader("Выберите изображение",
                                       type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    else:  # URL
        url = st.text_input("Введите URL изображения:",
                           placeholder="https://example.com/image.jpg")
        if url:
            with st.spinner("Загружаем изображение..."):
                image = load_image_from_url(url)

    # предсказание
    if image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Изображение")
            st.image(image, use_column_width=True)

            # предсказание
            with st.spinner("Анализируем изображение..."):
                predicted_class, confidence, probabilities, inference_time = predict_image(
                    model, image, transforms
                )

        with col2:
            st.subheader("Результат классификации")

            # основное предсказание
            st.success(f"**Предсказанный класс:** {class_names[predicted_class]}")
            st.info(f"**Уверенность:** {confidence:.2%}")

            # время ответа модели
            st.metric("Время предсказания", f"{inference_time:.3f} секунд")

            # все вероятности
            st.subheader("Вероятности по классам:")
            for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                progress = int(prob * 100)
                if i == predicted_class:
                    st.markdown(f"**{class_name}:** {prob:.4f} ({progress}%)")
                    st.progress(progress / 100)
                else:
                    st.write(f"• {class_name}: {prob:.4f} ({progress}%)")

            # визуализация вероятностей
            st.subheader("Визуализация вероятностей")
            prob_df = pd.DataFrame({
                'Class': class_names,
                'Probability': probabilities.numpy()
            })
            prob_df = prob_df.sort_values('Probability', ascending=False)

            fig_bar = px.bar(prob_df, x='Class', y='Probability',
                           color='Probability', color_continuous_scale='viridis')
            fig_bar.update_layout(title='Вероятности по классам', showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

# основное приложение
def main():
    st.sidebar.title("Навигация")
    page = st.sidebar.radio("Выберите страницу:",
                           ["Главная", "Классификация изображений"])

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **О приложении:**
    - Классификация изображений Intel
    - 6 классов: buildings, forest, glacier, mountain, sea, street
    - Модель: ResNet50 с Transfer Learning
    - Точность: ~90%
    - Путь к модели: {}
    """.format(MODEL_PATH))

    if page == "Главная":
        show_main_page()
    else:
        show_classification_page()

if __name__ == "__main__":
    main()