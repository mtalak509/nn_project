import streamlit as st
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
import torchutils as tu
from PIL import Image
import pandas as pd

# model_state_dict

from torchvision.models import MobileNet_V3_Small_Weights

image = st.sidebar.file_uploader("Загрузите фото вашей родинки", type='jpg')



try:
    
    # Importing model
    model = torchvision.models.mobilenet_v3_small(weights = None)
    model.classifier[3] = nn.Linear(1024,2)
    checkpoint = torch.load('models/model_2_params/skin_cancer_model_complete2.pth', weights_only=False)
    # st.write(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])

    # Prediction func
    def get_prediction(image, model, device='cpu') -> str:
        
        transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image = Image.open(image).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        model.eval()
        model.to(device)
        with torch.no_grad():
            res = model(image_tensor)
            _, predicted = torch.max(res, 1)
            predicted_label = predicted.item()

            predicted_class = checkpoint['idx_to_class'].get(predicted_label)

            probabilities = torch.nn.functional.softmax(res, dim=1)
            confidence = probabilities[0][predicted_label].item()

        return predicted_class, confidence
    
    # st.write(get_prediction(image, model))
        
    with st.spinner('Анализируем изображение...'):
        predicted_class, confidence = get_prediction(image, model)
    
    # Показываем результат
    if isinstance(predicted_class, str) and predicted_class.startswith("Error"):
        st.error(predicted_class)
    else:
        st.header('Предсказание модели на основе MobileNet_V3')
        st.success(f"**Результат анализа:** {predicted_class}")
        st.info(f"**Уверенность:** {confidence:.2%}")

    if image is not None:
        image = Image.open(image)
        st.image(image, caption="Загруженное изображение", use_container_width=True)
    else:
        st.info("Файл пока не загружен.")

    loss_df = pd.DataFrame({
        'train_loss': checkpoint['train_loss'],
        'valid_loss': checkpoint['valid_loss']
    })
    
    st.subheader("📈 История обучения")
    st.line_chart(loss_df)

    acc_df = pd.DataFrame({
        'train_accuracy': checkpoint['train_acc'],
        'valid_accuracy': checkpoint['valid_acc']
    })
    
    st.subheader("📈 ")
    st.line_chart(acc_df)

except Exception as e:
    st.header(e)


