pip install git+https://github.com/pytorch/vision.git

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = YourModel()  # Ganti dengan nama dan definisi model Anda
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define any preprocessing steps if necessary
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Make prediction
def predict(image, model):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        # Perform any postprocessing here
        prediction = outputs.argmax(1).item()
    return prediction

def main():
    st.title("Image Classifier")
    st.write("This is a simple image classifier.")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        model = load_model("best_model.pth")  # Ganti dengan lokasi model terbaik Anda
        prediction = predict(image, model)
        
        # Display prediction
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
