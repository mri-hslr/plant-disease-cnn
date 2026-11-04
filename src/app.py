import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import pathlib
from PIL import Image

# Set up the page configuration
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image, and the CNN will predict the disease.")

# 1. Load the trained model and class names
# We wrap this in a cache so it only loads once, not on every button click
@st.cache_resource
def load_model():
    model_path = "outputs/custom_cnn/best.keras"
    model = tf.keras.models.load_model(model_path)
    
    # Load class names saved during training
    with open("outputs/custom_cnn/class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model. Did you train it yet? Error: {e}")
    st.stop()

# 2. Image Preprocessing (Must match train.py exactly)
def preprocess_image(image):
    # Convert PIL Image to numpy array
    img = np.array(image.convert("RGB"))
    
    # Resize using bicubic interpolation (same as train.py)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # Apply CLAHE to L channel of LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    # Gaussian Blur
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0)
    
    # Normalize for Custom CNN
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    return np.expand_dims(img, axis=0)

# 3. UI for Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf..."):
            # Preprocess and predict
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            
            # Get the highest probability
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            # Clean up the class name for display (e.g., Apple___Apple_scab -> Apple - Apple Scab)
            display_name = predicted_class.replace("___", " - ").replace("_", " ")
            
            # Show Results
            st.markdown("---")
            st.subheader("Results:")
            st.write(f"**Prediction:** {display_name}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # If confidence is low, show a warning
            if confidence < 60.0:
                st.warning("The model is not highly confident. This image might not be in the dataset or the symptoms are unclear.")