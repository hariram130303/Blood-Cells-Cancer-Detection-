import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import io # Import io for in-memory image processing

# --- Configuration ---
IMG_SIZE = (128, 128)
MODEL_PATH_KERAS = 'blood_cells_cancer.keras' # Path to your .keras model file

# --- Load the Model ---
@st.cache_resource # Cache the model loading for efficiency
def load_my_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH_KERAS)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}. Make sure 'blood_cells_cancer.keras' is in the same directory.")
        st.stop() # Stop the app if model can't be loaded

model = load_my_model()

# --- Define Class Labels (Important!) ---
class_labels = [
    'Benign',
    '[Malignant] early Pre-B',
    '[Malignant] Pre-B',
    '[Malignant] Pro-B'
]

if len(class_labels) != model.output_shape[1]:
    st.warning(f"Mismatch: Model output classes ({model.output_shape[1]}) vs defined class_labels ({len(class_labels)}). Please verify `class_labels`.")

# --- Prediction Function ---
def predict_image(img_bytes, model, img_size):
    # Use BytesIO to load image from bytes directly, avoiding temporary file
    img = image.load_img(io.BytesIO(img_bytes), target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescale image (same as during training)

    predictions = model.predict(img_array)
    return predictions[0] # Return the probabilities for the single image

# --- Page Content Functions ---

def home_page():
    st.title("🔬 Blood Cell Cancer (ALL) Detector")
    st.write("Upload a microscopic blood cell image to classify if it's affected by Acute Lymphoblastic Leukemia (ALL) or healthy.")
    st.markdown("---")

    col1, col2 = st.columns([1, 2]) # 1:2 ratio, adjust as needed

    with col1:
        st.subheader("Upload Your Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    with col2:
        if uploaded_file is not None:
            st.subheader("Results")
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write("")
            st.write("Classifying...")

            predictions = predict_image(uploaded_file.read(), model, IMG_SIZE)

            predicted_class_index = np.argmax(predictions)
            predicted_class_name = class_labels[predicted_class_index]
            confidence = predictions[predicted_class_index] * 100

            st.success(f"Prediction: **{predicted_class_name}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

            st.markdown("---")
            st.subheader("Detailed Probabilities:")

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['skyblue' if i != predicted_class_index else 'lightcoral' for i in range(len(class_labels))]
            ax.bar(class_labels, predictions, color=colors)
            ax.set_ylabel("Probability")
            ax.set_title("Class Probabilities")
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        else:
            st.info("Please upload an image to get a prediction.")
            st.markdown(
                """
                ### How it Works:
                1.  **Upload an image** of a blood cell.
                2.  Our deep learning model will analyze it.
                3.  Get an instant prediction whether it's Benign or one of the Malignant ALL subtypes.
                """
            )

def about_model_page():
    st.title("🧠 About the ALL Detection Model")

    st.markdown("""
    This web application utilizes a Convolutional Neural Network (CNN) to classify microscopic blood cell images into different categories related to Acute Lymphoblastic Leukemia (ALL).

    ### Model Architecture:
    The model is built using TensorFlow/Keras and consists of several convolutional layers followed by pooling layers, and then dense (fully connected) layers for classification.
    It's designed to learn hierarchical features from the raw pixel data of blood cell images.

    ```python
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    ```
    * **Convolutional Layers:** Extract features like edges, textures, and patterns.
    * **Pooling Layers (MaxPooling):** Reduce spatial dimensions, making the model more robust to small variations.
    * **Flatten Layer:** Converts the 2D feature maps into a 1D vector.
    * **Dense Layers:** Standard neural network layers for final classification.
    * **Dropout Layer:** Helps prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
    * **Activation Functions (ReLU, Softmax):** ReLU introduces non-linearity, and Softmax provides probabilities for each class.

    ### Training Data:
    The model was trained on a dataset of blood cell images categorized as:
    * `Benign` (Healthy cells)
    * `[Malignant] early Pre-B`
    * `[Malignant] Pre-B`
    * `[Malignant] Pro-B`

    ### Limitations:
    * **Research Tool:** This application is for educational and demonstrative purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
    * **Image Quality:** The accuracy heavily depends on the quality and characteristics of the input images, which should be similar to the training data (microscopic blood cell images).
    * **Generalization:** While the model learns patterns, its performance on entirely new and unseen image types or different staining protocols might vary.

    For more detailed information or specific medical concerns, please consult a qualified healthcare professional.
    """)

def contact_page():
    st.title("✉️ Contact Us")

    st.markdown("""
    If you have any questions, feedback, or inquiries about this application or the underlying model, feel free to reach out.

    **Developer:** Hari Ram

    **Email:** [tmhariram@gmail.com](mailto:tmhariram@gmail.com)

    **GitHub (Optional):** [https://github.com/hariram130303](https://github.com/hariram130303)

    **LinkedIn (Optional):** [https://linkedin.com/in/hari-ram-thogata-madam](https://linkedin.com/in/hari-ram-thogata-madam)

    ---

    We appreciate your interest in the Blood Cell Cancer (ALL) Detector!
    """)

# --- Main App Logic ---

st.set_page_config(
    page_title="Blood Cell Cancer (ALL) Detector",
    page_icon="🔬",
    layout="wide" # Use wide layout for more space
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About the Model", "Contact"])

# Display selected page content
if page == "Home":
    home_page()
elif page == "About the Model":
    about_model_page()
elif page == "Contact":
    contact_page()

st.markdown("---")
st.caption("Developed by Hari Ram 💖. This tool is for informational purposes only and should not be used for medical diagnosis.")