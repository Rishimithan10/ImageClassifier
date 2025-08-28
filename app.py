import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

@st.cache_resource(show_spinner=False)
def load_cifar_model(path="CIFAR10.h5"):

    try:
        model = load_model(path)
        return model
    except Exception as e:
        # Try safer load
        try:
            model = load_model(path, compile=False)
            return model
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load model from {path}. "
                "If this is an older H5 file, convert to SavedModel format:\n"
                "  model = load_model('CIFAR10.h5')\n"
                "  model.save('saved_model_dir', save_format='tf')\n"
                "Then update the app to load the 'saved_model_dir' instead."
            ) from e2

def preprocess_pil_image(image: Image.Image) -> np.ndarray:

    img = image.convert("RGB").resize((32, 32))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape((1, 32, 32, 3))
    return arr

def predict_image(model, img_input: np.ndarray):
    preds = model.predict(img_input)
    prob = float(np.max(preds))
    idx = int(np.argmax(preds))
    return idx, prob, preds

def main():
    st.title("📸 CIFAR-10 Classifier — drag & drop image")
    st.write("Drop an image (jpg/png). Model expects 32×32 RGB input; the app will resize automatically.")

   
    st.sidebar.markdown("## Options")
    model_path = st.sidebar.text_input("Model path (.h5 or SavedModel dir)", value="CIFAR10.h5")
    show_probs = st.sidebar.checkbox("Show full probability vector", value=False)
    
 
    with st.spinner("Loading model..."):
        try:
            model = load_cifar_model(model_path)
        except RuntimeError as err:
            st.error(str(err))
            st.stop()

    
    uploaded = st.file_uploader("Drag & drop an image here (or click to browse)", type=["png","jpg","jpeg"], accept_multiple_files=False)

    if uploaded is not None:

        try:
            bytes_data = uploaded.read()
            img = Image.open(io.BytesIO(bytes_data))
        except Exception as e:
            st.error("Could not read the uploaded file as an image.")
            return

        st.subheader("Input Image")
        st.image(img, use_container_width=True)

        
        with st.spinner("Preprocessing & predicting..."):
            try:
                x = preprocess_pil_image(img)
                idx, prob, preds = predict_image(model, x)
                label = CLASS_NAMES[idx]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        st.markdown("### Prediction")
        st.write(f"**Label:** `{label}`")

        if show_probs and uploaded:
            probs_table = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
            st.table(probs_table)
        elif uploaded is None:
            st.error("Please Upload the image")
        
        st.download_button("Download preprocessed 32×32 image (PNG)", data=Image.fromarray((x[0]*255).astype(np.uint8)).tobytes(), file_name="preprocessed_32x32.png")

    else:
        st.info("Upload an image to get a prediction. Try photos of animals, planes, vehicles, etc.")


if __name__ == "__main__":
    main()
