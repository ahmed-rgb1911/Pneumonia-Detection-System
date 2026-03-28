import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import base64

def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{encoded_string.decode()}");
        background-attachment: fixed;
        background-size: cover;
        background-blend-mode: overlay;
        background-color: rgba(0, 0, 0, 0.7); 
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

set_bg_from_local('download.webp')


@st.cache_resource
def load_and_activate_model():
    loaded_model = tf.keras.models.load_model('my_pneumonia_model.keras')
    loaded_model.predict(np.zeros((1, 128, 128, 3)))
    return loaded_model

model = load_and_activate_model()

def get_gradcam_heatmap(img_array, model, layer_name):
    with tf.GradientTape() as tape:
        x = img_array
        conv_outputs = None
        
        for layer in model.layers:
            x = layer(x)
            if layer.name == layer_name:
                conv_outputs = x
        
        predictions = x
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:
        heatmap /= max_val
        
    return heatmap.numpy()

def display_heatmap(original_image, heatmap):
    img = np.array(original_image.resize((128, 128)))
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

st.title("Pneumonia Detection System")
uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Original Image", use_column_width=True)
    
    img_resized = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        st.success(f"Result: Normal ({prediction[0][0]:.2f})")
    else:
        st.error(f"Result: Pneumonia ({1 - prediction[0][0]:.2f})")
        
    if st.button("Show Grad-CAM Analysis"):
        with st.spinner('Generating Heatmap...'):
            try:
                heatmap = get_gradcam_heatmap(img_array, model, 'final_conv_layer')
                final_img = display_heatmap(image, heatmap)
                st.image(final_img, caption="Grad-CAM Hotspot", use_column_width=True)
            except Exception as e:
                st.error(f"Error generating heatmap: {e}")
