import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import base64



with st.sidebar:
    st.image("logo.png", width=100) 
    st.title("Project Info")
    
    st.markdown("---") 
    
    st.markdown("### **Developed by:**")
    st.write("Ahmed Gehad (team leader)")
    st.write("Tarek Omran")
    
    st.markdown("---")
    
    st.markdown("### **Project:**")
    st.info("Pneumonia Detection System")
    
    st.markdown("### **Course:**")
    st.success("Work Based Project")
    
    st.markdown("---")
    st.write("This system uses a Deep Learning model (CNN) to detect Pneumonia from X-ray images with high accuracy.")
     



def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{encoded_string.decode()}");
            background-attachment: fixed;
            background-size: 100% 100%;
            background-repeat: no-repeat;
            background-blend-mode: normal;
            background-color: rgba(255, 255, 255, 0.2); 
        }}

        [data-testid="stSidebar"] {{
            background-color: #26274d !important;
        }}
        
        [data-testid="stSidebar"] h1 {{
            color: white !important;
        }}
        
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] span {{
            color: white !important;
        }}

        .stAlert {{
            background-color: rgb(255, 255, 255) !important; 
            color: #1e1e2f !important; 
            border: 3px solid #4cc9f0 !important;
            border-radius: 15px !important;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3) !important;
        }}
        
        .stAlert p {{
            font-weight: bold !important;
            font-size: 22px !important;
            color: #1e1e2f !important;
        }}

        h1 {{ color: #1e1e2f !important; }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local('OIP.webp')


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
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
    
    img_resized = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    
    prediction = model.predict(img_array)
    
    class_names = ['Normal', 'Pneumonia'] 
    result_idx = np.argmax(prediction[0])
    confidence = prediction[0][result_idx] * 100

    if class_names[result_idx] == 'Normal':
        st.success(f"Result: Normal (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"Result: Pneumonia (Confidence: {confidence:.2f}%)")
        
    if st.button("Show Grad-CAM Analysis"):
        with st.spinner('Generating Heatmap...'):
            try:
                heatmap = get_gradcam_heatmap(img_array, model, 'final_conv_layer')
                final_img = display_heatmap(image, heatmap)
                st.image(final_img, caption="Grad-CAM Visualization", use_column_width=True)
            except Exception as e:
                st.error(f"Error generating heatmap: {e}")