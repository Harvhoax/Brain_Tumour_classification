import streamlit as st
import numpy as np
import tensorflow as tf
import time
import plotly.graph_objects as go
from PIL import Image

# Download models from Google Drive if not present
def download_model(drive_id, output):
    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={drive_id}", output, quiet=False)
        st.success(f"✅ Downloaded {output}")

download_model("1hEK-4-qQPJGf9nR30NvoeZobC7KnyIrb", "custom_cnn_final2.h5")
download_model("eYfRsPzILz1FO1yBPatVyQIhVRShG3-Q", "mobilenetv2_final2.h5")

# Load your saved models
custom_model = tf.keras.models.load_model('custom_cnn_final2.h5')
transfer_model = tf.keras.models.load_model('mobilenetv2_final2.h5')

# Class names
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# Helper function
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Custom CSS for style
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e0c3fc, #8ec5fc);
        padding: 2rem;
        border-radius: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🧠✨ Brain MRI Tumor Classifier")
st.caption("**An animated brain MRI prediction dashboard.**")

uploaded_file = st.file_uploader("📤 Upload MRI Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded MRI Image", use_container_width=True)
    img_array = preprocess_image(img)

    # Streaming loader animation
    loading_placeholder = st.empty()
    for i in range(100):
        loading_placeholder.progress(i+1, text=f" AI analyzing MRI scan... {i+1}%")
        time.sleep(0.01)  # fast animation

    # Predict
    custom_preds = custom_model.predict(img_array)[0]
    transfer_preds = transfer_model.predict(img_array)[0]

    loading_placeholder.empty()

    # Animated Plotly bar chart
    st.markdown("### Animated Probability Comparison")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=class_names,
        y=[0]*4,
        name='Custom CNN',
        marker_color='#8e24aa'
    ))
    fig.add_trace(go.Bar(
        x=class_names,
        y=[0]*4,
        name='MobileNetV2',
        marker_color='#3949ab'
    ))

    fig.update_layout(
        barmode='group',
        yaxis=dict(range=[0, max(max(custom_preds), max(transfer_preds)) + 0.1]),
        title="Prediction Probability Comparison (Animated)"
    )

    # Animate the build-up
    frames = []
    for step in np.linspace(0, 1, 20):
        frames.append(go.Frame(
            data=[
                go.Bar(y=(custom_preds * step), x=class_names, marker_color='#8e24aa'),
                go.Bar(y=(transfer_preds * step), x=class_names, marker_color='#3949ab'),
            ]
        ))

    fig.frames = frames
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="▶️ Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 100, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 50}}])]
        )]
    )

    st.plotly_chart(fig, use_container_width=True)

    # Predictions summary
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎨 Custom CNN")
        for i, prob in enumerate(custom_preds):
            st.write(f"**{class_names[i]}:** `{prob*100:.2f}%`")
        st.success(f"🧾 Predicted: `{class_names[np.argmax(custom_preds)]}`")

    with col2:
        st.subheader("🖌️ MobileNetV2")
        for i, prob in enumerate(transfer_preds):
            st.write(f"**{class_names[i]}:** `{prob*100:.2f}%`")
        st.success(f"🧾 Predicted: `{class_names[np.argmax(transfer_preds)]}`")

st.markdown("</div>", unsafe_allow_html=True)
