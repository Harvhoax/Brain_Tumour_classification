🧠 Brain MRI Tumor Classification Dashboard



🚀 A highly interactive, visually stunning Streamlit app that uses deep learning models (Custom CNN & MobileNetV2) to classify brain MRI scans into:

Glioma

Meningioma

Pituitary

No Tumor

It features smooth animated bar chart races, sparkline trend history, and a gorgeous abstract aesthetic UI.

✨ Features

✅ Dual Model PredictionsClassifies MRI scans using both a custom CNN and a transfer-learned MobileNetV2 model for robust comparison.

✅ Smooth Animated VisualizationsBar chart race animations with easing, interactive tooltips, and dynamic legends.

✅ Sparkline History TrackingTracks prediction probabilities across multiple uploaded scans.

✅ Streaming Loader AnimationSimulates an AI pipeline actively analyzing MRI scans for immersive UX.

✅ Fully Responsive & BeautifulGradient abstract UI that adapts perfectly from desktops to mobiles.

📂 Dataset & Model Training

📚 Dataset

We used a publicly available brain MRI dataset consisting of four classes:

Glioma

Meningioma

Pituitary

No Tumor

⚠️ Typically sourced from datasets like Kaggle Brain Tumor MRI Dataset (or similar).Each category contained T1-weighted MRI slices pre-labeled into folders.

🏗️ How the models were trained

🧬 1️⃣ Custom CNN

Built from scratch using TensorFlow/Keras.

Architecture included:

Convolutional layers with ReLU activations.

MaxPooling layers to downsample.

Batch Normalization for faster convergence.

Dense layers with softmax for multiclass output.

Trained for 25 epochs with Adam optimizer, using categorical_crossentropy.

🪄 2️⃣ Transfer Learning: MobileNetV2

Loaded a pre-trained MobileNetV2 (ImageNet weights).

Froze the base layers initially.

Added custom Dense + Dropout head for 4-class classification.

Later unfroze and fine-tuned last few layers with a low learning rate.

⚙️ Common preprocessing

All images resized to 224x224.

Pixel values normalized (/255.0).

Applied ImageDataGenerator for augmentation: rotations, zooms, flips.

🚀 How It Works

Upload a brain MRI image (jpg, jpeg, png).

The app:

Preprocesses the image (resize & normalize).

Feeds it to both the Custom CNN & MobileNetV2 models.

Shows prediction probabilities with:

🎯 Smooth animated bar chart race (custom easing).

📈 Sparkline trends over multiple uploads.

Highlights the top predicted class for each model.

🛠️ Tech Stack

Layer

Details

Frontend

Streamlit

Visualization

Plotly (animated bars & sparklines)

Deep Learning

TensorFlow / Keras

Image Handling

PIL, NumPy

State Mgmt

Streamlit Session State

🔥 Running the app

Make sure your trained model files (custom_cnn_final2.h5, mobilenetv2_final2.h5) are in the project directory.Then run:

streamlit run app.py

❤️ Credits

Dataset: Brain MRI Images for Brain Tumor Detection

TensorFlow & Keras community for awesome tutorials.

Streamlit & Plotly teams for powerful visualization tools.

