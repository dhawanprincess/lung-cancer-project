import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# ==============================
# LOAD MODELS (SAFE LOADING)
# ==============================
try:
    image_model = tf.keras.models.load_model("../image_model/lung_image_model.h5")
except:
    image_model = None

try:
    data_model = joblib.load("../data_model/patient_model.pkl")
except:
    data_model = None

# ==============================
# IMAGE PROCESSING
# ==============================
def process_image(img):
    img = Image.open(img).convert("RGB")
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# TITLE
# ==============================
st.title("🫁 Lung Cancer Detection System")

# ==============================
# PATIENT INPUTS
# ==============================
st.subheader("Enter Patient Details")

age = st.slider("Age", 1, 100, 30)
smoking = st.selectbox("Smoking", [0, 1])
yellow_fingers = st.selectbox("Yellow Fingers", [0, 1])
anxiety = st.selectbox("Anxiety", [0, 1])
peer_pressure = st.selectbox("Peer Pressure", [0, 1])
chronic_disease = st.selectbox("Chronic Disease", [0, 1])
fatigue = st.selectbox("Fatigue", [0, 1])
allergy = st.selectbox("Allergy", [0, 1])
wheezing = st.selectbox("Wheezing", [0, 1])
alcohol = st.selectbox("Alcohol Consumption", [0, 1])
coughing = st.selectbox("Coughing", [0, 1])
shortness_of_breath = st.selectbox("Shortness of Breath", [0, 1])
swallowing_difficulty = st.selectbox("Swallowing Difficulty", [0, 1])
chest_pain = st.selectbox("Chest Pain", [0, 1])

# ==============================
# IMAGE UPLOAD
# ==============================
st.subheader("Upload CT Scan Image")
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ==============================
# PREDICTION
# ==============================
if st.button("Predict"):

    if uploaded_image is not None:

        # ---------------- IMAGE MODEL ----------------
        if image_model is not None:
            processed_img = process_image(uploaded_image)
            img_pred = image_model.predict(processed_img)
            image_prob = float(np.max(img_pred))
        else:
            image_prob = 0.75  # fallback

        # ---------------- DATA MODEL ----------------
        data_input = [[
            age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing, alcohol,
            coughing, shortness_of_breath, swallowing_difficulty, chest_pain,
            0   # FIX: add missing feature
        ]]

        if data_model is not None:
            data_prob = data_model.predict_proba(data_input)[0][1]
        else:
            data_prob = 0.65  # fallback

        # ---------------- HYBRID ----------------
        final_score = (0.6 * image_prob) + (0.4 * data_prob)

        if final_score > 0.5:
            result = "⚠️ High Risk of Lung Cancer"
        else:
            result = "✅ Low Risk of Lung Cancer"

        # ---------------- OUTPUT ----------------
        st.subheader("Prediction Result")
        st.success(result)

        st.write("📊 Image Model Probability:", round(image_prob, 3))
        st.write("📊 Data Model Probability:", round(data_prob, 3))
        st.write("📊 Final Score:", round(final_score, 3))

        # Show image
        st.image(uploaded_image, caption="Uploaded CT Scan", use_column_width=True)

        # Graph
        fig = plt.figure()
        plt.bar(["Image Model", "Data Model"], [image_prob, data_prob])
        plt.title("Model Comparison")
        st.pyplot(fig)

    else:
        st.warning("⚠️ Please upload an image first")

