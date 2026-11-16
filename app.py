import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="AI Road Safety Assistant", page_icon="🛣️", layout="wide")

CLASS_NAMES = [
    "Speed limit (20 km/h)", "Speed limit (30 km/h)", "Speed limit (50 km/h)",
    "Speed limit (60 km/h)", "Speed limit (70 km/h)", "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)", "Speed limit (100 km/h)", "Speed limit (120 km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

ADVICE = {
    0: "Keep your speed at or below 20 km/h.",
    1: "Keep your speed at or below 30 km/h.",
    2: "Keep your speed at or below 50 km/h.",
    3: "Keep your speed at or below 60 km/h.",
    4: "Keep your speed at or below 70 km/h.",
    5: "Keep your speed at or below 80 km/h.",
    6: "The previous 80 km/h speed restriction has ended. Follow the next posted limit.",
    7: "Keep your speed at or below 100 km/h.",
    8: "Keep your speed at or below 120 km/h.",
    9: "Do not overtake other vehicles.",
    10: "Do not overtake vehicles covered by this restriction.",
    11: "Give way as required at the next intersection.",
    12: "You are travelling on a priority road. Continue carefully.",
    13: "Slow down and give way to traffic with priority.",
    14: "Come to a complete stop and proceed only when safe.",
    15: "Vehicles are prohibited from entering this road.",
    16: "Vehicles over 3.5 metric tons are prohibited.",
    17: "Do not enter this road from this direction.",
    18: "Take extra care and watch for hazards.",
    19: "Slow down and prepare for a left-hand curve.",
    20: "Slow down and prepare for a right-hand curve.",
    21: "Reduce speed and be prepared for two successive curves.",
    22: "Slow down and watch for an uneven road surface.",
    23: "Reduce speed; the road may be slippery.",
    24: "Be prepared for the road to become narrower on the right.",
    25: "Slow down and watch for road workers and temporary hazards.",
    26: "Watch the traffic signals and obey their instructions.",
    27: "Slow down and watch carefully for pedestrians.",
    28: "Slow down; children may be crossing.",
    29: "Watch for cyclists crossing the road.",
    30: "Watch for icy or snowy road conditions.",
    31: "Watch for wild animals crossing.",
    32: "The indicated restrictions have ended. Continue to obey other signs.",
    33: "Turn right ahead, when safe.",
    34: "Turn left ahead, when safe.",
    35: "Continue straight ahead.",
    36: "Continue straight or turn right.",
    37: "Continue straight or turn left.",
    38: "Keep to the right of the traffic island or obstruction.",
    39: "Keep to the left of the traffic island or obstruction.",
    40: "Follow the roundabout direction and give way as required.",
    41: "The no-passing restriction has ended.",
    42: "The heavy-vehicle no-passing restriction has ended."
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("road_sign_model.keras", compile=False)

st.title("🛣️ AI Road Safety Assistant")
st.write("Upload a road-sign photo to classify it with your trained 43-class CNN model.")
st.info("Educational prototype: always follow official road signs, local traffic laws, and road conditions.")

try:
    model = load_model()
except Exception as e:
    st.error("The trained model could not be loaded. Check that road_sign_model.keras is in the same GitHub folder as app.py.")
    st.exception(e)
    st.stop()

uploaded = st.file_uploader("Upload a road-sign image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Uploaded road-sign image", use_container_width=True)

        # This matches the notebook: RGB, resize to 32x32.
        # The model itself contains a Rescaling(1./255) layer.
        resized = image.resize((32, 32))
        array = np.asarray(resized, dtype=np.float32)
        batch = np.expand_dims(array, axis=0)

        with st.spinner("Analysing the road sign..."):
            probabilities = model.predict(batch, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        with right:
            st.subheader("🤖 AI detection result")
            st.success(CLASS_NAMES[predicted_index])
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            st.progress(max(0.0, min(1.0, confidence)))
            st.markdown("**Road safety guidance**")
            st.write(ADVICE.get(predicted_index, "Observe the sign and follow official traffic instructions."))

            st.markdown("**Top 3 predictions**")
            top_indices = np.argsort(probabilities)[-3:][::-1]
            for idx in top_indices:
                st.write(f"{CLASS_NAMES[int(idx)]}: {float(probabilities[idx]) * 100:.1f}%")
    except Exception as e:
        st.error("Could not analyse this image. Please try another clear JPG or PNG image.")
        st.exception(e)

with st.expander("About this project"):
    st.write("""
    - **Dataset:** German Traffic Sign Recognition Benchmark (GTSRB), loaded in the training notebook.
    - **Model:** Convolutional Neural Network (CNN) built with TensorFlow/Keras.
    - **Input:** RGB image resized to 32 × 32 pixels.
    - **Output:** One of 43 road-sign classes, confidence score, top-three predictions, and safety guidance.
    - **Database:** No separate database is used by this prototype; the trained model file stores learned parameters.
    """)
