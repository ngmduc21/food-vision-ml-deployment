import os
import json
import requests
import SessionState
import streamlit as st
import tensorflow as tf
from utils import load_and_prep_image, classes_and_models, update_logger, predict_json

# Add environment credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "nt114-ml-deploy-0548f8fef41b.json" 
PROJECT = "nt114-ml-deploy" 
REGION = "asia-southeast1" 

### Streamlit code  ###
st.title("Website nhận diện thức ăn online")
st.header("Đó là món ăn gì? thử ngay!")

@st.cache # đưa function vào cache để prediction không tự lặp lại (streamlit refresh sau mỗi lần click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image = load_and_prep_image(image)
    # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    # image = tf.expand_dims(image, axis=0)
    preds = predict_json(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    return image, pred_class, pred_conf

# Chọn version của model
choose_model = st.sidebar.selectbox(
    "Chọn mô hình bạn muốn sử dụng",
    ("Model 1 (10 food classes)", # original 10 classes
     "Model 2 (11 food classes)", # original 10 classes + donuts
     "Model 3 (11 food classes + non-food class)") # 11 classes (same as above) + not_food class
)

# Model choice logic
if choose_model == "Model 1 (10 food classes)":
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]
elif choose_model == "Model 2 (11 food classes)":
    CLASSES = classes_and_models["model_2"]["classes"]
    MODEL = classes_and_models["model_2"]["model_name"]
else:
    CLASSES = classes_and_models["model_3"]["classes"]
    MODEL = classes_and_models["model_3"]["model_name"]

# Display info about model and classes
if st.checkbox("Hiện các mô hình dự đoán"):
    st.write(f"Bạn đã chọn {MODEL}, đây là các loại thức ăn chúng tôi có thể dự đoán:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload ảnh thức ăn",
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Hãy upload ảnh thức ăn!")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Dự đoán!")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    st.write(f"Kết quả dự đoán: {session_state.pred_class}, \
               Độ chính xác: {session_state.pred_conf:.3f}")

    # Create feedback mechanism (building a data flywheel)
    session_state.feedback = st.selectbox(
        "Kết quả dự đoán có đúng không?",
        ("Chọn kết quả", "Có", "Không"))
    if session_state.feedback == "Lựa chọn":
        pass
    elif session_state.feedback == "Có":
        st.write("Cám ơn về phản hồi của bạn!")
        # Log prediction information to terminal (this could be stored in Big Query or something...)
        print(update_logger(image=session_state.image,
                            model_used=MODEL,
                            pred_class=session_state.pred_class,
                            pred_conf=session_state.pred_conf,
                            correct=True))
    elif session_state.feedback == "Không":
        session_state.correct_class = st.text_input("Hãy cho chúng tôi biết đây là món gì?")
        if session_state.correct_class:
            st.write("Cám ơn! Kết quả của bạn sẽ giúp chúng tôi cải thiện độ chính xác")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=session_state.image,
                                model_used=MODEL,
                                pred_class=session_state.pred_class,
                                pred_conf=session_state.pred_conf,
                                correct=False,
                                user_label=session_state.correct_class))

