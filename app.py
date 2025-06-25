import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

def detect_objects(img):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    count_dict = {}
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            count_dict[label] = count_dict.get(label, 0) + 1
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img, count_dict

# Streamlit UI
st.set_page_config(page_title="YOLO Object Counter", layout="centered")
st.title("🔍 YOLO Object Detection & Counting")
st.sidebar.title("Choose Mode")

mode = st.sidebar.radio("Select input type:", ["Upload Image", "Webcam Live"])

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        result_img, obj_counts = detect_objects(img)

        st.subheader("📦 Object Detection Result")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB")

        if obj_counts:
            st.subheader("📊 Object Counts")
            for obj, count in obj_counts.items():
                st.write(f"**{obj.capitalize()}**: {count}")
        else:
            st.info("No objects detected.")

elif mode == "Webcam Live":
    st.warning("📷 Press 'Start' to open webcam (limited support in browsers).")

    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break
            result_img, obj_counts = detect_objects(frame)
            FRAME_WINDOW.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()
    else:
        st.info("☝️ Tick 'Start Webcam' to begin.")
