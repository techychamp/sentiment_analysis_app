import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
import gdown
import os

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File ID from your Google Drive link
FILE_ID = "1ab82yhOovCQThm5Feg2F4VKNOlAZSiGc"
OUTPUT_PATH = "./saved_model/model.safetensors"
model_dir = "./saved_model"

@st.cache_resource
def load_model():
    if not os.path.exists(OUTPUT_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, OUTPUT_PATH, quiet=True)
    return AutoModelForSequenceClassification.from_pretrained(model_dir)

model = load_model()
model.to(device)

# Emotion to color mapping
features = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
emotion_colors = {
    "joy": "#fddb3a",
    "anger": "#ff6347",
    "sadness": "#6495ed",
    "fear": "#800080",
    "surprise": "#ffa500",
    "love": "#ff69b4"
}

st.set_page_config(page_title="Emotion Analyzer", page_icon="🧠", layout="centered")
st.title("🧠 Emotion Detection from Text")
st.write("Enter a sentence and see how emotions are distributed!")

# User Input
user_input = st.text_area("Your text here:", placeholder="E.g., I'm feeling excited and happy!", height=150)

if st.button("Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():  
                outputs = model(**inputs)

            predicted_labels = [(features[i],j.item()) for i,j in enumerate(outputs.logits[0]) if j.item()>0]
            total = sum(map(lambda x: x[1], predicted_labels))
            sorted_outputs = sorted(predicted_labels, key=lambda x: x[1], reverse=True)

            st.subheader("Detected Emotions:")

            for item in sorted_outputs:
                emotion = item[0].lower()
                score = item[1]
                percent = int((score//total) * 100)
                color = emotion_colors.get(emotion, "#d3d3d3")  # default to light gray if unknown

                # Create gradient badge
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(90deg, {color} {percent}%, #f0f0f0 {percent}%);
                        border-radius: 20px;
                        padding: 10px 20px;
                        margin: 5px 0;
                        color: #000;
                        font-weight: 600;
                        font-size: 16px;
                        width: fit-content;
                        border: 1px solid #ccc;
                    ">
                        {emotion.capitalize()}: {percent:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("Please enter some text to analyze.")
