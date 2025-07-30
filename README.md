# 🧠 Emotion Detection in Text using RoBERTa

This project focuses on detecting emotions from text using a fine-tuned transformer-based model — DistilRoBERTa. It goes beyond traditional sentiment analysis by classifying text into **six emotion categories**: `joy`, `sadness`, `love`, `anger`, `fear`, and `surprise`.

---

## 🚀 Project Objectives

- Build an emotion detection model using `DistilRoBERTa`.
- Classify sentences into six emotion categories.
- Fine-tune using the `dair-ai/emotion` dataset.
- Evaluate and save the trained model.
- Deploy the model using a Streamlit web app.

---

## 📂 Dataset

- **Source:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (Hugging Face)
- **Classes:** `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`
- **Structure:**
  ```json
  {
    "text": "I'm feeling happy today!",
    "label": "joy"
  }
  ```

- **Splits:**
  - Train
  - Validation
  - Test

---

## 🔧 Preprocessing

- Tokenized using `distilroberta-base` tokenizer.
- Applied padding and truncation.
- Used Hugging Face's `map()` method to process the full dataset.

---

## 🧠 Model Architecture

- **Base Model:** `DistilRoBERTa` (from Hugging Face Transformers)
- **Final Layer:** Fully connected + Softmax classifier.
- **Training Framework:** Hugging Face `Trainer` API.
- **Output:** Logits → softmax → emotion label.

---

## ⚙️ Training Configuration

| Parameter         | Value               |
|------------------|---------------------|
| Epochs           | 3                   |
| Batch Size       | 16                  |
| Optimizer        | AdamW               |
| Loss Function    | CrossEntropyLoss    |
| Evaluation Metric| Accuracy            |

---

## 📈 Results

- Achieved good accuracy on the validation set.
- Model accurately detects emotions like:
  ```
  Input: "I just got promoted!"
  Output: joy
  ```

---

## 💾 Model Saving & Inference

- Model saved to: `./saved_model/`
- Tokenizer and configuration saved alongside.
- Inference example:
  ```python
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  import torch

  tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
  model = AutoModelForSequenceClassification.from_pretrained("./saved_model")

  text = "I miss you so much"
  inputs = tokenizer(text, return_tensors="pt")
  outputs = model(**inputs)
  logits = outputs.logits
  predicted_class = torch.argmax(logits, dim=1)
  ```

---

## 🌐 Streamlit Web App

- Interactive UI using `Streamlit`.
- Enter a sentence and view detected emotions with colored percentage bars.
- Model loaded from Google Drive using `gdown`.
- Emotion-color mapping for better visualization.

### ✅ Features:
- Color-coded emotion bars
- GPU/CPU compatibility
- Gradient percentage display
- Easy-to-use UI

---

## 📦 Installation

```bash
pip install streamlit transformers datasets scikit-learn torch gdown
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Sample Emotions

| Emotion   | Example Sentence                                  |
|-----------|---------------------------------------------------|
| Joy       | "I just got accepted into my dream college!"      |
| Sadness   | "I miss my dog so much, it hurts."                |
| Love      | "You mean the world to me."                       |
| Anger     | "I’m furious that they ignored my complaint."     |
| Fear      | "I'm scared to go on stage."                      |
| Surprise  | "Whoa! I didn't see that coming at all!"          |

---

## ⚠️ Challenges

- Ambiguity in sarcastic or mixed-emotion text
- Only six fixed emotion categories
- Domain-specific inputs may confuse the model

---

## 🔮 Future Work

- Add neutral, disgust, and complex emotions
- Fine-tune on healthcare, reviews, etc.
- Deploy as API or integrate with chatbots

---

## 📌 Conclusion

This project demonstrates a working implementation of emotion detection using transformer models. It classifies emotions more precisely than basic sentiment analysis and has potential applications in mental health, feedback systems, and intelligent assistants.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Hugging Face 🤗
- dair-ai/emotion dataset
- Streamlit community
