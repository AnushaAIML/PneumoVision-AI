# 🩺 PneumoVision AI
### Explainable Pneumonia Detection from Chest X-rays using Deep Learning

An end-to-end AI-powered medical imaging system that detects Pneumonia from Chest X-ray images using Transfer Learning with ResNet18, FastAPI backend, Streamlit frontend, and GradCAM-based Explainable AI visualization.

---

# 🚀 Features

- ✅ Pneumonia Detection using Deep Learning
- ✅ Transfer Learning with ResNet18
- ✅ Chest X-ray Image Classification
- ✅ FastAPI Backend for Model Inference
- ✅ Streamlit Interactive Web UI
- ✅ GradCAM Heatmap Visualization
- ✅ Explainable AI (XAI)
- ✅ Real-time Prediction System
- ✅ Confidence Score Generation
- ✅ External Unseen Image Testing

---

# 🧠 Tech Stack

| Category | Technologies |
|---|---|
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Model Architecture | ResNet18 |
| Explainable AI | GradCAM |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |

---

# 📂 Project Structure

```bash
pneumonia-detection/
│
├── api/
│   └── main.py
│
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── inference.py
│   ├── gradcam.py
│   └── evaluate.py
│
├── ui/
│   └── app.py
│
├── model/
│   └── model.pt
│
├── dataset/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone <your-github-repo-link>
cd pneumonia-detection
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv myenv
```

---

## 3️⃣ Activate Environment

### Windows

```bash
myenv\Scripts\activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Model Training

```bash
python -m src.train
```

---

# 🧪 Model Evaluation

```bash
python -m src.evaluate
```

---

# 🚀 Run Backend API

```bash
python -m uvicorn api.main:app --reload
```

---

# 💻 Run Streamlit Frontend

```bash
streamlit run ui/app.py
```

---

# 🔥 GradCAM Explainability

GradCAM heatmaps are used to visualize the important lung regions influencing the model’s prediction, improving interpretability and transparency in medical AI systems.

---

# 📊 Example Output

- Prediction: PNEUMONIA
- Confidence Score: 0.9966
- GradCAM Heatmap Visualization

---

# 🎯 Future Improvements

- Improve model accuracy with larger datasets
- Add Docker deployment
- Deploy on HuggingFace/Render
- Add multi-disease detection
- Add DICOM support
- Add cloud inference pipeline

---

# 👩‍💻 Author

Anusha 

AI/ML Graduate 