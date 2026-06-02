# 🎨 AI Art Classifier

[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)]()
[![UI](https://img.shields.io/badge/Gradio-FF7C00?style=flat&logo=gradio&logoColor=white)]()

A deep learning computer vision application designed to classify paintings and artwork into several distinct art movements and styles. 

🌍 **Live Demo:** [ai.thinkwiser.eu](https://ai.thinkwiser.eu) *(Coming Soon)*

---

## 📊 Dataset Reference

This project is trained and evaluated on the curated **WikiArt Art Movements/Styles** dataset:
* **Dataset Link:** [Kaggle - WikiArt Art Movements/Styles by sivarazadi](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles)
* **Source:** Scraped directly from `wikiart.org` mapping historically significant fine art collections across nuanced, complex stylistic categories.

---

## 🚀 Features

* **Multi-Model Architecture**: Seamlessly switch between three state-of-the-art base architectures:
  * `EfficientNetV2-S`
  * `ConvNeXt (Tiny)`
  * `DenseNet121`
* **"Heavy Armor" Classification Head**: Replaces the default classifier with a custom deep neural network head featuring Batch Normalization, `LeakyReLU`, and cascading Dropout layers for maximum feature extraction.
* **Focal Loss Training**: Trained using a custom Focal Loss function (`gamma=2.0, alpha=0.25`) to handle extreme class imbalances and force the model to focus on hard-to-distinguish art styles.
* **Interactive UI**: Powered by Gradio 6.14, featuring real-time single-image inference, confidence distribution charts, and a responsive design.
* **Hardware Agnostic**: Automatically detects and runs on NVIDIA GPUs (CUDA) if available, with a graceful, optimized fallback to CPU/Intel Integrated Graphics.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/neo802/art-classification.git](https://github.com/neo802/art-classification.git)
cd art-classification
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install gradio pandas numpy matplotlib seaborn scikit-learn Pillow
```

### 4. Setup Data & Weights
To run the app locally, you must provide the dataset and the trained model weights. Create two directories in the root folder:

- `datasets/`: Place your categorized images here (each subfolder name becomes a class label).
- `models/`: Place your trained PyTorch .pth files here.

Required Directory Structure:
```
art-classification/
├── mainAppCUDA.py         # Gradio application optimized for NVIDIA GPUs, with both single image test and full test set evaluation
├── mainAppUniversal.py    # Main Gradio application, only featuring single image test, by file, webcam or clipboard
├── datasets/              # (Required) Image dataset
│   ├── Impressionism/
│   ├── Cubism/
│   └── ... (7 classes)
├── models/                # (Required) Trained weights
│   ├── efficientnetv2_best.pth
│   ├── convnext_best.pth
│   └── densenet_best.pth
└── README.md
```

### 💻 Usage
Run the Gradio web server:
```bash
python mainAppCUDA.py
```
or
```bash
python mainAppUniversal.py
```
The application will boot up in your terminal and automatically open in your default web browser.

### 🧠 Architecture Deep-Dive
#### The Custom Deep Head
Standard transfer learning often suffers from "Gradient Shock" when attaching an untrained linear layer to a pre-trained base model. To combat this, all models in this repository are routed through a highly regularized, custom deep head before final classification:
```python
nn.Sequential(
    nn.Flatten(),
    nn.BatchNorm1d(in_features),
    nn.Linear(in_features, 512),
    nn.LeakyReLU(negative_slope=0.3),
    nn.BatchNorm1d(512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.LeakyReLU(negative_slope=0.3),
    nn.BatchNorm1d(256),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.LeakyReLU(negative_slope=0.3),
    nn.BatchNorm1d(128),
    nn.Linear(128, NUM_CLASSES)
)
```

#### Why Focal Loss?
Distinguishing between closely related art styles (e.g., High Renaissance vs. Mannerism) is notoriously difficult. Standard Cross-Entropy loss allows the model to get "lazy" by only learning easy classes. By implementing Focal Loss, the model dynamically scales down the loss for easy, well-classified examples and hyper-focuses on the nuanced, difficult styles.

### 📈 Future Roadmap
[x] Migrate from Keras/TensorFlow to PyTorch

[x] Migrate UI from Streamlit to Gradio 6.0

[x] Optimize for CPU fallback environments (Intel UHD Graphics)

[ ] Deploy the Gradio application to the Raspberry Pi

[ ] Deploy as a full web application to ai.thinkwiser.eu

### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
Developed by Neo802 as part of the Computer Vision course.