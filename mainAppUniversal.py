import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gradio as gr
from functools import lru_cache

print("\n" + "="*50)
print("🚀 INITIALIZING AI ART CLASSIFIER FOR ALL DEVICES")
print("="*50)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
DATA_DIR = 'datasets'
MODELS_DIR = 'models'

print(f"[1/4] Checking dataset directory: '{DATA_DIR}'...")
def get_class_names(data_dir):
    if os.path.exists(data_dir):
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        return classes
    return []

CLASSES = get_class_names(DATA_DIR)
NUM_CLASSES = len(CLASSES)
print(f"      ✅ Found {NUM_CLASSES} art styles.")

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"[2/4] Detecting Hardware Infrastructure...")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
HARDWARE_DISPLAY = "🟢 NVIDIA GPU (CUDA Active)" if DEVICE_STR == "cuda" else "🟡 CPU Fallback (Intel Graphics / standard RAM)"
print(f"      ✅ Hardware set to: {HARDWARE_DISPLAY}")

# ==========================================
# 2. MODEL CACHING 
# ==========================================
@lru_cache(maxsize=3)
def load_model(model_choice):
    print(f"\n[SYSTEM] Loading {model_choice} into RAM. This may take a moment on CPU...")
    start_time = time.time()
    device = torch.device(DEVICE_STR)
    
    def build_heavy_head(in_features, num_classes):
        return nn.Sequential(
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
            nn.Linear(128, num_classes)
        )

    if model_choice == "EfficientNetV2":
        model = models.efficientnet_v2_s(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = build_heavy_head(num_features, NUM_CLASSES)
        weights_path = 'models/efficientnetv2_best.pth'
        
    elif model_choice == "ConvNeXt (Tiny)":
        model = models.convnext_tiny(weights=None)
        num_features = model.classifier[2].in_features
        model.classifier = build_heavy_head(num_features, NUM_CLASSES)
        weights_path = 'models/convnext_best.pth'
        
    elif model_choice == "DenseNet121":
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = build_heavy_head(num_features, NUM_CLASSES)
        weights_path = 'models/densenet_best.pth'

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"[SYSTEM] ✅ {model_choice} loaded successfully in {time.time() - start_time:.2f} seconds!")
        return model, device, True
    except FileNotFoundError:
        print(f"[SYSTEM] ❌ Error: Could not find weights at {weights_path}")
        return None, device, False

def predict_image(model, image_tensor, device):
    with torch.no_grad():
        inputs = image_tensor.unsqueeze(0).to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
    return probs

print(f"[3/4] Compiling Gradio Interface...")
# ==========================================
# 3. GRADIO UI EVENT HANDLERS (BUG FIXED)
# ==========================================
def process_single_image(image, selected_model, progress=gr.Progress()):
    if image is None:
        # Return empty strings and None to clear the UI
        return ["", "", None] * 3
        
    progress(0.1, desc="Processing Image...")
    image_tensor = val_transform(image)
    models_to_run = ["EfficientNetV2", "ConvNeXt (Tiny)", "DenseNet121"] if selected_model == "Compare All" else [selected_model]
    
    outputs = []
    for i in range(3):
        if i < len(models_to_run):
            m_name = models_to_run[i]
            progress(0.3 + (i*0.2), desc=f"Loading & Running {m_name}...")
            model, device, loaded = load_model(m_name)
            
            if not loaded:
                outputs.extend([f"### **{m_name}**", f"⚠️ Weights missing", None])
                continue
                
            probs = predict_image(model, image_tensor, device)
            top_idx = probs.argmax()
            success_text = f"✅ **{CLASSES[top_idx].title()}** ({probs[top_idx] * 100:.1f}%)"
            
            df_probs = pd.DataFrame({"Style": [c.title() for c in CLASSES], "Probability": probs * 100})
            
            # Directly return the raw data, no buggy gr.update()
            outputs.extend([f"### **{m_name}**", success_text, df_probs])
        else:
            outputs.extend(["", "", None])
            
    progress(1.0, desc="Done!")
    return outputs

# ==========================================
# 4. GRADIO APP LAYOUT (BULLETPROOF STATIC LAYOUT)
# ==========================================
with gr.Blocks(title="AI Art Style Classifier") as demo:
    gr.Markdown("# 🎨 AI Art Style Classifier")
    
    with gr.Row():
        gr.Info(f"**System Status:** {HARDWARE_DISPLAY} | **Classes Loaded:** {NUM_CLASSES}")
    
    if NUM_CLASSES == 0:
        gr.Warning("⚠️ No dataset classes found. Please ensure your images are inside subfolders in the `datasets/` directory.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            selected_model = gr.Dropdown(choices=["EfficientNetV2", "ConvNeXt (Tiny)", "DenseNet121", "Compare All"], value="EfficientNetV2", label="Active Model:")
            gr.Markdown("---")
            uploaded_file = gr.Image(type="pil", label="Upload an Art Piece...")
            
        with gr.Column(scale=3):
            gr.Markdown("### 🖼️ Prediction Results")
            with gr.Row():
                # Removed all "visible=False" logic. Columns are static.
                headers, labels, plots = [], [], []
                for i in range(3):
                    with gr.Column():
                        header = gr.Markdown("")
                        label = gr.Markdown("")
                        plot = gr.BarPlot(x="Style", y="Probability", tooltip=["Style", "Probability"], height=250)
                        
                        headers.append(header)
                        labels.append(label)
                        plots.append(plot)

            input_components = [uploaded_file, selected_model]
            output_components = []
            for i in range(3):
                output_components.extend([headers[i], labels[i], plots[i]])
                
            uploaded_file.change(fn=process_single_image, inputs=input_components, outputs=output_components)
            selected_model.change(fn=process_single_image, inputs=input_components, outputs=output_components)

print(f"[4/4] Launching Web Server...")
print("="*50)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", inbrowser=True)