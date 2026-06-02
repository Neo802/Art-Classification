import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg') # Ensures Matplotlib works safely in a web server environment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gradio as gr
from functools import lru_cache

print("\n" + "="*50)
print("🚀 INITIALIZING AI ART CLASSIFIER FOR CUDA")
print("="*50)

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
DATA_DIR = 'datasets'
MODELS_DIR = 'models'

print(f"[1/4] Checking dataset directory: '{DATA_DIR}'...")
# --- Helper Functions ---
def get_class_names(data_dir):
    """Scans the dataset directory to get class names and count."""
    if os.path.exists(data_dir):
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        return classes
    else:
        print(f"Dataset directory '{data_dir}' not found. Cannot infer class names.")
        return []

# Define globals dynamically from the unified dataset folder
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
# 2. MODEL & DATA CACHING (Replaced @st.cache with @lru_cache)
# ==========================================
@lru_cache(maxsize=3)
def load_model(model_choice):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Helper function to build your custom Heavy Head
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
        num_features = model.classifier[2].in_features # ConvNeXt has 768 features here
        model.classifier = build_heavy_head(num_features, NUM_CLASSES)
        weights_path = 'models/convnext_best.pth'
        
    elif model_choice == "DenseNet121":
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features # DenseNet has 1024 features here
        model.classifier = build_heavy_head(num_features, NUM_CLASSES)
        weights_path = 'models/densenet_best.pth'

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device, True
    except FileNotFoundError:
        return None, device, False

def predict_image(model, image_tensor, device):
    with torch.no_grad():
        inputs = image_tensor.unsqueeze(0).to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
    return probs

@lru_cache(maxsize=3)
def evaluate_on_test_set(model_choice):
    model, device, loaded = load_model(model_choice)
    if not loaded:
        return None, None, None
        
    # --- Programmatic Split Logic (80% Train, 10% Val, 10% Test) ---
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # We use a FIXED manual seed (42) to match Notebook holdouts exactly
    generator = torch.Generator().manual_seed(42)
    _, _, test_data = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100
    
    return accuracy, all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    plt.ylabel('Actual Style')
    plt.xlabel('Predicted Style')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

print(f"[3/4] Compiling Gradio Interface...")
# ==========================================
# 3. GRADIO UI EVENT HANDLERS
# ==========================================
def process_single_image(image, selected_model):
    if image is None:
        # Return all elements as invisible if no image
        return [gr.update(visible=False)] * 12
        
    image_tensor = val_transform(image)
    models_to_run = ["EfficientNetV2", "ConvNeXt (Tiny)", "DenseNet121"] if selected_model == "Compare All" else [selected_model]
    
    outputs = []
    for i in range(3):
        if i < len(models_to_run):
            m_name = models_to_run[i]
            model, device, loaded = load_model(m_name)
            
            if not loaded:
                outputs.extend([
                    gr.update(visible=True), 
                    f"### **{m_name}**", 
                    f"⚠️ Weights missing for {m_name}", 
                    None
                ])
                continue
                
            probs = predict_image(model, image_tensor, device)
            top_idx = probs.argmax()
            success_text = f"✅ **{CLASSES[top_idx].title()}** ({probs[top_idx] * 100:.1f}%)"
            
            df_probs = pd.DataFrame({
                "Style": [c.title() for c in CLASSES],
                "Probability": probs * 100
            })
            
            outputs.extend([
                gr.update(visible=True),        # Column visibility
                f"### **{m_name}**",            # Header
                success_text,                   # Top Prediction Text
                gr.update(value=df_probs)       # Barplot data
            ])
        else:
            outputs.extend([gr.update(visible=False), "", "", None])
            
    return outputs

def run_evaluation(selected_model, progress=gr.Progress()):
    if not os.path.exists(DATA_DIR):
        return gr.update(visible=True), None, pd.DataFrame(), [gr.update(visible=False), None] * 3
        
    models_to_run = ["EfficientNetV2", "ConvNeXt (Tiny)", "DenseNet121"] if selected_model == "Compare All" else [selected_model]
    
    accuracies = {}
    cm_figures = {}
    
    for i, m_name in enumerate(models_to_run):
        progress(i / len(models_to_run), desc=f"Evaluating {m_name}...")
        acc, y_true, y_pred = evaluate_on_test_set(m_name)
        if acc is not None:
            accuracies[m_name] = acc
            cm_figures[m_name] = plot_confusion_matrix(y_true, y_pred, f"{m_name} Confusion Matrix")
            
    progress(1.0, desc="Evaluation Complete!")
    
    if not accuracies:
        return gr.update(visible=True), None, pd.DataFrame(), [gr.update(visible=False), None] * 3

    acc_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy (%)'])
    
    # Base updates: Show the results container, update bar plot, update dataframe
    updates = [gr.update(visible=True), acc_df, acc_df]
    
    # Dynamic CM updates
    for i in range(3):
        if i < len(models_to_run):
            m_name = models_to_run[i]
            updates.extend([gr.update(visible=True), cm_figures[m_name]])
        else:
            updates.extend([gr.update(visible=False), None])
            
    return tuple(updates)

# ==========================================
# 4. GRADIO APP LAYOUT (Replacing Streamlit Layout)
# ==========================================
with gr.Blocks(title="AI Art Style Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 AI Art Style Classifier")
    
    if NUM_CLASSES == 0:
        gr.Warning("⚠️ No dataset classes found. Please ensure your images are inside subfolders in the `datasets/` directory.")
    
    with gr.Row():
        # --- SIDEBAR EQUIVALENT ---
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            model_options = ["EfficientNetV2", "ConvNeXt (Tiny)", "DenseNet121", "Compare All"]
            selected_model = gr.Dropdown(choices=model_options, value="EfficientNetV2", label="Choose a Model to activate:")
            gr.Markdown("---")
            uploaded_file = gr.Image(type="pil", label="Upload an Art Piece...")
            
        # --- MAIN PAGE TABS EQUIVALENT ---
        with gr.Column(scale=3):
            with gr.Tabs():
                
                # TAB 1: SINGLE IMAGE TEST
                with gr.TabItem("🖼️ Single Image Test"):
                    gr.Markdown("Upload a painting to see what art movement the AI thinks it belongs to.")
                    
                    with gr.Row():
                        # We build 3 dynamic columns to mimic Streamlit's `st.columns(3)` for "Compare All"
                        res_cols = []
                        headers = []
                        labels = []
                        plots = []
                        
                        for i in range(3):
                            with gr.Column(visible=False) as col:
                                header = gr.Markdown()
                                label = gr.Markdown()
                                plot = gr.BarPlot(x="Style", y="Probability", tooltip=["Style", "Probability"], height=250)
                                
                                res_cols.append(col)
                                headers.append(header)
                                labels.append(label)
                                plots.append(plot)

                    # Trigger prediction when image is uploaded OR model dropdown changes
                    input_components = [uploaded_file, selected_model]
                    output_components = []
                    for i in range(3):
                        output_components.extend([res_cols[i], headers[i], labels[i], plots[i]])
                        
                    uploaded_file.change(fn=process_single_image, inputs=input_components, outputs=output_components)
                    selected_model.change(fn=process_single_image, inputs=input_components, outputs=output_components)

                # TAB 2: FULL TEST SET EVALUATION
                with gr.TabItem("📊 Full Test Set Evaluation"):
                    gr.Markdown("Run the active model(s) against the dynamically allocated 10% test set to generate accuracy metrics and confusion matrices.")
                    eval_btn = gr.Button("🚀 Run Test Set Evaluation", variant="primary")
                    
                    with gr.Column(visible=False) as eval_results:
                        gr.Markdown("### 🏆 Model Accuracies")
                        acc_bar = gr.BarPlot(x="Model", y="Accuracy (%)", tooltip=["Model", "Accuracy (%)"], height=300)
                        acc_df = gr.Dataframe()
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🧩 Confusion Matrices")
                        with gr.Row():
                            cm_cols = []
                            cm_plots = []
                            for i in range(3):
                                with gr.Column(visible=False) as cm_col:
                                    cm_plot = gr.Plot()
                                    cm_cols.append(cm_col)
                                    cm_plots.append(cm_plot)
                    
                    # Output list for Evaluate function
                    eval_outputs = [eval_results, acc_bar, acc_df]
                    for i in range(3):
                        eval_outputs.extend([cm_cols[i], cm_plots[i]])
                        
                    eval_btn.click(fn=run_evaluation, inputs=selected_model, outputs=eval_outputs)

print(f"[4/4] Launching Web Server...")
print("="*50)

if __name__ == "__main__":
    demo.launch()