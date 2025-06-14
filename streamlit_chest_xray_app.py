import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import matplotlib.cm as cm

# Class labels
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# Medical suggestions
suggestions = {
    "COVID": (
        "1. Initiate strict isolation protocols to prevent viral transmission.\n"
        "2. Recommend immediate confirmatory testing (e.g., RT-PCR or antigen testing).\n"
        "3. Consult with an infectious disease specialist and monitor respiratory function closely."
    ),
    "Normal": (
        "1. No radiological evidence of pulmonary pathology detected.\n"
        "2. Maintain regular health monitoring and follow-up if symptoms persist.\n"
        "3. Consider clinical correlation and laboratory tests if concerns remain."
    ),
    "Viral Pneumonia": (
        "1. Recommend supportive care and consider antiviral therapy based on clinical context.\n"
        "2. Obtain a comprehensive viral panel (PCR or rapid testing).\n"
        "3. Monitor for potential complications such as hypoxia or secondary bacterial infection."
    ),
    "Lung_Opacity": (
        "1. Advise high-resolution CT (HRCT) scan to further assess abnormal opacities.\n"
        "2. Recommend pulmonology consultation for differential diagnosis.\n"
        "3. Consider bronchoscopy or biopsy if malignancy or atypical infection is suspected."
    )
}


# Explanations
explanation = {
    "COVID": (
        "Diffuse bilateral ground-glass opacities suggest a pattern commonly associated with COVID-19 pneumonia. "
        "These abnormalities are consistent with inflammatory alveolar damage and impaired gas exchange."
    ),
    "Lung_Opacity": (
        "Significant focal or diffuse opacities identified, which may indicate consolidation, edema, hemorrhage, "
        "or neoplastic processes. Further imaging and diagnostic testing are recommended."
    ),
    "Viral Pneumonia": (
        "Patchy, bilateral opacities typical of viral pneumonia observed. Imaging features may vary depending on the viral agent, "
        "but bilateral involvement and peribronchial thickening are common."
    ),
    "Normal": (
        "The lung fields appear clear with no radiographic signs of consolidation, effusion, or nodules. "
        "However, imaging findings should be interpreted in conjunction with clinical presentation."
    )
}


# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, len(class_names))
    )

    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Grad-CAM hooks
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

model.layer4.register_forward_hook(forward_hook)
model.layer4.register_full_backward_hook(backward_hook)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Grad-CAM function
def generate_gradcam(model, image_tensor, class_idx):
    activations.clear()
    gradients.clear()

    output = model(image_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    pooled_grad = torch.mean(grad, dim=(1, 2))

    for i in range(act.shape[0]):
        act[i] *= pooled_grad[i]

    heatmap = torch.mean(act, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

# Streamlit UI
st.title("🩻🧠 Chest X-ray Classifier")
st.write("Upload a chest X-ray image to detect **COVID-19**, **Normal**, **Viral Pneumonia**, or **Lung Opacity**.")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Quick grayscale check to verify X-ray
        grayscale_ratio = np.mean(np.abs(np.array(image)[:, :, 0] - np.array(image)[:, :, 1]))
        if grayscale_ratio > 30:
            st.error("🚫 This doesn't appear to be a valid chest X-ray.")
        else:
            st.image(image, caption="Uploaded Image", use_container_width=True)

            img_tensor = transform(image).unsqueeze(0)
            img_tensor.requires_grad_()

            outputs = model(img_tensor)
            probs = F.softmax(outputs[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            pred_conf = probs[pred_idx].item()
            confidence_threshold = 0.70

            if pred_conf >= confidence_threshold:
                pred_class = class_names[pred_idx]

                st.subheader("🔍 Prediction Results:")
                for i, prob in enumerate(probs):
                    st.write(f"{class_names[i]}: {prob.item() * 100:.2f}%")

                st.success(f"🔬 Most likely: **{pred_class}** ({pred_conf * 100:.2f}%)")

                st.subheader("📋 Suggested Medical Step:")
                st.info(suggestions[pred_class])

                st.subheader("🌡️ Grad-CAM Heatmap:")
                heatmap = generate_gradcam(model, img_tensor, pred_idx)

                heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
                heatmap_resized = np.array(heatmap_resized)
                image_np = np.array(image).astype(np.float32) / 255.0
                heatmap_color = cm.jet(heatmap_resized / 255.0)[..., :3]
                blended = 0.6 * image_np + 0.4 * heatmap_color
                blended = np.clip(blended, 0, 1)

                st.image(blended, caption="Model Decision Heatmap", use_container_width=True)

                st.subheader("🧠 Model Explanation:")
                st.write(explanation.get(pred_class, "No explanation available."))
            else:
                st.warning(f"🔎 Model confidence too low: {pred_conf * 100:.2f}%.")
                st.info("Image classified as **Uncertain**. Try a clearer chest X-ray.")

    except Exception as e:
        st.error(f"🚫 Unexpected error: {e}")
