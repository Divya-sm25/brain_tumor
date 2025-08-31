# ============================================================
# Gradio App: ResNet18 Classification + SwinUNet Segmentation
# With Basic AES Encryption for Uploaded Images
# ============================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm
import gradio as gr
import numpy as np
from PIL import Image
import cv2

# Security (AES encryption)
from cryptography.fernet import Fernet

# Generate or load a key
KEY_PATH = "secret.key"
if not os.path.exists(KEY_PATH):
    with open(KEY_PATH, "wb") as f:
        f.write(Fernet.generate_key())
with open(KEY_PATH, "rb") as f:
    key = f.read()
fernet = Fernet(key)

# Utility: encrypt image before saving
def encrypt_image(image_bytes, save_path):
    encrypted = fernet.encrypt(image_bytes)
    with open(save_path, "wb") as f:
        f.write(encrypted)

# Utility: decrypt image before loading
def decrypt_image(path):
    with open(path, "rb") as f:
        encrypted = f.read()
    decrypted = fernet.decrypt(encrypted)
    return Image.open(io.BytesIO(decrypted))

import io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Classification Model ----------------
class BrainTumorResNet18(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):  # 4 classes
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# Load classification model
clf_model = BrainTumorResNet18(num_classes=4).to(DEVICE)
clf_model.load_state_dict(torch.load(
    "models/best_resnet18_mri.pth",  # ✅ updated path
    map_location=DEVICE
))
clf_model.eval()

clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ---------------- Segmentation Model ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class SwinUNet(nn.Module):
    def __init__(self, encoder_name="swin_small_patch4_window7_224", pretrained=True, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained,
                                         features_only=True, out_indices=(0,1,2,3))
        enc_chs = self.encoder.feature_info.channels()
        self.up3 = nn.ConvTranspose2d(enc_chs[3], enc_chs[2], 2, stride=2)
        self.dec3 = ConvBlock(enc_chs[2]*2, enc_chs[2])
        self.up2 = nn.ConvTranspose2d(enc_chs[2], enc_chs[1], 2, stride=2)
        self.dec2 = ConvBlock(enc_chs[1]*2, enc_chs[1])
        self.up1 = nn.ConvTranspose2d(enc_chs[1], enc_chs[0], 2, stride=2)
        self.dec1 = ConvBlock(enc_chs[0]*2, enc_chs[0])
        self.final_up = nn.ConvTranspose2d(enc_chs[0], 64, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
    def _ensure_nchw(self, feat, expected_ch):
        if feat.ndim==4:
            if feat.shape[1]==expected_ch: return feat
            if feat.shape[-1]==expected_ch: return feat.permute(0,3,1,2).contiguous()
        return feat
    def forward(self, x):
        feats = self.encoder(x)
        expected = self.encoder.feature_info.channels()
        for i in range(len(feats)):
            feats[i] = self._ensure_nchw(feats[i], expected[i])
        f0,f1,f2,f3 = feats
        d3 = self.up3(f3)
        if d3.shape[-2:] != f2.shape[-2:]:
            d3 = nn.functional.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3,f2], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-2:] != f1.shape[-2:]:
            d2 = nn.functional.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2,f1], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != f0.shape[-2:]:
            d1 = nn.functional.interpolate(d1, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1,f0], dim=1))
        out = self.final_up(d1)
        return self.final_conv(out)

# Load segmentation model
seg_model = SwinUNet().to(DEVICE)
seg_model.load_state_dict(
    torch.load("models/swinunet_best (6).pth", map_location=DEVICE),  # ✅ updated path
    strict=False
)
seg_model.eval()

seg_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------- Inference Function ----------------
def predict(img):
    pil_img = Image.fromarray(img).convert("RGB")

    # ---- Encrypt uploaded image (security step) ----
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="PNG")
    encrypt_image(img_bytes.getvalue(), "temp/encrypted_input.img")

    # ---- Classification ----
    x = clf_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = clf_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = CLASS_NAMES[np.argmax(probs)]
    conf = float(np.max(probs))

    # ---- Segmentation ----
    seg_in = seg_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = seg_model(seg_in)[0,0].cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)

    # Resize mask to match original
    img_np = np.array(pil_img.resize((224,224)))
    mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay mask
    overlay = img_np.copy()
    overlay[mask_resized > 0] = [255, 0, 0]  # red
    blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

    return blended, f"Prediction: {pred_class} (conf: {conf:.2f})"

# ---------------- Gradio UI ----------------
example_images = [
    "images/img1.jpg",
    "images/img2.jpg",
    "images/img3.jpg",
    "images/img4.jpg",
    "images/img5.jpg",
    "images/img6.jpg",
    "images/img7.jpg",
    "images/img8.jpg",
    "images/img9.jpg",
    "images/img10.jpg",
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Textbox()],
    title="🧠 Brain Tumor Classification + Segmentation",
    description="Upload an MRI or click on one of the example images. The app will classify tumor type (ResNet18) and segment tumor region (SwinUNet).",
    examples=example_images,
    cache_examples=False
)

demo.launch(inline=True, debug=True, share=True)
