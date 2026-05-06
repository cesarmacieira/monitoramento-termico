"""
============================================================
app.py — Monitoramento Térmico do Polisher
============================================================
Pipeline de dois modelos:
  MODELO 1 — Classificador (EfficientNet-B3)
    Entrada : frame grayscale 1 canal
    Saída   : Normal / Atenção / Crítico + Grad-CAM

  MODELO 2 — Segmentador (U-Net, 3 canais)
    Entrada : [canal_spatial, canal_temp_abs, mapa_gradcam]
    Saída   : mapa de probabilidade de anomalia por pixel

Checkpoints esperados em ./checkpoints/:
    classifier_best.pt
    unet_best.pt

Como usar:
    streamlit run app.py
============================================================
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import numpy as np
from PIL import Image
import cv2
import matplotlib.cm as cm
import matplotlib
import tempfile
import os
import json
from pathlib import Path
import io

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# ─────────────────────────────────────────────────────────
#  DEFAULTS
# ─────────────────────────────────────────────────────────
CHECKPOINTS_DIR         = Path("checkpoints")
DEFAULT_MASK_CUTOFF     = 0.5
ANOMALY_SCORE_THRESHOLD = 5.0

VALID_STD_MIN        = 5.0
VALID_STD_MAX        = 110.0  # térmicas com anomalia quente podem ter contraste moderadamente alto
VALID_EDGE_MAX       = 0.12   # reduzido: térmicas têm gradientes muito suaves
VALID_SATURATION_MAX = 0.10   # reduzido: menos tolerância a pixels saturados

CLASS_NAMES  = ["normal", "atencao", "critico"]
CLASS_LABELS = {"normal": "Normal", "atencao": "Atenção", "critico": "Crítico"}
CLASS_COLORS = {"normal": "#2a9d5c", "atencao": "#e06c00", "critico": "#c0392b"}

# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
def _cmap(name):
    try:
        return cm.get_cmap(name)
    except AttributeError:
        return matplotlib.colormaps[name]


def validar_dominio(frame_gray_np: np.ndarray, pil_original: "Image.Image | None" = None):
    img = frame_gray_np.astype(np.float32)
    H, W = img.shape[:2]
    total_pixels = H * W
    detalhes = {}

    # 0. Verificação de cor para distinguir foto real de imagem térmica falsa-cor.
    #
    # PROBLEMA: tanto fotos reais quanto imagens térmicas com colormap (inferno/jet)
    # são coloridas — NÃO se pode usar "desvio de cor" como critério.
    #
    # SOLUÇÃO: usar o canal de luminância (grayscale) para discriminar.
    # Fotos reais têm textura de alta frequência (bordas finas, detalhes de pele/cabelo).
    # Imagens térmicas têm gradientes suaves mesmo com cores vivas.
    #
    # Critério principal: Laplaciano no canal grayscale.
    #   - Foto real:    Laplaciano alto (muitas bordas e texturas)
    #   - Imagem térmica: Laplaciano baixo (gradientes suaves)
    #
    # Critério secundário: correlação entre canais RGB.
    #   - Colormaps térmicos (inferno/jet/hot): cada nível de cinza mapeia para
    #     uma cor específica → os canais RGB têm correlação MONOTÔNICA com o cinza.
    #   - Fotos reais: cores independentes do brilho local → correlação BAIXA.
    if pil_original is not None:
        try:
            arr_rgb = np.array(pil_original.convert("RGB")).astype(np.float32)
            r, g, b = arr_rgb[:,:,0], arr_rgb[:,:,1], arr_rgb[:,:,2]

            # ── CRITÉRIO 0: Detecção de face (Haar Cascade) ─────────────────
            # Selfies preservam a estrutura espacial de um rosto.
            # IMPORTANTE: imagens térmicas com hotspot circular/oval podem gerar
            # falsos positivos no Haar cascade. Proteções:
            #   1. minNeighbors=10 — muito mais restritivo que o padrão (3-4)
            #   2. minSize=(60,60) — ignora detecções pequenas
            #   3. Validação por Laplaciano na ROI da face detectada:
            #      face real > 80 (textura pele/cabelo), hotspot térmico << 80
            try:
                face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(face_cascade_path)
                img_u8_rgb = np.array(pil_original.convert("RGB"))
                img_gray_face = cv2.cvtColor(img_u8_rgb, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(
                    img_gray_face, scaleFactor=1.05, minNeighbors=10, minSize=(60, 60)
                )
                n_faces = len(faces) if faces is not None and len(faces) > 0 else 0
                detalhes["faces_detectadas"] = n_faces
                if n_faces > 0:
                    # Validação extra: textura de pele (Laplaciano alto) vs hotspot térmico
                    confirmed_faces = 0
                    for (fx, fy, fw, fh) in faces:
                        roi_gray = img_gray_face[fy:fy+fh, fx:fx+fw]
                        lap_face = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
                        # Face real: Laplaciano > 80; hotspot térmico: Laplaciano << 80
                        if lap_face > 80:
                            confirmed_faces += 1
                    if confirmed_faces > 0:
                        return False, (f"Envie um frame da camera térmica."), detalhes
            except Exception:
                pass  # Se Haar falhar, continua com outros critérios

            # Correlação entre canais e luminância: colormaps têm correlação alta
            lum = (0.299*r + 0.587*g + 0.114*b).flatten()
            lum_std = float(np.std(lum))
            if lum_std > 1e-3:
                corr_r = float(np.corrcoef(r.flatten(), lum)[0, 1])
                corr_g = float(np.corrcoef(g.flatten(), lum)[0, 1])
                corr_b = float(np.corrcoef(b.flatten(), lum)[0, 1])
                max_chan_corr = max(abs(corr_r), abs(corr_g), abs(corr_b))
                min_chan_corr = min(abs(corr_r), abs(corr_g), abs(corr_b))
                detalhes["corr_canal_lum_max"] = round(max_chan_corr, 3)
                detalhes["corr_canal_lum_min"] = round(min_chan_corr, 3)

                # Foto real: todos os canais têm correlação BAIXA com luminância
                # Colormap térmico: pelo menos um canal tem correlação ALTA (> 0.85)
                # E a correlação mínima ainda é razoável (> 0.3) — padrão monotônico
                is_colormap_like = (max_chan_corr > 0.85) and (min_chan_corr > 0.25)
                is_real_photo    = (max_chan_corr < 0.75) or (min_chan_corr < 0.15)

                # Laplaciano no grayscale: alto = foto real, baixo = térmica
                img_u8_chk = img.astype(np.uint8)
                lap_gray = float(cv2.Laplacian(img_u8_chk, cv2.CV_64F).var())
                detalhes["laplacian_gray"] = round(lap_gray, 1)

                # ── CRITÉRIO D: foto com colormap aplicado (ex: selfie inferno) ──
                # Uma foto real com colormap térmico aplicado fica com:
                #   - is_colormap_like = True  (enganando os critérios A e B)
                #   - Laplaciano ALTO no canal de saturação HSV
                # Imagens térmicas reais têm saturação HSV muito suave/homogênea.
                # Fotos reais colorizadas preservam textura de pele/cabelo no canal S.
                try:
                    arr_rgb_u8 = arr_rgb.astype(np.uint8)
                    hsv = cv2.cvtColor(arr_rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
                    sat_channel = hsv[:, :, 1]  # canal S (saturação)
                    lap_sat = float(cv2.Laplacian(sat_channel.astype(np.uint8), cv2.CV_64F).var())
                    detalhes["laplacian_saturacao"] = round(lap_sat, 1)
                    # Térmicas reais: saturação varia de forma suave → lap_sat < 80
                    # Selfie com colormap: saturação preserva bordas do rosto → lap_sat > 80
                    if is_colormap_like and lap_sat > 80:
                        return False, (f"Envie um frame válido."), detalhes
                except Exception:
                    pass

                # Critério A — foto com textura densa e sem padrão de colormap
                if lap_gray > 600 and not is_colormap_like:
                    return False, (f"Envie um frame válido."), detalhes

                # Critério B — selfie em ambiente escuro / baixo contraste
                # Fotos reais em ambiente escuro têm Laplaciano moderado (150–600)
                # mas NÃO têm o padrão monotônico de colormap térmico.
                # Câmera Pi térmica real: canais RGB formam colormap genuíno
                # (inferno/jet) com correlação alta E Laplaciano < 150 (gradiente suave).
                # Selfie escura: Laplaciano moderado (>150) E sem colormap genuíno.
                if lap_gray > 150 and not is_colormap_like:
                    return False, (f"Envie um frame válido."), detalhes

                # Critério C — imagem com histograma concentrado (fundo uniforme/screenshot)
                # Avaliado aqui com o pil_original RGB para pegar casos como screenshots
                hist_rgb, _ = np.histogram(
                    (0.299*r + 0.587*g + 0.114*b).flatten(), bins=32, range=(0, 256)
                )
                hist_rgb_norm = hist_rgb / (hist_rgb.sum() + 1e-8)
                hist_rgb_max = float(hist_rgb_norm.max())
                detalhes["hist_pico_rgb"] = round(hist_rgb_max, 4)
                # Térmicas reais com hotspot concentrado chegam até ~56% — margem segura em 65%
                if hist_rgb_max > 0.65:
                    return False, (
                        f"Histograma muito concentrado na imagem original (pico={hist_rgb_max*100:.1f}%). "
                        f"Fundo uniforme, screenshot ou imagem sintetica."
                    ), detalhes

        except Exception:
            pass  # Se falhar, continua com validação normal

    # 1. STD global
    std_val = float(np.std(img))
    detalhes["std_intensidade"] = round(std_val, 2)
    if std_val < VALID_STD_MIN:
        return False, f"Imagem muito uniforme (std={std_val:.1f} < {VALID_STD_MIN}). Frame vazio ou corrompido.", detalhes
    if std_val > VALID_STD_MAX:
        return False, f"Contraste muito alto (std={std_val:.1f} > {VALID_STD_MAX}). Nao parece imagem termica.", detalhes

    # 2. Pixels saturados
    n_black  = int(np.sum(img <= 2))
    n_white  = int(np.sum(img >= 253))
    frac_sat = (n_black + n_white) / total_pixels
    detalhes["fracao_saturada"] = round(frac_sat, 4)
    if frac_sat > VALID_SATURATION_MAX:
        return False, f"Alta saturacao ({frac_sat*100:.1f}% pixels pretos/brancos). Indica imagem nao-termica.", detalhes

    # 3. Densidade de bordas Canny
    img_u8 = img.astype(np.uint8)
    edges  = cv2.Canny(img_u8, threshold1=30, threshold2=100)
    edge_ratio = float(np.sum(edges > 0)) / total_pixels
    detalhes["densidade_bordas"] = round(edge_ratio, 4)
    if edge_ratio > VALID_EDGE_MAX:
        return False, f"Bordas muito densas ({edge_ratio*100:.1f}%). Imagens termicas tem gradientes suaves.", detalhes

    # 4. Patches de alto contraste local
    patch_stds = []
    ps = 16
    for r in range(0, H - ps, ps):
        for c in range(0, W - ps, ps):
            patch_stds.append(np.std(img[r:r+ps, c:c+ps]))
    frac_high_patch = float(np.mean(np.array(patch_stds) > 60))
    detalhes["fracao_patches_alto_contraste"] = round(frac_high_patch, 4)
    if frac_high_patch > 0.30:
        return False, f"Muitos patches com alto contraste local ({frac_high_patch*100:.1f}%). Indica textura artificial.", detalhes

    # 5. Suavidade espacial - Laplaciano
    # Fotos reais tem muito detalhe de textura; termicas sao intrinsecamente suaves
    laplacian_var = float(cv2.Laplacian(img_u8, cv2.CV_64F).var())
    detalhes["laplacian_var"] = round(laplacian_var, 1)
    if laplacian_var > 800:
        return False, f"Textura muito detalhada (Laplaciano={laplacian_var:.0f}). Nao parece imagem termica.", detalhes

    # 6. Histograma: pico concentrado indica fundo uniforme (foto/selfie)
    #
    # PROBLEMA: câmeras térmicas com hotspot pequeno (ponto quente no fundo frio)
    # têm a maioria dos pixels em tons escuros → histograma naturalmente concentrado
    # no bin baixo, chegando a 50–70% sem ser uma imagem sintética ou selfie.
    #
    # SOLUÇÃO: só rejeitar se o pico estiver nos tons MÉDIOS ou ALTOS (bins 8–31),
    # que é o padrão de fundos neutros de fotos/selfies. Pico no bin 0–7 (escuro)
    # é característico de térmica com fundo frio + hotspot concentrado → aceitar.
    hist, _ = np.histogram(img.flatten(), bins=32, range=(0, 256))
    hist_norm = hist / (hist.sum() + 1e-8)
    hist_max       = float(hist_norm.max())
    hist_argmax    = int(np.argmax(hist_norm))   # bin onde está o pico (0–31)
    detalhes["hist_pico_max"]    = round(hist_max, 4)
    detalhes["hist_pico_bin"]    = hist_argmax

    # Pico nos bins escuros (0–5, ou seja, 0–40 em valor de pixel):
    # padrão de câmera térmica com fundo frio — NÃO rejeitar mesmo com pico alto.
    pico_em_escuro = hist_argmax <= 5

    # Threshold generoso: rejeita apenas imagens com pico extremo
    # fora da região escura (fundos neutros, screenshots, etc.)
    if hist_max > 0.55 and not pico_em_escuro:
        return False, f"Histograma muito concentrado (pico={hist_max*100:.1f}%, bin={hist_argmax}). Fundo uniforme ou imagem sintetica.", detalhes

    return True, "OK", detalhes


# ─────────────────────────────────────────────────────────
#  ARQUITETURA — Classificador EfficientNet-B3
# ─────────────────────────────────────────────────────────
class ThermalClassifier(nn.Module):
    GRADCAM_LAYER_TIMM = "backbone.blocks.6"
    GRADCAM_LAYER_TV   = "backbone.features.7"

    def __init__(self, num_classes=3, dropout=0.40, backend="timm"):
        super().__init__()
        self._backend = backend if (backend == "timm" and HAS_TIMM) else "torchvision"

        if self._backend == "timm":
            self.backbone = timm.create_model(
                "efficientnet_b3",
                pretrained=False,
                num_classes=0,
                global_pool="avg",
                in_chans=1,
            )
            feat_dim = self.backbone.num_features
        else:
            _tv = efficientnet_b3(weights=None)
            orig = _tv.features[0][0]
            new_conv = nn.Conv2d(
                1, orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.copy_(orig.weight.mean(dim=1, keepdim=True))
            _tv.features[0][0] = new_conv
            feat_dim = _tv.classifier[1].in_features
            _tv.classifier = nn.Identity()
            self.backbone = _tv

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self._extract_features(x[:, 0:1])
        return self.head(feat)

    def _extract_features(self, x1ch):
        return self.backbone(x1ch)

    def get_features(self, x):
        if self._backend == "timm":
            return self.backbone.forward_features(x[:, 0:1])
        else:
            return self.backbone.features(x[:, 0:1])

    @property
    def gradcam_layer_name(self):
        return self.GRADCAM_LAYER_TIMM if self._backend == "timm" else self.GRADCAM_LAYER_TV


# ─────────────────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = None
        self.feats = None

        target_layer_name = getattr(model, "gradcam_layer_name", "backbone.features.7")
        target_layer = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break

        if target_layer is None:
            bb = model.backbone
            if hasattr(bb, "blocks"):
                target_layer = bb.blocks[-1]
            elif hasattr(bb, "features"):
                target_layer = bb.features[-2]
            else:
                target_layer = list(bb.modules())[-3]

        self.handle_f = target_layer.register_forward_hook(self._save_feats)
        self.handle_b = target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, module, input, output):
        self.feats = (output[0] if isinstance(output, tuple) else output).detach()

    def _save_grads(self, module, grad_input, grad_output):
        self.grads = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.feats).sum(dim=1)
        cam     = F.relu(cam).squeeze()
        if cam.dim() == 0:
            cam = cam.unsqueeze(0).unsqueeze(0)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, class_idx

    def remove_hooks(self):
        self.handle_f.remove()
        self.handle_b.remove()


# ─────────────────────────────────────────────────────────
#  ARQUITETURA — U-Net (3 canais: spatial + temp + gradcam)
# ─────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ThermalUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=32, dropout=0.15):
        super().__init__()
        b = base_ch
        self.enc1 = DoubleConv(in_channels, b,    dropout=0.0)
        self.enc2 = DoubleConv(b,           b*2,  dropout=dropout)
        self.enc3 = DoubleConv(b*2,         b*4,  dropout=dropout)
        self.enc4 = DoubleConv(b*4,         b*8,  dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(b*8, b*16, dropout=dropout)
        self.up4  = nn.ConvTranspose2d(b*16, b*8,  2, stride=2)
        self.dec4 = DoubleConv(b*16, b*8)
        self.up3  = nn.ConvTranspose2d(b*8,  b*4,  2, stride=2)
        self.dec3 = DoubleConv(b*8,  b*4)
        self.up2  = nn.ConvTranspose2d(b*4,  b*2,  2, stride=2)
        self.dec2 = DoubleConv(b*4,  b*2)
        self.up1  = nn.ConvTranspose2d(b*2,  b,    2, stride=2)
        self.dec1 = DoubleConv(b*2,  b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)  # logits sem sigmoid


# ─────────────────────────────────────────────────────────
#  CARREGAMENTO
# ─────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelos(checkpoints_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_path  = checkpoints_dir / "classifier_best.pt"
    unet_path = checkpoints_dir / "unet_best.pt"

    if not cls_path.exists() or not unet_path.exists():
        return None, None, None, device

    ckpt_cls  = torch.load(cls_path,  map_location=device, weights_only=False)
    ckpt_unet = torch.load(unet_path, map_location=device, weights_only=False)

    # Instancia classificador com o backend salvo no checkpoint
    backend = ckpt_cls.get("backend", "torchvision")
    classifier = ThermalClassifier(
        num_classes=len(ckpt_cls.get("class_names", CLASS_NAMES)),
        dropout=0.0,
        backend=backend,
    ).to(device)
    classifier.load_state_dict(ckpt_cls["model_state"])
    classifier.eval()

    # Instancia U-Net (sempre 3 canais: spatial + temp + gradcam)
    unet = ThermalUNet(
        in_channels=3,
        base_ch=32,
        dropout=0.0,
    ).to(device)
    unet.load_state_dict(ckpt_unet["model_state"])
    unet.eval()

    cfg = {
        "img_size"        : ckpt_cls["img_size"],
        "g_min"           : ckpt_cls["g_min"],
        "g_max"           : ckpt_cls["g_max"],
        "thresh_atencao"  : ckpt_cls.get("thresh_atencao", None),
        "thresh_critico"  : ckpt_cls.get("thresh_critico", None),
        "class_names"     : ckpt_cls.get("class_names", CLASS_NAMES),
        "backend"         : backend,
        "val_acc"         : ckpt_cls.get("val_acc", "—"),
        "epoch_cls"       : ckpt_cls.get("epoch", "—"),
        "val_loss_unet"   : ckpt_unet.get("val_loss", "—"),
        "val_iou"         : ckpt_unet.get("val_iou", "—"),
        "val_f1"          : ckpt_unet.get("val_f1",  "—"),
        "epoch_unet"      : ckpt_unet.get("epoch",   "—"),
        "gradcam_thresh"  : ckpt_unet.get("gradcam_thresh_pct", 70),
    }
    return classifier, unet, cfg, device


# ─────────────────────────────────────────────────────────
#  INFERÊNCIA
# ─────────────────────────────────────────────────────────
def inferir_frame(frame_gray_np, classifier, unet, cfg, device,
                  mean_abs=None, mask_cutoff=DEFAULT_MASK_CUTOFF):
    """
    Pipeline:
      1. Prepara tensor 2ch (spatial + temp_norm)
      2. Classificador → classe predita + Grad-CAM
      3. Monta tensor 3ch (spatial + temp_norm + gradcam)
      4. U-Net → máscara de probabilidade
    """
    H, W = cfg["img_size"]
    g_min, g_max = cfg["g_min"], cfg["g_max"]
    tf = T.Compose([T.Resize((H, W), antialias=True), T.ToTensor()])

    if frame_gray_np.ndim == 3:
        frame_gray_np = frame_gray_np[:, :, 0]

    # Guarda dimensões originais para restaurar o overlay ao final
    orig_H, orig_W = frame_gray_np.shape[:2]

    pil_gray = Image.fromarray(frame_gray_np.astype(np.uint8), mode="L")
    ch_spatial = tf(pil_gray)  # (1, H, W)

    mean_norm = (
        float(np.clip((mean_abs - g_min) / (g_max - g_min + 1e-8), 0.0, 1.0))
        if mean_abs is not None else 0.5
    )
    ch_temp = torch.full_like(ch_spatial, mean_norm)
    x2ch = torch.cat([ch_spatial, ch_temp], dim=0).unsqueeze(0).to(device)  # (1, 2, H, W)

    # ── Classificador + Grad-CAM ──────────────────────────
    gradcam = GradCAM(classifier)
    with torch.enable_grad():
        cam_map, pred_class_idx = gradcam.generate(x2ch)
    gradcam.remove_hooks()

    pred_class_name = cfg["class_names"][pred_class_idx]
    with torch.no_grad():
        logits = classifier(x2ch)
        probs_cls = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    # ── Monta entrada 3ch para U-Net ─────────────────────
    ch_gradcam = torch.from_numpy(cam_map).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    ch_gradcam = F.interpolate(ch_gradcam, size=(H, W), mode="bilinear", align_corners=False)
    x3ch = torch.cat([
        x2ch[:, 0:1],   # canal spatial
        x2ch[:, 1:2],   # canal temp
        ch_gradcam.to(device),
    ], dim=1)            # (1, 3, H, W)

    with torch.no_grad():
        seg_probs = torch.sigmoid(unet(x3ch)).squeeze().cpu().numpy()

    # ── Visualização ─────────────────────────────────────
    mask_bin = (seg_probs > mask_cutoff).astype(np.uint8)
    gray_norm = np.array(pil_gray.resize((W, H)), dtype=np.float32) / 255.0

    # Cores por classe (RGB)
    _CLASS_COLORS_CV2 = {
        "normal":  (0,   200,  80),
        "atencao": (255, 165,   0),
        "critico": (220,  30,  30),
    }
    cor_anom   = _CLASS_COLORS_CV2.get(pred_class_name, (255, 80, 80))

    # Base: colormap inferno fundido com heatmap hot da U-Net
    viz_rgb  = (_cmap("inferno")(gray_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap  = (_cmap("hot")(seg_probs)[:, :, :3] * 255).astype(np.uint8)
    overlay  = cv2.addWeighted(viz_rgb, 0.60, heatmap, 0.40, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_c = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_c = cv2.morphologyEx(mask_c,   cv2.MORPH_OPEN,  kernel)

    contornos, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regioes = []
    g_min, g_max = cfg["g_min"], cfg["g_max"]
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Contorno pixel-level + bounding box
        cv2.drawContours(overlay, [cnt], -1, cor_anom, 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), cor_anom, 1)
        region_mean_norm = float(seg_probs[y:y + h, x:x + w].mean())
        temp_est = g_min + region_mean_norm * (g_max - g_min)
        regioes.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area), "temp_estimada": round(float(temp_est), 2),
        })

    # Sem regiões binárias mas classificador detectou anomalia:
    # sobrepõe heatmap difuso da U-Net (ou Grad-CAM como fallback)
    if not regioes and pred_class_name in ("atencao", "critico"):
        probs_up = cv2.resize(seg_probs, (W, H), interpolation=cv2.INTER_LINEAR)
        p_min, p_max = probs_up.min(), probs_up.max()
        if p_max - p_min > 1e-6:
            probs_norm_vis = (probs_up - p_min) / (p_max - p_min)
        else:
            probs_norm_vis = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_LINEAR)
        heat_rgb = (_cmap("hot")(probs_norm_vis)[:, :, :3] * 255).astype(np.uint8)
        alpha = (probs_norm_vis * 0.55).clip(0, 0.55)
        for c in range(3):
            overlay[:, :, c] = np.clip(
                overlay[:, :, c] * (1 - alpha) + heat_rgb[:, :, c] * alpha, 0, 255
            ).astype(np.uint8)
        thresh_vis = float(np.percentile(probs_norm_vis, 85))
        hot_mask = (probs_norm_vis >= thresh_vis).astype(np.uint8)
        cnts_vis, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_vis:
            if cv2.contourArea(cnt) < 30:
                continue
            xv, yv, wv, hv = cv2.boundingRect(cnt)
            cv2.drawContours(overlay, [cnt], -1, cor_anom, 2)
            cv2.rectangle(overlay, (xv, yv), (xv + wv, yv + hv), cor_anom, 1)

    score_pct = float(mask_c.mean() * 100)


    # ── Redimensiona overlay de volta para o tamanho original do frame ───
    if (orig_H, orig_W) != (H, W):
        overlay = cv2.resize(overlay, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)

    # ── Decisão unificada: classificador + U-Net ─────────
    cls_is_critico  = pred_class_name == "critico"
    cls_is_atencao  = pred_class_name == "atencao"
    unet_has_signal = len(regioes) > 0 or score_pct > ANOMALY_SCORE_THRESHOLD

    if cls_is_critico:
        is_anomaly = True
    elif cls_is_atencao and unet_has_signal:
        is_anomaly = True
    else:
        is_anomaly = False

    return (overlay, score_pct, is_anomaly, regioes, seg_probs, pred_class_name, probs_cls, cam_map)


def processar_video(video_path, classifier, unet, cfg, device,
                    mean_abs=None, mask_cutoff=DEFAULT_MASK_CUTOFF,
                    progress_bar=None):
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H, W  = cfg["img_size"]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name

    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    relatorio = []
    rejeitados = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        valido, motivo, _ = validar_dominio(gray)  # video: sem pil_original
        if not valido:
            rejeitados.append(frame_idx)
            warn = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(warn, "FORA DO DOMINIO", (10, H//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
            cv2.putText(warn, motivo[:60], (10, H//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            writer.write(warn)
            relatorio.append({"frame": frame_idx, "anomalia_pct": None,
                               "tem_anomalia": None, "num_regioes": 0,
                               "classe": None, "dominio_valido": False,
                               "motivo_rejeicao": motivo})
        else:
            (overlay, score, is_anom, regioes, _,
             pred_class, probs_cls, _) = inferir_frame(gray, classifier, unet, cfg, device, mean_abs=mean_abs, mask_cutoff=mask_cutoff,)
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            relatorio.append({"frame": frame_idx, "anomalia_pct": round(score, 3),
                               "tem_anomalia": is_anom, "num_regioes": len(regioes),
                               "classe": pred_class, "dominio_valido": True,
                               "probs": {n: round(float(p), 4)
                                         for n, p in zip(cfg["class_names"], probs_cls)}})
        if progress_bar is not None:
            progress_bar.progress(min((frame_idx + 1) / max(total, 1), 1.0), text=f"Frame {frame_idx + 1} / {total}")
        frame_idx += 1

    cap.release()
    writer.release()
    with open(out_path, "rb") as f:
        video_bytes = f.read()
    os.unlink(out_path)
    return video_bytes, relatorio, rejeitados


# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Monitoramento do Polisher", page_icon="🌡️", layout="wide")

# ─────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1340px; }

.cabecalho-app {
    background: #1c2b3a; border-radius: 16px; padding: 2rem 2.4rem;
    margin-bottom: 1.8rem; display: flex; align-items: center;
    gap: 1.5rem; border-left: 5px solid #e06c00;
}
.cabecalho-texto h1 { margin:0; font-size:1.85rem; font-weight:800; color:#f0f4f8; }
.cabecalho-texto p  { margin:0.3rem 0 0 0; font-size:0.95rem; color:#8daabf; }

.card-info  { background:#f7f9fb; border:1.5px solid #d5dde6; border-radius:14px; padding:1.2rem 1.3rem; }
.card-soft  { background:#ffffff; border:1.5px solid #dce5ef; border-radius:14px; padding:1rem 1.15rem; transition:box-shadow 0.2s; }
.card-soft:hover { box-shadow:0 2px 12px rgba(60,100,150,0.08); }
.card-alert { background:#fff8f3; border:1.5px solid #f5c6a0; border-radius:14px; padding:1.2rem 1.3rem; }
.card-ok    { background:#f3fbf6; border:1.5px solid #a0d9b4; border-radius:14px; padding:1.2rem 1.3rem; }
.card-error { background:#fff3f3; border:1.5px solid #f5a0a0; border-radius:14px; padding:1.2rem 1.3rem; }

.section-title {
    font-size:0.78rem; font-weight:700; color:#6b8399;
    letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.7rem;
}
.section-divider { border:none; border-top:1.5px solid #e4ecf4; margin:1.4rem 0; }
.small-muted { color:#7a90a4; font-size:0.88rem; }
.result-caption { text-align:center; color:#7a90a4; font-size:0.82rem; margin-top:0.3rem; font-style:italic; }

.stButton > button {
    background:#1c4f7a; color:white; border:none; border-radius:10px;
    padding:0.6rem 1.2rem; font-weight:600; font-size:0.9rem;
    width:100%; transition:background 0.2s;
}
.stButton > button:hover { background:#164061 !important; color:white !important; }

.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:2px solid #e4ecf4; }
.stTabs [data-baseweb="tab"] {
    font-weight:600; font-size:0.9rem; padding:0.5rem 1.2rem;
    border-radius:8px 8px 0 0; color:#6b8399;
}
.stTabs [aria-selected="true"] { color:#e06c00 !important; border-bottom:2px solid #e06c00 !important; }

code { background:#edf2f7 !important; color:#2a7a4f !important;
    border-radius:5px; padding:1px 6px !important; font-size:0.83em !important; }

.bar-bg { background:#dde5ee; border-radius:999px; height:10px; overflow:hidden; margin-top:6px; }
.bar-fill-ok   { background:#2a9d5c; height:10px; border-radius:999px; transition:width 0.4s; }
.bar-fill-warn { background:#e06c00; height:10px; border-radius:999px; transition:width 0.4s; }
.bar-fill-err  { background:#c0392b; height:10px; border-radius:999px; transition:width 0.4s; }

.classe-badge {
    display:inline-block; padding:0.25rem 0.8rem; border-radius:999px;
    font-weight:700; font-size:0.95rem; margin-top:0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  CABEÇALHO
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="cabecalho-app">
    <div class="cabecalho-texto">
        <h1>🌡️ Monitoramento do Polisher</h1>
        <p>Detecção de anomalias térmicas com EfficientNet-B3 (Grad-CAM) + U-Net · Câmera Pi térmica</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  CARREGAR MODELOS
# ─────────────────────────────────────────────────────────
for key, default in [("analisado", False), ("resultado_img", None),
                     ("upload_key", 0), ("nome_arquivo", None),
                     ("mask_cutoff", DEFAULT_MASK_CUTOFF),
                     ("mean_abs_val", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

cls_model, unet_model, model_cfg, device = carregar_modelos(CHECKPOINTS_DIR)

if cls_model is None:
    st.markdown(f"""
    <div class="card-error">
        <div style="font-size:1.3rem;font-weight:800;color:#c0392b;">⛔ Checkpoints não encontrados</div>
        <div style="margin-top:0.5rem;color:#5e6d7c;">
            Esperado em <code>checkpoints/classifier_best.pt</code> e <code>checkpoints/unet_best.pt</code>.<br>
            Ajuste o caminho abaixo se necessário.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────
aba_analise, aba_modelo, aba_exemplos = st.tabs(["Análise", "Sobre o Modelo", "Exemplos"])


# ══════════════════════════════════════════════════════════
#  ABA — ANÁLISE
# ══════════════════════════════════════════════════════════
with aba_analise:

    st.markdown('<div class="section-title">Parâmetros de inferência</div>', unsafe_allow_html=True)
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        mask_cutoff_val = st.number_input(
            "Limiar da máscara (mask cutoff)",
            min_value=0.10, max_value=0.90,
            value=DEFAULT_MASK_CUTOFF, step=0.05,
            help="Probabilidade mínima para marcar um pixel como anômalo.",
        )

    use_mean = st.checkbox(
        "Informar temperatura real do frame (°C)",
        value=False,
        help=(
            f"Se souber a temperatura média real do frame (lida na câmera), informe aqui. "
            f"Isso melhora a precisão da análise. "
            f"Se deixar desmarcado, o modelo assume {(model_cfg['g_min'] + model_cfg['g_max']) / 2:.1f}°C "
            f"(centro da faixa de treinamento: {model_cfg['g_min']:.1f}°C – {model_cfg['g_max']:.1f}°C)."))
    mean_abs_val = None
    if use_mean:
        col_temp, _ = st.columns([1, 2])
        with col_temp:
            mean_abs_val = st.number_input("Temperatura média do frame (°C)",
                value=float((model_cfg['g_min'] + model_cfg['g_max']) / 2), step=0.5)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_upload, col_visualizacao = st.columns(2)
    with col_upload:
        st.markdown('<div class="section-title">Envio do arquivo térmico</div>', unsafe_allow_html=True)
        arquivo = st.file_uploader("Selecione um frame (PNG / JPG) ou vídeo (MP4 / AVI / MOV)",
            type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"], key=f"uploader_{st.session_state.upload_key}",)

        is_video = False
        if arquivo is not None:
            ext = Path(arquivo.name).suffix.lower()
            is_video = ext in {".mp4", ".avi", ".mov", ".mkv"}

        if arquivo is None:
            st.session_state.analisado     = False
            st.session_state.resultado_img = None
            st.session_state.nome_arquivo  = None

        # ════════════════════════════════════════════════════
        #  FLUXO — IMAGEM (dentro de col_upload)
        # ════════════════════════════════════════════════════
        if arquivo is not None and not is_video:
            if st.session_state.nome_arquivo != arquivo.name:
                st.session_state.analisado     = False
                st.session_state.resultado_img = None
                st.session_state.nome_arquivo  = arquivo.name

            b1, b2, _ = st.columns([1, 1, 2])
            with b1:
                clicar_analisar = st.button("Analisar frame", use_container_width=True, type="primary")
            with b2:
                clicar_limpar = st.button("Limpar", use_container_width=True)

            if clicar_limpar:
                st.session_state.upload_key   += 1
                st.session_state.analisado     = False
                st.session_state.resultado_img = None
                st.session_state.nome_arquivo  = None
                st.rerun()

            if clicar_analisar:
                pil_img  = Image.open(arquivo).convert("L")
                frame_np = np.array(pil_img)

                valido, motivo, detalhes = validar_dominio(frame_np, pil_original=Image.open(arquivo).convert("RGB"))

                if not valido:
                    st.session_state.resultado_img = {"valido": False, "motivo": motivo, "detalhes": detalhes,}
                else:
                    with st.spinner("Processando frame..."):
                        (overlay, score, is_anom, regioes, seg_probs,
                         pred_class, probs_cls, cam_map) = inferir_frame(
                            frame_np, cls_model, unet_model, model_cfg, device,
                            mean_abs=mean_abs_val, mask_cutoff=mask_cutoff_val)

                    # Prob map com stretch de contraste
                    probs_arr = np.clip(seg_probs, 0.0, 1.0)
                    p_min, p_max = probs_arr.min(), probs_arr.max()
                    probs_norm = (probs_arr - p_min) / (p_max - p_min) if p_max - p_min > 1e-6 else probs_arr
                    prob_rgb = (_cmap("hot")(probs_norm)[:, :, :3] * 255).astype(np.uint8)

                    # Grad-CAM visualização
                    cam_rgb = (_cmap("jet")(cam_map)[:, :, :3] * 255).astype(np.uint8)

                    buf_overlay = io.BytesIO()
                    Image.fromarray(overlay).save(buf_overlay, format="PNG")
                    buf_overlay.seek(0)

                    st.session_state.resultado_img = {
                        "valido"      : True,
                        "overlay"     : buf_overlay.getvalue(),
                        "prob_rgb"    : prob_rgb,
                        "cam_rgb"     : cam_rgb,
                        "score"       : score,
                        "is_anom"     : is_anom,
                        "regioes"     : regioes,
                        "detalhes"    : detalhes,
                        "mask_cutoff" : mask_cutoff_val,
                        "pred_class"  : pred_class,
                        "probs_cls"   : {n: float(p) for n, p in zip(model_cfg["class_names"], probs_cls)}}
                st.session_state.analisado = True

    # ── Pré-visualização na coluna de visualização ───────────────
    with col_visualizacao:
        if arquivo is not None and not is_video:
            pil_prev = Image.open(arquivo).convert("L")
            np_prev  = np.array(pil_prev)
            gray_norm_prev = np_prev.astype(np.float32) / 255.0
            viz_prev = (_cmap("inferno")(gray_norm_prev)[:, :, :3] * 255).astype(np.uint8)
            st.markdown('<div class="section-title">Pré-visualização</div>', unsafe_allow_html=True)
            st.image(viz_prev, width=300)

    # ── Resultado da análise abaixo das duas colunas ─────────────
    if arquivo is not None and not is_video and st.session_state.analisado and st.session_state.resultado_img:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        res = st.session_state.resultado_img
        st.markdown('<div class="section-title">Resultado da análise</div>', unsafe_allow_html=True)

        if not res["valido"]:
            st.markdown(f"""
            <div class="card-error">
                <div style="font-size:1.5rem;font-weight:800;color:#c0392b;">⛔ Imagem fora do domínio</div>
                <div style="margin-top:0.6rem;color:#5e6d7c;font-size:0.95rem;">{res["motivo"]}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            score     = res["score"]
            is_anom   = res["is_anom"]
            regioes   = res["regioes"]
            pred_class = res["pred_class"]
            probs_cls  = res["probs_cls"]
            bar_color  = "bar-fill-err" if score > 30 else ("bar-fill-warn" if score > 5 else "bar-fill-ok")
            bar_pct    = min(score, 100)
            badge_color = CLASS_COLORS.get(pred_class, "#6b8399")

            # ── Overlay + Grad-CAM lado a lado ──
            col_overlay, col_cam, col_extra = st.columns(3)

            with col_overlay:
                st.image(res["overlay"], width=350)

            with col_cam:
                # ── Veredito unificado abaixo das imagens ──
                conf_pct = int(round(float(probs_cls.get(pred_class, 0)) * 100))
                st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
                if is_anom:
                    _detail = (f"Classificador: <strong>{CLASS_LABELS.get(pred_class,'?').upper()}</strong> (confiança {conf_pct}%)")
                    st.markdown(f"""
                    <div class="card-alert">
                        <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
                            <div style="font-size:1.3rem;font-weight:800;color:#c0392b;">⚠️ Anomalia detectada</div>
                            <span style="background:{badge_color}22;color:{badge_color};border:1.5px solid {badge_color};
                                        font-size:0.9rem;font-weight:700;padding:0.2rem 0.8rem;border-radius:8px;">
                                {CLASS_LABELS.get(pred_class, pred_class).upper()}
                            </span>
                        </div>
                        <div style="margin-top:0.5rem;color:#5e6d7c;font-size:0.9rem;">{_detail}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if pred_class == "atencao":
                        _detail = f"Classificador: <strong>ATENÇÃO</strong> (confiança {conf_pct}%)"
                        _card_cls = "card-alert"
                        _icon = "🔶"
                        _title_color = "#e06c00"
                        _verdict_txt = "Monitorar"
                    else:
                        _detail = f"Classificador: <strong>NORMAL</strong> (confiança {conf_pct}%)"
                        _card_cls = "card-ok"
                        _icon = "✅"
                        _title_color = "#2a9d5c"
                        _verdict_txt = "Frame normal"
                    st.markdown(f"""
                    <div class="{_card_cls}">
                        <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
                            <div style="font-size:1.3rem;font-weight:800;color:{_title_color};">{_icon} {_verdict_txt}</div>
                            <span style="background:{badge_color}22;color:{badge_color};border:1.5px solid {badge_color};
                                        font-size:0.9rem;font-weight:700;padding:0.2rem 0.8rem;border-radius:8px;">
                                {CLASS_LABELS.get(pred_class, pred_class).upper()}
                            </span>
                        </div>
                        <div style="margin-top:0.5rem;color:#5e6d7c;font-size:0.9rem;">{_detail}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            if st.button("Analisar outro arquivo", use_container_width=False):
                st.session_state.upload_key   += 1
                st.session_state.analisado     = False
                st.session_state.resultado_img = None
                st.session_state.nome_arquivo  = None
                st.rerun()

    # ════════════════════════════════════════════════════
    #  FLUXO — VÍDEO
    # ════════════════════════════════════════════════════
    if arquivo is not None and is_video:
        uploaded_vid = arquivo
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_vid.name).suffix, delete=False) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        cap   = cv2.VideoCapture(tmp_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_v   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_v   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, first_frame = cap.read()
        cap.release()

        st.markdown(f"""
        <div class="card-info" style="margin-bottom:1rem;">
            <div style="font-weight:700;font-size:1.05rem;">📹 {uploaded_vid.name}</div>
            <div style="color:#5e6d7c;font-size:0.9rem;margin-top:0.4rem;">
                {total} frames &nbsp;·&nbsp; {fps:.1f} fps &nbsp;·&nbsp; {w_v}×{h_v} px
            </div>
        </div>
        """, unsafe_allow_html=True)

        if ret:
            first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            valido_prev, motivo_prev, _ = validar_dominio(first_gray)
            col_prev, col_prev_msg = st.columns([1, 1])
            with col_prev:
                st.image(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                         caption="Preview — primeiro frame", width=380)
            with col_prev_msg:
                if not valido_prev:
                    st.markdown(f"""
                    <div class="card-error" style="margin-top:0.5rem;">
                        <div style="font-weight:700;color:#c0392b;">⚠️ Primeiro frame fora do domínio</div>
                        <div style="color:#5e6d7c;font-size:0.88rem;margin-top:0.4rem;">{motivo_prev}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="card-ok" style="margin-top:0.5rem;">
                        <div style="font-weight:700;color:#2a9d5c;">✅ Domínio válido</div>
                        <div style="color:#5e6d7c;font-size:0.88rem;margin-top:0.4rem;">
                            Pronto para processar.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        bv1, bv2, _ = st.columns([1, 1, 5])
        with bv1:
            clicar_video = st.button("▶️ Processar vídeo", use_container_width=True, type="primary")

        if clicar_video:
            prog = st.progress(0.0, text="Iniciando...")
            with st.spinner("Processando vídeo..."):
                video_bytes, relatorio, rejeitados = processar_video(
                    tmp_path, cls_model, unet_model, model_cfg, device,
                    mean_abs=mean_abs_val, mask_cutoff=mask_cutoff_val,
                    progress_bar=prog)
            prog.progress(1.0, text="Concluído!")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            validos = [r for r in relatorio if r.get("dominio_valido")]
            n_anom  = sum(1 for r in validos if r["tem_anomalia"])
            scores  = [r["anomalia_pct"] for r in validos]

            # Distribuição de classes
            from collections import Counter
            class_dist = Counter(r.get("classe") for r in validos if r.get("classe"))

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Resultado do processamento</div>', unsafe_allow_html=True)

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            def _metric_card(col, label, value, sub=""):
                col.markdown(f"""
                <div class="card-soft" style="text-align:center;padding:0.9rem 0.5rem;">
                    <div class="small-muted">{label}</div>
                    <div style="font-size:1.6rem;font-weight:800;color:#1c2b3a;margin-top:0.2rem;">{value}</div>
                    <div class="small-muted">{sub}</div>
                </div>""", unsafe_allow_html=True)

            _metric_card(mc1, "Total frames", total)
            _metric_card(mc2, "Fora do domínio", len(rejeitados))
            _metric_card(mc3, "Processados", len(validos))
            _metric_card(mc4, "Anômalos", n_anom, f"{100*n_anom/max(len(validos),1):.1f}%" if validos else "—")
            _metric_card(mc5, "Score médio", f"{np.mean(scores):.1f}%" if scores else "—")

            # Distribuição de classes
            if class_dist:
                st.markdown('<div class="section-title" style="margin-top:1.2rem;">Distribuição de classes (Classificador)</div>', unsafe_allow_html=True)
                cols_cls = st.columns(len(CLASS_NAMES))
                for i, cn in enumerate(CLASS_NAMES):
                    count = class_dist.get(cn, 0)
                    pct   = 100 * count / max(len(validos), 1)
                    color = CLASS_COLORS.get(cn, "#888")
                    cols_cls[i].markdown(f"""
                    <div class="card-soft" style="text-align:center;padding:0.9rem 0.5rem;">
                        <div class="small-muted">{CLASS_LABELS.get(cn, cn)}</div>
                        <div style="font-size:1.5rem;font-weight:800;color:{color};">{count}</div>
                        <div class="small-muted">{pct:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

            if scores:
                st.markdown('<div class="section-title" style="margin-top:1.2rem;">Score de anomalia por frame</div>', unsafe_allow_html=True)
                st.line_chart({"Score (%)": scores}, height=200)

            st.download_button("⬇️ Baixar vídeo anotado", data=video_bytes, file_name=f"anomalia_{uploaded_vid.name}",
                mime="video/mp4", use_container_width=False)

            if n_anom:
                st.markdown('<div class="section-title" style="margin-top:1rem;">Frames anômalos</div>', unsafe_allow_html=True)
                st.dataframe([r for r in validos if r["tem_anomalia"]], use_container_width=True,
                    hide_index=True)

# ══════════════════════════════════════════════════════════
#  ABA — SOBRE O MODELO
# ══════════════════════════════════════════════════════════
with aba_modelo:
 
    # ── Cabeçalho da aba ──────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1c2b3a 0%,#243447 100%);border-radius:16px;
                padding:2rem 2.4rem;margin-bottom:1.8rem;border-left:5px solid #e06c00;">
        <div style="font-size:1.5rem;font-weight:800;color:#f0f4f8;margin-bottom:0.4rem;">
            🔬 Sobre o Modelo e o Dataset
        </div>
        <div style="color:#8daabf;font-size:0.95rem;line-height:1.6;">
            Este sistema detecta anomalias térmicas na superfície do <strong style="color:#e0c070;">Polisher</strong>
            monitorando o processo de polimento de aço via câmera Pi térmica.
            Os modelos foram treinados com dados reais do projeto
            <strong style="color:#e0c070;">Pitch-In LBAM</strong> da Universidade de Sheffield,
            publicados no Kaggle pelo pesquisador <strong style="color:#e0c070;">D. B. Miller et al.</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════════════
    #  SEÇÃO 1 — DATASET
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Origem dos dados — Pitch-In LBAM Thermal Imaging Dataset</div>', unsafe_allow_html=True)
 
    col_ds1, col_ds2 = st.columns([3, 2])
    with col_ds1:
        st.markdown("""
        <div class="card-info" style="margin-bottom:1rem;">
            <div style="font-weight:700;font-size:1rem;color:#1c2b3a;margin-bottom:0.6rem;">
                📦 O que é o dataset?
            </div>
            <div style="color:#5e6d7c;font-size:0.9rem;line-height:1.7;">
                O <strong>Pitch-In LBAM Thermal Imaging Dataset</strong> é um conjunto de dados
                de imagens térmicas de processo de fabricação aditiva a laser (<em>Laser-Based
                Additive Manufacturing</em>), coletado pela Universidade de Sheffield em parceria
                com a <strong>Rolls-Royce</strong> no âmbito do projeto <em>Pitch-In</em> de IoT
                industrial.<br><br>
                Os dados são gravados por uma <strong>câmera Pi térmica</strong> acoplada a uma
                bancada de polimento de superfícies de aço (<em>Polisher</em>),
                armazenados no formato <code>HDF5</code> com a estrutura
                <code>pi-camera-1[H × W × T]</code>, em que cada "fatia" ao longo do eixo
                temporal é um frame de temperatura absoluta por pixel, em graus Celsius.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card-soft" style="margin-bottom:1rem;">
            <div style="font-weight:700;font-size:0.95rem;color:#1c2b3a;margin-bottom:0.5rem;">
                🔗 Referência
            </div>
            <div style="color:#5e6d7c;font-size:0.88rem;line-height:1.65;">
                Miller, D. B., Song, B., Farnsworth, M. &amp; Tiwari, D. (2021).
                <em>Pitch-In LBAM Thermal Imaging Dataset</em>. Kaggle.<br>
                <a href="https://www.kaggle.com/datasets/dbmiller/pitchin-lbam-thermal-imaging-dataset"
                   target="_blank"
                   style="color:#1a6fb5;font-size:0.85rem;word-break:break-all;">
                    kaggle.com/datasets/dbmiller/pitchin-lbam-thermal-imaging-dataset
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    with col_ds2:
        st.markdown("""
        <div class="card-soft" style="margin-bottom:0.7rem;">
            <div class="section-title" style="margin-bottom:0.5rem;">Características do arquivo bruto</div>
            <table style="width:100%;font-size:0.87rem;border-collapse:collapse;">
                <tr><td style="color:#5e6d7c;padding:4px 0;">Formato</td>
                    <td style="text-align:right;"><code>HDF5</code></td></tr>
                <tr><td style="color:#5e6d7c;padding:4px 0;">Dataset interno</td>
                    <td style="text-align:right;"><code>pi-camera-1</code></td></tr>
                <tr><td style="color:#5e6d7c;padding:4px 0;">Dimensões</td>
                    <td style="text-align:right;"><code>H × W × frames</code></td></tr>
                <tr><td style="color:#5e6d7c;padding:4px 0;">Unidade</td>
                    <td style="text-align:right;"><code>°C / pixel</code></td></tr>
                <tr><td style="color:#5e6d7c;padding:4px 0;">Câmera</td>
                    <td style="text-align:right;"><code>Pi Thermal Cam</code></td></tr>
                <tr><td style="color:#5e6d7c;padding:4px 0;">Resolução saída</td>
                    <td style="text-align:right;"><code>320 × 240 px</code></td></tr>
                <tr><td style="color:#5e6d7c;padding:4px 0;">Frames extraídos</td>
                    <td style="text-align:right;"><code>2 000</code></td></tr>
                <tr><td style="color:#5e6d7c;padding:4px 0;">Colormap visualização</td>
                    <td style="text-align:right;"><code>inferno</code></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════════════
    #  SEÇÃO 2 — EXTRAÇÃO E LABELING
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Como os dados foram preparados</div>', unsafe_allow_html=True)
 
    col_ext1, col_ext2 = st.columns(2)
    with col_ext1:
        st.markdown("""
        <div class="card-soft" style="margin-bottom:0.75rem;">
            <div style="font-weight:700;font-size:0.95rem;color:#1c2b3a;margin-bottom:0.4rem;">
                🗂️ Amostragem estratificada
            </div>
            <div style="color:#5e6d7c;font-size:0.87rem;line-height:1.65;">
                Para evitar redundância temporal e garantir diversidade de padrões,
                os 2 000 frames foram selecionados em quatro estratos:
                <ul style="margin:0.4rem 0 0 1rem;padding:0;">
                    <li><strong>40%</strong> — cobertura temporal uniforme</li>
                    <li><strong>25%</strong> — frames de alta variação espacial (alto desvio padrão)</li>
                    <li><strong>25%</strong> — frames de alto delta entre frames consecutivos</li>
                    <li><strong>10%</strong> — frames em temperaturas extremas (mín/máx)</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    with col_ext2:
        st.markdown("""
        <div class="card-soft" style="margin-bottom:0.75rem;">
            <div style="font-weight:700;font-size:0.95rem;color:#1c2b3a;margin-bottom:0.4rem;">
                🏷️ Labels automáticos por temperatura absoluta
            </div>
            <div style="color:#5e6d7c;font-size:0.87rem;line-height:1.65;">
                Cada frame recebe um label com base na <strong>temperatura média absoluta</strong>
                do frame em relação à distribuição global, via percentis:
                <ul style="margin:0.4rem 0 0 1rem;padding:0;">
                    <li><strong>Normal</strong> — abaixo do percentil 75</li>
                    <li><strong>Atenção</strong> — entre percentil 75 e 95</li>
                    <li><strong>Crítico</strong> — acima do percentil 95</li>
                </ul>
                Os limiares exatos (em °C) são derivados da distribuição real
                do arquivo HDF5 e salvos nos checkpoints.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════════════
    #  SEÇÃO 3 — MÉTRICAS DOS MODELOS
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Desempenho dos modelos</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
 
    val_acc = model_cfg["val_acc"]
    val_iou = model_cfg["val_iou"]
    val_f1  = model_cfg["val_f1"]
    acc_fmt = f"{val_acc*100:.1f}%" if isinstance(val_acc, float) else val_acc
    iou_fmt = f"{val_iou*100:.1f}%" if isinstance(val_iou, float) else val_iou
    f1_fmt  = f"{val_f1*100:.1f}%"  if isinstance(val_f1,  float) else val_f1
 
    mc1.markdown(f"""
    <div class="card-soft" style="text-align:center;padding:1rem 0.5rem;">
        <div class="small-muted">Acurácia (validação)</div>
        <div style="font-size:1.8rem;font-weight:800;color:#1c2b3a;">{acc_fmt}</div>
        <div class="small-muted">EfficientNet-B3 — frames classificados corretamente</div>
    </div>""", unsafe_allow_html=True)
 
    mc2.markdown(f"""
    <div class="card-soft" style="text-align:center;padding:1rem 0.5rem;">
        <div class="small-muted">IoU (segmentação)</div>
        <div style="font-size:1.8rem;font-weight:800;color:#1c2b3a;">{iou_fmt}</div>
        <div class="small-muted">U-Net — sobreposição máscara predita vs. pseudo-mask</div>
    </div>""", unsafe_allow_html=True)
 
    mc3.markdown(f"""
    <div class="card-soft" style="text-align:center;padding:1rem 0.5rem;">
        <div class="small-muted">F1-Score (segmentação)</div>
        <div style="font-size:1.8rem;font-weight:800;color:#1c2b3a;">{f1_fmt}</div>
        <div class="small-muted">U-Net — equilíbrio precisão/recall por pixel</div>
    </div>""", unsafe_allow_html=True)
 
    mc4.markdown(f"""
    <div class="card-soft" style="text-align:center;padding:1rem 0.5rem;">
        <div class="small-muted">Épocas treinadas</div>
        <div style="font-size:1.8rem;font-weight:800;color:#1c2b3a;">{model_cfg["epoch_cls"]} / {model_cfg["epoch_unet"]}</div>
        <div class="small-muted">Classificador / U-Net</div>
    </div>""", unsafe_allow_html=True)
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════════════
    #  SEÇÃO 4 — ARQUITETURA DO PIPELINE
    # ══════════════════════════════════════════════════════
    col_desc, col_pipe = st.columns(2)
    with col_desc:
        st.markdown('<div class="section-title">Arquitetura do pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card-soft" style="margin-bottom:0.8rem;border-left:4px solid #1a6fb5;">
            <div style="font-weight:700;font-size:0.95rem;color:#1c2b3a;margin-bottom:0.4rem;">
                Modelo 1 — Classificador (EfficientNet-B3)
            </div>
            <div style="color:#5e6d7c;font-size:0.87rem;line-height:1.65;">
                Recebe o frame térmico em <strong>2 canais</strong>: o padrão espacial
                em grayscale (canal 0) e a temperatura média absoluta normalizada globalmente
                (canal 1). Classifica o frame em <strong>Normal / Atenção / Crítico</strong>
                e gera um mapa <strong>Grad-CAM</strong> que localiza as regiões mais
                influentes — sem nenhuma anotação manual.
            </div>
        </div>
        <div class="card-soft" style="border-left:4px solid #e06c00;">
            <div style="font-weight:700;font-size:0.95rem;color:#1c2b3a;margin-bottom:0.4rem;">
                Modelo 2 — Segmentador (U-Net, 3 canais)
            </div>
            <div style="color:#5e6d7c;font-size:0.87rem;line-height:1.65;">
                Recebe <strong>3 canais</strong>: frame spatial, temperatura normalizada
                e o mapa Grad-CAM do classificador. Produz uma
                <strong>máscara pixel-a-pixel</strong> de probabilidade de anomalia,
                refinando a localização com precisão geométrica. As pseudo-masks de
                treinamento são derivadas automaticamente do Grad-CAM — sem supervisão manual.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    with col_pipe:
        st.markdown('<div class="section-title">Etapas do pipeline</div>', unsafe_allow_html=True)
        etapas = [
            ("1. Captura do frame", "A câmera Pi térmica registra a temperatura por pixel da superfície do Polisher em formato HDF5 (°C/pixel)."),
            ("2. Pré-processamento", "O frame é convertido para PNG grayscale 320×240, normalizado localmente (canal spatial) e globalmente (canal temperatura)."),
            ("3. Classificação + Grad-CAM", "O EfficientNet-B3 classifica o frame e gera um mapa de ativação que localiza as regiões determinantes — sem anotações."),
            ("4. Segmentação (U-Net)", "A U-Net recebe os 3 canais e produz uma máscara de probabilidade por pixel, identificando exatamente as regiões anômalas."),
            ("5. Resultado visual", "Contornos coloridos (verde / laranja / vermelho) e bounding boxes são sobrepostos ao frame com colormap inferno."),
        ]
        for titulo, desc in etapas:
            st.markdown(f"""
            <div class="card-soft" style="margin-bottom:0.75rem;">
                <div style="font-weight:700;font-size:0.95rem;">{titulo}</div>
                <div style="color:#5e6d7c;font-size:0.86rem;margin-top:0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════════════
    #  SEÇÃO 5 — CLASSES
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">O que cada classe significa</div>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)
    for col, cls, cor, titulo, desc in [
        (cc1, "normal",  "#2a9d5c", "✅ Normal",
         "Temperatura e padrão espacial dentro do esperado para o processo. Nenhuma ação necessária."),
        (cc2, "atencao", "#e06c00", "🔶 Atenção",
         "Temperatura acima do percentil 75 da distribuição ou padrão incomum. Recomenda-se monitorar os próximos frames."),
        (cc3, "critico", "#c0392b", "🔴 Crítico",
         "Temperatura acima do percentil 95 — anomalia térmica significativa. Verificação imediata da superfície é recomendada."),
    ]:
        col.markdown(f"""
        <div class="card-soft" style="border-left:4px solid {cor};padding:1rem;height:100%;">
            <div style="font-weight:800;font-size:1rem;color:{cor};">{titulo}</div>
            <div style="color:#5e6d7c;font-size:0.87rem;margin-top:0.4rem;">{desc}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════════════
    #  SEÇÃO 6 — PARÂMETROS TÉCNICOS
    # ══════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Parâmetros técnicos do treinamento</div>', unsafe_allow_html=True)
    cfg_items = [
        ("Resolução dos frames",      f"{model_cfg['img_size'][1]} × {model_cfg['img_size'][0]} pixels"),
        ("Faixa de temperatura",      f"{model_cfg['g_min']:.1f}°C  →  {model_cfg['g_max']:.1f}°C"),
        ("Limiar classe Atenção",     f"acima de {model_cfg['thresh_atencao']:.1f}°C (p75)" if model_cfg['thresh_atencao'] else "—"),
        ("Limiar classe Crítico",     f"acima de {model_cfg['thresh_critico']:.1f}°C (p95)" if model_cfg['thresh_critico'] else "—"),
        ("Percentil Grad-CAM",        f"Top {model_cfg['gradcam_thresh']}% de ativação → pseudo-mask"),
        ("Backbone classificador",    f"EfficientNet-B3 ({model_cfg['backend']})"),
        ("Segmentador",               "U-Net — 3 canais de entrada (spatial + temp + Grad-CAM)"),
        ("Divisão treino/val/teste",  "80% / 10% / 10%"),
        ("Dropout classificador",     "0.40 (head) — previne overfitting"),
        ("Dropout U-Net",             "0.20 (encoder/bottleneck)"),
        ("Épocas máx. classificador", "50 (early stopping patience=12)"),
        ("Épocas máx. U-Net",         "40 (early stopping patience=12)"),
        ("Hardware de inferência",    str(device).upper()),
        ("Fonte dos dados",           "Pitch-In LBAM — Univ. Sheffield / Kaggle"),
    ]
    ci1, ci2 = st.columns(2)
    for i, (k, v) in enumerate(cfg_items):
        with (ci1 if i % 2 == 0 else ci2):
            st.markdown(f"""
            <div class="card-soft" style="margin-bottom:0.55rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.3rem;">
                <span style="font-weight:600;color:#3a4f63;">{k}</span>
                <code style="font-size:0.85rem;">{v}</code>
            </div>
            """, unsafe_allow_html=True)
 
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════════════
    #  SEÇÃO 7 — AVISO DE LIMITAÇÕES
    # ══════════════════════════════════════════════════════
    st.markdown("""
    <div class="card-alert">
        <div style="font-weight:700;font-size:0.95rem;color:#b85c00;margin-bottom:0.4rem;">
            ⚠️ Limitações e considerações importantes
        </div>
        <div style="color:#5e6d7c;font-size:0.87rem;line-height:1.7;">
            <strong>Labels automáticos:</strong> as classes (Normal / Atenção / Crítico) são definidas
            por percentis da distribuição de temperatura do próprio dataset — não por um especialista
            de processo. Os limiares devem ser validados com conhecimento do domínio real.<br>
            <strong>Pseudo-masks:</strong> a U-Net é treinada com máscaras derivadas automaticamente
            do Grad-CAM, não de anotações humanas pixel-a-pixel. Isso torna o treinamento viável
            sem custo de anotação, mas pode introduzir ruído nas bordas das regiões anômalas.<br>
            <strong>Domínio:</strong> o sistema foi treinado exclusivamente com dados de câmera Pi
            térmica em processo de polimento de aço. Imagens de outras câmeras ou processos podem
            não ser representadas adequadamente pelo modelo.
        </div>
    </div>
    """, unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════
#  ABA — EXEMPLOS
# ══════════════════════════════════════════════════════════
with aba_exemplos:
    st.markdown("## Exemplos de frames térmicos")
    st.markdown("Baixe os frames abaixo e envie na aba **Análise** para testar o sistema.")
    _exemplos = [{"arquivo": "exemplo1.png"},{"arquivo": "exemplo2.png"},{"arquivo": "exemplo3.png"}]
    cols_ex = st.columns(3)
    for col, ex in zip(cols_ex, _exemplos):
        _path = Path("frames_exemplos") / ex["arquivo"]
        _fallback = Path(ex["arquivo"])  # raiz do projeto
        _img_path = _path if _path.exists() else (_fallback if _fallback.exists() else None)
        with col:
            if _img_path:
                st.image(str(_img_path), use_container_width=True)
                with open(str(_img_path), "rb") as _f:
                    _bytes = _f.read()
                st.download_button(label=f"⬇️ Download {ex['arquivo']}", data=_bytes,
                    file_name=ex["arquivo"], mime="image/png", use_container_width=True,
                    key=f"dl_{ex['arquivo']}")
            else:
                st.info(f"Arquivo não encontrado: {ex['arquivo']}")

# ─────────────────────────────────────────────────────────
#  RODAPÉ
# ─────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="small-muted" style="text-align:center;">'
    'Monitoramento térmico de superfícies de aço'
    '</div>', unsafe_allow_html=True)