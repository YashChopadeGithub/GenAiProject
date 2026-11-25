"""Streamlit entrypoint for Loan Document Processing using Generative AI.

Designed for both local runs and Streamlit Community Cloud:
- Downloads missing inference checkpoints from Hugging Face Hub on demand.
- Caches at most one heavy model in memory at a time (max_entries=1) to stay
  within Streamlit Community Cloud memory limits.
"""

import base64
import gc
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForTokenClassification, AutoTokenizer

from api.extractor import DocumentExtractor
from autoencoder.config import CHECKPOINT_DIR, DATA_DIR, DEVICE
from autoencoder.dataset import ResizeWithPad
from autoencoder.evaluate import compute_psnr, load_best_model
from autoencoder.losses import SSIMLoss
from autoencoder.model import DocumentVAE

HF_WEIGHTS_REPO = os.environ.get("HF_WEIGHTS_REPO", "yashchopade05/loan-document-ai-weights")
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "samples")

st.set_page_config(
    page_title="Loan Document AI",
    page_icon="📄",
    layout="wide",
)


def _ensure_checkpoint(rel_path: str) -> str:
    """Return local checkpoint path, downloading from Hugging Face Hub if absent."""
    local_path = os.path.join(CHECKPOINT_DIR, rel_path)
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    hf_hub_download(
        repo_id=HF_WEIGHTS_REPO,
        filename=rel_path.replace("\\", "/"),
        local_dir=CHECKPOINT_DIR,
    )
    return local_path


def _ensure_transformer_checkpoint() -> str:
    """Ensure all LayoutLM checkpoint files exist locally and return model directory."""
    rel_dir = os.path.join("transformer_model", "best_model")
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for fname in required_files:
        _ensure_checkpoint(os.path.join(rel_dir, fname))
    return os.path.join(CHECKPOINT_DIR, rel_dir)


@st.cache_resource(show_spinner="Initializing OCR & document extractor...")
def get_doc_extractor() -> DocumentExtractor:
    """Keep the lightweight RapidOCR/rule extractor cached across tabs."""
    return DocumentExtractor(DATA_DIR)


@st.cache_resource(max_entries=1, show_spinner="Loading model weights...")
def load_stage_model(stage: str) -> Dict[str, Any]:
    """Load one heavy generative/transformer model at a time to conserve RAM."""
    gc.collect()
    if stage == "ae":
        _ensure_checkpoint("best_autoencoder.pth")
        model, _ = load_best_model()
        ssim_fn = SSIMLoss().to(DEVICE)
        return {"model": model, "ssim_fn": ssim_fn}

    if stage == "vae":
        ckpt_path = _ensure_checkpoint("best_vae.pth")
        vae = DocumentVAE().to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        vae.load_state_dict(ckpt["model_state_dict"])
        vae.eval()
        return {"model": vae}

    if stage == "transformer":
        model_dir = _ensure_transformer_checkpoint()
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir).to(DEVICE)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}

    if stage == "cgan":
        from cgan.synthesizer import DocumentSynthesizer

        ckpt_path = _ensure_checkpoint("best_cgan_generator.pth")
        return {"synthesizer": DocumentSynthesizer(checkpoint_path=ckpt_path)}

    if stage == "diffusion":
        from diffusion.synthesizer import DiffusionDocumentSynthesizer

        _ensure_checkpoint("best_diffusion_unet_ema.pth")
        return {"synthesizer": DiffusionDocumentSynthesizer(use_ema=True)}

    raise ValueError(f"Unknown model stage: {stage}")


def _decode_data_url(data_url: str) -> Image.Image:
    """Decode a data:image/...;base64 string into a PIL Image."""
    _, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _list_sample_files() -> List[str]:
    if not os.path.exists(SAMPLES_DIR):
        return []
    return sorted(
        f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def _load_active_document() -> Tuple[Optional[Image.Image], Optional[bytes], str]:
    """Sidebar document source selector: uploaded file or bundled sample receipt."""
    st.sidebar.header("Document Input")
    uploaded = st.sidebar.file_uploader(
        "Upload a document image",
        type=["jpg", "jpeg", "png"],
        help="Upload any scanned receipt or loan document image.",
    )
    if uploaded is not None:
        raw = uploaded.getvalue()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        st.sidebar.image(img, caption=f"Active: {uploaded.name}", use_container_width=True)
        return img, raw, uploaded.name

    samples = _list_sample_files()
    if not samples:
        st.sidebar.warning("No sample images found in static/samples/. Please upload an image.")
        return None, None, ""

    default_idx = samples.index("210.jpg") if "210.jpg" in samples else 0
    selected = st.sidebar.selectbox("Or choose a sample receipt", samples, index=default_idx)
    sample_path = os.path.join(SAMPLES_DIR, selected)
    with open(sample_path, "rb") as f:
        raw = f.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    st.sidebar.image(img, caption=f"Sample: {selected}", use_container_width=True)
    return img, raw, selected


def _run_transformer_extraction(
    image: Image.Image, raw_bytes: bytes, filename: str
) -> Dict[str, Any]:
    """Run OCR + LayoutLM token classification + rule-based risk verification."""
    extractor = get_doc_extractor()
    bundle = load_stage_model("transformer")
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]

    width, height = image.size
    box_path, key_path, base_name = extractor.find_box_and_key(filename, raw_bytes, image)
    if box_path and os.path.exists(box_path):
        words, boxes, raw_lines = extractor.parse_box_file(box_path, width, height)
    else:
        words, boxes, raw_lines = extractor.process_unseen_image(image)

    gt_data: Dict[str, Any] = {}
    if key_path and os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8", errors="ignore") as f:
            gt_data = json.load(f)

    transformer_extracted: Dict[str, str] = {}
    transformer_confs: Dict[str, float] = {}

    if words:
        encoding = tokenizer(
            words,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=512,
        )
        word_ids = encoding.word_ids()
        bbox_list = [
            [0, 0, 0, 0] if (wid is None or wid >= len(boxes)) else boxes[wid]
            for wid in word_ids
        ]
        encoding["bbox"] = torch.tensor([bbox_list])
        encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = outputs.logits.argmax(-1).squeeze().tolist()
        confs = probs.max(-1).values.squeeze().tolist()
        id2label = model.config.id2label

        raw_ent: Dict[str, List[str]] = {"COMPANY": [], "ADDRESS": [], "DATE": [], "TOTAL": []}
        raw_c: Dict[str, List[float]] = {"COMPANY": [], "ADDRESS": [], "DATE": [], "TOTAL": []}

        cur_ent: Optional[str] = None
        cur_words: List[str] = []
        cur_confs: List[float] = []
        prev_wid: Optional[int] = None

        for i, wid in enumerate(word_ids):
            if wid is None or wid == prev_wid:
                continue
            label = id2label[preds[i]]
            word = words[wid]
            conf = confs[i]

            if label.startswith("B-"):
                if cur_ent:
                    raw_ent[cur_ent].append(" ".join(cur_words))
                    raw_c[cur_ent].append(sum(cur_confs) / len(cur_confs))
                cur_ent = label[2:]
                cur_words = [word]
                cur_confs = [conf]
            elif label.startswith("I-") and cur_ent == label[2:]:
                cur_words.append(word)
                cur_confs.append(conf)
            else:
                if cur_ent:
                    raw_ent[cur_ent].append(" ".join(cur_words))
                    raw_c[cur_ent].append(sum(cur_confs) / len(cur_confs))
                    cur_ent = None
                    cur_words = []
                    cur_confs = []
            prev_wid = wid

        if cur_ent:
            raw_ent[cur_ent].append(" ".join(cur_words))
            raw_c[cur_ent].append(sum(cur_confs) / len(cur_confs))

        for k, vals in raw_ent.items():
            if vals:
                transformer_extracted[k] = " ".join(vals).strip()
                transformer_confs[k] = round(sum(raw_c[k]) / len(raw_c[k]), 2)

    fields, confidences, risks = extractor.extract_rich_metadata(
        raw_lines, gt_data, transformer_extracted, transformer_confs
    )
    return {
        "fields": fields,
        "confidences": confidences,
        "risks": risks,
        "matched_document": base_name,
        "word_count": len(words),
    }


def main() -> None:
    st.title("📄 Loan Document Processing using Generative AI")
    st.caption(
        "Research prototype combining Conv-Autoencoder cleanup, Conv-VAE scenario sampling, "
        "LayoutLM field extraction + rule-based risk auditing, cGAN, and DDIM Diffusion synthesis."
    )

    image, raw_bytes, filename = _load_active_document()
    st.sidebar.divider()
    stage = st.sidebar.radio(
        "Pipeline Stage",
        [
            "1. Autoencoder Cleanup",
            "2. VAE Scenario Sampling",
            "3. Transformer Extraction & Risk Audit",
            "4. Conditional GAN Synthesis",
            "5. Diffusion (DDIM) Synthesis",
        ],
    )
    st.sidebar.info(f"Running on `{DEVICE}` | Weights: `{HF_WEIGHTS_REPO}`")

    if image is None or raw_bytes is None:
        st.warning("Upload a document image in the sidebar to begin.")
        return

    resize_pad = ResizeWithPad(512)
    to_tensor = T.ToTensor()

    if stage == "1. Autoencoder Cleanup":
        st.subheader("Model 1 — Convolutional Autoencoder Reconstruction")
        if st.button("Run Autoencoder Cleanup", type="primary"):
            bundle = load_stage_model("ae")
            padded = resize_pad(image)
            inp = to_tensor(padded).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = bundle["model"](inp)
                ssim_val = float(bundle["ssim_fn"].compute_ssim_value(out, inp))
                psnr_val = float(compute_psnr(out, inp))
            recon = to_pil_image(out.squeeze(0).cpu())

            m1, m2 = st.columns(2)
            m1.metric("SSIM", f"{ssim_val:.4f}")
            m2.metric("PSNR", f"{psnr_val:.2f} dB")

            c1, c2 = st.columns(2)
            c1.image(padded, caption="Original (512×512 Padded)", use_container_width=True)
            c2.image(recon, caption="AE Cleaned Reconstruction", use_container_width=True)

    elif stage == "2. VAE Scenario Sampling":
        st.subheader("Model 2 — Convolutional VAE Scenario Generation")
        temp = st.slider("Latent Temperature", min_value=0.1, max_value=1.0, value=0.35, step=0.05)
        if st.button("Generate 3 Latent Variations", type="primary"):
            bundle = load_stage_model("vae")
            padded = resize_pad(image)
            inp = to_tensor(padded).unsqueeze(0).to(DEVICE)
            samples: List[Image.Image] = []
            with torch.no_grad():
                for _ in range(3):
                    out, _, _ = bundle["model"](inp, temperature=temp, sample_mode=False)
                    samples.append(to_pil_image(out.squeeze(0).cpu()))

            cols = st.columns(4)
            cols[0].image(padded, caption="Original", use_container_width=True)
            for idx, s_img in enumerate(samples, start=1):
                cols[idx].image(s_img, caption=f"Variation {idx}", use_container_width=True)

    elif stage == "3. Transformer Extraction & Risk Audit":
        st.subheader("Model 3 — LayoutLM Field Extraction & Rule-Based Risk Flags")
        if st.button("Extract Fields & Run Risk Audit", type="primary"):
            result = _run_transformer_extraction(image, raw_bytes, filename)
            st.caption(
                f"Document match: `{result['matched_document']}` | "
                f"OCR tokens processed: `{result['word_count']}`"
            )

            col_left, col_right = st.columns([1.2, 1.0])
            with col_left:
                st.markdown("#### Extracted Structured Fields")
                rows = [
                    {
                        "Field": k,
                        "Value": v,
                        "Confidence": f"{int(round(result['confidences'].get(k, 0.0) * 100))}%",
                    }
                    for k, v in result["fields"].items()
                ]
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No fields could be extracted from this image.")

            with col_right:
                st.markdown("#### Underwriting Risk & Discrepancy Flags")
                risks = result["risks"]
                if not risks:
                    st.success("No rule-based discrepancies or risk flags triggered.")
                else:
                    for r in risks:
                        sev = r.get("severity", "low").upper()
                        msg = (
                            f"**[{sev}] {r.get('field', '')} — {r.get('issue', '')}**\n\n"
                            f"- Rule: {r.get('rule', '')}\n"
                            f"- Recommendation: {r.get('recommendation', '')}"
                        )
                        if sev == "HIGH":
                            st.error(msg)
                        elif sev == "MEDIUM":
                            st.warning(msg)
                        else:
                            st.info(msg)

    elif stage == "4. Conditional GAN Synthesis":
        st.subheader("Phase 2 — Conditional GAN (Pix2Pix) Document Synthesis")
        c_ctrl1, c_ctrl2 = st.columns(2)
        style = c_ctrl1.selectbox(
            "Style Preset",
            ["clean", "thermal_fade", "stamped", "fax_noise", "crumpled_scan", "ink_bleed"],
        )
        noise_level = c_ctrl2.slider(
            "Noise / Artifact Intensity", min_value=0.1, max_value=0.9, value=0.35, step=0.05
        )
        if st.button("Synthesize with cGAN", type="primary"):
            bundle = load_stage_model("cgan")
            extractor = get_doc_extractor()
            box_path, key_path, _ = extractor.find_box_and_key(filename, raw_bytes, image)
            res = bundle["synthesizer"].synthesize(
                image=image,
                box_path=box_path,
                key_path=key_path,
                style=style,
                noise_level=noise_level,
            )
            m = res["metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("SSIM", m["ssim"])
            mc2.metric("PSNR", f"{m['psnr']} dB")
            mc3.metric("FID (est.)", m["fid_score"])
            mc4.metric("OCR Readability", f"{m['ocr_readability']}%")

            ic1, ic2, ic3 = st.columns(3)
            ic1.image(_decode_data_url(res["images"]["original"]), caption="Original", use_container_width=True)
            ic2.image(_decode_data_url(res["images"]["condition"]), caption="Layout Condition Map", use_container_width=True)
            ic3.image(_decode_data_url(res["images"]["synthetic"]), caption=f"cGAN ({style})", use_container_width=True)

    elif stage == "5. Diffusion (DDIM) Synthesis":
        st.subheader("Phase 2 — DDPM / DDIM Diffusion Document Synthesis")
        d_ctrl1, d_ctrl2 = st.columns(2)
        cfg_scale = d_ctrl1.slider("CFG Scale", min_value=1.5, max_value=5.0, value=2.5, step=0.5)
        steps = d_ctrl2.slider("DDIM Sampling Steps", min_value=10, max_value=50, value=20, step=5)
        if st.button("Synthesize with Diffusion (DDIM)", type="primary"):
            bundle = load_stage_model("diffusion")
            extractor = get_doc_extractor()
            box_path, key_path, _ = extractor.find_box_and_key(filename, raw_bytes, image)
            res = bundle["synthesizer"].synthesize(
                image=image,
                box_path=box_path,
                key_path=key_path,
                cfg_scale=cfg_scale,
                sampling_steps=steps,
            )
            m = res["metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("SSIM", m["ssim"])
            mc2.metric("PSNR", f"{m['psnr']} dB")
            mc3.metric("FID (est.)", m["fid_score"])
            mc4.metric("Latency", f"{res['timing_ms']} ms")

            ic1, ic2, ic3 = st.columns(3)
            ic1.image(_decode_data_url(res["images"]["original"]), caption="Original", use_container_width=True)
            ic2.image(_decode_data_url(res["images"]["condition_map"]), caption="Layout Condition Map", use_container_width=True)
            ic3.image(_decode_data_url(res["images"]["synthesized"]), caption=f"Diffusion ({steps} steps)", use_container_width=True)


if __name__ == "__main__":
    main()

