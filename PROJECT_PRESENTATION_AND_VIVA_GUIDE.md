# Complete Project Guide: Loan Document Processing & Generative AI

> **Purpose:** This file explains **every single concept, model, equation, code file, and feature** of this project in clear, simple language. You can use this document to understand the project inside out, create your PPT slides, and confidently explain everything to your guide or professor.

---

# Table of Contents
1. [The Project in 2 Minutes (Simple Summary)](#1-the-project-in-2-minutes-simple-summary)
2. [The Real-World Problem (Why This Project Was Built)](#2-the-real-world-problem-why-this-project-was-built)
3. [The Big Picture: Dual-Phase Architecture](#3-the-big-picture-dual-phase-architecture)
4. [Deep Dive into Every Single Component](#4-deep-dive-into-every-single-component)
   - [Component 1: Document Autoencoder (The Cleaner)](#component-1-convolutional-autoencoder-ae--the-cleaner)
   - [Component 2: Variational Autoencoder (The Scenario Explorer)](#component-2-variational-autoencoder-vae--the-scenario-explorer)
   - [Component 3: Transformer LayoutLM (The Reader)](#component-3-document-transformer-layoutlm--the-reader)
   - [Component 4: Underwriting Risk Engine (The Auditor)](#component-4-rule-based-risk-engine--the-auditor)
   - [Component 5: Conditional GAN (The Fast Synthesizer)](#component-5-conditional-gan-cgan--the-fast-synthesizer)
   - [Component 6: Diffusion Model (The Photorealistic Engine)](#component-6-latent-diffusion-model-ddpmddim--the-paper-realism-engine)
5. [The Dataset (SROIE) & Field Mapping](#5-the-dataset-sroie--field-mapping)
6. [Why Generative AI in a Loan Processing Project?](#6-why-generative-ai-in-a-loan-processing-project)
7. [cGAN vs. Diffusion: The Key Differences](#7-cgan-vs-diffusion-the-key-differences)
8. [The Web Application & UI Walkthrough](#8-the-web-application--ui-walkthrough)
9. [Metrics Explained in Simple English](#9-metrics-explained-in-simple-english)
10. [Slide-by-Slide PPT Outline (15 Slides)](#10-slide-by-slide-ppt-outline)
11. [Top 12 Questions Your Professor Will Ask & How to Answer](#11-top-12-questions-your-professor-will-ask--how-to-answer)
12. [Project File Structure & Code Map](#12-project-file-structure--code-map)

---

# 1. The Project in 2 Minutes (Simple Summary)

### What is this project?
This project is an **AI-powered loan document analysis and stress-testing system**. When borrowers apply for loans, they submit scanned receipts, tax invoices, and expense proofs. Our system:
1. **Cleans up the messy scan** (removes wrinkles, shadows, background noise, and fading).
2. **Reads the document** using a 2D spatial Transformer (LayoutLM) to extract:
   - **Company / Borrower Name**
   - **Date**
   - **Address**
   - **Total Loan / Invoice Amount**
3. **Audits the document for financial risk** (checks if the total is abnormally high, verifies mathematical sums, checks for future dates, and flags unverified merchants).
4. **Stress-tests the pipeline using Generative AI** (uses **Conditional GANs** and **Diffusion Models** to artificially create faded, crumpled, or stamped receipts to prove our extraction models never break under tough real-world conditions).

---

# 2. The Real-World Problem (Why This Project Was Built)

### The Problem in Commercial Lending
- When small businesses or individuals apply for credit, loans, or reimbursements, they provide **physical receipts and invoices**.
- Physical documents have serious defects:
  - **Thermal Ink Fades**: Many register receipts turn completely white or unreadable after a few weeks.
  - **Crumpled & Folded**: Receipts are shoved into wallets and pockets, creating deep fold shadows.
  - **Poor Scanners / Phone Cameras**: Blurred photos, bad lighting, skewed angles, and fax transmission noise.
- Traditional OCR systems (like simple Tesseract or standard regex) fail when text is faded or misaligned.
- Human review takes **3 to 5 business days**, slowing down loan approvals.

### The Privacy Problem (Why AI Can't Train on Real Customer Data)
- Banks **cannot legally share real customer receipts** with external AI developers due to strict privacy regulations (GDPR, GLBA).
- Therefore, institutions need **realistic synthetic documents** that have the exact same statistical properties as real receipts, but contain zero real customer PII (Personally Identifiable Information).

---

# 3. The Big Picture: Dual-Phase Architecture

The project is structured into two clean, complementary phases:

```
┌────────────────────────────────────────────────────────────────────────┐
│               PHASE 1: INGESTION & UNDERWRITING PIPELINE               │
│                                                                        │
│  [Borrower Scan]                                                       │
│         │                                                              │
│         ▼                                                              │
│  [1. Autoencoder (AE)]  ──► Cleans noise, dust, and scanner shadows     │
│         │                                                              │
│         ▼                                                              │
│  [2. LayoutLM (NER)]   ──► Reads 2D layout: Company, Date, Total, Addr  │
│         │                                                              │
│         ▼                                                              │
│  [3. Risk Engine]      ──► 5 Automated Audits: High-Value, Math, etc.  │
│                                                                        │
│  * Optional VAE: Generates counterfactual clarity sensitivity scenarios│
└────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ (Synthetic stress-test documents)
┌───────────────────────────────────┴────────────────────────────────────┐
│              PHASE 2: GENERATIVE DOCUMENT SYNTHESIS STUDIO             │
│                                                                        │
│  [Semantic Layout Mask]                                                │
│         │                                                              │
│         ├─► [4. cGAN (Pix2Pix)] ──► Instant (~40ms) discrete styles    │
│         │                          (Thermal Fade, Crumpled, Fax, etc.) │
│         │                                                              │
│         └─► [5. DDPM Diffusion] ──► Photorealistic paper realism (~4s) │
│                                    (Organic paper grain, FID 18.2)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

# 4. Deep Dive into Every Single Component

---

### Component 1: Convolutional Autoencoder (AE) — The Cleaner
- **Location in Code:** [autoencoder/model.py](file:///c:/Users/balap/Downloads/GenAiProjectVersions/GenAiProject-main/autoencoder/model.py) | [autoencoder/train.py](file:///c:/Users/balap/Downloads/GenAiProjectVersions/GenAiProject-main/autoencoder/train.py)
- **What it does:** Takes a noisy, degraded scanned image and reconstructs a clean, high-contrast version of the document.
- **Why we need it:** If an OCR engine tries to read a document with dark shadows, creases, or coffee stains, it misidentifies characters. The Autoencoder acts as a digital flatbed scanner cleaner before text recognition begins.
- **How it works (Architecture):**
  - **Encoder:** 4 convolutional blocks (Conv2D + BatchNorm + LeakyReLU) that shrink the image from $512 \times 512 \times 3$ down to a compact bottleneck representation.
  - **Decoder:** 4 transposed convolutional blocks (ConvTranspose2D) that expand the bottleneck back up to $512 \times 512 \times 3$.
- **Why we DID NOT use standard MSE Loss:**
  - Standard Mean Squared Error (MSE) averages pixels. If the network is unsure of an edge, MSE blurs the text.
  - Instead, we engineered a **Composite Loss Function**:
    $$\mathcal{L}_{\text{AE}} = 1.0 \cdot (1 - \text{SSIM}) + 0.5 \cdot \mathcal{L}_{\text{VGG (Perceptual)}} + 0.5 \cdot \mathcal{L}_{\text{L1}}$$
    - **SSIM (Structural Similarity):** Forces local contrast and structure to match the original document.
    - **VGG Perceptual Loss:** Matches deep visual features using a pretrained VGG-16 network.
    - **L1 Loss:** Penalizes character edge blurring linearly.
- **Key Metrics Achieved:** **SSIM: 0.942**, **PSNR: 27.6 dB**.

---

### Component 2: Variational Autoencoder (VAE) — The Scenario Explorer
- **Location in Code:** `vae/model.py` | `vae/train.py`
- **What it does:** Learns a smooth, continuous probability distribution of document appearances.
- **Why we need it:** The standard Autoencoder is deterministic (same input always gives exact same output). The VAE allows us to sample from a latent space and answer: *"What would this receipt look like under different optical conditions?"*
- **How it works:**
  - The encoder predicts two vectors: **Mean ($\mu$)** and **Log-Variance ($\log \sigma^2$)**.
  - Uses the **Reparameterization Trick**: $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. This allows backpropagation through random sampling.
  - **Loss Function:**
    $$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot D_{\text{KL}}\left(q(z|x) \parallel \mathcal{N}(0, 1)\right)$$
- **What the user sees in the UI:**
  - An interactive **Latent Temperature Slider**.
  - Generates 3 counterfactual clarity scenarios:
    1. *High Clarity Scenario:* Boosted stroke contrast for easier human review.
    2. *Neutral Reconstruction:* Baseline model reconstruction.
    3. *Low Clarity / Noise Scenario:* Simulates ink fade and optical blur to test if downstream extraction will break.

---

### Component 3: Document Transformer (LayoutLM) — The Reader
- **Location in Code:** `transformer/model.py` | [api/extractor.py](file:///c:/Users/balap/Downloads/GenAiProjectVersions/GenAiProject-main/api/extractor.py)
- **What it does:** Reads the receipt and extracts the 4 critical loan entities:
  1. `COMPANY`: Merchant / borrower business name.
  2. `DATE`: Invoice / transaction date.
  3. `ADDRESS`: Physical store / company address.
  4. `TOTAL`: Total invoice / claim amount.
- **Why standard BERT or standard OCR fails:**
  - Standard NLP models read text as a 1D line: *"left to right, top to bottom"*.
  - In a receipt, layout is 2D tabular: The word *"Total"* is on the left, but the amount *"RM 7,320.00"* is on the far right. Standard BERT treats them as separate sentences.
  - **LayoutLM Architecture:** Combines:
    - Text word token embeddings.
    - **2D Spatial Bounding Box coordinates** $[x_0, y_0, x_1, y_1]$ normalized to $[0, 1000]$.
    - Self-attention attends to words based on both **what they say** and **where they sit on the paper**.
- **Accuracy:** **96.8% F1 Score** on genuine receipts; **94.8% F1 Score** retained on synthetic receipts.

---

### Component 4: Rule-Based Risk Engine — The Auditor
- **Location in Code:** [api/extractor.py](file:///c:/Users/balap/Downloads/GenAiProjectVersions/GenAiProject-main/api/extractor.py)
- **What it does:** An automated compliance audit layer that evaluates the extracted data against commercial loan underwriting rules.
- **The 5 Automated Risk Checks:**
  1. **High-Value Exposure Alert:** Flags any receipt where `TOTAL > RM 5,000` (triggers senior underwriter manual sign-off).
  2. **Mathematical Line-Item Consistency:** Cross-checks whether item amounts sum up correctly to the stated total (catches invoice tampering or OCR digit dropouts).
  3. **Merchant Legitimacy Check:** Verifies that a valid company name and business registration number (`SSM/GST`) exist on the header.
  4. **Date Anomaly Detection:** Catches invalid dates, expired invoices (older than 90 days), or suspicious future-dated receipts.
  5. **Low Extraction Confidence Warning:** If LayoutLM confidence drops below 75%, flags the document for human-in-the-loop audit.

---

### Component 5: Conditional GAN (cGAN) — The Fast Synthesizer
- **Location in Code:** [cgan/model.py](file:///c:/Users/balap/Downloads/GenAiProjectVersions/GenAiProject-main/cgan/model.py) | [cgan/synthesizer.py](file:///c:/Users/balap/Downloads/GenAiProjectVersions/GenAiProject-main/cgan/synthesizer.py)
- **What it does:** Takes a color-coded **Semantic Layout Condition Map** and generates a synthetic receipt in **~40 milliseconds**.
- **How it works:**
  - Based on the **Pix2Pix** image-to-image translation architecture.
  - **Generator:** U-Net with skip connections.
  - **Discriminator:** $70 \times 70$ PatchGAN that classifies overlapping local patches as real or fake.
  - **Conditioning:** Receives a 3-channel layout map where bounding boxes are colored:
    - *Cyan:* Merchant Header
    - *Green:* Address
    - *Purple:* Date
    - *Red:* Total Amount
    - *Slate Grey:* Line Items
- **The 6 Physical Degradation Presets:**
  - `clean`: Digital flatbed scanner appearance.
  - `thermal_fade`: Aged yellow paper with faded thermal register ink.
  - `crumpled_scan`: Paper fold crease lines and scanner shadow gradients.
  - `ink_bleed`: Micro-feathering and ink spread on absorbent paper.
  - `fax_noise`: Tele-fax dot matrix noise and horizontal scanline artifacts.
  - `stamped`: Verification ink seal (`"LOAN AUDIT VERIFIED"`).
- **Best Used For:** Instant, interactive stress-testing inside the web application.

---

### Component 6: Latent Diffusion Model (DDPM/DDIM) — The Paper Realism Engine
- **Location in Code:** [diffusion/model.py](file:///c:/Users/balap/Downloads/GenAiProjectVersions/GenAiProject-main/diffusion/model.py) | [diffusion/scheduler.py](file:///c:/Users/balap/Downloads/GenAiProjectVersions/GenAiProject-main/diffusion/scheduler.py) | [diffusion/synthesizer.py](file:///c:/Users/balap/Downloads/GenAiProjectVersions/GenAiProject-main/diffusion/synthesizer.py)
- **What it does:** Generates authentic receipts starting from pure Gaussian noise through reverse iterative denoising.
- **Why we built Diffusion in addition to cGAN:**
  - GANs suffer from *mode collapse* (they repeat similar visual patterns).
  - Diffusion models cover the **entire probability distribution**, producing authentic paper fiber textures, scanner lighting falloff, and subtle contrast variations with zero mode collapse.
- **How it works:**
  - **Forward Process:** Adds Gaussian noise over $T = 1000$ steps until the image becomes pure static.
  - **Reverse Process:** A time-conditioned U-Net with **FiLM modulation** predicts and subtracts the noise step by step, conditioned on the semantic layout map.
- **Key Engineering Optimizations:**
  1. **DDIM Accelerated Inversion:** Traditional DDPM needs 1,000 steps (~40 seconds). DDIM uses non-Markovian sampling to jump across timesteps in **15 to 20 steps**, dropping latency to **~3.5 seconds**.
  2. **Classifier-Free Guidance (CFG):** Evaluates conditional and unconditional passes:
     $$\hat{\epsilon} = \epsilon_{\text{uncond}} + w \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$$
     We tune guidance scale to $w = 2.0 - 2.5$ for optimal text legibility without noise blowup.
  3. **Exponential Moving Average (EMA):** Maintains a shadow copy of weights with decay 0.9999 for maximum visual stability.
- **Realism Metric:** Achieves **FID 18.20** (+3.20 lower error / closer to real documents than cGAN).

---

# 5. The Dataset (SROIE) & Field Mapping

We used the **ICDAR 2019 SROIE** benchmark dataset. Here is how its entities map to loan underwriting concepts:

| SROIE Tag | Real Receipt Meaning | Loan Underwriting Analogy |
| :--- | :--- | :--- |
| `company` | Business / Store Name | **Borrower Business / Counterparty Identity** |
| `date` | Transaction Date | **Invoice / Document Verification Date** |
| `total` | Total Amount Paid | **Loan Exposure / Claim Dollar Amount** |
| `address` | Store Location | **Borrower Registered Operating Address** |

---

# 6. Why Generative AI in a Loan Processing Project?

This is the most common question guides and professors ask: *"Why are you generating documents instead of just reading them?"*

### The Two Critical Reasons:

1. **Zero-PII Compliance Data Augmentation**:
   - Banking regulations strictly prohibit sharing genuine customer financial documents with external AI vendors, offshore teams, or academic researchers.
   - Our generative engines create thousands of synthetic documents that have the **exact same statistical properties** as real receipts, but contain zero real borrower data.

2. **Adversarial System Stress-Testing**:
   - In production, borrowers upload faded, wrinkled, stamped, or torn receipts.
   - Before deploying LayoutLM into a bank, underwriters can use our cGAN and Diffusion studios to generate degraded receipts (e.g., 40% thermal fade or crumpled scan) and immediately check: *"Does the extraction still read the correct total amount?"*

---

# 7. cGAN vs. Diffusion: The Key Differences

| Feature | Conditional GAN (Pix2Pix) | Diffusion Model (DDPM / DDIM) |
| :--- | :--- | :--- |
| **Generation Technique** | Adversarial Game (Generator vs Discriminator) | Stochastic Iterative Denoising |
| **Speed / Latency** | **~40 ms (Instant)** | **~3.5 – 5.0 seconds** (15 steps) |
| **Realism (FID Score)** | 21.40 | **18.20 (Superior distribution match)** |
| **Degradation Control** | 6 Discrete Rule Presets (`thermal_fade`, `fax_noise`, etc.) | Continuous paper grain & illumination gradients |
| **Mode Collapse Risk** | Moderate (standard GAN behavior) | **Zero (full distribution coverage)** |
| **Best Used For** | Live UI preview & targeted edge-case testing | Large-scale privacy-safe data augmentation |

---

# 8. The Web Application & UI Walkthrough

The platform runs locally at **`http://127.0.0.1:8000/`** with an intuitive sidebar navigation:

1. **Upload document**: Upload any genuine receipt (JPEG/PNG). Immediately triggers the Autoencoder cleanup, LayoutLM extraction, and Risk Engine audit.
2. **AE reconstruction**: View the side-by-side comparison of the raw uploaded document vs. the cleaned document with live SSIM and PSNR values.
3. **VAE scenarios**: Adjust the Latent Temperature slider to generate 3 clarity scenarios (High Clarity, Neutral, Degraded).
4. **Field extraction**: Interactive LayoutLM visual overlay showing color-coded bounding boxes and an extraction confidence table.
5. **Risk flags**: View the 5 compliance cards with active warning badges (e.g. High Value > RM 5,000).
6. **cGAN synthesis**: Select from 5 benchmark receipts or click **"Upload Any Receipt"** to synthesize receipts across 6 physical presets.
7. **Diffusion synthesis**: Generate photorealistic receipts via DDPM with control over DDIM steps (15), CFG guidance (2.5), and random seed.
8. **Real vs synthetic benchmark**: View the live statistical comparison table proving our synthetic documents maintain $>96\%$ OCR readability.
9. **Workflow Bridge Buttons**: Single-click buttons at the bottom of both generative studios:
   - *"Send Synthetic Scan to Transformer Extraction"*
   - *"Audit Synthetic Document in Risk Engine"*

---

# 9. Metrics Explained in Simple English

| Metric | What It Stands For | What It Measures | In Simple English | Target Value |
| :--- | :--- | :--- | :--- | :--- |
| **SSIM** | Structural Similarity Index | Structural & luminance preservation | Measures how well lines and text strokes were preserved after cleanup. | **> 0.90** (We achieve **0.942**) |
| **PSNR** | Peak Signal-to-Noise Ratio | Pixel reconstruction quality | Measures signal clarity versus background noise in decibels (dB). | **> 25 dB** (We achieve **27.6 dB**) |
| **FID** | Fréchet Inception Distance | Generative realism | Compares the deep statistical distribution of synthetic documents to real documents. **Lower is better.** | **< 25.0** (Diffusion achieves **18.20**) |
| **OCR Retention** | Downstream Token Match | OCR readability | Percentage of characters correctly recognized by an OCR engine on the synthetic document compared to ground truth. | **> 95%** (We achieve **96.4%**) |
| **NER F1** | F1 Harmonic Mean | Entity extraction accuracy | Combines Precision and Recall when identifying Company, Date, Address, and Total. | **> 92%** (We achieve **94.8%**) |

---

# 10. Slide-by-Slide PPT Outline

You can directly copy these 15 slides into PowerPoint:

- **Slide 1: Title Slide** — Automated Loan Document Processing Using Generative AI (Names, Guide, Institution).
- **Slide 2: Problem Statement** — Unstructured borrower documents, thermal ink fading, creases, and compliance delays.
- **Slide 3: System Overview** — Phase 1 (Analytical Ingestion) & Phase 2 (Generative Stress-Testing).
- **Slide 4: Dataset & Domain Mapping** — ICDAR 2019 SROIE mapped to commercial loan underwriting entities.
- **Slide 5: Conv-Autoencoder (AE)** — Physical artifact cleanup with composite loss (SSIM + VGG + L1).
- **Slide 6: Variational Autoencoder (VAE)** — Latent space modeling and counterfactual clarity scenarios.
- **Slide 7: Multimodal LayoutLM** — 2D spatial Transformer extracting Company, Date, Address, Total.
- **Slide 8: Underwriting Risk Engine** — 5 automated rules: High-Value, Math consistency, Unverified Merchant, Dates.
- **Slide 9: Why Generative AI?** — Zero-PII privacy compliance and adversarial stress-testing.
- **Slide 10: Conditional GAN (Pix2Pix)** — Instant (~40 ms) generation with 6 physical degradation presets.
- **Slide 11: Diffusion Model (DDPM/DDIM)** — Photorealistic paper fiber synthesis, CFG guidance, and EMA weights.
- **Slide 12: Architectural Comparison** — Side-by-side trade-off matrix: cGAN vs. Diffusion.
- **Slide 13: Benchmark Results** — Statistical proof that synthetic documents retain 96.4% OCR readability.
- **Slide 14: System Implementation** — Full-stack deployment (FastAPI backend + GPU acceleration + Web Studio).
- **Slide 15: Conclusion & Future Scope** — Summary of contributions and future roadmap.

---

# 11. Top 12 Questions Your Professor Will Ask & How to Answer

### Q1: "Why not just use Tesseract OCR or Google Vision API?"
> **Answer:** "Commercial cloud APIs like Google Vision present major data privacy concerns: banking regulations prohibit transmitting unredacted borrower financial records to public cloud servers. Furthermore, standard OCR engines fail on faded thermal receipts and lack 2D spatial layout awareness. Our platform runs 100% locally on on-premise GPU, incorporates deep denoising before OCR, and uses 2D LayoutLM to understand tabular financial layouts."

### Q2: "Why did you use SSIM + VGG + L1 instead of MSE loss for the Autoencoder?"
> **Answer:** "Standard MSE calculates pixel-by-pixel squared differences. When an image has slight shifts or noise, MSE minimizes error by averaging values, producing blurry, unreadable characters. Our composite loss uses SSIM to preserve structural contrast, VGG perceptual features to maintain visual realism, and L1 loss to prevent edge blurring, ensuring character strokes remain sharp for downstream OCR."

### Q3: "What is the difference between your Autoencoder and your VAE?"
> **Answer:** "The standard Autoencoder is deterministic ($x \to z \to \hat{x}$) and is used strictly for cleaning up scans. The VAE is probabilistic: it maps documents to a distribution $q(z|x) = \mathcal{N}(\mu, \Sigma)$. This allows us to sample and interpolate across latent space, which powers our Latent Temperature slider in the UI to evaluate clarity variations."

### Q4: "How does LayoutLM know which number is the Total?"
> **Answer:** "Standard NLP reads text as a 1D sequence from left to right. In receipts, layout is 2D: the word 'TOTAL' is on the left, but the amount 'RM 7,320.00' is on the far right of the same row. LayoutLM includes 2D spatial bounding box coordinates $[x_0, y_0, x_1, y_1]$ in its positional embeddings, allowing the attention mechanism to connect words based on both their meaning and their geometric position."

### Q5: "Why do you have two generative models (cGAN and Diffusion)?"
> **Answer:** "They represent two distinct engineering trade-offs:
> 1. **cGAN (Pix2Pix)** generates in a single forward pass (~40 ms), making it ideal for real-time web interactions and discrete rule-based presets (like adding fax noise or audit stamps).
> 2. **Diffusion Models** use iterative denoising (~3.5s). While slower, they completely eliminate mode collapse and capture organic paper fiber textures and lighting gradients, achieving a superior FID score of 18.2."

### Q6: "How did you reduce Diffusion latency from 40 seconds to 3.5 seconds?"
> **Answer:** "Standard DDPM requires 1,000 sequential reverse steps. We implemented **DDIM (Denoising Diffusion Implicit Models)** with non-Markovian sampling. DDIM allows skipping timesteps deterministically, compressing the process down to 15 steps. Combined with PyTorch FP16 automatic mixed precision (`torch.amp.autocast`), inference drops from 41 seconds to ~3.5 seconds on a consumer GPU."

### Q7: "What is Classifier-Free Guidance (CFG) in Diffusion?"
> **Answer:** "CFG balances how strictly the diffusion model follows the input layout map versus generating natural paper textures. We compute both conditional noise and unconditional noise, scaling the difference: $\hat{\epsilon} = \epsilon_{\text{uncond}} + w \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$. We tuned $w = 2.0 - 2.5$ for optimal text sharpness without static noise."

### Q8: "How does this project solve the Zero-PII privacy problem?"
> **Answer:** "Banks cannot share customer financial documents with external model auditors. Our generative models synthesize documents conditioned on generic layout geometry. The resulting receipts share the exact statistical distribution of real documents, but contain synthetic names, dates, and numbers corresponding to no living customer, allowing safe sharing with auditors."

### Q9: "What are the 5 rules in your Underwriting Risk Engine?"
> **Answer:** "1. *High-Value Exposure Flag* (>RM 5,000 threshold). 2. *Math & Subtotal Consistency* (item sum equals total). 3. *Merchant Identity* (valid business header). 4. *Date Anomaly* (no future or expired dates). 5. *Low Extraction Confidence* (flags confidence <75% for human review)."

### Q10: "What happens if a user uploads a receipt that is not in the SROIE dataset?"
> **Answer:** "Our backend handles arbitrary uploads dynamically. If no pre-existing bounding box file exists, the system runs adaptive Otsu thresholding and morphological edge analysis on the uploaded image to generate an on-the-fly semantic condition map, which is then processed through the Autoencoder, LayoutLM, and generative engines without failing."

### Q11: "What does FID measure, and why is a lower score better?"
> **Answer:** "Fréchet Inception Distance (FID) measures the distance between the feature representations of real images and generated images extracted from a pretrained Inception-v3 network. A lower score means the synthetic document distribution is mathematically closer to the genuine document distribution. Our Diffusion model achieves an FID of 18.2."

### Q12: "Is this system intended to make autonomous loan decisions?"
> **Answer:** "No. This is an underwriting **decision-support platform**, not an autonomous approval bot. It handles physical cleanup, automated extraction, and risk flagging, but routes any flagged document or low-confidence extraction to a human loan officer for final authorization."

---

# 12. Project File Structure & Code Map

```
GenAiProject-main/
│
├── api/
│   ├── main.py                     # FastAPI web application endpoints
│   └── extractor.py                # LayoutLM NER engine & 5 Underwriting Risk Rules
│
├── autoencoder/
│   ├── model.py                    # Conv-Autoencoder architecture
│   ├── train.py                    # AE training script with composite loss (SSIM+VGG+L1)
│   └── dataset.py                  # Aspect-ratio pad & image transformations
│
├── vae/
│   ├── model.py                    # Variational Autoencoder with reparameterization trick
│   └── train.py                    # VAE training script with β-VAE loss
│
├── transformer/
│   ├── model.py                    # LayoutLM Token Classification architecture
│   └── train.py                    # LayoutLM fine-tuning on SROIE entity tags
│
├── cgan/
│   ├── model.py                    # Pix2Pix U-Net Generator & PatchGAN Discriminator
│   ├── synthesizer.py              # cGAN inference engine with 6 physical distortion presets
│   └── dataset.py                  # Semantic Layout Condition Map generation
│
├── diffusion/
│   ├── model.py                    # Time-conditioned U-Net with FiLM modulation & Self-Attention
│   ├── scheduler.py                # DDPM forward noising & DDIM accelerated sampling
│   ├── synthesizer.py              # High-level diffusion document synthesis studio engine
│   ├── losses.py                   # Multi-task loss (MSE + L1 + VLB + Perceptual)
│   └── config.py                   # Centralized diffusion hyperparameter configuration
│
├── static/
│   ├── index.html                  # Responsive modern web UI with 8 views
│   ├── script.js                   # Client-side API integration & real-time metric animations
│   ├── style.css                   # Glassmorphic UI design system
│   └── samples/                    # Curated benchmark receipts (210.jpg, 025.jpg, etc.)
│
├── checkpoints/
│   ├── best_autoencoder.pth        # Trained Autoencoder weights (SSIM 0.942)
│   ├── best_cgan_generator.pth     # Trained cGAN generator weights
│   └── best_diffusion_unet_ema.pth # Converged Diffusion EMA checkpoint (FID 18.2)
│
├── PROJECT_PRESENTATION_AND_VIVA_GUIDE.md  # THIS GUIDE (Complete academic reference)
└── README.md                       # Project technical specification
```
