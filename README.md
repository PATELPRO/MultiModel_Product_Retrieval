# Multimodal Product Retrieval — CLIP Zero-Shot vs. LoRA Fine-Tuning on Amazon Berkeley Objects

A text↔image retrieval system for e-commerce product catalogs, built end-to-end on the [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset — from raw S3 data through a cleaned, deduplicated catalog, a zero-shot CLIP baseline, and a LoRA-fine-tuned model, with a fully quantified, per-category comparison between them.

**Given a text query, return the most relevant product images. Given a product image, return the most relevant listing text.**

## Headline Result

Fine-tuning CLIP with LoRA (updating **~1.07% of parameters**) on ~14,500 catalog-specific pairs improved retrieval performance substantially over zero-shot CLIP, with gains concentrated specifically in the product categories the zero-shot baseline was measurably weakest on:

| Metric | Zero-shot CLIP | LoRA Fine-Tuned | Δ | Relative gain |
|---|---|---|---|---|
| Text→Image R@10 | 40.11% | 54.22% | +14.11 pp | +35% |
| Image→Text R@10 | 38.11% | 52.35% | +14.24 pp | +37% |

**Improvement correlated negatively with baseline performance (r ≈ −0.5):** categories that scored worst under zero-shot CLIP (e.g. `HARDWARE_HANDLE`, +29.2pp; `LIGHT_BULB`, +23.5pp) saw the largest absolute gains — direct, quantitative evidence that fine-tuning closed the specific gap identified during baseline evaluation, rather than producing an unexplained uniform lift. No category regressed meaningfully in either search direction. Full breakdown in [Results](#results).

## Table of Contents

- [Motivation](#motivation)
- [Approach](#approach)
- [Dataset & Cleaning Methodology](#dataset--cleaning-methodology)
- [Model & Architecture](#model--architecture)
- [Fine-Tuning: LoRA](#fine-tuning-lora)
- [Evaluation Methodology](#evaluation-methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Reproduction](#setup--reproduction)
- [Limitations & Future Work](#limitations--future-work)
- [Key Engineering Notes](#key-engineering-notes)

## Motivation

Most portfolio-scale retrieval projects work with clean, pre-packaged benchmark datasets and stop at a single reported metric. This project instead starts from a real, messy, multi-marketplace product catalog (multilingual listings, duplicate products across regions, shared placeholder images, SEO-stuffed keyword fields) and treats the data cleaning itself as a first-class part of the methodology — every filtering and deduplication decision is quantified and justified, not asserted. The modeling side deliberately compares a zero-shot foundation model against a lightweight, parameter-efficient fine-tune, with an explicit, pre-registered hypothesis (fine-tuning should help most on the categories the baseline analysis flagged as weak) that is then tested against real per-category results rather than just reporting an aggregate number.

## Approach

Several architectures for cross-modal retrieval were considered before implementation, reasoned through from first principles rather than defaulting to the most common one:

1. **Zero-shot pretrained CLIP** (`openai/clip-vit-base-patch32`) — implemented, serves as the baseline.
2. **Independently pretrained, naively fused unimodal encoders** — reasoned through analytically and *not* implemented: two encoders never trained with a shared objective have no principled reason to produce comparable embeddings, and would be expected to perform close to random chance for cross-modal retrieval (bracketed below by chance and above by zero-shot CLIP's demonstrated result). Treated as a low-value experiment given the expected outcome was already inferable with reasonable confidence, and time was allocated to the higher-value approach below instead.
3. **Joint contrastive fine-tuning of CLIP (LoRA)** — implemented; this is the core modeling contribution of the project.
4. **Classical statistical alignment (CCA / Deep CCA)** — researched and deliberately not implemented as a working baseline: CCA's linear-only alignment, lack of use of semantic/label information, and requirement to hold the full covariance matrix in memory make it poorly suited to this scale; understanding *why* the field moved to contrastive/metric learning informed the choice of approach 3.
5. **Joint fusion / cross-attention architectures** — considered and set aside: this family cannot pre-compute independent catalog embeddings for fast nearest-neighbor search, which conflicts with the retrieval architecture (`torch.topk` over a pre-computed catalog) used throughout this project.

## Dataset & Cleaning Methodology

Built from all 16 listing shards of the [ABO dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), hosted publicly on S3 (`s3://amazon-berkeley-objects/`, no AWS account required).

**Key cleaning decisions** (each quantified, not just asserted — full reasoning in `docs/Documentation.pdf`):

- **Language filtering:** restricted to English-language marketplace variants (`en_IN`, `en_US`, `en_CA`, `en_GB`, `en_AU`, `en_AE`, `en_SG`) — ~79.4% of the dataset — rather than chasing a 95% cumulative-coverage threshold that would have required including genuinely non-English text unsuitable for an English-pretrained text encoder.
- **Text field construction:** `combined_text = item_name + " " + bullet_point`. `item_keywords` was evaluated via a quantitative word-overlap metric (median 62.5% overlap with existing text) and excluded — not because it was purely redundant, but because CLIP's 77-token limit meant including it risked truncating higher-value descriptive content.
- **Deduplication:** two distinct issues, both resolved via a richness-based strategy (keep the duplicate with the longest `combined_text`, country-priority as tiebreaker) — (1) the same product listed across multiple marketplaces/languages (`item_id` duplicates), and (2) different products sharing an identical, reused placeholder photo (`main_image_id` collisions) — the latter treated as a genuine data-quality issue independent of the former, since a shared image cannot visually distinguish between two different products.
- **Category scope:** `CELLULAR_PHONE_CASE` (~60% of the cleaned dataset) excluded entirely to prevent it from dominating aggregate metrics; a minimum 300-listing floor applied per remaining category for evaluation reliability; a maximum per-category cap was considered but rejected after confirming the largest remaining category was only 18.9% of the final set.

**Final dataset:** 29 product categories, 25,479 listings — English-only, one row per unique product, no duplicate images across products, no empty text fields.

## Model & Architecture

**Base model:** `openai/clip-vit-base-patch32` (~152.9M parameters) — a dual-encoder architecture: a 12-layer Vision Transformer (patch size 32) and a 12-layer, 512-wide text Transformer, each with a final linear projection into a shared 512-dimensional embedding space. Loaded via Hugging Face `transformers` (chosen over OpenAI's original repository specifically to support the LoRA fine-tuning workflow via the `peft` ecosystem).

**Retrieval mechanism:** catalog image and text embeddings are computed once via `get_image_features()`/`get_text_features()` (independently, not the combined forward pass), L2-normalized, and saved. A query (text or image) is embedded on demand, compared against the full catalog via a single matrix multiplication, and ranked via `torch.topk` — no softmax at inference time (softmax is reserved for the training objective; retrieval uses raw cosine similarity ranking, since softmax over a large or noisy candidate set was found to distort confidence in a way that doesn't affect ranking correctness).

## Fine-Tuning: LoRA

Full fine-tuning of all ~152.9M parameters was rejected given the small fine-tuning set (~14,500 pairs) relative to CLIP's original 400M-pair pretraining — a ratio strongly favoring catastrophic forgetting under unrestricted weight updates. **LoRA** (rank 8, alpha 16) was applied to the Q/K/V/output projection matrices across all 24 attention blocks (12 vision + 12 text), with the two final projection layers fully fine-tuned (`modules_to_save`) given their small size and outsized importance to the shared embedding space. Result: **1,638,400 trainable parameters — 1.07% of the model.**

Training used the same symmetric contrastive (InfoNCE-style) cross-entropy loss CLIP was originally pretrained with — each training batch defines a fresh N-way matching problem, with the diagonal of the batch similarity matrix as ground truth — for 15 epochs (~4 min/epoch on a single Colab T4), with validation loss monitored throughout (no increase observed across the full run, indicating no overfitting at this scale/epoch count).

## Evaluation Methodology

- **Fixed, stratified test set** (`test_query_ids.csv`, 20% of the catalog, stratified by `product_type`) reserved *before* fine-tuning and explicitly excluded from all training data — guaranteeing both models are compared on identical, unseen items with zero contamination risk.
- **Bidirectional Recall@K** (K = 1, 5, 10), computed both as an overall pooled (micro-averaged) figure and broken down per category, since a single aggregate number would be dominated by the largest categories and could mask category-specific weaknesses.
- **Ground truth:** only the exact original `item_id` counts as a correct retrieval — a stated simplification, since near-duplicate listings could arguably also be acceptable matches.

## Results

Full per-category tables, correlation analysis, and an accompanying formula-driven Excel workbook (delta/%-change/correlation/weighted-average cross-checks, fully recalculating from source data) are in `docs/`. Headline findings:

- **Consistent, substantial gains in both directions:** ~35–41% relative Recall@10 improvement, text→image and image→text alike.
- **Targeted, not uniform, improvement:** Pearson r ≈ −0.5 between baseline category performance and improvement magnitude — the weakest zero-shot categories (visually/textually homogeneous ones: hardware, lighting, furniture) improved the most.
- **No regressions:** across a 4-tier classification (Strong / Moderate / Marginal / Regressed) applied to all 29 categories in both directions, zero categories regressed meaningfully.
- **Fine-tuning strengthened cross-directional consistency:** correlation between text→image and image→text per-category performance rose from r = 0.916 (baseline) to r = 0.939 (fine-tuned) — though the *magnitude* of each category's gain was largely direction-independent (r ≈ −0.07), a distinction discussed in full in the results documentation.

## Repository Structure

```
├── README.md
├── docs/
│   ├── ABO_Dataset_Construction.pdf        # data cleaning methodology
│   ├── CLIP_Zero_Shot_Baseline.pdf         # baseline results & CLIP technical reference
│   ├── Results_Comparison.pdf              # zero-shot vs. fine-tuned analysis
│   └── Results_Analysis.xlsx               # formula-driven results workbook
├── notebooks/
│   └── Image_Text_Retrieval_Pipeline.ipynb # end-to-end: data cleaning → embeddings → training → evaluation
└── requirements.txt
```

*(The pipeline was developed iteratively across several working notebooks; the version in this repository is consolidated into a single, cleaned, linearly runnable notebook covering data cleaning, embedding generation, LoRA fine-tuning, and evaluation.)*

## Setup & Reproduction

```bash
pip install -r requirements.txt
```

Core dependencies: `transformers`, `peft`, `torch`, `boto3`, `pandas`, `Pillow`, `scikit-learn`.

The notebook is written for Google Colab (GPU runtime) with Google Drive mounted for persistent storage across sessions, given multi-hour training/embedding runs. Data is pulled directly from the public ABO S3 bucket (`--no-sign-request`, no AWS account needed). Key artifacts (catalog embeddings, `item_id` alignment files, the fixed train/val/test split, and LoRA checkpoints) are saved to Drive at each stage so the pipeline can resume after a session disconnect rather than restart from scratch.

## Limitations & Future Work

Identified during the project but deliberately deferred given time constraints — listed transparently rather than omitted:

- **Statistical significance testing (McNemar's test):** the paired test-set design (same items evaluated by both models) supports a direct significance test on the Recall@K improvement; not yet run.
- **Catastrophic forgetting check:** LoRA was chosen specifically to preserve CLIP's general-purpose knowledge, but this was never directly verified — e.g. by comparing zero-shot accuracy on a generic, out-of-domain dataset (such as CIFAR-10) before and after fine-tuning.
- **`item_keywords` ablation:** flagged during data cleaning as a candidate experiment (with vs. without in `combined_text`) once the pipeline existed; not yet run.
- **Prompt-template experiment:** CLIP's own paper shows templated prompts ("a photo of a {label}") measurably improve zero-shot performance; not yet tested against this catalog's baseline.
- **Approach 2 (naive fusion) as a measured result:** currently a reasoned analytical argument rather than an implemented experiment.
- **Larger CLIP variants** (ViT-B/16, ViT-L/14) as an alternative or complementary starting point.

## Key Engineering Notes

A few non-obvious, real engineering problems worth noting for anyone reproducing this pipeline:

- **Google Drive read latency, not GPU compute, was the dominant bottleneck** for both catalog embedding and training — diagnosed via manual timing splits, resolved via an in-memory image cache (training) and parallelized `boto3` downloads (`ThreadPoolExecutor`) rather than sequential per-file `aws s3 cp` calls.
- **`os.system` silently swallows errors** in Colab; all shell calls use `subprocess.run(capture_output=True)` instead, after a failure mode where commands appeared to succeed while doing nothing.
- **Checkpointing was necessary, not optional**, given unpredictable Colab session disconnects during multi-hour embedding/training runs — incremental progress (embeddings, `item_id`s, resume position) is saved periodically to Drive rather than only at completion.
