# RSNA Knee Abnormality Detection

A multimodal medical-AI project for the **RSNA Knee Abnormality Detection Kaggle Competition**.

The goal is to develop machine-learning models that analyze knee MRI examinations and predict the probability of **12 clinically important abnormalities**. The competition uses a **macro-averaged ROC-AUC** across the twelve targets.

> **Project status:** Phase 0 — Competition understanding / Phase 1 — Dataset discovery and EDA  
> **Primary development environment:** Kaggle  
> **Local AI development assistant:** Ollama + Qwen2.5-Coder 14B + Phi-4-mini-reasoning

---

## Competition

A single knee MRI examination can contain evidence of multiple abnormalities. This competition asks participants to build models capable of detecting twelve clinically important knee abnormalities.

The competition is particularly interesting because it is designed as a **multimodal problem**, pairing knee MRI examinations with radiology reports.

### Prediction targets

The model must produce a confidence score for each of the following 12 targets:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

This is a **multi-label classification** problem: one study may contain zero, one, or multiple abnormalities.

---

## Evaluation Metric

The competition evaluates submissions using the **macro-averaged ROC-AUC** across the twelve target abnormalities.

Conceptually:

```text
AUC(ACL)
AUC(MCL)
AUC(Medial Meniscus)
...
AUC(Fracture)
        ↓
Mean of all 12 AUC values
        ↓
Final competition score
```

Because the metric is macro-averaged, the project will track both:

- Overall macro ROC-AUC
- Per-target ROC-AUC

Accuracy alone is not an appropriate primary optimization metric for this competition.

---

## Submission Format

The submission contains one row per `StudyInstanceUID` and one confidence score for each target.

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
```

The model therefore needs to output **probability/confidence scores**, rather than binary predictions.

---

# Project Objectives

The project will be developed progressively rather than attempting to build a complex architecture immediately.

### Primary objectives

- Understand the competition dataset
- Perform detailed exploratory data analysis
- Understand MRI study/series/slice organization
- Understand available MRI sequences
- Analyze the twelve target labels
- Investigate class imbalance
- Build a leakage-safe validation strategy
- Develop a reproducible baseline
- Progress toward study-level MRI modeling
- Investigate multi-sequence modeling
- Investigate multimodal MRI + report modeling
- Optimize the final model for macro ROC-AUC
- Produce a valid Kaggle submission

---

# Development Philosophy

The project follows an iterative approach:

```text
Competition Understanding
        ↓
Dataset Discovery
        ↓
EDA
        ↓
Validation Strategy
        ↓
Baseline
        ↓
Study-Level Modeling
        ↓
Multi-Sequence Modeling
        ↓
Multimodal MRI + Report
        ↓
Ensembling / Optimization
        ↓
Final Submission
```

We will **not** start with a large 3D Transformer or complex multimodal architecture.

The first priority is to build a correct and reproducible data pipeline.

---

# Planned Modeling Roadmap

## Phase 0 — Competition Understanding

- Understand the task
- Identify all 12 targets
- Understand macro ROC-AUC
- Understand submission requirements
- Understand available modalities

**Status:** In progress / completed

---

## Phase 1 — Dataset Discovery & EDA

The first notebook will investigate the actual Kaggle data without making assumptions.

### Planned analysis

- Discover dataset paths
- Inspect directory structure
- Identify train/test data
- Identify metadata
- Identify labels
- Identify radiology reports
- Map `StudyInstanceUID`
- Identify MRI file format
- Identify series
- Identify sequences
- Count studies
- Count series
- Count images/slices
- Analyze image dimensions
- Analyze target distribution
- Identify missing values
- Inspect patient/study identifiers
- Visualize representative MRI data

**Status:** Next milestone

---

## Phase 2 — Baseline Model

Initial baseline:

```text
MRI
 ↓
Preprocessing
 ↓
2D CNN
 ↓
12-output classification head
 ↓
BCEWithLogitsLoss
 ↓
12 probabilities
```

The baseline is intended to establish:

- Correct data loading
- Correct labels
- Correct validation
- Correct training loop
- Correct inference
- Correct submission generation

---

## Phase 3 — Study-Level Modeling

MRI examinations contain multiple slices.

Instead of treating every slice as an independent final prediction:

```text
Slice 1 ─┐
Slice 2 ─┤
Slice 3 ─┤
   ...   ├──> Aggregation ──> 12 predictions
Slice N ─┘
```

Potential approaches:

- Mean/max pooling
- Learned attention
- Multiple Instance Learning (MIL)
- Slice importance weighting

---

## Phase 4 — Multi-Sequence MRI

The model will investigate combining information from multiple MRI sequences.

```text
Sequence A ──> Encoder ─┐
Sequence B ──> Encoder ─┤
Sequence C ──> Encoder ─┤
                         ├──> Fusion/Attention
Sequence D ──> Encoder ─┤
                        └──> 12-label classifier
```

Experiments may include:

- Sequence-specific encoders
- Shared encoders
- Feature-level fusion
- Cross-sequence attention
- Sequence weighting

---

## Phase 5 — Multimodal MRI + Radiology Report

If the competition's test/inference data provides the required report information, the project will investigate:

```text
                MRI
                 │
          Vision Encoder
                 │
                 ▼
           MRI Embedding
                 │
                 ├───────┐
                 │       │
                 ▼       ▼
               Fusion  Text Encoder
                         ▲
                         │
                  Radiology Report
                 │
                 ▼
             12-label head
```

Experiments will compare:

- MRI-only
- Report-only
- MRI + Report

Special attention will be given to avoiding information leakage from reports.

---

# Validation Strategy

Medical imaging requires careful validation.

A major project requirement is to avoid leakage caused by splitting individual images/slices from the same examination across training and validation sets.

The preferred hierarchy to investigate is:

```text
Patient
   ↓
Study
   ↓
Series
   ↓
Slice
```

The validation strategy must ensure that information from the same patient/study does not unintentionally appear in both training and validation.

Before model optimization, the project will verify:

- Patient identifiers
- Study identifiers
- Duplicate studies
- Duplicate images
- Series relationships
- Train/validation overlap

---

# Class Imbalance

The twelve abnormalities may have different prevalence.

The project will measure:

- Positive count
- Negative count
- Positive percentage
- Class imbalance ratio
- Per-class validation AUC

Potential techniques to investigate later:

- `BCEWithLogitsLoss`
- Positive class weighting
- Focal Loss
- Asymmetric Loss
- Sampling strategies
- Per-class threshold analysis

Loss-function changes will only be introduced after establishing a reliable baseline.

---

# Local AI Development Workflow

The project uses local AI models to accelerate development while keeping the final engineering decisions under manual review.

### Qwen2.5-Coder 14B

Primary coding assistant.

Used for:

- Python
- PyTorch
- Dataset loaders
- Preprocessing
- Training loops
- Inference
- Kaggle notebooks
- Debugging
- Refactoring

### Phi-4-mini-reasoning

Secondary reasoning and code-review assistant.

Used for:

- Logic checking
- Debugging
- Architecture review
- Leakage analysis
- Validation review
- Experiment analysis

### Development loop

```text
Requirement
    ↓
Qwen2.5-Coder
    ↓
Generated Code
    ↓
Run/Test in Kaggle
    ↓
Phi-4 Review (when needed)
    ↓
Project Review
    ↓
Refinement
    ↓
Working Version
```

The local models are assistants; generated code is tested and reviewed before being treated as part of the project.

---

# Recommended Repository Structure

The repository will evolve toward:

```text
rsna-knee-abnormality-detection/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_study_level_model.ipynb
│   ├── 04_multisequence_model.ipynb
│   └── 05_multimodal_model.ipynb
│
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── transforms.py
│   │   └── split.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── mil.py
│   │   ├── multisequence.py
│   │   └── multimodal.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── losses.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   └── utils/
│       ├── metrics.py
│       ├── seed.py
│       └── logging.py
│
├── configs/
│   └── baseline.yaml
│
├── experiments/
│   └── experiment_log.csv
│
└── submission/
    └── submission.csv
```

Dataset files, model weights, Kaggle credentials, API keys, and other large/private files should **not** be committed to Git.

---

# Experiment Tracking

Every meaningful experiment should record:

| Experiment | Model | Input | Loss | Validation AUC | Notes |
|---|---|---|---|---:|---|
| E01 | Baseline CNN | TBD | BCE | TBD | Initial baseline |
| E02 | Improved CNN | TBD | BCE | TBD | TBD |
| E03 | 2.5D | Multi-slice | TBD | TBD | TBD |
| E04 | MIL | Study | TBD | TBD | TBD |
| E05 | Multi-sequence | MRI | TBD | TBD | TBD |
| E06 | Multimodal | MRI + Report | TBD | TBD | TBD |

The experiment log will prevent uncontrolled changes and make it possible to identify which improvements actually increase the competition score.

---

# Reproducibility

Experiments should use:

- Fixed random seeds where practical
- Versioned configuration
- Recorded model architecture
- Recorded hyperparameters
- Recorded dataset/split information
- Saved checkpoints
- Logged validation metrics
- Explicit package versions

---

# Medical AI Considerations

This project is intended as a **research/competition project**, not a clinical diagnostic system.

Important considerations include:

- Data leakage
- Patient-level separation
- Class imbalance
- Calibration
- False positives and false negatives
- Dataset bias
- Generalization
- Interpretability
- Appropriate use of radiology reports
- Limitations of automated medical-image interpretation

High competition performance should not be interpreted as clinical validation.

---

# Current Status

### Completed

- [x] Competition overview reviewed
- [x] Twelve prediction targets identified
- [x] Macro ROC-AUC identified as primary metric
- [x] Multi-label nature identified
- [x] Multimodal MRI + report direction identified
- [x] Initial modeling roadmap defined
- [x] Local AI development workflow planned

### In Progress

- [ ] Kaggle dataset discovery
- [ ] Dataset structure analysis
- [ ] Label distribution analysis
- [ ] MRI sequence analysis
- [ ] Radiology report availability analysis
- [ ] Patient/study leakage analysis

### Upcoming

- [ ] EDA notebook
- [ ] Validation split
- [ ] Baseline model
- [ ] Baseline submission
- [ ] Study-level modeling
- [ ] Multi-sequence modeling
- [ ] Multimodal modeling
- [ ] Ensemble experiments
- [ ] Final Kaggle submission

---

# Project Principles

1. **Understand the data before modeling.**
2. **Never assume the dataset structure.**
3. **Prevent patient/study leakage.**
4. **Use macro ROC-AUC as the primary optimization metric.**
5. **Track per-target performance.**
6. **Start simple and establish a strong baseline.**
7. **Change one major variable at a time where possible.**
8. **Measure every experiment.**
9. **Optimize for reproducibility, not just leaderboard performance.**
10. **Use local AI to accelerate coding, but validate generated code.**

---

## Disclaimer

This repository is an independent machine-learning research project for the RSNA Kaggle competition. It is not a clinically validated diagnostic system and should not be used to make medical decisions.

---

## Acknowledgements

This project is based on the **RSNA Knee Abnormality Detection** Kaggle competition and its associated dataset.

Competition:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

RSNA:
https://www.rsna.org/

---

## License

Choose an appropriate license before publishing the repository. Note that the Kaggle competition dataset and associated medical data may have separate terms of use; do not redistribute competition data in this repository unless explicitly permitted.
