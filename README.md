# Numerical Uncertainty in Linear MRI Registration

This repository contains the code and experimental setup for the project  
**“**[Numerical Uncertainty in Linear Registration: An Experimental Study](https://arxiv.org/abs/2508.00781v1)**.”**.

The goal of this work is to investigate how floating-point numerical perturbations
affect the stability, reliability, and quality of commonly used **linear MRI
registration tools**.

---

## 🧠 Background & Motivation

Linear registration is a core step in most neuroimaging preprocessing pipelines.
Despite its widespread use, its **numerical stability**—that is, sensitivity to
small floating-point errors introduced by software, hardware, or compiler
differences—has been largely understudied.

Even minor numerical perturbations can:
- Steer optimizers toward different local minima
- Cause sporadic registration failures
- Propagate uncertainty into downstream analyses

This project systematically quantifies these effects using **Monte-Carlo
Arithmetic (MCA)**.

---

## 🔬 What This Project Does

We assess numerical uncertainty across:

- **Registration software**
  - SPM (spm_affreg)
  - FSL (FLIRT)
  - ANTs (antsRegistration)

- **Similarity measures**
  - SSD, NCC, CR, MI, NMI (tool-dependent)

- **Templates**
  - Symmetric and asymmetric MNI152 (1 mm resolution)

- **Cohorts**
  - Healthy controls (HC)
  - Parkinson’s disease (PD)

Each registration is run once using standard IEEE floating-point arithmetic and
multiple times under MCA perturbations to quantify variability.

---

## 📏 Numerical Uncertainty Metric

Numerical uncertainty is measured using the **standard deviation of Framewise
Displacement (FD)** across perturbed runs.

FD provides a single, interpretable scalar that summarizes variability in affine
registration parameters (translation, rotation, scale, and shear). Higher FD
variability indicates greater numerical instability.

---

## 📊 Key Findings

- **SPM is the most numerically stable** among the evaluated tools.
- **FSL and ANTs show larger variability**, with ANTs occasionally exhibiting
  catastrophic failures under perturbation.
- **Similarity measure choice matters**: NCC and CR are generally more stable
  than MI/NMI in FSL.
- In some cases, numerical variability reaches magnitudes comparable to **voxel
  size or in-scanner head motion**, making it practically significant.
- **Cohort (HC vs PD) and template choice** do not significantly affect numerical
  stability.
- MCA-derived variability metrics show promise as **features for automated
  quality control (QC)**.

---

## 🤖 Automated QC (Proof of Concept)

Subjects that fail visual QC consistently show higher numerical variability.
This suggests that MCA-based uncertainty measures can support:
- Automated QC
- Anomaly detection
- More reproducible preprocessing pipelines

---

## 🧰 Tools & Frameworks

- **Monte-Carlo Arithmetic**
  - Verificarlo (fuzzy-libm backend)
  - Verrou (validation experiments)

- **Neuroimaging software**
  - SPM12
  - FSL 6.x
  - ANTs 2.5+

- **Infrastructure**
  - Dockerized pipelines for reproducibility

---

## 📁 Repository Structure

```text
.
├── docker/              # Dockerfiles for perturbed & unperturbed tools
├── scripts/             # Registration and execution scripts
├── analysis/            # FD computation and statistical analyses
├── qc/                  # QC labels and exploratory models
├── figures/             # Figures used in the manuscript
└── README.md
