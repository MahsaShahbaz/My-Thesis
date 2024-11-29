# My-Thesis  
**Differential Privacy Techniques in Federated Learning: Application to Diabetic Retinopathy Image Processing**

## Overview  
This repository contains the implementation and results of my master's thesis, which presents a privacy-preserving **Federated Learning (FL)** framework for diagnosing **Diabetic Retinopathy (DR)**. The framework incorporates **Differential Privacy (DP)** using Gaussian and Laplace mechanisms to balance privacy and diagnostic accuracy, addressing sensitive data concerns in healthcare.

---

## Proposed Method  
The framework evaluates four models:  
1. **Centralized Non-Private ML:** Baseline model trained with all data aggregated on a central server.  
2. **Non-Private FL:** Decentralized model trained on client devices without privacy mechanisms.  
3. **Private FL (Gaussian):** FL with Gaussian noise added to gradients to ensure privacy.  
4. **Private FL (Laplace):** FL with Laplace noise, offering another privacy-preserving approach.

Key contributions include:  
- Comparative analysis of privacy-utility trade-offs using \(\epsilon\) (Laplace) and noise multipliers \(\sigma\) (Gaussian).  
- Evaluation of robustness under **inversion attack simulations**, testing the ability to reconstruct original images from shared gradients.  
- Implementation using **SqueezeNet**, a lightweight model architecture tailored for medical image classification.  

---

## Dataset  
- **Source:** APTOS 2019 Blindness Detection dataset from Kaggle.  
- **Task:** Classify retinal images into five categories based on DR severity: No DR, Mild, Moderate, Severe, Proliferative.  
- **Preprocessing:** Images resized to 224x224 pixels for SqueezeNet compatibility.  

---

## Highlights  
- **Gaussian vs. Laplace Mechanisms:**  
  - Gaussian offers stability and consistent privacy protection.  
  - Laplace retains structural details but introduces occasional noise spikes.  

- **Inversion Attack Analysis:**  
  - Simulated attacks show significant privacy improvements with DP mechanisms.  
  - Metrics like PSNR, SSIM, and perceptual loss quantitatively validate privacy robustness.

---

## Results  
- **Gaussian Mechanism:** Achieved a strong privacy-utility balance with consistent results.  
- **Laplace Mechanism:** Suitable for scenarios prioritizing structural detail retention.  

---

## Acknowledgements  
This research was conducted under the guidance of:  
- [Professor Federica Battisti](https://www.linkedin.com/in/federica-battisti-b9901447/) – Università degli Studi di Padova  
- Professor Luis Alberto da Silva Cruz – Universidade de Coimbra  

Special thanks to:  
- [Università degli Studi di Padova](https://www.linkedin.com/school/university-of-padova/posts/?feedView=all)  
- [Universidade de Coimbra](https://www.linkedin.com/school/universidade-de-coimbra/posts/?feedView=all)  

---

## Contact  
For questions or collaboration, reach out at:  
**mahsa.shahbazi@studenti.unipd.it**
