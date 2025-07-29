# TDCA-MHFCA-for-Lung-Cancer-Detection


A robust and computationally efficient deep learning framework for lung cancer classification based on the **Contextual Heuristic Attention (CHA)** architecture, enhanced by TDCA and MHFCA modules. This repository presents statistical evaluations, component-wise ablation, and benchmarking against state-of-the-art methods.

---

## 📊 Table 1: Ablation Study Comparing Optimization Techniques with the Proposed CHA Architecture

| SL No. | Optimization Technique        | Selected Features (%) | Computation Time (s) | Accuracy (%) |
|--------|-------------------------------|------------------------|-----------------------|---------------|
| 1      | Genetic Algorithm (GA)        | 82.3                   | 12,457.80             | 94.2          |
| 2      | Particle Swarm Optimization   | 85.7                   | 10,389.40             | 95.4          |
| 3      | Grey Wolf Optimizer (GWO)     | 87.2                   | 9,847.20              | 95.9          |
| 4      | Dragonfly Algorithm (DA)      | 89.1                   | 8,634.50              | 96.5          |
| 5      | Ant Colony Optimization (ACO) | 86.5                   | 9,276.30              | 95.7          |
| 6      | **CHA Proposed Architecture** | **84.97**              | **7,134.80**          | **98.1**      |

---

## 🧪 Table 2: Ablation Study of TDCA-MHFCA – Component-Wise and Parameter-Wise Impact on Performance

| Ablation Model                               | Accuracy | Macro F1 | Weighted Precision | Weighted Recall |
|---------------------------------------------|----------|----------|---------------------|------------------|
| Full Model (TDCA + MHFCA, Red=16, Drop=0.4/0.2) | 0.9818   | 0.9804   | 0.9825              | 0.9818           |
| No MHFCA (TDCA only)                         | 0.8000   | 0.7991   | 0.8052              | 0.8000           |
| No TDCA (MHFCA only)                         | 0.9636   | 0.9548   | 0.9650              | 0.9636           |
| Baseline (No TDCA / MHFCA)                   | 0.4818   | 0.3254   | 0.4356              | 0.4818           |
| Low Reduction Ratio (Red=8)                  | 0.4273   | 0.3563   | 0.3950              | 0.4273           |
| Low Dropout Rate (Drop=0.2/0.1)              | 0.7000   | 0.6790   | 0.6921              | 0.7000           |

---

## 📉 Table 3: Statistical Validation Metrics with Baseline Comparison and Confidence Intervals

| Metric            | Baseline | Mean of Scores | Shapiro-Wilk p (Normality) | Wilcoxon p | 95% Confidence Interval |
|------------------|----------|----------------|-----------------------------|------------|--------------------------|
| Accuracy          | 0.770    | 0.996          | 0.0002                      | 0.0020     | (0.9930, 0.9997)         |
| Cohen’s Kappa     | 0.720    | 0.994          | 0.0002                      | 0.0020     | (0.9879, 0.9995)         |
| Jaccard Similarity| 0.720    | 0.994          | 0.0011                      | 0.0020     | (0.9839, 1.0007)         |

---

## 🔍 Table 4: Comparative Performance Analysis of Deep Learning and Hybrid Models for Lung Cancer Classification

| Paper                        | Method                                | Dataset              | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-----------------------------|----------------------------------------|----------------------|--------------|----------------|-------------|---------------|
| Lanjewar et al. (2023)      | Modified DenseNet + ML classifiers     | CT scans             | 95.23        | –              | 92.50       | –             |
| AR et al. (2023)            | LCD-Capsule Network                    | CT images            | 96.10        | 94.80          | 95.60       | –             |
| Dunn et al. (2023)          | DL + Radiomic analysis                 | CT scan images       | 92.40        | –              | –           | –             |
| Murthy & Prasad (2023)      | Adversarial Transformer Network        | CT scan dataset      | 93.75        | –              | –           | –             |
| Lyu et al. (2022)           | Transformer-based DL model             | Whole-brain MRI      | 88.60        | –              | –           | –             |
| Wang et al. (2020)          | ResNet + Transfer Learning             | CT images            | 96.00        | –              | –           | –             |
| Narin & Onur (2022)         | DL with tuned hyperparameters          | Lung cancer images   | 90.30        | –              | –           | –             |
| Ashhar et al. (2021)        | VGG19 (Best among CNNs)                | CT lung images       | 94.80        | –              | –           | –             |
| Pandit et al. (2023)        | DL + Enhanced optimization             | CT images            | 97.20        | –              | –           | –             |
| Reddy & Khanaa (2023)       | Intelligent DL algorithm               | CT dataset           | 95.80        | –              | –           | –             |
| Marappan et al. (2022)      | Lightweight DL model                   | Low-res CT images    | 93.40        | –              | –           | –             |
| Kumar et al. (2023)         | Improved UNet model                    | CT nodule dataset    | 91.30        | –              | –           | –             |
| Alsheikhy et al. (2023)     | Hybrid DL techniques                   | Custom dataset       | 94.70        | –              | –           | –             |
| Shimazaki et al. (2022)     | DL + Segmentation                      | Chest radiographs    | 89.90        | –              | –           | –             |
| Chaunzwa et al. (2021)      | DL for histology classification        | CT images            | 90.20        | –              | –           | –             |
| Masud et al. (2021)         | ML + DL classification framework       | CT images            | 92.10        | –              | –           | –             |
| Nithya & Vinod Chandra (2023)| ExtRanFS + ML                         | CT lung cancer images| 94.40        | –              | –           | –             |
| Ren et al. (2022)           | LCDAE ensemble model                   | CT lung datasets     | 93.70        | –              | –           | –             |
| Ren et al. (2024)           | Foundation model for segmentation      | Cancer image data    | 96.30        | –              | –           | –             |
| Ren et al. (2024)           | Triplet representation learning        | Biomedical datasets  | 91.50        | –              | –           | –             |
| Meeradevi et al. (2025)     | ML + DL + MADM system                  | CT datasets          | 95.20        | –              | –           | –             |
| Durgam et al. (2025)        | DL + Transformer integration           | Custom lung CT data  | 97.50        | –              | –           | –             |
| **Proposed TDCA-MHFCA Method** | **TDCA-MHFCA**                       | CT lung cancer images| **98.18**    | **98.25**      | **98.18**   | **98.04**     |

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For any queries or contributions, please email:

📧 your.email@domain.com


