# 🤖 Shadow AI in HR Recruitment
### Detecting Bias and Governance Risks Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-Embeddings-orange?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=flat)
![Fairlearn](https://img.shields.io/badge/Fairlearn-Bias_Audit-blueviolet?style=flat)
![AIF360](https://img.shields.io/badge/AIF360-Fairness_Metrics-red?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=flat)

> **Built by Tobi Omoode — People Scientist & HR Analytics practitioner**
> A production-grade 5-agent AI recruitment pipeline that detects bias and governance risks at every stage of hiring — from job description to post-hire monitoring.

---

## 🎯 The Problem

AI is increasingly being used in recruitment — but most organisations don't know whether their hiring systems are biased. This project investigates **Shadow AI in HR**: the hidden use of AI tools in recruitment pipelines and the governance risks they create.

**Key questions this project answers:**
- Do job descriptions contain gendered or age-biased language at scale?
- Do AI shortlisting systems treat candidates fairly across gender and ethnicity?
- Can ML models predict hiring decisions with high accuracy — and are those decisions fair?
- What happens to attrition risk for employees hired through a potentially biased process?

---

## 🏗️ Architecture — 5-Agent Pipeline

Each agent passes its output to the next, creating an end-to-end auditable recruitment system.

```
Raw Data (Job Descriptions + Resumes + Interview Data + IBM Watson HR)
       ↓
Agent 1: Job Description Analysis
  → Bias term extraction (gender, age, ethnicity)
  → TF-IDF · Word2Vec · BERT embeddings
  → 2,277 job descriptions analysed
       ↓
Agent 2: Resume Screening
  → ML-based shortlisting with fairness evaluation
  → Fairlearn + AIF360 bias metrics
  → 22,872 candidate-role combinations scored
       ↓
Agent 3: Interview Evaluation
  → Sentence Transformer semantic similarity (all-MiniLM-L6-v2)
  → DistilBERT response quality prediction
  → 10,000 candidates assessed
       ↓
Agent 4: Hiring Decision Prediction
  → Ensemble model (XGBoost 50% + Random Forest 30% + Logistic Regression 20%)
  → 1,664 features (SBERT embeddings + TF-IDF + engineered features)
  → 10,000 hiring decisions predicted
       ↓
Agent 5: Post-Hire Monitoring
  → Attrition + performance prediction with multi-dimensional fairness audit
  → Fairness metrics by gender, age, and ethnicity
  → 1,470 employees monitored (IBM Watson HR dataset)
```

---

## 📊 Real Results

### Agent 4 — Hiring Decision Ensemble Model

| Metric | Score |
|--------|-------|
| **Accuracy** | **91.5%** |
| **Precision** | 91.1% |
| **Recall** | 91.7% |
| **F1-Score** | 91.4% |
| **ROC-AUC** | **0.985** |

Ensemble breakdown: XGBoost (50%) + Random Forest (30%) + Logistic Regression (20%)

### Agent 1 — Job Description Bias

| Metric | Finding |
|--------|---------|
| Job descriptions analysed | 2,277 |
| Average masculine terms per JD | 1.02 |
| Average feminine terms per JD | 1.06 |
| Average age bias terms per JD | 0.23 |
| Average gender bias ratio | 1.30 |

> Gender bias ratio > 1.0 indicates masculine language slightly dominates across the corpus.

### Agent 2 — Resume Shortlisting Fairness

| Metric | Score |
|--------|-------|
| Demographic Parity Difference | **0.000** |
| Disparate Impact (AIF360) | **0.985** |
| Statistical Parity Difference | -0.009 |
| Shortlisting rate | 99.9% |

> DI of 0.985 is within the 0.8–1.2 "four-fifths rule" acceptable range — the shortlisting model is statistically fair.

### Agent 5 — Post-Hire Attrition Model

| Metric | Score |
|--------|-------|
| Accuracy | 82.7% |
| Precision | 35.7% |
| Recall | 10.6% |
| F1-Score | 16.4% |
| Best params | RF: 200 trees, max depth 12 |

---


## 📈 Pipeline Visualisations

> Generated automatically when the notebook runs end-to-end. Charts saved to `/content/outputs/`.

### Bias Progression Across the Pipeline
![Bias Progression](outputs/pipeline_bias_progression.png)
*How bias signals change at each stage — from job description through to hiring decision*

### Model Performance Comparison
![Model Performance](outputs/pipeline_model_performance.png)
*Accuracy, F1-Score and ROC-AUC across all 5 agents*

### Fairness Dashboard
![Fairness Dashboard](outputs/fairness_dashboard.png)
*Demographic parity and disparate impact metrics by gender and age group*

---
## 🔬 Technical Details

### Models Used

| Agent | Model | Purpose |
|-------|-------|---------|
| Agent 1 | BERT (bert-base-uncased) | Job description embeddings |
| Agent 1 | Word2Vec (9,796 vocab) | Semantic word relationships |
| Agent 2 | Logistic Regression | Resume shortlisting (0.999 test accuracy) |
| Agent 2 | SHAP + LIME | Explainability |
| Agent 3 | all-MiniLM-L6-v2 | Semantic similarity (384-dim embeddings) |
| Agent 3 | DistilBERT (66.9M params) | Response quality prediction |
| Agent 4 | XGBoost (500 estimators) | Hiring decision component |
| Agent 4 | Random Forest (200 trees) | Hiring decision component |
| Agent 4 | Logistic Regression | Hiring decision component |
| Agent 5 | Random Forest + GridSearchCV | Attrition prediction |
| Agent 5 | Logistic Regression (L2, C=10) | Performance prediction |

### Fairness Frameworks

Two industry-standard fairness libraries used in parallel:

**Fairlearn** — measures:
- Demographic Parity Difference
- Equalized Odds Difference
- Selection rates by sensitive attribute (gender, age, ethnicity)

**AIF360 (IBM)** — measures:
- Disparate Impact (DI)
- Statistical Parity Difference (SPD)
- Applied to hiring and post-hire outcomes

### Feature Engineering (Agent 4)

```
Total features: 1,664
  ├── Resume SBERT embeddings:      384 dimensions
  ├── Interview SBERT embeddings:   384 dimensions
  ├── Interaction embeddings:       384 dimensions
  ├── TF-IDF features:              500 features
  └── Engineered numeric features:   12 features
       (bias_signal, similarity_score, interview_score,
        distilbert_quality, resume_length, answer_resume_ratio,
        semantic_deviation, bias_weighted_answer...)
```

---

## 📁 Generated Outputs

When the notebook runs end-to-end, it produces:

**Data outputs** (`/content/shared_data/`):
- `agent1_job_bias.csv` — bias scores for 2,277 job descriptions
- `agent2_resume_shortlisting.csv` — shortlisting results for 22,872 candidates
- `agent3_interview_results.csv` — interview scores for 10,000 candidates
- `agent4_hiring_predictions.csv` — hiring decisions for 10,000 candidates
- `agent5_post_hire_results.csv` — attrition/performance results for 1,470 employees
- `agent5_metrics.json` — full fairness metrics report
- `mitigation_results.json` — bias mitigation analysis

**Visualisations** (`/content/outputs/`):
- `pipeline_bias_progression.png` — bias scores tracked across all 5 pipeline stages
- `pipeline_model_performance.png` — accuracy/F1/AUC comparison across agents
- `fairness_dashboard.png` — demographic parity and disparate impact dashboard

---

## 🗂️ Repository Structure

```
shadow-ai-recruitment-bias-detection/
│
├── shadow_ai_recruitment_pipeline.ipynb   # Full 5-agent pipeline (119 cells)
└── README.md
```

---

## ⚡ How to Run

### Google Colab (recommended — GPU required for BERT)

1. Open `shadow_ai_recruitment_pipeline.ipynb` in Google Colab
2. **Runtime → Change runtime type → GPU** (T4 GPU recommended)
3. Run all cells — the pipeline takes approximately 30–45 minutes end-to-end

**Datasets required** (place in `/content/` or upload when prompted):
- Job descriptions dataset (2,277 records)
- Resume dataset (962 records — Kaggle: [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset))
- Interview dataset (`ankshi_interview_dataset.csv` — 20,000 records)
- IBM Watson HR dataset (1,470 records — auto-loaded from GitHub in Agent 5)

### Dependencies

```bash
pip install pandas numpy scikit-learn nltk gensim transformers torch sentence-transformers
pip install fairlearn aif360 shap lime xgboost
pip install matplotlib seaborn plotly
```

---

## 📦 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 (Google Colab) |
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP Models | BERT, DistilBERT, Sentence Transformers, Word2Vec |
| ML | XGBoost, scikit-learn (RF, LR, GridSearchCV) |
| Explainability | SHAP, LIME |
| Fairness | Fairlearn, IBM AIF360 |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Feature Engineering | TF-IDF, SBERT embeddings, interaction features |

---

## 💡 Key Findings

**1. Bias in job descriptions is measurable but mild.** The average gender bias ratio of 1.30 indicates a slight lean towards masculine language, but no extreme outliers. Age bias terms appeared in 23% of descriptions on average.

**2. The shortlisting model is statistically fair.** Disparate Impact of 0.985 (well within the 0.8–1.2 acceptable range) and Demographic Parity Difference of 0.000 indicate the resume screening agent does not systematically disadvantage candidates by gender.

**3. Hiring prediction is highly accurate.** 91.5% accuracy and 0.985 ROC-AUC on 10,000 candidates — the ensemble model learns genuine signal from the combination of semantic embeddings, interview performance, and bias features.

**4. Post-hire attrition remains hard to predict.** 82.7% accuracy but only 16.4% F1-score on leavers — consistent with the IBM Watson dataset challenge seen in standalone attrition models. This finding connects directly to Project 01 of this portfolio.

---

## 🔭 What I Would Build Next

- [ ] Real-time bias monitoring dashboard (Streamlit) with live JD scoring
- [ ] Adversarial debiasing using AIF360's `AdversarialDebiasing` algorithm
- [ ] Counterfactual fairness testing — would the same candidate be hired if their gender/age were different?
- [ ] Integration with live ATS (Greenhouse, Lever) via API
- [ ] Audit trail generation — GDPR-compliant logging of every AI decision with explainability

---

## 📚 Datasets

| Dataset | Source | Records |
|---------|--------|---------|
| Job Descriptions | Kaggle | 2,277 |
| Resumes | [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) | 962 |
| Interview Questions | ankshi_interview_dataset | 20,000 |
| IBM Watson HR | [Kaggle IBM HR](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) | 1,470 |

---

## 👤 About

I'm **Tobi Omoode**, a People Scientist building at the intersection of people data, machine learning, and strategic HR. This is Project 02 of a 6-project HR Analytics portfolio.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tobi_Omoode-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/tobi8943)
[![GitHub](https://img.shields.io/badge/GitHub-mztatee15--stack-181717?style=flat&logo=github)](https://github.com/mztatee15-stack)
[![Project 01](https://img.shields.io/badge/Portfolio-Attrition_Engine-185FA5?style=flat)](https://github.com/mztatee15-stack/predictive-attrition-model-by-Tobi)

---

## ⚠️ Disclaimer

All datasets used are publicly available or synthetically generated. No real candidate data was used. Fairness metrics are reported for research purposes — this system is designed to audit and flag bias, not to make real hiring decisions. Any AI tool used in recruitment should be subject to human oversight and regular bias auditing in accordance with EU AI Act and GDPR requirements.

---

*Project 02 of 6 · HR Analytics Portfolio · Tobi Omoode · 2026*
