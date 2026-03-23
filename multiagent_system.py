#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy nltk gensim scikit-learn torch transformers')


import pandas as pd
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch

nltk.download('punkt')
nltk.download('punkt_tab')


# 1. Load Dataset

# Expect CSV with columns:
df = pd.read_csv("job_title_des.csv")

# 2. Text Preprocessing

def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

df['clean_description'] = df['Job Description'].apply(clean_text)
df.dropna(subset=['clean_description'], inplace=True)

# 3. Tokenization

df['tokens'] = df['clean_description'].apply(nltk.word_tokenize)

# 4. TF-IDF Representation

vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['clean_description'])
tfidf_feature_names = vectorizer.get_feature_names_out()

# 5. Gender-coded and Exclusionary Terms Detection

masculine_words = [
    "ambitious", "confident", "decisive", "dominant", "fearless",
    "assertive", "independent", "leader", "objective"
]
feminine_words = [
    "supportive", "understanding", "committed", "interpersonal",
    "loyal", "responsible", "collaborative", "sensitive", "empathic"
]
exclusionary_terms = ["native speaker", "young", "energetic", "able-bodied"]

def detect_terms(tokens, term_list):
    return sum(1 for t in tokens if t in term_list)

df['masculine_terms'] = df['tokens'].apply(lambda x: detect_terms(x, masculine_words))
df['feminine_terms']  = df['tokens'].apply(lambda x: detect_terms(x, feminine_words))
df['exclusionary_terms'] = df['tokens'].apply(lambda x: detect_terms(x, exclusionary_terms))

df['gender_bias_ratio'] = (df['masculine_terms'] + 1) / (df['feminine_terms'] + 1)

# 6. Word2Vec Embeddings

model_w2v = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=2, workers=4)
word_vectors = model_w2v.wv

similarities = {
    'assertive-supportive': word_vectors.similarity('assertive', 'supportive') if 'assertive' in word_vectors and 'supportive' in word_vectors else None,
    'leader-collaborative': word_vectors.similarity('leader', 'collaborative') if 'leader' in word_vectors and 'collaborative' in word_vectors else None
}

# 7. BERT Embeddings

print("\nGenerating BERT embeddings...")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model_bert = AutoModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    # Mean pooling over token embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

# Generate embeddings for a subset
df['bert_embedding'] = df['clean_description'].sample(50).apply(get_bert_embedding)

print("✅ BERT embeddings generated successfully.")

# 8. Summary Statistics

summary = {
    "Total Jobs": len(df),
    "Avg Masculine Terms": df['masculine_terms'].mean(),
    "Avg Feminine Terms": df['feminine_terms'].mean(),
    "Avg Exclusionary Terms": df['exclusionary_terms'].mean(),
    "Avg Gender Bias Ratio": df['gender_bias_ratio'].mean()
}

print("\n=== Linguistic Bias Summary ===")
for k, v in summary.items():
    print(f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}")

print("\n=== Word2Vec Similarities ===")
for k, v in similarities.items():
    print(f"{k}: {v}")

# 9. Save Results

df.to_csv("job_bias_analysis_results.csv", index=False)
print("\nAnalysis complete. Results saved to 'job_bias_analysis_results.csv'.")


# In[2]:


get_ipython().system('pip install pandas numpy nltk torch transformers scikit-learn shap')

# resume_shortlisting_bias_analysis

import pandas as pd
import numpy as np
import re
import nltk
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import shap
import warnings

warnings.filterwarnings("ignore")

# 0. NLTK Downloads

nltk.download('punkt')

# 1. Load Datasets

# Resume dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

if not {'Category', 'Resume'}.issubset(df.columns):
    raise ValueError("The dataset must contain 'Category' and 'Skills' columns.")

print(f"✅ Loaded {len(df)} resumes across {df['Category'].nunique()} job categories.")

# Load bias indicators from Agent 1
job_bias_df = pd.read_csv("job_bias_analysis_results.csv")

# Keep relevant columns
job_bias_df = job_bias_df[[
    'Job Title', 'masculine_terms', 'feminine_terms',
    'exclusionary_terms', 'gender_bias_ratio'
]]

# 2. Merge Job Bias Data with Resume Data

print("\n🔹 Merging resume and job bias data...")

job_bias_df['job_title_clean'] = job_bias_df['Job Title'].astype(str).str.lower()
df['category_clean'] = df['Category'].astype(str).str.lower()

# Merge on job title
df = df.merge(job_bias_df, left_on='category_clean', right_on='job_title_clean', how='left')

# Fill missing bias columns
bias_cols = ['masculine_terms', 'feminine_terms', 'exclusionary_terms', 'gender_bias_ratio']
df[bias_cols] = df[bias_cols].fillna(0)

print(f"✅ Merged dataset shape: {df.shape}")
print("Bias feature sample:\n", df[bias_cols].head())

# 3. Text Preprocessing

print("\n🔹 Cleaning resume text...")

def clean_text(text):
    text = re.sub(r'[^A-Za-z\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

df['clean_resume'] = df['Resume'].apply(clean_text)

# 4. Encode Job Categories

label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['Category'])

# 5. Remove Tiny Classes to Avoid Stratify Error

category_counts = df['Category'].value_counts()
valid_categories = category_counts[category_counts >= 3].index
df = df[df['Category'].isin(valid_categories)]

print(f"✅ Filtered dataset to {len(df)} resumes across {len(valid_categories)} valid categories.\n")

# 6. Generate BERT Embeddings

print("🔹 Generating BERT embeddings (this may take several minutes)...")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Sample subset for computational efficiency
sample_df = df.sample(min(500, len(df)), random_state=42)
sample_df['embeddings'] = sample_df['clean_resume'].apply(get_bert_embedding)

# Combine embeddings with bias features
bias_features = sample_df[bias_cols].values
X_text = np.vstack(sample_df['embeddings'].values)
X = np.hstack((X_text, bias_features))

y = sample_df['category_encoded'].values

# 7. Train-Test Split

print("\n🔹 Splitting data into train/test sets...")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    print("⚠️ Some categories have too few examples — proceeding without stratification.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# 8. Train Logistic Regression

print("\n🔹 Training logistic regression classifier...")

clf = LogisticRegression(max_iter=2000, multi_class='multinomial')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.3f}")

# 9. Fairness Metrics

print("\n🔹 Calculating fairness metrics...")

np.random.seed(42)
results_df = pd.DataFrame({
    'true_category': label_encoder.inverse_transform(y_test),
    'predicted_category': label_encoder.inverse_transform(y_pred),
    'group': np.random.choice(['Group_A', 'Group_B'], size=len(y_test))
})

# Binary outcome: 1 = correctly shortlisted, 0 = not
results_df['shortlisted'] = np.where(
    results_df['true_category'] == results_df['predicted_category'], 1, 0
)

# Demographic Parity Difference
group_means = results_df.groupby('group')['shortlisted'].mean()
demographic_parity_diff = abs(group_means['Group_A'] - group_means['Group_B'])
print(f"📊 Demographic Parity Difference: {demographic_parity_diff:.3f}")

# Equal Opportunity Ratio
def tpr(df, group_name):
    subset = df[df['group'] == group_name]
    tp = (subset['shortlisted'] == 1).sum()
    fn = (subset['shortlisted'] == 0).sum()
    return tp / (tp + fn + 1e-6)

group_a_tpr = tpr(results_df, 'Group_A')
group_b_tpr = tpr(results_df, 'Group_B')
equal_opportunity_ratio = group_a_tpr / (group_b_tpr + 1e-6)
print(f"📊 Equal Opportunity Ratio (GroupA / GroupB): {equal_opportunity_ratio:.3f}")

# 10. SHAP Explainability

print("\n🔹 Explaining model predictions with SHAP...")

explainer = shap.LinearExplainer(clf, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=(0, 1))
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

top_features_idx = np.argsort(mean_abs_shap)[-10:][::-1]
top_features_values = mean_abs_shap[top_features_idx]

print("\n=== Top 10 Influential Features (Including Bias Indicators) ===")
for i, val in zip(top_features_idx, top_features_values):
    val_float = float(np.array(val).flatten()[0]) if isinstance(val, np.ndarray) else float(val)
    print(f"Feature {i}: SHAP Impact {val_float:.4f}")

# 11. Save Results

print("\n🔹 Saving results to file...")

results_df['gender_bias_ratio'] = sample_df['gender_bias_ratio'].values[:len(results_df)]
results_df.to_csv("resume_shortlisting_bias_results.csv", index=False)

print("\n✅ Integrated analysis complete.")
print("Results saved to 'resume_shortlisting_bias_results.csv'.")


# In[4]:


get_ipython().system('pip install pandas numpy torch transformers sentence-transformers scikit-learn')

import pandas as pd
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
import torch
import re

# -------------------------------
# 1. Load datasets
# -------------------------------
hr_df = pd.read_csv("hr_interview_dataset.csv")
agent2_df = pd.read_csv("resume_shortlisting_bias_results.csv")

# Limit for testing
hr_df = hr_df.head(100)

# -------------------------------
# 2. Prepare Agent 2 signals
# -------------------------------
n_repeat = int(np.ceil(len(hr_df) / len(agent2_df)))
hr_df["shortlisted"] = np.tile(agent2_df["shortlisted"].values, n_repeat)[:len(hr_df)]
hr_df["gender_bias_ratio"] = np.tile(agent2_df["gender_bias_ratio"].values, n_repeat)[:len(hr_df)]

scaler = MinMaxScaler()
hr_df["bias_signal"] = scaler.fit_transform(hr_df[["gender_bias_ratio"]])

# -------------------------------
# 3. Load T5 Model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# -------------------------------
# 4. Generate answers
# -------------------------------
generated_answers = []
for _, row in hr_df.iterrows():
    question = row["question"]
    shortlisted = "yes" if row["shortlisted"] == 1 else "no"
    bias_level = round(float(row["bias_signal"]), 2)
    prompt = (
        f"Question: {question} | "
        f"Bias level: {bias_level} | Shortlisted: {shortlisted}. "
        "Generate a professional and unbiased interview answer."
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(input_ids, max_length=80, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answers.append(answer)

hr_df["generated_answer"] = generated_answers

# -------------------------------
# 5. Compute similarity with ideal answers
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings_generated = embedder.encode(hr_df["generated_answer"], convert_to_tensor=True)
embeddings_ideal = embedder.encode(hr_df["ideal_answer"], convert_to_tensor=True)

similarities = util.pytorch_cos_sim(embeddings_generated, embeddings_ideal)
similarity_scores = [float(similarities[i][i]) for i in range(len(similarities))]
hr_df["similarity_score"] = similarity_scores

# -------------------------------
# 6. Save initial CSV
# -------------------------------
initial_file = "interview_answers_raw.csv"
hr_df.to_csv(initial_file, index=False, encoding="utf-8")
print(f"✅ Initial raw results saved to {initial_file}")

# -------------------------------
# 7. Clean messy CSV (incorporate your cleaning snippet)
# -------------------------------
with open(initial_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Remove BOM and stray quotes
clean_text = raw_text.replace("﻿", "").replace('"', "")

lines = [line.strip() for line in clean_text.strip().splitlines() if line.strip()]
header_line = lines[0]
data_lines = lines[1:]

rows = []
for line in data_lines:
    parts = re.split(r'\s{2,}|\t|,(?=\S)', line)
    parts = [p.strip() for p in parts if p.strip()]
    rows.append(parts)

expected_cols = [
    "question", "category", "role", "experience", "difficulty",
    "source_type", "ideal_answer", "keywords", "shortlisted",
    "gender_bias_ratio", "bias_signal", "generated_answer", "similarity_score"
]

normalized_rows = [
    row[:len(expected_cols)] + [""] * (len(expected_cols) - len(row))
    for row in rows
]

cleaned_df = pd.DataFrame(normalized_rows, columns=expected_cols)

# Convert numeric columns
numeric_cols = ["shortlisted", "gender_bias_ratio", "similarity_score"]
for col in numeric_cols:
    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

# Save cleaned CSV
cleaned_file = "interview_answers_cleaned.csv"
cleaned_df.to_csv(cleaned_file, index=False, encoding="utf-8")
print(f"✅ Cleaned results saved to {cleaned_file}")
print(cleaned_df.head(10))


# In[6]:


get_ipython().system('pip install pandas numpy scikit-learn scipy joblib fairlearn aif360 gender-guesser')

# hiring_decision

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, demographic_parity_difference, equalized_odds_difference
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import gender_guesser.detector as gender

warnings.filterwarnings("ignore")

# 1. Load datasets

structured_df = pd.read_csv("dataset.csv")
agent3_df = pd.read_csv("interview_answers_cleaned.csv")  # Agent 3 outputs

print("Structured dataset shape:", structured_df.shape)
print("Agent 3 dataset shape:", agent3_df.shape)

# 2. Align and merge datasets

n_rows = min(len(structured_df), len(agent3_df))
structured_df = structured_df.head(n_rows).reset_index(drop=True)
agent3_df = agent3_df.head(n_rows).reset_index(drop=True)

df = pd.concat(
    [structured_df, agent3_df[['generated_answer', 'similarity_score', 'shortlisted', 'gender_bias_ratio']]],
    axis=1
)
print("Merged dataset shape:", df.shape)

# 3. Encode target and sensitive features

# Encode hiring decision
le_target = LabelEncoder()
df['decision_encoded'] = le_target.fit_transform(df['decision'])

# Infer gender from Name
d = gender.Detector()
def infer_gender(name):
    first_name = str(name).split()[0]
    g = d.get_gender(first_name)
    if g in ['male', 'mostly_male']:
        return 1
    elif g in ['female', 'mostly_female']:
        return 0
    else:
        return np.nan

df['gender'] = df['Name'].apply(infer_gender)
df = df.dropna(subset=['gender'])
df['gender'] = df['gender'].astype(int)

# 4. Prepare features

# Structured numeric features: encode categorical columns
categorical_cols = ['Role']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Features: structured + Agent3 outputs
feature_cols = ['similarity_score', 'shortlisted', 'gender_bias_ratio'] + categorical_cols
X = df[feature_cols]
y = df['decision_encoded']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train classifier

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 6. Model performance

print("\n=== Model Performance ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# 7. Fairness metrics (Fairlearn)

sensitive_features = ['gender']
for feature in sensitive_features:
    metric_frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=df.loc[y_test.index, feature],
    )
    print(f"\nFairness for feature: {feature}")
    print(metric_frame.by_group)
    print(f"Demographic parity difference: {demographic_parity_difference(y_test, y_pred, sensitive_features=df.loc[y_test.index, feature]):.3f}")
    print(f"Equalized odds difference: {equalized_odds_difference(y_test, y_pred, sensitive_features=df.loc[y_test.index, feature]):.3f}")

# 8. Fairness metrics (AIF360)

# Prepare X_test and y_test for AIF360
X_test_df = pd.DataFrame(X_test, columns=feature_cols).reset_index(drop=True)
y_test_df = pd.Series(y_test).reset_index(drop=True).rename("hired")
X_test_df['gender'] = df.loc[y_test.index, 'gender'].reset_index(drop=True)

# Remove any rows with NA to satisfy AIF360
aif360_input = pd.concat([X_test_df, y_test_df], axis=1).dropna().reset_index(drop=True)

aif_data = BinaryLabelDataset(
    df=aif360_input,
    label_names=["hired"],
    protected_attribute_names=["gender"],
)

metric = BinaryLabelDatasetMetric(
    aif_data,
    privileged_groups=[{"gender": 1}],
    unprivileged_groups=[{"gender": 0}],
)

print("\nAIF360 Metrics:")
print("Difference in mean outcomes:", metric.mean_difference())
print("Statistical parity difference:", metric.disparate_impact())

# 9. Save output for Agent 5

df['predicted_decision'] = le_target.inverse_transform(clf.predict(X_scaled))
df.to_csv("agent4_hiring_predictions.csv", index=False)
print("\n✅ Agent 4 complete. Output saved to 'agent4_hiring_predictions.csv'.")


# In[9]:


get_ipython().system('pip install pandas numpy scikit-learn scipy joblib fairlearn aif360')

# post_hire_analysis

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

warnings.filterwarnings("ignore")

# 1. Load datasets

attrition_df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
agent4_df = pd.read_csv("agent4_hiring_predictions.csv")

print("Attrition dataset shape:", attrition_df.shape)
print("Agent 4 predictions shape:", agent4_df.shape)

# 2. Merge Agent 4 predictions

# Use index to merge since Attrition dataset has no Name column
n_rows = min(len(attrition_df), len(agent4_df))
attrition_df = attrition_df.head(n_rows).reset_index(drop=True)
agent4_df = agent4_df.head(n_rows).reset_index(drop=True)

# Add predicted_decision from Agent 4
df = pd.concat([attrition_df, agent4_df['predicted_decision']], axis=1)

# Encode Agent 4 prediction
le_agent4 = LabelEncoder()
df['predicted_decision_encoded'] = le_agent4.fit_transform(df['predicted_decision'])

# 3. Preprocess Attrition dataset

# Sensitive feature
df['gender_num'] = df['Gender'].map({'Female':0, 'Male':1})

# Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in ['Gender', 'Attrition']]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode target
df['Attrition'] = df['Attrition'].map({'No':0, 'Yes':1})
df['HighPerformance'] = (df['PerformanceRating'] > 3).astype(int)

# 4. Feature selection

# Include Agent 4 predicted hiring decision as a feature
feature_cols_attr = [c for c in df.columns if c not in ['Attrition','HighPerformance','Gender','predicted_decision']]
feature_cols_attr.append('predicted_decision_encoded')  # add agent4 prediction
X_attr = df[feature_cols_attr]
y_attr = df['Attrition']

feature_cols_perf = [c for c in df.columns if c not in ['HighPerformance','Attrition','Gender','predicted_decision']]
feature_cols_perf.append('predicted_decision_encoded')
X_perf = df[feature_cols_perf]
y_perf = df['HighPerformance']

# Standardize
scaler_attr = StandardScaler()
X_attr_scaled = scaler_attr.fit_transform(X_attr)

scaler_perf = StandardScaler()
X_perf_scaled = scaler_perf.fit_transform(X_perf)

# Train/test split
X_train_attr, X_test_attr, y_train_attr, y_test_attr, idx_train_attr, idx_test_attr = train_test_split(
    X_attr_scaled, y_attr, df.index, test_size=0.2, random_state=42, stratify=y_attr
)
X_train_perf, X_test_perf, y_train_perf, y_test_perf, idx_train_perf, idx_test_perf = train_test_split(
    X_perf_scaled, y_perf, df.index, test_size=0.2, random_state=42, stratify=y_perf
)

# 5. Model training - Attrition

print("\nTraining Random Forest for Attrition prediction...")
rf_attr = RandomForestClassifier(n_estimators=200, random_state=42)
rf_attr.fit(X_train_attr, y_train_attr)
y_attr_pred = rf_attr.predict(X_test_attr)

print("\nAttrition Prediction Performance:")
print(classification_report(y_test_attr, y_attr_pred))
print("Accuracy:", accuracy_score(y_test_attr, y_attr_pred))

# 6. Model training - Performance

print("\nTraining Logistic Regression for Performance prediction...")
lr_perf = LogisticRegression(max_iter=1000, C=0.1, penalty='l2')
lr_perf.fit(X_train_perf, y_train_perf)
y_perf_pred = lr_perf.predict(X_test_perf)

print("\nPerformance Prediction:")
print(classification_report(y_test_perf, y_perf_pred))
print("Accuracy:", accuracy_score(y_test_perf, y_perf_pred))

# 7. Fairness metrics - Attrition

print("\n=== Fairness Assessment (Attrition) ===")
metric_frame_attr = MetricFrame(
    metrics={
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    },
    y_true=y_test_attr,
    y_pred=y_attr_pred,
    sensitive_features=df.loc[idx_test_attr,'gender_num']
)
print(metric_frame_attr.by_group)
print("Demographic parity difference:", demographic_parity_difference(y_test_attr, y_attr_pred, sensitive_features=df.loc[idx_test_attr,'gender_num']))
print("Equalized odds difference:", equalized_odds_difference(y_test_attr, y_attr_pred, sensitive_features=df.loc[idx_test_attr,'gender_num']))

# AIF360 metric
aif_data_attr = BinaryLabelDataset(
    df=pd.concat([
        pd.DataFrame(X_test_attr, columns=feature_cols_attr).reset_index(drop=True),
        pd.Series(y_test_attr).reset_index(drop=True).rename('attrition'),
        df.loc[idx_test_attr,'gender_num'].reset_index(drop=True).rename('gender')
    ], axis=1),
    label_names=['attrition'],
    protected_attribute_names=['gender']
)
metric_attr = BinaryLabelDatasetMetric(
    aif_data_attr,
    privileged_groups=[{'gender':1}],
    unprivileged_groups=[{'gender':0}]
)
print("Difference in mean outcomes (attrition):", metric_attr.mean_difference())
print("Statistical parity difference:", metric_attr.disparate_impact())

# 8. Fairness metrics - Performance

print("\n=== Fairness Assessment (Performance) ===")
metric_frame_perf = MetricFrame(
    metrics={
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    },
    y_true=y_test_perf,
    y_pred=y_perf_pred,
    sensitive_features=df.loc[idx_test_perf,'gender_num']
)
print(metric_frame_perf.by_group)
print("Demographic parity difference:", demographic_parity_difference(y_test_perf, y_perf_pred, sensitive_features=df.loc[idx_test_perf,'gender_num']))
print("Equalized odds difference:", equalized_odds_difference(y_test_perf, y_perf_pred, sensitive_features=df.loc[idx_test_perf,'gender_num']))

# 9. Save results

results_df = pd.DataFrame({
    'EmployeeIndex': df.loc[idx_test_attr].index,
    'Gender': df.loc[idx_test_attr,'Gender'],
    'AttritionTrue': y_test_attr.values,
    'AttritionPred': y_attr_pred,
    'HighPerformanceTrue': y_test_perf.values,
    'HighPerformancePred': y_perf_pred,
    'PredictedHiringDecision': df.loc[idx_test_attr,'predicted_decision'].values
})
results_df.to_csv('post_hire_analysis', index=False)
print("\n✅ Post-hire analysis results saved to 'post_hire_analysis_with_agent4.csv'.")

