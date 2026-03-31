```md
# 🚀 Theme 3 – End-to-End ML Pipeline (High-Scoring Version)

## 🎯 Objective
Build a complete AI system that:
1. Recommends **optimal Set Points (SPs)** to achieve **GOOD product quality**
2. Detects **production downtime (anomalies)**

---

# 🧩 1. Overall System Architecture

```

INPUT: Current PV (sensor data) + current SP

→ Model 1: Quality Prediction
→ Model 2: SP Optimization (Prescriptive Engine)
→ Model 3: Downtime Detection

OUTPUT:

* Predicted Quality (good / low_bad / high_bad)
* Recommended SP values
* Downtime risk / anomaly alert

```

---

# 📦 2. Data Understanding

## Data Sources
- 3 files:
  - `good.csv`
  - `low_bad.csv`
  - `high_bad.csv`
- Downtime dataset (2 months)

## Key Columns
- `SP`: controllable variables
- `PV`: sensor values
- `Part`: yeast type ⚠️ IMPORTANT
- `Set Time`: timestamp

---

# 🧹 3. Data Preprocessing

## 3.1 Merge Dataset
- Combine all 3 files
- Add label:
```

good → 0
low_bad → 1
high_bad → 2

```

---

## 3.2 Split by Part (CRITICAL)
```

for each Part:
train separate model

```

---

## 3.3 Cleaning
- Handle missing values
- Remove duplicates
- Clip extreme outliers (IQR or percentile)

---

## 3.4 Time Handling
- Convert `Set Time` → datetime
- Extract:
  - hour
  - day
  - shift (optional)

---

# ⚙️ 4. Feature Engineering

## 4.1 Base Features
- All SP columns
- All PV columns

---

## 4.2 Derived Features
- Rolling mean (window=3,5)
- Rolling std
- Delta:
```

x(t) - x(t-1)

```

---

## 4.3 Interaction Features
- SP × PV (important for nonlinear behavior)

---

## 4.4 Encoding
- Part → Label Encoding or One-hot

---

# 🧠 5. TASK 1 — Quality Prediction Model

## 🎯 Goal
Predict product quality:
```

(SP + PV) → {good, low_bad, high_bad}

```

---

## ✅ Model Selection

### Main Model:
- **XGBoost Classifier**

### Backup Models:
- Random Forest
- CatBoost

---

## ⚙️ Training Setup

- Train per Part
- Train/Validation split: time-based or 80/20
- Handle imbalance:
  - class_weight
  - SMOTE (optional)

---

## 📊 Evaluation Metrics
- Accuracy
- F1-score (important)
- Confusion Matrix
- ROC-AUC (optional)

---

# 🔥 6. TASK 1 — SP Recommendation (Prescriptive Engine)

## 🎯 Goal
Find SP values that maximize probability of GOOD

---

## 🧠 Approach: Optimization on top of ML

### Step 1 — Fix current PV
```

PV = current state

```

---

### Step 2 — Search SP space

#### Option A — Grid Search (simple)
```

for each SP combination:
predict probability of GOOD
select best SP

```

---

#### Option B — Bayesian Optimization (BEST)
Use:
- Optuna

Objective:
```

maximize P(good | SP, PV)

```

---

## 📤 Output
- Optimal SP values
- Confidence score

---

# ⚠️ 7. TASK 2 — Downtime Detection

## 🎯 Goal
Detect anomaly / production failure

---

## 🧠 Approach 1 — Supervised Classification

### Model:
- XGBoost
- Random Forest

### Input:
- SP + PV

### Output:
- downtime / normal

---

## 🧠 Approach 2 — Anomaly Detection (RECOMMENDED)

### Train only on GOOD data

### Models:
- Isolation Forest ⭐
- One-Class SVM
- AutoEncoder (advanced)

---

## ⚙️ Logic
```

if anomaly_score > threshold:
→ downtime risk

```

---

## 📊 Metrics
- Precision / Recall
- F1-score
- ROC-AUC

---

# 🧪 8. Model Validation Strategy

## Cross Validation
- Time-based split preferred

---

## Comparison Table
| Model | Accuracy | F1 | Notes |
|------|--------|----|------|
| XGBoost |  |  | |
| RF |  |  | |

---

# 📈 9. Explainability (HIGH SCORE BONUS)

## Use:
- Feature Importance (XGBoost)
- SHAP values

---

## Show:
- Top features affecting quality
- Impact of SP on output

---

# 🖥️ 10. AI Demonstrator (UI)

## Input:
- Current PV + SP

## Output:
- Predicted Quality
- Recommended SP
- Downtime risk

---

## Suggested UI:
- Dashboard (Streamlit / React)
- Panels:
  - Prediction
  - Recommendation
  - Alert

---

# 🏆 11. Key Points to Maximize Marks

✅ Train separate model per Part  
✅ Implement SP optimization (NOT just prediction)  
✅ Include downtime detection  
✅ Compare multiple models  
✅ Use proper evaluation metrics  
✅ Show explainability  
✅ Build working demo  

---

# ❌ Common Mistakes

- Only classification (no recommendation)
- Ignoring Part separation
- No optimization step
- No anomaly detection
- No evaluation comparison

---

# ✅ Final Tech Stack

| Component | Tool |
|----------|------|
| Model | XGBoost |
| Optimization | Optuna |
| Anomaly | Isolation Forest |
| Visualization | SHAP |
| UI | Streamlit |

---

# 🚀 Final Summary

This project is NOT just ML — it is:
- Predictive + Prescriptive + Anomaly Detection system

👉 Core winning combo:
- XGBoost + Optuna + Isolation Forest

---
```
