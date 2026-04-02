# Machine Learning Pipeline: Vegemite Prescriptive Production System

This document delineates the comprehensive Machine Learning architecture and data processing pipeline as implemented in the primary computational notebook (`group-1-theme-3.ipynb`).

## 1. Data Profiling & Understanding
This initial phase executes a systematic traversal of the dataset repositories to validate structural dimensions and identify ubiquitous missing values. Furthermore, it incorporates error-handling mechanisms for character encoding discrepancies (e.g., UTF-8 versus UTF-16) to ensure robust data ingestion algorithms.

## 2. Quality Sensor Data Unification
To facilitate downstream quality analysis, raw sensory data files are filtered based on discrete machine classifications. Target labels are programmatically inferred and unified from file nomenclatures, while redundant records are excised to synthesize a cohesive and statistically valid master dataset.

## 3. Downtime Log Cleaning
This step isolates empirical equipment failure logs and standardizes disparate string-based temporal entries into a unified, chronologically ordered datetime schema. This transformation is pivotal for establishing deterministic anchor points utilized in predictive warning window generation.

## 4. Sensor Value Imputation
To rectify decoding artifact anomalies within continuous sensory Trend logs, missing data points are deterministically imputed utilizing a Time-Series Forward-Fill (ffill) approach. This methodology strictly preserves physical continuity and structurally precludes the risk of future data leakage.

## 5. Downtime Integration & Windowed Features
Preprocessed time-series sensory telemetry is chronologically integrated with respective downtime event flags. The pipeline subsequently derives higher-order statistical features—such as rolling means, standard deviations, and differential deltas—over sliding windows to quantitatively delineate early-warning risk intervals.

## 6. EDA: Class Data Imbalance Check
A systematic evaluation of the sanitized quality dataset is conducted to quantify the categorical distribution of product batches. Utilizing statistical visualizations, this phase pre-emptively identifies class imbalances, an essential prerequisite for addressing potential algorithmic bias prior to the training phase.

## 7. EDA: Multicollinearity & SP vs PV Evaluation
This empirical phase investigates the presence of multicollinearity among sensory variables utilizing a correlation matrix heatmap. Additionally, time-series visualizations are generated to quantitatively contrast theoretical Setpoint (SP) configurations against actual Process Variables (PV), thereby highlighting macroscopic operational deviations.

## 8. EDA: Feature Contribution Analysis
An empirical assessment of feature importance is executed for both quality classification (Task 1) and downtime risk prediction (Task 2) tasks. A baseline Random Forest ensemble is utilized to estimate feature significance and discern dominant explanatory variables before commencing formal algorithm training.

## 9. Task 1: Product Quality Classification & Prescriptive Engine
A computationally efficient LightGBM classifier is trained to systematically detect early trajectories of product quality drift. Synergistically, a Multi-Output Random Forest Prescriptive Regressor is deployed to autonomously recommend optimally adjusted Setpoints (SP), presenting a mathematical paradigm for salvaging sub-optimal production batches.

## 10. Task 2: Hybrid Downtime Tracking Pipeline
A sophisticated Dual-Stage Machine Learning architecture is implemented to mitigate the prevalence of excessive False Alarms. Initially, an Isolation Forest constructs an unsupervised anomaly detection boundary; subsequently, a supervised LightGBM operates as a secondary diagnostic mechanism to ascertain the risk severity and interpret the root cause of the flagged anomalies.

## 11. Checkpoint Consolidation: Feature & Artifact Export
In the final deployment phase, all synthesized Machine Learning models, feature encoders, computational scalers, and structural configurations are serialized into `.joblib` and `.json` representations. This systematic export guarantees seamless integration with the backend user interface and ensures high-fidelity execution within the production inference environment.


