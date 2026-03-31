
"""
Production Serving Script – Advanced Revision
Backend logic for the Vegemite Prescriptive Production System.

Features:
- Loads the 6 new task-specific model artifacts and Configs.
- Synchronized feature engineering mapping directly to Window-Based (mean, max, min, delta) formats.
- Safe Grid Optimization honoring the rigorous +-5% bound constraints.
- Multi-class root-cause inference combined with Isolation Forest.
"""

import itertools
import json
import os
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Replicates the Window-based feature derivation for inference."""
    
    @staticmethod
    def compute_for_row(row_dict_clean, features_list):
        out = {}
        for f in features_list:
            if f.endswith('_mean'):
                out[f] = row_dict_clean.get(f[:-5], 0.0)
            elif f.endswith('_std'):
                out[f] = 0.0  # Require history for actual std, bypass for point-prediction
            elif f.endswith('_max'):
                out[f] = row_dict_clean.get(f[:-4], 0.0)
            elif f.endswith('_min'):
                out[f] = row_dict_clean.get(f[:-4], 0.0)
            elif f.endswith('_delta'):
                out[f] = 0.0  # Instantaneous diff is zero
            else:
                out[f] = row_dict_clean.get(f, 0.0)
                
        # Return dataframe strictly aligned to trained feature order
        return pd.DataFrame([out])[features_list]

# Configuration and Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models" 
CONFIG_DIR = BASE_DIR / "config"

# UI key to Canonical SP column mapping
UI_TO_SP = {
    "ffteFeedSolidsSP":       "FFTE Feed solids SP",
    "ffteProductionSolidsSP": "FFTE Production solids SP",
    "ffteSteamPressureSP":    "FFTE Steam pressure SP",
    "tfeOutFlowSP":           "TFE Out flow SP",
    "tfeProductionSolidsSP":  "TFE Production solids SP",
    "tfeVacuumPressureSP":    "TFE Vacuum pressure SP",
    "tfeSteamPressureSP":     "TFE Steam pressure SP",
}

class VegemiteServer:
    def __init__(self):
        self.m1_parts = {}
        self.m2_lgb = None
        self.m2_iso = None
        self.m2_scaler = None
        
        self.task1_features = {}
        self.task2_features = []
        self.task2_class_mapping = {}

        self.load_models()

    def load_models(self):
        """Loads SOTA configurations and joblibs delivered by ML Engineer."""
        try:
            # 1. Load Configurations
            if (CONFIG_DIR / "task1_features.json").exists():
                with open(CONFIG_DIR / "task1_features.json", 'r') as f:
                    self.task1_features = json.load(f)
                    
            if (CONFIG_DIR / "task2_features.json").exists():
                with open(CONFIG_DIR / "task2_features.json", 'r') as f:
                    self.task2_features = json.load(f)
                    
            if (CONFIG_DIR / "task2_class_mapping.json").exists():
                with open(CONFIG_DIR / "task2_class_mapping.json", 'r') as f:
                    # JSON keys are always strings -> convert back to int
                    self.task2_class_mapping = {int(k): v for k, v in json.load(f).items()}
                
            # 2. Load Task 1 (Quality Specialist Models)
            for part in ["Yeast - BRD", "Yeast - BRN", "Yeast - FMX"]:
                safe_name = part.replace(" ", "_").replace("-", "")
                path = MODELS_DIR / f"task1_model_{safe_name}.joblib"
                if path.exists():
                    self.m1_parts[part] = joblib.load(path)
            
            # 3. Load Task 2 (Downtime Ensemble & Scaler)
            lgb_path = MODELS_DIR / "task2_lightgbm_multiclass.joblib"
            if lgb_path.exists(): self.m2_lgb = joblib.load(lgb_path)
            
            iso_path = MODELS_DIR / "task2_isolation_forest.joblib"
            if iso_path.exists(): self.m2_iso = joblib.load(iso_path)
            
            scaler_path = MODELS_DIR / "task2_scaler.joblib"
            if scaler_path.exists(): self.m2_scaler = joblib.load(scaler_path)

            sys.stderr.write(f"Server successfully initialized from {MODELS_DIR}\n")
        except Exception as e:
            sys.stderr.write(f"Initialization error: {e}\n")

    def get_model1(self, part):
        return self.m1_parts.get(part)

    def optimize_sp(self, clean_row, part, p_good_curr):
        """Prescriptive Engine matching Optuna's +-5% physics limits."""
        model1 = self.get_model1(part)
        safe_part = part.replace(" ", "_").replace("-", "")
        t1_feats = self.task1_features.get(safe_part, [])
        
        # Base failure handling
        default_rec = {ui: float(clean_row.get(re.sub(r'[^A-Za-z0-9_]+', '_', canon), 0.0)) for ui, canon in UI_TO_SP.items()}
        
        if not model1 or not t1_feats:
            return default_rec, p_good_curr, 0.0

        # Dynamically discover which Set Points exist in model features
        sp_to_opt = []
        for feat in t1_feats:
            if "SP" in feat and not any(ext in feat for ext in ['_mean', '_std', '_max', '_min', '_delta']):
                sp_to_opt.append(feat)
                
        # Limit combinations to maintain sub-second api latency
        sp_to_opt = sp_to_opt[:3] 
        if not sp_to_opt:
            return default_rec, p_good_curr, 0.0

        # Constrained Ranges: strict +- 5% bound mapped exactly to ML review logic
        ranges = {}
        for c in sp_to_opt:
            val = clean_row.get(c, 0.1) + 1e-9
            if val > 0:
                ranges[c] = np.linspace(val * 0.95, val * 1.05, 3)
            elif val < 0:
                ranges[c] = np.linspace(val * 1.05, val * 0.95, 3)
            else:
                ranges[c] = np.linspace(-0.05, 0.05, 3)

        best_score = -999.0
        best_cand = clean_row.copy()
        best_pg = p_good_curr
        best_pdt = 0.0
        
        lambda_penalty = 0.02
        normal_idx = next((k for k,v in self.task2_class_mapping.items() if str(v).lower() == 'normal'), 0)

        for vals in itertools.product(*ranges.values()):
            cand_row = clean_row.copy()
            penalty = 0
            for k, v in zip(sp_to_opt, vals):
                orig = clean_row.get(k, 0.1) + 1e-9
                penalty += abs(v - float(orig)) / abs(float(orig))
                cand_row[k] = v
            
            # Formulate prediction
            X_t1 = FeatureEngineer.compute_for_row(cand_row, t1_feats)
            p_vec = model1.predict_proba(X_t1)[0]
            pg = float(p_vec[0]) # Assuming class 0 is Good
            
            pdt = 0.0
            if self.m2_lgb and self.task2_features:
                X_t2 = FeatureEngineer.compute_for_row(cand_row, self.task2_features)
                p_dt_vec = self.m2_lgb.predict_proba(X_t2)[0]
                pdt = 1.0 - float(p_dt_vec[normal_idx])
            
            # Objective criteria factoring safe bounds + quality + machine safety
            score = pg - (lambda_penalty * penalty) - (0.4 * pdt)
            if score > best_score:
                best_score = score
                best_cand = cand_row
                best_pg = pg
                best_pdt = pdt
                
        # Formulate mapping back to UI keys
        rec_sp = {}
        for ui, canonical in UI_TO_SP.items():
            clean_col = re.sub(r'[^A-Za-z0-9_]+', '_', canonical)
            rec_sp[ui] = float(best_cand.get(clean_col, clean_row.get(clean_col, 0.0)))
            
        return rec_sp, best_pg, best_pdt


def main():
    server = VegemiteServer()
    
    # Read input from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data: return
        body = json.loads(input_data)
    except Exception as e:
        sys.stderr.write(f"JSON input error: {e}\n")
        return

    # Extract inputs
    part = body.get("part", "Yeast - BRD")
    
    try:
        # Formulate base row from UI keys + extra sensors
        raw_row = {}
        for ui_key, col_name in UI_TO_SP.items():
            if ui_key in body:
                raw_row[col_name] = float(body[ui_key])
        
        sensors = body.get("sensors", {})
        if isinstance(sensors, dict):
            for k, v in sensors.items():
                raw_row[k] = float(v)

        # Standardize Names matching LightGBM Training Data
        clean_row = {}
        for k, v in raw_row.items():
            clean_k = re.sub(r'[^A-Za-z0-9_]+', '_', k)
            clean_row[clean_k] = v

        # -------------------------------------------------------------
        # TASK 1: QUALITY PREDICTION (Current Settings)
        # -------------------------------------------------------------
        pred_label = "UNKNOWN"
        p_good = 0.0
        
        model1 = server.get_model1(part)
        safe_part = part.replace(" ", "_").replace("-", "")
        t1_feats = server.task1_features.get(safe_part, [])
        
        if model1 and t1_feats:
            X_t1 = FeatureEngineer.compute_for_row(clean_row, t1_feats)
            p_vec = model1.predict_proba(X_t1)[0]
            
            p_good = float(p_vec[0])
            pred_idx = int(np.argmax(p_vec))
            
            # Map index strictly to notebook mapping
            if pred_idx == 0:
                pred_label = "GOOD"
            elif pred_idx == 1:
                pred_label = "LOW_BAD"
            elif pred_idx == 2:
                pred_label = "HIGH_BAD"

        # -------------------------------------------------------------
        # TASK 2: DOWNTIME ALERT (Multi-Class + Isolation Check)
        # -------------------------------------------------------------
        pred_dt_class = "Normal"
        p_dt_risk = 0.0
        iso_anomaly = False

        if server.m2_lgb and server.m2_iso and server.m2_scaler and server.task2_features:
            X_t2 = FeatureEngineer.compute_for_row(clean_row, server.task2_features)
            
            # Sub-Task 2A: Isolation Forest (Requires Scaling)
            X_t2_scaled = server.m2_scaler.transform(X_t2)
            iso_preds = server.m2_iso.predict(X_t2_scaled)
            if iso_preds[0] == -1:
                iso_anomaly = True

            # Sub-Task 2B: Root Cause Multi-Class LightGBM
            p_dt_vec = server.m2_lgb.predict_proba(X_t2)[0]
            pred_dt_idx = int(np.argmax(p_dt_vec))
            pred_dt_class = server.task2_class_mapping.get(pred_dt_idx, "Normal")
            
            normal_idx = next((k for k,v in server.task2_class_mapping.items() if str(v).lower() == 'normal'), 0)
            p_dt_risk = 1.0 - float(p_dt_vec[normal_idx])
            
            # Hard fallback risk mapping if isolation forest disagrees strongly
            if iso_anomaly and p_dt_risk < 0.5:
                p_dt_risk = 0.5 + (p_dt_risk / 2) # boost risk context
                if pred_dt_class == "Normal":
                    pred_dt_class = "Unknown_Anomaly"

        # -------------------------------------------------------------
        # PRESCRIPTIVE ENGINE (Recommendations)
        # -------------------------------------------------------------
        rec_sp, rec_p_good, rec_p_dt = server.optimize_sp(clean_row, part, p_good)

        response = {
            "recommendedSP": rec_sp,
            "pGood": float(round(p_good, 4)),
            "pDowntime": float(round(p_dt_risk, 4)),
            "prediction": pred_label,
            "downtimeRisk": float(round(p_dt_risk * 100, 2)),
            "rootCause": pred_dt_class,
            "isoAnomaly": iso_anomaly,
            "recommendedPGood": float(round(rec_p_good, 4)),
            "recommendedPDowntime": float(round(rec_p_dt, 4))
        }

        # 4. Safe Logging
        try:
            from datetime import datetime
            log_dir = BASE_DIR / "data"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "prediction_logs.csv"
            
            log_data = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Part": part}
            for ui_key in UI_TO_SP.keys():
                log_data[ui_key] = body.get(ui_key, "")
            log_data["Prediction"] = pred_label
            log_data["pGood"] = float(round(p_good, 4))
            log_data["pDowntimeRisk"] = float(round(p_dt_risk * 100, 2))
            log_data["RootCause"] = pred_dt_class
            
            pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=not log_file.exists(), index=False)
        except:
            pass

    except Exception as e:
        sys.stderr.write(f"Inference Runtime Error: {e}\n")
        response = {
            "error": str(e), "prediction": "ERROR",
            "pGood": 0.0, "pDowntime": 0.0, "downtimeRisk": 0.0, "rootCause": "Error",
            "recommendedSP": {k: 0.0 for k in UI_TO_SP.keys()},
            "recommendedPGood": 0.0, "recommendedPDowntime": 0.0
        }
    
    sys.stdout.write(json.dumps(response, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
