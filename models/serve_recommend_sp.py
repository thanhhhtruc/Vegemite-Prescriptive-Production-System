
"""
Production Serving Script – Academic Revision
Backend logic for the Vegemite Prescriptive Production System.

Features:
- Loads pre-trained Global and Per-Part ensemble models (XGB, LGBM, Cat, RF).
- Implements synchronized feature engineering (Rolling/Difference) to match training.
- Joint Set Point (SP) optimization using objective grid search.
- Task 1 (Quality) and Task 2 (Downtime) simultaneous prediction.
"""

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# ML Frameworks - Ensure all potential winners are supported
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import joblib
except ImportError:
    # Minimal fallback or error if critical libraries are missing
    pass

class FeatureEngineer:
    """Replicates Step 4 of the training pipeline for real-time inference."""
    
    @staticmethod
    def compute_for_row(row_dict, history_df, use_cols, medians):
        """Calculates features for a candidate row given its part-specific context."""
        # 1. Base DataFrame from the candidate row
        df_cand = pd.DataFrame([row_dict])
        
        # 2. Append history if available to compute rolling/diff
        if history_df is not None and len(history_df) >= 2:
            df_combined = pd.concat([history_df, df_cand], ignore_index=True)
        else:
            df_combined = df_cand.copy()

        # 3. Dynamic Feature Generation (aligned with fuse_data_notebook.py)
        # SOTA V7.3+ uses suffix _roll3, _diff, _lag1, and _lag3
        sensor_cols = [c for c in df_combined.columns if any(x in c for x in [' PV', ' Level', ' speed', ' SP'])]
        
        # Limit to first 20 as in notebook to prevent feature explosion
        for col in sensor_cols[:20]:
            # Rolling mean (window=3)
            df_combined[f'{col}_roll3'] = df_combined[col].rolling(3, min_periods=1).mean()
            # First-order difference
            df_combined[f'{col}_diff'] = df_combined[col].diff().fillna(0)
            # Lags (capture preemptive patterns)
            df_combined[f'{col}_lag1'] = df_combined[col].shift(1).bfill()
            df_combined[f'{col}_lag3'] = df_combined[col].shift(3).bfill()
            
        # 4. Extract only the last row (the candidate) and align columns
        final_row = df_combined.tail(1).copy()
        
        # 5. Ensure all columns expected by the model exist and fill NaNs
        for col in use_cols:
            if col not in final_row.columns or pd.isna(final_row[col].iloc[0]):
                final_row[col] = medians.get(col, 0.0)
                
        return final_row[use_cols]

# Configuration and Paths
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output" # Assuming models are in output/ as per latest save cell
MODELS_DIR = BASE_DIR / "models" # Fallback search location

# Model Artifact Filenames
MODEL_PATHS = {
    'global': "vegemite_quality_global.joblib",
    'per_part': "vegemite_quality_specialists.joblib",
    'downtime': "vegemite_downtime.joblib",
    'artifacts': "vegemite_artifacts.joblib"
}

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
        self.m1_global = None
        self.m1_parts = {}
        self.m2_downtime = None
        self.artifacts = {}
        self.load_models()

    def load_models(self):
        """Loads all SOTA artifacts from the output directory."""
        search_dirs = [OUTPUT_DIR, MODELS_DIR, Path(".")]
        
        for d in search_dirs:
            art_path = d / MODEL_PATHS['artifacts']
            if art_path.exists():
                try:
                    self.artifacts = joblib.load(art_path)
                    self.m1_global = joblib.load(d / MODEL_PATHS['global'])
                    self.m1_parts = joblib.load(d / MODEL_PATHS['per_part'])
                    self.m2_downtime = joblib.load(d / MODEL_PATHS['downtime'])
                    sys.stderr.write(f"Server successfully initialized from {d}\n")
                    return
                except Exception as e:
                    sys.stderr.write(f"Initialization error in {d}: {e}\n")
        
        sys.stderr.write("Critical Error: Pre-trained models not found. Please run the training notebook first.\n")

    def get_model(self, part):
        """Dynamic routing to specialist or global model."""
        m = self.m1_parts.get(part, self.m1_global)
        if m is None:
            # Fallback to global if specialist failed to load
            return self.m1_global
        return m

    def optimize_sp(self, base_row, part, history_df):
        """Joint grid search optimization for best SP combination."""
        use_cols = self.artifacts.get('use_cols', [])
        medians = self.artifacts.get('medians', {})
        sp_cols = self.artifacts.get('sp_cols', [])
        le_q = self.artifacts.get('le_q')
        
        if not sp_cols or not le_q:
            return {k: base_row.get(v, medians.get(v, 0.0)) for k, v in UI_TO_SP.items()}, 0.0, 0.0

        # Optimization Parameters
        n_grid = 3 # 3^3 combinations for 3 most important SPs to stay responsive
        lambda_penalty = 0.4 # Weight for downtime avoidance
        
        # Select top 3 SPs to optimize
        target_sp = [c for c in UI_TO_SP.values() if c in sp_cols][:4]
        
        # Calculate bounds from medians if history not enough
        ranges = {}
        for c in target_sp:
            val = base_row.get(c, medians.get(c, 0.0))
            ranges[c] = np.linspace(val * 0.9, val * 1.1, n_grid)

        best_score = -999.0
        best_sp_vals = {k: base_row.get(v, medians.get(v, 0.0)) for k, v in UI_TO_SP.items()}
        best_p_good = 0.0
        best_p_dt = 0.0
        
        model = self.get_model(part)
        if model is None: return best_sp_vals, 0.0, 0.0
        
        good_idx = list(le_q.classes_).index('good') if 'good' in le_q.classes_ else 0

        # Perform Grid Search
        keys = list(ranges.keys())
        for values in itertools.product(*ranges.values()):
            cand_row = base_row.copy()
            for k, v in zip(keys, values):
                cand_row[k] = v
            
            # Recalculate features for this candidate
            X_cand = FeatureEngineer.compute_for_row(cand_row, history_df, use_cols, medians)
            
            # Predict
            try:
                p_vec = model.predict_proba(X_cand)[0]
                pg = p_vec[good_idx]
                p_dt = self.m2_downtime.predict_proba(X_cand)[0][1] if self.m2_downtime else 0.0
                
                score = pg - (lambda_penalty * p_dt)
                if score > best_score:
                    best_score = score
                    best_p_good = pg
                    best_p_dt = p_dt
                    for k, v in zip(keys, values):
                        # Find the UI key for this SP col
                        ui_key = SP_TO_UI_REVERSE(k)
                        best_sp_vals[ui_key] = v
            except:
                continue

        return best_sp_vals, best_p_good, best_p_dt

def SP_TO_UI_REVERSE(col_name):
    # Simple reverse mapping
    for k, v in UI_TO_SP.items():
        if v == col_name: return k
    return col_name

def main():
    server = VegemiteServer()
    
    # Read input from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            return
        body = json.loads(input_data)
    except Exception as e:
        sys.stderr.write(f"JSON input error: {e}\n")
        return

    # Extract inputs
    part = body.get("part", "Yeast - BRD")
    medians = server.artifacts.get('medians', {})
    use_cols = server.artifacts.get('use_cols', [])
    le_q = server.artifacts.get('le_q')
    
    # ── Robust Inference Loop ─────────────────────────────────────────────
    try:
        # Build base feature row
        current_row = medians.copy()
        for ui_key, col_name in UI_TO_SP.items():
            if ui_key in body:
                current_row[col_name] = float(body[ui_key])
        
        # Include any raw sensor data if provided
        sensors = body.get("sensors", {})
        if isinstance(sensors, dict):
            for k, v in sensors.items():
                if k in current_row: current_row[k] = float(v)

        # Contextual history for rolling features (placeholder)
        history_df = None 

        # 1. Prediction for Current Settings
        X_curr = FeatureEngineer.compute_for_row(current_row, history_df, use_cols, medians)
        model = server.get_model(part)
        
        if model is None:
            raise ValueError("Model not found in any search path.")
            
        p_vec = model.predict_proba(X_curr)[0]
        
        # SOTA V7.3: Threshold-Aware Prediction
        classes = list(le_q.classes_) if le_q else []
        low_bad_idx = classes.index('low_bad') if 'low_bad' in classes else -1
        
        thresh = 0.35
        if low_bad_idx != -1 and p_vec[low_bad_idx] >= thresh:
            pred_label = "LOW_BAD"
        else:
            pred_idx = int(np.argmax(p_vec))
            pred_label = str(le_q.inverse_transform([pred_idx])[0]).upper() if le_q else "UNKNOWN"
        
        good_idx = classes.index('good') if 'good' in classes else 0
        p_good = float(p_vec[good_idx])
        
        p_dt = 0.0
        if server.m2_downtime:
            p_dt = float(server.m2_downtime.predict_proba(X_curr)[0][1])

        # 2. Recommended Set Points
        rec_sp, rec_p_good, rec_p_dt = server.optimize_sp(current_row, part, history_df)

        # 3. Formulate Response
        response = {
            "recommendedSP": {k: float(v) for k, v in rec_sp.items()},
            "pGood": float(round(p_good, 4)),
            "pDowntime": float(round(p_dt, 4)),
            "prediction": pred_label,
            "downtimeRisk": float(round(p_dt * 100, 2)),
            "recommendedPGood": float(round(rec_p_good, 4)),
            "recommendedPDowntime": float(round(rec_p_dt, 4))
        }
    except Exception as e:
        sys.stderr.write(f"Inference Runtime Error: {e}\n")
        import traceback
        traceback.print_exc()
        # Safe Fallback
        response = {
            "error": str(e),
            "prediction": "ERROR",
            "pGood": 0.0,
            "pDowntime": 0.0,
            "downtimeRisk": 0.0,
            "recommendedSP": {k: float(medians.get(UI_TO_SP[k], 0.0)) if k in UI_TO_SP else 0.0 for k in UI_TO_SP.keys()},
            "recommendedPGood": 0.0,
            "recommendedPDowntime": 0.0
        }
    
    sys.stdout.write(json.dumps(response, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
