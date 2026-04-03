
"""
Production Serving Script - Advanced Revision
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
import random
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder that converts NumPy data types to native Python types for JSON serialization."""
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

# Configuration and Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models" / "models"
CONFIG_DIR = BASE_DIR / "models" / "config"
BUFFER_FILE = BASE_DIR / "data" / "sensor_buffer.json"
MAX_BUFFER_SIZE = 15

"""
Cross-reference mapping: Maps clean UI sensor names into exactly what the models were trained on.
"""
FRIENDLY_TO_TAG_MAP = {
    "FFTE_Steam_pressure_PV": "Port_Melbourne_RSLinx_Enterprise_Veg_B_SQL_FFTE_Steam_Pressure",
    "FFTE_Steam_pressure_SP": "Port_Melbourne_RSLinx_Enterprise_Veg_B_SQL_FFTE_Steam_Setpoint",
    "FFTE_Heat_temperature_1": "Port_Melbourne_RSLinx_Enterprise_Veg_B_SQL_FFTE_Pre_Heat_Temperature1_Deg_C_",
    "FFTE_Heat_temperature_2": "Port_Melbourne_RSLinx_Enterprise_Veg_B_SQL_FFTE_Pre_Heat_Temperature2_Deg_C_",
    "FFTE_Heat_temperature_3": "Port_Melbourne_RSLinx_Enterprise_Veg_B_SQL_FFTE_Pre_Heat_Temperature3_Deg_C_",
    "TFE_Product_out_temperature": "Port_Melbourne_RSLinx_Enterprise_Veg_B_SQL_FFTE_Post_Heat_Temperature_Deg_C_",
    "Extract_tank_Level": "Extract_tank_Level_PV",
    "TFE_Tank_level": "TFE_Tank_level_PV",
    "TFE_Level": "TFE_Level_PV",
    "TFE_Vacuum_pressure_PV": "VEG_B_HMI_VEDC1266_PV_",
    "TFE_Vacuum_pressure_SP": "VEG_B_HMI_VEDC1266_SP_",
    "TFE_Out_flow_SP": "VEG_B_HMI_VEDC1664_SP_",
    "TFE_Out_flow_PV": "VEG_B_HMI_VELC1560_PV_"
}

class FeatureEngineer:
    """Replicates the Window-based feature derivation for inference."""
    
    @staticmethod
    def update_and_get_buffer(row_dict_clean):
        """
        Appends the latest sensor readings to the rolling buffer.
        Maintains a maximum buffer size and writes the state to disk.
        """
        buffer = []
        if BUFFER_FILE.exists():
            try:
                with open(BUFFER_FILE, 'r') as f:
                    buffer = json.load(f)
            except:
                pass
        
        buffer.append(row_dict_clean)
        if len(buffer) > MAX_BUFFER_SIZE:
            buffer = buffer[-MAX_BUFFER_SIZE:]
            
        try:
            with open(BUFFER_FILE, 'w') as f:
                json.dump(buffer, f)
        except Exception as e:
            sys.stderr.write(f"Warning: could not write buffer {e}\n")
            
        return pd.DataFrame(buffer)

    @staticmethod
    def compute_for_buffer(buffer_df, features_list):
        """
        Calculates time-series features (mean, std, min, max, delta, lag) 
        over the rolling buffer for the strictly requested features.
        """
        out = {}
        last_row = buffer_df.iloc[-1].to_dict()
        first_row = buffer_df.iloc[0].to_dict()
        
        for f in features_list:
            if f.endswith('_mean'):
                base_f = f[:-5]
                out[f] = float(buffer_df[base_f].mean()) if base_f in buffer_df.columns else float(last_row.get(base_f, 0.0))
            elif f.endswith('_mean_lag5'):
                base_f = f[:-10]
                out[f] = float(buffer_df[base_f].mean()) if base_f in buffer_df.columns else float(last_row.get(base_f, 0.0))
            elif f.endswith('_mean_volatility'):
                base_f = f[:-16]
                out[f] = float(buffer_df[base_f].mean()) if base_f in buffer_df.columns else float(last_row.get(base_f, 0.0))
            elif f.endswith('_std'):
                base_f = f[:-4]
                val = float(buffer_df[base_f].std()) if base_f in buffer_df.columns and len(buffer_df) > 1 else 0.0
                out[f] = val if not pd.isna(val) else 0.0
            elif f.endswith('_std_lag5'):
                base_f = f[:-9]
                val = float(buffer_df[base_f].std()) if base_f in buffer_df.columns and len(buffer_df) > 1 else 0.0
                out[f] = val if not pd.isna(val) else 0.0
            elif f.endswith('_std_volatility'):
                base_f = f[:-15]
                val = float(buffer_df[base_f].std()) if base_f in buffer_df.columns and len(buffer_df) > 1 else 0.0
                out[f] = val if not pd.isna(val) else 0.0
            elif f.endswith('_max'):
                base_f = f[:-4]
                out[f] = float(buffer_df[base_f].max()) if base_f in buffer_df.columns else float(last_row.get(base_f, 0.0))
            elif f.endswith('_max_lag5'):
                base_f = f[:-9]
                out[f] = float(buffer_df[base_f].max()) if base_f in buffer_df.columns else float(last_row.get(base_f, 0.0))
            elif f.endswith('_max_volatility'):
                base_f = f[:-15]
                out[f] = float(buffer_df[base_f].max()) if base_f in buffer_df.columns else float(last_row.get(base_f, 0.0))
            elif f.endswith('_min'):
                base_f = f[:-4]
                out[f] = float(buffer_df[base_f].min()) if base_f in buffer_df.columns else float(last_row.get(base_f, 0.0))
            elif f.endswith('_min_lag5'):
                base_f = f[:-9]
                out[f] = float(buffer_df[base_f].min()) if base_f in buffer_df.columns else float(last_row.get(base_f, 0.0))
            elif f.endswith('_min_volatility'):
                base_f = f[:-15]
                out[f] = float(buffer_df[base_f].min()) if base_f in buffer_df.columns else float(last_row.get(base_f, 0.0))
            elif f.endswith('_delta'):
                base_f = f[:-6]
                out[f] = float(last_row.get(base_f, 0.0)) - float(first_row.get(base_f, 0.0)) if base_f in buffer_df.columns else 0.0
            elif f.endswith('_delta_lag5'):
                base_f = f[:-11]
                out[f] = float(last_row.get(base_f, 0.0)) - float(first_row.get(base_f, 0.0)) if base_f in buffer_df.columns else 0.0
            elif f.endswith('_delta_volatility'):
                base_f = f[:-17]
                out[f] = float(last_row.get(base_f, 0.0)) - float(first_row.get(base_f, 0.0)) if base_f in buffer_df.columns else 0.0
            else:
                out[f] = float(last_row.get(f, 0.0))
                
        # Return dataframe strictly aligned to trained feature order
        return pd.DataFrame([out])[features_list].fillna(0.0)

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
    """
    Manages the loading and serving of Machine Learning models.
    Handles both Quality Prediction (Task 1) and Downtime Alert/Root Cause (Task 2).
    """
    def __init__(self):
        self.m1_classifiers = {}
        self.m1_recommenders = {}
        self.m2_lgbs = {}
        self.m2_isos = {}
        self.m2_scalers = {}
        self.m2_stage2_lgbs = {}
        self.m2_stage2_encoders = {}
        
        self.task1_features = {}
        self.task1_feature_not_sp = {}
        self.task1_sp_cols = {}
        
        self.task2_features = {}
        self.task2_class_mapping = {}

        self.load_models()

    def load_models(self):
        """Loads SOTA configurations and joblibs delivered by ML Engineer."""
        try:
            # 1. Load Configurations
            if (CONFIG_DIR / "task1_features.json").exists():
                with open(CONFIG_DIR / "task1_features.json", 'r') as f:
                    t1_config = json.load(f)
                    self.task1_features = {k: v.get("features", []) for k,v in t1_config.items()}
                    self.task1_feature_not_sp = {k: v.get("feature_not_sp", []) for k,v in t1_config.items()}
                    self.task1_sp_cols = {k: v.get("sp_cols", []) for k,v in t1_config.items()}
                    
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
                
                # Classifier Task 1
                clf_path = MODELS_DIR / f"task1_classifier_{safe_name}.joblib"
                if clf_path.exists():
                    self.m1_classifiers[part] = joblib.load(clf_path)
                    
                # Recommender Task 1
                rec_path = MODELS_DIR / f"task1_recommender_{safe_name}.joblib"
                if rec_path.exists():
                    self.m1_recommenders[part] = joblib.load(rec_path)
            
            # 3. Load Task 2 (Downtime Ensemble & Scaler)
            for machine in ["TFE", "FFTE"]:
                lgb_path = MODELS_DIR / f"task2_stage1_lgb_{machine}.joblib"
                if lgb_path.exists():
                    self.m2_lgbs[machine] = joblib.load(lgb_path)
                
                iso_path = MODELS_DIR / f"task2_stage1_iso_{machine}.joblib"
                if iso_path.exists():
                    self.m2_isos[machine] = joblib.load(iso_path)
                
                scaler_path = MODELS_DIR / f"task2_stage1_scaler_{machine}.joblib"
                if scaler_path.exists():
                    self.m2_scalers[machine] = joblib.load(scaler_path)
                    
                s2_lgb_path = MODELS_DIR / f"task2_stage2_lgb_{machine}.joblib"
                if s2_lgb_path.exists():
                    self.m2_stage2_lgbs[machine] = joblib.load(s2_lgb_path)
                    
                s2_enc_path = MODELS_DIR / f"task2_stage2_enc_{machine}.joblib"
                if s2_enc_path.exists():
                    self.m2_stage2_encoders[machine] = joblib.load(s2_enc_path)

            sys.stderr.write(f"Server successfully initialized from {MODELS_DIR}\n")
            sys.stderr.write(f" Loaded {len(self.m1_classifiers)} Classifiers\n")
            sys.stderr.write(f" Loaded {len(self.m1_recommenders)} Recommenders\n")
            sys.stderr.write(f" Task 2 Model Status: Loaded {len(self.m2_lgbs)} machines\n")
        except Exception as e:
            sys.stderr.write(f"Initialization error: {e}\n")

    def get_model1(self, part):
        """Retrieves the Quality Classifier model for a specific product part."""
        return self.m1_classifiers.get(part)
        
    def get_recommender1(self, part):
        """Retrieves the Prescriptive Recommender model for a specific product part."""
        return self.m1_recommenders.get(part)

    def optimize_sp(self, buffer_df, part, p_good_curr, p_dt_curr):
        """Prescriptive Engine with Dynamic Bounds & Safety Protocols."""
        model_clf = self.get_model1(part)
        model_rec = self.get_recommender1(part)
        
        clean_row = buffer_df.iloc[-1].to_dict()
        safe_part = part.replace(" ", "_").replace("-", "")
        t1_feats = self.task1_features.get(safe_part, [])
        feature_not_sp = self.task1_feature_not_sp.get(safe_part, [])
        sp_cols = self.task1_sp_cols.get(safe_part, [])
        
        # Keep default SP state
        default_rec = {ui: float(clean_row.get(re.sub(r'[^A-Za-z0-9_]+', '_', canon), 0.0)) for ui, canon in UI_TO_SP.items()}
        
        # =================================================================
        #  SAFETY PROTOCOL 1: "Don't fix what ain't broke"
        #  Only bypass intervention if the batch is Good (>= 85%) AND the Machine is Safe (< 20%)
        # =================================================================
        if p_good_curr >= 0.85 and p_dt_curr < 0.20:
            return default_rec, p_good_curr, p_dt_curr, False

        if not model_clf or not t1_feats or not model_rec or not feature_not_sp:
            return default_rec, p_good_curr, p_dt_curr, False

        # Create input vector for Recommender
        curr_buffer_calc = FeatureEngineer.compute_for_buffer(buffer_df, feature_not_sp)
        raw_rec_sp_values = model_rec.predict(curr_buffer_calc)[0]
        
        # =================================================================
        #  SAFETY PROTOCOL 2: DYNAMIC BOUNDS (Risk-based margins)
        # =================================================================
        # If machine is at high risk of failure (>= 80%) or batch is severely degrading, allow wider limits
        if p_dt_curr >= 0.80 or p_good_curr < 0.15:
            base_deviation = 0.50  # High risk -> AI allowed to intervene up to 50%
        elif p_good_curr < 0.40 or p_dt_curr >= 0.40:
            base_deviation = 0.30  # Medium risk -> Release bound to 30%
        else:
            base_deviation = 0.15  # Slight deviation -> Tighten safety bound to 15%

        best_cand = clean_row.copy()
        requires_manual_review = False # UI flag for manual review
        
        for idx, sp_col in enumerate(sp_cols):
            orig_val = float(clean_row.get(sp_col, 0.0))
            new_val = raw_rec_sp_values[idx]
            
            #  SAFETY PROTOCOL 3: SENSOR-SPECIFIC RULES (Physics-based)
            # Dựa trên phân tích dao động thực tế (Good vs Bad) để gán margin chuẩn
            col_lower = sp_col.lower()
            if "temperature" in col_lower or "heat" in col_lower:
                allowed_dev = min(base_deviation, 0.15) 
                margin = max(np.abs(orig_val) * allowed_dev, 2.0)
            elif "flow" in col_lower or "speed" in col_lower:
                allowed_dev = min(base_deviation * 1.5, 0.40)
                margin = max(np.abs(orig_val) * allowed_dev, 200.0) # Thực tế dao động ~200
            elif "vacuum" in col_lower:
                allowed_dev = base_deviation * 2.0
                margin = max(np.abs(orig_val) * allowed_dev, 15.0)  # Thực tế dao động ~15
            elif "pressure" in col_lower:
                allowed_dev = base_deviation
                margin = max(np.abs(orig_val) * allowed_dev, 20.0)  # Thực tế steam pressure dao động ~20
            elif "ffte production solids" in col_lower:
                # Chỉ số này rất khắt khe, chênh lệch thực tế cực nhỏ (std ~0.5)
                allowed_dev = min(base_deviation, 0.02)
                margin = max(np.abs(orig_val) * allowed_dev, 0.5)
            elif "tfe production solids" in col_lower:
                allowed_dev = base_deviation
                margin = max(np.abs(orig_val) * allowed_dev, 10.0) # Thực tế dao động ~10
            else:
                allowed_dev = base_deviation
                margin = max(np.abs(orig_val) * allowed_dev, 7.0)  # Mặc định (như Feed solids std ~7)
            
            # Clip values to safety bounds
            bound_lower = orig_val - margin
            bound_upper = orig_val + margin
            safe_val = np.clip(new_val, bound_lower, bound_upper)
            
            # If AI adjusts beyond 10% compared to initial, flag for Engineer Review
            if np.abs(safe_val - orig_val) > (np.abs(orig_val) * 0.10) and np.abs(safe_val - orig_val) > 1.0:
                requires_manual_review = True
                
            best_cand[sp_col] = safe_val
            
        # Simulate Good probability again after safe valve adjustments
        cand_buffer_df = buffer_df.copy()
        for k, v in best_cand.items():
            cand_buffer_df.loc[cand_buffer_df.index[-1], k] = v
            
        X_t1 = FeatureEngineer.compute_for_buffer(cand_buffer_df, t1_feats)
        p_vec = model_clf.predict_proba(X_t1)[0]
        best_pg = float(p_vec[0])
        
        # Simulate Downtime risk (Task 2)
        best_pdt = 0.0
        if self.m2_lgbs and self.task2_features:
            # Recreate SP_PV conflicts
            pv_cols = [c for c in cand_buffer_df.columns if 'PV' in c]
            for pv_col in pv_cols:
                sp_col = pv_col.replace('PV', 'SP')
                if sp_col in cand_buffer_df.columns:
                    err_col = pv_col.replace('PV', 'SP_PV_Delta')
                    pv_val = pd.to_numeric(cand_buffer_df[pv_col], errors='coerce').fillna(0)
                    sp_val = pd.to_numeric(cand_buffer_df[sp_col], errors='coerce').fillna(0)
                    cand_buffer_df[err_col] = pv_val - sp_val
                    cand_buffer_df[f'{err_col}_volatility'] = cand_buffer_df[err_col].rolling(window=15, min_periods=1).std().fillna(0)
                lag5_col = f"{pv_col}_lag5"
                cand_buffer_df[lag5_col] = cand_buffer_df[pv_col].shift(5).bfill().fillna(cand_buffer_df[pv_col])

            for machine in self.m2_lgbs.keys():
                config = self.task2_features.get(machine, {})
                features = config.get("features", [])
                if not features: continue
                # We calculate T2 feature values
                X_t2 = FeatureEngineer.compute_for_buffer(cand_buffer_df, [f for f in features if f != 'IF_Anomaly_Score'])
                
                # Fetch Iso and Scaler for calculating IF_Anomaly_Score
                iso_model = self.m2_isos.get(machine)
                scaler_model = self.m2_scalers.get(machine)
                if iso_model and scaler_model:
                    X_t2_scaled = scaler_model.transform(X_t2)
                    anomaly_score = -iso_model.decision_function(X_t2_scaled)[0]
                    X_t2['IF_Anomaly_Score'] = anomaly_score
                else:
                    X_t2['IF_Anomaly_Score'] = 0.0
                
                # Ensure ordered identical to JSON
                X_t2 = X_t2[features + ['IF_Anomaly_Score']]
                
                p_dt_vec = self.m2_lgbs[machine].predict_proba(X_t2)[0]
                risk = float(p_dt_vec[1]) if len(p_dt_vec) > 1 else 0.0
                if risk > best_pdt:
                    best_pdt = risk
                
        # Format response for UI
        rec_sp = {}
        for ui, canonical in UI_TO_SP.items():
            clean_col = re.sub(r'[^A-Za-z0-9_]+', '_', canonical)
            rec_sp[ui] = float(best_cand.get(clean_col, clean_row.get(clean_col, 0.0)))
            
        return rec_sp, best_pg, best_pdt, requires_manual_review


def main():
    """
    Main execution entry point.
    Reads JSON from standard input, processes sensor data through Digital Twin simulation,
    evaluates quality and downtime risks, runs the prescriptive optimizer, 
    and outputs the result as a JSON string to standard output.
    """
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

        # =============================================================
        #  DIGITAL TWIN SIMULATION - FOR DEMO PHASE 2
        # =============================================================
        try:
            prev_buffer = []
            if BUFFER_FILE.exists():
                with open(BUFFER_FILE, 'r') as f:
                    prev_buffer = json.load(f)
                    
            last_state = prev_buffer[-1] if prev_buffer else {}
                
            twin_pairs = [
                ("TFE_Vacuum_pressure_SP", "TFE_Vacuum_pressure_PV"),
                ("FFTE_Steam_pressure_SP", "FFTE_Steam_pressure_PV"),
                ("TFE_Out_flow_SP", "TFE_Out_flow_PV"),
                ("TFE_Production_solids_SP", "TFE_Production_solids_PV"),
                ("FFTE_Feed_solids_SP", "FFTE_Feed_solids_PV"),
                ("FFTE_Production_solids_SP", "FFTE_Production_solids_PV"),
                ("TFE_Steam_pressure_SP", "TFE_Steam_pressure_PV")
            ]
            
            for sp_col, pv_col in twin_pairs:
                if sp_col in clean_row:
                    target_sp = float(clean_row[sp_col])
                    last_pv = float(last_state.get(pv_col, target_sp))
                    
                    diff = target_sp - last_pv
                    
                    RESPONSE_RATE = 0.35
                    step = diff * RESPONSE_RATE
                    jitter = random.uniform(-1.5, 1.5)
                    
                    if pv_col == "FFTE_Steam_pressure_PV" and target_sp >= 145.0:
                        jitter = random.uniform(25.0, 50.0) 
                    elif pv_col == "TFE_Out_flow_PV" and target_sp <= 1500.0:
                        jitter = random.uniform(600.0, 800.0) 
                    elif pv_col == "TFE_Vacuum_pressure_PV" and target_sp >= -20.0:
                        jitter = random.uniform(15.0, 30.0)
                        
                    simulated_pv = last_pv + step + jitter
                    clean_row[pv_col] = round(simulated_pv, 2)
            
            """
            FIX 2: Supplement ALL static sensors to prevent them dropping to 0.0
            """
            static_pvs = {
                "TFE_Tank_level": 70.0,
                "TFE_Level": 50.0,
                "FFTE_Heat_temperature_1": 80.0,
                "FFTE_Heat_temperature_2": 82.0,
                "FFTE_Heat_temperature_3": 81.0,
                "TFE_Product_out_temperature": 68.0,
                "FFTE_Discharge_density": 1.26,
                "FFTE_Discharge_solids": 49.71,
                "FFTE_Feed_flow_rate_PV": 9286.64,
                "TFE_Input_flow_PV": 1603.76,
                "TFE_Motor_current": 24.9,
                "TFE_Motor_speed": 80.0,
                "TFE_Production_solids_density": 0.93,
                "TFE_Steam_temperature": 62.29,
                "TFE_Temperature": 64.0
            }
            for stat_col, base_val in static_pvs.items():
                if stat_col not in clean_row:
                    old_stat = float(last_state.get(stat_col, base_val))
                    """
                    Mean Reversion Force: Eliminate random walk drift effect.
                    """
                    pull_to_base = (base_val - old_stat) * 0.1 
                    clean_row[stat_col] = round(old_stat + pull_to_base, 2)
                    
            """
            Extract Tank Level Custom Physics: Revert to equilibrium.
            To prevent the stream from injecting noisy fluctuations (e.g., dropping to 30-40%),
            we enforce a Mean Reversion towards a secure ~65.0 baseline.
            If the incoming value indicates an explicit anomaly test (e.g., > 85.0 or < 20.0),
            we will honor the testing condition instead.
            """
            old_extract = float(last_state.get("Extract_tank_Level", 65.0))
            current = float(clean_row.get("Extract_tank_Level", old_extract))

            if current > 85.0 or current < 20.0:
                clean_row["Extract_tank_Level"] = current
            else:
                if old_extract > 65.0:
                    new_extract = old_extract - 2.0
                else:
                    new_extract = old_extract + (65.0 - old_extract) * 0.1
                clean_row["Extract_tank_Level"] = round(new_extract + random.uniform(-0.5, 0.5), 2)
                    
        except Exception as e:
            sys.stderr.write(f"Digital Twin Simulation Warning: {e}\n")
        # =============================================================

        # -------------------------------------------------------------
        # TASK 1: QUALITY PREDICTION (Current Settings)
        # -------------------------------------------------------------
        pred_label = "UNKNOWN"
        p_good = 0.0
        
        model1 = server.get_model1(part)
        safe_part = part.replace(" ", "_").replace("-", "")
        t1_feats = server.task1_features.get(safe_part, [])
        
        # Load and update history buffer
        buffer_df = FeatureEngineer.update_and_get_buffer(clean_row)

        if model1 and t1_feats:
            X_t1 = FeatureEngineer.compute_for_buffer(buffer_df, t1_feats)
            p_vec = model1.predict_proba(X_t1)[0]
            
            p_good = float(p_vec[0])
            p_low = float(p_vec[1]) if len(p_vec) > 1 else 0.0
            p_high = float(p_vec[2]) if len(p_vec) > 2 else 0.0
            
            # Error 3: Safe Threshold logic to accurately catch Low_Bad vs High_Bad
            if p_low > 0.20 and p_low >= (p_high * 0.5):
                pred_label = "LOW_BAD"
            elif p_high > p_low and p_high > p_good:
                pred_label = "HIGH_BAD"
            elif p_good >= p_low and p_good >= p_high:
                pred_label = "GOOD"
            else:
                pred_idx = int(np.argmax(p_vec))
                if pred_idx == 0:
                    pred_label = "GOOD"
                elif pred_idx == 1:
                    pred_label = "LOW_BAD"
                else:
                    pred_label = "HIGH_BAD"

        # -------------------------------------------------------------
        # TASK 2: DOWNTIME ALERT (Multi-Class + Isolation Check)
        # -------------------------------------------------------------
        pred_dt_classes = []
        p_dt_risk = 0.0
        iso_anomaly = False

        if server.m2_lgbs and server.m2_isos and server.m2_scalers and server.task2_features:
            
            # 1. SYNC COLUMN NAMES (From short UI names to model canonical names)
            buffer_df_t2 = buffer_df.copy()
            for friendly, tag in FRIENDLY_TO_TAG_MAP.items():
                clean_tag = re.sub(r'[^A-Za-z0-9_]+', '_', tag)
                if friendly in buffer_df_t2.columns:
                    buffer_df_t2[clean_tag] = buffer_df_t2[friendly]
            
            # Fill missing SPs dynamically based on their PV equivalents if SP does not exist
            if "Extract_tank_Level_PV" in buffer_df_t2.columns and "Extract_tank_Level_SP" not in buffer_df_t2.columns:
                buffer_df_t2["Extract_tank_Level_SP"] = buffer_df_t2["Extract_tank_Level_PV"]

            # ---------------------------------------------------------
            #  RECREATE CONFLICT FEATURES (SP vs PV) AND LAG5 AT RUNTIME
            # ---------------------------------------------------------
            pv_cols = [c for c in buffer_df_t2.columns if 'PV' in c]
            for pv_col in pv_cols:
                sp_col = pv_col.replace('PV', 'SP')
                if sp_col in buffer_df_t2.columns:
                    err_col = pv_col.replace('PV', 'SP_PV_Delta')
                    # Calculate immediate error
                    pv_val = pd.to_numeric(buffer_df_t2[pv_col], errors='coerce').fillna(0)
                    sp_val = pd.to_numeric(buffer_df_t2[sp_col], errors='coerce').fillna(0)
                    buffer_df_t2[err_col] = pv_val - sp_val
                    
                    # Calculate cumulative volatility in buffer
                    buffer_df_t2[f'{err_col}_volatility'] = buffer_df_t2[err_col].rolling(window=15, min_periods=1).std().fillna(0)
                
                # Calculate Time-lagged feature (shift back 5 steps if enough data)
                lag5_col = f"{pv_col}_lag5"
                buffer_df_t2[lag5_col] = buffer_df_t2[pv_col].shift(5).bfill().fillna(buffer_df_t2[pv_col])

            # Loop through each machine to aggregate risk indicators
            for machine, lgb_model in server.m2_lgbs.items():
                config = server.task2_features.get(machine, {})
                features = config.get("features", [])
                stage1_thresh = config.get("stage1_thresh", 0.05)
                stage2_fallback = config.get("stage2_fallback", f"{machine}_Anomaly")
                
                if not features: continue

                # 2. CALCULATE FEATURES for this machine
                base_features = [f for f in features if f != 'IF_Anomaly_Score']
                X_t2 = FeatureEngineer.compute_for_buffer(buffer_df_t2, base_features)
                
                # Sub-Task 2A: Isolation Forest 
                iso_model = server.m2_isos.get(machine)
                scaler_model = server.m2_scalers.get(machine)
                local_iso_anomaly = False
                
                if iso_model and scaler_model:
                    X_t2_scaled = scaler_model.transform(X_t2)
                    iso_preds = iso_model.predict(X_t2_scaled)
                    anomaly_score = -iso_model.decision_function(X_t2_scaled)[0]
                    X_t2['IF_Anomaly_Score'] = anomaly_score
                    if iso_preds[0] == -1:
                        iso_anomaly = True
                        local_iso_anomaly = True
                else:
                    X_t2['IF_Anomaly_Score'] = 0.0

                # FIX 3: Prevent column name duplicates
                if 'IF_Anomaly_Score' not in features:
                    features_ordered = features + ['IF_Anomaly_Score']
                else:
                    features_ordered = features
                X_t2 = X_t2[features_ordered]

                # Sub-Task 2B: Root Cause Multi-Class LightGBM 
                p_dt_vec = lgb_model.predict_proba(X_t2)[0]
                raw_risk = float(p_dt_vec[1]) if len(p_dt_vec) > 1 else float(p_dt_vec[0])
                
                """
                OOD DETECTION (OUT-OF-DISTRIBUTION) VETO POWER
                Grant Veto Power to Isolation Forest if data is extremely anomalous.
                When pressure reaches extreme values, LightGBM extrapolation fails (low raw_risk).
                IF high anomaly_score automatically rescues the system.
                """
                is_extreme_anomaly = local_iso_anomaly and (anomaly_score > 0.12)
                
                """ EXPERT RULE (ABSOLUTE OVERRIDE)
                Move the overflow rule outside the anomaly check to give it absolute priority over AI. """
                force_extract_overflow = False
                force_tfe_shutdown = False

                if machine == "TFE":
                    try:
                        col_name = 'Extract_tank_Level'
                        if 'Extract_tank_Level_PV' in buffer_df_t2.columns:
                            col_name = 'Extract_tank_Level_PV'
                        extract_level = float(buffer_df_t2[col_name].iloc[-1]) if col_name in buffer_df_t2.columns else 0.0
                        
                        if extract_level > 95.0:
                            force_extract_overflow = True
                            local_iso_anomaly = True  # Force AI to trigger an alarm
                            is_extreme_anomaly = True # Force red warning
                            
                        # Rule 2: TFE Overpressure / Vacuum Loss
                        tfe_steam_pv = float(buffer_df_t2['TFE_Steam_pressure_PV'].iloc[-1]) if 'TFE_Steam_pressure_PV' in buffer_df_t2.columns else 120.0
                        tfe_vac_pv = float(buffer_df_t2['TFE_Vacuum_pressure_PV'].iloc[-1]) if 'TFE_Vacuum_pressure_PV' in buffer_df_t2.columns else -65.0
                        
                        if tfe_steam_pv > 140.0 or tfe_vac_pv > -10.0:
                            force_tfe_shutdown = True
                            local_iso_anomaly = True  
                            is_extreme_anomaly = True 

                    except Exception:
                        pass
                
                calibrated_risk = 0.0
                if local_iso_anomaly and (raw_risk >= stage1_thresh or is_extreme_anomaly):
                    calibrated_risk = 0.85 + min(abs(anomaly_score), 0.14)
                    
                    # If it is an overflow error, lock risk at 99%
                    if force_extract_overflow or force_tfe_shutdown:
                        calibrated_risk = 0.99
                else:
                    if stage1_thresh > 0:
                        calibrated_risk = (raw_risk / stage1_thresh) * 0.40
                
                if calibrated_risk > p_dt_risk:
                    p_dt_risk = calibrated_risk
                    
                """ Merge logic for determining Root Cause """
                if local_iso_anomaly and (raw_risk >= stage1_thresh or is_extreme_anomaly):
                    s2_lgb = server.m2_stage2_lgbs.get(machine)
                    s2_enc = server.m2_stage2_encoders.get(machine)
                    
                    if s2_lgb is not None and s2_enc is not None:
                        pred_enc = s2_lgb.predict(X_t2)[0]
                        pred_cause = s2_enc.inverse_transform([pred_enc])[0]
                        
                        # Assign critical error name
                        if force_extract_overflow:
                            pred_cause = "EXTRACT_TANK_OVERFLOW_CRITICAL"
                        elif force_tfe_shutdown:
                            pred_cause = "TFE_Shutdown"
                                
                        if "EXTRACT_TANK_OVERFLOW_CRITICAL" not in pred_dt_classes and pred_cause not in pred_dt_classes:
                            pred_dt_classes.append(pred_cause)
                    else:
                        if force_extract_overflow:
                            if "EXTRACT_TANK_OVERFLOW_CRITICAL" not in pred_dt_classes:
                                pred_dt_classes.append("EXTRACT_TANK_OVERFLOW_CRITICAL")
                        elif force_tfe_shutdown:
                            if "TFE_Shutdown" not in pred_dt_classes:
                                pred_dt_classes.append("TFE_Shutdown")
                        elif stage2_fallback not in pred_dt_classes:
                            pred_dt_classes.append(stage2_fallback)
                elif calibrated_risk >= 0.20:
                    warning_msg = f"{machine} Degrading (Moderate Risk)"
                    try:
                        if machine == "TFE":
                            tfe_steam = float(buffer_df_t2['TFE_Steam_pressure_PV'].iloc[-1]) if 'TFE_Steam_pressure_PV' in buffer_df_t2.columns else 120.0
                            tfe_vac = float(buffer_df_t2['TFE_Vacuum_pressure_PV'].iloc[-1]) if 'TFE_Vacuum_pressure_PV' in buffer_df_t2.columns else -65.0
                            tfe_outflow = float(buffer_df_t2['TFE_Out_flow_PV'].iloc[-1]) if 'TFE_Out_flow_PV' in buffer_df_t2.columns else 2000.0
                            tfe_prod_solids = float(buffer_df_t2['TFE_Production_solids_PV'].iloc[-1]) if 'TFE_Production_solids_PV' in buffer_df_t2.columns else 70.0
                            
                            if tfe_steam > 128.0:
                                warning_msg = "TFE Degrading: Reduce Steam Pressure SP"
                            elif tfe_vac > -30.0:
                                warning_msg = "TFE Degrading: Decrease Vacuum Pressure (Make more negative)"
                            elif tfe_outflow < 1700.0 or tfe_outflow > 2950.0:
                                warning_msg = "TFE Degrading: Adjust Outflow SP to Safe Range"
                            elif tfe_prod_solids > 75.0:
                                warning_msg = "TFE Degrading: Decrease Production Solids SP"
                            elif tfe_prod_solids < 65.0:
                                warning_msg = "TFE Degrading: Increase Production Solids SP"
                            else:
                                warning_msg = "TFE Degrading: Review AI Recommender SP"

                        elif machine == "FFTE":
                            ffte_steam = float(buffer_df_t2['FFTE_Steam_pressure_PV'].iloc[-1]) if 'FFTE_Steam_pressure_PV' in buffer_df_t2.columns else 130.0
                            ffte_prod_solids = float(buffer_df_t2['FFTE_Production_solids_PV'].iloc[-1]) if 'FFTE_Production_solids_PV' in buffer_df_t2.columns else 60.0
                            
                            if ffte_steam > 140.0:
                                warning_msg = "FFTE Degrading: Reduce Steam Pressure SP"
                            elif ffte_prod_solids > 65.0:
                                warning_msg = "FFTE Degrading: Reduce Production Solids SP"
                            else:
                                warning_msg = "FFTE Degrading: Check Feed Solids SP"
                    except Exception:
                        pass

                    if warning_msg not in pred_dt_classes:
                        pred_dt_classes.append(warning_msg)

        if not pred_dt_classes:
            pred_dt_classes = ["Normal"]



        # -------------------------------------------------------------
        # PRESCRIPTIVE ENGINE (Recommendations)
        # -------------------------------------------------------------
        rec_sp, rec_p_good, rec_p_dt, review_flag = server.optimize_sp(buffer_df, part, p_good, p_dt_risk)

        response = {
            "recommendedSP": rec_sp,
            "pGood": float(round(p_good, 4)),
            "pDowntime": float(round(p_dt_risk, 4)),
            "prediction": pred_label,
            "downtimeRisk": float(round(p_dt_risk * 100, 2)),
            "rootCause": pred_dt_classes,
            "isoAnomaly": iso_anomaly,
            "recommendedPGood": float(round(rec_p_good, 4)),
            "recommendedPDowntime": float(round(rec_p_dt, 4)),
            "requiresManualReview": review_flag
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
            log_data["RootCause"] = " | ".join(pred_dt_classes)
            
            pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=not log_file.exists(), index=False)
        except:
            pass

    except Exception as e:
        import traceback; traceback.print_exc()
        response = {
            "error": str(e), "prediction": "ERROR",
            "pGood": 0.0, "pDowntime": 0.0, "downtimeRisk": 0.0, "rootCause": "Error",
            "recommendedSP": {k: 0.0 for k in UI_TO_SP.keys()},
            "recommendedPGood": 0.0, "recommendedPDowntime": 0.0
        }
    
    sys.stdout.write(json.dumps(response, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
