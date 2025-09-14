# # farmapp/ml_utils.py
# import pandas as pd
# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import joblib
# import os

# CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'daily_record.csv')  # adjust

# # columns in your CSV; adapt if different
# FEATURE_COLS = [
#     'breed','age_years','parity','lactation_stage','weight_kg','feed_kg',
#     'walking_km','rumination_min','temp_c','humidity'
# ]
# CLASS_TARGET = 'disease_label'
# REG_TARGET = 'milk_liters'

# CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml_cache')
# os.makedirs(CACHE_DIR, exist_ok=True)
# CLASS_MODEL_FILE = os.path.join(CACHE_DIR, 'clf.joblib')
# REG_MODEL_FILE = os.path.join(CACHE_DIR, 'reg.joblib')
# PREPROC_FILE = os.path.join(CACHE_DIR, 'preproc.joblib')

# def load_data():
#     df = pd.read_csv(CSV_PATH, parse_dates=['date'], dayfirst=True)
#     return df

# def build_preprocessor(df):
#     # auto-detect categorical vs numeric
#     cat_cols = [c for c in FEATURE_COLS if df[c].dtype == object or c in ['breed','lactation']]
#     num_cols = [c for c in FEATURE_COLS if c not in cat_cols]
#     num_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler())
#     ])
#     cat_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
#     ])
#     preproc = ColumnTransformer([
#         ('num', num_pipeline, num_cols),
#         ('cat', cat_pipeline, cat_cols)
#     ])
#     return preproc, num_cols, cat_cols

# def train_models(force_retrain=False):
#     # trains classifier and regressor from CSV, caches them
#     if not force_retrain and os.path.exists(CLASS_MODEL_FILE) and os.path.exists(REG_MODEL_FILE) and os.path.exists(PREPROC_FILE):
#         clf = joblib.load(CLASS_MODEL_FILE)
#         reg = joblib.load(REG_MODEL_FILE)
#         preproc = joblib.load(PREPROC_FILE)
#         return preproc, clf, reg

#     df = load_data()
#     # drop rows with missing targets for respective tasks
#     df_clf = df.dropna(subset=[CLASS_TARGET])
#     df_reg = df.dropna(subset=[REG_TARGET])

#     preproc, num_cols, cat_cols = build_preprocessor(df)
#     X = df[FEATURE_COLS]
#     X_pre = preproc.fit_transform(X)

#     # classifier
#     y_clf = df_clf[CLASS_TARGET]
#     Xc = preproc.transform(df_clf[FEATURE_COLS])
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(Xc, y_clf)

#     # regressor
#     yr = df_reg[REG_TARGET]
#     Xr = preproc.transform(df_reg[FEATURE_COLS])
#     reg = RandomForestRegressor(n_estimators=100, random_state=42)
#     reg.fit(Xr, yr)

#     joblib.dump(clf, CLASS_MODEL_FILE)
#     joblib.dump(reg, REG_MODEL_FILE)
#     joblib.dump(preproc, PREPROC_FILE)
#     return preproc, clf, reg

# def predict_from_input(input_dict):
#     """
#     input_dict: dict of FEATURE_COLS keys to values (strings/numbers)
#     returns: {'disease_label': pred_label, 'milk_liters': pred_float}
#     """
#     preproc, clf, reg = train_models()
#     df = pd.DataFrame([input_dict], columns=FEATURE_COLS)
#     Xp = preproc.transform(df)
#     pred_label = clf.predict(Xp)[0]
#     pred_proba = clf.predict_proba(Xp).max()  # optional confidence
#     pred_milk = float(reg.predict(Xp)[0])
#     return {'disease_label': int(pred_label), 'disease_conf': float(pred_proba), 'milk_liters': pred_milk}






# farmapp/ml_utils.py
import os
from django.conf import settings
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Project data path (BASE_DIR is from settings.py)
BASE_DIR = settings.BASE_DIR
CSV_PATH = os.path.join(BASE_DIR, 'data', 'daily_record.csv')

# Features (ensure these match your CSV header exactly)
FEATURE_COLS = [
    'breed', 'age_years', 'parity', 'lactation_stage', 'weight_kg', 'feed_kg',
    'walking_km', 'rumination_min', 'temp_c', 'humidity'
]
CLASS_TARGET = 'disease_label'
REG_TARGET = 'milk_liters'

# cache dir for models
CACHE_DIR = os.path.join(BASE_DIR, 'ml_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
CLASS_MODEL_FILE = os.path.join(CACHE_DIR, 'clf.joblib')
REG_MODEL_FILE = os.path.join(CACHE_DIR, 'reg.joblib')
PREPROC_FILE = os.path.join(CACHE_DIR, 'preproc.joblib')

def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Put daily_record.csv there.")
    # try to parse date if present, dayfirst True because your CSV uses dd-mm-yyyy
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=['date'], dayfirst=True)
    except Exception:
        df = pd.read_csv(CSV_PATH)
    return df

def build_preprocessor(df):
    # decide categorical vs numeric based on dtype and known categorical names
    cat_guards = {'breed', 'lactation_stage'}
    cat_cols = [c for c in FEATURE_COLS if (c in cat_guards) or df.get(c, pd.Series()).dtype == object]
    num_cols = [c for c in FEATURE_COLS if c not in cat_cols]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Backwards/forwards compatible OneHotEncoder init
    try:
        # older sklearn supports 'sparse' keyword
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        # newer sklearn (1.2+) uses sparse_output
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe)
    ])

    # sparse_threshold=0 forces dense output (so preproc.transform returns ndarray)
    preproc = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop',
        sparse_threshold=0
    )
    return preproc, num_cols, cat_cols

def train_models(force_retrain=False):
    # load from cache if available and not forcing retrain
    if (not force_retrain) and os.path.exists(CLASS_MODEL_FILE) and os.path.exists(REG_MODEL_FILE) and os.path.exists(PREPROC_FILE):
        clf = joblib.load(CLASS_MODEL_FILE)
        reg = joblib.load(REG_MODEL_FILE)
        preproc = joblib.load(PREPROC_FILE)
        return preproc, clf, reg

    df = load_data()

    # defensive: ensure FEATURE_COLS exist in df
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in CSV: {missing}")

    preproc, num_cols, cat_cols = build_preprocessor(df)

    X_all = df[FEATURE_COLS]
    X_pre = preproc.fit_transform(X_all)  # fit once on all data (pipeline includes imputers)

    # classifier: drop rows with missing class target
    if CLASS_TARGET in df.columns:
        df_clf = df.dropna(subset=[CLASS_TARGET])
        if not df_clf.empty:
            Xc = preproc.transform(df_clf[FEATURE_COLS])
            y_clf = df_clf[CLASS_TARGET]
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(Xc, y_clf)
        else:
            clf = RandomForestClassifier(n_estimators=10, random_state=42)  # dummy small model
    else:
        clf = RandomForestClassifier(n_estimators=10, random_state=42)

    # regressor: drop rows with missing reg target
    if REG_TARGET in df.columns:
        df_reg = df.dropna(subset=[REG_TARGET])
        if not df_reg.empty:
            Xr = preproc.transform(df_reg[FEATURE_COLS])
            yr = df_reg[REG_TARGET]
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(Xr, yr)
        else:
            reg = RandomForestRegressor(n_estimators=10, random_state=42)
    else:
        reg = RandomForestRegressor(n_estimators=10, random_state=42)

    joblib.dump(preproc, PREPROC_FILE)
    joblib.dump(clf, CLASS_MODEL_FILE)
    joblib.dump(reg, REG_MODEL_FILE)

    return preproc, clf, reg

def predict_from_input(input_dict):
    """
    input_dict must contain keys exactly matching FEATURE_COLS
    returns: {'disease_label': int, 'disease_conf': float, 'milk_liters': float}
    """
    # ensure keys present
    missing = [k for k in FEATURE_COLS if k not in input_dict]
    if missing:
        raise KeyError(f"Missing keys for prediction: {missing}")

    preproc, clf, reg = train_models()

    df = pd.DataFrame([input_dict], columns=FEATURE_COLS)
    Xp = preproc.transform(df)  # returns 2D ndarray
    pred_label = int(clf.predict(Xp)[0])
    # try to get confidence, else 0.0
    try:
        if hasattr(clf, "predict_proba"):
            pred_proba = float(clf.predict_proba(Xp).max())
        else:
            pred_proba = 0.0
    except Exception:
        pred_proba = 0.0

    pred_milk = float(reg.predict(Xp)[0])
    return {'disease_label': pred_label, 'disease_conf': pred_proba, 'milk_liters': pred_milk}

