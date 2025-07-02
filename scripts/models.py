import os
import numpy as np
import pandas as pd
import scipy.io
# preprocessing, models, pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# regression & classification models
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# model evaluation
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, balanced_accuracy_score
)
# cross-validation & search
from sklearn.model_selection import (
    LeaveOneGroupOut, GroupShuffleSplit,
    GridSearchCV
)

# # --------- functions ----------
def load_AMIGOS_data(path):
    """ Loops through directory with AMIGOS data and load .mat files.
    Args: 
        Parth, excluded participants
    Returns: 
        mats, tuple.
    """
    mats = []
    # loop over all possible participant IDs
    for i in range(40): 
        pid = i + 1
        # construct expected filename
        filename = f"Data_Preprocessed_P{'0' if i < 9 else ''}{pid}"
        src = os.path.join(path, filename, f"{filename}.mat")
        if os.path.exists(src):
            # load .mat and append tuple
            mats.append((pid, scipy.io.loadmat(src)))
        else:
            print(f"Warning: File not found - {src}")
    return mats

def exp_condition(df, session_df):
    """ Merge in session type and add flag for if person watched the long videos alone or with people.
    Args: main long df and df with session type (metadata). In the metadata, there is a column called 'Session_Type_Exp_2' that tracks 'Alone' or 'Group'
    Returns: df with new column that holds a binary indicator for 1 (alone) or 0 (group)
    """
    # ensure the merge keys share type
    df = df.copy()
    df['ParticipantID'] = df['ParticipantID'].astype(str)
    session_df = session_df.copy()
    session_df['UserID'] = session_df['UserID'].astype(str)

    # extract and rename
    flags = session_df[['UserID', 'Session_Type_Exp_2']].copy()
    flags.rename(columns={'UserID': 'ParticipantID'}, inplace=True)
    flags['Alone_long'] = (flags['Session_Type_Exp_2'] == 'Alone').astype(int)

    # now merge on ParticipantID (both strings)
    out = df.merge(flags[['ParticipantID', 'Alone_long']],
                   on='ParticipantID', how='left')
    return out

# ################## prepare for analysis ##################
def process_labels_amigos(mats,
                            long_vid_codes=None,
                            short_only_pids=None):
    """ Extract the labels from the mats variable
    Return df of labels with ParticipantID / VideoID keys.
    """
    
    label_names = ["arousal", "valence"]  
                 
    long_vid_codes  = long_vid_codes or {"N1", "P1", "B1", "U1"}
    short_only_pids = short_only_pids or {8, 24, 32}

    rows = [] # collect extracted rows

    for pid, mat in mats:
        # get video IDs and self-assessment label matrix
        vids   = mat.get("VideoIDs")
        labels = mat.get("labels_selfassessment")

        # loop over videos for each participant
        for idx in range(vids.shape[1]): 
            vid_raw = np.squeeze(vids[0, idx]) # extract and flatten arrays
            if isinstance(vid_raw, np.ndarray):
                vid_raw = vid_raw.item()
            if isinstance(vid_raw, (bytes, np.bytes_)):
                vid_raw = vid_raw.decode()

            if pid in short_only_pids and str(vid_raw) in long_vid_codes:
                continue

            try:
                vid_id = str(int(vid_raw))
            except (ValueError, TypeError):
                vid_id = str(vid_raw).strip()

            lblvec = np.squeeze(labels[0, idx])
            if lblvec.size != 12:
                continue

            row = {"ParticipantID": str(pid),
                   "VideoID":      vid_id}
            row.update({name: float(lblvec[i])
                        for i, name in enumerate(label_names)})
            rows.append(row)
    return pd.DataFrame(rows)

def process_labels_phy(path: str) -> pd.DataFrame:
    """ Reads path to labels, select relevant rows, extract IDs & apply z-score normalization.
    Args:
       path to PhyMER label file
    Returns:
        pd.DataFrame with columns (among others) "ParticipantID", "VideoID", "Arousal_z", "Valence_z"
    """
    labels = pd.read_csv(path)

    # Keep only video rows
    vid_labels = labels[labels["experiment_code"].str.contains("VID")].copy()

    # Extract numeric IDs
    vid_labels["ParticipantID"] = (
        vid_labels["experiment_code"]
        .str.extract(r"SUB0*(\d+)", expand=False)
        .astype(int)
    )
    vid_labels["VideoID"] = (
        vid_labels["experiment_code"]
        .str.extract(r"VID0*(\d+)", expand=False)
        .astype(int)
    )

    # Keep relevant columns
    vid_labels = vid_labels[["ParticipantID", "VideoID", "arousal", "valence"]]
    
    return vid_labels

def prepare_dataset(merged_features, labels_df):
    """ Prepares dataset by merging features and labels by VideoID and ParticipantID
    Return X, y_arousal, y_valence, groups arrays and videoIDs
    """
    # cast keys to string for merge
    for col in ['ParticipantID', 'VideoID']:
        merged_features[col] = merged_features[col].astype(str)
        labels_df[col] = labels_df[col].astype(str)
    # filter labels to those VideoIDs present
    vid_ids = merged_features['VideoID'].unique()
    labels = labels_df[labels_df['VideoID'].isin(vid_ids)].copy()
    
    # Merge features with labels to make sure labels align
    merged = merged_features.merge(
        labels[['ParticipantID','VideoID','arousal','valence']], # merge with non-zscaled features for 
        on=['ParticipantID','VideoID'], how='inner'
    )
    # prepare X by dropping non-feature columns (dropping PC2 because this is only introducing noise)
    X = merged.drop(
        columns=[
            'ParticipantID','VideoID',
            'Cluster_PC2',
            'arousal','valence' 
        ],
        errors='ignore'
    ).values

    # prepare targets and group labels
    y_arousal = merged['arousal'].values
    y_valence = merged['valence'].values 
    groups = merged['ParticipantID'].values
    
    return X, y_arousal, y_valence, groups, merged, merged["VideoID"].values

def evaluate_models(
    datasets,
    models, *,
    split_kwargs=None,
    fixed_params = None 
    ):
    """ Wrapper function to run the experiment over multiple (dataset, model) combinations & collect results
    Returns results
    """
    results = {}
    for dname, data in datasets.items():
        for mname, model_fn in models.items():
            print(f"{mname} on {dname} …")
            
            # Pass fixed_params if available for this model-dataset combo
            extra_args = {}
            # if fixed_params contains the exact model,dataset pair
            if fixed_params is not None:
                combo_key = (mname, dname)
                if combo_key in fixed_params:
                    extra_args["fixed_params"] = fixed_params[combo_key]

            metrics, final_model = model_fn(
                X=data["X"],
                y=data["y"],
                groups=data["groups"],
                video_ids=data["video_ids"],
                split_kwargs=split_kwargs,
                **extra_args  # Pass fixed params here
            )
            results[(mname, dname)] = {
                "metrics": metrics,
                "model":   final_model
            }
    return results


############## models ################

# seperate tuning and evaluation:
# seperate test + training set. test set = 1 record per person.
# LOPO on training set. find best parameters based on mean peformance.
# with established HP, we may want to retrain the final model on the whole train/validation set 
# test that model on the test set.

def split_groups(X, y, groups, *,
                      test_size=0.20,
                      random_state=42,
                      explicit_test_groups=None):
    """
    Returns train_idx, test_idx with entire groups in one side.
    If `explicit_test_groups` is given you hard-code which
    participants belong to the test set.
    """
    if explicit_test_groups is not None:
        test_mask  = np.isin(groups, explicit_test_groups)
        train_idx  = np.flatnonzero(~test_mask)
        test_idx   = np.flatnonzero(test_mask)
    else:
        gss = GroupShuffleSplit(test_size=test_size,
                                n_splits=1,
                                random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
    return train_idx, test_idx

def lopo_tune(X_tr, y_tr, groups_tr, pipeline, param_grid,
              scoring, n_jobs=-1):
    """
    One GridSearchCV with Leave-One-Group-Out on *training* data only.
    Returns the fitted search object.
    """
    logo   = LeaveOneGroupOut()
    search = GridSearchCV(pipeline,
                          param_grid,
                          cv=logo,
                          scoring=scoring,
                          n_jobs=n_jobs,
                          refit=True)        # refit on full train set. basically retrain the model with best hyper parameters
    search.fit(X_tr, y_tr, groups=groups_tr)
    return search          # best_estimator_ lives inside

def run_experiment(X, y, groups, *,
                   pipeline, param_grid, scoring,
                   is_classification,
                   split_kwargs=None,
                   fixed_params=None,
                   video_ids=None):  

    split_kwargs = split_kwargs or {}
    tr_idx, te_idx = split_groups(X, y, groups, **split_kwargs) # split groups CHANGE THIS BACK TO SPLIT_GROUPS! 

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr, g_te = groups[tr_idx], groups[te_idx]

    # slice test video IDs so we can reuse later
    v_te = video_ids[te_idx] if video_ids is not None else None

    # Fit model
    if fixed_params is not None:
        pipeline.set_params(**fixed_params)
        pipeline.fit(X_tr, y_tr)
        best = pipeline
        y_hat = best.predict(X_te)
        search_results = {"best_params": fixed_params, "cv_results": None}
    else:
        search = lopo_tune(X_tr, y_tr, g_tr, pipeline, param_grid, scoring)
        best = search.best_estimator_
        y_hat = best.predict(X_te)
        search_results = {
            "best_params": search.best_params_,
            "cv_results": search.cv_results_
        }

    #  Classification or regression output
    if is_classification:
        acc = accuracy_score(y_te, y_hat)
        f1 = f1_score(y_te, y_hat)
        precision = precision_score(y_te, y_hat)
        recall = recall_score(y_te, y_hat)
        balanced_acc = balanced_accuracy_score(y_te, y_hat)
        auc = (
            roc_auc_score(y_te, best.predict_proba(X_te)[:, 1])
            if hasattr(best, "predict_proba") and len(np.unique(y)) == 2
            else np.nan
        )
        y_proba = (
            best.predict_proba(X_te)[:, 1]
            if hasattr(best, "predict_proba")
            else np.full_like(y_hat, fill_value=np.nan, dtype=float)
        )

        metrics = {
            "acc": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "balanced_acc": balanced_acc,
            "auc": auc,
            **search_results,
            "y_true": y_te,
            "y_pred": y_hat,
            "y_proba": y_proba,
            "groups": g_te
        }


    else:
        r2   = r2_score(y_te, y_hat)
        mae  = mean_absolute_error(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        metrics = {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            **search_results,
            "y_true": y_te,
            "y_pred": y_hat,
            "groups": g_te
        }
        if v_te is not None:
            metrics["y_video"] = v_te

    return metrics, best


# ############### SVR AND RF FOR REGRESSION ###############
def SVM_reg(X, y, groups, *, split_kwargs=None, fixed_params=None, video_ids=None):
    """ Execute SVR for regression
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        # ("pca",     PCA(random_state=42)),
        ("svr",     SVR())
    ])

    grid = [
    {
        "svr__kernel": ["rbf"],
        "svr__C": [0.1, 1, 10], # large c; low bias, high variance
        "svr__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1],
        "svr__epsilon": [0.05, 0.1, 0.2]
    },
    {
        "svr__kernel": ["poly"],
        "svr__C": [0.1, 1, 10],
        "svr__degree": [2, 3],
        "svr__coef0": [0.0, 0.5],
        "svr__epsilon": [0.05, 0.1, 0.2]
    },
    {
        "svr__kernel": ["sigmoid"],
        "svr__C": [0.1, 1],
        "svr__gamma": [0.1],
        "svr__coef0": [0.0],
        "svr__epsilon": [0.05, 0.1, 0.2]
    }
    ]
    
    metrics, search = run_experiment(
        X, y, groups,
        pipeline=pipe,
        param_grid=grid,
        scoring="neg_mean_squared_error",
        is_classification=False,
        split_kwargs=split_kwargs,
        fixed_params=fixed_params,
        video_ids=video_ids
    )
    return metrics, search

def RF_reg(X, y, groups, *, split_kwargs=None, fixed_params=None, video_ids=None): 
    """ Execute RF using LOPO.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        # ("pca", PCA(random_state=42)),
        ("rf",      RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    grid = {
    'rf__n_estimators': [100, 150, 200],
    'rf__max_depth': [5, 7, 10],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [3, 5, 10],
    'rf__max_features': ['sqrt', 0.5],
    "rf__max_samples":    [None, 0.8],
    }
    
    metrics, search = run_experiment(
        X, y, groups,
        pipeline=pipe,
        param_grid=grid,
        scoring="neg_mean_squared_error",
        is_classification=False,
        split_kwargs=split_kwargs,
        fixed_params=fixed_params,
        video_ids=video_ids
    )
    return metrics, search

######### BINARY CLASSIFICATION ############
def make_binary_labels(df, labels_z=False):
    """
    Adds two Boolean columns for binary classification:
      - If labels_z=True, thresholds at 0.0 (z-score)
      - Else, thresholds at the median value of each label (per dataset)

    Output columns:
      • Arousal_bin  = 1 if arousal > threshold else 0
      • Valence_bin  = 1 if valence > threshold else 0
    """
    df = df.copy()
    
    if labels_z:
        arousal_thresh = 0.0
        valence_thresh = 0.0
    else:
        arousal_thresh = df["arousal"].median()
        valence_thresh = df["valence"].median()

    df["Arousal_bin"] = (df["arousal"] > arousal_thresh).astype(int)
    df["Valence_bin"] = (df["valence"] > valence_thresh).astype(int)
    return df

# svc for classification
def SVC_class(X, y, groups, *, split_kwargs=None, fixed_params=None, video_ids=None):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), # impute nans
        ("scaler", StandardScaler()), # scale features only for SVC
        # ("pca", PCA(random_state=42)),
        ("svc",    SVC(probability= True))
    ])
    
    param_grid = [
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.1, 1, 10], # large c; low bias, high variance
        "svc__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1],
        
    },
    {
        "svc__kernel": ["poly"],
        "svc__C": [0.1, 1, 10],
        "svc__degree": [2, 3],
        "svc__coef0": [0.0, 0.5],
    },
    {
        "svc__kernel": ["sigmoid"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": [0.1],
        "svc__coef0": [0.0],
    }
    ]
        
    metrics, search = run_experiment(
        X, y, groups,
        pipeline=pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        is_classification=True,
        split_kwargs=split_kwargs,
        fixed_params=fixed_params,
        video_ids=video_ids  
    )
    return metrics, search

# svm for random forests
def RF_class(X, y, groups, *, split_kwargs=None, fixed_params=None, video_ids=None):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        # ("pca", PCA(random_state=42)),
        ("rf",     RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # new, expanded grid
    param_grid = {
    'rf__n_estimators': [100, 150, 200],
    'rf__max_depth': [5, 7, 10],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [3, 5, 10],
    'rf__max_features': ['sqrt', 0.5],
    "rf__max_samples":    [None, 0.8],
    "rf__class_weight": [None, "balanced"]  # for classification only
    }

    metrics, search = run_experiment(
        X, y, groups,
        pipeline=pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        is_classification=True,
        split_kwargs=split_kwargs,
        fixed_params=fixed_params,
        video_ids=video_ids 
    )
    return metrics, search


# ------- final execution functions --------

# amigos
def perform_analysis(merged_short, merged_long, labels_df,
    pers_short: pd.DataFrame = None,
    pers_long:  pd.DataFrame = None,
    bi_class: bool = True,
    fixed_params=None,
    fixed_params_class=None
):
    """ Prepare datasets (X, y, groups) for regression & classification. Runs regressionmodels, if bi_class = True also the classification.
    Returns: all_results
    """
    # Ensure same dtype *before* merging
    merged_short["ParticipantID"] = merged_short["ParticipantID"].astype(str)
    merged_long ["ParticipantID"] = merged_long ["ParticipantID"].astype(str)

    #  Merge in personality if provided
    if pers_short is not None:
        pers_short = pers_short.copy()                       # safety
        pers_short["ParticipantID"] = pers_short["ParticipantID"].astype(str)
        merged_short = merged_short.merge(
            pers_short, on="ParticipantID", how="left"
        )
    if pers_long is not None:
        pers_long = pers_long.copy()
        pers_long["ParticipantID"] = pers_long["ParticipantID"].astype(str)
        merged_long  = merged_long.merge(
            pers_long, on="ParticipantID", how="left"
        )
    # REGRESSION - mask participants 8 and 28
    # Prepare dataset (by combining features + labels. get: X, y_arousal, y_valence and groups
    X_s, y_ar_s, y_val_s, grp_s, merged_short_labeled, video_s = prepare_dataset(merged_short, labels_df)
    X_l, y_ar_l, y_val_l, grp_l, merged_long_labeled,  video_l = prepare_dataset(merged_long, labels_df)

    exclude_pids = {"8", "28"}
    mask_s = ~np.isin(grp_s.astype(str), list(exclude_pids))
    mask_l = ~np.isin(grp_l.astype(str), list(exclude_pids))

    # apply mask
    X_s_reg   = X_s[mask_s]
    y_ar_s_reg = y_ar_s[mask_s]
    y_val_s_reg = y_val_s[mask_s]
    grp_s_reg = grp_s[mask_s]

    X_l_reg   = X_l[mask_l]
    y_ar_l_reg = y_ar_l[mask_l]
    y_val_l_reg = y_val_l[mask_l]
    grp_l_reg = grp_l[mask_l]

    # build regression dataset. Only masekd arrays 
    reg_datasets = {
        "short-arousal": {"X": X_s_reg, "y": y_ar_s_reg, "groups": grp_s_reg, "video_ids": video_s[mask_s]},   # include video id for later analysis, but mask because it shouldn't be used my the model
        "short-valence": {"X": X_s_reg, "y": y_val_s_reg, "groups": grp_s_reg, "video_ids": video_s[mask_s]},
        "long-arousal":  {"X": X_l_reg, "y": y_ar_l_reg, "groups": grp_l_reg, "video_ids": video_l[mask_l]},
        "long-valence":  {"X": X_l_reg, "y": y_val_l_reg, "groups": grp_l_reg, "video_ids": video_l[mask_l]},
    }
    reg_models = {"RF": RF_reg, "SVR": SVM_reg}
    reg_results = evaluate_models(
        reg_datasets, 
        reg_models, 
        fixed_params=fixed_params
        )
    
    # CLASSIFICATION
    if bi_class:        
        # create binary labels. As y is centered around the mean, we split at 0 (labels_z = True)
        short_bin = make_binary_labels(merged_short_labeled, labels_z = False)
        long_bin  = make_binary_labels(merged_long_labeled, labels_z = False)
        # same x, but new ys (bins)
        yA_short = short_bin["Arousal_bin"].values
        yV_short = short_bin["Valence_bin"].values
        yA_long  = long_bin ["Arousal_bin"].values
        yV_long  = long_bin ["Valence_bin"].values
        
        yA_short_masked = yA_short[mask_s]   # short slice, participants 18 & 28 removed
        yV_short_masked = yV_short[mask_s]
        yA_long_masked  = yA_long[mask_l]    # long slice, participants 18 & 28 removed
        yV_long_masked  = yV_long[mask_l]

        class_datasets = {
            "short-Arousal": {"X": X_s_reg, "y": yA_short_masked, "groups": grp_s_reg, "video_ids": video_s[mask_s]},
            "short-Valence": { "X": X_s_reg, "y": yV_short_masked, "groups": grp_s_reg, "video_ids": video_s[mask_s]},
            "long-Arousal": {"X": X_l_reg, "y": yA_long_masked, "groups": grp_l_reg,  "video_ids": video_l[mask_l]},
            "long-Valence": { "X": X_l_reg, "y": yV_long_masked, "groups": grp_l_reg,"video_ids": video_l[mask_l]},
        }
        # run classifiers!
        class_models = {"SVC":  SVC_class, "RF":   RF_class}
        class_results = {} # save results in dict
        
        class_results = evaluate_models(
        class_datasets,
        class_models,
        split_kwargs={},          # default LOPO; change if you need
        fixed_params=fixed_params_class
        )
    else:
        class_results = {}
    return {
    "results": {"regression": reg_results, "classification": class_results},
    "labeled_short": merged_short_labeled,
    "labeled_long": merged_long_labeled
} 

# more or less same function but for phymer
def phy_perform_analysis(
    feature_df: pd.DataFrame,
    labels_df:  pd.DataFrame,
    bi_class:   bool = True,
    fixed_params:       dict = None,
    fixed_params_class: dict = None,
):
    """ Prepare datasets for regression & classification
    Args:
        - feature_df: DataFrame with ParticipantID, VideoID and all features
        - labels_df:  DataFrame with ParticipantID, VideoID, Arousal_z, Valence_z
        - bi_class:   whether to run binary classification too
        - fixed_params:       dict for regression {(model,dset): params} or None
        - fixed_params_class: dict for classification or None
    Returns: results for regression & classification
    """
    
    # prepare dataset: merge & split into X, y_ar, y_val, groups, merged, video_ids
    X, y_ar, y_val, groups, merged, video_ids = prepare_dataset(
        merged_features=feature_df,
        labels_df      =labels_df     
    )

    # Regression
    reg_datasets = {
        "arousal": {"X": X, "y": y_ar, "groups": groups, "video_ids": video_ids},
        "valence": {"X": X, "y": y_val, "groups": groups, "video_ids": video_ids},
    } # build datasets
    reg_models   = {"RF": RF_reg, "SVR": SVM_reg} # through here, run_experiment is called
    # keep results
    reg_results = evaluate_models(
        reg_datasets,
        reg_models,
        split_kwargs={},            
        fixed_params=fixed_params  
    )
    
    # Classification
    class_results = {}
    if bi_class:
        # derive binary labels
        bins = make_binary_labels(merged, labels_z = False) 
        class_datasets = {
            "arousal": {"X": X, "y": bins["Arousal_bin"].values, "groups": groups, "video_ids": video_ids},
            "valence": {"X": X, "y": bins["Valence_bin"].values, "groups": groups, "video_ids": video_ids},
        }
        class_models = {"SVC": SVC_class, "RF": RF_class}
        
        class_results = evaluate_models(
            class_datasets,
            class_models,
            split_kwargs={},             
            fixed_params=fixed_params_class
        )
    else:
        class_results = {}
    return {"regression": reg_results, "classification": class_results, "labeled_full": merged}
