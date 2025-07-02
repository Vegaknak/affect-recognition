from Preprocessing import extract_videos, convert_gsr, run_pipeline
from models import load_AMIGOS_data, process_labels_amigos, exp_condition, perform_analysis
import pandas as pd
import pickle
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

DATA_DIR = project_root / "Amigos" / "Physiological Data Preprocessed"
METADATA_PATH = project_root / "Amigos" / "Metadata" / "Participant_Questionnaires.xlsx"

PCA_PATH = project_root / "Code" / "AMIGOS_pers.pkl"
PARAMS_REG = project_root / "Code" / "best_params_regression.csv"
PARAMS_CLASS = project_root / "Code" / "best_params_classification.csv" # Uncomment if used

MERGED_SHORT_PKL = script_dir / "AMIGOS_features_s.pkl"
MERGED_LONG_PKL  = script_dir / "AMIGOS_features_l.pkl"
FS = 128

# ---- best parameters ----
best_params_reg = pd.read_csv(PARAMS_REG,
dtype={"rf__n_estimators":"Int16", "rf__max_depth":"Int16", "rf__min_samples_split":"Int16",
    "rf__min_samples_leaf":"Int16", "svr__degree":"Int16"})

# best_params_class = pd.read_csv(PARAMS_CLASS, # Use PARAMS_CLASS here
# dtype={"rf__n_estimators":"Int16", "rf__max_depth":"Int16",  "rf__min_samples_split":"Int16",
#     "rf__min_samples_leaf":"Int16", "svc__degree":"Int16",})

if __name__ == "__main__":
    # Load data using the portable Path objects
    mats = load_AMIGOS_data(DATA_DIR)
    # Get labels 
    labels_df = process_labels_amigos(mats)

    # ---- Run preprocessing ---
    ecg_s, gsr_s = extract_videos(mats, range(0,16), excluded_ppn=[9], label="short")
    ecg_l, gsr_l = extract_videos(mats, range(16,20), excluded_ppn=[8,24,28], label="long")
    gsr_s = convert_gsr(gsr_s)
    gsr_l = convert_gsr(gsr_l)

    merged_short, merged_long = run_pipeline(
        ecg_s, gsr_s, ecg_l, gsr_l,
        save_csv=(MERGED_SHORT_PKL, MERGED_LONG_PKL) 
    )

    # ---- Run analysis ---
    merged_short = pd.read_pickle(MERGED_SHORT_PKL)
    merged_long  = pd.read_pickle(MERGED_LONG_PKL)

    pers_df = pd.read_pickle(PCA_PATH)
    session_df = pd.read_excel(METADATA_PATH, sheet_name=0, dtype={"UserID": str})
    merged_long  = exp_condition(merged_long, session_df) # add Alone_long column

    # define different feature sets 
    traits_cols      = ["ParticipantID", "Extroversion", "Agreeableness",
                        "Conscientiousness", "Openness", "Neuroticism"]
    trait_pca_cols   = ["ParticipantID", "PC1", "PC2"]
    cluster_pca_cols = ["ParticipantID", "Cluster_PC1", "Cluster_PC2"]

    traits_df      = pers_df[traits_cols]
    trait_pca_df   = pers_df[trait_pca_cols]
    cluster_pca_df = pers_df[cluster_pca_cols]

    variants = {
        "No_personality": (None, None),
        "Traits":         (traits_df, traits_df),
        "Clusters":       (cluster_pca_df, cluster_pca_df),
    }


    results_by_personality = {}
    for name, (ps, pl) in variants.items():
        print(f"\n=== Running models with {name.replace('_',' ')} ===")
        df_personality = best_params_reg[best_params_reg["personality"] == name]
        fixed_params = {(row["model"], row["dataset"]): {
                k: row[k]
                for k in best_params_reg.columns
                if k not in ["personality", "model", "dataset"]
                and not pd.isna(row[k])
            }
            for _, row in df_personality.iterrows()
        }

        # df_clf_personality = best_params_class[best_params_class["personality"] == name]
        # fixed_params_class = {
        #     (row["model"], row["dataset"]): {
        #         k: row[k]
        #         for k in best_params_class.columns
        #         if k not in ["personality", "model", "dataset"]
        #         and not pd.isna(row[k])
        #     }
        #     for _, row in df_clf_personality.iterrows()    # filter so it doesnt take full table
        # }

        res = perform_analysis(
            merged_short,
            merged_long,
            labels_df,
            pers_short=ps,
            pers_long=pl,
            bi_class=False, 
            fixed_params=None,  
            fixed_params_class=None
        )
        results_by_personality[name] = res

    with open(script_dir / "Amigos_results_rawlabels.pkl", "wb") as f:
        pickle.dump(results_by_personality, f)
