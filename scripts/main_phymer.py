from Preprocessing import run_phymer_pipeline
from models import process_labels_phy, phy_perform_analysis 
import pandas as pd
import pickle
from pathlib import Path


script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

PCA_PATH = script_dir / "personality_phymer.pkl"
FEATURES = script_dir / "combined_PHY_features.csv"
LABELS = project_root / "PhyMER" / "PhyMER Dataset" / "labels.csv"
PHYMER_DATA_ROOT = project_root / "PhyMER" / "PhyMER Dataset"

if __name__ == "__main__":
    
    # ---- Run preprocessing ----
    # root = "/Users/anna/Downloads/Scriptie_code/PhyMER"
    # combined_features = run_phymer_pipeline(
    #     root_dir        = root,
    #     fs_eda          = 4,      
    #     fs_bvp          = 64,     
    #     bpf_fc          = 0.45,   
    #     amp_min         = 0.1,   
    #     save_csv        = "PHY_features.csv"
    # )
    # print(combined_features.head())


    # ---- Run analysis ----
    # when preprocessing once has ran this can be uncommented
    combined_features = pd.read_csv("PHY_features.csv")
    
    combined_features = pd.read_csv(FEATURES) # Load features
    labels_z = process_labels_phy(LABELS)     # Load labels
    pers_df = pd.read_pickle(PCA_PATH)        # Load personality variant info

    # define different variants (without personality vs with personality)
    traits_cols    = ["ParticipantID", "Extroversion", "Agreeableness",
                    "Conscientiousness", "Openness", "Neuroticism"]
    trait_pca_cols = ["ParticipantID", "PC1", "PC2"]
    cluster_cols   = ["ParticipantID", "Cluster_PC1"]

    traits_df      = pers_df[traits_cols]
    trait_pca_df   = pers_df[trait_pca_cols]
    cluster_df     = pers_df[cluster_cols]
    
    variants = {
        # "No_personality": None,
        # "Traits":         traits_df,
        "Clusters": cluster_df,
    }

    
    phy_results = {}
    for name, pers_df_variant in variants.items():
        print(f"\n=== Running PhyMER with {name.replace('_', ' ')} ===")

        if pers_df_variant is not None:
            feat = combined_features.merge(pers_df_variant, on="ParticipantID", how="left")
        else:
            feat = combined_features.copy()

        res = phy_perform_analysis(
            feature_df=feat,
            labels_df=labels_z,
            bi_class=False, 
            fixed_params=None,
            fixed_params_class=None
        )
        phy_results[name] = res

    # Pickle results
    with open("phymer_results.pkl", "wb") as f:
        pickle.dump(phy_results, f)
        