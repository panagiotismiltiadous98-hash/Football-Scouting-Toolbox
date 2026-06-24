import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

BASE = "C:/Users/User/Desktop/Dataset/sofifa-web-scraper-main/"

################ FEATURE SPACES (exact match to similarity engine) ################

ATT_FEATURES = [
    'attacking_finishing', 'attacking_volleys', 'attacking_heading_accuracy',
    'attacking_short_passing', 'skill_dribbling', 'skill_curve',
    'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
    'movement_reactions', 'movement_balance', 'power_shot_power',
    'power_jumping', 'power_strength', 'power_long_shots',
    'mentality_vision', 'mentality_penalties', 'mentality_composure',
    'mentality_attack_position', 'weak_foot', 'skill_moves',
    'height_cm', 'weight_kg', 'age', 'num_positions'
]

MF_FEATURES = [
    'attacking_short_passing', 'attacking_volleys', 'skill_dribbling',
    'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
    'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
    'movement_agility', 'movement_reactions', 'movement_balance',
    'power_shot_power', 'power_jumping', 'power_stamina',
    'power_strength', 'power_long_shots', 'mentality_aggression',
    'mentality_interceptions', 'mentality_vision', 'mentality_penalties',
    'mentality_composure', 'mentality_attack_position',
    'defending_defensive_awareness', 'defending_standing_tackle',
    'defending_sliding_tackle', 'weak_foot', 'skill_moves',
    'height_cm', 'weight_kg', 'age', 'num_positions'
]

DEF_FEATURES = [
    'defending_defensive_awareness', 'defending_standing_tackle',
    'defending_sliding_tackle', 'power_strength', 'power_jumping',
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
    'movement_reactions', 'movement_balance', 'attacking_short_passing',
    'skill_long_passing', 'skill_ball_control', 'mentality_interceptions',
    'mentality_aggression', 'mentality_composure', 'mentality_vision',
    'attacking_crossing', 'weak_foot', 'skill_moves',
    'height_cm', 'weight_kg', 'age', 'num_positions'
]

GK_FEATURES = [
    'goalkeeping_gk_diving', 'goalkeeping_gk_handling',
    'goalkeeping_gk_kicking', 'goalkeeping_gk_positioning',
    'goalkeeping_gk_reflexes', 'attacking_short_passing',
    'mentality_vision', 'mentality_composure', 'movement_reactions',
    'movement_agility', 'movement_balance', 'power_jumping',
    'power_strength', 'power_shot_power', 'weak_foot',
    'height_cm', 'weight_kg', 'age'
]

################ HELPER FUNCTION ################

def compare_pair(df, features, name1_search, name2_search, role, cluster_col):
    """
    Finds two players by name substring, computes their cosine similarity
    using the full-population StandardScaler (matching the dashboard exactly),
    and prints a full side-by-side attribute comparison.
    """
    features = [f for f in features if f in df.columns]

    # Scale across full population — same as dashboard
    scaler = StandardScaler()
    scaled_all = scaler.fit_transform(df[features].fillna(0))

    idx1 = df[df['name'].str.contains(name1_search, na=False, case=False)].index
    idx2 = df[df['name'].str.contains(name2_search, na=False, case=False)].index

    print(f"\n{'='*70}")
    print(f"  {role.upper()} PAIR VERIFICATION")
    print(f"{'='*70}")
    print(f"  Search 1: '{name1_search}' — {len(idx1)} match(es) found")
    print(f"  Search 2: '{name2_search}' — {len(idx2)} match(es) found")

    if len(idx1) == 0 or len(idx2) == 0:
        print("  ERROR: One or both players not found. Check name spelling.")
        return

    i, j = idx1[0], idx2[0]
    name1 = df.loc[i, 'name']
    name2 = df.loc[j, 'name']

    vec_i = scaled_all[i].reshape(1, -1)
    vec_j = scaled_all[j].reshape(1, -1)
    score = cosine_similarity(vec_i, vec_j)[0][0]

    cluster1 = df.loc[i, cluster_col]
    cluster2 = df.loc[j, cluster_col]

    print(f"\n  Player 1:  {name1}  (age {df.loc[i,'age']:.0f})  →  {cluster1}")
    print(f"  Player 2:  {name2}  (age {df.loc[j,'age']:.0f})  →  {cluster2}")
    print(f"\n  Cosine Similarity (full {role} population scaling): {score:.4f}")
    print(f"  Same cluster: {'YES ✅' if cluster1 == cluster2 else 'NO ❌'}")

    col1 = name1.split('.')[-1].strip()[:10]
    col2 = name2.split('.')[-1].strip()[:10]
    print(f"\n  {'Feature':<35} {col1:>12} {col2:>12} {'Diff':>8}")
    print(f"  {'-'*68}")
    for feat in features:
        v1 = df.loc[i, feat]
        v2 = df.loc[j, feat]
        diff = v1 - v2
        print(f"  {feat:<35} {v1:>12.1f} {v2:>12.1f} {diff:>+8.1f}")

################ LOAD DATASETS ################

print("Loading datasets...")
df_att = pd.read_csv(BASE + "attackers_similarity_dataset.csv")
df_mid = pd.read_csv(BASE + "midfielders_similarity_dataset.csv")
df_def = pd.read_csv(BASE + "defenders_similarity_dataset.csv")
df_gk  = pd.read_csv(BASE + "goalkeepers_similarity_dataset.csv")
print("All datasets loaded successfully.")

################ ATTACKER PAIR ################
compare_pair(df_att, ATT_FEATURES,
             name1_search="Haaland",
             name2_search="Gyökeres",
             role="Attackers",
             cluster_col="att_cluster_label")

################ MIDFIELDER PAIR ################
compare_pair(df_mid, MF_FEATURES,
             name1_search="Caicedo",
             name2_search="Varela",
             role="Midfielders",
             cluster_col="mid_cluster_label")

################ DEFENDER PAIR ################
# Rhys Norrington-Davies vs Dan McNamara
compare_pair(df_def, DEF_FEATURES,
             name1_search="Norrington",
             name2_search="McNamara",
             role="Defenders",
             cluster_col="def_cluster_label")

################ GOALKEEPER PAIR ################
# R. Gurtner vs J. Placide — both Experienced GK
compare_pair(df_gk, GK_FEATURES,
             name1_search="Gurtner",
             name2_search="Placide",
             role="Goalkeepers",
             cluster_col="gk_cluster_label")

print(f"\n{'='*70}")
print("  VERIFICATION COMPLETE")
print(f"{'='*70}\n")