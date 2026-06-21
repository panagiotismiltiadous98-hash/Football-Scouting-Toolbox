import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ROLE_FILES = {
    "Attackers": "attackers_similarity_dataset.csv",
    "Midfielders": "midfielders_similarity_dataset.csv",
    "Defenders": "defenders_similarity_dataset.csv",
    "Goalkeepers": "goalkeepers_similarity_dataset.csv",
}

CLUSTER_COL_BY_ROLE = {
    "Attackers": "att_cluster_label",
    "Midfielders": "mid_cluster_label",
    "Defenders": "def_cluster_label",
    "Goalkeepers": "gk_cluster_label",
}

FEATURES_BY_ROLE = {
    "Attackers": [
        'attacking_finishing', 'attacking_volleys', 'attacking_heading_accuracy',
        'attacking_short_passing', 'skill_dribbling', 'skill_curve',
        'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
        'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
        'movement_reactions', 'movement_balance', 'power_shot_power',
        'power_jumping', 'power_strength', 'power_long_shots',
        'mentality_vision', 'mentality_penalties', 'mentality_composure',
        'mentality_attack_position', 'weak_foot', 'skill_moves',
        'height_cm', 'weight_kg', 'age', 'num_positions'
    ],
    "Midfielders": [
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
    ],
    "Defenders": [
        'defending_defensive_awareness', 'defending_standing_tackle',
        'defending_sliding_tackle', 'power_strength', 'power_jumping',
        'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
        'movement_reactions', 'movement_balance', 'attacking_short_passing',
        'skill_long_passing', 'skill_ball_control', 'mentality_interceptions',
        'mentality_aggression', 'mentality_composure', 'mentality_vision',
        'attacking_crossing', 'weak_foot', 'skill_moves',
        'height_cm', 'weight_kg', 'age', 'num_positions'
    ],
    "Goalkeepers": [
        'goalkeeping_gk_diving', 'goalkeeping_gk_handling',
        'goalkeeping_gk_kicking', 'goalkeeping_gk_positioning',
        'goalkeeping_gk_reflexes', 'attacking_short_passing',
        'mentality_vision', 'mentality_composure', 'movement_reactions',
        'movement_agility', 'movement_balance', 'power_jumping',
        'power_strength', 'power_shot_power', 'weak_foot',
        'height_cm', 'weight_kg', 'age'
    ]
}

TOP_N = 10  # how many similar players to check per query player

# ─── COMPUTATION ────────────────────────────────────────────────────────────────
results = []

for role, file in ROLE_FILES.items():
    df = pd.read_csv(file)
    cluster_col = CLUSTER_COL_BY_ROLE[role]
    features = [f for f in FEATURES_BY_ROLE[role] if f in df.columns]

    # Build similarity matrix (same pipeline as dashboard)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features].fillna(0))
    sim = cosine_similarity(scaled)
    sim_df = pd.DataFrame(sim, index=df.index, columns=df.index)

    match_percentages = []

    for idx in df.index:
        query_cluster = df.loc[idx, cluster_col]
        if pd.isna(query_cluster):
            continue

        # Get top-N similar players (excluding self)
        scores = sim_df.loc[idx].sort_values(ascending=False).drop(idx)
        top_idx = scores.head(TOP_N).index

        # Count how many share the same cluster
        top_clusters = df.loc[top_idx, cluster_col]
        same_cluster_count = (top_clusters == query_cluster).sum()
        match_pct = (same_cluster_count / TOP_N) * 100
        match_percentages.append(match_pct)

    avg_match_pct = np.mean(match_percentages)
    median_match_pct = np.median(match_percentages)
    n_players = len(match_percentages)

    results.append({
        'Role': role,
        'N Players': n_players,
        'Avg % Same Cluster (Top-10)': round(avg_match_pct, 1),
        'Median % Same Cluster (Top-10)': round(median_match_pct, 1),
    })

    print(f"{role}: avg={avg_match_pct:.1f}%, median={median_match_pct:.1f}% (n={n_players})")

# ─── SUMMARY TABLE ──────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(results)
print("\n" + "="*70)
print("CONVERGENT VALIDATION SUMMARY")
print("="*70)
print(summary_df.to_string(index=False))

summary_df.to_csv("cluster_validation_summary.csv", index=False)
print("\nSaved: cluster_validation_summary.csv")
