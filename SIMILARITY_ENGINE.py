import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

################ LOAD OUTFIELD MODEL DATASET ################
df = pd.read_csv("outfield_model_dataset.csv", encoding="utf-8")

################ BASIC CLEANING / NORMALISATION ################

# Remove noisy feature
df = df.drop(columns=['international_reputation'], errors='ignore')

# Normalize height & weight to 0-100
for col in ['height_cm', 'weight_kg']:
    if col in df.columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100

# Normalize skill_moves and weak_foot (1-5 -> 0-100)
for col in ['skill_moves', 'weak_foot']:
    if col in df.columns:
        df[col] = (df[col] - 1) / (5 - 1) * 100

# Normalize num_positions (1-4 -> 0-100)
if 'num_positions' in df.columns:
    df['num_positions'] = (df['num_positions'] - 1) / (4 - 1) * 100

################ FILTER ATTACKER GROUP ################

# ATT + MF/ATT
df_att = df[df['position_universal'].isin(['ATT', 'MF/ATT'])].copy()

print("\nAttacker group size:", len(df_att))
print(df_att['position_universal'].value_counts())

################ SELECT ATTACKER FEATURES ################

att_features = [
    'attacking_finishing',
    'attacking_volleys',
    'attacking_heading_accuracy',
    'attacking_short_passing',
    'skill_dribbling',
    'skill_curve',
    'skill_fk_accuracy',
    'skill_long_passing',
    'skill_ball_control',
    'movement_acceleration',
    'movement_sprint_speed',
    'movement_agility',
    'movement_reactions',
    'movement_balance',
    'power_shot_power',
    'power_jumping',
    'power_strength',
    'power_long_shots',
    'mentality_vision',
    'mentality_penalties',
    'mentality_composure',
    'mentality_attack_position',
    'weak_foot',
    'skill_moves',
    'height_cm',
    'weight_kg',
    'age',
    'num_positions'
]

# Keep only columns that actually exist
att_features = [c for c in att_features if c in df_att.columns]

# Metadata
att_metadata = [
    'player_id',
    'name',
    'positions',
    'position_universal',
    'club_name',
    'overall_rating',
    'potential',
    'value',
    'wage'
]
att_metadata = [c for c in att_metadata if c in df_att.columns]

# Final modelling dataframe
df_att_model = df_att[att_metadata + att_features].copy()

print("\nAttacker modelling dataframe shape:", df_att_model.shape)
print("\nAttacker features used:")
print(att_features)

################ OPTIONAL: ADD ATTACKER CLUSTER LABELS ################
# Keep this only if you want cluster context in the similarity output

def classify_attacker_cluster(row):
    # Simple rule-based reuse of your interpreted clusters
    # You can remove this whole block if not needed.
    finishing = row.get('attacking_finishing', np.nan)
    num_positions = row.get('num_positions', np.nan)
    age = row.get('age', np.nan)

    if pd.notna(finishing) and pd.notna(age) and finishing >= 72 and age >= 26:
        return "Elite Finisher"
    elif pd.notna(num_positions) and num_positions >= 60:
        return "Complete Attacker"
    elif pd.notna(age) and age <= 23:
        return "Young Prospect"
    else:
        return "Role-Specific Attacker"

df_att_model['att_cluster_label'] = df_att_model.apply(classify_attacker_cluster, axis=1)

################ SCALE FEATURES ################

att_features_only = df_att_model[att_features].copy()

scaler = StandardScaler()
att_scaled = scaler.fit_transform(att_features_only)

att_scaled_df = pd.DataFrame(
    att_scaled,
    columns=att_features,
    index=df_att_model.index
)

print("\nScaled attacker data sample:")
print(att_scaled_df.head())

################ COSINE SIMILARITY MATRIX ################

att_similarity = cosine_similarity(att_scaled_df)

att_similarity_df = pd.DataFrame(
    att_similarity,
    index=df_att_model['name'],
    columns=df_att_model['name']
)

print("\nAttacker cosine similarity matrix shape:")
print(att_similarity_df.shape)

################ HELPER: HANDLE DUPLICATE PLAYER NAMES ################
# If duplicate names exist, use player_id for exact lookup

name_counts = df_att_model['name'].value_counts()
duplicate_names = name_counts[name_counts > 1].index.tolist()

if duplicate_names:
    print("\nDuplicate player names detected:")
    print(duplicate_names[:20])

################ SIMILARITY FUNCTION ################

def find_similar_attackers(player_name, top_n=10):
    """
    Returns the top_n most similar attackers to the given player_name.
    If duplicate names exist, it takes the first match found.
    """

    # Check if player exists
    if player_name not in att_similarity_df.columns:
        print(f"\nPlayer '{player_name}' not found in attacker dataset.")
        return None

    # Similarity scores
    similar_scores = att_similarity_df[player_name].sort_values(ascending=False)

    # Remove self-match
    similar_scores = similar_scores.drop(player_name, errors='ignore')

    # Take top N
    top_players = similar_scores.head(top_n)

    # Build output dataframe
    results = df_att_model[df_att_model['name'].isin(top_players.index)].copy()

    # Keep only useful output columns
    output_cols = [
        'name',
        'positions',
        'position_universal',
        'club_name',
        'overall_rating',
        'potential',
        'value',
        'age',
        'num_positions',
        'att_cluster_label'
    ]
    output_cols = [c for c in output_cols if c in results.columns]

    results = results[output_cols].copy()
    results['similarity_score'] = results['name'].map(top_players)

    return results.sort_values(by='similarity_score', ascending=False)

################ MORE ROBUST FUNCTION USING PLAYER_ID ################

def find_similar_attackers_by_id(player_id, top_n=10):
    """
    Safer version when duplicate names exist.
    Finds top_n most similar attackers using player_id.
    """

    if player_id not in df_att_model['player_id'].values:
        print(f"\nPlayer ID '{player_id}' not found in attacker dataset.")
        return None

    player_row = df_att_model[df_att_model['player_id'] == player_id].iloc[0]
    player_name = player_row['name']
    player_index = player_row.name

    similarity_series = pd.Series(
        att_similarity[player_index],
        index=df_att_model.index
    ).sort_values(ascending=False)

    similarity_series = similarity_series.drop(player_index, errors='ignore')
    top_idx = similarity_series.head(top_n).index

    results = df_att_model.loc[top_idx].copy()

    output_cols = [
        'player_id',
        'name',
        'positions',
        'position_universal',
        'club_name',
        'overall_rating',
        'potential',
        'value',
        'age',
        'num_positions',
        'att_cluster_label'
    ]
    output_cols = [c for c in output_cols if c in results.columns]

    results = results[output_cols].copy()
    results['similarity_score'] = similarity_series.loc[top_idx].values

    print(f"\nTop {top_n} similar attackers to {player_name} (ID: {player_id}):")
    return results.sort_values(by='similarity_score', ascending=False)

################ EXAMPLE USAGE ################

# Example by player name
example_name = "O. McBurnie"

if example_name in df_att_model['name'].values:
    print(f"\nTop similar attackers to {example_name}:")
    print(find_similar_attackers(example_name, top_n=10))
else:
    print(f"\nExample player '{example_name}' not found. Use a valid name from your dataset.")

################ OPTIONAL: SAVE ATTACKER SIMILARITY DATA ################

# Save the cleaned attacker modelling dataset
df_att_model.to_csv("attackers_similarity_dataset.csv", index=False)

print("\nSaved: attackers_similarity_dataset.csv")