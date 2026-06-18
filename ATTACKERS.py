import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

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

################################################## FILTER ATTACKER GROUP ############################################

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
att_metadata = ['player_id', 'name', 'positions', 'position_universal', 'club_name',
                'overall_rating', 'potential', 'value']
att_metadata = [c for c in att_metadata if c in df_att.columns]

# Modelling dataframe
df_att_model = df_att[att_metadata + att_features].copy()

print("\nAttacker modelling dataframe shape:", df_att_model.shape)
print("\nAttacker features used:")
print(att_features)

################ EXPLORATORY CHECK ################

att_numeric = df_att_model[att_features].copy()

print("\nAverage attacker feature values:")
print(att_numeric.mean().sort_values(ascending=False))

print("\nAttacker feature variability:")
print(att_numeric.std().sort_values(ascending=False))

################ SCALE ATTACKER FEATURES ################

scaler = StandardScaler()
att_scaled = scaler.fit_transform(att_numeric)

att_scaled_df = pd.DataFrame(att_scaled, columns=att_numeric.columns, index=df_att_model.index)

print("\nScaled attacker data sample:")
print(att_scaled_df.head())

################ K-MEANS TUNING ################

inertia = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(att_scaled_df)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(att_scaled_df, cluster_labels))

print("\nK values tested:", list(k_values))
print("Inertia values:", inertia)
print("Silhouette scores:", silhouette_scores)

################ ELBOW METHOD PLOT ################

plt.figure(figsize=(8,5))
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method for Attacker K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()

################ SILHOUETTE SCORE PLOT ################

plt.figure(figsize=(8,5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Attacker K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()

################ FINAL ATTACKER K-MEANS MODEL ################

best_k = 4

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_att_model['att_cluster'] = kmeans_final.fit_predict(att_scaled_df)

print("\nAttacker cluster counts:")
print(df_att_model['att_cluster'].value_counts().sort_index())

################ ATTACKER CLUSTER LABELS ################

cluster_names = {
    0: "Versatile Attacker",
    1: "Complete Attacker",
    2: "Young Prospect",
    3: "Role-Specific Attacker"
}

df_att_model['att_cluster_label'] = df_att_model['att_cluster'].map(cluster_names)

print("\nAttacker cluster distribution:")
print(df_att_model['att_cluster_label'].value_counts())

################ ATTACKER CLUSTER PROFILE ################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

cluster_summary = df_att_model.groupby('att_cluster_label')[att_numeric.columns].mean().round(2)

print("\nCluster feature averages (modelling features only):")
print(cluster_summary)

context_cols = [c for c in ['overall_rating', 'potential'] if c in df_att_model.columns]
if context_cols:
    context_summary = df_att_model.groupby('att_cluster_label')[context_cols].mean().round(2)
    print("\nCluster context (NOT used in clustering, for narrative only):")
    print(context_summary)

################ UMAP VISUALISATION ################

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(att_scaled_df)

umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=df_att_model.index)
umap_df['cluster'] = df_att_model['att_cluster']
umap_df['cluster_label'] = df_att_model['att_cluster_label']

centroids = kmeans_final.cluster_centers_
centroids_2d = reducer.transform(centroids)

plt.figure(figsize=(9,7))

ax = sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='cluster_label',
    palette='Set2',
    s=60
)

ax.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    c='black',
    s=220,
    marker='X',
    label='Centroids'
)

# Add centroid labels
for cluster_id, (x, y) in enumerate(centroids_2d):
    ax.text(
        x + 0.08,
        y + 0.08,
        cluster_names[cluster_id],
        fontsize=10,
        weight='bold',
        color='black'
    )

plt.title("UMAP Projection of Attacker Clusters")
plt.legend(title="Attacker Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

################ EXAMPLE PLAYERS PER CLUSTER ################

print("\nExample attackers per cluster:\n")

for label in df_att_model['att_cluster_label'].unique():
    print(f"{label}:")
    print(df_att_model.loc[df_att_model['att_cluster_label'] == label, 'name'].head(5).to_list())
    print()

################ SAVE REAL K-MEANS LABELS FOR DOWNSTREAM USE ################
# SIMILARITY_ENGINE.py merges this in by player_id instead of running its
# own rule-based classify_attacker_cluster() heuristic.

df_att_model[['player_id', 'att_cluster', 'att_cluster_label']].to_csv(
    "attackers_cluster_labels.csv", index=False
)
print("Saved: attackers_cluster_labels.csv")