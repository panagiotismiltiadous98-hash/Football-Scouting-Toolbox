import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap

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

################################################## FILTER MIDFIELDER GROUP ############################################

# MF + DF/MF EXCLUDING MF/ATT AS MORE APPRORIATE IN ATTACKING FEATURE SPACE #
df_mid = df[df['position_universal'].isin(['MF', 'DF/MF'])].copy()

print("\nMidfielder group size:", len(df_mid))
print(df_mid['position_universal'].value_counts())

################ SELECT MIDFIELDER FEATURES ################

mid_features = [
    'attacking_short_passing',
    'attacking_volleys',
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
    'power_stamina',
    'power_strength',
    'power_long_shots',
    'mentality_aggression',
    'mentality_interceptions',
    'mentality_vision',
    'mentality_penalties',
    'mentality_composure',
    'mentality_attack_position',
    'defending_defensive_awareness',
    'defending_standing_tackle',
    'defending_sliding_tackle',
    'weak_foot',
    'skill_moves',
    'height_cm',
    'weight_kg',
    'age',
    'num_positions'
]

# Keep only columns that actually exist
mid_features = [c for c in mid_features if c in df_mid.columns]

# Metadata
mid_metadata = ['player_id', 'name', 'positions', 'position_universal', 'club_name',
                'overall_rating', 'potential', 'value']
mid_metadata = [c for c in mid_metadata if c in df_mid.columns]

# Modelling dataframe
df_mid_model = df_mid[mid_metadata + mid_features].copy()

print("\nMidfielder modelling dataframe shape:", df_mid_model.shape)
print("\nMidfielder features used:")
print(mid_features)

################ EXPLORATORY CHECK ################

mid_numeric = df_mid_model[mid_features].copy()

print("\nAverage midfielder feature values:")
print(mid_numeric.mean().sort_values(ascending=False))

print("\nMidfielder feature variability:")
print(mid_numeric.std().sort_values(ascending=False))

################ SCALE MIDFIELDER FEATURES ################

mid_scaler = StandardScaler()
mid_scaled = mid_scaler.fit_transform(mid_numeric)

mid_scaled_df = pd.DataFrame(mid_scaled, columns=mid_numeric.columns, index=df_mid_model.index)

print("\nScaled midfielder data sample:")
print(mid_scaled_df.head())

################ K-MEANS TUNING ################

mid_inertia = []
mid_silhouette_scores = []
mid_k_values = range(2, 11)

for k in mid_k_values:
    mid_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    mid_cluster_labels = mid_kmeans.fit_predict(mid_scaled_df)

    mid_inertia.append(mid_kmeans.inertia_)
    mid_silhouette_scores.append(silhouette_score(mid_scaled_df, mid_cluster_labels))

print("\nMidfielder K values tested:", list(mid_k_values))
print("Midfielder inertia values:", mid_inertia)
print("Midfielder silhouette scores:", mid_silhouette_scores)

################ ELBOW METHOD PLOT ################

plt.figure(figsize=(8,5))
plt.plot(mid_k_values, mid_inertia, marker='o')
plt.title("Elbow Method for Midfielder K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(mid_k_values))
plt.tight_layout()
plt.show()

################ SILHOUETTE SCORE PLOT ################

plt.figure(figsize=(8,5))
plt.plot(mid_k_values, mid_silhouette_scores, marker='o')
plt.title("Silhouette Scores for Midfielder K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(mid_k_values))
plt.tight_layout()
plt.show()

################ FINAL MIDFIELDER K-MEANS MODEL ################

mid_best_k = 4

mid_kmeans_final = KMeans(n_clusters=mid_best_k, random_state=42, n_init=10)
df_mid_model['mid_cluster'] = mid_kmeans_final.fit_predict(mid_scaled_df)

print("\nMidfielder cluster counts:")
print(df_mid_model['mid_cluster'].value_counts().sort_index())

################ MIDFIELDER CLUSTER LABELS ################

mid_cluster_names = {
    0: "Box-to-Box Midfielders",
    1: "Versatile Midfielders",
    2: "Elite Midfielders",
    3: "Raw Physical Midfielders"
}

df_mid_model['mid_cluster_label'] = df_mid_model['mid_cluster'].map(mid_cluster_names)

print("\nMidfielder cluster distribution:")
print(df_mid_model['mid_cluster_label'].value_counts())

################ MIDFIELDER CLUSTER PROFILE ################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

mid_cluster_summary = df_mid_model.groupby('mid_cluster_label')[mid_numeric.columns].mean().round(2)

print("\nMidfielder cluster feature averages:")
print(mid_cluster_summary)

################ UMAP VISUALISATION ################

mid_reducer = umap.UMAP(n_components=2, random_state=42)
mid_embedding = mid_reducer.fit_transform(mid_scaled_df)

mid_umap_df = pd.DataFrame(mid_embedding, columns=['UMAP1', 'UMAP2'], index=df_mid_model.index)
mid_umap_df['cluster'] = df_mid_model['mid_cluster']
mid_umap_df['cluster_label'] = df_mid_model['mid_cluster_label']

mid_centroids = mid_kmeans_final.cluster_centers_
mid_centroids_2d = mid_reducer.transform(mid_centroids)

plt.figure(figsize=(9,7))

mid_ax = sns.scatterplot(
    data=mid_umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='cluster_label',
    palette='Set2',
    s=60
)

mid_ax.scatter(
    mid_centroids_2d[:, 0],
    mid_centroids_2d[:, 1],
    c='black',
    s=220,
    marker='X',
    label='Centroids'
)

# Add centroid labels
for cluster_id, (x, y) in enumerate(mid_centroids_2d):
    mid_ax.text(
        x + 0.08,
        y + 0.08,
        mid_cluster_names[cluster_id],
        fontsize=10,
        weight='bold',
        color='black'
    )

plt.title("UMAP Projection of Midfielder Clusters")
plt.legend(title="Midfielder Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

################ EXAMPLE PLAYERS PER CLUSTER ################

print("\nExample midfielders per cluster:\n")

for label in df_mid_model['mid_cluster_label'].unique():
    print(f"{label}:")
    print(df_mid_model.loc[df_mid_model['mid_cluster_label'] == label, 'name'].head(5).to_list())
    print()

### save midfielders for dashboard ###
df_mid_model.to_csv("midfielders_similarity_dataset.csv", index=False)
print("Saved: midfielders_similarity_dataset.csv")