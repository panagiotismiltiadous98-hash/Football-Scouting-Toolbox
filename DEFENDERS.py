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

###################################### DEFENDER CLUSTERING PIPELINE ############################################

################ FILTER DEFENDER GROUP ################

# DF + DF/MF
df_def = df[df['position_universal'].isin(['DF'])].copy()

print("\nDefender group size:", len(df_def))
print(df_def['position_universal'].value_counts())

################ SELECT DEFENDER FEATURES ################

def_features = [

    # Defending core
    'defending_defensive_awareness',
    'defending_standing_tackle',
    'defending_sliding_tackle',

    # Physical
    'power_strength',
    'power_jumping',

    # Movement
    'movement_acceleration',
    'movement_sprint_speed',
    'movement_agility',
    'movement_reactions',
    'movement_balance',

    # Passing / build-up
    'attacking_short_passing',
    'skill_long_passing',
    'skill_ball_control',

    # Mental
    'mentality_interceptions',
    'mentality_aggression',
    'mentality_composure',
    'mentality_vision',

    # Crossing (important for full-backs)
    'attacking_crossing',

    # Extra
    'weak_foot',
    'skill_moves',
    'height_cm',
    'weight_kg',
    'age',
    'num_positions'
]

# Keep only existing columns
def_features = [c for c in def_features if c in df_def.columns]

# Metadata
def_metadata = ['player_id', 'name', 'positions', 'position_universal', 'club_name',
                'overall_rating', 'potential', 'value', 'age']
def_metadata = [c for c in def_metadata if c in df_def.columns]

# Modelling dataframe
df_def_model = df_def[def_metadata + def_features].copy()

print("\nDefender modelling dataframe shape:", df_def_model.shape)
print("\nDefender features used:")
print(def_features)

################ EXPLORATORY CHECK ################

def_numeric = df_def_model.select_dtypes(include=np.number).drop(columns=['player_id'], errors='ignore')

print("\nAverage defender feature values:")
print(def_numeric.mean().sort_values(ascending=False))

print("\nDefender feature variability:")
print(def_numeric.std().sort_values(ascending=False))

################ SCALE DEFENDER FEATURES ################

scaler = StandardScaler()
def_scaled = scaler.fit_transform(def_numeric)

def_scaled_df = pd.DataFrame(def_scaled, columns=def_numeric.columns, index=df_def_model.index)

print("\nScaled defender data sample:")
print(def_scaled_df.head())

################ K-MEANS TUNING ################

inertia = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(def_scaled_df)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(def_scaled_df, cluster_labels))

print("\nK values tested:", list(k_values))
print("Inertia values:", inertia)
print("Silhouette scores:", silhouette_scores)

################ ELBOW METHOD ################

plt.figure(figsize=(8,5))
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method for Defender K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()

################ SILHOUETTE ################

plt.figure(figsize=(8,5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Defender K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()

################ FINAL MODEL ################

best_k = 5  # we will validate after plots

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_def_model['def_cluster'] = kmeans_final.fit_predict(def_scaled_df)

print("\nDefender cluster counts:")
print(df_def_model['def_cluster'].value_counts().sort_index())

################ CLUSTER LABELS ################

def_cluster_names = {
    0: "Modern All-Round Defender",
    1: "Wide Defender Profile",
    2: "Technical Elite Center-Back",
    3: "Experienced Center-Back",
    4: "Young-Balanced Defender"
}

df_def_model['def_cluster_label'] = df_def_model['def_cluster'].map(def_cluster_names)

print("\nDefender cluster distribution:")
print(df_def_model['def_cluster_label'].value_counts())

################ CLUSTER PROFILE ################

cluster_summary = df_def_model.groupby('def_cluster_label')[def_numeric.columns].mean()

print("\nDefender cluster feature averages:")
print(cluster_summary)

################ UMAP VISUALISATION ################

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(def_scaled_df)

umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=df_def_model.index)
umap_df['cluster'] = df_def_model['def_cluster']
umap_df['cluster_label'] = df_def_model['def_cluster_label']

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

# Labels on centroids
# Custom offsets per cluster to avoid overlap
offsets = {
    0: (0.3, 0.3),
    1: (0.3, -0.5),
    2: (-2.0, 0.3),
    3: (0.3, 0.3),
    4: (0.3, -0.5),
}

for cluster_id, (x, y) in enumerate(centroids_2d):
    ox, oy = offsets.get(cluster_id, (0.3, 0.3))
    ax.text(
        x + ox,
        y + oy,
        def_cluster_names[cluster_id],
        fontsize=10,
        weight='bold'
    )

plt.title("UMAP Projection of Defender Clusters")
plt.legend(title="Defender Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

################ EXAMPLES ################

print("\nExample defenders per cluster:\n")

for label in df_def_model['def_cluster_label'].unique():
    print(f"{label}:")
    print(df_def_model.loc[df_def_model['def_cluster_label'] == label, 'name'].head(5).to_list())
    print()

### CLUSTER TABLE PROFILE ###
selected_features = [
    'movement_sprint_speed',
    'attacking_crossing',
    'mentality_aggression',
    'defending_defensive_awareness'
]

print("\nFeature availability check:")
for col in selected_features:
    print(f"{col}: {'✅' if col in df_def_model.columns else '❌'}")

selected_features = [c for c in selected_features if c in df_def_model.columns]

cluster_profile_selected = (
    df_def_model
    .groupby('def_cluster_label')[selected_features]
    .mean()
    .round(2)
)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
print("\nDefender cluster feature averages (selected features):")
print(cluster_profile_selected)

### DEFENDERS SAVED FOR DASHBOARD ##
df_def_model.to_csv("defenders_similarity_dataset.csv", index=False)
print("Saved: defenders_similarity_dataset.csv")