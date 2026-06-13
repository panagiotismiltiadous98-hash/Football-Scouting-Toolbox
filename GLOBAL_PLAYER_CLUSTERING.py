import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap

############################################################
# LOAD DATA
############################################################
df = pd.read_csv("outfield_model_dataset.csv", encoding="utf-8")

############################################################
# BASIC CLEANING
############################################################

df = df.drop(columns=['international_reputation'], errors='ignore')

# Normalize height & weight
if 'height_cm' in df.columns:
    df['height_cm'] = 100 * (df['height_cm'] - df['height_cm'].min()) / (df['height_cm'].max() - df['height_cm'].min())

if 'weight_kg' in df.columns:
    df['weight_kg'] = 100 * (df['weight_kg'] - df['weight_kg'].min()) / (df['weight_kg'].max() - df['weight_kg'].min())

############################################################
# REMOVE GOALKEEPERS
############################################################

df = df[df['position_universal'] != 'GK'].copy()

print("\nTotal outfield players:", len(df))

############################################################
# FEATURE SELECTION (GLOBAL)
############################################################

features = [

    # Attacking
    'attacking_finishing',
    'attacking_crossing',
    'attacking_short_passing',

    # Skill
    'skill_dribbling',
    'skill_ball_control',
    'skill_long_passing',

    # Movement
    'movement_acceleration',
    'movement_sprint_speed',
    'movement_agility',
    'movement_reactions',

    # Power
    'power_strength',
    'power_stamina',
    'power_long_shots',

    # Mental
    'mentality_vision',
    'mentality_aggression',
    'mentality_composure',
    'mentality_interceptions',

    # Defending
    'defending_defensive_awareness',
    'defending_standing_tackle',

    # Extra
    'weak_foot',
    'skill_moves',
    'height_cm',
    'weight_kg',
    'age'
]

# Keep only existing columns
features = [c for c in features if c in df.columns]

metadata = ['player_id', 'name', 'positions', 'club_name']
metadata = [c for c in metadata if c in df.columns]

df_model = df[metadata + features].copy()

############################################################
# NUMERIC DATA
############################################################

numeric = df_model.select_dtypes(include=np.number).drop(columns=['player_id'], errors='ignore')

############################################################
# SCALING
############################################################

scaler = StandardScaler()
scaled = scaler.fit_transform(numeric)
scaled_df = pd.DataFrame(scaled, columns=numeric.columns, index=df_model.index)

############################################################
# K SELECTION
############################################################

k_values = range(3, 9)
inertia = []
silhouette = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_df)

    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(scaled_df, labels))

plt.figure()
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.show()

plt.figure()
plt.plot(k_values, silhouette, marker='o')
plt.title("Silhouette Score")
plt.xlabel("K")
plt.ylabel("Score")
plt.show()

############################################################
# FINAL MODEL
############################################################

best_k = 6  # adjust after plots

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_model['cluster'] = kmeans.fit_predict(scaled_df)

print("\nCluster distribution:")
print(df_model['cluster'].value_counts())

############################################################
# CLUSTER PROFILE
############################################################

cluster_profile = df_model.groupby('cluster')[numeric.columns].mean().round(2)

pd.set_option('display.max_columns', None)
print("\nCluster feature averages:")
print(cluster_profile)

############################################################
# UMAP VISUALIZATION
############################################################

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(scaled_df)

umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=df_model.index)

cluster_names = {
    0: "Target Finishers",
    1: "Wide Defenders",
    2: "Defensive Players",
    3: "Elite Centre-Backs",
    4: "Elite Technical Players",
    5: "Explosive Attackers"
}

# Add numeric cluster and label
umap_df['cluster'] = df_model['cluster']
umap_df['cluster_label'] = df_model['cluster'].map(cluster_names)

# Also store labels in main dataframe
df_model['cluster_label'] = df_model['cluster'].map(cluster_names)

plt.figure(figsize=(9,7))
sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='cluster_label',
    palette='tab10',
    s=50
)

plt.title("UMAP Projection of Global Player Clusters")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

############################################################
# EXAMPLES
############################################################

print("\nExample players per cluster:\n")

for c in sorted(df_model['cluster'].unique()):
    label = cluster_names.get(c, f"Cluster {c}")
    print(f"{label}:")
    print(df_model[df_model['cluster'] == c]['name'].head(10).to_list())
    print()