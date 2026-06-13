import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap

################ LOAD GK MODEL DATASET ################
df = pd.read_csv("goalkeepers_model_dataset.csv", encoding="utf-8")

################ GK EXPLORATORY ANALYSIS ################

### 1) AVG ###

# Select ONLY numeric modelling features (EXCLUDE identifier)
gk_numeric = df.select_dtypes(include=np.number).drop(columns=['player_id'], errors='ignore')

# Mean profile
gk_mean = gk_numeric.mean().sort_values(ascending=False)

print("\nAverage Goalkeeper Attribute Values:")
print(gk_mean)


### 2) STD ###

gk_std = gk_numeric.std().sort_values(ascending=False)

print("\nGoalkeeper Attribute Variability:")
print(gk_std)

### 3) AVG PLOT
plt.figure(figsize=(10,6))

gk_mean.sort_values().plot(kind='barh')

plt.title("Average Goalkeeper Attribute Profile")
plt.xlabel("Average Value")
plt.ylabel("Attribute")

plt.tight_layout()
plt.show()

### 4) Core GK Attribute Boxplot
key_gk = [
'goalkeeping_gk_diving',
'goalkeeping_gk_handling',
'goalkeeping_gk_kicking',
'goalkeeping_gk_reflexes'
]

plt.figure(figsize=(10,6))

sns.boxplot(data=df[key_gk])

plt.title("Distribution of Core Goalkeeping Attributes")

plt.xticks(
    ticks=range(len(key_gk)),
    labels=['Diving','Handling','Kicking','Reflexes'],
    rotation=0
)

plt.tight_layout()
plt.show()

################ GK CORRELATION HEATMAP ################

gk_numeric = df.select_dtypes(include=np.number).drop(columns=['player_id'], errors='ignore')

gk_corr = gk_numeric.corr()

# Create short labels mapping
short_names = {
'goalkeeping_gk_diving': 'Diving',
'goalkeeping_gk_handling': 'Handling',
'goalkeeping_gk_kicking': 'Kicking',
'goalkeeping_gk_positioning': 'Positioning',
'goalkeeping_gk_reflexes': 'Reflexes',
'attacking_short_passing': 'ShortPass',
'mentality_vision': 'Vision',
'mentality_composure': 'Composure',
'movement_reactions': 'Reactions',
'movement_agility': 'Agility',
'power_jumping': 'Jumping',
'power_strength': 'Strength',
'power_shot_power': 'ShotPower',
'movement_balance': 'Balance',
'weak_foot': 'WeakFoot',
'height_cm': 'Height',
'weight_kg': 'Weight',
'age': 'Age'
}

# Rename for plotting only
gk_corr_renamed = gk_corr.rename(index=short_names, columns=short_names)

plt.figure(figsize=(10,8))

sns.heatmap(
    gk_corr_renamed,
    cmap="coolwarm",
    annot=False,
    square=True,
    linewidths=0.5
)

plt.title("Goalkeeper Feature Correlation Heatmap")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

################ SCALE GK FEATURES ################

# Select numeric features (exclude ID)
gk_numeric = df.select_dtypes(include=np.number).drop(columns=['player_id'], errors='ignore')

# Initialize scaler
scaler = StandardScaler()

# Fit and transform
gk_scaled = scaler.fit_transform(gk_numeric)

# keeping original index
gk_scaled_df = pd.DataFrame(gk_scaled, columns=gk_numeric.columns, index=df.index)

print("\nScaled GK data sample:")
print(gk_scaled_df.head())

################ K-MEANS TUNING FOR GK DATASET ################

inertia = []
silhouette_scores = []
k_values = range(2, 11)   # test K from 2 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(gk_scaled_df)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(gk_scaled_df, cluster_labels))

print("\nK values tested:", list(k_values))
print("Inertia values:", inertia)
print("Silhouette scores:", silhouette_scores)

################ ELBOW METHOD PLOT ################

plt.figure(figsize=(8,5))
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method for Goalkeeper K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()

################ SILHOUETTE SCORE PLOT ################

plt.figure(figsize=(8,5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Goalkeeper K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.tight_layout()
plt.show()

################ FINAL GK K-MEANS MODEL (4) ################

best_k = 4

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)

df['gk_cluster'] = kmeans_final.fit_predict(gk_scaled_df)

print("\nGoalkeeper cluster counts:")
print(df['gk_cluster'].value_counts().sort_index())

################ GK CLUSTER PROFILE ################

cluster_summary = df.groupby('gk_cluster')[gk_numeric.columns].mean()

print("\nCluster feature averages:")
print(cluster_summary)


################ UMAP VISUALISATION ################


cluster_names = {
    0: "Young-Developing GK",
    1: "Prime Balanced GK",
    2: "Elite GK",
    3: "Experienced GK"
}

df['gk_cluster_label'] = df['gk_cluster'].map(cluster_names)

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(gk_scaled_df)

umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=df.index)
umap_df['cluster'] = df['gk_cluster']
umap_df['cluster_label'] = df['gk_cluster_label']

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
    marker='X'
)

# Add text labels at centroids
for cluster_id, (x, y) in enumerate(centroids_2d):
    ax.text(
        x + 0.08,
        y + 0.08,
        cluster_names[cluster_id],
        fontsize=10,
        weight='bold',
        color='black'
    )

plt.title("Projection of Goalkeeper Clusters")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

########## ADDING LABELS TO CLUSTERS #############
cluster_names = {
    0: "Young-Developing GK",
    1: "Prime Balanced GK",
    2: "Elite GK",
    3: "Experienced GK"
}

df['gk_cluster_label'] = df['gk_cluster'].map(cluster_names)

## Count per Label ##
print(df['gk_cluster_label'].value_counts())

## EXAMPLES ##
print("\nExample goalkeepers per cluster:\n")

for label in df['gk_cluster_label'].unique():
    print(f"{label}:")
    print(df.loc[df['gk_cluster_label'] == label, 'name'].head(5).to_list())
    print()

# Merge overall_rating, potential, value from player_stats
player_stats = pd.read_csv("player_stats.csv", usecols=['name', 'overall_rating', 'potential', 'value'])
player_stats = player_stats.drop_duplicates(subset='name')
df = df.merge(player_stats, on='name', how='left')

### SAVE FOR DASHBOARD BACKEND ##
df.to_csv("goalkeepers_similarity_dataset.csv", index=False)
print("Saved: goalkeepers_similarity_dataset.csv")