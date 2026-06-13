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

#### POSITION RELATED COLUMNS ############
print("\nPosition-related columns:")
print([c for c in df.columns if 'position' in c.lower()])

#### raw positions ##########
print("\nSample raw positions:")
print(df['positions'].dropna().unique()[:20])

### GROUPED VERSION ####
print("\nPosition group distribution:")
print(df['position_group'].value_counts())

########## UNIVERSAL VERSION #########
print("\nUniversal positions:")
print(df['position_universal'].value_counts())

######### unique position ####
all_positions = set()

for pos in df['positions'].dropna():
    for p in pos.split(','):
        all_positions.add(p.strip())

print("\nAll unique position tokens:")
print(sorted(all_positions))

####### POSITION SUBGROUPS ########
def assign_subgroup(pos):

    pos = str(pos)

    # DEFENDERS
    if any(p in pos for p in ['CB']):
        return 'CB'

    elif any(p in pos for p in ['RB','LB']):
        return 'FB'

    # MIDFIELDERS
    elif any(p in pos for p in ['CDM']):
        return 'CDM'

    elif any(p in pos for p in ['CM']):
        return 'CM'

    elif any(p in pos for p in ['CAM']):
        return 'CAM'

    elif any(p in pos for p in ['RM','LM']):
        return 'WIDE_MF'

    # ATTACKERS
    elif any(p in pos for p in ['RW','LW']):
        return 'WINGER'

    elif any(p in pos for p in ['ST']):
        return 'ST'



df['position_subgroup'] = df['positions'].apply(assign_subgroup)

print(df['position_subgroup'].value_counts())

###### cb models##
df_cb = df[df['position_subgroup'] == 'CB']


    ####################################### CENTRE-BACK CLUSTERING PIPELINE ############################################

    ################ FILTER CB GROUP ################

    df_cb = df[
        (df['position_universal'].isin(['DF'])) &
        (df['defender_subgroup'] == 'CB')
        ].copy()

    print("\nCentre-Back group size:", len(df_cb))
    print(df_cb['defender_subgroup'].value_counts())

    ################ SELECT CB FEATURES ################

    cb_features = [

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

        # Extra
        'weak_foot',
        'skill_moves',
        'height_cm',
        'weight_kg',
        'age',
        'num_positions'
    ]

    # Keep only existing columns
    cb_features = [c for c in cb_features if c in df_cb.columns]

    # Metadata
    cb_metadata = ['player_id', 'name', 'positions', 'position_universal', 'defender_subgroup', 'club_name']
    cb_metadata = [c for c in cb_metadata if c in df_cb.columns]

    # Modelling dataframe
    df_cb_model = df_cb[cb_metadata + cb_features].copy()

    print("\nCentre-Back modelling dataframe shape:", df_cb_model.shape)
    print("\nCentre-Back features used:")
    print(cb_features)

    ################ EXPLORATORY CHECK ################

    cb_numeric = df_cb_model.select_dtypes(include=np.number).drop(columns=['player_id'], errors='ignore')

    print("\nAverage centre-back feature values:")
    print(cb_numeric.mean().sort_values(ascending=False))

    print("\nCentre-back feature variability:")
    print(cb_numeric.std().sort_values(ascending=False))

    ################ SCALE CB FEATURES ################

    scaler_cb = StandardScaler()
    cb_scaled = scaler_cb.fit_transform(cb_numeric)

    cb_scaled_df = pd.DataFrame(cb_scaled, columns=cb_numeric.columns, index=df_cb_model.index)

    print("\nScaled centre-back data sample:")
    print(cb_scaled_df.head())

    ################ K-MEANS TUNING ################

    cb_inertia = []
    cb_silhouette_scores = []
    cb_k_values = range(2, 11)

    for k in cb_k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cb_scaled_df)

        cb_inertia.append(kmeans.inertia_)
        cb_silhouette_scores.append(silhouette_score(cb_scaled_df, cluster_labels))

    print("\nCB K values tested:", list(cb_k_values))
    print("CB inertia values:", cb_inertia)
    print("CB silhouette scores:", cb_silhouette_scores)

    ################ ELBOW METHOD ################

    plt.figure(figsize=(8, 5))
    plt.plot(cb_k_values, cb_inertia, marker='o')
    plt.title("Elbow Method for Centre-Back K-Means")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(list(cb_k_values))
    plt.tight_layout()
    plt.show()

    ################ SILHOUETTE ################

    plt.figure(figsize=(8, 5))
    plt.plot(cb_k_values, cb_silhouette_scores, marker='o')
    plt.title("Silhouette Scores for Centre-Back K-Means")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(list(cb_k_values))
    plt.tight_layout()
    plt.show()

    ################ FINAL MODEL ################

    best_k_cb = 3  # adjust after elbow + silhouette + football interpretation

    kmeans_cb_final = KMeans(n_clusters=best_k_cb, random_state=42, n_init=10)
    df_cb_model['cb_cluster'] = kmeans_cb_final.fit_predict(cb_scaled_df)

    print("\nCentre-back cluster counts:")
    print(df_cb_model['cb_cluster'].value_counts().sort_index())

    ################ CLUSTER LABELS ################

    cb_cluster_names = {
        0: "Ball-Playing Centre-Back",
        1: "Defensive Centre-Back",
        2: "Physical Centre-Back"
    }

    df_cb_model['cb_cluster_label'] = df_cb_model['cb_cluster'].map(cb_cluster_names)

    print("\nCentre-back cluster distribution:")
    print(df_cb_model['cb_cluster_label'].value_counts())

    ################ CLUSTER PROFILE ################

    cb_cluster_summary = df_cb_model.groupby('cb_cluster_label')[cb_numeric.columns].mean()

    print("\nCentre-back cluster feature averages:")
    print(cb_cluster_summary)

    ################ UMAP VISUALISATION ################

    cb_reducer = umap.UMAP(n_components=2, random_state=42)
    cb_embedding = cb_reducer.fit_transform(cb_scaled_df)

    cb_umap_df = pd.DataFrame(cb_embedding, columns=['UMAP1', 'UMAP2'], index=df_cb_model.index)
    cb_umap_df['cluster'] = df_cb_model['cb_cluster']
    cb_umap_df['cluster_label'] = df_cb_model['cb_cluster_label']

    cb_centroids = kmeans_cb_final.cluster_centers_
    cb_centroids_2d = cb_reducer.transform(cb_centroids)

    plt.figure(figsize=(9, 7))

    ax = sns.scatterplot(
        data=cb_umap_df,
        x='UMAP1',
        y='UMAP2',
        hue='cluster_label',
        palette='Set2',
        s=60
    )

    ax.scatter(
        cb_centroids_2d[:, 0],
        cb_centroids_2d[:, 1],
        c='black',
        s=220,
        marker='X',
        label='Centroids'
    )

    for cluster_id, (x, y) in enumerate(cb_centroids_2d):
        ax.text(
            x + 0.08,
            y + 0.08,
            cb_cluster_names[cluster_id],
            fontsize=10,
            weight='bold'
        )

    plt.title("UMAP Projection of Centre-Back Clusters")
    plt.legend(title="Centre-Back Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    ################ EXAMPLES ################

    print("\nExample centre-backs per cluster:\n")

    for label in df_cb_model['cb_cluster_label'].unique():
        print(f"{label}:")
        print(df_cb_model.loc[df_cb_model['cb_cluster_label'] == label, 'name'].head(5).to_list())
        print()

    ####################################### FULL-BACK CLUSTERING PIPELINE ############################################

    ################ FILTER FB GROUP ################

    df_fb = df[
        (df['position_universal'].isin(['DF'])) &
        (df['defender_subgroup'] == 'FB')
        ].copy()

    print("\nFull-Back group size:", len(df_fb))
    print(df_fb['defender_subgroup'].value_counts())

    ################ SELECT FB FEATURES ################

    fb_features = [

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

        # Crossing
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
    fb_features = [c for c in fb_features if c in df_fb.columns]

    # Metadata
    fb_metadata = ['player_id', 'name', 'positions', 'position_universal', 'defender_subgroup', 'club_name']
    fb_metadata = [c for c in fb_metadata if c in df_fb.columns]

    # Modelling dataframe
    df_fb_model = df_fb[fb_metadata + fb_features].copy()

    print("\nFull-Back modelling dataframe shape:", df_fb_model.shape)
    print("\nFull-Back features used:")
    print(fb_features)

    ################ EXPLORATORY CHECK ################

    fb_numeric = df_fb_model.select_dtypes(include=np.number).drop(columns=['player_id'], errors='ignore')

    print("\nAverage full-back feature values:")
    print(fb_numeric.mean().sort_values(ascending=False))

    print("\nFull-back feature variability:")
    print(fb_numeric.std().sort_values(ascending=False))

    ################ SCALE FB FEATURES ################

    scaler_fb = StandardScaler()
    fb_scaled = scaler_fb.fit_transform(fb_numeric)

    fb_scaled_df = pd.DataFrame(fb_scaled, columns=fb_numeric.columns, index=df_fb_model.index)

    print("\nScaled full-back data sample:")
    print(fb_scaled_df.head())

    ################ K-MEANS TUNING ################

    fb_inertia = []
    fb_silhouette_scores = []
    fb_k_values = range(2, 11)

    for k in fb_k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(fb_scaled_df)

        fb_inertia.append(kmeans.inertia_)
        fb_silhouette_scores.append(silhouette_score(fb_scaled_df, cluster_labels))

    print("\nFB K values tested:", list(fb_k_values))
    print("FB inertia values:", fb_inertia)
    print("FB silhouette scores:", fb_silhouette_scores)

    ################ ELBOW METHOD ################

    plt.figure(figsize=(8, 5))
    plt.plot(fb_k_values, fb_inertia, marker='o')
    plt.title("Elbow Method for Full-Back K-Means")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(list(fb_k_values))
    plt.tight_layout()
    plt.show()

    ################ SILHOUETTE ################

    plt.figure(figsize=(8, 5))
    plt.plot(fb_k_values, fb_silhouette_scores, marker='o')
    plt.title("Silhouette Scores for Full-Back K-Means")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(list(fb_k_values))
    plt.tight_layout()
    plt.show()

    ################ FINAL MODEL ################

    best_k_fb = 3  # adjust after elbow + silhouette + football interpretation

    kmeans_fb_final = KMeans(n_clusters=best_k_fb, random_state=42, n_init=10)
    df_fb_model['fb_cluster'] = kmeans_fb_final.fit_predict(fb_scaled_df)

    print("\nFull-back cluster counts:")
    print(df_fb_model['fb_cluster'].value_counts().sort_index())

    ################ CLUSTER LABELS ################

    fb_cluster_names = {
        0: "Attacking Full-Back",
        1: "Balanced Full-Back",
        2: "Defensive Full-Back"
    }

    df_fb_model['fb_cluster_label'] = df_fb_model['fb_cluster'].map(fb_cluster_names)

    print("\nFull-back cluster distribution:")
    print(df_fb_model['fb_cluster_label'].value_counts())

    ################ CLUSTER PROFILE ################

    fb_cluster_summary = df_fb_model.groupby('fb_cluster_label')[fb_numeric.columns].mean()

    print("\nFull-back cluster feature averages:")
    print(fb_cluster_summary)

    ################ UMAP VISUALISATION ################

    fb_reducer = umap.UMAP(n_components=2, random_state=42)
    fb_embedding = fb_reducer.fit_transform(fb_scaled_df)

    fb_umap_df = pd.DataFrame(fb_embedding, columns=['UMAP1', 'UMAP2'], index=df_fb_model.index)
    fb_umap_df['cluster'] = df_fb_model['fb_cluster']
    fb_umap_df['cluster_label'] = df_fb_model['fb_cluster_label']

    fb_centroids = kmeans_fb_final.cluster_centers_
    fb_centroids_2d = fb_reducer.transform(fb_centroids)

    plt.figure(figsize=(9, 7))

    ax = sns.scatterplot(
        data=fb_umap_df,
        x='UMAP1',
        y='UMAP2',
        hue='cluster_label',
        palette='Set2',
        s=60
    )

    ax.scatter(
        fb_centroids_2d[:, 0],
        fb_centroids_2d[:, 1],
        c='black',
        s=220,
        marker='X',
        label='Centroids'
    )

    for cluster_id, (x, y) in enumerate(fb_centroids_2d):
        ax.text(
            x + 0.08,
            y + 0.08,
            fb_cluster_names[cluster_id],
            fontsize=10,
            weight='bold'
        )

    plt.title("UMAP Projection of Full-Back Clusters")
    plt.legend(title="Full-Back Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    ################ EXAMPLES ################

    print("\nExample full-backs per cluster:\n")

    for label in df_fb_model['fb_cluster_label'].unique():
        print(f"{label}:")
        print(df_fb_model.loc[df_fb_model['fb_cluster_label'] == label, 'name'].head(5).to_list())
        print()

    ################ TOP 30 CENTRE-BACKS ################

    cols = ['name', 'positions', 'club_name', 'overall', 'age']
    cols = [c for c in cols if c in df_cb.columns]

    # sort if possible
    if 'overall' in df_cb.columns:
        df_cb_sorted = df_cb.sort_values(by='overall', ascending=False)
    else:
        df_cb_sorted = df_cb.copy()

    print("\nTop 30 Centre-Backs:\n")
    print(df_cb_sorted[cols].head(30).to_string(index=False))

    ################ TOP 30 FULL-BACKS ################

    cols = ['name', 'positions', 'club_name', 'overall', 'age']
    cols = [c for c in cols if c in df_fb.columns]

    # sort if possible
    if 'overall' in df_fb.columns:
        df_fb_sorted = df_fb.sort_values(by='overall', ascending=False)
    else:
        df_fb_sorted = df_fb.copy()

    print("\nTop 30 Full-Backs:\n")
    print(df_fb_sorted[cols].head(30).to_string(index=False))

##### SECOND PIPELINE OF DEFENDERS######
####################################### DEFENDER SUBGROUP COLUMN ############################################

def assign_subgroup(pos):

    pos = str(pos).upper()

    # CENTRE-BACKS
    if any(p in pos for p in ['CB', 'LCB', 'RCB']):
        return 'CB'

    # FULL-BACKS / WING-BACKS
    elif any(p in pos for p in ['RB', 'LB', 'RWB', 'LWB']):
        return 'FB'

    else:
        return 'Other'

# Create subgroup column
df['defender_subgroup'] = df['positions'].apply(assign_subgroup)

print("\nDefender subgroup distribution:")
print(df['defender_subgroup'].value_counts())


####################################### DEFENDER CLUSTERING PIPELINE ############################################

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
def_metadata = ['player_id', 'name', 'positions', 'position_universal', 'club_name']
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

best_k = 4  # we will validate after plots

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_def_model['def_cluster'] = kmeans_final.fit_predict(def_scaled_df)

print("\nDefender cluster counts:")
print(df_def_model['def_cluster'].value_counts().sort_index())

################ CLUSTER LABELS ################

def_cluster_names = {
    0: "Ball-Playing Defender",
    1: "Defensive Anchor",
    2: "Modern Full-Back",
    3: "Physical Centre-Back"
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
for cluster_id, (x, y) in enumerate(centroids_2d):
    ax.text(
        x + 0.08,
        y + 0.08,
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

