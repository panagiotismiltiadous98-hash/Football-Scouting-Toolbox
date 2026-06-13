import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ─── LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv("player_stats.csv", encoding="utf-8")

# ─── CLEAN + PARSE DOB ──────────────────────────────────────────────────────
df['dob'] = df['dob'].replace(['###', '####', '#####', ''], np.nan)
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

cols = ['player_id', 'name', 'version', 'club_name', 'dob', 'age']
cols = [c for c in cols if c in df.columns]
missing = df.loc[df['dob'].isna(), cols]
print(missing)
print("Available columns:", df.columns.tolist())

# Fix Torrico's missing DOB
df.loc[df['player_id'] == 274195, 'dob'] = pd.to_datetime('2004-08-10')

# Create age from DOB
today = pd.Timestamp.today().normalize()
df['age'] = (today - df['dob']).dt.days // 365

# Remove duplicates (one row per player_id per version)
df = df.drop_duplicates(subset=['player_id', 'version'], keep='first')

print("Rows:", len(df))
print("Duplicate player_id+version:", df.duplicated(subset=['player_id', 'version']).sum())
print("Missing DOB:", df['dob'].isna().sum())
print("Missing Age:", df['age'].isna().sum())

# ─── DATASET SHAPE & TYPES ──────────────────────────────────────────────────
print("\nDataset shape (rows, columns):")
print(df.shape)

missing_pct = df.isna().mean().sort_values(ascending=False)
print("\nMissing values percentage (non-zero only):")
print(missing_pct[missing_pct > 0] * 100)

print("\nCount of columns by dtype:")
print(df.dtypes.value_counts())

# ─── CLEAN COLUMNS ──────────────────────────────────────────────────────────
url_cols = sorted(set(
    [c for c in df.columns if c.lower() in {"url", "image", "club_logo", "country_flag"}] +
    [c for c in df.columns if any(k in c.lower() for k in ["url", "logo", "flag", "image"])]
))

def clean_url_series(s):
    s = s.astype("string").str.strip()
    s = s.replace(["", "nan", "NaN", "None", "NULL", "null", "###", "####", "#####"], pd.NA)
    s = s.where(s.str.startswith(("http://", "https://", "/")), other=pd.NA)
    return s

for c in url_cols:
    df[c] = clean_url_series(df[c])

drop_cols = [
    'country_position', 'country_name', 'country_kit_number',
    'country_rating', 'country_id', 'country_flag',
    'country_league_name', 'country_league_id', 'specialities'
]
to_drop = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=to_drop)
print("Actually dropped:", to_drop)

if 'play_styles' in df.columns:
    df['play_styles'] = (df['play_styles']
                         .astype("string").str.strip()
                         .replace(["", "nan", "NaN", "None", "NULL", "null"], pd.NA)
                         .str.replace(r"\s*,\s*", "|", regex=True))

if 'release_clause' in df.columns:
    rc = df['release_clause'].astype("string").str.strip()
    rc = rc.replace(["", "nan", "NaN", "None", "NULL", "null"], pd.NA)
    rc = rc.str.replace(r"[^\d\.KMB]", "", regex=True)

    def parse_money(x):
        if pd.isna(x): return np.nan
        x = str(x)
        mult = 1
        if x.endswith("K"): mult, x = 1e3, x[:-1]
        elif x.endswith("M"): mult, x = 1e6, x[:-1]
        elif x.endswith("B"): mult, x = 1e9, x[:-1]
        try: return float(x) * mult
        except: return np.nan

    df['release_clause'] = rc.apply(parse_money)

for c in ['height_cm', 'weight_kg']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

missing = df.isna().mean().sort_values(ascending=False) * 100
print("\nMissing % (top 20):")
print(missing.head(30))

# ─── DESCRIPTIVE STATISTICS — WHOLE DATASET ─────────────────────────────────
desc_cols = ['age', 'height_cm', 'weight_kg']
for c in desc_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

summary = pd.DataFrame({
    'Mean':    df[desc_cols].mean(),
    'Std Dev': df[desc_cols].std(),
    'Min':     df[desc_cols].min(),
    'Median':  df[desc_cols].median(),
    'Max':     df[desc_cols].max()
})
print("\n#################### CUSTOM DESCRIPTIVE SUMMARY ####################")
print(summary)

# ─── POSITION GROUPING ───────────────────────────────────────────────────────
df['positions'] = df['positions'].astype(str)

def assign_position_group(pos):
    if 'GK' in pos:
        return 'Goalkeeper'
    elif any(p in pos for p in ['CB', 'RB', 'LB', 'RWB', 'LWB']):
        return 'Defender'
    elif any(p in pos for p in ['CM', 'CDM', 'CAM', 'RM', 'LM']):
        return 'Midfielder'
    elif any(p in pos for p in ['ST', 'CF', 'RW', 'LW']):
        return 'Attacker'
    else:
        return 'Other'

df['position_group'] = df['positions'].apply(assign_position_group)
print("\nPosition group counts:")
print(df['position_group'].value_counts())

# ─── DESCRIPTIVE STATISTICS BY POSITION GROUP ────────────────────────────────
group_summary = df.groupby('position_group')[['age', 'height_cm', 'weight_kg']].agg(
    ['mean', 'std', 'min', 'median', 'max']
)
print("\n################ POSITION GROUP DESCRIPTIVE SUMMARY ################")
print(group_summary)

# Overall rating by position group
overall_by_group = df.groupby('position_group')['overall_rating'].agg(
    ['mean', 'std', 'min', 'median', 'max']
).round(1)
print("\n################ OVERALL RATING BY POSITION GROUP ################")
print(overall_by_group)

# ─── VISUALISATIONS ──────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='position_group', y='age')
plt.title("Age Distribution by Position Group")
plt.xlabel("Position Group")
plt.ylabel("Age")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='position_group', y='height_cm')
plt.title("Height Distribution by Position Group")
plt.xlabel("Position Group")
plt.ylabel("Height (cm)")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='position_group', y='weight_kg')
plt.title("Weight Distribution by Position Group")
plt.xlabel("Position Group")
plt.ylabel("Weight (kg)")
plt.show()

plt.figure(figsize=(8, 5))
counts = df['position_group'].value_counts()
bars = plt.bar(counts.index, counts.values)
plt.title("Number of Players per Position Group", fontsize=14)
plt.xlabel("Position Group", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=30)
for i, v in enumerate(counts.values):
    plt.text(i, v + 50, str(v), ha='center')
plt.tight_layout()
plt.show()

# ─── OVERALL RATING VISUALISATION ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Histogram
ax1.hist(df['overall_rating'].dropna(), bins=35, color='#1F4E79', edgecolor='white', alpha=0.85)
ax1.axvline(df['overall_rating'].mean(), color='#FFD700', linewidth=2.5,
            linestyle='--', label=f"Mean: {df['overall_rating'].mean():.1f}", zorder=5)
ax1.axvline(df['overall_rating'].median(), color='#FF6B6B', linewidth=2.5,
            linestyle='-', label=f"Median: {df['overall_rating'].median():.1f}", zorder=4)
ax1.set_title("Overall Rating Distribution — All Players", fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel("Overall Rating", fontsize=12)
ax1.set_ylabel("Number of Players", fontsize=12)
ax1.legend(fontsize=11)

# Right: Boxplot by position group
colors = ['#2E75B6', '#70AD47', '#ED7D31', '#FFC000']
groups = ['Attacker', 'Midfielder', 'Defender', 'Goalkeeper']
data_by_group = [df[df['position_group'] == g]['overall_rating'].dropna() for g in groups]

bp = ax2.boxplot(data_by_group, patch_artist=True, labels=groups,
                 medianprops=dict(color='white', linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax2.set_title("Overall Rating by Position Group", fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel("Position Group", fontsize=12)
ax2.set_ylabel("Overall Rating", fontsize=12)

plt.suptitle("Overall Rating Overview", fontsize=15,
             fontweight='bold', y=0.98)
plt.tight_layout(pad=3.0)
plt.savefig("overall_rating_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as overall_rating_plot.png")

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────
df['positions'] = df['positions'].astype(str)
df['num_positions'] = df['positions'].apply(
    lambda x: len([p.strip() for p in x.split(',')])
)
df['multi_positioned'] = df['num_positions'].apply(
    lambda x: 'YES' if x >= 2 else 'NO'
)
print("\nMulti-positioned distribution:")
print(df['multi_positioned'].value_counts())

# ─── UNIVERSAL POSITION LABEL ────────────────────────────────────────────────
POS_TO_GROUP = {
    "GK": "GK",
    "CB": "DF", "RB": "DF", "LB": "DF", "RWB": "DF", "LWB": "DF",
    "CDM": "MF", "CM": "MF", "CAM": "MF", "RM": "MF", "LM": "MF",
    "ST": "ATT", "CF": "ATT", "RW": "ATT", "LW": "ATT",
    "RF": "ATT", "LF": "ATT", "RS": "ATT", "LS": "ATT",
    "RCF": "ATT", "LCF": "ATT"
}
ORDER = ["DF", "MF", "ATT"]

def universal_position_label(pos_str):
    if pd.isna(pos_str):
        return "Other"
    tokens = [p.strip() for p in str(pos_str).split(",") if p.strip()]
    groups = {POS_TO_GROUP.get(t) for t in tokens}
    groups.discard(None)
    if not groups:
        return "Other"
    if "GK" in groups:
        return "GK"
    groups = [g for g in ORDER if g in groups]
    if len(groups) == 1:
        return groups[0]
    if len(groups) == 2:
        return f"{groups[0]}/{groups[1]}"
    return "Other"

df["position_universal"] = df["positions"].apply(universal_position_label)
print("\nUniversal position label counts:")
print(df["position_universal"].value_counts())

print("\nExamples of mixed-role players:")
print(df.loc[df["position_universal"].isin(["DF/MF", "MF/ATT", "DF/ATT"]),
             ["name", "positions", "position_universal"]].head(15))

print("\nNumber of 'Other' players:", len(df[df["position_universal"] == "Other"]))

# Fix 'Other' and 'DF/ATT'
def fix_other_positions(pos_str):
    tokens = [p.strip() for p in str(pos_str).split(",")]
    groups = [POS_TO_GROUP.get(t) for t in tokens if POS_TO_GROUP.get(t) in ["DF", "MF", "ATT"]]
    if not groups:
        return "Other"
    freq = Counter(groups)
    top_two = sorted(freq, key=freq.get, reverse=True)[:2]
    if len(top_two) == 1:
        return top_two[0]
    order = ["DF", "MF", "ATT"]
    top_two = sorted(top_two, key=lambda x: order.index(x))
    return f"{top_two[0]}/{top_two[1]}"

df.loc[df["position_universal"] == "Other", "position_universal"] = \
    df.loc[df["position_universal"] == "Other", "positions"].apply(fix_other_positions)
df.loc[df["position_universal"] == "DF/ATT", "position_universal"] = "DF"

print("\nUpdated universal position counts:")
print(df["position_universal"].value_counts())
print(df['position_universal'].isna().sum())

# ─── PLAYSTYLES ──────────────────────────────────────────────────────────────
playstyle_counter = Counter()
for styles in df['play_styles'].dropna():
    for s in styles.split('|'):
        playstyle_counter[s.strip()] += 1

print("\nTotal unique playstyles:", len(playstyle_counter))
print("\nPlaystyle frequency:")
for style, count in playstyle_counter.most_common():
    print(f"{style}: {count}")

# ─── SPLIT INTO GK AND OUTFIELD ──────────────────────────────────────────────
df_goalkeepers = df[df["position_universal"] == "GK"].copy()
df_outfield    = df[df["position_universal"] != "GK"].copy()

print("\nGoalkeepers:", len(df_goalkeepers))
print("Outfield players:", len(df_outfield))

df_goalkeepers.to_csv("goalkeepers_dataset.csv", index=False)
df_outfield.to_csv("outfield_players_dataset.csv", index=False)
print("\nDatasets successfully saved.")

print("\nPosition distribution (GK dataset):")
print(df_goalkeepers["position_universal"].value_counts())
print("\nPosition distribution (Outfield dataset):")
print(df_outfield["position_universal"].value_counts())

print("Missing age values in GK dataset:", df_goalkeepers['age'].isna().sum())
print("Missing age values in Outfield dataset:", df_outfield['age'].isna().sum())

# ─── GOALKEEPER MODEL FEATURES ───────────────────────────────────────────────
gk_features = [
    'goalkeeping_gk_diving', 'goalkeeping_gk_handling', 'goalkeeping_gk_kicking',
    'goalkeeping_gk_positioning', 'goalkeeping_gk_reflexes',
    'attacking_short_passing', 'mentality_vision', 'mentality_composure',
    'movement_reactions', 'movement_agility', 'power_jumping', 'power_strength',
    'power_shot_power', 'movement_balance', 'weak_foot',
    'height_cm', 'weight_kg', 'age'
]
gk_metadata = ['player_id', 'name', 'positions', 'position_universal', 'club_name']
df_goalkeepers_model = df_goalkeepers[gk_metadata + gk_features].copy()
print("\nGK dataset shape:", df_goalkeepers_model.shape)

# ─── OUTFIELD MODEL FEATURES ─────────────────────────────────────────────────
gk_columns = [c for c in df_outfield.columns if "goalkeeping_" in c]
df_outfield_model = df_outfield.drop(columns=gk_columns)
print("\nRemoved GK columns:", gk_columns)
print("\nOutfield dataset shape:", df_outfield_model.shape)

df_goalkeepers_model.to_csv("goalkeepers_model_dataset.csv", index=False)
df_outfield_model.to_csv("outfield_model_dataset.csv", index=False)
print("\nFinal modelling datasets saved.")