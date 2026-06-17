import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import re
from pathlib import Path
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap

st.set_page_config(page_title="Football Similarity Toolbox", layout="wide")

st.markdown("""
    <style>
        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; font-size: 2.8rem; font-weight: 800;
               letter-spacing: 1px; margin-bottom: 1.5rem;'>
         Football Player Similarity Toolbox
    </h1>
""", unsafe_allow_html=True)

ROLE_FILES = {
    "Attackers": "attackers_similarity_dataset.csv",
    "Midfielders": "midfielders_similarity_dataset.csv",
    "Defenders": "defenders_similarity_dataset.csv",
    "Goalkeepers": "goalkeepers_similarity_dataset.csv",
}

ORIGINAL_FILES = {
    "Attackers": "outfield_model_dataset.csv",
    "Midfielders": "outfield_model_dataset.csv",
    "Defenders": "outfield_model_dataset.csv",
    "Goalkeepers": "goalkeepers_model_dataset.csv",
}

NATIONALITY_FILES = {
    "Attackers": "outfield_players_dataset.csv",
    "Midfielders": "outfield_players_dataset.csv",
    "Defenders": "outfield_players_dataset.csv",
    "Goalkeepers": "goalkeepers_dataset.csv",
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

PIZZA_FEATURES_BY_ROLE = {
    "Attackers": [
        'attacking_finishing', 'attacking_heading_accuracy', 'skill_dribbling',
        'movement_sprint_speed', 'power_shot_power', 'mentality_composure'
    ],
    "Midfielders": [
        'attacking_short_passing', 'skill_long_passing', 'mentality_vision',
        'defending_defensive_awareness', 'mentality_interceptions', 'skill_dribbling'
    ],
    "Defenders": [
        'defending_defensive_awareness', 'skill_long_passing',
        'movement_sprint_speed', 'mentality_interceptions',
        'power_strength', 'attacking_crossing'
    ],
    "Goalkeepers": [
        'goalkeeping_gk_diving', 'goalkeeping_gk_handling',
        'goalkeeping_gk_kicking', 'goalkeeping_gk_reflexes',
        'goalkeeping_gk_positioning', 'movement_reactions'
    ]
}

PIZZA_LABELS_BY_ROLE = {
    "Attackers": ['Finishing', 'Heading', 'Dribbling', 'Sprint Speed', 'Shot Power', 'Composure'],
    "Midfielders": ['Short Pass', 'Long Pass', 'Vision', 'Def Awareness', 'Interceptions', 'Dribbling'],
    "Defenders": ['Def Awareness', 'Long Pass', 'Sprint Speed', 'Interceptions', 'Strength', 'Crossing'],
    "Goalkeepers": ['Diving', 'Handling', 'Kicking', 'Reflexes', 'Positioning', 'Reactions']
}

CLUSTER_COL_BY_ROLE = {
    "Attackers": "att_cluster_label",
    "Midfielders": "mid_cluster_label",
    "Defenders": "def_cluster_label",
    "Goalkeepers": "gk_cluster_label",
}

POSITION_COORDS = {
    "GK":  (0.50, 0.08),
    "CB":  (0.50, 0.25), "RB": (0.80, 0.28), "LB": (0.20, 0.28),
    "RWB": (0.85, 0.40), "LWB": (0.15, 0.40), "SW": (0.50, 0.18),
    "CDM": (0.50, 0.36), "CM": (0.50, 0.53), "RM": (0.87, 0.52),
    "LM":  (0.13, 0.52), "CAM": (0.50, 0.71),
    "RW":  (0.87, 0.86), "LW": (0.13, 0.86), "CF": (0.50, 0.82),
    "ST":  (0.50, 0.86), "RS": (0.65, 0.86), "LS": (0.35, 0.86),
    "RF":  (0.65, 0.78), "LF": (0.35, 0.78),
}

NATIONALITY_TO_EMOJI = {
    'Norwegian': '🇳🇴', 'French': '🇫🇷', 'Portuguese': '🇵🇹', 'British': '🇬🇧',
    'English': '🇬🇧', 'Spanish': '🇪🇸', 'Egyptian': '🇪🇬', 'Brazilian': '🇧🇷',
    'Moroccan': '🇲🇦', 'Argentine': '🇦🇷', 'German': '🇩🇪', 'Italian': '🇮🇹',
    'Dutch': '🇳🇱', 'Belgian': '🇧🇪', 'Croatian': '🇭🇷', 'Polish': '🇵🇱',
    'Senegalese': '🇸🇳', 'Ivorian': '🇨🇮', 'Cameroonian': '🇨🇲', 'Ghanaian': '🇬🇭',
    'Colombian': '🇨🇴', 'Mexican': '🇲🇽', 'Chilean': '🇨🇱', 'Uruguayan': '🇺🇾',
    'Japanese': '🇯🇵', 'Korean': '🇰🇷', 'Australian': '🇦🇺', 'American': '🇺🇸',
    'Swedish': '🇸🇪', 'Danish': '🇩🇰', 'Swiss': '🇨🇭', 'Austrian': '🇦🇹',
    'Turkish': '🇹🇷', 'Ukrainian': '🇺🇦', 'Ukranian': '🇺🇦', 'Serbian': '🇷🇸',
    'Romanian': '🇷🇴', 'Czech': '🇨🇿', 'Slovak': '🇸🇰', 'Hungarian': '🇭🇺',
    'Greek': '🇬🇷', 'Scottish': '🏴', 'Welsh': '🏴', 'Irish': '🇮🇪',
    'Algerian': '🇩🇿', 'Tunisian': '🇹🇳', 'Nigerian': '🇳🇬', 'Malian': '🇲🇱',
    'Guinean': '🇬🇳', 'Congolese': '🇨🇩', 'Gabonese': '🇬🇦', 'Zambian': '🇿🇲',
    'Zimbabwean': '🇿🇼', 'Ecuadorian': '🇪🇨', 'Peruvian': '🇵🇪',
    'Bolivian': '🇧🇴', 'Paraguayan': '🇵🇾', 'Venezuelan': '🇻🇪', 'Russian': '🇷🇺',
    'Finnish': '🇫🇮', 'Icelandic': '🇮🇸', 'Chinese': '🇨🇳', 'Iranian': '🇮🇷',
    'Saudi': '🇸🇦', 'Qatari': '🇶🇦', 'Slovenian': '🇸🇮', 'Bulgarian': '🇧🇬',
    'Albanian': '🇦🇱', 'Bosnian': '🇧🇦', 'Montenegrin': '🇲🇪', 'Georgian': '🇬🇪',
    'Armenian': '🇦🇲', 'Azerbaijani': '🇦🇿', 'Israeli': '🇮🇱', 'Belarusian': '🇧🇾',
    'Latvian': '🇱🇻', 'Lithuanian': '🇱🇹', 'Estonian': '🇪🇪', 'Kenyan': '🇰🇪',
    'Tanzanian': '🇹🇿', 'Ugandan': '🇺🇬', 'Burkinabé': '🇧🇫', 'Jamaican': '🇯🇲',
    'Cypriot': '🇨🇾', 'Indonesian': '🇮🇩', 'Indian': '🇮🇳', 'Gambian': '🇬🇲',
    'Kosovan': '🇽🇰', 'Surinamese': '🇸🇷', 'Angolan': '🇦🇴', 'Togolese': '🇹🇬',
    'Haitian': '🇭🇹', 'Comorian': '🇰🇲', 'Panamanian': '🇵🇦', 'Iraqi': '🇮🇶',
    'South': '🇿🇦',
}

DISPLAY_COLUMNS = [
    "name", "positions", "club_name", "overall_rating", "potential", "age", "value"
]

## fix zoom issues ##
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
}

## REMOVE ZOOM FOR RADAR ##
RADAR_CONFIG = {
    'staticPlot': True,
}
## images for players ##
IMAGE_DIR = Path("player_images")


# ─── FUNCTIONS ───

@st.cache_data
def load_role_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


@st.cache_data
def load_nationality_lookup(players_file: str) -> tuple:
    try:
        df = pd.read_csv(players_file, usecols=['player_id', 'name', 'description'])
        def extract(desc):
            if pd.isna(desc): return None
            m = re.search(r'is an? (\w+) footballer', str(desc))
            return m.group(1) if m else None
        df['nationality'] = df['description'].apply(extract)
        by_id, by_name = {}, {}
        for _, row in df.iterrows():
            nat = row['nationality']
            emoji = NATIONALITY_TO_EMOJI.get(nat, '')
            val = (nat or '', emoji)
            by_id[int(row['player_id'])] = val
            if str(row['name']) not in by_name:
                by_name[str(row['name'])] = val
        return by_id, by_name
    except Exception:
        return {}, {}


@st.cache_data
def load_original_with_clusters(original_file: str, sim_file: str, cluster_col: str) -> pd.DataFrame:
    orig = pd.read_csv(original_file)
    if cluster_col:
        sim = pd.read_csv(sim_file, usecols=lambda c: c in ['name', cluster_col])
        sim = sim.drop_duplicates(subset='name')
        orig = orig.merge(sim, on='name', how='left')
    return orig


@st.cache_data
def load_id_lookup(original_file: str) -> tuple:
    try:
        orig = pd.read_csv(original_file, usecols=lambda c: c in ['name', 'player_id'])
        by_index = dict(zip(orig.index, orig['player_id'].astype(int)))
        by_name = {}
        for _, row in orig.iterrows():
            if str(row['name']) not in by_name:
                by_name[str(row['name'])] = int(row['player_id'])
        return by_index, by_name
    except Exception:
        return {}, {}


@st.cache_data
def load_value_lookup(original_file: str) -> dict:
    try:
        orig = pd.read_csv(original_file, usecols=lambda c: c in ['name', 'value', 'player_id'])
        # Fallback to player_stats for GKs (no value column)
        if 'value' not in orig.columns or orig['value'].isna().all():
            orig = pd.read_csv("player_stats.csv", usecols=['name', 'value', 'player_id'])
            orig = orig.drop_duplicates(subset='player_id', keep='first')
        else:
            if 'player_id' in orig.columns:
                orig = orig.drop_duplicates(subset='player_id', keep='first')
        result = {}
        for _, row in orig.iterrows():
            v = row['value']
            pid = row.get('player_id')
            if pd.notna(v) and pd.notna(pid):
                try:
                    s = str(v).replace('€', '').replace(',', '').strip()
                    if 'M' in s: display = f"€{float(s.replace('M','')):.1f}M"
                    elif 'K' in s: display = f"€{float(s.replace('K','')):.0f}K"
                    else:
                        m = float(s) / 1_000_000
                        display = f"€{m:.1f}M" if m >= 1 else f"€{float(s)/1000:.0f}K"
                except:
                    display = str(v)
                result[int(pid)] = display
        return result
    except Exception:
        return {}


def get_player_image(player_id):
    if player_id is None:
        return None
    img_path = IMAGE_DIR / f"{int(player_id)}.png"
    if img_path.exists():
        return Image.open(img_path)
    return None


def parse_value(value_str) -> float:
    if pd.isna(value_str): return 0.0
    s = str(value_str).replace('€', '').replace(',', '').strip()
    try:
        if 'M' in s: return float(s.replace('M', ''))
        elif 'K' in s: return float(s.replace('K', '')) / 1000
        else: return float(s) / 1_000_000
    except Exception:
        return 0.0


@st.cache_data
def build_similarity(df: pd.DataFrame, features: list):
    features = [c for c in features if c in df.columns]
    model_df = df[features].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_df)
    sim = cosine_similarity(scaled)
    sim_df = pd.DataFrame(sim, index=df.index, columns=df.index)
    return sim_df, features, scaled


@st.cache_data
def compute_umap(scaled_array: np.ndarray):
    reducer = umap.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(scaled_array)


def get_similar_players(df, sim_df, player_idx, top_n, cluster_col, max_value_m=None, value_lookup=None):
    scores = sim_df.loc[player_idx].sort_values(ascending=False).drop(player_idx)
    top_idx = scores.head(top_n * 5).index
    keep_cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    if cluster_col and cluster_col in df.columns:
        keep_cols = keep_cols + [cluster_col]
    result = df.loc[top_idx, keep_cols].copy()
    result["similarity_score"] = scores.loc[top_idx].values
    if max_value_m is not None and 'value' in result.columns:
        result["_value_m"] = result["value"].apply(parse_value)
        result = result[result["_value_m"] <= max_value_m].drop(columns=["_value_m"])

    # Format value column to match player card display (€XX.XM)
    if 'value' in result.columns:
        def format_value(v):
            m = parse_value(v)
            if m >= 1:
                return f"€{m:.1f}M"
            else:
                return f"€{m*1000:.0f}K"
        result['value'] = result['value'].apply(format_value)

    return result.head(top_n).sort_values("similarity_score", ascending=False)

def get_young_talents(df, sim_df, player_idx, player_age, role=""):
    scores = sim_df.loc[player_idx].sort_values(ascending=False).drop(player_idx)

    # GKs peak later — 26 is still young for a goalkeeper
    if role == "Goalkeepers":
        age_cutoff = 24 if player_age >= 26 else player_age - 1
    else:
        age_cutoff = 21 if player_age >= 23 else player_age - 1

    high_sim = scores[scores >= 0.85]
    if high_sim.empty:
        return pd.DataFrame()
    candidates = df.loc[high_sim.index].copy()
    candidates["similarity_score"] = high_sim.values
    if "age" in candidates.columns:
        candidates = candidates[candidates["age"] <= age_cutoff]
    if candidates.empty:
        return pd.DataFrame()
    keep_cols = [c for c in ["name", "positions", "club_name", "overall_rating", "potential", "age"] if
                 c in candidates.columns]
    result = candidates[keep_cols + ["similarity_score"]].copy()
    return result.sort_values("similarity_score", ascending=False).head(5)


def make_why_this_match_chart(player_row, talent_row, features, role, player_name, talent_name):
    pizza_features = PIZZA_FEATURES_BY_ROLE[role]
    pizza_labels = PIZZA_LABELS_BY_ROLE[role]
    compare_features = [f for f in pizza_features if f in player_row.index and f in talent_row.index]
    compare_labels = [pizza_labels[i] for i, f in enumerate(pizza_features) if f in compare_features]
    if not compare_features:
        return None
    vals_player = [float(player_row[f]) if pd.notna(player_row[f]) else 0 for f in compare_features]
    vals_talent = [float(talent_row[f]) if pd.notna(talent_row[f]) else 0 for f in compare_features]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=player_name, y=compare_labels, x=vals_player,
                         orientation='h', marker_color='#1f77b4', opacity=0.85))
    fig.add_trace(go.Bar(name=talent_name, y=compare_labels, x=vals_talent,
                         orientation='h', marker_color='#FFD700', opacity=0.85))
    fig.update_layout(
        barmode='group',
        title=dict(text="Key Attributes", x=0.4),
        xaxis=dict(title="Attribute Value", range=[0, 100]),
        yaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320, margin=dict(t=60, b=30, l=10, r=10),
    )
    return fig


def make_pitch_figure(positions_str: str) -> go.Figure:
    raw = str(positions_str).replace("/", ",").replace("|", ",")
    positions = [p.strip().upper() for p in raw.split(",") if p.strip()]
    primary_pos = positions[0] if positions else None

    LINE = dict(color="#222222", width=2)

    fig = go.Figure()

    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                  fillcolor="rgba(0,0,0,0)", line=dict(color="#222222", width=2.5))
    fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5, line=LINE)

    theta = np.linspace(0, 2*np.pi, 120)
    r = 0.12
    fig.add_trace(go.Scatter(x=0.5+r*np.cos(theta), y=0.5+r*np.sin(theta)*0.625,
                             mode="lines", line=LINE, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="markers",
                             marker=dict(color="#222222", size=5),
                             showlegend=False, hoverinfo="skip"))
    for y0, y1 in [(0, 0.20), (0.80, 1)]:
        fig.add_shape(type="rect", x0=0.18, y0=y0, x1=0.82, y1=y1,
                      line=LINE, fillcolor="rgba(0,0,0,0)")
    for y0, y1 in [(0, 0.08), (0.92, 1)]:
        fig.add_shape(type="rect", x0=0.33, y0=y0, x1=0.67, y1=y1,
                      line=LINE, fillcolor="rgba(0,0,0,0)")
    for y in [0.0, 1.0]:
        fig.add_shape(type="line", x0=0.38, y0=y, x1=0.62, y1=y,
                      line=dict(color="#222222", width=3.5))
    fig.add_trace(go.Scatter(x=[0.5, 0.5], y=[0.14, 0.86], mode="markers",
                             marker=dict(color="#222222", size=4),
                             showlegend=False, hoverinfo="skip"))

    arc_theta = np.linspace(np.pi*0.15, np.pi*0.85, 40)
    r_arc = 0.10
    fig.add_trace(go.Scatter(x=0.5+r_arc*np.cos(arc_theta),
                             y=0.20+r_arc*np.sin(arc_theta)*0.625*0.6,
                             mode="lines", line=LINE, showlegend=False, hoverinfo="skip"))
    arc_theta2 = np.linspace(np.pi*1.15, np.pi*1.85, 40)
    fig.add_trace(go.Scatter(x=0.5+r_arc*np.cos(arc_theta2),
                             y=0.80+r_arc*np.sin(arc_theta2)*0.625*0.6,
                             mode="lines", line=LINE, showlegend=False, hoverinfo="skip"))

    found_any = False
    for pos in positions:
        coords = POSITION_COORDS.get(pos)
        if coords:
            found_any = True
            px_val, py_val = coords
            is_primary = (pos == primary_pos)

            if is_primary:
                # Larger glow + gold marker for primary position
                fig.add_trace(go.Scatter(x=[px_val], y=[py_val], mode="markers",
                                         marker=dict(color="rgba(255,60,60,0.30)", size=54, line=dict(width=0)),
                                         showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[px_val], y=[py_val], mode="markers+text",
                                         marker=dict(color="#FF3B3B", size=34,
                                                     line=dict(color="#222222", width=2)),
                                         text=[pos], textposition="middle center",
                                         textfont=dict(color="white", size=11, family="Arial Black"),
                                         showlegend=False, hovertext=f"{pos} (Primary)", hoverinfo="text"))
            else:
                fig.add_trace(go.Scatter(x=[px_val], y=[py_val], mode="markers",
                                         marker=dict(color="rgba(255,220,0,0.22)", size=34, line=dict(width=0)),
                                         showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=[px_val], y=[py_val], mode="markers+text",
                                         marker=dict(color="#FFD700", size=22,
                                                     line=dict(color="#222222", width=1.5)),
                                         text=[pos], textposition="middle center",
                                         textfont=dict(color="#1a1a1a", size=7, family="Arial Black"),
                                         showlegend=False, hovertext=pos, hoverinfo="text"))

    if not found_any:
        fig.add_annotation(x=0.5, y=0.5, text=positions_str,
                           showarrow=False, font=dict(color="#222222", size=14))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.02, 1.02], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.02, 1.02], scaleanchor="x", scaleratio=1.6, fixedrange=True),
        margin=dict(l=0, r=0, t=0, b=0), height=300,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def make_pizza_chart(player_row, compare_row, features, labels, player_name, compare_name):
    vals_p, vals_c = [], []
    for f in features:
        vals_p.append(float(player_row[f]) if f in player_row and pd.notna(player_row[f]) else 0)
        vals_c.append(float(compare_row[f]) if f in compare_row and pd.notna(compare_row[f]) else 0)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_p+[vals_p[0]], theta=labels+[labels[0]],
                                  fill='toself', name=player_name,
                                  line=dict(color='#1f77b4', width=2),
                                  fillcolor='rgba(31,119,180,0.3)'))
    fig.add_trace(go.Scatterpolar(r=vals_c+[vals_c[0]], theta=labels+[labels[0]],
                                  fill='toself', name=compare_name,
                                  line=dict(color='#ff7f0e', width=2),
                                  fillcolor='rgba(255,127,14,0.3)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      showlegend=True,
                      title=dict(text=f"{player_name} vs {compare_name}", x=0.5),
                      height=420, margin=dict(t=60, b=20, l=40, r=40))

    return fig


def make_trajectory_chart(orig_df, cluster_col, metric, metric_label, role, player_age, player_overall=None, player_row=None):
    df_plot = orig_df.copy()
    if metric == 'value_m':
        df_plot['value_m'] = df_plot['value'].apply(parse_value)
        metric = 'value_m'
    df_plot = df_plot.dropna(subset=['age', metric, cluster_col])
    df_plot['age'] = df_plot['age'].astype(int)
    df_plot = df_plot[(df_plot['age'] >= 16) & (df_plot['age'] <= 40)]
    grouped = (df_plot.groupby([cluster_col, 'age'])[metric]
               .mean().reset_index()
               .rename(columns={metric: 'avg_value', cluster_col: 'cluster'}))
    counts = df_plot.groupby([cluster_col, 'age']).size().reset_index(name='count')
    counts.columns = ['cluster', 'age', 'count']
    grouped = grouped.merge(counts, on=['cluster', 'age'])
    grouped = grouped[grouped['count'] >= 5]
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, cluster in enumerate(sorted(grouped['cluster'].unique())):
        cdata = grouped[grouped['cluster'] == cluster].sort_values('age')
        fig.add_trace(go.Scatter(
            x=cdata['age'], y=cdata['avg_value'],
            mode='lines+markers', name=cluster,
            line=dict(width=2.5, color=colors[i % len(colors)]),
            marker=dict(size=5),
            hovertemplate=f"<b>{cluster}</b><br>Age: %{{x}}<br>{metric_label}: %{{y:.1f}}<extra></extra>"
        ))

    # Get the colour of the selected player's cluster
    player_cluster = player_row.get(cluster_col, '') if hasattr(player_row, 'get') else ''
    sorted_clusters = sorted(grouped['cluster'].unique())
    cluster_color = colors[
        sorted_clusters.index(player_cluster) % len(colors)] if player_cluster in sorted_clusters else 'red'

    # Vertical line in cluster colour
    fig.add_vline(x=player_age, line_dash="dash", line_color=cluster_color,
                  annotation_text=f"Selected player (age {player_age})",
                  annotation_position="top right",
                  annotation_font_color=cluster_color)

    # ⭐ Star marker for player's own rating on the chart
    if player_overall is not None and metric != 'value_m':
        fig.add_trace(go.Scatter(
            x=[player_age],
            y=[player_overall],
            mode='markers',
            marker=dict(symbol='star', size=18, color='red',
                        line=dict(color='white', width=1.5)),
            name='Selected Player',
            hovertemplate=f"<b>Selected Player</b><br>Age: {player_age}<br>{metric_label}: {player_overall}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=f"{role} — {metric_label} by Age & Cluster", x=0.5),
        xaxis=dict(title="Age", dtick=2),
        yaxis=dict(title=metric_label, range=[0, 100]),  # ← fixed 0-100
        legend=dict(title="Cluster"),
        height=420, margin=dict(t=60, b=40, l=50, r=20),
        hovermode="x unified"
    )
    return fig


# ─── SIDEBAR ───
role = st.sidebar.selectbox("Select role", list(ROLE_FILES.keys()))
top_n = st.sidebar.slider("Number of similar players", min_value=5, max_value=20, value=10)

df = load_role_data(ROLE_FILES[role])
sim_df, used_features, scaled_array = build_similarity(df, FEATURES_BY_ROLE[role])
cluster_col = CLUSTER_COL_BY_ROLE.get(role, "")
id_by_index, id_by_name = load_id_lookup(ORIGINAL_FILES[role])
value_lookup = load_value_lookup(ORIGINAL_FILES[role])  # ← FIXED: use role file, fallback handles GKs
nat_by_id, nat_by_name = load_nationality_lookup(NATIONALITY_FILES[role])
orig_with_clusters = load_original_with_clusters(ORIGINAL_FILES[role], ROLE_FILES[role], cluster_col)

player_options = df["name"].astype(str).tolist()
selected_name = st.sidebar.selectbox("Select player", player_options)

st.sidebar.markdown("---")
max_budget_m = None
with st.sidebar.expander("💰 Budget Filter", expanded=False):
    budget_enabled = st.checkbox("Filter by max market value")
    if budget_enabled:
        max_budget_m = st.slider("Max player value (€M)", min_value=0, max_value=200, value=50, step=5)
        st.caption(f"Showing similar players worth ≤ €{max_budget_m}M")

matches = df.index[df["name"] == selected_name].tolist()
if not matches:
    st.error("Player not found.")
    st.stop()

if len(matches) > 1:
    chosen_idx = st.sidebar.selectbox(
        "Duplicate Name — Choose Player", matches,
        format_func=lambda i: f"{df.loc[i,'name']} | {df.loc[i,'club_name']} | {df.loc[i,'positions']}"
    )
    # Show a small preview image + info right after selection
    preview_pid = int(df.loc[chosen_idx, 'player_id']) if 'player_id' in df.columns else None
    preview_img = get_player_image(preview_pid)
    if preview_img:
        prev_col1, prev_col2 = st.sidebar.columns([1, 2])
        with prev_col1:
            st.image(preview_img, width=60)
        with prev_col2:
            st.caption(f"{df.loc[chosen_idx,'club_name']} | Age {int(df.loc[chosen_idx,'age'])}")
else:
    chosen_idx = matches[0]

player_row = df.loc[chosen_idx]
player_name_str = str(player_row.get("name", ""))
player_age = int(player_row.get("age", 25))
# ← FIXED: player_id defined FIRST, then value lookup uses it
player_id = int(df.loc[chosen_idx, 'player_id']) if 'player_id' in df.columns else id_by_name.get(player_name_str)
player_value_str = value_lookup.get(player_id, "N/A")  # ← FIXED: uses player_id, no typo
nat_info = nat_by_id.get(player_id, nat_by_name.get(player_name_str, ('', '')))
nationality_str = nat_info[0]
flag_emoji = nat_info[1]

# ─── TOP ROW ───
col_card, col_table = st.columns([1, 2])

with col_card:
    st.markdown(f"<h3 style='text-align:center; margin-bottom:0.2rem;'>{player_name_str}</h3>",
                unsafe_allow_html=True)

    _, img_col, _ = st.columns([1, 2, 1])
    with img_col:
        img = get_player_image(player_id)
        if img:
            st.image(img, use_container_width=True)
        else:
            st.markdown("<div style='text-align:center; font-size:4rem;'>👤</div>",
                        unsafe_allow_html=True)

    if nationality_str and flag_emoji:
        st.markdown(
            f"<div style='text-align:center; font-size:1.1rem; margin-top:0.3rem;'>"
            f"{flag_emoji} {nationality_str}</div>",
            unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0.6rem 0;'>", unsafe_allow_html=True)

    stats_html = f"""
    <div style='text-align:center; line-height:1.9rem; font-size:0.95rem;'>
        Position: <b>{player_row.get('positions', '')}</b><br>
        Football Club: {player_row.get('club_name', '')}<br>
        Overall Rating: <b>{int(player_row.get('overall_rating', 0))}</b><br>
        Potential Rating: <b>{int(player_row.get('potential', 0))}</b><br>
        Market Value: <b>{player_value_str}</b><br>
        Age: <b>{player_age}</b><br>
    """
    if cluster_col and cluster_col in df.columns:
        stats_html += f"🏷️ <b>{player_row.get(cluster_col, '')}</b><br>"
    stats_html += "</div>"
    st.markdown(stats_html, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0.6rem 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center; font-weight:600; margin-bottom:0;'>Position on Pitch</p>",
                unsafe_allow_html=True)
    pitch_fig = make_pitch_figure(str(player_row.get("positions", "")))
    st.plotly_chart(pitch_fig, use_container_width=True, config=PLOTLY_CONFIG)

with col_table:
    st.subheader("Top similar players")
    similar_df = get_similar_players(
        df, sim_df, chosen_idx, top_n, cluster_col,
        max_value_m=max_budget_m,
        value_lookup=value_lookup if budget_enabled else None
    )
    st.dataframe(
        similar_df.rename(columns={
            "name": "👤 Name", "positions": "📍 Position", "club_name": "🏟️ Club",
            "overall_rating": "⭐ Overall", "potential": "📈 Potential", "age": "🎂 Age",
            "att_cluster_label": "🏷️ Type", "mid_cluster_label": "🏷️ Type",
            "def_cluster_label": "🏷️ Type", "gk_cluster_label": "🏷️ Type",
            "similarity_score": "🎯 Similarity", "value": "💰 Value",
        }),
        use_container_width=True
    )

    # ── Young Talent Alert ──
    st.markdown("---")
    if role == "Goalkeepers":
        age_cutoff = 24 if player_age >= 26 else player_age - 1
    else:
        age_cutoff = 21 if player_age >= 23 else player_age - 1
    young_df = get_young_talents(df, sim_df, chosen_idx, player_age, role=role)

    if not young_df.empty:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                        border-left: 4px solid #FFD700;
                        border-radius: 8px; padding: 1rem 1.2rem; margin-top: 0.5rem;'>
                <h4 style='color:#FFD700; margin:0 0 0.3rem 0;'>🌟 Young Talent Alert</h4>
                <p style='color:#ccc; margin:0; font-size:0.88rem;'>
                    Players aged ≤ {age_cutoff} with ≥ 85% similarity to {player_name_str}
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            young_df.rename(columns={
                "name": "👤 Name", "positions": "📍 Position", "club_name": "🏟️ Club",
                "overall_rating": "⭐ Overall", "potential": "📈 Potential", "age": "🎂 Age",
                "similarity_score": "🎯 Similarity",
            }),
            use_container_width=True
        )

        talent_names = young_df["name"].tolist()
        st.markdown(
            "<p style='font-size:1.1rem; font-weight:700; color:#1a1a2e; margin-bottom:0.3rem;'>"
            "🔍 Why This Match?</p>"
            "<p style='font-size:0.9rem; color:#666; margin-top:0;'>"
            "Select a young talent below to compare their key attributes side-by-side:</p>",
            unsafe_allow_html=True
        )
        selected_talent = st.selectbox(
            "Choose a player",
            talent_names,
            key="talent_compare",
            label_visibility="collapsed"
        )

        talent_matches = df[df["name"] == selected_talent]
        if not talent_matches.empty:
            talent_row = talent_matches.iloc[0]
            why_fig = make_why_this_match_chart(
                player_row, talent_row,
                PIZZA_FEATURES_BY_ROLE[role],
                role, player_name_str, selected_talent
            )
            if why_fig:
                st.plotly_chart(why_fig, use_container_width=True, config=PLOTLY_CONFIG)

    else:
        st.markdown(f"""
            <div style='background: #f8f9fa; border-left: 4px solid #ccc;
                        border-radius: 8px; padding: 1rem 1.2rem; margin-top: 0.5rem;'>
                <p style='color:#888; margin:0; font-size:1.2rem;'>
                    🌟 No young talents found with ≥ 85% similarity and age ≤ {age_cutoff}.
                </p>
                <p style='color:#888; margin:0; font-size:0.88rem;'>
                No players aged {age_cutoff} or younger matched with ≥ 85% similarity to {player_name_str}.
                Try increasing the number of similar players, or check a different player!
            </p>
            </div>
        """, unsafe_allow_html=True)

st.divider()

# ─── MIDDLE ROW: UMAP + Pizza ───
col_umap, col_pizza = st.columns(2)

with col_umap:
    st.subheader("UMAP Projection")
    with st.spinner("Computing UMAP..."):
        embedding = compute_umap(scaled_array)
    umap_plot_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=df.index)
    umap_plot_df["name"] = df["name"].values
    umap_plot_df["color"] = "Other players"
    umap_plot_df.loc[umap_plot_df.index.isin(similar_df.index), "color"] = "Similar players"
    umap_plot_df.loc[umap_plot_df.index == chosen_idx, "color"] = "Selected player"
    color_map = {"Other players": "#cccccc", "Similar players": "#1f77b4", "Selected player": "#d62728"}
    fig_umap = px.scatter(umap_plot_df, x="UMAP1", y="UMAP2",
                          color="color", color_discrete_map=color_map,
                          hover_name="name", title=f"UMAP — {role}", height=420)
    selected_point = umap_plot_df[umap_plot_df.index == chosen_idx]
    fig_umap.add_trace(go.Scatter(x=selected_point["UMAP1"], y=selected_point["UMAP2"],
                                  mode="markers+text",
                                  marker=dict(size=14, color="#d62728", symbol="star"),
                                  text=selected_point["name"], textposition="top center",
                                  showlegend=False))
    fig_umap.update_layout(legend_title_text="", margin=dict(t=50, b=20))
    st.plotly_chart(fig_umap, use_container_width=True, config=PLOTLY_CONFIG)

with col_pizza:
    st.subheader("Attribute Comparison (Radar Chart)")
    pizza_features = PIZZA_FEATURES_BY_ROLE[role]
    pizza_labels = PIZZA_LABELS_BY_ROLE[role]
    available = [f for f in pizza_features if f in df.columns]
    available_labels = [pizza_labels[i] for i, f in enumerate(pizza_features) if f in df.columns]
    top_similar_idx = similar_df.index[0]
    compare_row = df.loc[top_similar_idx]
    compare_name = str(compare_row.get("name", "Similar Player"))
    fig_pizza = make_pizza_chart(player_row, compare_row, available, available_labels,
                                 player_name_str, compare_name)
    st.plotly_chart(fig_pizza, use_container_width=True, config=RADAR_CONFIG)

st.divider()

# ─── BOTTOM ROW: Age Trajectory ───
st.subheader(f"📈 Career Trajectory by Cluster — {role}")
st.caption("Average rating and market value by age for each player cluster. Red dashed line = selected player's current age.")

tab_rating, tab_value = st.tabs(["⭐ Rating Trajectory", "💰 Value Trajectory"])

with tab_rating:
    if 'overall_rating' in orig_with_clusters.columns and cluster_col in orig_with_clusters.columns:
        fig_traj_rating = make_trajectory_chart(
            orig_with_clusters, cluster_col, 'overall_rating',
            'Avg Overall Rating', role, player_age,
            player_overall=int(player_row.get('overall_rating', 0)),
            player_row=player_row
        )
        st.plotly_chart(fig_traj_rating, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.warning("Rating or cluster data not available for this role.")

with tab_value:
    if 'value' in orig_with_clusters.columns and cluster_col in orig_with_clusters.columns:
        fig_traj_value = make_trajectory_chart(
            orig_with_clusters, cluster_col, 'value_m',
            'Avg Market Value (€M)', role, player_age)
        st.plotly_chart(fig_traj_value, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.warning("Value or cluster data not available for this role.")

with st.expander("Features used in this model"):
    st.write(used_features)