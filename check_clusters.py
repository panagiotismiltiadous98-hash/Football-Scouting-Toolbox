import pandas as pd

dfd = pd.read_csv('defenders_similarity_dataset.csv')
print('=== DEFENDERS ===')
for label in dfd['def_cluster_label'].unique():
    sub = dfd[dfd['def_cluster_label'] == label]
    print(f'{label}:')
    print(f'  age={sub["age"].mean():.1f}, rating={sub["overall_rating"].mean():.1f}')
    print(f'  def_awareness={sub["defending_defensive_awareness"].mean():.1f}')
    print(f'  strength={sub["power_strength"].mean():.1f}')
    print(f'  sprint={sub["movement_sprint_speed"].mean():.1f}')
    print()

mid = pd.read_csv('midfielders_similarity_dataset.csv')
print('=== MIDFIELDERS ===')
for label in mid['mid_cluster_label'].unique():
    sub = mid[mid['mid_cluster_label'] == label]
    print(f'{label}:')
    print(f'  age={sub["age"].mean():.1f}, rating={sub["overall_rating"].mean():.1f}')
    print(f'  def_awareness={sub["defending_defensive_awareness"].mean():.1f}')
    print(f'  vision={sub["mentality_vision"].mean():.1f}')
    print(f'  stamina={sub["power_stamina"].mean():.1f}')
    print()

att = pd.read_csv('attackers_similarity_dataset.csv')
print('=== ATTACKERS ===')
for label in att['att_cluster_label'].unique():
    sub = att[att['att_cluster_label'] == label]
    print(f'{label}:')
    print(f'  age={sub["age"].mean():.1f}, rating={sub["overall_rating"].mean():.1f}')
    print(f'  finishing={sub["attacking_finishing"].mean():.1f}')
    print(f'  dribbling={sub["skill_dribbling"].mean():.1f}')
    print(f'  sprint={sub["movement_sprint_speed"].mean():.1f}')
    print()
