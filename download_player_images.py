"""
download_player_images.py
─────────────────────────
Downloads all player face images from SoFIFA to a local folder called
'player_images/' inside your project directory.

Run ONCE from your project folder:
    python download_player_images.py

Requirements:
    pip install requests pandas

HOW IT WORKS:
  - Reads player_id from your CSV files
  - Builds the correct SoFIFA image URL (e.g. 239085 -> players/239/085/26_360.png)
  - Downloads each image and saves as player_images/{player_id}.png
  - Skips already-downloaded images so you can resume if interrupted
"""

import os
import time
import pandas as pd
import requests
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
CSV_FILES = [
    "outfield_model_dataset.csv",
    "goalkeepers_model_dataset.csv",
]

OUTPUT_DIR = "player_images"
SLEEP_BETWEEN = 0.3   # seconds between requests (be polite to server)
# ────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def player_id_to_image_url(player_id: int) -> str:
    """Convert player_id int to SoFIFA CDN URL with split folder format."""
    pid_str = str(player_id).zfill(6)   # pad to 6 digits e.g. 239085
    part1 = pid_str[:3]                  # 239
    part2 = pid_str[3:]                  # 085
    return f"https://cdn.sofifa.net/players/{part1}/{part2}/26_360.png"

def download_image(player_id: int, name: str, session: requests.Session) -> bool:
    out_path = Path(OUTPUT_DIR) / f"{player_id}.png"
    if out_path.exists():
        return True  # already downloaded

    url = player_id_to_image_url(player_id)
    try:
        r = session.get(url, timeout=10)
        if r.status_code == 200 and len(r.content) > 500:
            out_path.write_bytes(r.content)
            return True
        else:
            print(f"  ✗ {name} (ID:{player_id}) — HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ {name} (ID:{player_id}) — {e}")
        return False

# ── Collect all unique players ──
all_players = {}
for csv_file in CSV_FILES:
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found, skipping.")
        continue
    df = pd.read_csv(csv_file, usecols=lambda c: c in ['player_id', 'name'])
    for _, row in df.iterrows():
        pid = row.get('player_id')
        name = row.get('name', '')
        if pd.notna(pid):
            all_players[int(pid)] = name

print(f"Found {len(all_players)} unique players to download.\n")

# ── Session with browser-like headers ──
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Referer': 'https://sofifa.com/',
    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
})

# Visit sofifa.com first to pick up cookies
print("Fetching cookies from sofifa.com...")
try:
    session.get('https://sofifa.com/', timeout=10)
    print("Cookies acquired.\n")
except Exception as e:
    print(f"Could not fetch cookies: {e}\n")

# ── Download loop ──
ok = 0
fail = 0
total = len(all_players)

for i, (pid, name) in enumerate(all_players.items(), 1):
    success = download_image(pid, name, session)
    if success:
        ok += 1
    else:
        fail += 1

    if i % 50 == 0 or i == total:
        print(f"Progress: {i}/{total} — ✓{ok} saved, ✗{fail} failed")

    time.sleep(SLEEP_BETWEEN)

print(f"\nDone! {ok} images saved to '{OUTPUT_DIR}/' folder.")
print(f"{fail} images could not be downloaded.")
print("\nNow restart your Streamlit app — images will load from the local folder!")
