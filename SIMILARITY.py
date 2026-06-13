import pandas as pd
import numpy as np

##  LOAD ( read_csv only AND encode)
df = pd.read_csv("player_stats.csv", encoding="latin-1")

## CLEAN + parse DOB
# Replace common placeholders
df['dob'] = df['dob'].replace(['###', '####', '#####', ''], np.nan)

## CONVERT to datetime
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

# Find the player with missing DOB
cols = ['player_id','name','version','club_name','dob','age']
cols = [c for c in cols if c in df.columns]

missing = df.loc[df['dob'].isna(), cols]
print(missing)
print("Available columns:", df.columns.tolist())

## PLAYER WITH MISSING DOB, AGE FOUND ##
## FIX TORRICO'S MISSING AGE ##
# Set real DOB
df.loc[df['player_id'] == 274195, 'dob'] = pd.to_datetime('2004-08-10')

# Recalculate age
today = pd.Timestamp.today().normalize()
df['age'] = (today - df['dob']).dt.days // 365

## Create age (from DOB)
today = pd.Timestamp.today().normalize()
df['age'] = (today - df['dob']).dt.days // 365

## REMOVE duplicates (one row per player_id per version= FC26)
df = df.drop_duplicates(subset=['player_id', 'version'], keep='first')

# Verification #
print("Rows:", len(df))
print("Duplicate player_id+version:", df.duplicated(subset=['player_id', 'version']).sum())
print("Missing DOB:", df['dob'].isna().sum())
print("Missing Age:", df['age'].isna().sum())
