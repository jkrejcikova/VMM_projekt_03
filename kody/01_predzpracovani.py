# 01_predzpracovani.py

"""
Semestrální projekt VMM – Střelecká úspěšnost (Premier League)
Skript 1: Předzpracování a očištění dat
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_IN = os.path.join(DATA_DIR, 'shots_outcome_Final.csv')
DATA_OUT = os.path.join(DATA_DIR, 'clean_shots.csv')

# 1. Načtení dat
# V datasetu hodnota 'NA' = bez asistence a 'None' = samostatná akce (nikoli prázdná hodnota NaN).
# Výchozí chování pro detekci NaN hodnot je proto upraveno.

df_raw = pd.read_csv(
    DATA_IN,
    keep_default_na=False,
    na_values=['', 'NaN', 'nan', 'null', 'NULL']
)

print(f'Načteno {len(df_raw)} střel.')

# 2. Vlastní góly a cílová proměnná
# Vyřazení vlastních gólů (OwnGoal).
n_owngoals = (df_raw['result'] == 'OwnGoal').sum()
df = df_raw[df_raw['result'] != 'OwnGoal'].copy()

# Vytvoření binární cílové proměnné goal (1 = Goal, 0 = ostatní).
df['goal'] = (df['result'] == 'Goal').astype(int)
print(f'Vyřazeno {n_owngoals} vlastních gólů. Zbylo gólů: {df["goal"].sum()}.')

# 3. Vyřazení střel bez souřadnic X/Y
# Vyřazení řádků bez souřadnic (nezbytné pro následný výpočet geometrie).
mask_missing_xy = df['X'].isna() | df['Y'].isna()
df_clean = df[~mask_missing_xy].copy()
print(f'Smazáno {mask_missing_xy.sum()} střel bez souřadnic. Zůstává {len(df_clean)} střel.')

# 4. Přepočet na reálné metry
# Rozměry hřiště: 105x68 m. Původní X, Y jsou v rozsahu 0-1.
df_clean['x_m'] = df_clean['X'] * 105.0
df_clean['y_m'] = df_clean['Y'] * 68.0

# Vzdálenost od středu branky (x=105, y=34) v metrech.
dx = 105.0 - df_clean['x_m']
dy = 34.0 - df_clean['y_m']
df_clean['dist_m'] = np.sqrt(dx**2 + dy**2)

# Střelecký úhel (branka 7,32 m, tyče na y = 34 +- 3,66 m).
y_tyc1 = 34.0 - 3.66
y_tyc2 = 34.0 + 3.66
uhel1 = np.arctan2(y_tyc1 - df_clean['y_m'], dx)
uhel2 = np.arctan2(y_tyc2 - df_clean['y_m'], dx)
df_clean['angle_deg'] = np.degrees(np.abs(uhel2 - uhel1))

# 5. Úprava kategorií
df_clean['shotType'] = df_clean['shotType'].fillna('Unknown')
if 'OtherBodyPart' in df_clean['shotType'].values:
    df_clean['shotType'] = df_clean['shotType'].replace({'OtherBodyPart': 'Other'})

# Sloučení vzácných akcí (<= 5 výskytů) do kategorie 'Other'.
akce_counts = df_clean['lastAction'].value_counts()
vzacne_akce = akce_counts[akce_counts <= 5].index.tolist()
df_clean['lastAction'] = df_clean['lastAction'].replace({a: 'Other' for a in vzacne_akce})

# Tvorba binárních příznaků pro penaltu a asistenci.
df_clean['is_penalty'] = (df_clean['situation'] == 'Penalty').astype(int)
df_clean['has_assist'] = (df_clean['player_assisted'] != 'NA').astype(int)

# 6. Odstranění nepotřebných sloupců
# Vyřazení identifikátorů a konečného skóre (prevence úniku informací - data leakage).
sloupce_vyradit = ['id', 'match_id', 'player_id', 'h_goals', 'a_goals']
df_clean = df_clean.drop(columns=sloupce_vyradit).copy()

sloupce_poradi = [
    'minute', 'result', 'goal', 'X', 'Y', 'x_m', 'y_m', 'dist_m', 'angle_deg',
    'xG', 'player', 'h_a', 'situation', 'is_penalty', 'shotType', 'lastAction',
    'player_assisted', 'has_assist', 'year', 'date', 'h_team', 'a_team'
]
df_clean = df_clean[sloupce_poradi]

# Uložení finálního datasetu
df_clean.to_csv(DATA_OUT, index=False, encoding='utf-8')
print(f'Hotovo. Uloženo do {DATA_OUT}')