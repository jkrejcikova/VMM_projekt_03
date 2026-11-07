# -*- coding: utf-8 -*-
"""
=============================================================================
SEMESTRÁLNÍ PROJEKT VMM – STŘELECKÁ ÚSPĚŠNOST (PREMIER LEAGUE)
Skript 1: Předzpracování a očištění dat
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_IN = os.path.join(BASE_DIR, 'data', 'shots_outcome_Final.csv')
DATA_OUT = os.path.join(BASE_DIR, 'data', 'clean_shots.csv')

print('=' * 65)
print('1. PŘEDZPRACOVÁNÍ DAT: shots_outcome_Final.csv -> clean_shots.csv')
print('=' * 65)

# 1. Načtení datové sady
df = pd.read_csv(DATA_IN)
n_start = len(df)
print(f'Původní počet záznamů: {n_start} střel.')

# 2. Vyřazení vlastních gólů (vlastní góly nereprezentují střelecký záměr útočníka)
df = df[df['result'] != 'OwnGoal'].copy()
print(f'Po vyřazení vlastních gólů (OwnGoal): {len(df)} střel.')

# 3. Vytvoření cílové binární proměnné (1 = Gól, 0 = Ostatní)
df['goal'] = (df['result'] == 'Goal').astype(int)

# 4. Přeškálování prostorových souřadnic X a Y na fotbalové hřiště 100 × 100
df['X'] = df['X'] * 100.0
df['Y'] = df['Y'] * 100.0

# 5. Odvození geometrických veličin
# Eukleidovská vzdálenost střely od středu branky [X=100, Y=50] v metrech/jednotkách
df['dist_m'] = np.sqrt((100.0 - df['X'])**2 + (50.0 - df['Y'])**2)

# Střelecký zorný úhel sevřený brankovými tyčemi (branka Y v rozmezí 45.2 až 54.8, X = 100)
y_min_b, y_max_b, x_br = 45.2, 54.8, 100.0
df['angle_deg'] = np.degrees(
    np.abs(
        np.arctan2(df['X'] - x_br, df['Y'] - y_min_b) -
        np.arctan2(df['X'] - x_br, df['Y'] - y_max_b)
    )
)

# 6. Ošetření chybějících hodnot
df['player_assisted'] = df['player_assisted'].fillna('Bez asistence')
df['shotType']        = df['shotType'].fillna('Neznamy')
df['situation']       = df['situation'].fillna('Neznama')

# Věcné odstranění řádků s chybějícími souřadnicemi či xG
n_before_drop = len(df)
df = df.dropna(subset=['X', 'Y', 'xG']).copy()
print(f'Vyřazeno {n_before_drop - len(df)} střel bez prostorových souřadnic.')

# 7. Aplikace pravidla 3-sigma na proměnnou dist_m
z_dist = np.abs(stats.zscore(df['dist_m']))
n_before_3sigma = len(df)
df = df[z_dist < 3].copy()
print(f'Pravidlo 3-sigma vyřadilo {n_before_3sigma - len(df)} extrémně odlehlých střel.')

# 8. Uložení očištěných dat do clean_shots.csv
df.to_csv(DATA_OUT, index=False)
print(f'\n[OK] Očištěná data úspěšně uložena: {DATA_OUT}')
print(f'     Finální vzorek: n = {len(df)} střel')
print(f'     Z toho gólů:    {df["goal"].sum()} ({df["goal"].mean():.1%})')
print('=' * 65)
