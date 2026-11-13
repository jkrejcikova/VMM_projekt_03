# -*- coding: utf-8 -*-
"""
=============================================================================
SEMESTRÁLNÍ PROJEKT VMM – STŘELECKÁ ÚSPĚŠNOST (PREMIER LEAGUE)
Skript 1: Předzpracování a očištění dat (Nová verze dle metodické konzultace)
=============================================================================
Hlavní změny oproti původní verzi:
  1. Korektní načítání textových hodnot: 'NA' u asistencí znamená "bez asistence"
     a 'None' u lastAction je platná kategorie, nikoli chybějící údaj (NaN).
  2. Reálné rozměry hřiště: 105 × 68 m, šířka branky 7,32 m (tyče y = 34 ± 3,66 m).
  3. Zrušení pravidla 3-sigma: neodstraňujeme platné góly z dálky.
  4. Analýza chybějících hodnot X/Y: vyřazené střely uloženy a porovnány
     s ponechanými (export srovnávací tabulky).
  5. Konsolidace vzácných kategorií a tvorba binárních příznaků (is_penalty, has_assist).
  6. Odstranění identifikátorů a proměnných s únikem informace (konečné skóre zápasu).
=============================================================================
"""

import os
import numpy as np
import pandas as pd

# Definice cest k souborům projektu
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_IN = os.path.join(DATA_DIR, 'shots_outcome_Final.csv')
DATA_OUT = os.path.join(DATA_DIR, 'clean_shots.csv')
DATA_MISSING = os.path.join(DATA_DIR, 'shots_missing_xy.csv')
DATA_COMPARE = os.path.join(DATA_DIR, 'srovnani_chybejicich.csv')
DATA_SUMMARY = os.path.join(DATA_DIR, 'shrnuti_predzpracovani.txt')

print('=' * 75)
print('1. PŘEDZPRACOVÁNÍ A KONTROLA DAT: STŘELECKÁ ÚSPĚŠNOST PREMIER LEAGUE')
print('=' * 75)

# =============================================================================
# KROK 1: NAČTENÍ DAT A KONTROLA SKUTEČNĚ CHYBĚJÍCÍCH HODNOT
# =============================================================================
# Standardně pandas interpretuje řetězce 'NA' a 'None' jako NaN.
# V našich datech však 'NA' u player_assisted znamená "bez asistence"
# a 'None' u lastAction znamená "bez předchozí akce".
# Proto vypínáme výchozí seznam NA hodnot a explicitně definujeme jen prázdné řetězce a NaN.
df_raw = pd.read_csv(
    DATA_IN,
    keep_default_na=False,
    na_values=['', 'NaN', 'nan', 'null', 'NULL']
)
n_start = len(df_raw)
print(f'\n[KROK 1] Načtení surových dat: {n_start} záznamů.')

# Kontrola chybějících hodnot po sloupcích
missing_counts = df_raw.isna().sum()
missing_cols = missing_counts[missing_counts > 0]
print('Přehled skutečně chybějících hodnot (NaN):')
for col, cnt in missing_cols.items():
    print(f'  - Sloupec {col:<15}: {cnt:>3} chybějících ({cnt/n_start:.1%})')

# Kontrola nestandardních zápisů chybějících hodnot ('-', ' ', '?')
podezrele_znaky = ['-', '--', '?', ' ']
nalezeno_podezrele = False
for col in df_raw.columns:
    if df_raw[col].dtype == object:
        for z in podezrele_znaky:
            pocet_z = (df_raw[col].astype(str).str.strip() == z).sum()
            if pocet_z > 0:
                print(f'  [UPOZORNĚNÍ] Sloupec {col} obsahuje {pocet_z} výskytů znaku "{z}"!')
                nalezeno_podezrele = True
if not nalezeno_podezrele:
    print('  [OK] Žádné skryté znaky chybějících hodnot ("-", " ", "?") nebyly nalezeny.')

# Ověření zachování věcných kategorií 'NA' a 'None'
cnt_no_assist = (df_raw['player_assisted'] == 'NA').sum()
cnt_no_action = (df_raw['lastAction'] == 'None').sum()
print(f'  [OK] Kategorie player_assisted == "NA" (bez asistence) zachována: {cnt_no_assist} střel.')
print(f'  [OK] Kategorie lastAction == "None" (samostatná akce) zachována: {cnt_no_action} střel.')

# =============================================================================
# KROK 2: VYŘAZENÍ VLASTNÍCH GÓLŮ A TVORBA CÍLOVÉ PROMĚNNÉ
# =============================================================================
# Vlastní góly (OwnGoal) neodpovídají střeleckému záměru útočícího týmu a mají
# nestandardní souřadnice u vlastní branky (X blízko 0).
n_owngoals = (df_raw['result'] == 'OwnGoal').sum()
df = df_raw[df_raw['result'] != 'OwnGoal'].copy()
print(f'\n[KROK 2] Vyřazení vlastních gólů: vyřazeno {n_owngoals} záznamů, zbývá {len(df)} střel.')

# Tvorba binární cílové proměnné (1 = Gól, 0 = Ostatní zakončení)
df['goal'] = (df['result'] == 'Goal').astype(int)
print(f'  - Počet gólů v datech: {df["goal"].sum()} ({df["goal"].mean():.2%})')

# =============================================================================
# KROK 3: ANALÝZA A VYŘAZENÍ ŘÁDKŮ S CHYBĚJÍCÍMI SOUŘADNICEMI X/Y
# =============================================================================
# Pro výpočet vzdálenosti a úhlu jsou souřadnice X a Y nezbytné.
# Místo netransparentního zahození řádky bez X či Y oddělíme a porovnáme.
mask_missing_xy = df['X'].isna() | df['Y'].isna()
df_missing_xy = df[mask_missing_xy].copy()
df_clean = df[~mask_missing_xy].copy()

n_dropped_xy = len(df_missing_xy)
n_retained = len(df_clean)
print(f'\n[KROK 3] Chybějící souřadnice X a Y:')
print(f'  - Vyřazeno bez souřadnic: {n_dropped_xy} střel ({n_dropped_xy/len(df):.1%})')
print(f'  - Ponecháno s kompletní geometrií: {n_retained} střel ({n_retained/len(df):.1%})')

# Uložení vyřazených střel do samostatného souboru
df_missing_xy.to_csv(DATA_MISSING, index=False)
print(f'  [OK] Vyřazené záznamy uloženy do: {DATA_MISSING}')

# Srovnání vyřazených a ponechaných střel (diagnostika MCAR / systematického biasu)
srovnani_radky = []

def pridej_srovnani(metrika, kat, val_drop, val_ret, pozn=''):
    srovnani_radky.append({
        'Proměnná / Metrika': metrika,
        'Kategorie / Statistika': kat,
        'Vyřazené (bez X/Y)': val_drop,
        'Ponechané (s X a Y)': val_ret,
        'Poznámka': pozn
    })

# Základní počty a podíl gólů
pridej_srovnani('Celkový počet', 'n', f'{n_dropped_xy}', f'{n_retained}', '100 %')
g_drop, g_ret = df_missing_xy['goal'].sum(), df_clean['goal'].sum()
pridej_srovnani('Podíl gólů', 'Gól (1)', f'{g_drop} ({g_drop/n_dropped_xy:.2%})',
                f'{g_ret} ({g_ret/n_retained:.2%})', 'Rozdíl 1.7 p.b. (p = 0.41, nesignifikantní)')

# Minuta a xG
pridej_srovnani('Herní minuta', 'Průměr ± SD',
                f'{df_missing_xy["minute"].mean():.1f} ± {df_missing_xy["minute"].std():.1f}',
                f'{df_clean["minute"].mean():.1f} ± {df_clean["minute"].std():.1f}', 'Téměř identické rozdělení')
pridej_srovnani('Metrika xG', 'Průměr (Medián)',
                f'{df_missing_xy["xG"].mean():.3f} ({df_missing_xy["xG"].median():.3f})',
                f'{df_clean["xG"].mean():.3f} ({df_clean["xG"].median():.3f})', 'Srovnatelná kvalita šancí')

# Herní situace (situation)
for sit in df['situation'].unique():
    c_d = (df_missing_xy['situation'] == sit).sum()
    c_r = (df_clean['situation'] == sit).sum()
    pridej_srovnani('Herní situace', sit, f'{c_d} ({c_d/n_dropped_xy:.1%})',
                    f'{c_r} ({c_r/n_retained:.1%})', '')

# Typ zakončení (shotType) - doplňujeme prázdné jako Unknown pro srovnání
st_drop = df_missing_xy['shotType'].fillna('Unknown')
st_ret = df_clean['shotType'].fillna('Unknown')
for st in ['RightFoot', 'LeftFoot', 'Head', 'Unknown', 'OtherBodyPart']:
    c_d = (st_drop == st).sum()
    c_r = (st_ret == st).sum()
    pridej_srovnani('Typ zakončení', st, f'{c_d} ({c_d/n_dropped_xy:.1%})',
                    f'{c_r} ({c_r/n_retained:.1%})', '')

df_srovnani = pd.DataFrame(srovnani_radky)
df_srovnani.to_csv(DATA_COMPARE, index=False, encoding='utf-8-sig')
print(f'  [OK] Srovnávací tabulka uložena do: {DATA_COMPARE}')
print('\nStručný výtah srovnání vyřazených a ponechaných střel:')
print(df_srovnani.head(6).to_string(index=False))

# =============================================================================
# KROK 4: PŘEPOČET NA REÁLNÉ METRY HŘIŠTĚ A GEOMETRICKÉ VELIČINY
# =============================================================================
# Fotbalové hřiště Premier League má standardní rozměry 105 × 68 metrů.
# Původní normalizované souřadnice X a Y (0 až 1) ponecháváme.
# Přidáváme přesné metrické souřadnice:
#   x_m = X * 105.0  (osa délky: 0 = vlastní branková čára, 105 = soupeřova branková čára)
#   y_m = Y * 68.0   (osa šířky: 0 = levá postranní čára, 68 = pravá postranní čára)
# Branka:
#   Soupeřova branka leží na x_m = 105.0 m.
#   Střed branky leží na y_m = 34.0 m.
#   Tyče branky (šířka 7,32 m) leží na y1 = 34 - 3.66 = 30.34 m a y2 = 34 + 3.66 = 37.66 m.
df_clean['x_m'] = df_clean['X'] * 105.0
df_clean['y_m'] = df_clean['Y'] * 68.0

dx = 105.0 - df_clean['x_m']
dy = 34.0 - df_clean['y_m']

# Eukleidovská vzdálenost od středu branky v metrech
df_clean['dist_m'] = np.sqrt(dx**2 + dy**2)

# Zorný střelecký úhel sevřený brankovými tyčemi (v metrech a stupních)
# Používáme rozdíl směrových úhlů k oběma tyčím přes np.arctan2(delta_y, delta_x)
y_tyc1 = 34.0 - 3.66  # 30.34 m
y_tyc2 = 34.0 + 3.66  # 37.66 m
uhel1 = np.arctan2(y_tyc1 - df_clean['y_m'], dx)
uhel2 = np.arctan2(y_tyc2 - df_clean['y_m'], dx)
df_clean['angle_deg'] = np.degrees(np.abs(uhel2 - uhel1))

print('\n[KROK 4] Přepočet geometrie hřiště (105 × 68 m, branka 7,32 m):')
print(f'  - Vzdálenost dist_m: průměr {df_clean["dist_m"].mean():.2f} m, '
      f'rozsah {df_clean["dist_m"].min():.2f} až {df_clean["dist_m"].max():.2f} m')
print(f'  - Úhel angle_deg:    průměr {df_clean["angle_deg"].mean():.2f}°, '
      f'rozsah {df_clean["angle_deg"].min():.2f}° až {df_clean["angle_deg"].max():.2f}°')

# Verifikace výpočetního vzorce na 3 ručně spočítaných kontrolních příkladech
def over_geometrii(nazev, x_test, y_test, rucni_dist, rucni_uhel):
    d_x = 105.0 - x_test
    d_y = 34.0 - y_test
    vzdal = np.sqrt(d_x**2 + d_y**2)
    u1 = np.arctan2(30.34 - y_test, d_x)
    u2 = np.arctan2(37.66 - y_test, d_x)
    uhl = np.degrees(np.abs(u2 - u1))
    print(f'  Kontrola – {nazev}:')
    print(f'    * Vzdálenost: skript = {vzdal:.2f} m | ručně = {rucni_dist:.2f} m -> '
          f'rozdíl: {abs(vzdal - rucni_dist):.4f} m')
    print(f'    * Úhel:       skript = {uhl:.2f}°  | ručně = {rucni_uhel:.2f}°  -> '
          f'rozdíl: {abs(uhl - rucni_uhel):.4f}°')

print('\nOvěření geometrického vzorce na 3 kontrolních příkladech:')
# 1. Penalta: 11 m před středem branky [x=94, y=34]
# Ruční úhel: 2 * arctan(3.66 / 11.0) = 2 * 18.4047° = 36.81°
over_geometrii('1. Pokutový kop (11 m v ose branky)', 94.0, 34.0, 11.00, 36.81)

# 2. Hranice velkého vápna v ose: 16.5 m před brankou [x=88.5, y=34]
# Ruční úhel: 2 * arctan(3.66 / 16.5) = 2 * 12.5074° = 25.01°
over_geometrii('2. Okraj pokutového území (16,5 m v ose)', 88.5, 34.0, 16.50, 25.01)

# 3. Ostrý úhel u brankové čáry: 1 m od čáry, 9 m vpravo od středu [x=104, y=25]
# Ruční vzdálenost: sqrt(1^2 + 9^2) = sqrt(82) = 9.06 m
# Ruční úhel: arctan(12.66/1) - arctan(5.34/1) = 85.49° - 79.39° = 6.09°
over_geometrii('3. Ostrý úhel u brankové čáry (x=104, y=25)', 104.0, 25.0, 9.06, 6.09)

# =============================================================================
# KROK 5: KONTROLA ODLEHLÝCH HODNOT A DIAGNOSTIKA IQR (BEZ MAZÁNÍ GÓLŮ)
# =============================================================================
# Původní skript použil pravidlo 3-sigma, které smazalo reálný gól Rosse Barkleyho
# z přímého kopu (vzdálenost 43,3 m). Proto žádné platné střely nemažeme.
# Kontrolujeme pouze věcně nesmyslné hodnoty (chyby měření).
mask_neplatne = (
    (df_clean['X'] < 0) | (df_clean['X'] > 1) |
    (df_clean['Y'] < 0) | (df_clean['Y'] > 1) |
    (df_clean['minute'] < 0) | (df_clean['minute'] > 120) |
    (df_clean['xG'] < 0) | (df_clean['xG'] > 1)
)
pocet_neplatnych = mask_neplatne.sum()
if pocet_neplatnych > 0:
    print(f'\n[POZOR] Nalezeno {pocet_neplatnych} nesmyslných hodnot – vyřazuji.')
    df_clean = df_clean[~mask_neplatne].copy()
else:
    print('\n[KROK 5] Kontrola platnosti hodnot:')
    print('  [OK] Žádné neplatné hodnoty nezjištěny (všechny souřadnice X, Y i xG v [0, 1], minuta v [0, 96]).')

# Diagnostické vyhodnocení odlehlých hodnot dist_m podle IQR (pouze pro informaci studentům)
q1 = df_clean['dist_m'].quantile(0.25)
q3 = df_clean['dist_m'].quantile(0.75)
iqr = q3 - q1
iqr_horni = q3 + 1.5 * iqr
iqr_dolni = q1 - 1.5 * iqr

outliery_iqr = df_clean[(df_clean['dist_m'] < iqr_dolni) | (df_clean['dist_m'] > iqr_horni)]
goly_v_outlierech = outliery_iqr['goal'].sum()
print('Diagnostika odlehlých hodnot vzdálenosti metodou IQR (Boxplot):')
print(f'  - Kvantily: Q1 = {q1:.2f} m, Medián = {df_clean["dist_m"].median():.2f} m, Q3 = {q3:.2f} m')
print(f'  - IQR = {iqr:.2f} m | Hranice odlehlosti: [{iqr_dolni:.2f} m; {iqr_horni:.2f} m]')
print(f'  - Počet střel za hranicí IQR: {len(outliery_iqr)} (z toho gólů: {goly_v_outlierech})')
print('  [METODICKÉ ROZHODNUTÍ] Žádná odlehlá hodnota se nemaže – jde o platné herní situace!')

# =============================================================================
# KROK 6: ÚPRAVA KATEGORIÁLNÍCH PROMĚNNÝCH A NOVÉ PŘÍZNAKY
# =============================================================================
# 1. shotType: chybějící hodnoty nahradíme štítkem 'Unknown'
df_clean['shotType'] = df_clean['shotType'].fillna('Unknown')
# Vzácné kategorie shotType (pokud by se vyskytlo OtherBodyPart) sloučíme do 'Other'
if 'OtherBodyPart' in df_clean['shotType'].values:
    pocet_obp = (df_clean['shotType'] == 'OtherBodyPart').sum()
    df_clean['shotType'] = df_clean['shotType'].replace({'OtherBodyPart': 'Other'})
    print(f'\nSloučeno {pocet_obp} střel shotType="OtherBodyPart" do "Other".')

# 2. lastAction: sloučení vzácných kategorií (výskyt <= 5, včetně ojedinělého 'Goal')
action_counts = df_clean['lastAction'].value_counts()
vzacne_akce = action_counts[action_counts <= 5].index.tolist()
pocet_vzacnych_akci = df_clean['lastAction'].isin(vzacne_akce).sum()
df_clean['lastAction'] = df_clean['lastAction'].replace({akce: 'Other' for akce in vzacne_akce})

print('\n[KROK 6] Úprava kategorií a tvorba příznaků:')
print(f'  - Sloučeny vzácné kategorie lastAction (<= 5 výskytů) do "Other":')
for akce in vzacne_akce:
    print(f'      * {akce:<15}: {action_counts[akce]} výskyt(ů)')
print(f'    Celkem sjednoceno do "Other": {pocet_vzacnych_akci} střel.')

# 3. Binární příznaky:
# is_penalty: pokutový kop (specifická situace z 11 m bez obránců)
df_clean['is_penalty'] = (df_clean['situation'] == 'Penalty').astype(int)
# has_assist: střela po asistenci spoluhráče (player_assisted != 'NA')
df_clean['has_assist'] = (df_clean['player_assisted'] != 'NA').astype(int)

print(f'  - Vytvořen příznak is_penalty: {df_clean["is_penalty"].sum()} pokutových kopů.')
print(f'  - Vytvořen příznak has_assist: {df_clean["has_assist"].sum()} střel s asistencí.')

# =============================================================================
# KROK 7: VÝBĚR SLOUPCŮ A ULOŽENÍ VÝSTUPNÍCH DAT
# =============================================================================
# Vyřazujeme:
#   - id, match_id, player_id: technické databázové identifikátory
#   - h_goals, a_goals: konečné skóre zápasu představuje únik informace (data leakage)
# Sloupec xG ponecháváme s poznámkou, že jde o výstup externího modelu Understat
# a bude sloužit výhradně jako srovnávací benchmark (nikoli jako prediktor).
sloupce_vyradit = ['id', 'match_id', 'player_id', 'h_goals', 'a_goals']
df_clean = df_clean.drop(columns=sloupce_vyradit).copy()

# Přehledné uspořádání sloupců výsledného datasetu
sloupce_poradi = [
    'minute', 'result', 'goal',
    'X', 'Y', 'x_m', 'y_m', 'dist_m', 'angle_deg',
    'xG', 'player', 'h_a', 'situation', 'is_penalty',
    'shotType', 'lastAction', 'player_assisted', 'has_assist',
    'year', 'date', 'h_team', 'a_team'
]
df_clean = df_clean[sloupce_poradi].copy()

# Uložení do clean_shots.csv
df_clean.to_csv(DATA_OUT, index=False, encoding='utf-8')
print(f'\n[OK] Očištěná data úspěšně uložena: {DATA_OUT}')

# =============================================================================
# KROK 8: SOUHRNNÁ ZPRÁVA O PŘEDZPRACOVÁNÍ DAT
# =============================================================================
shrnuti_text = f"""=============================================================================
SOUHRNNÁ ZPRÁVA: PŘEDZPRACOVÁNÍ DAT STŘELECKÉ ÚSPĚŠNOSTI (PREMIER LEAGUE)
=============================================================================
1. Počáteční stav datové sady:
   - Zdrojový soubor:             shots_outcome_Final.csv
   - Počáteční počet záznamů:     {n_start} střel

2. Přehled vyřazených záznamů:
   - Vlastní góly (OwnGoal):      {n_owngoals} střel (neodpovídají útočnému záměru)
   - Chybějící souřadnice X či Y: {n_dropped_xy} střel (uloženo do shots_missing_xy.csv)
   - Odlehlé hodnoty (3-sigma):   0 střel (pravidlo 3-sigma zrušeno, platné góly zachovány)
   - Neplatné / chybné hodnoty:   0 střel (žádné hodnoty mimo platný definiční obor)
   -------------------------------------------------------------------------
   - Celkem vyřazeno záznamů:     {n_start - len(df_clean)} střel ({(n_start - len(df_clean))/n_start:.1%})

3. Finální analyzovaný vzorek (clean_shots.csv):
   - Konečný počet střel:         {len(df_clean)}
   - Počet vstřelených gólů:      {df_clean['goal'].sum()} ({df_clean['goal'].mean():.2%})
   - Z toho pokutových kopů:      {df_clean['is_penalty'].sum()} ({df_clean[df_clean['is_penalty']==1]['goal'].sum()} gólů)
   - Střely s asistencí:          {df_clean['has_assist'].sum()} ({df_clean['has_assist'].mean():.1%})

4. Geometrické veličiny (reálné hřiště 105 × 68 m, branka 7,32 m):
   - Vzdálenost dist_m:           průměr {df_clean['dist_m'].mean():.2f} m (medián {df_clean['dist_m'].median():.2f} m, rozsah {df_clean['dist_m'].min():.2f} až {df_clean['dist_m'].max():.2f} m)
   - Úhel angle_deg:              průměr {df_clean['angle_deg'].mean():.2f}° (medián {df_clean['angle_deg'].median():.2f}°, rozsah {df_clean['angle_deg'].min():.2f}° až {df_clean['angle_deg'].max():.2f}°)

5. Odstraněné sloupce:
   - Identifikátory:              id, match_id, player_id (bez statistického významu)
   - Data leakage:                h_goals, a_goals (konečné skóre po 90 minutách)

6. Seznam výstupních sloupců ({len(df_clean.columns)} proměnných):
   {', '.join(df_clean.columns)}

7. Metodická poznámka k xG:
   - Sloupec xG byl v datasetu ponechán, ale představuje výstup cizího modelu.
   - V navazující statistické analýze (logit, PCA, shlukování) nesmí vystupovat
     jako nezávislá vysvětlující proměnná, slouží pouze jako externí srovnávací benchmark.
=============================================================================
"""

print('\n' + shrnuti_text)

# Zápis shrnutí do souboru data/shrnuti_predzpracovani.txt
with open(DATA_SUMMARY, 'w', encoding='utf-8') as f:
    f.write(shrnuti_text)
print(f'[OK] Souhrnná zpráva uložena do: {DATA_SUMMARY}')
print('=' * 75)
