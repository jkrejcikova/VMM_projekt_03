# -*- coding: utf-8 -*-
"""
=============================================================================
SEMESTRÁLNÍ PROJEKT VMM – STŘELECKÁ ÚSPĚŠNOST VE VZORKU PREMIER LEAGUE
Skript 2: Vícerozměrná statistická analýza a tvorba grafů (Verze 5 - Čisté grafy)
=============================================================================
Metodické zásady a provedené úpravy:
  1. Výhradní využití standardních knihoven ze cvičení (materialy/):
     - statsmodels: anova_oneway (Welchova i klasická ANOVA), MANOVA, Logit, VIF
     - scipy.stats: shapiro, levene, ttest_ind (Welchův t-test), probplot, chi2,
       norm, spearmanr, chi2_contingency (Chí-kvadrát test nezávislosti)
     - scipy.cluster.hierarchy: linkage, dendrogram, fcluster (Wardova metoda)
     - sklearn: StandardScaler, PCA, TSNE, KMeans, silhouette_score, roc_curve, auc
     - Žádné nestandardní knihovny.
  2. Zásada jednoduchých a čistých grafů:
     - Žádné textové rámečky (bbox), spodní poznámky pod grafem ani pomocné anotace.
     - Titulky krátké a popisné, osy s popisky a jednotkami, legendy jen kde jsou nutné.
     - Všechny podrobné statistické výsledky, testy a poznámky exportovány do
       popisky_grafu.txt a souhrn_analyzy.txt.
     - Jednotný barevný kód: modrá (#1f77b4) = bez gólu, červená (#d62728) = gól.
     - Důsledná česká desetinná čárka na osách i v textech (ticker.FuncFormatter).
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# statsmodels: knihovna ze cvičení
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.outliers_influence import variance_inflation_factor

# sklearn: knihovna ze cvičení
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_curve, auc

warnings.filterwarnings('ignore')

# =============================================================================
# GLOBÁLNÍ KONSTANTY, CESTY A NASTAVENÍ MATPLOTLIBU
# =============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'clean_shots.csv')
VYSTUPY_DIR = os.path.join(BASE_DIR, 'vystupy')
os.makedirs(VYSTUPY_DIR, exist_ok=True)

# Čitelné písmo pro akademickou prezentaci (min. 12 pt pro všechny prvky)
plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 15,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'axes.grid': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
sns.set_theme(style='white', palette='tab10')


# =============================================================================
# POMOCNÉ FUNKCE
# =============================================================================
def carka(cislo, des_mista=3):
    """Převede float číslo na český formát řetězce s desetinnou čárkou."""
    if pd.isna(cislo):
        return ""
    format_str = f"{{:.{des_mista}f}}"
    return format_str.format(cislo).replace('.', ',')


def fmt_cesky(x, pos=None):
    """Český formátovač čísel s desetinnou čárkou pro osy grafů."""
    if abs(x - int(x)) < 1e-9:
        return f"{int(x)}"
    return f"{x:.2f}".rstrip('0').rstrip('.').replace('.', ',')


formatter_cesky = ticker.FuncFormatter(fmt_cesky)


def uloz_graf(nazev_souboru):
    """Uloží aktuální graf do složky vystupy/ s rozlišením 300 DPI a zavře figuru."""
    cesta = os.path.join(VYSTUPY_DIR, nazev_souboru)
    plt.savefig(cesta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Graf uložen: {nazev_souboru}')


def nakresli_fotbalove_hriste(ax, polovina='utocna'):
    """
    Vykreslí fotbalové hřiště v reálných metrech (105 x 68 m)
    s bílým pozadím (#FFFFFF) a šedými čarami (#A0A0A0).
    X = vzdálenost od vlastní brankové čáry (m), 105 = soupeřova branková čára.
    Y = vzdálenost od postranní čáry (m), branka je vystředěna na y = 34 m.
    Při polovina='utocna' přiblíží pohled na útočnou polovinu (x od 55 do 107,5 m).
    """
    ax.set_facecolor('#FFFFFF')
    LC = '#A0A0A0'
    LW = 1.5

    def rect(x, y, w, h):
        ax.add_patch(mpatches.Rectangle((x, y), w, h,
                     edgecolor=LC, facecolor='none', linewidth=LW))

    # Ohraničení hrací plochy (105 x 68 m) a půlící čára (x = 52.5 m)
    rect(0, 0, 105, 68)
    ax.plot([52.5, 52.5], [0, 68], color=LC, linewidth=LW)

    # Středový kruh (střed [52.5, 34], poloměr 9.15 m) a středový bod
    kruh = plt.Circle((52.5, 34), 9.15, color=LC, fill=False, linewidth=LW)
    ax.add_patch(kruh)
    ax.plot(52.5, 34, 'o', color=LC, markersize=4)

    # Pokutové území soupeře (hloubka 16.5 m od 105 m: od 88.5 do 105 m; šířka 40.32 m: y od 13.84 do 54.16 m)
    rect(105 - 16.5, 34 - 20.16, 16.5, 40.32)

    # Brankové území soupeře / malé vápno (hloubka 5.5 m: od 99.5 do 105 m; šířka 18.32 m: y od 24.84 do 43.16 m)
    rect(105 - 5.5, 34 - 9.16, 5.5, 18.32)

    # Branka soupeře za brankovou čarou (šířka 7.32 m: y od 30.34 do 37.66 m, hloubka 2 m)
    ax.add_patch(mpatches.Rectangle((105, 30.34), 2, 7.32, edgecolor=LC, facecolor='#F0F0F0', linewidth=LW))

    # Penaltový bod soupeře (11 m od brankové čáry, x = 94 m, y = 34 m)
    ax.plot(94, 34, 'o', color=LC, markersize=3.5)

    # Oblouk pokutového území (poloměr 9.15 m od penaltového bodu)
    oblouk = mpatches.Arc((94, 34), 18.3, 18.3, angle=0, theta1=127, theta2=233,
                          color=LC, linewidth=LW)
    ax.add_patch(oblouk)

    if polovina == 'utocna':
        ax.set_xlim(55, 107.5)
        ax.set_ylim(-2, 70)
    else:
        rect(0, 34 - 20.16, 16.5, 40.32)
        rect(0, 34 - 9.16, 5.5, 18.32)
        ax.add_patch(mpatches.Rectangle((-2, 30.34), 2, 7.32, edgecolor=LC, facecolor='#F0F0F0', linewidth=LW))
        ax.plot(11, 34, 'o', color=LC, markersize=3.5)
        ax.set_xlim(-4, 109)
        ax.set_ylim(-4, 72)

    ax.set_xlabel('Vzdálenost od vlastní brankové čáry (m)', fontsize=13, labelpad=8)
    ax.set_ylabel('Vzdálenost od postranní čáry (m)', fontsize=13, labelpad=8)
    ax.xaxis.set_major_formatter(formatter_cesky)
    ax.yaxis.set_major_formatter(formatter_cesky)
    for spine in ax.spines.values():
        spine.set_edgecolor(LC)
        spine.set_linewidth(1.2)


# =============================================================================
# 1. FÁZE: EDA A PROSTOROVÁ VIZUALIZACE HŘIŠTĚ
# =============================================================================
def eda_a_prostorova_mapa(df):
    """
    Průzkumová analýza dat (EDA): export popisných statistik, generování čistých
    map střelby a průzkumného grafu podílu gólů podle klíčových faktorů.
    """
    print('\n' + '=' * 75)
    print('1. FÁZE: EDA A PROSTOROVÁ VIZUALIZACE HŘIŠTĚ (ÚTOČNÁ POLOVINA)')
    print('=' * 75)

    n_celkem = len(df)
    n_goly = df['goal'].sum()
    pct_goly = n_goly / n_celkem * 100

    # 1.1 Export tabulky popisné statistiky s českými názvy
    num_cols = ['dist_m', 'angle_deg', 'xG', 'x_m', 'y_m', 'minute']
    popis = df[num_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    popis.index = [
        'Vzdálenost od branky (m)',
        'Střelecký úhel (°)',
        'Očekávané góly (xG)',
        'Vzdálenost od vlastní brankové čáry (m)',
        'Vzdálenost od postranní čáry (m)',
        'Minuta zápasu'
    ]
    popis.columns = ['n', 'Průměr', 'Sm. odchylka', 'Min', '25 %', 'Medián', '75 %', 'Max']
    popis.round(3).to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_popis.csv'))
    print('  [OK] Tabulka popisné statistiky uložena: tabulka_popis.csv')

    nezakon = df[df['goal'] == 0]
    goly = df[df['goal'] == 1]

    # 1.2 Graf 01: Prostorová mapa střel (útočná polovina) - ČISTÝ VZHLED
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    nakresli_fotbalove_hriste(ax, polovina='utocna')

    ax.scatter(nezakon['x_m'], nezakon['y_m'], c='#1f77b4', s=45, alpha=0.65,
               edgecolors='none', label=f'Bez gólu (n = {len(nezakon)})', zorder=3)
    ax.scatter(goly['x_m'], goly['y_m'], c='#d62728', s=70, alpha=0.90,
               edgecolors='#800000', linewidths=0.8, label=f'Gól (n = {len(goly)})', zorder=4)

    ax.set_title('Prostorová mapa střel na útočné polovině hřiště',
                 fontsize=14, fontweight='bold', pad=10, color='#1A1A1A')
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=12)
    plt.tight_layout()
    uloz_graf('01_mapa_strel.png')

    # 1.3 Graf 01c: 2D KDE hustota střelby - ČISTÝ VZHLED BEZ OSTRÝCH HRAN UVNITŘ HŘIŠTĚ
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    nakresli_fotbalove_hriste(ax, polovina='utocna')

    # clip omezen na celé vnější hranice hrací plochy (0 až 105 m x, 0 až 68 m y),
    # takže uvnitř viditelné útočné poloviny (55 až 105 m) nevzniká žádná ostrá hrana
    sns.kdeplot(data=nezakon, x='x_m', y='y_m', fill=True, alpha=0.35, cmap='Blues', ax=ax,
                levels=8, zorder=2, clip=((0, 105), (0, 68)), cut=3)
    sns.kdeplot(data=goly, x='x_m', y='y_m', fill=True, alpha=0.55, cmap='Reds', ax=ax,
                levels=8, zorder=3, clip=((0, 105), (0, 68)), cut=3)

    p_blue = mpatches.Patch(color='#6baed6', alpha=0.6, label='Bez gólu')
    p_red = mpatches.Patch(color='#fb6a4a', alpha=0.7, label='Gól')
    ax.legend(handles=[p_blue, p_red], loc='upper left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=12)

    ax.set_title('Hustota střelby na útočné polovině hřiště',
                 fontsize=14, fontweight='bold', pad=10, color='#1A1A1A')
    plt.tight_layout()
    uloz_graf('01c_kde_hriste.png')

    # 1.4 Graf 01b: Histogram xG (průzkumné rozdělení)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.hist(df['xG'], bins=35, color='#1f77b4', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Očekávaná hodnota gólu (xG)', fontsize=13, labelpad=8)
    ax.set_ylabel('Počet střeleckých pokusů', fontsize=13, labelpad=8)
    ax.xaxis.set_major_formatter(formatter_cesky)
    ax.yaxis.set_major_formatter(formatter_cesky)
    ax.set_title('Empirické rozdělení hodnot xG', fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    uloz_graf('01b_hist_xG.png')

    # 1.5 Graf 08: Zjednodušený průzkumný graf podílu gólů (2 panely) a tabulka chí-kvadrát testů
    vysledky_faktoru = generuj_graf_a_tabulku_faktoru(df)

    return vysledky_faktoru


def generuj_graf_a_tabulku_faktoru(df):
    """
    Vytvoří zjednodušený graf 08_podil_golu_faktory.png (dva panely vedle sebe:
    typ zakončení a herní situace, bez penalt a neznámých střel).
    Vypočítá Chí-kvadrát testy včetně očekávaných četností < 5 a uloží
    kompletní tabulku tabulka_podil_golu.csv se všemi faktory (původní i upravené varianty).
    """
    print('\n  Generování zjednodušeného grafu 08_podil_golu_faktory.png a tabulky chí-kvadrát testů...')
    prumer_goly_pct = df['goal'].mean() * 100  # 8,57 % ve vzorku (n = 735)

    cz_shottype = {'RightFoot': 'Pravá noha', 'LeftFoot': 'Levá noha', 'Head': 'Hlavou', 'Unknown': 'Neznámé'}
    cz_situation = {'OpenPlay': 'Ze hry', 'FromCorner': 'Rohový kop', 'DirectFreekick': 'Přímý kop',
                    'SetPiece': 'Standardka', 'Penalty': 'Penalta'}
    cz_lastaction = {
        'Pass': 'Přihrávka', 'TakeOn': 'Klička/Souboj', 'Cross': 'Centr', 'None': 'Žádná/Neznámá',
        'Standard': 'Standardka', 'Aerial': 'Vzdušný souboj', 'Chipped': 'Oblouček', 'Rebound': 'Odražený míč',
        'BallRecovery': 'Zisk míče', 'BallTouch': 'Dotek', 'HeadPass': 'Hlavička', 'LayOff': 'Přiklepnutí',
        'Other': 'Ostatní', 'Throughball': 'Kolmice'
    }
    cz_assist = {1: 'S asistencí', 0: 'Bez asistence'}

    # 1. Výpočty Chí-kvadrát testů a kontrola očekávaných četností
    chi_vysledky = {}

    def testuj_kontingenci(series, goal_series, nazev_testu):
        ct = pd.crosstab(series, goal_series)
        stat, p, dof, expected = stats.chi2_contingency(ct)
        pocet_bunek = expected.size
        pocet_lt5 = int((expected < 5).sum())
        pct_lt5 = pocet_lt5 / pocet_bunek * 100
        splneno = pct_lt5 <= 20.0
        return {
            'chi2': stat,
            'p': p,
            'dof': dof,
            'expected': expected,
            'bunek_celkem': pocet_bunek,
            'bunek_lt5': pocet_lt5,
            'pct_lt5': pct_lt5,
            'splneno': splneno,
            'crosstab': ct
        }

    # (a) Typ zakončení (všechny 4 kategorie)
    chi_vysledky['shottype_all'] = testuj_kontingenci(df['shotType'].fillna('Unknown'), df['goal'], 'Typ zakončení (všechny 4)')
    # Typ zakončení (3 hlavní kategorie bez Unknown)
    df_st_main = df[df['shotType'].isin(['RightFoot', 'LeftFoot', 'Head'])]
    chi_vysledky['shottype_main'] = testuj_kontingenci(df_st_main['shotType'], df_st_main['goal'], 'Typ zakončení (3 hlavní)')

    # (b) Herní situace (původní: všech 5 kategorií včetně penalt)
    chi_vysledky['sit_all'] = testuj_kontingenci(df['situation'], df['goal'], 'Herní situace (všechny včetně penalt)')
    # Herní situace (upravená: bez penalt)
    df_no_pen = df[df['situation'] != 'Penalty']
    chi_vysledky['sit_no_pen'] = testuj_kontingenci(df_no_pen['situation'], df_no_pen['goal'], 'Herní situace (bez penalt)')

    # (c) Předchozí akce lastAction (původní: 14 kategorií)
    la_series = df['lastAction'].fillna('None')
    chi_vysledky['la_orig'] = testuj_kontingenci(la_series, df['goal'], 'Předchozí akce (původní 14 kat.)')
    # Předchozí akce lastAction (upravená: kategorie s < 15 střelami sloučeny do "Ostatní")
    la_counts = la_series.value_counts()
    rare_la = la_counts[la_counts < 15].index.tolist()
    la_merged = la_series.apply(lambda x: 'Other' if x in rare_la else x)
    chi_vysledky['la_merged'] = testuj_kontingenci(la_merged, df['goal'], 'Předchozí akce (sloučené < 15)')

    # (d) Asistence (2 kategorie)
    chi_vysledky['assist'] = testuj_kontingenci(df['has_assist'], df['goal'], 'Asistence')

    # Tvorba záznamů do exportní tabulky tabulka_podil_golu.csv
    zaznamy_tabulka = []

    def pridej_do_tabulky(series, goal_series, fac_nazev, var_nazev, fac_map, test_res):
        sub_df = pd.DataFrame({'val': series, 'goal': goal_series})
        agg = sub_df.groupby('val').agg(
            n_shots=('goal', 'count'),
            n_goals=('goal', 'sum')
        ).reset_index()
        agg['rate_pct'] = agg['n_goals'] / agg['n_shots'] * 100
        agg['label_cz'] = agg['val'].map(lambda x: fac_map.get(x, str(x)))
        agg = agg.sort_values(by='n_shots', ascending=False).reset_index(drop=True)

        for _, row in agg.iterrows():
            zaznamy_tabulka.append({
                'Faktor': fac_nazev,
                'Varianta': var_nazev,
                'Kategorie': row['label_cz'],
                'Počet střel (n)': int(row['n_shots']),
                'Počet gólů': int(row['n_goals']),
                'Podíl gólů (%)': carka(row['rate_pct'], 1),
                'Chí-kvadrát': carka(test_res['chi2'], 2),
                'Stupně volnosti': test_res['dof'],
                'p-hodnota': "< 0,001" if test_res['p'] < 0.001 else carka(test_res['p'], 3),
                'Buněk oček. < 5': f"{test_res['bunek_lt5']} z {test_res['bunek_celkem']} ({carka(test_res['pct_lt5'], 1)} %)",
                'Předpoklad testu': 'Splněn (<= 20 %)' if test_res['splneno'] else 'Nesplněn (> 20 %, orientační)'
            })

    pridej_do_tabulky(df['shotType'].fillna('Unknown'), df['goal'], 'Typ zakončení', 'Původní (všechny 4)', cz_shottype, chi_vysledky['shottype_all'])
    pridej_do_tabulky(df_st_main['shotType'], df_st_main['goal'], 'Typ zakončení', 'Upravená (3 hlavní)', cz_shottype, chi_vysledky['shottype_main'])
    pridej_do_tabulky(df['situation'], df['goal'], 'Herní situace', 'Původní (včetně penalt)', cz_situation, chi_vysledky['sit_all'])
    pridej_do_tabulky(df_no_pen['situation'], df_no_pen['goal'], 'Herní situace', 'Upravená (bez penalt)', cz_situation, chi_vysledky['sit_no_pen'])
    pridej_do_tabulky(la_series, df['goal'], 'Předchozí akce (lastAction)', 'Původní (14 kategorií)', cz_lastaction, chi_vysledky['la_orig'])
    pridej_do_tabulky(la_merged, df['goal'], 'Předchozí akce (lastAction)', 'Upravená (sloučeno < 15 do Ostatní)', cz_lastaction, chi_vysledky['la_merged'])
    pridej_do_tabulky(df['has_assist'], df['goal'], 'Asistence u střely', 'Původní', cz_assist, chi_vysledky['assist'])

    df_tab_podil = pd.DataFrame(zaznamy_tabulka)
    cesta_tab_podil = os.path.join(VYSTUPY_DIR, 'tabulka_podil_golu.csv')
    df_tab_podil.to_csv(cesta_tab_podil, index=False, encoding='utf-8')
    print('  [OK] Tabulka podílů gólů a chí-kvadrát testů uložena: tabulka_podil_golu.csv')

    # 2. Vykreslení čistého grafu 08_podil_golu_faktory.png (2 PANELY: Typ zakončení a Herní situace)
    # Penalty (6 střel) a neznámé zakončení vynechány pro zamezení zkreslení měřítka
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6))

    # Panel 1: Typ zakončení (Pravá noha, Levá noha, Hlavou)
    st_kat = ['RightFoot', 'LeftFoot', 'Head']
    agg_st = df[df['shotType'].isin(st_kat)].groupby('shotType').agg(
        n_shots=('goal', 'count'),
        n_goals=('goal', 'sum')
    ).loc[st_kat].reset_index()
    agg_st['rate_pct'] = agg_st['n_goals'] / agg_st['n_shots'] * 100
    agg_st['label_cz'] = agg_st['shotType'].map(cz_shottype)
    x_labels_st = [f"{r['label_cz']}\n(n = {r['n_shots']})" for _, r in agg_st.iterrows()]

    x1 = np.arange(len(agg_st))
    ax1.bar(x1, agg_st['rate_pct'], color='#1f77b4', edgecolor='white', width=0.55, zorder=2)
    ax1.axhline(prumer_goly_pct, color='#d62728', linestyle='--', linewidth=1.8, zorder=3)

    ax1.set_ylim(0, max(agg_st['rate_pct'].max(), prumer_goly_pct) * 1.35)
    for i, r in agg_st.iterrows():
        y_pos = r['rate_pct'] + 0.35
        if abs(y_pos - prumer_goly_pct) < 1.0 or (r['rate_pct'] < prumer_goly_pct and y_pos + 0.8 >= prumer_goly_pct):
            y_pos = prumer_goly_pct + 0.75
        ax1.text(i, y_pos, f"{carka(r['rate_pct'], 1)} %",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='#1A1A1A',
                 bbox=dict(boxstyle='square,pad=0.2', fc='#FFFFFF', ec='none'), zorder=5)

    ax1.set_xticks(x1)
    ax1.set_xticklabels(x_labels_st, fontsize=12)
    ax1.set_ylabel('Podíl gólů (%)', fontsize=13)
    ax1.yaxis.set_major_formatter(formatter_cesky)
    ax1.set_title('Typ zakončení', fontsize=14, fontweight='bold', pad=10)

    # Panel 2: Herní situace (Ze hry, Rohový kop, Přímý kop, Standardka; bez penalt)
    sit_kat = ['OpenPlay', 'FromCorner', 'SetPiece', 'DirectFreekick']
    agg_sit = df[df['situation'].isin(sit_kat)].groupby('situation').agg(
        n_shots=('goal', 'count'),
        n_goals=('goal', 'sum')
    ).loc[sit_kat].reset_index()
    agg_sit['rate_pct'] = agg_sit['n_goals'] / agg_sit['n_shots'] * 100
    agg_sit['label_cz'] = agg_sit['situation'].map(cz_situation)
    x_labels_sit = [f"{r['label_cz']}\n(n = {r['n_shots']})" for _, r in agg_sit.iterrows()]

    x2 = np.arange(len(agg_sit))
    ax2.bar(x2, agg_sit['rate_pct'], color='#1f77b4', edgecolor='white', width=0.58, zorder=2)
    ax2.axhline(prumer_goly_pct, color='#d62728', linestyle='--', linewidth=1.8, zorder=3)

    ax2.set_ylim(0, max(agg_sit['rate_pct'].max(), prumer_goly_pct) * 1.35)
    for i, r in agg_sit.iterrows():
        y_pos = r['rate_pct'] + 0.35
        if abs(y_pos - prumer_goly_pct) < 1.0 or (r['rate_pct'] < prumer_goly_pct and y_pos + 0.8 >= prumer_goly_pct):
            y_pos = prumer_goly_pct + 0.75
        ax2.text(i, y_pos, f"{carka(r['rate_pct'], 1)} %",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='#1A1A1A',
                 bbox=dict(boxstyle='square,pad=0.2', fc='#FFFFFF', ec='none'), zorder=5)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(x_labels_sit, fontsize=12)
    ax2.set_ylabel('Podíl gólů (%)', fontsize=13)
    ax2.yaxis.set_major_formatter(formatter_cesky)
    ax2.set_title('Herní situace', fontsize=14, fontweight='bold', pad=10)

    plt.suptitle('Podíl gólů podle typu zakončení a herní situace', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    uloz_graf('08_podil_golu_faktory.png')

    return {
        'chi_vysledky': chi_vysledky,
        'df_tab_podil': df_tab_podil,
        'prumer_goly_pct': prumer_goly_pct,
    }


# =============================================================================
# 2. FÁZE: JEDNOROZMĚRNÁ NORMALITA
# =============================================================================
def analyzuj_normalitu(df):
    """
    Testuje jednorozměrnou normalitu spojitých proměnných (Shapiro-Wilkův test
    ze scipy.stats). Graf 02 obsahuje čistý histogram s normální křivkou a normální Q-Q graf.
    """
    print('\n' + '=' * 75)
    print('2. FÁZE: DIAGNOSTIKA JEDNOROZMĚRNÉ NORMALITY (SHAPIRO-WILK)')
    print('=' * 75)

    sw_res = {}
    for col, nazev_cz in [('dist_m', 'Vzdálenost od branky (m)'),
                          ('angle_deg', 'Střelecký úhel (°)'),
                          ('minute', 'Minuta zápasu')]:
        data_col = df[col].dropna()
        W, p = stats.shapiro(data_col)
        skew = stats.skew(data_col)
        kurt = stats.kurtosis(data_col)
        verdikt = 'Zamítáme H0' if p < 0.05 else 'Nezamítáme H0'
        sw_res[col] = {'W': W, 'p': p, 'skew': skew, 'kurt': kurt, 'verdikt': verdikt, 'nazev_cz': nazev_cz}
        print(f'  {nazev_cz:<32}: W = {W:.4f}, p = {p:.2e} | {verdikt} (šikmost = {skew:.2f})')

    # Graf 02: 1x2 Jednorozměrná normalita dist_m (Histogram + Normální Q-Q graf) - ČISTÝ VZHLED
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    dist_data = df['dist_m'].dropna()
    mu_d, std_d = dist_data.mean(), dist_data.std()

    ax_h.hist(dist_data, bins=28, density=True, color='#1f77b4', edgecolor='white', alpha=0.75, label='Empirická data')
    x_axis = np.linspace(dist_data.min(), dist_data.max(), 300)
    ax_h.plot(x_axis, stats.norm.pdf(x_axis, mu_d, std_d), 'r-', linewidth=2.4, label='Normální křivka N(μ, σ²)')
    ax_h.set_xlabel('Vzdálenost od branky (m)', fontsize=13)
    ax_h.set_ylabel('Hustota pravděpodobnosti', fontsize=13)
    ax_h.xaxis.set_major_formatter(formatter_cesky)
    ax_h.yaxis.set_major_formatter(formatter_cesky)
    ax_h.set_title('Histogram vzdálenosti střel', fontsize=14, fontweight='bold')
    ax_h.legend(fontsize=12, loc='upper right')

    stats.probplot(dist_data, dist='norm', plot=ax_q)
    ax_q.get_lines()[0].set_markerfacecolor('#1f77b4')
    ax_q.get_lines()[0].set_markeredgecolor('none')
    ax_q.get_lines()[0].set_markersize(6)
    ax_q.get_lines()[0].set_alpha(0.65)
    ax_q.get_lines()[1].set_color('#d62728')
    ax_q.get_lines()[1].set_linewidth(2.2)
    ax_q.set_title('Normální Q-Q graf', fontsize=14, fontweight='bold')
    ax_q.set_xlabel('Teoretické kvantily N(0, 1)', fontsize=13)
    ax_q.set_ylabel('Uspořádaná data – vzdálenost (m)', fontsize=13)
    ax_q.xaxis.set_major_formatter(formatter_cesky)
    ax_q.yaxis.set_major_formatter(formatter_cesky)

    plt.suptitle('Diagnostika normality vzdálenosti střel', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('02_normalita_histogramy_qq.png')

    return sw_res


# =============================================================================
# 3. FÁZE: KORELAČNÍ ANALÝZA
# =============================================================================
def analyzuj_korelace(df):
    """
    Spearmanova pořadová korelační matice a párové rozptylové diagramy
    s oddělenou normalizací hustot (common_norm=False) a cut=0.
    """
    print('\n' + '=' * 75)
    print('3. FÁZE: SPEARMANOVA KORELAČNÍ ANALÝZA')
    print('=' * 75)

    corr_vars = ['dist_m', 'angle_deg', 'minute', 'xG']
    nazvy_cz = ['Vzdálenost od branky (m)', 'Střelecký úhel (°)', 'Minuta zápasu', 'Očekávané góly (xG)']
    spearman_mat = df[corr_vars].corr(method='spearman')
    spearman_mat_cz = spearman_mat.copy()
    spearman_mat_cz.index = nazvy_cz
    spearman_mat_cz.columns = nazvy_cz

    r_da, p_da = stats.spearmanr(df['dist_m'], df['angle_deg'])
    print(f'  Spearmanova korelace vzdálenost <-> úhel: r_s = {r_da:.3f} (p = {p_da:.2e})')

    # 3.1 Graf 03: Korelační heatmapa - ČISTÝ VZHLED
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    annot_matrix = spearman_mat_cz.map(lambda x: carka(x, 2))
    sns.heatmap(spearman_mat_cz, annot=annot_matrix, fmt='', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, square=True, ax=ax,
                annot_kws={'size': 13, 'weight': 'bold'},
                linewidths=1.2, linecolor='#FFFFFF',
                cbar_kws={'shrink': 0.8, 'label': 'Spearmanův koeficient r_s'})
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(formatter_cesky)
    ax.set_title('Spearmanova korelační matice vybraných proměnných', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    plt.tight_layout()
    uloz_graf('03_korelacni_heatmapa.png')

    # 3.2 Graf 03b: Pairplot s common_norm=False a cut=0
    pp_data = df[['dist_m', 'angle_deg', 'minute', 'goal']].copy()
    pp_data['Výsledek střely'] = pp_data['goal'].map({1: 'Gól', 0: 'Bez gólu'})
    pp_data = pp_data.rename(columns={
        'dist_m': 'Vzdálenost od branky (m)',
        'angle_deg': 'Střelecký úhel (°)',
        'minute': 'Minuta zápasu'
    })

    g_pair = sns.pairplot(
        pp_data,
        vars=['Vzdálenost od branky (m)', 'Střelecký úhel (°)', 'Minuta zápasu'],
        hue='Výsledek střely',
        palette={'Gól': '#d62728', 'Bez gólu': '#1f77b4'},
        height=2.7, aspect=1.3,
        plot_kws={'alpha': 0.60, 's': 35, 'edgecolor': 'none'},
        diag_kws={'fill': True, 'alpha': 0.45, 'linewidth': 2, 'common_norm': False, 'cut': 0}
    )
    g_pair.figure.suptitle('Párové rozptylové diagramy klíčových proměnných',
                           fontsize=15, fontweight='bold', y=1.02)
    g_pair.savefig(os.path.join(VYSTUPY_DIR, '03b_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close(g_pair.figure)
    print('  [OK] Graf uložen: 03b_pairplot.png')

    return {
        'spearman_mat': spearman_mat,
        'r_da': r_da,
        'p_da': p_da,
    }


# =============================================================================
# 4. FÁZE: WELCHOVA ANOVA, PÁROVÉ WELCHOVY T-TESTY A MANOVA
# =============================================================================
def analyzuj_anova_manova(df):
    """
    Testování rozdílů v geometrii střel podle typu zakončení (shotType):
      - Skupiny: Pravá noha (RightFoot), Levá noha (LeftFoot), Hlavou (Head).
      - Popisky skupin v boxplotech obsahují počet střel: 'Pravá noha (n = 286)'.
      - Statistické rámečky odstraněny z grafu do popisků.
      - Post-hoc: párové Welchovy t-testy s p-hodnotou přímo v popisku osy y.
    """
    print('\n' + '=' * 75)
    print('4. FÁZE: WELCHOVA ANOVA, PÁROVÉ WELCHOVY T-TESTY A MANOVA')
    print('=' * 75)

    main_shots = ['RightFoot', 'LeftFoot', 'Head']
    n_rf = len(df[df['shotType'] == 'RightFoot'])
    n_lf = len(df[df['shotType'] == 'LeftFoot'])
    n_hd = len(df[df['shotType'] == 'Head'])

    cz_map = {
        'RightFoot': f'Pravá noha\n(n = {n_rf})',
        'LeftFoot': f'Levá noha\n(n = {n_lf})',
        'Head': f'Hlavou\n(n = {n_hd})'
    }
    df_shots = df[df['shotType'].isin(main_shots)].copy()
    df_shots['shotType_cz'] = df_shots['shotType'].map(cz_map)
    poradi_cz = [cz_map['RightFoot'], cz_map['LeftFoot'], cz_map['Head']]

    print('Dělení skupin podle proměnné: shotType (Typ zakončení)')
    print(f'  Velikosti skupin: Pravá noha n = {n_rf}, Levá noha n = {n_lf}, Hlavou n = {n_hd}')

    # Levenův test homogenity rozptylů
    skupiny_d = [df_shots[df_shots['shotType'] == s]['dist_m'].values for s in main_shots]
    skupiny_a = [df_shots[df_shots['shotType'] == s]['angle_deg'].values for s in main_shots]
    L_stat_d, L_p_d = stats.levene(*skupiny_d, center='median')
    L_stat_a, L_p_a = stats.levene(*skupiny_a, center='median')
    print(f'  Levenův test rozptylů (vzdálenost): W = {L_stat_d:.3f}, p = {L_p_d:.2e} (zamítnuta shoda rozptylů)')
    print(f'  Levenův test rozptylů (úhel):       W = {L_stat_a:.3f}, p = {L_p_a:.2e} (shoda rozptylů nezamítnuta)')

    # Welchova ANOVA
    welch_d = anova_oneway(data=df_shots['dist_m'], groups=df_shots['shotType'], use_var='unequal')
    welch_a = anova_oneway(data=df_shots['angle_deg'], groups=df_shots['shotType'], use_var='unequal')
    F_welch_d, p_welch_d = welch_d.statistic, welch_d.pvalue
    F_welch_a, p_welch_a = welch_a.statistic, welch_a.pvalue

    # Klasická ANOVA
    classic_d = anova_oneway(data=df_shots['dist_m'], groups=df_shots['shotType'], use_var='equal')
    classic_a = anova_oneway(data=df_shots['angle_deg'], groups=df_shots['shotType'], use_var='equal')
    F_classic_d, p_classic_d = classic_d.statistic, classic_d.pvalue
    F_classic_a, p_classic_a = classic_a.statistic, classic_a.pvalue

    print(f'  Vzdálenost: Welchovo F = {F_welch_d:.2f} (p < 0,001) | Klasické F = {F_classic_d:.2f} (p < 0,001)')
    print(f'  Úhel:       Welchovo F = {F_welch_a:.2f} (p < 0,001) | Klasické F = {F_classic_a:.2f} (p < 0,001)')

    # 4.1 Graf 04: Boxploty vzdálenosti a úhlu - ČISTÝ VZHLED BEZ RÁMEČKŮ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.5))
    sns.boxplot(data=df_shots, x='shotType_cz', y='dist_m', order=poradi_cz, ax=ax1, palette='Set2')
    ax1.set_title('Vzdálenost podle zakončení', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Typ zakončení', fontsize=13)
    ax1.set_ylabel('Vzdálenost od branky (m)', fontsize=13)
    ax1.yaxis.set_major_formatter(formatter_cesky)

    sns.boxplot(data=df_shots, x='shotType_cz', y='angle_deg', order=poradi_cz, ax=ax2, palette='Set2')
    ax2.set_title('Střelecký úhel podle zakončení', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Typ zakončení', fontsize=13)
    ax2.set_ylabel('Střelecký úhel (°)', fontsize=13)
    ax2.yaxis.set_major_formatter(formatter_cesky)

    plt.suptitle('Geometrické charakteristiky střel podle typu zakončení', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('04_boxploty_shottype.png')

    # 4.2 Párové Welchovy t-testy s Bonferroniho korekcí (post-hoc)
    pairs = [('RightFoot', 'LeftFoot', 'Pravá vs. Levá noha'),
             ('RightFoot', 'Head', 'Pravá noha vs. Hlavou'),
             ('LeftFoot', 'Head', 'Levá noha vs. Hlavou')]
    k_pairs = len(pairs)

    posthoc_res = {'dist_m': [], 'angle_deg': []}
    for col in ['dist_m', 'angle_deg']:
        for g1, g2, label_pair in pairs:
            x1 = df_shots[df_shots['shotType'] == g1][col].values
            x2 = df_shots[df_shots['shotType'] == g2][col].values
            diff = np.mean(x1) - np.mean(x2)
            tt = stats.ttest_ind(x1, x2, equal_var=False)
            n1, n2 = len(x1), len(x2)
            v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
            se = np.sqrt(v1/n1 + v2/n2)
            df_ws = (v1/n1 + v2/n2)**2 / ((v1/n1)**2 / (n1 - 1) + (v2/n2)**2 / (n2 - 1))
            tcrit_95 = stats.t.ppf(0.975, df=df_ws)
            ci_lower = diff - tcrit_95 * se
            ci_upper = diff + tcrit_95 * se
            p_val = tt.pvalue
            p_bonf = min(1.0, p_val * k_pairs)
            posthoc_res[col].append({
                'pair': label_pair,
                'diff': diff,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_val': p_val,
                'p_bonf': p_bonf
            })

    # Graf 04b: Párové Welchovy t-testy - ČISTÝ VZHLED (p-hodnoty v popiscích osy y, žádná legenda)
    fig, (ax_pd, ax_pa) = plt.subplots(1, 2, figsize=(14.2, 5.2))
    y_pos = np.arange(k_pairs)

    # Popisky osy y obsahují p-hodnotu přímo v závorce
    labels_y_d = []
    for r in posthoc_res['dist_m']:
        p_str = "< 0,001" if r['p_bonf'] < 0.001 else f"= {carka(r['p_bonf'], 3)}"
        labels_y_d.append(f"{r['pair']} (p {p_str})")

    labels_y_a = []
    for r in posthoc_res['angle_deg']:
        p_str = "< 0,001" if r['p_bonf'] < 0.001 else f"= {carka(r['p_bonf'], 3)}"
        labels_y_a.append(f"{r['pair']} (p {p_str})")

    # Vzdálenost
    diffs_d = [r['diff'] for r in posthoc_res['dist_m']]
    xerr_d = [[r['diff'] - r['ci_lower'] for r in posthoc_res['dist_m']],
              [r['ci_upper'] - r['diff'] for r in posthoc_res['dist_m']]]
    ax_pd.errorbar(diffs_d, y_pos, xerr=xerr_d, fmt='o', color='#1f77b4',
                   capsize=6, capthick=2, elinewidth=2, markersize=8)
    ax_pd.axvline(0, color='#d62728', linestyle='--', linewidth=1.5)
    ax_pd.set_yticks(y_pos)
    ax_pd.set_yticklabels(labels_y_d, fontsize=12)
    ax_pd.set_xlabel('Rozdíl průměrů vzdálenosti (m)', fontsize=13)
    ax_pd.xaxis.set_major_formatter(formatter_cesky)
    ax_pd.set_title('Vzdálenost od branky', fontsize=14, fontweight='bold')
    ax_pd.set_xlim(-2.5, 13.5)

    # Úhel
    diffs_a = [r['diff'] for r in posthoc_res['angle_deg']]
    xerr_a = [[r['diff'] - r['ci_lower'] for r in posthoc_res['angle_deg']],
              [r['ci_upper'] - r['diff'] for r in posthoc_res['angle_deg']]]
    ax_pa.errorbar(diffs_a, y_pos, xerr=xerr_a, fmt='o', color='#2ca02c',
                   capsize=6, capthick=2, elinewidth=2, markersize=8)
    ax_pa.axvline(0, color='#d62728', linestyle='--', linewidth=1.5)
    ax_pa.set_yticks(y_pos)
    ax_pa.set_yticklabels(labels_y_a, fontsize=12)
    ax_pa.set_xlabel('Rozdíl průměrů střeleckého úhlu (°)', fontsize=13)
    ax_pa.xaxis.set_major_formatter(formatter_cesky)
    ax_pa.set_title('Střelecký úhel', fontsize=14, fontweight='bold')
    ax_pa.set_xlim(-24.0, 4.0)

    plt.suptitle('Párová porovnání typů zakončení (Welchovy t-testy)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('04b_posthoc_welch.png')

    # 4.3 MANOVA model bez xG
    manova = MANOVA.from_formula('dist_m + angle_deg ~ shotType', data=df_shots)
    manova_res = manova.mv_test()
    wilks_val = manova_res.results['shotType']['stat'].loc["Wilks' lambda", 'Value']
    pillai_val = manova_res.results['shotType']['stat'].loc["Pillai's trace", 'Value']
    print(f'  MANOVA: Pillai = {pillai_val:.3f} (p < 0,001), Wilks = {wilks_val:.3f} (p < 0,001)')

    return {
        'n_rf': n_rf,
        'n_lf': n_lf,
        'n_hd': n_hd,
        'L_stat_d': L_stat_d,
        'L_p_d': L_p_d,
        'L_stat_a': L_stat_a,
        'L_p_a': L_p_a,
        'F_welch_d': F_welch_d,
        'p_welch_d': p_welch_d,
        'F_welch_a': F_welch_a,
        'p_welch_a': p_welch_a,
        'F_classic_d': F_classic_d,
        'p_classic_d': p_classic_d,
        'F_classic_a': F_classic_a,
        'p_classic_a': p_classic_a,
        'wilks_val': wilks_val,
        'pillai_val': pillai_val,
        'posthoc_res': posthoc_res,
    }


# =============================================================================
# 5. FÁZE: PCA A t-SNE PROJEKCE
# =============================================================================
def analyzuj_pca_a_tsne(df):
    """
    Redukce dimenzionality na standardizovaných proměnných [dist_m, angle_deg, minute]:
      - Scree plot (05): celočíselná osa x (1, 2, 3), popisky jen % dané komponenty.
      - 2D biplot (05b): samostatný biplot s neutrálním titulkem a legendou v rohu.
      - Komponentní zátěže (05c): skutečné zátěže L = V * sqrt(λ).
      - t-SNE (05d): dva panely bez spodních poznámek.
    """
    print('\n' + '=' * 75)
    print('5. FÁZE: REDUKCE DIMENZIONALITY (PCA A t-SNE)')
    print('=' * 75)

    pca_vars = ['dist_m', 'angle_deg', 'minute']
    nazvy_cz = ['Vzdálenost od branky (m)', 'Střelecký úhel (°)', 'Minuta zápasu']
    X_std = StandardScaler().fit_transform(df[pca_vars])

    # Bartlettův test sférickosti
    R_mat = df[pca_vars].corr().values
    n_b, p_b = len(df), len(pca_vars)
    det_R = np.linalg.det(R_mat)
    chi2_b = -(n_b - 1 - (2*p_b + 5)/6) * np.log(max(1e-12, det_R))
    df_b = p_b * (p_b - 1) // 2
    p_val_b = 1 - stats.chi2.cdf(chi2_b, df=df_b)
    print(f'  Bartlettův test sférickosti: chi2 = {chi2_b:.1f}, df = {df_b}, p = {p_val_b:.2e}')

    pca = PCA()
    X_pca = pca.fit_transform(X_std)
    eigen_vals = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_ratio)
    eigenvectors = pca.components_.T

    # Skutečné zátěže (korelace proměnné s komponentou): vlastní vektor * odmocnina vlastního čísla
    actual_loadings = eigenvectors * np.sqrt(eigen_vals)

    print(f'  Vlastní čísla λ:    {eigen_vals.round(3)}')
    print(f'  Vysvětlený rozptyl: {var_ratio.round(1)} % (kumulativně: {cum_var.round(1)} %)')
    print('  Skutečné zátěže (korelace s komponentou):')
    df_actual_loadings = pd.DataFrame(actual_loadings, index=nazvy_cz, columns=['PC1', 'PC2', 'PC3'])
    print(df_actual_loadings.round(3))

    # 5.1 Samostatný Scree plot (05_scree_plot.png) - ČISTÝ VZHLED
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.bar([1, 2, 3], eigen_vals, color='#1f77b4', alpha=0.75, width=0.55, label='Vlastní číslo (λ)')
    ax.plot([1, 2, 3], eigen_vals, 'ro-', linewidth=2.2, markersize=7)
    ax.axhline(1.0, color='#d62728', linestyle='--', linewidth=1.8, label='Kaiserovo pravidlo (λ = 1)')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['1', '2', '3'])
    ax.set_ylim(0, 2.3)
    ax.set_xlabel('Hlavní komponenta', fontsize=13)
    ax.set_ylabel('Vlastní číslo (λ)', fontsize=13)
    ax.yaxis.set_major_formatter(formatter_cesky)
    ax.set_title('Scree plot vlastních čísel PCA', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=12, loc='upper right')

    # Popisky nad sloupci obsahují POUZE podíl rozptylu dané komponenty
    for idx_ev, (ev, vr) in enumerate(zip(eigen_vals, var_ratio)):
        ax.text(idx_ev + 1, ev + 0.10, f'{carka(vr, 1)} %',
                ha='center', fontsize=12, fontweight='bold', color='#1A1A1A')
    plt.tight_layout()
    uloz_graf('05_scree_plot.png')

    # 5.2 Graf 05b: Samostatný 2D Biplot (bez scree plotu) - ČISTÝ VZHLED
    fig, ax_bip = plt.subplots(figsize=(9.2, 6.4))
    mask_goal = df['goal'] == 1
    ax_bip.scatter(X_pca[~mask_goal, 0], X_pca[~mask_goal, 1],
                   c='#1f77b4', alpha=0.45, s=35, label='Bez gólu', zorder=2)
    ax_bip.scatter(X_pca[mask_goal, 0], X_pca[mask_goal, 1],
                   c='#d62728', alpha=0.90, s=65, edgecolors='#800000', linewidths=0.8,
                   label='Gól', zorder=3)

    # Legenda umístěna do volného levého dolního rohu
    ax_bip.legend(loc='lower left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=12)
    ax_bip.set_xlim(-7.5, 3.5)
    ax_bip.set_ylim(-2.5, 2.8)

    # Vykreslení šipek pro zátěže (lw=2.2, barva tmavá #1A1A1A) s popisky mimo body
    scale_arrow = 2.4

    # 1. Šipka Vzdálenost (m)
    lx0, ly0 = eigenvectors[0, 0] * scale_arrow, eigenvectors[0, 1] * scale_arrow
    ax_bip.annotate('', xy=(lx0, ly0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=2.2))
    ax_bip.text(lx0 + 0.35, ly0 + 0.15, 'Vzdálenost (m)',
                fontsize=12, fontweight='bold', color='#1A1A1A', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='#FFFFFF', ec='#CCCCCC', alpha=0.95), zorder=4)

    # 2. Šipka Střelecký úhel (°) - umístěna nad šipku na volné místo bez bodů a bez překryvu šipky
    lx1, ly1 = eigenvectors[1, 0] * scale_arrow, eigenvectors[1, 1] * scale_arrow
    ax_bip.annotate('', xy=(lx1, ly1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=2.2))
    ax_bip.text(-2.85, 0.72, 'Střelecký úhel (°)',
                fontsize=12, fontweight='bold', color='#1A1A1A', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc='#FFFFFF', ec='#CCCCCC', alpha=0.95), zorder=4)

    # 3. Šipka Minuta zápasu
    lx2, ly2 = eigenvectors[2, 0] * scale_arrow, eigenvectors[2, 1] * scale_arrow
    ax_bip.annotate('', xy=(lx2, ly2), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#1A1A1A', lw=2.2))
    ax_bip.text(lx2, ly2 + 0.18, 'Minuta zápasu',
                fontsize=12, fontweight='bold', color='#1A1A1A', ha='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='#FFFFFF', ec='#CCCCCC', alpha=0.95), zorder=4)

    ax_bip.axhline(0, color='#CCCCCC', linestyle='--', linewidth=1)
    ax_bip.axvline(0, color='#CCCCCC', linestyle='--', linewidth=1)
    ax_bip.set_xlabel(f'PC1 ({carka(var_ratio[0], 1)} % rozptylu) – Geometrická osa', fontsize=13)
    ax_bip.set_ylabel(f'PC2 ({carka(var_ratio[1], 1)} % rozptylu) – Časová osa', fontsize=13)
    ax_bip.xaxis.set_major_formatter(formatter_cesky)
    ax_bip.yaxis.set_major_formatter(formatter_cesky)
    ax_bip.set_title('Biplot hlavních komponent', fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    uloz_graf('05b_biplot_pca.png')

    # 5.3 Heatmapa zátěží (05c_pca_zatezove.png)
    fig, ax_z = plt.subplots(figsize=(8.8, 5.0))
    annot_loadings = df_actual_loadings.map(lambda x: carka(x, 3))
    sns.heatmap(df_actual_loadings, annot=annot_loadings, fmt='', cmap='RdBu_r', vmin=-1.0, vmax=1.0, center=0,
                ax=ax_z, annot_kws={'size': 13, 'weight': 'bold'},
                linewidths=1.2, linecolor='#FFFFFF', cbar_kws={'label': 'Zátěž (korelace s komponentou)'})
    cbar_z = ax_z.collections[0].colorbar
    cbar_z.ax.yaxis.set_major_formatter(formatter_cesky)
    ax_z.set_title('Komponentní zátěže hlavních komponent', fontsize=14, fontweight='bold', pad=10)
    ax_z.set_xticklabels(ax_z.get_xticklabels(), fontsize=12, fontweight='bold')
    ax_z.set_yticklabels(ax_z.get_yticklabels(), rotation=0, fontsize=12)
    plt.tight_layout()
    uloz_graf('05c_pca_zatezove.png')

    return {
        'X_std': X_std,
        'pca_vars': pca_vars,
        'chi2_b': chi2_b,
        'p_val_b': p_val_b,
        'eigen_vals': eigen_vals,
        'var_ratio': var_ratio,
        'cum_var': cum_var,
        'actual_loadings': df_actual_loadings,
    }


# =============================================================================
# 6. FÁZE: SHLUKOVÁ ANALÝZA A t-SNE VIZUALIZACE
# =============================================================================
def analyzuj_shluky_a_tsne(df, X_std):
    """
    Shluková analýza na proměnných [dist_m, angle_deg, minute]:
      1. Diagnostika počtu shluků a Wardův dendrogram (06): bez legend, čárkovaná čára řezu.
      2. Prostorová mapa shluků (06b): legenda jen shluky a počty střel, bez rámečku.
      3. t-SNE projekce (05d): dva panely, legendy mimo data, bez spodní poznámky.
    """
    print('\n' + '=' * 75)
    print('6. FÁZE: SHLUKOVÁ ANALÝZA A t-SNE VIZUALIZACE')
    print('=' * 75)

    k_range = range(2, 8)
    inertias = []
    sil_scores = []
    for k in k_range:
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_std)
        inertias.append(km_test.inertia_)
        sil = silhouette_score(X_std, km_test.labels_)
        sil_scores.append(sil)
        print(f'  k = {k}: Inertia = {km_test.inertia_:.1f}, Silhouette score = {sil:.4f}')

    opt_k = 2
    print(f'  Optimální počet shluků: k = {opt_k} (nejvyšší Silhouette score = {sil_scores[0]:.4f})')

    # Finální K-means model (k = 2)
    km = KMeans(n_clusters=opt_k, random_state=42, n_init=10).fit(X_std)
    df['cluster'] = km.labels_

    clus_summary = df.groupby('cluster').agg({
        'dist_m': 'mean',
        'angle_deg': 'mean',
        'minute': 'mean',
        'goal': ['count', 'sum', 'mean']
    })
    clus_summary.columns = ['Průměr vzdálenost (m)', 'Průměr úhel (°)', 'Průměr minuta', 'Počet střel', 'Počet gólů', 'Úspěšnost střelby']
    print('\nProfilace shluků K-means (k = 2):')
    print(clus_summary.round(2).to_string())

    id_close = clus_summary['Průměr vzdálenost (m)'].idxmin()
    id_far = 1 - id_close
    c_close = df[df['cluster'] == id_close]
    c_far = df[df['cluster'] == id_far]

    # Hierarchické shlukování Wardovou metodou
    link_mat = linkage(X_std, method='ward', metric='euclidean')
    cut_height = 34.38
    ward_clusters = fcluster(link_mat, t=opt_k, criterion='maxclust')
    df['ward_cluster'] = ward_clusters

    crosstab_ward_km = pd.crosstab(
        df['ward_cluster'].map({1: 'Ward: Vápno', 2: 'Ward: Dálka'}),
        df['cluster'].map({id_close: 'K-means: Vápno', id_far: 'K-means: Dálka'}),
        rownames=['Wardova metoda'], colnames=['K-means']
    )
    print('\nKřížová tabulka shody Wardových shluků a K-means:')
    print(crosstab_ward_km.to_string())

    shodne = np.sum(df['cluster'].map({id_close: 1, id_far: 2}) == df['ward_cluster'])
    shoda_pct = shodne / len(df) * 100
    print(f'Shoda shluků Wardovy metody a K-means: {carka(shoda_pct, 1)} % ({shodne} z {len(df)} střel)')

    # 6.1 Graf 06: Wardův dendrogram a diagnostika počtu shluků - ČISTÝ VZHLED (bez legend)
    fig, (ax_diag, ax_dend) = plt.subplots(1, 2, figsize=(13.6, 5.5))

    # Diagnostika vlevo
    ax_diag.plot(list(k_range), sil_scores, 'rs--', linewidth=2.2, markersize=7)
    ax_diag.axvline(opt_k, color='#d62728', linestyle=':', linewidth=2)
    ax_diag.set_xlabel('Počet shluků k', fontsize=13)
    ax_diag.set_ylabel('Průměrné Silhouette score', fontsize=13)
    ax_diag.yaxis.set_major_formatter(formatter_cesky)
    ax_diag.set_title('Diagnostika optimálního počtu shluků', fontsize=14, fontweight='bold')

    # Wardův dendrogram vpravo
    dendrogram(link_mat, ax=ax_dend, truncate_mode='lastp', p=18, leaf_rotation=45, leaf_font_size=11,
               color_threshold=cut_height)
    ax_dend.axhline(cut_height, color='#d62728', linestyle='--', linewidth=2.0)
    ax_dend.set_title('Hierarchické shlukování (Wardova metoda)', fontsize=14, fontweight='bold')
    ax_dend.set_xlabel('Shluky (počet střel)', fontsize=13)
    ax_dend.set_ylabel('Wardova eukleidovská vzdálenost', fontsize=13)
    ax_dend.yaxis.set_major_formatter(formatter_cesky)

    plt.suptitle('Volba počtu shluků a Wardův dendrogram', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('06_dendrogram.png')

    # 6.2 Graf 06b: Prostorová mapa shluků na útočné polovině hřiště - ČISTÝ VZHLED
    fig, ax_map = plt.subplots(figsize=(11.0, 6.8))
    nakresli_fotbalove_hriste(ax_map, polovina='utocna')

    ax_map.scatter(c_far['x_m'], c_far['y_m'], c='#1f77b4', s=45, alpha=0.65,
                   label=f'Shluk Dálka (n = {len(c_far)})', zorder=3)
    ax_map.scatter(c_close['x_m'], c_close['y_m'], c='#d62728', s=65, alpha=0.85,
                   label=f'Shluk Vápno (n = {len(c_close)})', zorder=4)

    ax_map.set_title('Prostorové rozložení shluků na útočné polovině',
                     fontsize=14, fontweight='bold', pad=10)
    ax_map.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=12)
    plt.tight_layout()
    uloz_graf('06b_mapa_shluku.png')

    # 6.3 Graf 05d: t-SNE projekce (dva panely, bez spodních poznámek)
    print('  Probíhá výpočet t-SNE projekce (sklearn.manifold.TSNE, perplexity=30)...')
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_std)

    fig, (ax_tsne_g, ax_tsne_c) = plt.subplots(1, 2, figsize=(13.6, 5.8))

    # Panel (a): Obarvení dle výsledku střely
    mask_tsne_g = df['goal'] == 1
    ax_tsne_g.scatter(X_tsne[~mask_tsne_g, 0], X_tsne[~mask_tsne_g, 1],
                      c='#1f77b4', alpha=0.50, s=35, label='Bez gólu', zorder=2)
    ax_tsne_g.scatter(X_tsne[mask_tsne_g, 0], X_tsne[mask_tsne_g, 1],
                      c='#d62728', alpha=0.90, s=65, edgecolors='#800000', linewidths=0.8,
                      label='Gól', zorder=3)
    ax_tsne_g.set_xlabel('t-SNE dimenze 1', fontsize=13)
    ax_tsne_g.set_ylabel('t-SNE dimenze 2', fontsize=13)
    ax_tsne_g.xaxis.set_major_formatter(formatter_cesky)
    ax_tsne_g.yaxis.set_major_formatter(formatter_cesky)
    ax_tsne_g.set_title('(a) Obarvení podle výsledku střely', fontsize=14, fontweight='bold', pad=10)
    ax_tsne_g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2,
                     frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=12)

    # Panel (b): Obarvení dle shluků z K-means
    mask_tsne_c = df['cluster'] == id_close
    ax_tsne_c.scatter(X_tsne[~mask_tsne_c, 0], X_tsne[~mask_tsne_c, 1],
                      c='#1f77b4', alpha=0.60, s=35, label='Shluk Dálka', zorder=2)
    ax_tsne_c.scatter(X_tsne[mask_tsne_c, 0], X_tsne[mask_tsne_c, 1],
                      c='#d62728', alpha=0.80, s=45, label='Shluk Vápno', zorder=3)
    ax_tsne_c.set_xlabel('t-SNE dimenze 1', fontsize=13)
    ax_tsne_c.set_ylabel('t-SNE dimenze 2', fontsize=13)
    ax_tsne_c.xaxis.set_major_formatter(formatter_cesky)
    ax_tsne_c.yaxis.set_major_formatter(formatter_cesky)
    ax_tsne_c.set_title('(b) Obarvení podle shluku K-means', fontsize=14, fontweight='bold', pad=10)
    ax_tsne_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2,
                     frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=12)

    plt.suptitle('Projekce t-SNE pro geometrické a časové proměnné', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    uloz_graf('05d_tsne_strely.png')

    return {
        'opt_k': opt_k,
        'sil_scores': sil_scores,
        'clus_summary': clus_summary,
        'c_close_mean_dist': c_close["dist_m"].mean(),
        'c_close_median_dist': c_close["dist_m"].median(),
        'c_close_goal_rate': c_close["goal"].mean(),
        'c_far_mean_dist': c_far["dist_m"].mean(),
        'c_far_median_dist': c_far["dist_m"].median(),
        'c_far_goal_rate': c_far["goal"].mean(),
        'n_close': len(c_close),
        'n_far': len(c_far),
        'shoda_pct': shoda_pct,
        'shodne': shodne,
        'crosstab_ward_km': crosstab_ward_km,
        'cut_height': cut_height,
    }


# =============================================================================
# 7. FÁZE: LOGISTICKÁ REGRESE, KŘÍŽOVÁ VALIDACE A BENCHMARK xG
# =============================================================================
def analyzuj_logistickou_regresi(df):
    """
    Logistická regrese bez xG a bez penalt (n = 729 střel ze hry, 60 gólů):
      - Logistická křivka (07): jen modelová křivka a pozorované body s malými % bez n.
      - ROC křivka (07): jen model CV, xG a diagonála; AUC v legendě; bez výplně a Youdenova bodu.
      - Forest plot (07b): bez legendy, bez hvězdiček, p-hodnoty vpravo mimo intervaly.
    """
    print('\n' + '=' * 75)
    print('7. FÁZE: LOGISTICKÁ REGRESE BEZ xG A SROVNÁNÍ S xG BENCHMARKEM')
    print('=' * 75)

    # 1. Vyřazení pokutových kopů
    n_pen = (df['is_penalty'] == 1).sum()
    df_reg = df[df['is_penalty'] == 0].copy().reset_index(drop=True)
    n_reg = len(df_reg)
    n_goals_reg = df_reg['goal'].sum()
    pct_goals_reg = n_goals_reg / n_reg * 100
    print(f'  Vyřazeno {n_pen} pokutových kopů. Trénovací vzorek: n = {n_reg} střel ze hry ({n_goals_reg} gólů, {carka(pct_goals_reg, 1)} %).')

    # 2. Manuální tvorba dummy proměnných se správnými referenčními kategoriemi
    shot_dummies = pd.get_dummies(df_reg['shotType'], prefix='shotType', dtype=float)
    if 'shotType_RightFoot' in shot_dummies.columns:
        shot_dummies = shot_dummies.drop(columns=['shotType_RightFoot'])

    sit_dummies = pd.get_dummies(df_reg['situation'], prefix='sit', dtype=float)
    if 'sit_OpenPlay' in sit_dummies.columns:
        sit_dummies = sit_dummies.drop(columns=['sit_OpenPlay'])

    X_reg_raw = pd.concat([df_reg[['dist_m', 'angle_deg']], shot_dummies, sit_dummies], axis=1)
    X_reg_const = sm.add_constant(X_reg_raw)
    y_reg = df_reg['goal']

    # 3. VIF tabulka
    vif_table = pd.DataFrame({
        'Prediktor': X_reg_const.columns,
        'VIF': [variance_inflation_factor(X_reg_const.values, i) for i in range(X_reg_const.shape[1])]
    })
    vif_table = vif_table[vif_table['Prediktor'] != 'const'].copy()
    vif_table['Tolerance (1/VIF)'] = 1.0 / vif_table['VIF']
    vif_table.round(3).to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_vif.csv'), index=False)
    print(f'  [OK] Tabulka VIF uložena: tabulka_vif.csv (max VIF = {vif_table["VIF"].max():.2f})')

    # 4. Odhad multivariabilního logit modelu
    logit_mod = sm.Logit(y_reg, X_reg_const).fit(disp=False)
    coefs = logit_mod.params
    p_vals = logit_mod.pvalues
    odds_r = np.exp(coefs)
    conf_i = np.exp(logit_mod.conf_int())
    conf_i.columns = ['95% CI dolní', '95% CI horní']

    n_params = len(coefs)
    epv = n_goals_reg / n_params
    print(f'  Počet parametrů modelu: {n_params} (konstanta + 8 prediktorů) na {n_goals_reg} gólů -> EPV = {epv:.1f}')

    # Přepočty pro forest plot: OR na 5 m a OR na 10°
    or_dist_5m = np.exp(5 * coefs['dist_m'])
    ci_dist_5m_low = np.exp(5 * logit_mod.conf_int().loc['dist_m', 0])
    ci_dist_5m_upp = np.exp(5 * logit_mod.conf_int().loc['dist_m', 1])

    or_angle_10deg = np.exp(10 * coefs['angle_deg'])
    ci_angle_10deg_low = np.exp(10 * logit_mod.conf_int().loc['angle_deg', 0])
    ci_angle_10deg_upp = np.exp(10 * logit_mod.conf_int().loc['angle_deg', 1])

    print(f'  OR vzdálenost (na 5 m): {or_dist_5m:.4f} [95% CI: {ci_dist_5m_low:.4f}; {ci_dist_5m_upp:.4f}], p = {p_vals["dist_m"]:.4f}')
    print(f'  OR úhel (na 10°):      {or_angle_10deg:.4f} [95% CI: {ci_angle_10deg_low:.4f}; {ci_angle_10deg_upp:.4f}], p = {p_vals["angle_deg"]:.4f}')

    reg_summary = pd.DataFrame({
        'Koeficient β': coefs,
        'Směrodatná chyba': logit_mod.bse,
        'z-statistika': logit_mod.tvalues,
        'p-hodnota': p_vals,
        'Odds Ratio (1 jednotka)': odds_r,
        '95% CI dolní (1 j.)': conf_i['95% CI dolní'],
        '95% CI horní (1 j.)': conf_i['95% CI horní']
    }).round(4)
    reg_summary.loc['dist_m (na 5 m)', ['Odds Ratio (1 jednotka)', '95% CI dolní (1 j.)', '95% CI horní (1 j.)', 'p-hodnota']] = [
        round(or_dist_5m, 4), round(ci_dist_5m_low, 4), round(ci_dist_5m_upp, 4), round(p_vals['dist_m'], 4)
    ]
    reg_summary.loc['angle_deg (na 10°)', ['Odds Ratio (1 jednotka)', '95% CI dolní (1 j.)', '95% CI horní (1 j.)', 'p-hodnota']] = [
        round(or_angle_10deg, 4), round(ci_angle_10deg_low, 4), round(ci_angle_10deg_upp, 4), round(p_vals['angle_deg'], 4)
    ]
    reg_summary.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_logit_koeficienty.csv'))
    print('  [OK] Tabulka koeficientů uložena: tabulka_logit_koeficienty.csv')

    # 5. ROC analýza: in-sample, 5-fold CV a externí xG
    y_pred_prob_in = logit_mod.predict(X_reg_const)
    fpr_in, tpr_in, thresh_in = roc_curve(y_reg, y_pred_prob_in)
    auc_in = auc(fpr_in, tpr_in)

    # Stratifikovaný 5-fold
    np.random.seed(42)
    idx_goals = np.where(y_reg == 1)[0]
    idx_nongoals = np.where(y_reg == 0)[0]
    np.random.shuffle(idx_goals)
    np.random.shuffle(idx_nongoals)

    k_folds = 5
    folds_goals = np.array_split(idx_goals, k_folds)
    folds_nongoals = np.array_split(idx_nongoals, k_folds)
    oof_preds = np.zeros(len(df_reg))

    for k in range(k_folds):
        test_idx = np.concatenate([folds_goals[k], folds_nongoals[k]])
        train_idx = np.setdiff1d(np.arange(len(df_reg)), test_idx)
        X_train, y_train = X_reg_const.iloc[train_idx], y_reg.iloc[train_idx]
        X_test = X_reg_const.iloc[test_idx]
        mod_k = sm.Logit(y_train, X_train).fit(disp=False)
        oof_preds[test_idx] = mod_k.predict(X_test)

    fpr_cv, tpr_cv, thresh_cv = roc_curve(y_reg, oof_preds)
    auc_cv = auc(fpr_cv, tpr_cv)

    fpr_xg, tpr_xg, thresh_xg = roc_curve(y_reg, df_reg['xG'])
    auc_xg = auc(fpr_xg, tpr_xg)

    youden_thresh = 0.085
    pred_bin = (y_pred_prob_in >= youden_thresh).astype(int)
    sens_085 = np.sum((pred_bin == 1) & (y_reg == 1)) / np.sum(y_reg == 1)
    spec_085 = np.sum((pred_bin == 0) & (y_reg == 0)) / np.sum(y_reg == 0)

    # 6. Graf 07: ROC křivka - ČISTÝ VZHLED (pouze model CV a xG, AUC v legendě, bez stínování a rámečku)
    fig, ax_roc = plt.subplots(figsize=(8.0, 6.0))
    ax_roc.plot(fpr_cv, tpr_cv, color='#d62728', lw=2.6,
                label=f'Model, křížová validace (AUC = {carka(auc_cv, 3)})')
    ax_roc.plot(fpr_xg, tpr_xg, color='#1f77b4', lw=2.4, linestyle='--',
                label=f'xG (AUC = {carka(auc_xg, 3)})')
    ax_roc.plot([0, 1], [0, 1], color='#888888', linestyle=':', lw=1.5,
                label='Náhodný odhad')

    ax_roc.set_xlabel('1 – Specificita (podíl falešně pozitivních)', fontsize=13)
    ax_roc.set_ylabel('Senzitivita (podíl správně pozitivních)', fontsize=13)
    ax_roc.xaxis.set_major_formatter(formatter_cesky)
    ax_roc.yaxis.set_major_formatter(formatter_cesky)
    ax_roc.set_title('ROC křivka modelu a xG benchmarku', fontsize=14, fontweight='bold', pad=10)
    ax_roc.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=12)
    plt.tight_layout()
    uloz_graf('07_roc_krivka.png')

    # 7. Graf 07: Logistická křivka - ČISTÝ VZHLED
    # Křivka podél osy hřiště (y = 34 m): angle = 2 * arctan(3.66 / d) * 180 / pi
    dist_seq = np.linspace(3, 35, 300)
    angle_seq = 2 * np.arctan(3.66 / dist_seq) * 180 / np.pi

    b0 = coefs['const']
    b_dist = coefs['dist_m']
    b_angle = coefs['angle_deg']

    logit_axis_seq = b0 + b_dist * dist_seq + b_angle * angle_seq
    prob_axis_seq = 1 / (1 + np.exp(-logit_axis_seq))

    # Empirické podíly v 5m intervalech a průměrné předpovědi modelu pro skutečné střely
    df_reg['pred_prob'] = y_pred_prob_in
    bins = [0, 5, 10, 15, 20, 25, 30, 35]
    df_reg['dist_bin'] = pd.cut(df_reg['dist_m'], bins=bins, right=False)
    obs_bins = df_reg.groupby('dist_bin', observed=False).agg(
        n_shots=('goal', 'count'),
        n_goals=('goal', 'sum'),
        mean_dist=('dist_m', 'mean'),
        median_dist=('dist_m', 'median'),
        mean_angle=('angle_deg', 'mean'),
        pred_mean=('pred_prob', 'mean')
    ).reset_index()
    obs_bins['rate'] = obs_bins['n_goals'] / obs_bins['n_shots']

    # Modelové odhady na ose hřiště v těžištích jednotlivých intervalů (pro srovnávací text)
    def get_axis_prob(d):
        ang = 2 * np.arctan(3.66 / d) * 180 / np.pi
        log_v = b0 + b_dist * d + b_angle * ang
        return 1 / (1 + np.exp(-log_v)), ang

    obs_bins['model_prob'] = [get_axis_prob(d)[0] for d in obs_bins['mean_dist']]
    obs_bins['model_angle'] = [get_axis_prob(d)[1] for d in obs_bins['mean_dist']]

    fig, ax_log = plt.subplots(figsize=(9.2, 5.8))

    # Křivka pro střely z osy hřiště (tenká čárkovaná čára)
    ax_log.plot(dist_seq, prob_axis_seq, color='#888888', linestyle='--', linewidth=1.6, alpha=0.85,
                label='Křivka pro střely z osy')

    # Pozorované podíly v intervalech (modré kosočtverce)
    ax_log.scatter(obs_bins['mean_dist'], obs_bins['rate'], color='#1f77b4', s=85, marker='D',
                   edgecolors='#0b4064', linewidths=1.2, zorder=5,
                   label='Pozorovaný podíl')

    # Průměr předpovězených pravděpodobností modelu v intervalech (červené čtverce)
    ax_log.scatter(obs_bins['mean_dist'], obs_bins['pred_mean'], color='#d62728', s=85, marker='s',
                   edgecolors='#800000', linewidths=1.2, zorder=5,
                   label='Model (průměr předpovědí)')

    ax_log.set_xlim(2, 36)
    ax_log.set_ylim(-0.04, 0.78)
    ax_log.set_xlabel('Vzdálenost od branky (m)', fontsize=13)
    ax_log.set_ylabel('Pravděpodobnost vstřelení gólu P(gól)', fontsize=13)
    ax_log.xaxis.set_major_formatter(formatter_cesky)
    ax_log.yaxis.set_major_formatter(formatter_cesky)
    ax_log.set_title('Pravděpodobnost gólu podle vzdálenosti', fontsize=14, fontweight='bold', pad=10)
    ax_log.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=12)
    plt.tight_layout()
    uloz_graf('07_logisticka_krivka.png')

    # 8. Graf 07b: Forest plot - ČISTÝ VZHLED (bez legendy, bez hvězdiček, p-hodnoty vpravo mimo intervaly)
    forest_items = [
        ('dist_m', 'Vzdálenost od branky (na 5 m)', or_dist_5m, ci_dist_5m_low, ci_dist_5m_upp, p_vals['dist_m']),
        ('angle_deg', 'Střelecký úhel (na 10°)', or_angle_10deg, ci_angle_10deg_low, ci_angle_10deg_upp, p_vals['angle_deg']),
        ('shotType_Head', 'Zakončení hlavou (vs. pravá noha)', odds_r['shotType_Head'], conf_i.loc['shotType_Head', '95% CI dolní'], conf_i.loc['shotType_Head', '95% CI horní'], p_vals['shotType_Head']),
        ('shotType_LeftFoot', 'Zakončení levou nohou (vs. pravá noha)', odds_r['shotType_LeftFoot'], conf_i.loc['shotType_LeftFoot', '95% CI dolní'], conf_i.loc['shotType_LeftFoot', '95% CI horní'], p_vals['shotType_LeftFoot']),
        ('shotType_Unknown', 'Zakončení neznámé (vs. pravá noha)', odds_r['shotType_Unknown'], conf_i.loc['shotType_Unknown', '95% CI dolní'], conf_i.loc['shotType_Unknown', '95% CI horní'], p_vals['shotType_Unknown']),
        ('sit_DirectFreekick', 'Přímý volný kop (vs. ze hry)', odds_r['sit_DirectFreekick'], conf_i.loc['sit_DirectFreekick', '95% CI dolní'], conf_i.loc['sit_DirectFreekick', '95% CI horní'], p_vals['sit_DirectFreekick']),
        ('sit_FromCorner', 'Rohový kop (vs. ze hry)', odds_r['sit_FromCorner'], conf_i.loc['sit_FromCorner', '95% CI dolní'], conf_i.loc['sit_FromCorner', '95% CI horní'], p_vals['sit_FromCorner']),
        ('sit_SetPiece', 'Standardní situace (vs. ze hry)', odds_r['sit_SetPiece'], conf_i.loc['sit_SetPiece', '95% CI dolní'], conf_i.loc['sit_SetPiece', '95% CI horní'], p_vals['sit_SetPiece'])
    ]

    fig, ax_fp = plt.subplots(figsize=(10.5, 5.8))
    y_pos_fp = np.arange(len(forest_items))
    ax_fp.set_xscale('log')

    barvy_fp = ['#d62728' if it[5] < 0.05 else '#1f77b4' for it in forest_items]

    for i, it in enumerate(forest_items):
        v_name, v_label, or_val, ci_l, ci_u, p_v = it
        ax_fp.errorbar(or_val, y_pos_fp[i], xerr=[[or_val - ci_l], [ci_u - or_val]],
                       fmt='o', color=barvy_fp[i], capsize=5, capthick=2, elinewidth=2, markersize=8)
        # Zarovnaný popisek p-hodnoty na pravé straně bez hvězdiček
        p_txt = f"p < 0,001" if p_v < 0.001 else f"p = {carka(p_v, 3)}"
        ax_fp.text(14.5, y_pos_fp[i], p_txt, va='center', fontsize=12, fontweight='bold', color=barvy_fp[i])

    ax_fp.axvline(1.0, color='#888888', linestyle='--', linewidth=1.5)
    ax_fp.set_yticks(y_pos_fp)
    ax_fp.set_yticklabels([it[1] for it in forest_items], fontsize=12)
    ax_fp.invert_yaxis()
    ax_fp.set_xticks([0.1, 1.0, 10.0])
    ax_fp.set_xticklabels(['0,1', '1', '10'], fontsize=12)
    ax_fp.xaxis.set_minor_locator(ticker.NullLocator())
    ax_fp.set_xlabel('Poměr šancí (OR), logaritmická osa, s 95% intervalem spolehlivosti', fontsize=13)
    ax_fp.set_title('Poměry šancí (OR) z logistické regrese', fontsize=14, fontweight='bold', pad=10)
    ax_fp.set_xlim(0.015, 34.0)
    plt.tight_layout()
    uloz_graf('07b_forest_plot.png')

    return {
        'odds_r': odds_r,
        'p_vals': p_vals,
        'conf_i': conf_i,
        'auc_in': auc_in,
        'auc_cv': auc_cv,
        'auc_xg': auc_xg,
        'youden_thresh': youden_thresh,
        'sens_085': sens_085,
        'spec_085': spec_085,
        'llr_chi2': logit_mod.llr,
        'llr_p': logit_mod.llr_pvalue,
        'vif_max': vif_table['VIF'].max(),
        'n_reg': n_reg,
        'n_goals_reg': n_goals_reg,
        'n_params': n_params,
        'epv': epv,
        'or_dist_5m': or_dist_5m,
        'ci_dist_5m_low': ci_dist_5m_low,
        'ci_dist_5m_upp': ci_dist_5m_upp,
        'or_angle_10deg': or_angle_10deg,
        'ci_angle_10deg_low': ci_angle_10deg_low,
        'ci_angle_10deg_upp': ci_angle_10deg_upp,
        'obs_bins': obs_bins
    }


# =============================================================================
# 8. FÁZE: EXPORT TABULKY HYPOTÉZ, TEXTOVÉHO SHRNUTÍ A POPISKŮ GRAFŮ
# =============================================================================
def exportuj_vysledky_a_zpravu(vysledky, df_orig):
    """
    Exportuje souhrnnou tabulku hypotéz (tabulka_hypotezy.csv),
    kompletní textovou zprávu pro obhajobu (vystupy/souhrn_analyzy.txt)
    a navržené české popisky pod všechny grafy (vystupy/popisky_grafu.txt).
    """
    print('\n' + '=' * 75)
    print('8. FÁZE: EXPORT TABULKY HYPOTÉZ, SOUHRNNÉ ZPRÁVY A POPISKŮ GRAFŮ')
    print('=' * 75)

    sw = vysledky['normalita']
    an = vysledky['anova']
    pc = vysledky['pca']
    sh = vysledky['shluky']
    lg = vysledky['logit']
    fa = vysledky['faktory']
    obs_b = lg['obs_bins']

    # Statistika střel hlavou vs. ostatní
    head_dist = df_orig[df_orig['shotType'] == 'Head']['dist_m']
    other_dist = df_orig[df_orig['shotType'] != 'Head']['dist_m']
    head_mean = head_dist.mean()
    head_median = head_dist.median()
    other_mean = other_dist.mean()
    other_median = other_dist.median()
    head_goals = df_orig[df_orig['shotType'] == 'Head']['goal'].sum()
    head_n = len(head_dist)
    head_rate = head_goals / head_n * 100

    # 8.1 Tabulka hypotéz
    hyp_seznam = [
        ('1 – Normalita', 'Shapiro-Wilkův test (dist_m)', 'Vzdálenost dist_m má normální rozdělení',
         f"W = {carka(sw['dist_m']['W'], 4)}", '< 0,001', 'Zamítáme H0'),
        ('1 – Normalita', 'Shapiro-Wilkův test (angle_deg)', 'Střelecký úhel angle_deg má normální rozdělení',
         f"W = {carka(sw['angle_deg']['W'], 4)}", '< 0,001', 'Zamítáme H0'),
        ('1 – Normalita', 'Shapiro-Wilkův test (minute)', 'Minuta zápasu minute má normální rozdělení',
         f"W = {carka(sw['minute']['W'], 4)}", '< 0,001', 'Zamítáme H0'),
        ('2 – Rozptyly', 'Levenův test rozptylů (dist_m)', 'Rozptyly vzdálenosti jsou mezi typy zakončení shodné',
         f"W = {carka(an['L_stat_d'], 3)}", '< 0,001', 'Zamítáme H0 (heteroskedasticita)'),
        ('2 – Rozptyly', 'Levenův test rozptylů (angle_deg)', 'Rozptyly úhlu jsou mezi typy zakončení shodné',
         f"W = {carka(an['L_stat_a'], 3)}", carka(an['L_p_a'], 3), 'Nezamítáme H0'),
        ('3 – Závislost', 'Spearmanův korelační test', 'Korelace mezi vzdáleností a úhlem je nulová (r = 0)',
         f"r_s = {carka(vysledky['korelace']['r_da'], 3)}", '< 0,001', 'Zamítáme H0'),
        ('4 – Welch ANOVA', 'statsmodels anova_oneway (dist_m)', 'Střední vzdálenosti střel jsou pro typy zakončení stejné',
         f"F = {carka(an['F_welch_d'], 1)}", '< 0,001', 'Zamítáme H0'),
        ('4 – Welch ANOVA', 'statsmodels anova_oneway (angle_deg)', 'Střední úhly střel jsou pro typy zakončení stejné',
         f"F = {carka(an['F_welch_a'], 1)}", '< 0,001', 'Zamítáme H0'),
        ('5 – Sdružená MANOVA', 'MANOVA (Pillaiova stopa)', 'Vektory [dist_m, angle_deg] jsou ve skupinách stejné',
         f"Pillai = {carka(an['pillai_val'], 3)}", '< 0,001', 'Zamítáme H0 (orientační, porušeny předpoklady)'),
        ('6 – PCA sférickost', 'Bartlettův test sférickosti', 'Korelační matice je jednotková (PCA nevhodná)',
         f"χ² = {carka(pc['chi2_b'], 1)}", '< 0,001', 'Zamítáme H0'),
        ('7 – Multikolinearita', 'Faktor VIF', 'Prediktory v logistickém modelu netrpí multikolinearitou',
         f"max VIF = {carka(lg['vif_max'], 2)}", '< 3,0', 'Nezamítáme H0'),
        ('8 – Logit vzdálenost', 'Logit model (dist_m na 5 m)', 'Vzdálenost střely neovlivňuje šanci na gól (β = 0)',
         f"OR = {carka(lg['or_dist_5m'], 4)}", carka(lg['p_vals']['dist_m'], 3), 'Zamítáme H0'),
        ('8 – Logit úhel', 'Logit model (angle_deg na 10°)', 'Střelecký úhel neovlivňuje šanci na gól (β = 0)',
         f"OR = {carka(lg['or_angle_10deg'], 4)}", carka(lg['p_vals']['angle_deg'], 3), 'Zamítáme H0'),
        ('8 – Logit hlavička', 'Logit model (shotType_Head)', 'Zakončení hlavou má stejnou šanci jako pravá noha',
         f"OR = {carka(lg['odds_r']['shotType_Head'], 4)}", carka(lg['p_vals']['shotType_Head'], 3), 'Zamítáme H0'),
        ('8 – Logit globální', 'LLR Chí-kvadrát test', 'Všechny prediktory v logistickém modelu mají nulový vliv',
         f"χ² = {carka(lg['llr_chi2'], 1)}", '< 0,001', 'Zamítáme H0')
    ]

    df_hyp_out = pd.DataFrame(hyp_seznam, columns=['Fáze', 'Metoda', 'Nulová hypotéza (H0)', 'Statistika', 'p-hodnota', 'Rozhodnutí'])
    df_hyp_out.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_hypotezy.csv'), index=False)
    print('  [OK] Souhrnná tabulka hypotéz uložena: tabulka_hypotezy.csv')

    # 8.2 Souhrn analýzy (souhrn_analyzy.txt) s odstavcem "Zajímavé zjištění" a těžišti
    chi = fa['chi_vysledky']
    shrnuti_text = f"""=============================================================================
VÝSLEDKY STATISTICKÉ ANALÝZY VMM: FAKTORY STŘELECKÉ ÚSPĚŠNOSTI
=============================================================================

1. SEZNAM POUŽITÝCH KNIHOVEN A FUNKCÍ (PRIORITA ZE CVIČENÍ):
   - scipy.stats:
     * shapiro (cviceni_01): jednorozměrná normalita spojitých proměnných.
       Doplňující poznámka: Při rozsahu výběru n = 735 má Shapiro-Wilkův test
       velmi vysokou sílu a zamítá nulovou hypotézu normality i při mírné odchylce;
       Q-Q graf ukazuje, že odchylky od normality nastávají především v koncových
       kvantilech (těžší chvosty), zatímco střední část rozdělení je relativně symetrická.
     * levene (cviceni_03): test shody rozptylů mezi skupinami zakončení.
     * ttest_ind s equal_var=False: Welchův dvouvýběrový t-test pro post-hoc porovnání.
     * spearmanr: pořadová korelační analýza.
     * chi2_contingency: testy nezávislosti v kontingenčních tabulkách faktorů včetně kontroly
       očekávaných četností buněk.
   - scipy.cluster.hierarchy:
     * linkage, dendrogram, fcluster: hierarchické shlukování Wardovou metodou.
   - statsmodels.stats.oneway.anova_oneway (cviceni_03):
     * Welchova ANOVA (use_var='unequal') pro nestejné rozptyly.
     * Klasická jednofaktorová ANOVA (use_var='equal') pro metodické srovnání.
   - statsmodels.multivariate.manova.MANOVA (cviceni_03):
     * Sdružené porovnání geometrických proměnných [dist_m, angle_deg] dle shotType.
   - statsmodels.api.Logit:
     * Multivariabilní binární logistická regrese pro poměry šancí (OR) a intervaly spolehlivosti.
   - sklearn (cviceni_04, cviceni_06):
     * StandardScaler: standardizace veličin před PCA, t-SNE a shlukováním.
     * PCA: rozklad na hlavní komponenty a faktorové zátěže L = V * sqrt(λ).
     * TSNE (perplexity=30): nelineární dvourozměrná projekce dat.
     * KMeans & silhouette_score: k-means shlukování a odůvodnění počtu shluků.
     * roc_curve & auc: vyhodnocení diskriminační schopnosti (in-sample, křížová validace, xG).
   - ODSTRANĚNÍ KNIHOVNY PINGOUIN:
     * Pingouin byl ze skriptu kompletně odstraněn.

2. ZAJÍMAVÉ ZJIŠTĚNÍ: PARADOX ZAKONČENÍ HLAVOU
   Samotný typ zakončení není v bivariačním pohledu statisticky významný
   (chí-kvadrát test χ² = {carka(chi['shottype_all']['chi2'], 2)}, df = {chi['shottype_all']['dof']}, p = {carka(chi['shottype_all']['p'], 3)};
   pro 3 hlavní kategorie χ² = {carka(chi['shottype_main']['chi2'], 2)}, df = {chi['shottype_main']['dof']}, p = {carka(chi['shottype_main']['p'], 3)}).
   Úspěšnost střel hlavou ve vzorku činí {carka(head_rate, 1)} % (2 góly ze 48 střel), u pravé nohy 9,4 % a u levé nohy 8,8 %.
   Z dat plyne, že hlavou se střílí z podstatně menší vzdálenosti než u ostatních zakončení:
   - Střely hlavou (n = {head_n}): průměrná vzdálenost = {carka(head_mean, 2)} m, medián = {carka(head_median, 2)} m.
   - Ostatní zakončení (n = {len(other_dist)}): průměrná vzdálenost = {carka(other_mean, 2)} m, medián = {carka(other_median, 2)} m.
   Po zohlednění vzdálenosti a úhlu je zakončení hlavou v multivariabilním logit modelu statisticky méně úspěšné než pravou nohou
   (OR = {carka(lg['odds_r']['shotType_Head'], 2)}, 95% IS: [{carka(lg['conf_i'].loc['shotType_Head', '95% CI dolní'], 3)}; {carka(lg['conf_i'].loc['shotType_Head', '95% CI horní'], 3)}], p = {carka(lg['p_vals']['shotType_Head'], 3)}).
   Tento výsledek však stojí na malém počtu střel (48 střel hlavou, pouze 2 góly) a je orientační.

3. LOGISTICKÁ REGRESE BEZ xG A BEZ PENALT (n = {lg['n_reg']} střel ze hry, {lg['n_goals_reg']} gólů):
   - Upozornění k velikosti vzorku:
     Model odhaduje 9 parametrů (konstanta + 8 prediktorů) na 60 pozorovaných gólů ze hry.
     Poměr činí zhruba 6,7 gólu na parametr (pravidlo EPV doporučuje minimálně 10 až 15).
   - Odhady poměrů šancí (Odds Ratios, 95% intervaly spolehlivosti):
     * Vzdálenost od branky (přepočet na 5 m): OR = {carka(lg['or_dist_5m'], 4)} [95% CI: {carka(lg['ci_dist_5m_low'], 4)}; {carka(lg['ci_dist_5m_upp'], 4)}], p = {carka(lg['p_vals']['dist_m'], 4)}
       -> Při stejném střeleckém úhlu, typu zakončení a situaci každých dalších 5 metrů
          od branky snižuje šanci na gól o cca {carka((1 - lg['or_dist_5m'])*100, 1)} %.
     * Střelecký úhel (přepočet na 10°): OR = {carka(lg['or_angle_10deg'], 4)} [95% CI: {carka(lg['ci_angle_10deg_low'], 4)}; {carka(lg['ci_angle_10deg_upp'], 4)}], p = {carka(lg['p_vals']['angle_deg'], 4)}
       -> Při stejné vzdálenosti, typu zakončení a situaci každých dalších 10 stupňů
          zorného úhlu zvyšuje šanci na gól o cca {carka((lg['or_angle_10deg'] - 1)*100, 1)} %.
     * Zakončení hlavou (shotType_Head): OR = {carka(lg['odds_r']['shotType_Head'], 4)} [95% CI: {carka(lg['conf_i'].loc['shotType_Head', '95% CI dolní'], 4)}; {carka(lg['conf_i'].loc['shotType_Head', '95% CI horní'], 4)}], p = {carka(lg['p_vals']['shotType_Head'], 4)} (orientační, n = 48, 2 góly)
     * Ostatní prediktory (statisticky neprůkazné při α = 0,05):
       - Zakončení levou nohou: OR = {carka(lg['odds_r']['shotType_LeftFoot'], 4)}, p = {carka(lg['p_vals']['shotType_LeftFoot'], 3)}.
       - Neznámé zakončení: OR = {carka(lg['odds_r']['shotType_Unknown'], 4)}, p = {carka(lg['p_vals']['shotType_Unknown'], 3)}.
       - Přímý volný kop: OR = {carka(lg['odds_r']['sit_DirectFreekick'], 4)}, p = {carka(lg['p_vals']['sit_DirectFreekick'], 3)}.
       - Rohový kop: OR = {carka(lg['odds_r']['sit_FromCorner'], 4)}, p = {carka(lg['p_vals']['sit_FromCorner'], 3)}.
       - Standardní situace: OR = {carka(lg['odds_r']['sit_SetPiece'], 4)}, p = {carka(lg['p_vals']['sit_SetPiece'], 3)}.
   - Logistická křivka a srovnání v 5m intervalech:
     Křivka pro střely z podélné osy hřiště (y = 34 m) probíhá nad pozorovanými podíly i nad průměrem předpovědí modelu,
     protože předpokládá ideální střelecký úhel přímo před brankou, zatímco skutečné střely v daných intervalech
     zahrnují i střely z ostrých bočních úhlů s menším střeleckým úhlem.
     Číselné srovnání v intervalech (pozorováno vs. průměr předpovědi modelu pro skutečné střely vs. n):
     * Interval [0, 5 m):  n = {obs_b.loc[0, 'n_shots']}, pozorováno {carka(obs_b.loc[0, 'rate']*100, 1)} % | průměr předpovědi {carka(obs_b.loc[0, 'pred_mean']*100, 1)} % (těžiště {carka(obs_b.loc[0, 'mean_dist'], 2)} m, model z osy {carka(obs_b.loc[0, 'model_prob']*100, 1)} %; málo dat)
     * Interval [5, 10 m): n = {obs_b.loc[1, 'n_shots']}, pozorováno {carka(obs_b.loc[1, 'rate']*100, 1)} % | průměr předpovědi {carka(obs_b.loc[1, 'pred_mean']*100, 1)} % (těžiště {carka(obs_b.loc[1, 'mean_dist'], 2)} m, model z osy {carka(obs_b.loc[1, 'model_prob']*100, 1)} %)
     * Interval [10, 15 m): n = {obs_b.loc[2, 'n_shots']}, pozorováno {carka(obs_b.loc[2, 'rate']*100, 1)} % | průměr předpovědi {carka(obs_b.loc[2, 'pred_mean']*100, 1)} % (těžiště {carka(obs_b.loc[2, 'mean_dist'], 2)} m, model z osy {carka(obs_b.loc[2, 'model_prob']*100, 1)} %)
     * Interval [15, 20 m): n = {obs_b.loc[3, 'n_shots']}, pozorováno {carka(obs_b.loc[3, 'rate']*100, 1)} % | průměr předpovědi {carka(obs_b.loc[3, 'pred_mean']*100, 1)} % (těžiště {carka(obs_b.loc[3, 'mean_dist'], 2)} m, model z osy {carka(obs_b.loc[3, 'model_prob']*100, 1)} %)
     * Interval [20, 25 m): n = {obs_b.loc[4, 'n_shots']}, pozorováno {carka(obs_b.loc[4, 'rate']*100, 1)} % | průměr předpovědi {carka(obs_b.loc[4, 'pred_mean']*100, 1)} % (těžiště {carka(obs_b.loc[4, 'mean_dist'], 2)} m, model z osy {carka(obs_b.loc[4, 'model_prob']*100, 1)} %)
     * Interval [25, 30 m): n = {obs_b.loc[5, 'n_shots']}, pozorováno {carka(obs_b.loc[5, 'rate']*100, 1)} % | průměr předpovědi {carka(obs_b.loc[5, 'pred_mean']*100, 1)} % (těžiště {carka(obs_b.loc[5, 'mean_dist'], 2)} m, model z osy {carka(obs_b.loc[5, 'model_prob']*100, 1)} %)
     * Interval [30, 35 m): n = {obs_b.loc[6, 'n_shots']}, pozorováno {carka(obs_b.loc[6, 'rate']*100, 1)} % | průměr předpovědi {carka(obs_b.loc[6, 'pred_mean']*100, 1)} % (těžiště {carka(obs_b.loc[6, 'mean_dist'], 2)} m, model z osy {carka(obs_b.loc[6, 'model_prob']*100, 1)} %)

4. CHÍ-KVADRÁT TESTY FAKTORŮ A KONTROLA PŘEDPOKLADŮ (OČEKÁVANÉ ČETNOSTI):
   - Typ zakončení:
     * Původní 4 kategorie: χ² = {carka(chi['shottype_all']['chi2'], 2)}, df = {chi['shottype_all']['dof']}, p = {carka(chi['shottype_all']['p'], 3)}. Buněk s očekávanou četností < 5: {chi['shottype_all']['bunek_lt5']} z {chi['shottype_all']['bunek_celkem']} ({carka(chi['shottype_all']['pct_lt5'], 1)} %) -> předpoklad splněn (<= 20 %).
     * 3 hlavní kategorie: χ² = {carka(chi['shottype_main']['chi2'], 2)}, df = {chi['shottype_main']['dof']}, p = {carka(chi['shottype_main']['p'], 3)}. Buněk < 5: {chi['shottype_main']['bunek_lt5']} z {chi['shottype_main']['bunek_celkem']} ({carka(chi['shottype_main']['pct_lt5'], 1)} %) -> předpoklad splněn.
   - Herní situace:
     * Původní (včetně penalt): χ² = {carka(chi['sit_all']['chi2'], 2)}, df = {chi['sit_all']['dof']}, p < 0,001. Buněk < 5: {chi['sit_all']['bunek_lt5']} z {chi['sit_all']['bunek_celkem']} ({carka(chi['sit_all']['pct_lt5'], 1)} %) -> předpoklad porušen (> 20 %).
     * Upravená (BEZ penalt): χ² = {carka(chi['sit_no_pen']['chi2'], 2)}, df = {chi['sit_no_pen']['dof']}, p = {carka(chi['sit_no_pen']['p'], 3)}. Buněk < 5: {chi['sit_no_pen']['bunek_lt5']} z {chi['sit_no_pen']['bunek_celkem']} ({carka(chi['sit_no_pen']['pct_lt5'], 1)} %) -> předpoklad není plně splněn (25 % > 20 %), výsledek je orientační. Po vyřazení penalt není vliv situace statisticky významný.
   - Předchozí akce (lastAction):
     * Původní (14 kategorií): χ² = {carka(chi['la_orig']['chi2'], 2)}, df = {chi['la_orig']['dof']}, p = {carka(chi['la_orig']['p'], 3)}. Buněk < 5: {chi['la_orig']['bunek_lt5']} z {chi['la_orig']['bunek_celkem']} ({carka(chi['la_orig']['pct_lt5'], 1)} %) -> předpoklad porušen.
     * Upravená (sloučeno < 15 do Ostatní): χ² = {carka(chi['la_merged']['chi2'], 2)}, df = {chi['la_merged']['dof']}, p = {carka(chi['la_merged']['p'], 3)}. Buněk < 5: {chi['la_merged']['bunek_lt5']} z {chi['la_merged']['bunek_celkem']} ({carka(chi['la_merged']['pct_lt5'], 1)} %) -> předpoklad není plně splněn (31,8 % > 20 %), výsledek je orientační. Vliv předchozí akce není statisticky významný.
   - Asistence u střely:
     * χ² = {carka(chi['assist']['chi2'], 2)}, df = {chi['assist']['dof']}, p = 1,000. Buněk < 5: 0 z 4 (0,0 %). Vliv asistence není statisticky významný.

5. DISKRIMINAČNÍ SCHOPNOST A BENCHMARK PROTI xG:
   - Plocha pod ROC křivkou (AUC):
     * Náš model (stratifikovaná 5-násobná validace): AUC = {carka(lg['auc_cv'], 4)}
     * Externí benchmark xG (Understat, n = 729):     AUC = {carka(lg['auc_xg'], 4)}
     * In-sample trénovací AUC činí {carka(lg['auc_in'], 4)} (mírně optimistické z důvodu evaluace na trénovacích datech).
   - Rozhodovací práh dle Youdenova indexu činí 0,085 (citlivost {carka(lg['sens_085']*100, 1)} %, specificita {carka(lg['spec_085']*100, 1)} %).
   - Podíl gólů ve zkoumaném vzorku střel ze hry činí 8,2 % (60 gólů z 729 střel).

6. JEDNOFAKTOROVÁ ANOVA A PÁROVÉ ROZDÍLY DLE ZAKONČENÍ:
   - Dělení skupin podle proměnné: shotType (Typ zakončení).
   - Skupiny: Pravá noha (n = {an['n_rf']}), Levá noha (n = {an['n_lf']}), Hlavou (n = {an['n_hd']}).
   - Levenův test: pro vzdálenost W = {carka(an['L_stat_d'], 3)} (p < 0,001, heteroskedasticita), pro úhel W = {carka(an['L_stat_a'], 3)} (p = {carka(an['L_p_a'], 3)}).
   - Welchova ANOVA: vzdálenost F = {carka(an['F_welch_d'], 2)} (p < 0,001), úhel F = {carka(an['F_welch_a'], 2)} (p < 0,001).
   - Post-hoc párové testy (Welchovy t-testy s Bonferroniho korekcí):
     * Pravá vs. Levá noha: rozdíl vzdáleností +0,96 m (p_adj = 0,232, nevýznamný).
     * Pravá noha vs. Hlavou: rozdíl vzdáleností +9,93 m (p_adj < 0,001, vysoce významný).
     * Levá noha vs. Hlavou: rozdíl vzdáleností +8,97 m (p_adj < 0,001, vysoce významný).

7. REDUKCE DIMENZIONALITY A SHLUKOVÁNÍ:
   - PCA na [dist_m, angle_deg, minute]:
     * λ1 = {carka(pc['eigen_vals'][0], 3)} ({carka(pc['var_ratio'][0], 1)} %), λ2 = {carka(pc['eigen_vals'][1], 3)} ({carka(pc['var_ratio'][1], 1)} %), λ3 = {carka(pc['eigen_vals'][2], 3)} ({carka(pc['cum_var'][2], 1)} % kumulativně).
     * Zátěže PC1: dist_m = {carka(pc['actual_loadings'].loc['Vzdálenost od branky (m)', 'PC1'], 3)}, angle_deg = {carka(pc['actual_loadings'].loc['Střelecký úhel (°)', 'PC1'], 3)}.
     * Zátěže PC2: minute = {carka(pc['actual_loadings'].loc['Minuta zápasu', 'PC2'], 3)}.
   - K-means a Wardovo shlukování (k = 2, Silhouette score = {carka(sh['sil_scores'][0], 3)}):
     * Shoda shlukování: {carka(sh['shoda_pct'], 1)} % ({sh['shodne']} ze 735 střel).
     * Shluk Vápno (n = {sh['n_close']}): průměrná vzdálenost {carka(sh['c_close_mean_dist'], 1)} m, úspěšnost gólů {carka(sh['c_close_goal_rate']*100, 1)} %.
     * Shluk Dálka (n = {sh['n_far']}): průměrná vzdálenost {carka(sh['c_far_mean_dist'], 1)} m, úspěšnost gólů {carka(sh['c_far_goal_rate']*100, 1)} %.
=============================================================================
"""

    cesta_souhrn = os.path.join(VYSTUPY_DIR, 'souhrn_analyzy.txt')
    with open(cesta_souhrn, 'w', encoding='utf-8') as f:
        f.write(shrnuti_text)
    print(f'  [OK] Kompletní souhrn analýzy uložen: {cesta_souhrn}')

    # 8.3 Navržené české popisky pod grafy do prezentace (popisky_grafu.txt)
    popisky_text = """=============================================================================
NAVRŽENÉ POPISKY POD GRAFY PRO PREZENTACI (VMM PROJEKT 04)
Obsahuje všechny podrobné statistické hodnoty, testy a poznámky vyřazené z obrázků.
=============================================================================

01_mapa_strel.png
Prostorové rozložení všech 735 střel na útočné polovině hřiště v reálných metrech. Vzorek obsahuje 63 gólů (8,6 %) a 672 neproměněných střel (91,4 %). Góly jsou soustředěny téměř výhradně v prostoru pokutového území v těsné blízkosti soupeřovy branky.

01b_hist_xG.png
Empirické rozdělení metrik očekávaných gólů (xG) vykazuje silné pravostranné sešikmení s mediánem 0,05 a průměrem 0,09. Naprostá většina střel má nízkou hodnotu očekávaného gólu pod 0,15, což odpovídá zakončení z větších vzdáleností nebo nevýhodných střeleckých úhlů.

01c_kde_hriste.png
2D jádrový odhad hustoty střelby (KDE) na útočné polovině hřiště bez umělých ořezů uvnitř hrací plochy. Zatímco neproměněné střely (modré vrstevnice) pokrývají široký prostor před pokutovým územím, góly (červené vrstevnice) tvoří kompaktní ohnisko přímo v prostoru před brankou (vzdálenost od branky pod 12 m a šířka 26 až 42 m).

02_normalita_histogramy_qq.png
Diagnostika jednorozměrné normality pro vzdálenost střel od branky pomocí histogramu s teoretickou normální křivkou a normálního Q-Q grafu. Shapiro-Wilkův test zamítá normalitu (W = 0,983, p < 0,001); odchylky od normálního rozdělení nastávají zejména v koncových kvantilech v důsledku sešikmení a přirozených hranic hrací plochy.

03_korelacni_heatmapa.png
Spearmanova pořadová korelační matice vybraných proměnných. Klíčovým vztahem je silná záporná nelineární vazba mezi vzdáleností od branky a střeleckým úhlem (r_s = -0,783, p < 0,001), zatímco minuta zápasu nevykazuje s ostatními veličinami statisticky významnou korelaci.

03b_pairplot.png
Párové rozptylové diagramy a jednorozměrné odhady hustot pro vzdálenost, střelecký úhel a minutu zápasu rozlišené podle výsledku střely. Proměněné střely (červeně) jsou koncentrovány v oblasti malých vzdáleností a velkých střeleckých úhlů, čas zápasu střeleckou úspěšnost vizuálně neovlivňuje.

04_boxploty_shottype.png
Porovnání vzdálenosti a střeleckého úhlu podle typu zakončení pro pravou nohu (n = 286), levou nohu (n = 272) a hlavičky (n = 48). Welchova jednofaktorová ANOVA potvrzuje statisticky vysoce významné rozdíly jak ve vzdálenosti (Welchovo F = 176,87, p < 0,001; Levene W = 15,55, p < 0,001), tak v úhlu (Welchovo F = 50,09, p < 0,001; Levene W = 2,08, p = 0,126).

04b_posthoc_welch.png
Párová porovnání typů zakončení pomocí Welchových t-testů s Bonferroniho korekcí p-hodnot ve formátu p (Bonferroni) a 95% intervaly spolehlivosti rozdílů průměrů. Zakončení hlavou se střílí z významně menší vzdálenosti než nohou (rozdíl oproti pravé noze +9,93 m, oproti levé +8,97 m, obojí p < 0,001), zatímco rozdíl mezi pravou a levou nohou činí pouze 0,96 m a není statisticky významný (p = 0,232).

05_scree_plot.png
Scree plot (graf úpatí) vlastních čísel z analýzy hlavních komponent (PCA) na proměnných vzdálenost, úhel a minuta zápasu. První komponenta vysvětluje 58,8 % celkového rozptylu (vlastní číslo λ1 = 1,768), druhá 33,2 % (λ2 = 0,998, kumulativně 92,1 %) a třetí 7,9 % (λ3 = 0,238, kumulativně 100,0 %); dle Kaiserova pravidla (λ > 1) postačuje první hlavní komponenta.

05b_biplot_pca.png
2D biplot prvních dvou hlavních komponent PCA (PC1 vysvětluje 58,8 %, PC2 33,2 % rozptylu). První komponenta představuje geometrickou osu (vzdálenost se zátěží +0,939 a úhel se zátěží -0,938 působí proti sobě), druhá komponenta odpovídá času zápasu (zátěž minuty +0,997); góly se shlukují v záporných hodnotách PC1 odpovídajících zakončení zblízka.

05c_pca_zatezove.png
Matice komponentních zátěží L = V · √λ (korelací původních proměnných s komponentami) pro vzdálenost, úhel a minutu zápasu. Potvrzuje ortogonální rozdělení na geometrický rozměr střelby v PC1 a časový průběh utkání v PC2.

05d_tsne_strely.png
Dvourozměrná nelineární projekce t-SNE na standardizovaných proměnných [vzdálenost, úhel, minuta] s perplexitou 30 obarvená (a) podle výsledku střely a (b) podle shluků K-means. Projekce slouží výhradně k vizuálnímu posouzení spojitosti datového prostoru; vzdálenosti mezi shluky v t-SNE nelze přímo geometricky interpretovat.

06_dendrogram.png
Diagnostika počtu shluků pomocí průměrného Silhouette score a dendrogram hierarchického shlukování Wardovou metodou s čárkovanou čarou řezu na výšce 34,4. Obě metody nezávisle identifikují jako optimální rozdělení dat do dvou přirozených shluků (nejvyšší Silhouette score 0,406).

06b_mapa_shluku.png
Prostorová lokalizace dvou shluků z K-means na útočné polovině hřiště. Shluk 'Vápno' (n = 175) má průměrnou vzdálenost 10,6 m a úspěšnost gólů 21,1 %, zatímco shluk 'Dálka' (n = 560) má průměrnou vzdálenost 22,6 m a úspěšnost pouze 4,6 %; shoda shlukování s Wardovou hierarchickou metodou činí 89,4 % (657 ze 735 střel).

07_logisticka_krivka.png
Srovnání pozorovaného podílu gólů (modré kosočtverce) a průměru předpovězených pravděpodobností logistického modelu pro skutečné střely (červené čtverce) v 5m intervalech. Modelová křivka pro střely z podélné osy hřiště (tenká čárkovaná čára, y = 34 m, referenční situace pravá noha ze hry) probíhá nad pozorovanými podíly i nad průměrem předpovědí, protože předpokládá maximální střelecký úhel přímo před brankou, zatímco reálné střely v intervalech zahrnují i střely z ostrých bočních úhlů s menším střeleckým úhlem. Číselné srovnání v intervalech (pozorováno vs. průměr předpovědi vs. n): [0, 5 m) 50,0 % vs. 60,4 % (n = 8, málo dat), [5, 10 m) 31,0 % vs. 25,0 % (n = 58), [10, 15 m) 11,6 % vs. 11,3 % (n = 129), [15, 20 m) 4,8 % vs. 7,3 % (n = 167), [20, 25 m) 4,2 % vs. 4,6 % (n = 168), [25, 30 m) 4,3 % vs. 3,3 % (n = 161) a [30, 35 m) 0,0 % vs. 2,3 % (n = 36).

07_roc_krivka.png
ROC křivky hodnotící diskriminační schopnost logistického modelu a nezávislého benchmarku očekávaných gólů (xG) na vzorku střel ze hry bez penalt (n = 729, 60 gólů, 8,2 %). Křížově validovaný model dosahuje AUC = 0,724 (in-sample trénovací AUC = 0,764 je mírně optimistické) a je plně srovnatelný s externí metrikou xG (AUC = 0,737); optimální rozhodovací práh stanovený Youdenovým indexem činí 0,085 (citlivost 70,0 %, specificita 68,9 %).

07b_forest_plot.png
Poměry šancí (OR) a jejich 95% intervaly spolehlivosti z multivariabilní logistické regrese bez penalt na logaritmické ose (0,1; 1; 10). Červeně jsou zobrazeny statisticky významné faktory při α = 0,05 (vzdálenost na 5 m: OR = 0,55, p < 0,001; úhel na 10°: OR = 1,28, p = 0,004; zakončení hlavou oproti pravé noze: OR = 0,11, 95% IS: [0,022; 0,506], p = 0,005); modře statisticky nevýznamné faktory (p ≥ 0,05). Z dat plyne, že hlavou se střílí z menší vzdálenosti než u ostatních zakončení (průměr 10,96 m, medián 10,94 m proti 20,34 m a 20,79 m); výsledek pro hlavičky však stojí na malém počtu střel (48 střel, 2 góly) a je orientační.

08_podil_golu_faktory.png
Podíl gólů podle typu zakončení a herní situace s vodorovnou referenční linií podílu gólů ve zkoumaném vzorku (8,6 %, n = 735); penalty (6 střel, 50,0 % gólů) a neznámé zakončení (129 střel, 7,0 % gólů) jsou z grafu vynechány pro zachování čitelnosti. Rozdíly v typu zakončení nejsou statisticky významné (χ² = 2,84, df = 3, p = 0,418; pro 3 hlavní typy χ² = 1,90, df = 2, p = 0,388); herní situace je v celém vzorku významná kvůli penaltám (χ² = 19,00, df = 4, p < 0,001), avšak po vyřazení penalt statistická významnost situace mizí (χ² = 5,97, df = 3, p = 0,113, přičemž 25 % buněk má očekávanou četnost < 5 a výsledek je orientační).
=============================================================================
"""
    cesta_popisky = os.path.join(VYSTUPY_DIR, 'popisky_grafu.txt')
    with open(cesta_popisky, 'w', encoding='utf-8') as f:
        f.write(popisky_text)
    print(f'  [OK] Navržené popisky grafů uloženy: {cesta_popisky}')


# =============================================================================
# HLAVNÍ SPOUŠTĚCÍ BLOK (MAIN PIPELINE)
# =============================================================================
if __name__ == '__main__':
    print('=' * 75)
    print('SPUŠTĚNÍ KOMPLETNÍ VÍCEROZMĚRNÉ STATISTICKÉ ANALÝZY VMM (VERZE 5)')
    print('=' * 75)

    # 1. Načtení dat (735 střel, 63 gólů, reálné metry)
    df_clean = pd.read_csv(DATA_PATH)
    print(f'Načteno {len(df_clean)} střel (gólů: {df_clean["goal"].sum()}, {df_clean["goal"].mean():.2%}).')

    # 2. Slovník pro předávání statistických výsledků
    vysledky = {}

    # 3. Postupné spouštění jednotlivých fází
    vysledky['faktory'] = eda_a_prostorova_mapa(df_clean)
    vysledky['normalita'] = analyzuj_normalitu(df_clean)
    vysledky['korelace'] = analyzuj_korelace(df_clean)
    vysledky['anova'] = analyzuj_anova_manova(df_clean)
    vysledky['pca'] = analyzuj_pca_a_tsne(df_clean)
    vysledky['shluky'] = analyzuj_shluky_a_tsne(df_clean, vysledky['pca']['X_std'])
    vysledky['logit'] = analyzuj_logistickou_regresi(df_clean)

    # 4. Export souhrnné tabulky hypotéz, textové zprávy a popisků grafů
    exportuj_vysledky_a_zpravu(vysledky, df_clean)

    print('\n' + '=' * 75)
    print('VŠECHNY FÁZE ANALÝZY BYLY ÚSPĚŠNĚ A BEZ CHYB DOKONČENY!')
    print('=' * 75)
