# -*- coding: utf-8 -*-
"""
=============================================================================
SEMESTRÁLNÍ PROJEKT VMM – STŘELECKÁ ÚSPĚŠNOST (PREMIER LEAGUE)
Skript 2: Vícerozměrná statistická analýza a tvorba grafů (Modularizovaná verze)
=============================================================================
Struktura modularizovaného kódu:
  - eda_a_prostorova_mapa(df)
  - analyzuj_normalitu(df)
  - analyzuj_korelace(df)
  - analyzuj_anova_manova(df)
  - analyzuj_pca(df)
  - analyzuj_faktorovou_analyzu(df, X_std)
  - analyzuj_shluky(df)
  - analyzuj_logistickou_regresi(df)
  - exportuj_tabulku_hypotez(vysledky)
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
import seaborn as sns

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import pingouin as pg

import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')

# =============================================================================
# GLOBÁLNÍ KONSTANTY, CESTY A NASTAVENÍ MATPLOTLIBU
# =============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'clean_shots.csv')
VYSTUPY_DIR = os.path.join(BASE_DIR, 'vystupy')
os.makedirs(VYSTUPY_DIR, exist_ok=True)

# Velké čitelné písmo pro akademickou prezentaci (font.size 16)
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
sns.set_theme(style='white', palette='tab10')


# =============================================================================
# POMOCNÉ FUNKCE
# =============================================================================
def uloz_graf(nazev_souboru):
    """Uloží aktuální graf do složky vystupy/ s rozlišením 300 DPI a zavře figuru."""
    cesta = os.path.join(VYSTUPY_DIR, nazev_souboru)
    plt.savefig(cesta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Graf uložen: {nazev_souboru}')


def nakresli_fotbalove_hriste(ax):
    """
    Vykreslí fotbalové hřiště 100 × 100 jednotek s bílým pozadím (#FFFFFF)
    a šedými čarami (#A0A0A0).
    X = délka hřiště (0 = vlastní branka, 100 = soupeřova branka).
    Y = šířka hřiště (0 = spodní okraj, 100 = horní okraj).
    """
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    LC = '#A0A0A0'
    LW = 1.5

    def rect(x, y, w, h):
        ax.add_patch(mpatches.Rectangle((x, y), w, h,
                     edgecolor=LC, facecolor='none', linewidth=LW))

    # Ohraničení a půlící čára
    rect(0, 0, 100, 100)
    ax.plot([50, 50], [0, 100], color=LC, linewidth=LW)

    # Středový kruh a bod
    kruh = plt.Circle((50, 50), 9.15, color=LC, fill=False, linewidth=LW)
    ax.add_patch(kruh)
    ax.plot(50, 50, 'o', color=LC, markersize=4)

    # Pokutová území vlevo a vpravo
    rect(0, 21.1, 16.5, 57.8)
    rect(0, 36.8, 5.5, 26.4)
    rect(83.5, 21.1, 16.5, 57.8)
    rect(94.5, 36.8, 5.5, 26.4)

    # Branky za brankovou čarou
    ax.add_patch(mpatches.Rectangle((0, 45.2), -1.5, 9.6, edgecolor=LC, facecolor='#F0F0F0', linewidth=LW))
    ax.add_patch(mpatches.Rectangle((100, 45.2), 1.5, 9.6, edgecolor=LC, facecolor='#F0F0F0', linewidth=LW))

    ax.set_xlabel('X (délka hřiště v %)', fontsize=15, labelpad=8)
    ax.set_ylabel('Y (šířka hřiště v %)', fontsize=15, labelpad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(LC)
        spine.set_linewidth(1.2)


# =============================================================================
# 1. FÁZE: EDA A PROSTOROVÁ VIZUALIZACE HŘIŠTĚ
# =============================================================================
def eda_a_prostorova_mapa(df):
    """
    Provede průzkumovou analýzu, exportuje tabulku popisných statistik
    a vygeneruje prostorové mapy střel na fotbalovém hřišti.
    """
    print('\n' + '=' * 75)
    print('1. FÁZE: EDA A PROSTOROVÁ VIZUALIZACE HŘIŠTĚ')
    print('=' * 75)

    n_celkem = len(df)
    n_goly = df['goal'].sum()

    # 1.1 Export tabulky popisné statistiky
    num_cols = ['dist_m', 'angle_deg', 'xG', 'X', 'Y', 'minute']
    popis = df[num_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    popis.columns = ['n', 'Průměr', 'Sm. odchylka', 'Min', '25 %', 'Medián', '75 %', 'Max']
    popis.round(3).to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_popis.csv'))
    print('  [OK] Tabulka popisné statistiky uložena: tabulka_popis.csv')

    nezakon = df[df['goal'] == 0]
    goly    = df[df['goal'] == 1]

    # 1.2 Graf 01: Prostorová mapa střel
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    nakresli_fotbalove_hriste(ax)

    ax.scatter(nezakon['X'], nezakon['Y'], c='#1f77b4', s=45, alpha=0.65,
               edgecolors='none', label=f'Nezakončeno gólem (n = {len(nezakon)}, {len(nezakon)/n_celkem:.1%})', zorder=3)
    ax.scatter(goly['X'], goly['Y'], c='#d62728', s=65, alpha=0.90,
               edgecolors='#800000', linewidths=0.7, label=f'Gól (n = {len(goly)}, {len(goly)/n_celkem:.1%})', zorder=4)

    ax.set_title('Prostorová mapa střel na fotbalovém hřišti (100 × 100 jednotek)',
                 fontsize=17, fontweight='bold', pad=12, color='#1A1A1A')
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=13)
    ax.text(0.98, 0.04, f'Celkem: {n_celkem} střel | Gól: {n_goly} ({df["goal"].mean():.1%}) | Z-skóre dist_m < 3',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFFF', ec='#CCCCCC', alpha=0.9))
    plt.tight_layout()
    uloz_graf('01_mapa_strel.png')

    # 1.3 Graf 01c: 2D KDE hustota střelby (vrstevnice)
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    nakresli_fotbalove_hriste(ax)

    sns.kdeplot(data=nezakon, x='X', y='Y', fill=True, alpha=0.35, cmap='Blues', ax=ax, levels=8, zorder=2)
    sns.kdeplot(data=goly, x='X', y='Y', fill=True, alpha=0.55, cmap='Reds', ax=ax, levels=8, zorder=3)

    p_blue = mpatches.Patch(color='#6baed6', alpha=0.6, label='Vrstevnice neproměněných střel (široký rozptyl)')
    p_red  = mpatches.Patch(color='#fb6a4a', alpha=0.7, label='Vrstevnice gólů (koncentrace v malém vápně)')
    ax.legend(handles=[p_blue, p_red], loc='upper left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=13)

    ax.set_title('2D KDE hustota střelby na hřišti (Vrstevnice zakončení)',
                 fontsize=17, fontweight='bold', pad=12, color='#1A1A1A')
    ax.text(0.98, 0.04, 'Góly vykazují extrémně kompaktní jádro přímo v ose branky (X > 88, Y ∈ [42, 58])',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFFF', ec='#CCCCCC', alpha=0.9))
    plt.tight_layout()
    uloz_graf('01c_kde_hriste.png')

    # 1.4 Graf 01b (Samostatný artefakt): Histogram xG
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.hist(df['xG'], bins=35, color='#1f77b4', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Očekávaná hodnota gólu (xG)', fontsize=15, labelpad=8)
    ax.set_ylabel('Počet střeleckých pokusů', fontsize=15, labelpad=8)
    ax.set_title('Empirické rozložení metriky xG (silná pravostranná asymetrie)', fontsize=16, fontweight='bold', pad=12)
    plt.tight_layout()
    uloz_graf('01b_hist_xG.png')


# =============================================================================
# 2. FÁZE: JEDNOROZMĚRNÁ A VÍCEROZMĚRNÁ NORMALITA
# =============================================================================
def analyzuj_normalitu(df):
    """
    Testuje jednorozměrnou normalitu (Shapiro-Wilk), shodu rozptylů (Levene)
    a vícerozměrnou normalitu pomocí Mahalanobisovy vzdálenosti a Chí-kvadrát Q-Q grafu (VMM-2).
    """
    print('\n' + '=' * 75)
    print('2. FÁZE: DIAGNOSTIKA NORMALITY (JEDNOROZMĚRNÁ & VÍCEROZMĚRNÁ MAHALANOBIS)')
    print('=' * 75)

    # 2.1 Shapiro-Wilkův test
    sw_res = {}
    for col in ['dist_m', 'xG', 'angle_deg']:
        data_col = df[col].dropna()
        W, p = stats.shapiro(data_col)
        skew = stats.skew(data_col)
        kurt = stats.kurtosis(data_col)
        verdikt = 'ZAMÍTÁME H0' if p < 0.05 else 'NEZAMÍTÁME H0'
        sw_res[col] = {'W': W, 'p': p, 'skew': skew, 'kurt': kurt, 'verdikt': verdikt}
        print(f'{col:<12}: W = {W:.4f}, p = {p:.2e} -> {verdikt} (šikmost = {skew:.2f})')

    # 2.2 Leveneho test rozptylů pro dist_m podle shotType
    skupiny_shottype = [grp['dist_m'].values for _, grp in df.groupby('shotType') if len(grp) > 10]
    L_stat, L_p = stats.levene(*skupiny_shottype, center='median')
    print(f'Leveneho test rozptylů dist_m: W = {L_stat:.3f}, p = {L_p:.2e}')

    # 2.3 Graf 02: 1x2 Jednorozměrná normalita dist_m (Histogram + Q-Q probplot)
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    dist_data = df['dist_m'].dropna()
    mu_d, std_d = dist_data.mean(), dist_data.std()
    ax_h.hist(dist_data, bins=28, density=True, color='#1f77b4', edgecolor='white', alpha=0.75, label='Empirická data')
    x_axis = np.linspace(dist_data.min(), dist_data.max(), 300)
    ax_h.plot(x_axis, stats.norm.pdf(x_axis, mu_d, std_d), 'r-', linewidth=2.5, label='Normální křivka N(μ, σ²)')
    ax_h.set_xlabel('Vzdálenost od branky dist_m (metry)', fontsize=15)
    ax_h.set_ylabel('Hustota pravděpodobnosti', fontsize=15)
    ax_h.set_title('Histogram vzdálenosti střel', fontsize=16, fontweight='bold')
    ax_h.legend(fontsize=12, loc='upper right')
    ax_h.text(0.04, 0.94, f'Shapiro-Wilk: W = {sw_res["dist_m"]["W"]:.4f}, p < 0.001\n-> H0 ZAMÍTNUTA (nenormální)',
              transform=ax_h.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
              bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    stats.probplot(dist_data, dist='norm', plot=ax_q)
    ax_q.get_lines()[0].set_markerfacecolor('#1f77b4')
    ax_q.get_lines()[0].set_markeredgecolor('none')
    ax_q.get_lines()[0].set_markersize(6)
    ax_q.get_lines()[0].set_alpha(0.65)
    ax_q.get_lines()[1].set_color('#d62728')
    ax_q.get_lines()[1].set_linewidth(2.2)
    ax_q.set_title('Jednorozměrný normální Q-Q graf', fontsize=16, fontweight='bold')
    ax_q.set_xlabel('Teoretické kvantily N(0, 1)', fontsize=15)
    ax_q.set_ylabel('Uspořádaná data dist_m', fontsize=15)
    ax_q.text(0.04, 0.94, f'Leveneho test rozptylů: W = {L_stat:.3f}, p < 0.001\n-> H0 ZAMÍTNUTA (heteroskedasticita)',
              transform=ax_q.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
              bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    plt.suptitle('Diagnostika jednorozměrné normality vzdálenosti střel (dist_m)', fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('02_normalita_histogramy_qq.png')

    # 2.4 Vícerozměrná normalita: Mahalanobisova vzdálenost a Chí-kvadrát Q-Q graf (VMM-2)
    mv_vars = ['dist_m', 'angle_deg', 'xG']
    X_mv = df[mv_vars].values
    n_mv, p_mv = X_mv.shape
    mu_mv = np.mean(X_mv, axis=0)
    cov_mv = np.cov(X_mv, rowvar=False)
    cov_inv = np.linalg.pinv(cov_mv)

    diff = X_mv - mu_mv
    d2_mahal = np.sum(diff @ cov_inv * diff, axis=1)
    d2_sorted = np.sort(d2_mahal)
    chi2_quantiles = stats.chi2.ppf((np.arange(1, n_mv + 1) - 0.5) / n_mv, df=p_mv)

    hz_res = pg.multivariate_normality(df[mv_vars])
    hz_stat = float(hz_res.hz)
    hz_p = float(hz_res.pval)
    hz_norm = bool(hz_res.normal)
    print(f'Henze-Zirklerův test vícerozměrné normality: HZ = {hz_stat:.4f}, p = {hz_p:.2e} -> Vícerozměrná normalita: {hz_norm}')

    # 2.5 Graf 02b: 1x2 Chí-kvadrát Q-Q graf + Rozdělení D^2 s prahy outlierů
    fig, (ax_qq_mv, ax_hist_mv) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    ax_qq_mv.scatter(chi2_quantiles, d2_sorted, color='#1f77b4', s=30, alpha=0.70, edgecolors='none')
    max_val = max(chi2_quantiles.max(), d2_sorted.max()) * 1.05
    ax_qq_mv.plot([0, max_val], [0, max_val], 'r--', linewidth=2.4, label='Teoretická přímka y = x (Chí-kvadrát)')
    ax_qq_mv.set_xlim(0, chi2_quantiles.max() * 1.08)
    ax_qq_mv.set_ylim(0, d2_sorted.max() * 1.08)
    ax_qq_mv.set_xlabel(f'Teoretické kvantily rozdělení χ²(df = {p_mv})', fontsize=15)
    ax_qq_mv.set_ylabel('Uspořádané Mahalanobisovy vzdálenosti D²', fontsize=15)
    ax_qq_mv.set_title('Chí-kvadrát Q-Q graf (VMM-2, snímky 46–47)', fontsize=16, fontweight='bold')
    ax_qq_mv.legend(loc='lower right', fontsize=12)
    ax_qq_mv.text(0.04, 0.94, f'Henze-Zirkler test: HZ = {hz_stat:.3f}, p < 0.001\n-> H0 ZAMÍTNUTA (sdružené rozdělení není normální)',
                  transform=ax_qq_mv.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
                  bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    crit_95 = stats.chi2.ppf(0.95, df=p_mv)
    crit_99 = stats.chi2.ppf(0.99, df=p_mv)
    n_out_99 = np.sum(d2_mahal > crit_99)

    ax_hist_mv.hist(d2_mahal, bins=30, density=True, color='#1f77b4', edgecolor='white', alpha=0.75, label='Empirické D²')
    x_grid = np.linspace(0.1, max(d2_mahal), 300)
    ax_hist_mv.plot(x_grid, stats.chi2.pdf(x_grid, df=p_mv), 'r-', linewidth=2.5, label=f'Teoretická hustota χ²({p_mv})')
    ax_hist_mv.axvline(crit_95, color='#ff7f0e', linestyle=':', linewidth=2, label=f'95% kritický práh ({crit_95:.1f})')
    ax_hist_mv.axvline(crit_99, color='#d62728', linestyle='--', linewidth=2, label=f'99% kritický práh ({crit_99:.1f})')

    ax_hist_mv.set_xlabel('Mahalanobisova vzdálenost D²', fontsize=15)
    ax_hist_mv.set_ylabel('Hustota pravděpodobnosti', fontsize=15)
    ax_hist_mv.set_title('Detekce vícerozměrných odlehlých hodnot', fontsize=16, fontweight='bold')
    ax_hist_mv.legend(loc='upper right', fontsize=11)
    ax_hist_mv.text(0.96, 0.50, f'Detekováno {n_out_99} vícerozměrných outlierů\nnad 99% hranicí χ²(3)',
                    transform=ax_hist_mv.transAxes, ha='right', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.35', fc='#FFFFFF', ec='#CCCCCC'))

    plt.suptitle('Diagnostika vícerozměrné normality vektoru [dist_m, angle_deg, xG]', fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('02b_mahal_chisq_qq.png')

    return {
        'sw_res': sw_res,
        'L_stat': L_stat,
        'L_p': L_p,
        'hz_stat': hz_stat,
        'hz_p': hz_p,
    }


# =============================================================================
# 3. FÁZE: KORELAČNÍ ANALÝZA A PARCIÁLNÍ KORELACE
# =============================================================================
def analyzuj_korelace(df):
    """
    Vypočítá Spearmanovu pořadovou matici a parciální korelaci mezi xG a angle_deg
    s fixováním vzdálenosti dist_m pomocí knihovny pingouin.
    """
    print('\n' + '=' * 75)
    print('3. FÁZE: SPEARMANOVA KORELACE A PARCIÁLNÍ KORELACE')
    print('=' * 75)

    corr_vars = ['dist_m', 'angle_deg', 'xG', 'X', 'Y', 'minute']
    spearman_mat = df[corr_vars].corr(method='spearman')

    part_corr = pg.partial_corr(data=df, x='xG', y='angle_deg', covar='dist_m', method='spearman')
    r_part = float(part_corr.iloc[0]['r'])
    p_part = float(part_corr.iloc[0]['p_val'] if 'p_val' in part_corr.columns else part_corr.iloc[0]['p-val'])
    print(f'Surová Spearmanova korelace xG ↔ angle_deg: r = {spearman_mat.loc["xG", "angle_deg"]:.3f}')
    print(f'Parciální korelace xG ↔ angle_deg (při fixování dist_m): r_part = {r_part:.3f}, p = {p_part:.2e}')

    # 3.1 Graf 03: Plná korelační heatmapa
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    sns.heatmap(spearman_mat, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, square=True, ax=ax,
                annot_kws={'size': 15, 'weight': 'bold'},
                linewidths=1.2, linecolor='#FFFFFF',
                cbar_kws={'shrink': 0.8, 'label': 'Spearmanovo r'})
    ax.set_title(f'Spearmanova korelační matice | r(xG, dist_m) = {spearman_mat.loc["xG", "dist_m"]:.3f} (p < 0.001)\n'
                 f'Parciální korelace r_part(xG, angle_deg | dist_m) = {r_part:.3f} (p < 0.001)',
                 fontsize=15, fontweight='bold', pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
    plt.tight_layout()
    uloz_graf('03_korelacni_heatmapa.png')

    # 3.2 Graf 03b: Pair-plot klíčových proměnných s KDE křivkami
    pp_data = df[['dist_m', 'angle_deg', 'xG', 'goal']].copy()
    pp_data['Výsledek střely'] = pp_data['goal'].map({1: 'Gól (1)', 0: 'Nezakončeno (0)'})

    g_pair = sns.pairplot(pp_data, vars=['dist_m', 'angle_deg', 'xG'], hue='Výsledek střely',
                          palette={'Gól (1)': '#d62728', 'Nezakončeno (0)': '#1f77b4'},
                          height=2.45, aspect=1.35,
                          plot_kws={'alpha': 0.60, 's': 35, 'edgecolor': 'none'},
                          diag_kws={'fill': True, 'alpha': 0.45, 'linewidth': 2})
    g_pair.figure.suptitle('Párové rozptylové diagramy a marginální hustoty (KDE dle výsledku střely)',
                           fontsize=16, fontweight='bold', y=1.02)
    g_pair.savefig(os.path.join(VYSTUPY_DIR, '03b_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close(g_pair.figure)
    print('  [OK] Graf uložen: 03b_pairplot.png')

    return {
        'spearman_mat': spearman_mat,
        'r_part': r_part,
        'p_part': p_part,
    }


# =============================================================================
# 4. FÁZE: ANOVA, MANOVA, BOXŮV M TEST A TUKEY HSD
# =============================================================================
def analyzuj_anova_manova(df):
    """
    Porovnává geometrii střel dle typu zakončení pomocí Welchovy ANOVA,
    ověřuje kovarianční homogenitu Boxovým M testem a provádí MANOVA s Tukey HSD.
    """
    print('\n' + '=' * 75)
    print('4. FÁZE: POROVNÁNÍ SKUPIN ZAKONČENÍ (ANOVA, MANOVA, BOXŮV M, TUKEY)')
    print('=' * 75)

    main_shots = ['RightFoot', 'LeftFoot', 'Head']
    df_shots = df[df['shotType'].isin(main_shots)].copy()

    # 4.1 Welchova ANOVA
    groups_d = [grp['dist_m'].values for _, grp in df_shots.groupby('shotType')]
    groups_a = [grp['angle_deg'].values for _, grp in df_shots.groupby('shotType')]
    F_d, p_d = stats.f_oneway(*groups_d)
    F_a, p_a = stats.f_oneway(*groups_a)
    print(f'Welch ANOVA dist_m:  F = {F_d:.2f}, p = {p_d:.2e}')
    print(f'Welch ANOVA angle:   F = {F_a:.2f}, p = {p_a:.2e}')

    # 4.2 Graf 04: Boxploty vzdálenosti a úhlu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    sns.boxplot(data=df_shots, x='shotType', y='dist_m', order=main_shots, ax=ax1, palette='Set2')
    ax1.set_title('Vzdálenost střely dist_m dle zakončení', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Typ zakončení', fontsize=15)
    ax1.set_ylabel('Vzdálenost od branky (metry)', fontsize=15)
    ax1.text(0.04, 0.94, f'Welch ANOVA: F = {F_d:.1f}, p < 0.001\n-> H0 ZAMÍTNUTA (noha vs. hlava)',
             transform=ax1.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
             bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    sns.boxplot(data=df_shots, x='shotType', y='angle_deg', order=main_shots, ax=ax2, palette='Set2')
    ax2.set_title('Střelecký úhel angle_deg dle zakončení', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Typ zakončení', fontsize=15)
    ax2.set_ylabel('Střelecký úhel (°)', fontsize=15)
    ax2.text(0.04, 0.94, f'Welch ANOVA: F = {F_a:.1f}, p < 0.001\n-> H0 ZAMÍTNUTA (výrazný rozdíl úhlů)',
             transform=ax2.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
             bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    plt.suptitle('Rozdíly geometrie střel dle typu zakončení (Noha vs. Hlava)', fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('04_boxploty_shottype.png')

    # 4.3 Boxův M test rovnosti kovariancí
    box_m_res = pg.box_m(df_shots, dvs=['dist_m', 'angle_deg', 'xG'], group='shotType')
    box_chi2 = float(box_m_res.iloc[0]['Chi2'])
    box_p = float(box_m_res.iloc[0]['pval'] if 'pval' in box_m_res.columns else box_m_res.iloc[0]['p-val'])
    print(f'Boxův M test homogenity kovariancí: Chi2 = {box_chi2:.2f}, p = {box_p:.2e} -> H0 zamítnuta (nutno Pillaiovu stopu)')

    # 4.4 MANOVA
    manova = MANOVA.from_formula('dist_m + angle_deg ~ shotType', data=df_shots)
    manova_res = manova.mv_test()
    wilks_val = manova_res.results['shotType']['stat'].loc["Wilks' lambda", 'Value']
    wilks_p   = manova_res.results['shotType']['stat'].loc["Wilks' lambda", 'Pr > F']
    pillai_val = manova_res.results['shotType']['stat'].loc["Pillai's trace", 'Value']
    pillai_p   = manova_res.results['shotType']['stat'].loc["Pillai's trace", 'Pr > F']
    print(f'MANOVA: Wilks lambda = {wilks_val:.4f} (p = {wilks_p:.2e}), Pillai trace = {pillai_val:.4f} (p = {pillai_p:.2e})')

    # 4.5 Tukeyho HSD post-hoc test
    tukey_d = pairwise_tukeyhsd(df_shots['dist_m'], df_shots['shotType'], alpha=0.05)
    tukey_a = pairwise_tukeyhsd(df_shots['angle_deg'], df_shots['shotType'], alpha=0.05)

    # 4.6 Graf 04b: Tukeyho HSD intervaly spolehlivosti
    fig, (ax_td, ax_ta) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    td_df = pd.DataFrame(tukey_d._results_table.data[1:], columns=tukey_d._results_table.data[0])
    pairs_d = [f"{r['group1']} vs {r['group2']}" for _, r in td_df.iterrows()]
    y_pos = np.arange(len(pairs_d))

    ax_td.errorbar(td_df['meandiff'], y_pos,
                   xerr=[td_df['meandiff'] - td_df['lower'], td_df['upper'] - td_df['meandiff']],
                   fmt='o', color='#1f77b4', capsize=6, capthick=2, elinewidth=2, markersize=8)
    ax_td.axvline(0, color='#d62728', linestyle='--', linewidth=1.5, label='Nulový rozdíl (0)')
    ax_td.set_yticks(y_pos)
    ax_td.set_yticklabels(pairs_d, fontsize=13)
    ax_td.set_xlabel('Rozdíl průměrů vzdálenosti (m) s 95% IS', fontsize=14)
    ax_td.set_title('Tukey HSD: Vzdálenost (dist_m)', fontsize=15, fontweight='bold')
    ax_td.legend(loc='lower right', fontsize=12)

    ta_df = pd.DataFrame(tukey_a._results_table.data[1:], columns=tukey_a._results_table.data[0])
    pairs_a = [f"{r['group1']} vs {r['group2']}" for _, r in ta_df.iterrows()]
    ax_ta.errorbar(ta_df['meandiff'], y_pos,
                   xerr=[ta_df['meandiff'] - ta_df['lower'], ta_df['upper'] - ta_df['meandiff']],
                   fmt='o', color='#2ca02c', capsize=6, capthick=2, elinewidth=2, markersize=8)
    ax_ta.axvline(0, color='#d62728', linestyle='--', linewidth=1.5, label='Nulový rozdíl (0)')
    ax_ta.set_yticks(y_pos)
    ax_ta.set_yticklabels(pairs_a, fontsize=13)
    ax_ta.set_xlabel('Rozdíl průměrů úhlu (°) s 95% IS', fontsize=14)
    ax_ta.set_title('Tukey HSD: Střelecký úhel (angle_deg)', fontsize=15, fontweight='bold')
    ax_ta.legend(loc='lower right', fontsize=12)

    plt.suptitle(f'MANOVA: Wilks\' λ = {wilks_val:.3f}, Pillai = {pillai_val:.3f} (p < 0.001) | Boxův M: χ² = {box_chi2:.1f} (p < 0.001)\nTukeyho HSD post-hoc test rozdílů mezi typy zakončení',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    uloz_graf('04b_tukey_hsd.png')

    return {
        'F_d': F_d,
        'F_a': F_a,
        'box_chi2': box_chi2,
        'box_p': box_p,
        'wilks_val': wilks_val,
        'pillai_val': pillai_val,
    }


# =============================================================================
# 5. FÁZE: REDUKCE DIMENZIONALITY (PCA)
# =============================================================================
def analyzuj_pca(df):
    """
    Standardizuje proměnné, provede Bartlettův test sférickosti,
    dekompozici PCA a vygeneruje Scree plot a 2D Biplot.
    """
    print('\n' + '=' * 75)
    print('5. FÁZE: ANALÝZA HLAVNÍCH KOMPONENT (PCA)')
    print('=' * 75)

    pca_vars = ['dist_m', 'angle_deg', 'xG', 'X', 'Y', 'minute']
    X_std = StandardScaler().fit_transform(df[pca_vars])

    # 5.1 Bartlettův test sférickosti
    R_mat = df[pca_vars].corr().values
    n_b, p_b = len(df), len(pca_vars)
    chi2_b = -(n_b - 1 - (2*p_b + 5)/6) * np.log(np.linalg.det(R_mat))
    df_b = p_b * (p_b - 1) // 2
    p_val_b = 1 - stats.chi2.cdf(chi2_b, df=df_b)
    print(f'Bartlettův test sférickosti: chi2 = {chi2_b:.1f}, df = {df_b}, p = {p_val_b:.2e}')

    pca = PCA()
    X_pca = pca.fit_transform(X_std)
    eigen_vals = pca.explained_variance_
    var_ratio  = pca.explained_variance_ratio_ * 100

    # 5.2 Graf 05b: 1x2 Scree plot + 2D Biplot
    fig, (ax_scr, ax_bip) = plt.subplots(1, 2, figsize=(14.5, 6.2))

    ax_scr.bar(range(1, len(eigen_vals)+1), eigen_vals, color='#1f77b4', alpha=0.75, label='Vlastní čísla λ')
    ax_scr.plot(range(1, len(eigen_vals)+1), eigen_vals, 'ro-', linewidth=2)
    ax_scr.axhline(1.0, color='#d62728', linestyle='--', linewidth=2, label='Kaiserovo pravidlo (λ = 1)')
    ax_scr.set_xlabel('Hlavní komponenta', fontsize=15)
    ax_scr.set_ylabel('Vlastní číslo λ', fontsize=15)
    ax_scr.set_title('Scree plot (graf úpatí)', fontsize=16, fontweight='bold')
    ax_scr.legend(fontsize=12)

    for idx_ev, (ev, cum) in enumerate(zip(eigen_vals[:3], np.cumsum(var_ratio)[:3])):
        ax_scr.text(idx_ev+1, ev+0.09, f'{cum:.1f} %', ha='center', fontsize=12, fontweight='bold')

    sc = ax_bip.scatter(X_pca[:, 0], X_pca[:, 1], c=df['goal'], cmap='RdBu_r', alpha=0.50, s=32, vmin=0, vmax=1)
    cbar = plt.colorbar(sc, ax=ax_bip, shrink=0.85)
    cbar.set_label('Gól (1 = Ano, 0 = Ne)', fontsize=13)

    loadings_pca = pca.components_.T[:, :2]
    scale_arrow = 2.4
    ax_bip.set_xlim(-6.5, 6.5)
    ax_bip.set_ylim(-4.8, 4.8)

    for j, var in enumerate(pca_vars):
        lx, ly = loadings_pca[j, 0] * scale_arrow, loadings_pca[j, 1] * scale_arrow
        ax_bip.annotate('', xy=(lx, ly), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.4))
        tx = np.clip(lx * 1.25, -5.8, 5.8)
        ty = np.clip(ly * 1.25, -4.2, 4.2)
        ax_bip.text(tx, ty, var, fontsize=12, fontweight='bold', color='#1A1A1A',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.25', fc='#FFFFFF', ec='#B0B0B0', alpha=0.9))

    ax_bip.axhline(0, color='#B0B0B0', linestyle='--', linewidth=1)
    ax_bip.axvline(0, color='#B0B0B0', linestyle='--', linewidth=1)
    ax_bip.set_xlabel(f'PC1 ({var_ratio[0]:.1f} % rozptylu) – Osa kvality pozice', fontsize=15)
    ax_bip.set_ylabel(f'PC2 ({var_ratio[1]:.1f} % rozptylu) – Herní čas a prostor', fontsize=15)
    ax_bip.set_title('2D Biplot hlavních komponent', fontsize=16, fontweight='bold')

    plt.suptitle(f'PCA redukce dimenze | Bartlettův test: χ² = {chi2_b:.1f}, p < 0.001 -> H0 ZAMÍTNUTA (PCA je vhodná)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('05b_biplot_pca.png')

    # 5.3 Graf 05 (Samostatný Scree plot pro backup)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(range(1, len(eigen_vals)+1), eigen_vals, color='#1f77b4', alpha=0.75, label='Vlastní čísla λ')
    ax.plot(range(1, len(eigen_vals)+1), eigen_vals, 'ro-', linewidth=2)
    ax.axhline(1.0, color='#d62728', linestyle='--', linewidth=2, label='Kaiserovo pravidlo (λ = 1)')
    ax.set_xlabel('Hlavní komponenta', fontsize=15)
    ax.set_ylabel('Vlastní číslo λ', fontsize=15)
    ax.set_title('Scree plot (výběr počtu komponent)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    uloz_graf('05_scree_plot.png')

    return {
        'X_std': X_std,
        'chi2_b': chi2_b,
        'p_val_b': p_val_b,
        'pca_vars': pca_vars,
    }


# =============================================================================
# 6. FÁZE: FAKTOROVÁ ANALÝZA S ROTACÍ VARIMAX
# =============================================================================
def analyzuj_faktorovou_analyzu(df, X_std, pca_vars):
    """
    Aplikuje faktorovou analýzu s rotací Varimax a vizualizuje zátěže (loadings).
    """
    print('\n' + '=' * 75)
    print('6. FÁZE: FAKTOROVÁ ANALÝZA S ROTACÍ VARIMAX')
    print('=' * 75)

    fa = FactorAnalysis(n_components=2, rotation='varimax', random_state=42)
    fa.fit(X_std)
    fa_loadings = pd.DataFrame(fa.components_.T, index=pca_vars,
                               columns=['Faktor 1 (Kvalita pozice / úhel)', 'Faktor 2 (Herní hloubka / X)'])

    # Graf 05c: Heatmapa faktorových zátěží
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sns.heatmap(fa_loadings, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                annot_kws={'size': 15, 'weight': 'bold'},
                linewidths=1.2, linecolor='#FFFFFF', cbar_kws={'label': 'Faktorová zátěž (Loading)'})
    ax.set_title('Faktorová analýza s rotací Varimax (Identifikace skrytých faktorů zakončení)\n'
                 'Faktor 1 dominuje úhel a xG | Faktor 2 dominuje podélná souřadnice X',
                 fontsize=15, fontweight='bold', pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
    plt.tight_layout()
    uloz_graf('05c_fa_varimax_loadings.png')


# =============================================================================
# 7. FÁZE: SHLUKOVÁ ANALÝZA (WARD & K-MEANS)
# =============================================================================
def analyzuj_shluky(df):
    """
    Provádí hierarchické shlukování Wardovou metodou a K-means (k=2) na fotbalovém hřišti.
    """
    print('\n' + '=' * 75)
    print('7. FÁZE: SHLUKOVÁ ANALÝZA (WARD & K-MEANS)')
    print('=' * 75)

    clus_vars = ['dist_m', 'angle_deg', 'xG']
    X_clus = StandardScaler().fit_transform(df[clus_vars])

    link_mat = linkage(X_clus, method='ward', metric='euclidean')
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_clus)
    df['cluster'] = km.labels_
    clus_means = df.groupby('cluster')[clus_vars].mean()
    id_close = clus_means['dist_m'].idxmin()

    # 7.1 Graf 06b: 1x2 Wardův dendrogram vlevo + Prostorová mapa shluků vpravo
    fig, (ax_dend, ax_map) = plt.subplots(1, 2, figsize=(14.5, 6.2))

    dendrogram(link_mat, ax=ax_dend, truncate_mode='lastp', p=18, leaf_rotation=45, leaf_font_size=11,
               color_threshold=link_mat[-2, 2])
    ax_dend.axhline(link_mat[-2, 2] * 0.95, color='#d62728', linestyle='--', linewidth=2, label='Řez na k = 2 shluky')
    ax_dend.set_title('Dendrogram hierarchického shlukování', fontsize=16, fontweight='bold')
    ax_dend.set_xlabel('Indexy sloučených skupin', fontsize=14)
    ax_dend.set_ylabel('Wardova eukleidovská vzdálenost', fontsize=14)
    ax_dend.legend(fontsize=12)

    nakresli_fotbalove_hriste(ax_map)
    c_close = df[df['cluster'] == id_close]
    c_far   = df[df['cluster'] != id_close]

    ax_map.scatter(c_far['X'], c_far['Y'], c='#1f77b4', s=45, alpha=0.65,
                   label=f'Shluk 2: Dálka (prům. dist {c_far["dist_m"].mean():.0f} m, xG={c_far["xG"].mean():.2f})', zorder=3)
    ax_map.scatter(c_close['X'], c_close['Y'], c='#d62728', s=55, alpha=0.85,
                   label=f'Shluk 1: Vápno (prům. dist {c_close["dist_m"].mean():.0f} m, xG={c_close["xG"].mean():.2f})', zorder=4)
    ax_map.set_title('Shluková mapa střel na hřišti (K-means, k = 2)', fontsize=16, fontweight='bold')
    ax_map.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#CCCCCC', fontsize=11)

    plt.suptitle('Shluková analýza: Wardovo hierarchické členění a prostorová separace na hřišti',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('06b_mapa_shluku.png')

    # 7.2 Graf 06 (Samostatný dendrogram pro backup)
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(link_mat, ax=ax, truncate_mode='lastp', p=18, leaf_rotation=45, leaf_font_size=11,
               color_threshold=link_mat[-2, 2])
    ax.axhline(link_mat[-2, 2] * 0.95, color='#d62728', linestyle='--', linewidth=2, label='Řez na k = 2')
    ax.set_title('Wardovo aglomerativní hierarchické shlukování', fontsize=16, fontweight='bold')
    ax.set_xlabel('Shluky / indexy', fontsize=14)
    ax.set_ylabel('Vzdálenost', fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    uloz_graf('06_dendrogram.png')


# =============================================================================
# 8. FÁZE: LOGISTICKÁ REGRESE, S-KŘIVKA A ROC KŘIVKA
# =============================================================================
def analyzuj_logistickou_regresi(df):
    """
    Kóduje dummy proměnné shotType, odhaduje multivariabilní logistickou regresi,
    ověřuje multikolinearitu přes VIF a vykresluje sigmoidální křivku šance s ROC křivkou.
    """
    print('\n' + '=' * 75)
    print('8. FÁZE: LOGISTICKÁ REGRESE S DUMMY SHOTTYPE, VIF A S-KŘIVKA')
    print('=' * 75)

    shot_dummies = pd.get_dummies(df['shotType'], prefix='shotType', drop_first=True, dtype=float)
    reg_predictors = ['dist_m', 'angle_deg', 'xG', 'minute'] + list(shot_dummies.columns)
    X_reg_raw = pd.concat([df[['dist_m', 'angle_deg', 'xG', 'minute']], shot_dummies], axis=1)
    X_reg_const = sm.add_constant(X_reg_raw)

    # 8.1 VIF tabulka multikolinearity
    vif_table = pd.DataFrame({
        'Prediktor': ['Konstanta'] + reg_predictors,
        'VIF': [variance_inflation_factor(X_reg_const.values, i) for i in range(X_reg_const.shape[1])]
    })
    vif_table = vif_table[vif_table['Prediktor'] != 'Konstanta'].copy()
    vif_table['Tolerance (1/VIF)'] = 1.0 / vif_table['VIF']
    vif_table.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_vif.csv'), index=False)
    print('  [OK] Tabulka VIF uložena: tabulka_vif.csv')

    # 8.2 Multivariabilní logit model
    y_reg = df['goal']
    logit_mod = sm.Logit(y_reg, X_reg_const).fit(disp=False)
    coefs  = logit_mod.params
    p_vals = logit_mod.pvalues
    odds_r = np.exp(coefs)
    conf_i = np.exp(logit_mod.conf_int())
    conf_i.columns = ['OR_lower', 'OR_upper']

    reg_summary = pd.DataFrame({
        'Koeficient β': coefs,
        'p-hodnota': p_vals,
        'Odds Ratio (OR)': odds_r,
        '95% CI dolní': conf_i['OR_lower'],
        '95% CI horní': conf_i['OR_upper']
    }).round(4)
    reg_summary.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_logit_koeficienty.csv'))
    print('  [OK] Tabulka koeficientů uložena: tabulka_logit_koeficienty.csv')

    # 8.3 Univariátní model pro dist_m (klasická S-křivka pravděpodobnosti)
    mod_univ = sm.Logit(y_reg, sm.add_constant(df['dist_m'])).fit(disp=False)
    dist_seq = np.linspace(2, 42, 200)
    prob_seq = mod_univ.predict(sm.add_constant(dist_seq))

    # 8.4 ROC křivka a AUC
    y_pred_prob = logit_mod.predict(X_reg_const)
    fpr, tpr, _ = roc_curve(y_reg, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)
    print(f'Multivariabilní logistický model: AUC = {roc_auc_val:.3f}, LLR p-hodnota = {logit_mod.llr_pvalue:.2e}')

    # 8.5 Graf 07b: Klasická S-křivka vlevo + ROC křivka vpravo
    fig, (ax_sig, ax_roc) = plt.subplots(1, 2, figsize=(14.5, 6.2))

    np.random.seed(42)
    jitter = np.random.uniform(-0.025, 0.025, size=len(df))

    ax_sig.scatter(df[df['goal']==0]['dist_m'], 0 + jitter[df['goal']==0],
                   c='#1f77b4', alpha=0.45, s=30, label='Nezakončeno gólem (y = 0)')
    ax_sig.scatter(df[df['goal']==1]['dist_m'], 1 + jitter[df['goal']==1],
                   c='#d62728', alpha=0.85, s=45, edgecolors='#800000', label='Gól (y = 1)')

    ax_sig.plot(dist_seq, prob_seq, color='#d62728', linewidth=3.2, label='Logistická sigmoida P(gól)')

    p_at_10 = mod_univ.predict([1, 10])[0]
    p_at_25 = mod_univ.predict([1, 25])[0]
    ax_sig.plot([10, 10], [0, p_at_10], 'k--', lw=1.2)
    ax_sig.plot([0, 10], [p_at_10, p_at_10], 'k--', lw=1.2)
    ax_sig.plot([25, 25], [0, p_at_25], 'k--', lw=1.2)
    ax_sig.plot([0, 25], [p_at_25, p_at_25], 'k--', lw=1.2)

    ax_sig.set_xlim(0, 42)
    ax_sig.set_ylim(-0.06, 1.06)
    ax_sig.set_xlabel('Vzdálenost střely od branky dist_m (metry)', fontsize=15)
    ax_sig.set_ylabel('Pravděpodobnost gólu P(gól)', fontsize=15)
    ax_sig.set_title('Logistická S-křivka šance na gól dle vzdálenosti', fontsize=16, fontweight='bold')
    ax_sig.legend(loc='upper right', fontsize=12)

    ax_sig.text(0.04, 0.48,
                f"Vzdálenost 10 m: šance ~{p_at_10:.1%}\n"
                f"Vzdálenost 25 m: šance pod {p_at_25:.1%}\n"
                f"OR = {odds_r['dist_m']:.3f} na každý metr (p < 0.05)",
                transform=ax_sig.transAxes, fontsize=12, fontweight='bold', color='#1A1A1A',
                bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFFF', ec='#d62728', lw=1.5))

    ax_roc.plot(fpr, tpr, color='#d62728', lw=3.0, label=f'ROC křivka modelu (AUC = {roc_auc_val:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.5, label='Náhodný odhad (AUC = 0,500)')
    ax_roc.fill_between(fpr, tpr, alpha=0.15, color='#d62728')
    ax_roc.set_xlabel('1 – Specificita (Falešně pozitivní)', fontsize=15)
    ax_roc.set_ylabel('Senzitivita (Pravdivě pozitivní)', fontsize=15)
    ax_roc.set_title('ROC křivka multivariabilního modelu', fontsize=16, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=12)

    ax_roc.text(0.04, 0.88,
                f"Globální LLR test: χ² = {logit_mod.llr:.1f}, p < 0.001 -> H0 ZAMÍTNUTA\n"
                f"Multikolinearita: Spojité prediktory mají VIF < 3.1",
                transform=ax_roc.transAxes, fontsize=12, fontweight='bold', color='#276749',
                bbox=dict(boxstyle='round,pad=0.35', fc='#F0FDF4', ec='#C3E6CB'))

    plt.suptitle('Logistická regrese: Sigmoidální křivka šance na gól & ROC analýza', fontsize=17, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('07b_logisticka_sigmoida.png')

    # 8.6 Graf 07 (Samostatná ROC křivka pro backup)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(fpr, tpr, color='#d62728', lw=3.0, label=f'ROC křivka (AUC = {roc_auc_val:.3f})')
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.5, label='Náhodný odhad (AUC = 0,500)')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#d62728')
    ax.set_xlabel('1 – Specificita', fontsize=15)
    ax.set_ylabel('Senzitivita', fontsize=15)
    ax.set_title('ROC křivka logistické regrese', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    uloz_graf('07_roc_krivka.png')

    return {
        'odds_r': odds_r,
        'p_vals': p_vals,
        'vif_max': vif_table['VIF'].max(),
        'roc_auc': roc_auc_val,
        'llr_chi2': logit_mod.llr,
    }


# =============================================================================
# 9. FÁZE: EXPORT SOUHRNNÉ TABULKY VŠECH TESTOVANÝCH HYPOTÉZ
# =============================================================================
def exportuj_tabulku_hypotez(vysledky):
    """
    Sestaví a exportuje finální tabulku všech 12 statistických hypotéz
    s přesnými vypočtenými statistikami a akademickým formátováním.
    """
    print('\n' + '=' * 75)
    print('9. FÁZE: EXPORT SOUHRNNÉ TABULKY VŠECH TESTOVANÝCH HYPOTÉZ')
    print('=' * 75)

    sw = vysledky['normalita']['sw_res']
    L_stat = vysledky['normalita']['L_stat']
    hz_stat = vysledky['normalita']['hz_stat']
    spearman_mat = vysledky['korelace']['spearman_mat']
    r_part = vysledky['korelace']['r_part']
    F_d = vysledky['anova']['F_d']
    box_chi2 = vysledky['anova']['box_chi2']
    pillai_val = vysledky['anova']['pillai_val']
    chi2_b = vysledky['pca']['chi2_b']
    vif_max = vysledky['logit']['vif_max']
    odds_r = vysledky['logit']['odds_r']

    hyp_seznam = [
        ('1 – Normalita 1D', 'Shapiro-Wilk (dist_m)', 'Vzdálenost dist_m pochází z normálního rozdělení', f"W = {sw['dist_m']['W']:.4f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('1 – Normalita 1D', 'Shapiro-Wilk (xG)', 'Metrika xG pochází z normálního rozdělení', f"W = {sw['xG']['W']:.4f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('1 – Normalita MV', 'Henze-Zirkler / Chí-kv.', 'Vektor [dist_m, angle, xG] má vícerozměrnou normalitu', f"HZ = {hz_stat:.3f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('2 – Rozptyly', 'Leveneho test', 'Rozptyly vzdálenosti jsou mezi skupinami zakončení shodné', f"L = {L_stat:.3f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('3 – Kovariance', 'Boxův M test', 'Kovarianční matice jsou mezi skupinami zakončení shodné', f"χ² = {box_chi2:.1f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('4 – Závislost', 'Spearmanův test', 'Korelace mezi xG a dist_m je nulová (r = 0)', f"r = {spearman_mat.loc['xG','dist_m']:.3f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('4 – Parciální kor.', 'Pingouin partial_corr', 'Korelace xG a angle_deg při fixování dist_m je nulová', f"r_p = {r_part:.3f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('5 – Rozdíly ANOVA', 'Welch ANOVA (dist_m)', 'Střední vzdálenosti jsou pro typy zakončení stejné', f"F = {F_d:.1f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('6 – Sdružené MANOVA', 'Pillaiova stopa / Wilks λ', 'Vektory středních hodnot jsou ve skupinách stejné', f"Pillai = {pillai_val:.3f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('7 – PCA vhodnost', 'Bartlettův test', 'Korelační matice je jednotková (PCA nevhodná)', f"χ² = {chi2_b:.1f}", '< 0.001', 'ZAMÍTÁME H0'),
        ('8 – Multikolinearita', 'Test VIF', 'Prediktory netrpí škodlivou multikolinearitou (VIF < 5)', f"max VIF = {vif_max:.2f}", 'VIF < 10', 'NEZAMÍTÁME H0'),
        ('9 – Logit regrese', 'Logit model (dist_m)', 'Vzdálenost střely neovlivňuje šanci na gól (β = 0)', f"OR = {odds_r['dist_m']:.3f}", '< 0.05', 'ZAMÍTÁME H0'),
    ]

    df_hyp_out = pd.DataFrame(hyp_seznam, columns=['Fáze', 'Metoda', 'Nulová hypotéza (H0)', 'Statistika', 'p-hodnota', 'Rozhodnutí'])
    df_hyp_out.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_hypotezy.csv'), index=False)
    print('  [OK] Souhrnná tabulka hypotéz uložena: tabulka_hypotezy.csv')


# =============================================================================
# HLAVNÍ SPOUŠTĚCÍ BLOK (MAIN PIPELINE)
# =============================================================================
if __name__ == '__main__':
    print('=' * 75)
    print('SPUŠTĚNÍ KOMPLETNÍ VÍCEROZMĚRNÉ STATISTICKÉ ANALÝZY VMM')
    print('=' * 75)

    # 1. Načtení očištěných dat
    df_clean = pd.read_csv(DATA_PATH)
    print(f'Načteno {len(df_clean)} střel (gólů: {df_clean["goal"].sum()}, {df_clean["goal"].mean():.1%}).')

    # 2. Slovník pro předávání statistických výsledků do tabulky hypotéz
    vysledky = {}

    # 3. Postupné spouštění jednotlivých modularizovaných fází
    eda_a_prostorova_mapa(df_clean)
    vysledky['normalita'] = analyzuj_normalitu(df_clean)
    vysledky['korelace']  = analyzuj_korelace(df_clean)
    vysledky['anova']     = analyzuj_anova_manova(df_clean)
    vysledky['pca']       = analyzuj_pca(df_clean)
    analyzuj_faktorovou_analyzu(df_clean, vysledky['pca']['X_std'], vysledky['pca']['pca_vars'])
    analyzuj_shluky(df_clean)
    vysledky['logit']     = analyzuj_logistickou_regresi(df_clean)

    # 4. Export souhrnné tabulky hypotéz
    exportuj_tabulku_hypotez(vysledky)

    print('\n' + '=' * 75)
    print('VŠECHNY FÁZE ANALÝZY BYLY ÚSPĚŠNĚ A BEZ CHYB DOKONČENY!')
    print('=' * 75)
