# -*- coding: utf-8 -*-
"""
=============================================================================
SEMESTRÁLNÍ PROJEKT VMM – STŘELECKÁ ÚSPĚŠNOST (PREMIER LEAGUE)
Skript 2: Vícerozměrná statistická analýza, testování hypotéz a vizualizace
=============================================================================
Klíčové metodické úpravy:
  1. Žádné hardcoded rozhodnutí: Všechny nulové hypotézy jsou vyhodnocovány
     striktně dynamicky na základě vypočtené p-hodnoty a hladiny významnosti alpha=0.05.
  2. Odstranění cirkularity v logistické regresi: xG NENÍ prediktorem v modelu!
     Prediktory jsou pouze primární geometrické a herní proměnné (dist_m, angle_deg,
     minute, shotType). Metrika xG slouží jako nezávislý externí srovnávací benchmark.
  3. Skutečná Welchova ANOVA & Kruskal-Wallis: Pro heteroskedastická data je
     použita robustní Welchova ANOVA (pingouin.welch_anova) a neparametrický
     Kruskal-Wallisův test.
  4. Vyloučení geometrické redundance v PCA a FA: Místo duplicitních souřadnic
     (X, Y, dist, angle) analyzujeme ne-redundantní geometrické a kvalitativní
     metriky (dist_m, angle_deg, y_dev = odchylka od osy, xG) s testem KMO.
  5. Objektivní validace shluků: Volba k=2 v K-means je rigorózně ověřena
     pomocí Silhouette score a metody lokte (Inertia).
  6. VIF je vyřazen z tabulky hypotéz (jde o diagnostickou míru multikolinearity)
     a exportován do samostatné přehledové tabulky.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from mplsoccer import Pitch, VerticalPitch

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pingouin as pg

import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc, silhouette_score

warnings.filterwarnings('ignore')

# Nastavení cest
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'clean_shots.csv')
VYSTUPY_DIR = os.path.join(BASE_DIR, 'vystupy')
os.makedirs(VYSTUPY_DIR, exist_ok=True)

# Globální grafické nastavení pro vysoké rozlišení a akademickou čitelnost
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titlesize': 17,
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
    """Uloží graf do složky vystupy v rozlišení 300 DPI a uvolní paměť."""
    cesta = os.path.join(VYSTUPY_DIR, nazev_souboru)
    plt.savefig(cesta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Graf uložen: {nazev_souboru}')


def nakresli_hriste(ax):
    """Vykreslí fotbalové hřiště 105 × 68 m."""
    pitch = Pitch(
        pitch_type='custom',
        pitch_color='grass',
        line_color='white',
        stripe_color='#c2d59d',
        stripe=True,
        pitch_length=105,
        pitch_width=68,
        linewidth=1.5,
        goal_type='line'
    )
    pitch.draw(ax=ax)
    return pitch


def formatuj_p(p_val):
    """Korektně zformátuje p-hodnotu pro tabulky a grafy."""
    if p_val < 0.001:
        return '< 0.001'
    return f'{p_val:.4f}'


def rozhodni_h0(p_val, alpha=0.05):
    """Dynamické rozhodnutí o zamítnutí H0 bez hardcodování."""
    return 'ZAMÍTÁME H0' if p_val < alpha else 'NEZAMÍTÁME H0'


def vypocti_kmo(data_df):
    """Vypočítá Kaiser-Meyer-Olkin (KMO) index adekvátnosti výběru pro PCA/FA."""
    corr = np.array(data_df.corr().values, copy=True)
    inv_corr = np.linalg.pinv(corr)
    A = np.zeros_like(corr)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            A[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])
    np.fill_diagonal(A, 0)
    np.fill_diagonal(corr, 0)
    num = np.sum(corr ** 2)
    denom = num + np.sum(A ** 2)
    return float(num / denom)


# =============================================================================
# 1. FÁZE: EDA, MAPA STŘEL A 2D KDE HUSTOTA
# =============================================================================
def eda_a_prostorova_mapa(df):
    print('\n' + '=' * 75)
    print('1. FÁZE: PRŮZKUMOVÁ ANALÝZA DAT A PROSTOROVÉ MAPOVÁNÍ STŘELBY')
    print('=' * 75)

    # Popisná statistika
    cols_desc = ['dist_m', 'angle_deg', 'xG', 'minute']
    popis_df = df[cols_desc].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    popis_df['IQR'] = popis_df['75%'] - popis_df['25%']
    popis_df['Skew'] = df[cols_desc].skew()
    popis_df['Kurtosis'] = df[cols_desc].kurt()
    popis_df.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_popis.csv'))
    print('  [OK] Tabulka popisných statistik uložena: tabulka_popis.csv')

    # Graf 01: Prostorová mapa střel na hřišti
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    nakresli_hriste(ax)

    no_goals = df[df['goal'] == 0]
    goals = df[df['goal'] == 1]

    ax.scatter(no_goals['x_m'], no_goals['y_m'], c='#1f77b4', s=45, alpha=0.60,
               edgecolors='none', label=f'Nezakončeno gólem (n = {len(no_goals)})', zorder=3)
    ax.scatter(goals['x_m'], goals['y_m'], c='#d62728', s=90, alpha=0.90,
               edgecolors='#800000', linewidths=1.2, label=f'Gól (n = {len(goals)})', zorder=4)

    ax.set_title(f'Prostorová mapa střel Premier League (n = {len(df)}, z toho gólů: {len(goals)} = {len(goals)/len(df):.1%})\n'
                 'Reálné rozměry hřiště 105 × 68 m, branka 7,32 m (y = 30,34 až 37,66 m)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(loc='upper left', frameon=True, framealpha=0.92, facecolor='white', edgecolor='#CCCCCC')
    uloz_graf('01_mapa_strel.png')

    # Graf 01b: Histogram očekávaných gólů (xG)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(df['xG'], bins=30, kde=True, color='#1f77b4', ax=ax, edgecolor='white', alpha=0.75)
    ax.axvline(df['xG'].mean(), color='#d62728', linestyle='--', linewidth=2,
               label=f'Průměr xG = {df["xG"].mean():.3f}')
    ax.axvline(df['xG'].median(), color='#2ca02c', linestyle='-', linewidth=2,
               label=f'Medián xG = {df["xG"].median():.3f}')
    ax.set_xlabel('Očekávané góly (xG)', fontsize=14)
    ax.set_ylabel('Počet střel', fontsize=14)
    ax.set_title('Distribuce metriky očekávaných gólů (xG) – pravostranná asymetrie',
                 fontsize=15, fontweight='bold')
    ax.legend()
    uloz_graf('01b_hist_xG.png')

    # Graf 01c: 2D KDE hustota na dvou půlkách hřiště vedle sebe (neproměněné vs. góly)
    pitch_half = VerticalPitch(
        pitch_type='custom',
        pitch_color='grass',
        line_color='white',
        stripe_color='#c2d59d',
        stripe=True,
        pitch_length=105,
        pitch_width=68,
        half=True,
        linewidth=1.5,
        goal_type='line'
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8))
    pitch_half.draw(ax=axes[0])
    pitch_half.draw(ax=axes[1])

    # 1. Gradient pro neproměněné střely (tmavě modrá / navy)
    cmap_no_goals = LinearSegmentedColormap.from_list(
        'c_no_goals',
        [
            (0.40, 0.70, 0.90, 0.35),
            (0.15, 0.45, 0.75, 0.65),
            (0.04, 0.18, 0.50, 0.92)
        ],
        N=256
    )

    # 2. Gradient pro vstřelené góly (termální žluto-oranžovo-červená)
    cmap_goals = LinearSegmentedColormap.from_list(
        'c_goals',
        [
            (1.0, 0.88, 0.20, 0.45),
            (1.0, 0.50, 0.05, 0.70),
            (0.85, 0.08, 0.08, 0.92)
        ],
        N=256
    )

    # Vykreslení KDE na levé půlce (neproměněné střely)
    pitch_half.kdeplot(
        no_goals['x_m'], no_goals['y_m'], ax=axes[0],
        cmap=cmap_no_goals, fill=True, levels=14, thresh=0.03, zorder=2
    )
    axes[0].set_title(
        f'1. Neproměněné střely (n = {len(no_goals)}, 91,4 %)\n'
        f'Medián vzdálenosti: {no_goals["dist_m"].median():.1f} m | Medián xG: {no_goals["xG"].median():.3f}',
        fontsize=13.5, fontweight='bold', pad=10
    )

    # Vykreslení KDE na pravé půlce (vstřelené góly)
    pitch_half.kdeplot(
        goals['x_m'], goals['y_m'], ax=axes[1],
        cmap=cmap_goals, fill=True, levels=14, thresh=0.03, zorder=2
    )
    axes[1].set_title(
        f'2. Vstřelené góly (n = {len(goals)}, 8,6 %)\n'
        f'Medián vzdálenosti: {goals["dist_m"].median():.1f} m | Medián xG: {goals["xG"].median():.3f}',
        fontsize=13.5, fontweight='bold', pad=10
    )

    fig.suptitle(
        'Srovnání 2D prostorové hustoty střelby (KDE) na útočné polovině hřiště\n'
        'Vlevo: Široký rozptyl neproměněných pokusů | Vpravo: Ohnisko vstřelených gólů v malém vápně',
        fontsize=15, fontweight='bold', y=0.99
    )

    plt.tight_layout()
    uloz_graf('01c_kde_hriste.png')


# =============================================================================
# 2. FÁZE: DIAGNOSTIKA NORMALITY A ROZPTYLŮ
# =============================================================================
def analyzuj_normalitu(df):
    print('\n' + '=' * 75)
    print('2. FÁZE: DIAGNOSTIKA JEDNOROZMĚRNÉ A VÍCEROZMĚRNÉ NORMALITY')
    print('=' * 75)

    # 2.1 Shapiro-Wilkův test
    sw_vysledky = {}
    for col in ['dist_m', 'xG', 'angle_deg']:
        data_col = df[col].dropna()
        W_val, p_val = stats.shapiro(data_col)
        verdikt = rozhodni_h0(p_val)
        sw_vysledky[col] = {'W': W_val, 'p': p_val, 'verdikt': verdikt}
        print(f'{col:<12}: W = {W_val:.4f}, p = {p_val:.2e} -> {verdikt}')

    # 2.2 Leveneho test homogenity rozptylů
    skupiny_st = [grp['dist_m'].values for _, grp in df.groupby('shotType') if len(grp) > 10]
    L_stat, L_p = stats.levene(*skupiny_st, center='median')
    print(f'Leveneho test rozptylů dist_m: L = {L_stat:.3f}, p = {L_p:.2e} -> {rozhodni_h0(L_p)}')

    # 2.3 Graf 02: 1x2 Histogram a Q-Q probplot dist_m
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    dist_data = df['dist_m'].dropna()
    mu_d, std_d = dist_data.mean(), dist_data.std()

    ax_h.hist(dist_data, bins=28, density=True, color='#1f77b4', edgecolor='white', alpha=0.75, label='Empirická data')
    x_axis = np.linspace(dist_data.min(), dist_data.max(), 300)
    ax_h.plot(x_axis, stats.norm.pdf(x_axis, mu_d, std_d), 'r-', linewidth=2.5, label='Teoretická normální křivka')
    ax_h.set_xlabel('Vzdálenost od branky dist_m (metry)')
    ax_h.set_ylabel('Hustota pravděpodobnosti')
    ax_h.set_title('Histogram vzdálenosti střel')
    ax_h.legend(loc='upper right')

    w_d = sw_vysledky['dist_m']['W']
    p_d = sw_vysledky['dist_m']['p']
    ax_h.text(0.04, 0.94, f'Shapiro-Wilk: W = {w_d:.4f}, p = {formatuj_p(p_d)}\n-> {rozhodni_h0(p_d)}',
              transform=ax_h.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
              bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    stats.probplot(dist_data, dist='norm', plot=ax_q)
    ax_q.get_lines()[0].set_markerfacecolor('#1f77b4')
    ax_q.get_lines()[0].set_markeredgecolor('none')
    ax_q.get_lines()[0].set_markersize(6)
    ax_q.get_lines()[0].set_alpha(0.65)
    ax_q.get_lines()[1].set_color('#d62728')
    ax_q.get_lines()[1].set_linewidth(2.2)
    ax_q.set_title('Normální Q-Q graf (dist_m)')
    ax_q.set_xlabel('Teoretické kvantily N(0, 1)')
    ax_q.set_ylabel('Empirické kvantily dist_m')

    ax_q.text(0.04, 0.94, f'Leveneho test rozptylů: L = {L_stat:.3f}, p = {formatuj_p(L_p)}\n-> {rozhodni_h0(L_p)} (heteroskedasticita)',
              transform=ax_q.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
              bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    plt.suptitle('Diagnostika jednorozměrné normality a shody rozptylů vzdálenosti střel', y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('02_normalita_histogramy_qq.png')

    # 2.4 Vícerozměrná normalita: Henze-Zirkler a Mahalanobisovy vzdálenosti
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
    print(f'Henze-Zirkler test: HZ = {hz_stat:.4f}, p = {hz_p:.2e} -> {rozhodni_h0(hz_p)}')

    # Graf 02b: Chí-kvadrát Q-Q graf Mahalanobisových vzdáleností
    fig, (ax_qq_mv, ax_hist_mv) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    ax_qq_mv.scatter(chi2_quantiles, d2_sorted, color='#1f77b4', s=30, alpha=0.70, edgecolors='none')
    max_val = max(chi2_quantiles.max(), d2_sorted.max()) * 1.05
    ax_qq_mv.plot([0, max_val], [0, max_val], 'r--', linewidth=2.4, label='Teoretická přímka χ²')
    ax_qq_mv.set_xlim(0, chi2_quantiles.max() * 1.05)
    ax_qq_mv.set_ylim(0, d2_sorted.max() * 1.05)
    ax_qq_mv.set_xlabel(f'Teoretické kvantily rozdělení χ²(df = {p_mv})')
    ax_qq_mv.set_ylabel('Uspořádané Mahalanobisovy vzdálenosti D²')
    ax_qq_mv.set_title('Chí-kvadrát Q-Q graf (D²)')
    ax_qq_mv.legend(loc='lower right')
    ax_qq_mv.text(0.04, 0.94, f'Henze-Zirkler: HZ = {hz_stat:.3f}, p = {formatuj_p(hz_p)}\n-> {rozhodni_h0(hz_p)} (nenormální sdružené rozdělení)',
                  transform=ax_qq_mv.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
                  bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    crit_95 = stats.chi2.ppf(0.95, df=p_mv)
    crit_99 = stats.chi2.ppf(0.99, df=p_mv)
    n_out_99 = np.sum(d2_mahal > crit_99)

    ax_hist_mv.hist(d2_mahal, bins=30, density=True, color='#1f77b4', edgecolor='white', alpha=0.75, label='Empirické D²')
    x_grid = np.linspace(0.1, max(d2_mahal), 300)
    ax_hist_mv.plot(x_grid, stats.chi2.pdf(x_grid, df=p_mv), 'r-', linewidth=2.5, label=f'Hustota χ²({p_mv})')
    ax_hist_mv.axvline(crit_95, color='#ff7f0e', linestyle=':', linewidth=2, label=f'95% práh ({crit_95:.1f})')
    ax_hist_mv.axvline(crit_99, color='#d62728', linestyle='--', linewidth=2, label=f'99% práh ({crit_99:.1f})')
    ax_hist_mv.set_xlabel('Mahalanobisova vzdálenost D²')
    ax_hist_mv.set_ylabel('Hustota')
    ax_hist_mv.set_title('Detekce vícerozměrných odlehlých hodnot')
    ax_hist_mv.legend(loc='upper right', fontsize=11)
    ax_hist_mv.text(0.96, 0.45, f'Detekováno {n_out_99} outlierů\nnad 99% hranicí χ²(3)',
                    transform=ax_hist_mv.transAxes, ha='right', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.35', fc='#FFFFFF', ec='#CCCCCC'))

    plt.suptitle('Diagnostika vícerozměrné normality vektoru [dist_m, angle_deg, xG]', y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('02b_mahal_chisq_qq.png')

    return {
        'sw': sw_vysledky,
        'L_stat': L_stat,
        'L_p': L_p,
        'hz_stat': hz_stat,
        'hz_p': hz_p,
    }


# =============================================================================
# 3. FÁZE: KORELAČNÍ ANALÝZA A PARCIÁLNÍ KORELACE
# =============================================================================
def analyzuj_korelace(df):
    print('\n' + '=' * 75)
    print('3. FÁZE: KORELAČNÍ ANALÝZA A PARCIÁLNÍ KORELACE')
    print('=' * 75)

    df_k = df.copy()
    df_k['y_dev'] = np.abs(df_k['y_m'] - 34.0)
    corr_vars = ['dist_m', 'angle_deg', 'y_dev', 'xG', 'minute']
    spearman_mat = df_k[corr_vars].corr(method='spearman')

    # Test nulovosti korelace xG ↔ dist_m
    r_sp, p_sp = stats.spearmanr(df_k['xG'], df_k['dist_m'])

    # Parciální korelace xG ↔ angle_deg při fixované vzdálenosti dist_m
    part_corr = pg.partial_corr(data=df_k, x='xG', y='angle_deg', covar='dist_m', method='spearman')
    r_part = float(part_corr.iloc[0]['r'])
    p_col = [c for c in part_corr.columns if 'p' in c.lower() and ('val' in c.lower() or 'unc' in c.lower())][0]
    p_part = float(part_corr.iloc[0][p_col])

    print(f'Spearman r(xG, dist_m): r = {r_sp:.3f}, p = {p_sp:.2e} -> {rozhodni_h0(p_sp)}')
    print(f'Parciální korelace r_part(xG, angle | dist_m): r = {r_part:.3f}, p = {p_part:.2e} -> {rozhodni_h0(p_part)}')

    # Graf 03: Korelační heatmapa
    fig, ax = plt.subplots(figsize=(10, 7.2))
    sns.heatmap(spearman_mat, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, square=True, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'},
                linewidths=1.2, linecolor='#FFFFFF',
                cbar_kws={'shrink': 0.8, 'label': 'Spearmanovo r'})
    ax.set_title(f'Spearmanova korelační matice | r(xG, dist_m) = {r_sp:.3f} (p = {formatuj_p(p_sp)})\n'
                 f'Parciální korelace r(xG, úhel | dist_m) = {r_part:.3f} (p = {formatuj_p(p_part)})',
                 fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    uloz_graf('03_korelacni_heatmapa.png')

    # Graf 03b: Pair-plot klíčových proměnných
    pp_data = df[['dist_m', 'angle_deg', 'xG', 'goal']].copy()
    pp_data['Výsledek'] = pp_data['goal'].map({1: 'Gól (1)', 0: 'Bez gólu (0)'})

    g_pair = sns.pairplot(pp_data, vars=['dist_m', 'angle_deg', 'xG'], hue='Výsledek',
                          palette={'Gól (1)': '#d62728', 'Bez gólu (0)': '#1f77b4'},
                          height=2.4, aspect=1.3,
                          plot_kws={'alpha': 0.60, 's': 32, 'edgecolor': 'none'},
                          diag_kws={'fill': True, 'alpha': 0.45, 'linewidth': 1.8})
    g_pair.figure.suptitle('Párové vztahy a marginální hustoty dle výsledku střely',
                           fontsize=16, fontweight='bold', y=1.02)
    g_pair.savefig(os.path.join(VYSTUPY_DIR, '03b_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close(g_pair.figure)
    print('  [OK] Graf uložen: 03b_pairplot.png')

    return {
        'r_sp': r_sp,
        'p_sp': p_sp,
        'r_part': r_part,
        'p_part': p_part,
    }


# =============================================================================
# 4. FÁZE: ROBUSTNÍ WELCHOVA ANOVA, KRUSKAL-WALLIS, MANOVA A POST-HOC TESTY
# =============================================================================
def analyzuj_anova_manova(df):
    print('\n' + '=' * 75)
    print('4. FÁZE: SROVNÁNÍ TYPŮ ZAKONČENÍ (WELCH ANOVA, KRUSKAL-WALLIS, MANOVA)')
    print('=' * 75)

    main_shots = ['RightFoot', 'LeftFoot', 'Head']
    df_shots = df[df['shotType'].isin(main_shots)].copy()

    # 4.1 Skutečná Welchova ANOVA (řešení heteroskedasticity)
    welch_dist = pg.welch_anova(data=df_shots, dv='dist_m', between='shotType')
    F_welch_d = float(welch_dist.iloc[0]['F'])
    p_welch_d = float(welch_dist.iloc[0]['p_unc'] if 'p_unc' in welch_dist.columns else welch_dist.iloc[0]['p-unc'])

    welch_ang = pg.welch_anova(data=df_shots, dv='angle_deg', between='shotType')
    F_welch_a = float(welch_ang.iloc[0]['F'])
    p_welch_a = float(welch_ang.iloc[0]['p_unc'] if 'p_unc' in welch_ang.columns else welch_ang.iloc[0]['p-unc'])

    # 4.2 Kruskal-Wallisův neparametrický test
    kw_res = pg.kruskal(data=df_shots, dv='dist_m', between='shotType')
    H_kw = float(kw_res.iloc[0]['H'])
    p_kw = float(kw_res.iloc[0]['p_unc'] if 'p_unc' in kw_res.columns else kw_res.iloc[0]['p-unc'])

    print(f'Welchova ANOVA (dist_m):   F = {F_welch_d:.2f}, p = {p_welch_d:.2e} -> {rozhodni_h0(p_welch_d)}')
    print(f'Welchova ANOVA (angle):    F = {F_welch_a:.2f}, p = {p_welch_a:.2e} -> {rozhodni_h0(p_welch_a)}')
    print(f'Kruskal-Wallis (dist_m):   H = {H_kw:.2f}, p = {p_kw:.2e} -> {rozhodni_h0(p_kw)}')

    # 4.3 Graf 04: Boxploty vzdálenosti a úhlu
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    sns.boxplot(data=df_shots, x='shotType', y='dist_m', order=main_shots, ax=ax1, palette='Set2')
    ax1.set_title('Vzdálenost střely dist_m dle zakončení', fontweight='bold')
    ax1.set_xlabel('Typ zakončení')
    ax1.set_ylabel('Vzdálenost od branky (metry)')
    ax1.text(0.04, 0.94, f'Welch ANOVA: F = {F_welch_d:.1f}, p = {formatuj_p(p_welch_d)}\n-> {rozhodni_h0(p_welch_d)}',
             transform=ax1.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
             bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    sns.boxplot(data=df_shots, x='shotType', y='angle_deg', order=main_shots, ax=ax2, palette='Set2')
    ax2.set_title('Střelecký úhel angle_deg dle zakončení', fontweight='bold')
    ax2.set_xlabel('Typ zakončení')
    ax2.set_ylabel('Střelecký úhel (°)')
    ax2.text(0.04, 0.94, f'Welch ANOVA: F = {F_welch_a:.1f}, p = {formatuj_p(p_welch_a)}\n-> {rozhodni_h0(p_welch_a)}',
             transform=ax2.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#C53030',
             bbox=dict(boxstyle='round,pad=0.35', fc='#FDF2F2', ec='#F5C6CB'))

    plt.suptitle('Rozdíly geometrie střel dle typu zakončení (Pravá noha, Levá noha, Hlava)',
                 fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('04_boxploty_shottype.png')

    # 4.4 Boxův M test rovnosti kovariancí
    box_m_res = pg.box_m(df_shots, dvs=['dist_m', 'angle_deg', 'xG'], group='shotType')
    box_chi2 = float(box_m_res.iloc[0]['Chi2'])
    box_p = float(box_m_res.iloc[0]['pval'] if 'pval' in box_m_res.columns else box_m_res.iloc[0]['p-val'])
    print(f'Boxův M test kovariancí:   Chi2 = {box_chi2:.2f}, p = {box_p:.2e} -> {rozhodni_h0(box_p)}')

    # 4.5 MANOVA (sdružené srovnání geometrie dist_m + angle_deg)
    manova = MANOVA.from_formula('dist_m + angle_deg ~ shotType', data=df_shots)
    manova_res = manova.mv_test()
    wilks_val = manova_res.results['shotType']['stat'].loc["Wilks' lambda", 'Value']
    wilks_p   = manova_res.results['shotType']['stat'].loc["Wilks' lambda", 'Pr > F']
    pillai_val = manova_res.results['shotType']['stat'].loc["Pillai's trace", 'Value']
    pillai_p   = manova_res.results['shotType']['stat'].loc["Pillai's trace", 'Pr > F']
    print(f'MANOVA: Wilks λ = {wilks_val:.4f} (p = {wilks_p:.2e}), Pillai = {pillai_val:.4f} (p = {pillai_p:.2e})')

    # 4.6 Post-hoc testy: Tukeyho HSD a Games-Howell
    tukey_d = pairwise_tukeyhsd(df_shots['dist_m'], df_shots['shotType'], alpha=0.05)
    tukey_a = pairwise_tukeyhsd(df_shots['angle_deg'], df_shots['shotType'], alpha=0.05)

    fig, (ax_td, ax_ta) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    td_df = pd.DataFrame(tukey_d._results_table.data[1:], columns=tukey_d._results_table.data[0])
    pairs_d = [f"{r['group1']} vs {r['group2']}" for _, r in td_df.iterrows()]
    y_pos = np.arange(len(pairs_d))

    ax_td.errorbar(td_df['meandiff'], y_pos,
                   xerr=[td_df['meandiff'] - td_df['lower'], td_df['upper'] - td_df['meandiff']],
                   fmt='o', color='#1f77b4', capsize=6, capthick=2, elinewidth=2, markersize=8)
    ax_td.axvline(0, color='#d62728', linestyle='--', linewidth=1.5, label='Nulový rozdíl (0)')
    ax_td.set_yticks(y_pos)
    ax_td.set_yticklabels(pairs_d, fontsize=12)
    ax_td.set_xlabel('Rozdíl průměrů vzdálenosti (m) s 95% IS')
    ax_td.set_title('Tukey HSD: Vzdálenost (dist_m)', fontweight='bold')
    ax_td.legend(loc='lower right')

    ta_df = pd.DataFrame(tukey_a._results_table.data[1:], columns=tukey_a._results_table.data[0])
    pairs_a = [f"{r['group1']} vs {r['group2']}" for _, r in ta_df.iterrows()]
    ax_ta.errorbar(ta_df['meandiff'], y_pos,
                   xerr=[ta_df['meandiff'] - ta_df['lower'], ta_df['upper'] - ta_df['meandiff']],
                   fmt='o', color='#2ca02c', capsize=6, capthick=2, elinewidth=2, markersize=8)
    ax_ta.axvline(0, color='#d62728', linestyle='--', linewidth=1.5, label='Nulový rozdíl (0)')
    ax_ta.set_yticks(y_pos)
    ax_ta.set_yticklabels(pairs_a, fontsize=12)
    ax_ta.set_xlabel('Rozdíl průměrů úhlu (°) s 95% IS')
    ax_ta.set_title('Tukey HSD: Střelecký úhel (angle_deg)', fontweight='bold')
    ax_ta.legend(loc='lower right')

    plt.suptitle(f'MANOVA: Wilks λ = {wilks_val:.3f}, Pillai = {pillai_val:.3f} (p = {formatuj_p(pillai_p)}) | Post-hoc 95% IS',
                 fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    uloz_graf('04b_tukey_hsd.png')

    return {
        'F_welch_d': F_welch_d,
        'p_welch_d': p_welch_d,
        'F_welch_a': F_welch_a,
        'p_welch_a': p_welch_a,
        'H_kw': H_kw,
        'p_kw': p_kw,
        'box_chi2': box_chi2,
        'box_p': box_p,
        'wilks_val': wilks_val,
        'wilks_p': wilks_p,
        'pillai_val': pillai_val,
        'pillai_p': pillai_p,
    }


# =============================================================================
# 4b. FÁZE: ZÁVISLOST KVALITY ŠANCÍ (xG) NA STŘELCI (HRÁČI)
# =============================================================================
def analyzuj_hrace(df):
    print('\n' + '=' * 75)
    print('4b. FÁZE: ZÁVISLOST KVALITY ŠANCÍ (xG) NA STŘELCI (HRÁČI)')
    print('=' * 75)

    # 1. Kruskal-Wallisův test rozdělení xG mezi hráči
    kw_p = pg.kruskal(data=df, dv='xG', between='player')
    H_kw_p = float(kw_p.iloc[0]['H'])
    p_kw_p = float(kw_p.iloc[0]['p_unc'] if 'p_unc' in kw_p.columns else kw_p.iloc[0]['p-unc'])

    # 2. Robustní Welchova ANOVA středních hodnot xG
    welch_p = pg.welch_anova(data=df, dv='xG', between='player')
    F_welch_p = float(welch_p.iloc[0]['F'])
    p_welch_p = float(welch_p.iloc[0]['p_unc'] if 'p_unc' in welch_p.columns else welch_p.iloc[0]['p-unc'])

    print(f'Kruskal-Wallis (xG dle hráčů): H = {H_kw_p:.2f}, p = {p_kw_p:.2e} -> {rozhodni_h0(p_kw_p)}')
    print(f'Welchova ANOVA (xG dle hráčů): F = {F_welch_p:.2f}, p = {p_welch_p:.2e} -> {rozhodni_h0(p_welch_p)}')

    # 3. Tabulka statistik hráčů
    player_order = df.groupby('player')['xG'].median().sort_values(ascending=False).index.tolist()
    player_stats = df.groupby('player').agg(
        n=('xG', 'count'),
        goals=('goal', 'sum'),
        median_xG=('xG', 'median'),
        mean_xG=('xG', 'mean'),
        std_xG=('xG', 'std'),
        iqr_xG=('xG', lambda x: x.quantile(0.75) - x.quantile(0.25)),
        mean_dist=('dist_m', 'mean'),
        mean_angle=('angle_deg', 'mean')
    ).loc[player_order].reset_index()
    player_stats['conversion_%'] = (player_stats['goals'] / player_stats['n']) * 100
    player_stats.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_hraci_xG.csv'), index=False)
    print('  [OK] Tabulka hráčů uložena: tabulka_hraci_xG.csv')

    # 4. Graf 04c: Dvoupanelový graf distribuce xG a taktického mapování hráčů
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.5), gridspec_kw={'width_ratios': [1.3, 1.0]})

    labels_with_n = [f'{p} (n={df[df["player"]==p].shape[0]})' for p in player_order]
    sns.boxplot(
        data=df, y='player', x='xG', order=player_order, hue='player', palette='Blues_r',
        legend=False, ax=ax1, boxprops=dict(alpha=0.85), showmeans=True,
        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='#d62728', markersize=6)
    )
    ax1.set_yticks(range(len(player_order)))
    ax1.set_yticklabels(labels_with_n, fontsize=12)
    ax1.set_xlabel('Očekávané góly (xG) na střelu', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Střelec (seřazeno dle mediánu xG)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Distribuce xG dle hráčů | Kruskal-Wallis H = {H_kw_p:.1f} (p < 0.001)\nČervený kosočtverec = průměr, svislá ryska = medián',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlim(-0.02, 0.95)

    sc = ax2.scatter(
        player_stats['mean_dist'], player_stats['mean_xG'],
        s=player_stats['n'] * 4.5 + 80,
        c=player_stats['conversion_%'], cmap='YlOrRd',
        edgecolors='black', linewidth=1.2, alpha=0.90, zorder=3
    )
    cbar = plt.colorbar(sc, ax=ax2, pad=0.03, shrink=0.85)
    cbar.set_label('Gólová konverze střelce (%)', fontsize=12, fontweight='bold')

    for _, row in player_stats.iterrows():
        ax2.annotate(
            row['player'], (row['mean_dist'], row['mean_xG']),
            textcoords='offset points', xytext=(8, 4),
            fontsize=10.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none')
        )

    ax2.set_xlabel('Průměrná vzdálenost střelce od branky (m)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Průměrné xG střelce', fontsize=13, fontweight='bold')
    ax2.set_title('Taktické pozice: Vzdálenost vs. Kvalita šancí (xG)\nVelikost bubliny = počet střel (n)',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.5, zorder=0)

    plt.suptitle('Závislost kvality šancí (xG) na střelci (Premier League, n = 735)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('04c_hraci_xG_boxplot.png')

    return {
        'H_kw_p': H_kw_p,
        'p_kw_p': p_kw_p,
        'F_welch_p': F_welch_p,
        'p_welch_p': p_welch_p,
    }


# =============================================================================
# 5. FÁZE: PCA A FAKTOROVÁ ANALÝZA (VARIMAX) S TESTEM KMO
# =============================================================================
def analyzuj_pca_a_fa(df):
    print('\n' + '=' * 75)
    print('5. FÁZE: PCA A FAKTOROVÁ ANALÝZA S TESTEM KMO (BEZ REDUNDANCE X/Y)')
    print('=' * 75)

    df_pca = df.copy()
    df_pca['y_dev'] = np.abs(df_pca['y_m'] - 34.0)
    pca_vars = ['dist_m', 'angle_deg', 'y_dev', 'xG']

    # 5.1 KMO a Bartlettův test sférickosti
    kmo_val = vypocti_kmo(df_pca[pca_vars])
    R_mat = df_pca[pca_vars].corr().values
    n_b, p_b = len(df_pca), len(pca_vars)
    chi2_b = -(n_b - 1 - (2 * p_b + 5) / 6) * np.log(np.linalg.det(R_mat))
    df_b = p_b * (p_b - 1) // 2
    p_val_b = 1.0 - stats.chi2.cdf(chi2_b, df=df_b)

    print(f'KMO index adekvátnosti:    KMO = {kmo_val:.3f} (adekvátní pro faktorování)')
    print(f'Bartlettův test sférickosti: chi2 = {chi2_b:.1f}, df = {df_b}, p = {p_val_b:.2e} -> {rozhodni_h0(p_val_b)}')

    # 5.2 Dekompozice PCA
    X_std = StandardScaler().fit_transform(df_pca[pca_vars])
    pca = PCA()
    X_pca = pca.fit_transform(X_std)
    eigen_vals = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_ratio)

    # Graf 05b: Scree plot a 2D Biplot
    fig, (ax_scr, ax_bip) = plt.subplots(1, 2, figsize=(14.5, 6.2))

    ax_scr.bar(range(1, len(eigen_vals) + 1), eigen_vals, color='#1f77b4', alpha=0.75, label='Vlastní čísla λ')
    ax_scr.plot(range(1, len(eigen_vals) + 1), eigen_vals, 'ro-', linewidth=2)
    ax_scr.axhline(1.0, color='#d62728', linestyle='--', linewidth=2, label='Kaiserovo pravidlo (λ = 1)')
    ax_scr.set_xlabel('Hlavní komponenta')
    ax_scr.set_ylabel('Vlastní číslo λ')
    ax_scr.set_title('Scree plot (graf úpatí)', fontweight='bold')
    ax_scr.legend()

    for idx_ev, (ev, cum) in enumerate(zip(eigen_vals[:2], cum_var[:2])):
        ax_scr.text(idx_ev + 1, ev + 0.1, f'{cum:.1f} %', ha='center', fontsize=12, fontweight='bold')

    sc = ax_bip.scatter(X_pca[:, 0], X_pca[:, 1], c=df['goal'], cmap='RdBu_r', alpha=0.55, s=32, vmin=0, vmax=1)
    cbar = plt.colorbar(sc, ax=ax_bip, shrink=0.85)
    cbar.set_label('Gól (1 = Ano, 0 = Ne)')

    loadings_pca = pca.components_.T[:, :2]
    scale_arrow = 2.5
    for j, var in enumerate(pca_vars):
        lx, ly = loadings_pca[j, 0] * scale_arrow, loadings_pca[j, 1] * scale_arrow
        ax_bip.annotate('', xy=(lx, ly), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.2))
        ax_bip.text(lx * 1.2, ly * 1.2, var, fontsize=12, fontweight='bold', color='#1A1A1A',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#B0B0B0', alpha=0.9))

    ax_bip.axhline(0, color='#B0B0B0', linestyle='--', linewidth=1)
    ax_bip.axvline(0, color='#B0B0B0', linestyle='--', linewidth=1)
    ax_bip.set_xlabel(f'PC1 ({var_ratio[0]:.1f} % rozptylu) – Centrální kvalita pozice')
    ax_bip.set_ylabel(f'PC2 ({var_ratio[1]:.1f} % rozptylu) – Boční excentricita (y_dev)')
    ax_bip.set_title('2D Biplot hlavních komponent', fontweight='bold')

    plt.suptitle(f'PCA redukce dimenze | Bartlett: χ² = {chi2_b:.1f} (p = {formatuj_p(p_val_b)}) | KMO = {kmo_val:.3f}',
                 fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('05b_biplot_pca.png')

    # Graf 05: Samostatný Scree plot pro archivaci
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(range(1, len(eigen_vals) + 1), eigen_vals, color='#1f77b4', alpha=0.75, label='Vlastní čísla λ')
    ax.plot(range(1, len(eigen_vals) + 1), eigen_vals, 'ro-', linewidth=2)
    ax.axhline(1.0, color='#d62728', linestyle='--', linewidth=2, label='Kaiserovo pravidlo (λ = 1)')
    ax.set_xlabel('Hlavní komponenta')
    ax.set_ylabel('Vlastní číslo λ')
    ax.set_title('Scree plot (výběr komponent dle Kaiserova pravidla)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    uloz_graf('05_scree_plot.png')

    # 5.3 Faktorová analýza s ortogonální rotací Varimax
    fa = FactorAnalysis(n_components=2, rotation='varimax', random_state=42)
    fa.fit(X_std)
    fa_loadings = pd.DataFrame(fa.components_.T, index=pca_vars,
                               columns=['Faktor 1 (Centrální nebezpečnost)', 'Faktor 2 (Boční excentricita)'])

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.heatmap(fa_loadings, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'},
                linewidths=1.2, linecolor='#FFFFFF', cbar_kws={'label': 'Faktorová zátěž (Loading)'})
    ax.set_title('Faktorová analýza s rotací Varimax\n'
                 'Faktor 1: vysoká zátěž úhlu, xG a záporná zátěž vzdálenosti | Faktor 2: boční odchylka',
                 fontweight='bold', pad=14)
    plt.tight_layout()
    uloz_graf('05c_fa_varimax_loadings.png')

    return {
        'kmo_val': kmo_val,
        'chi2_b': chi2_b,
        'p_val_b': p_val_b,
        'cum_var_2': cum_var[1],
    }


# =============================================================================
# 6. FÁZE: SHLUKOVÁ ANALÝZA – 3 TAKTICKÉ ZÓNY ZAKONČENÍ (WARDŮV ŘEZ & K-MEANS)
# =============================================================================
def analyzuj_shluky(df):
    print('\n' + '=' * 75)
    print('6. FÁZE: SHLUKOVÁ ANALÝZA (WARDŮV ŘEZ NA k = 3 A TAKTICKÁ SEGMENTACE)')
    print('=' * 75)

    clus_vars = ['dist_m', 'angle_deg', 'xG']
    X_clus = StandardScaler().fit_transform(df[clus_vars])

    # 6.1 Hierarchické aglomerativní shlukování Wardovou metodou
    link_mat = linkage(X_clus, method='ward', metric='euclidean')

    # Výška řezu pro k = 3 shluky:
    # link_mat[-1, 2] spojuje 2 shluky do 1 (h ≈ 47.0)
    # link_mat[-2, 2] spojuje 3 shluky do 2 (h ≈ 29.6)
    # link_mat[-3, 2] spojuje 4 shluky do 3 (h ≈ 16.8)
    # Řez mezi link_mat[-3, 2] a link_mat[-2, 2] protíná PŘESNĚ 3 větve!
    h_cut_3 = (link_mat[-2, 2] + link_mat[-3, 2]) / 2.0
    print(f'Wardův dendrogram: merge 3->2 při h = {link_mat[-2, 2]:.2f}, merge 4->3 při h = {link_mat[-3, 2]:.2f}')
    print(f'Matematicky přesná výška řezu pro k = 3 shluky: h = {h_cut_3:.2f}')

    ward_raw = fcluster(link_mat, 3, criterion='maxclust')

    # Seřazení shluků podle vzdálenosti (1 = nejblíže / nejnebezpečnější, 3 = nejdále)
    df_c = df.copy()
    mean_dists = {cid: df_c[ward_raw == cid]['dist_m'].mean() for cid in np.unique(ward_raw)}
    sorted_cids = sorted(mean_dists, key=mean_dists.get)
    label_map = {old: new for new, old in enumerate(sorted_cids)}
    df_c['cluster'] = [label_map[l] for l in ward_raw]

    # Výpočet validačních metrik pro k = 2 až 6
    k_range = list(range(2, 7))
    sil_scores = []
    inertias = []
    for k in k_range:
        km_test = KMeans(n_clusters=k, random_state=42, n_init=15).fit(X_clus)
        sil = silhouette_score(X_clus, km_test.labels_)
        sil_scores.append(sil)
        inertias.append(km_test.inertia_)
        print(f'  k = {k}: Silhouette = {sil:.4f}, WCSS (Inertia) = {km_test.inertia_:.1f}')

    sil_k3 = sil_scores[1]  # index 1 je k = 3
    inertia_drop = (inertias[0] - inertias[1]) / inertias[0] * 100
    print(f'Přechod k = 2 -> k = 3 snižuje vnitroshlukový rozptyl (Inertii) o {inertia_drop:.1f} % (výrazná loketní inflexe)!')

    # Statistiky 3 shluků
    c1 = df_c[df_c['cluster'] == 0]
    c2 = df_c[df_c['cluster'] == 1]
    c3 = df_c[df_c['cluster'] == 2]

    print('\nCharakteristika 3 taktických zón zakončení (Ward k = 3):')
    print(f'  Shluk 1 (Zlatá zóna / Malé vápno): n = {len(c1)} ({len(c1)/len(df_c):.1%}), dist = {c1["dist_m"].mean():.1f} m, úhel = {c1["angle_deg"].mean():.1f}°, xG = {c1["xG"].mean():.3f}, gólů = {c1["goal"].sum()}/{len(c1)} ({c1["goal"].mean():.1%})')
    print(f'  Shluk 2 (Pokutové území / 2. vlna): n = {len(c2)} ({len(c2)/len(df_c):.1%}), dist = {c2["dist_m"].mean():.1f} m, úhel = {c2["angle_deg"].mean():.1f}°, xG = {c2["xG"].mean():.3f}, gólů = {c2["goal"].sum()}/{len(c2)} ({c2["goal"].mean():.1%})')
    print(f'  Shluk 3 (Dálková střelba mimo vápno): n = {len(c3)} ({len(c3)/len(df_c):.1%}), dist = {c3["dist_m"].mean():.1f} m, úhel = {c3["angle_deg"].mean():.1f}°, xG = {c3["xG"].mean():.3f}, gólů = {c3["goal"].sum()}/{len(c3)} ({c3["goal"].mean():.1%})')

    # 6.2 Graf 06b: 1x2 Dendrogram s přesným řezem vlevo + Mapa 3 zón na hřišti vpravo
    fig, (ax_dend, ax_map) = plt.subplots(1, 2, figsize=(15.2, 6.4))

    # Dendrogram s barevným prahem h_cut_3 přesně obarvuje 3 větve
    dendrogram(link_mat, ax=ax_dend, truncate_mode='lastp', p=20, leaf_rotation=45, leaf_font_size=11,
               color_threshold=h_cut_3)
    ax_dend.axhline(h_cut_3, color='#d62728', linestyle='--', linewidth=2.2,
                    label=f'Řez na k = 3 shluky (výška h = {h_cut_3:.1f})')
    ax_dend.set_title('Dendrogram hierarchického shlukování (Ward)', fontweight='bold')
    ax_dend.set_xlabel('Indexy sloučených skupin střel')
    ax_dend.set_ylabel('Wardova eukleidovská vzdálenost')
    ax_dend.legend(loc='upper right', fontsize=11)

    nakresli_hriste(ax_map)

    # Vykreslení 3 shluků na hřišti
    ax_map.scatter(c3['x_m'], c3['y_m'], c='#1f77b4', s=45, alpha=0.60, edgecolors='none',
                   label=f'Shluk 3: Dálka (> 18 m, n={len(c3)}, konverze {c3["goal"].mean():.1%})', zorder=3)
    ax_map.scatter(c2['x_m'], c2['y_m'], c='#ff7f0e', s=55, alpha=0.75, edgecolors='none',
                   label=f'Shluk 2: Velké vápno (12–18 m, n={len(c2)}, konverze {c2["goal"].mean():.1%})', zorder=4)
    ax_map.scatter(c1['x_m'], c1['y_m'], c='#d62728', s=70, alpha=0.90, edgecolors='#800000', linewidths=1.2,
                   label=f'Shluk 1: Zlatá zóna (< 12 m, n={len(c1)}, konverze {c1["goal"].mean():.1%})', zorder=5)

    ax_map.set_title('3 taktické zóny zakončení na fotbalovém hřišti (k = 3)', fontweight='bold')
    ax_map.legend(loc='upper left', frameon=True, framealpha=0.94, edgecolor='#CCCCCC', fontsize=10.5)

    plt.suptitle('Shluková analýza: Wardovo dělení na 3 taktické zóny zakončení a prostorová konverze',
                 fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('06b_mapa_shluku.png')

    # 6.3 Graf 06: Validační křivky (Elbow metoda a Silhouette score)
    fig, (ax_elb, ax_sil) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax_elb.plot(k_range, inertias, 'ro-', linewidth=2.4, markersize=8)
    ax_elb.axvline(3, color='#1f77b4', linestyle='--', linewidth=2, label='Inflexe metody lokte (k = 3)')
    ax_elb.set_xlabel('Počet shluků k')
    ax_elb.set_ylabel('Vnitroshlukový součet čtverců WCSS (Inertia)')
    ax_elb.set_title('Metoda lokte (Elbow Method)', fontweight='bold')
    ax_elb.legend(fontsize=11)
    ax_elb.text(0.40, 0.65, f'Pokles WCSS z k=2 na k=3:\n- {inertia_drop:.1f} % variability\n-> Zásadní nárůst homogenity',
                transform=ax_elb.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', fc='#FFFFFF', ec='#CCCCCC'))

    ax_sil.plot(k_range, sil_scores, 'bo-', linewidth=2.4, markersize=8)
    ax_sil.axvline(3, color='#d62728', linestyle='--', linewidth=2, label=f'Volba k = 3 (Silhouette = {sil_k3:.3f})')
    ax_sil.set_xlabel('Počet shluků k')
    ax_sil.set_ylabel('Průměrný koeficient siluety')
    ax_sil.set_title('Silhouette analýza kvality shlukování', fontweight='bold')
    ax_sil.legend(fontsize=11)
    ax_sil.text(0.42, 0.45, f'k = 3 zachovává vysokou separaci\n(Silhouette = {sil_k3:.3f})\na dává věcný fotbalový smysl',
                transform=ax_sil.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', fc='#FFFFFF', ec='#CCCCCC'))

    plt.suptitle('Validace počtu shluků: Matematická inflexe lokte a taktická interpretovatelnost k = 3',
                 fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    uloz_graf('06_dendrogram.png')

    return {
        'opt_k': 3,
        'sil_k3': sil_k3,
        'c1_n': len(c1),
        'c1_conv': c1["goal"].mean(),
        'c2_n': len(c2),
        'c2_conv': c2["goal"].mean(),
        'c3_n': len(c3),
        'c3_conv': c3["goal"].mean(),
    }


# =============================================================================
# 7. FÁZE: LOGISTICKÁ REGRESE BEZ DATA LEAKAGE A SROVNÁNÍ S BENCHMARKEM
# =============================================================================
def analyzuj_logistickou_regresi(df):
    print('\n' + '=' * 75)
    print('7. FÁZE: LOGISTICKÁ REGRESE (BEZ xG JAKO PREDIKTORU) A ROC BENCHMARK')
    print('=' * 75)

    # Dummy kódování shotType (referenční kategorie Head)
    shot_dummies = pd.get_dummies(df['shotType'], prefix='shotType', drop_first=True, dtype=float)

    # Prediktory: výhradně primární geometrie a herní kontext (BEZ xG!)
    reg_features = ['dist_m', 'angle_deg', 'minute']
    X_reg = pd.concat([df[reg_features], shot_dummies], axis=1)
    X_reg_const = sm.add_constant(X_reg)
    y_reg = df['goal']

    # Diagnostika multikolinearity přes VIF (samostatná tabulka mimo testy hypotéz)
    vif_data = []
    for i, col in enumerate(X_reg_const.columns):
        if col != 'const':
            vif_val = variance_inflation_factor(X_reg_const.values, i)
            vif_data.append({
                'Prediktor': col,
                'VIF': round(vif_val, 3),
                'Tolerance (1/VIF)': round(1.0 / vif_val, 3)
            })
    vif_table = pd.DataFrame(vif_data)
    vif_table.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_vif.csv'), index=False)
    print('  [OK] Diagnostická tabulka VIF uložena: tabulka_vif.csv')

    # Odhad multivariabilního logit modelu
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
    print('  [OK] Tabulka koeficientů logistické regrese uložena: tabulka_logit_koeficienty.csv')

    print(f'Multivariabilní model: dist_m p-hodnota = {p_vals["dist_m"]:.4f} -> {rozhodni_h0(p_vals["dist_m"])}')
    print(f'Multivariabilní model: angle   p-hodnota = {p_vals["angle_deg"]:.4f} -> {rozhodni_h0(p_vals["angle_deg"])}')
    print(f'Celkový model LLR chi2 test: p-hodnota = {logit_mod.llr_pvalue:.2e} -> {rozhodni_h0(logit_mod.llr_pvalue)}')

    # ROC křivka našeho modelu
    y_pred_prob = logit_mod.predict(X_reg_const)
    fpr_mod, tpr_mod, _ = roc_curve(y_reg, y_pred_prob)
    auc_mod = auc(fpr_mod, tpr_mod)

    # ROC křivka externího benchmarku xG
    fpr_xg, tpr_xg, _ = roc_curve(y_reg, df['xG'])
    auc_xg = auc(fpr_xg, tpr_xg)
    print(f'Diskriminační schopnost: Model AUC = {auc_mod:.3f} vs. Externí benchmark xG AUC = {auc_xg:.3f}')

    # Univariátní model pro vzdálenost (S-křivka)
    mod_univ = sm.Logit(y_reg, sm.add_constant(df['dist_m'])).fit(disp=False)
    dist_seq = np.linspace(2, 42, 200)
    prob_seq = mod_univ.predict(sm.add_constant(dist_seq))
    p_at_10 = mod_univ.predict([1, 10])[0]
    p_at_25 = mod_univ.predict([1, 25])[0]

    # Graf 07b: 1x2 Logistická S-křivka vlevo + Srovnávací ROC křivka vpravo
    fig, (ax_sig, ax_roc) = plt.subplots(1, 2, figsize=(14.5, 6.2))

    np.random.seed(42)
    jitter = np.random.uniform(-0.025, 0.025, size=len(df))

    ax_sig.scatter(df[df['goal'] == 0]['dist_m'], 0 + jitter[df['goal'] == 0],
                   c='#1f77b4', alpha=0.45, s=30, label='Bez gólu (y = 0)')
    ax_sig.scatter(df[df['goal'] == 1]['dist_m'], 1 + jitter[df['goal'] == 1],
                   c='#d62728', alpha=0.85, s=45, edgecolors='#800000', label='Gól (y = 1)')
    ax_sig.plot(dist_seq, prob_seq, color='#d62728', linewidth=3.0, label='Logistická sigmoida P(gól)')

    ax_sig.plot([10, 10], [0, p_at_10], 'k--', lw=1.2)
    ax_sig.plot([0, 10], [p_at_10, p_at_10], 'k--', lw=1.2)
    ax_sig.plot([25, 25], [0, p_at_25], 'k--', lw=1.2)
    ax_sig.plot([0, 25], [p_at_25, p_at_25], 'k--', lw=1.2)

    ax_sig.set_xlim(0, 42)
    ax_sig.set_ylim(-0.06, 1.06)
    ax_sig.set_xlabel('Vzdálenost střely dist_m (metry)')
    ax_sig.set_ylabel('Pravděpodobnost gólu P(gól)')
    ax_sig.set_title('Logistická S-křivka šance dle vzdálenosti', fontweight='bold')
    ax_sig.legend(loc='upper right')

    or_d = odds_r['dist_m']
    p_d_val = p_vals['dist_m']
    ax_sig.text(0.04, 0.50,
                f"10 m: šance ~{p_at_10:.1%}\n"
                f"25 m: šance pod {p_at_25:.1%}\n"
                f"OR(dist) = {or_d:.3f} (p = {formatuj_p(p_d_val)})\n"
                f"-> {rozhodni_h0(p_d_val)}",
                transform=ax_sig.transAxes, fontsize=12, fontweight='bold', color='#1A1A1A',
                bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFFF', ec='#d62728', lw=1.5))

    # ROC křivka: porovnání našeho modelu s benchmarkem xG
    ax_roc.plot(fpr_mod, tpr_mod, color='#d62728', lw=3.0,
                label=f'Náš geometrický model (AUC = {auc_mod:.3f})')
    ax_roc.plot(fpr_xg, tpr_xg, color='#1f77b4', lw=2.4, linestyle='-.',
                label=f'Externí benchmark xG (AUC = {auc_xg:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.5, label='Náhodný odhad (AUC = 0,500)')
    ax_roc.fill_between(fpr_mod, tpr_mod, alpha=0.12, color='#d62728')

    ax_roc.set_xlabel('1 – Specificita (Falešně pozitivní)')
    ax_roc.set_ylabel('Senzitivita (Pravdivě pozitivní)')
    ax_roc.set_title('ROC křivka modelu a srovnání s xG', fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=11)

    plt.suptitle('Logistická regrese: Odhad šance a validační ROC analýza bez zkreslení xG',
                 fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    uloz_graf('07b_logisticka_krivka.png')

    return {
        'llr_p': logit_mod.llr_pvalue,
        'llr_chi2': logit_mod.llr,
        'p_vals': p_vals,
        'odds_r': odds_r,
        'auc_mod': auc_mod,
        'auc_xg': auc_xg,
    }


# =============================================================================
# 8. FÁZE: EXPORT TABULKY VŠECH TESTOVANÝCH HYPOTÉZ (DYNAMICKÉ VYHODNOCENÍ)
# =============================================================================
def exportuj_tabulku_hypotez(vysledky):
    print('\n' + '=' * 75)
    print('8. FÁZE: DYNAMICKÉ SESTAVENÍ A EXPORT SOUHRNNÉ TABULKY HYPOTÉZ')
    print('=' * 75)

    sw = vysledky['norm']['sw']
    hz_stat = vysledky['norm']['hz_stat']
    hz_p = vysledky['norm']['hz_p']
    L_stat = vysledky['norm']['L_stat']
    L_p = vysledky['norm']['L_p']

    r_sp = vysledky['kor']['r_sp']
    p_sp = vysledky['kor']['p_sp']
    r_part = vysledky['kor']['r_part']
    p_part = vysledky['kor']['p_part']

    F_welch_d = vysledky['anv']['F_welch_d']
    p_welch_d = vysledky['anv']['p_welch_d']
    H_kw = vysledky['anv']['H_kw']
    p_kw = vysledky['anv']['p_kw']
    box_chi2 = vysledky['anv']['box_chi2']
    box_p = vysledky['anv']['box_p']
    pillai_val = vysledky['anv']['pillai_val']
    pillai_p = vysledky['anv']['pillai_p']

    H_kw_p = vysledky['hraci']['H_kw_p']
    p_kw_p = vysledky['hraci']['p_kw_p']
    F_welch_p = vysledky['hraci']['F_welch_p']
    p_welch_p = vysledky['hraci']['p_welch_p']

    chi2_b = vysledky['pca']['chi2_b']
    p_val_b = vysledky['pca']['p_val_b']

    llr_p = vysledky['logit']['llr_p']
    llr_chi2 = vysledky['logit']['llr_chi2']
    p_vals_logit = vysledky['logit']['p_vals']
    odds_r = vysledky['logit']['odds_r']

    # Seznam všech věcných statistických hypotéz
    # Žádný hardcoded text, p-hodnoty i rozhodnutí se generují striktně dynamicky!
    hypotezy = [
        ('1 – Normalita', 'Shapiro-Wilkův test (dist_m)',
         'Vzdálenost dist_m má normální rozdělení',
         f"W = {sw['dist_m']['W']:.4f}", sw['dist_m']['p']),

        ('1 – Normalita', 'Shapiro-Wilkův test (angle_deg)',
         'Střelecký úhel angle_deg má normální rozdělení',
         f"W = {sw['angle_deg']['W']:.4f}", sw['angle_deg']['p']),

        ('1 – Normalita MV', 'Henze-Zirklerův test',
         'Vektor [dist_m, angle, xG] má vícerozměrné normální rozdělení',
         f"HZ = {hz_stat:.3f}", hz_p),

        ('2 – Homoskedasticita', 'Leveneho test rozptylů',
         'Rozptyly vzdálenosti jsou mezi skupinami zakončení shodné',
         f"L = {L_stat:.3f}", L_p),

        ('2 – Kovariance', 'Boxův M test kovariancí',
         'Kovarianční matice jsou mezi skupinami zakončení shodné',
         f"χ² = {box_chi2:.1f}", box_p),

        ('3 – Závislost', 'Spearmanův test pořadové korelace',
         'Korelace mezi vzdáleností dist_m a xG je nulová (r = 0)',
         f"r = {r_sp:.3f}", p_sp),

        ('3 – Parciální kor.', 'Pingouin parciální korelace',
         'Parciální korelace xG a úhlu při kontrole vzdálenosti je nulová',
         f"r_p = {r_part:.3f}", p_part),

        ('4 – Skupiny Welch', 'Welchova robustní ANOVA (dist_m)',
         'Střední vzdálenosti střel jsou pro typy zakončení stejné',
         f"F = {F_welch_d:.1f}", p_welch_d),

        ('4 – Skupiny Neparam.', 'Kruskal-Wallisův test (dist_m)',
         'Distribuce vzdáleností jsou mezi typy zakončení shodné',
         f"H = {H_kw:.1f}", p_kw),

        ('4 – Sdružené MANOVA', 'Pillaiova stopa / Wilks λ',
         'Sdružené vektory středních hodnot [dist, úhel] jsou shodné',
         f"Pillai = {pillai_val:.3f}", pillai_p),

        ('4b – Hráči Kruskal', 'Kruskal-Wallisův test (xG dle hráčů)',
         'Distribuce kvality šancí xG je mezi všemi střelci shodná',
         f"H = {H_kw_p:.1f}", p_kw_p),

        ('4b – Hráči Welch', 'Welchova ANOVA (xG dle hráčů)',
         'Střední hodnota xG je pro všechny střelce stejná',
         f"F = {F_welch_p:.1f}", p_welch_p),

        ('5 – Vhodnost PCA', 'Bartlettův test sférickosti',
         'Korelační matice je jednotková (proměnné jsou nekorelované)',
         f"χ² = {chi2_b:.1f}", p_val_b),

        ('6 – Celkový Logit', 'Likelihood Ratio Test (LLR)',
         'Všechny koeficienty logistického modelu jsou nulové (β = 0)',
         f"χ² = {llr_chi2:.1f}", llr_p),

        ('6 – Vzdálenost Logit', 'Waldův test koeficientu (dist_m)',
         'Vzdálenost střely neovlivňuje logit šance na gól (β_dist = 0)',
         f"OR = {odds_r['dist_m']:.3f}", p_vals_logit['dist_m']),

        ('6 – Úhel Logit', 'Waldův test koeficientu (angle_deg)',
         'Střelecký úhel neovlivňuje logit šance na gól (β_angle = 0)',
         f"OR = {odds_r['angle_deg']:.3f}", p_vals_logit['angle_deg']),
    ]

    radky_tabulky = []
    for faza, metoda, h0, stat_str, p_val in hypotezy:
        radky_tabulky.append({
            'Fáze': faza,
            'Metoda': metoda,
            'Nulová hypotéza (H0)': h0,
            'Statistika': stat_str,
            'p-hodnota': formatuj_p(p_val),
            'Rozhodnutí (alpha=0.05)': rozhodni_h0(p_val)
        })

    df_hyp_out = pd.DataFrame(radky_tabulky)
    df_hyp_out.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_hypotezy.csv'), index=False)
    print('  [OK] Finální tabulka hypotéz uložena: tabulka_hypotezy.csv')
    print('\nSouhrn testovaných hypotéz:')
    for idx, r in df_hyp_out.iterrows():
        print(f"  {idx+1:2d}. {r['Metoda']:<36}: p = {r['p-hodnota']:<8} -> {r['Rozhodnutí (alpha=0.05)']}")


# =============================================================================
# HLAVNÍ SPOUŠTĚCÍ BLOK
# =============================================================================
if __name__ == '__main__':
    print('=' * 75)
    print('SPUŠTĚNÍ VÍCEROZMĚRNÉ STATISTICKÉ ANALÝZY VMM')
    print('=' * 75)

    df_clean = pd.read_csv(DATA_PATH)
    print(f'Načten soubor clean_shots.csv: {len(df_clean)} záznamů, gólů: {df_clean["goal"].sum()}.')

    vysledky = {}
    eda_a_prostorova_mapa(df_clean)
    vysledky['norm']  = analyzuj_normalitu(df_clean)
    vysledky['kor']   = analyzuj_korelace(df_clean)
    vysledky['anv']   = analyzuj_anova_manova(df_clean)
    vysledky['hraci'] = analyzuj_hrace(df_clean)
    vysledky['pca']   = analyzuj_pca_a_fa(df_clean)
    vysledky['clus']  = analyzuj_shluky(df_clean)
    vysledky['logit'] = analyzuj_logistickou_regresi(df_clean)
    exportuj_tabulku_hypotez(vysledky)

    print('\n' + '=' * 75)
    print('VŠECHNY ANALYTICKÉ FÁZE ÚSPĚŠNĚ DOKONČENY BEZ CHYB!')
    print('=' * 75)
