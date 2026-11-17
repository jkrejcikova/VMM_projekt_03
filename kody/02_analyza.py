# -*- coding: utf-8 -*-
"""
Semestrální projekt VMM – Střelecká úspěšnost (střely z fotbalových lig)
Skript 2: Vícerozměrná statistická analýza a vizualizace
"""

import os
import warnings
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

# Nastavení cest a grafiky
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'clean_shots.csv')
VYSTUPY_DIR = os.path.join(BASE_DIR, 'vystupy')
os.makedirs(VYSTUPY_DIR, exist_ok=True)

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


# Pomocné funkce
def uloz_graf(nazev_souboru):
    cesta = os.path.join(VYSTUPY_DIR, nazev_souboru)
    plt.savefig(cesta, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Uloženo: {nazev_souboru}')


def nakresli_hriste(ax):
    pitch = Pitch(
        pitch_type='custom', pitch_color='grass', line_color='white',
        stripe_color='#c2d59d', stripe=True, pitch_length=105,
        pitch_width=68, linewidth=1.5, goal_type='line'
    )
    pitch.draw(ax=ax)
    return pitch


def formatuj_p(p_val):
    return '< 0.001' if p_val < 0.001 else f'{p_val:.4f}'


def rozhodni_h0(p_val, alpha=0.05):
    return 'ZAMÍTÁME H0' if p_val < alpha else 'NEZAMÍTÁME H0'


def vypocti_kmo(data_df):
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


# 1. Průzkumná analýza (EDA) a prostorové mapy
def eda_a_prostorova_mapa(df):
    print('\nZpracovávám EDA a mapy střel...')

    cols_desc = ['dist_m', 'angle_deg', 'xG', 'minute']
    popis_df = df[cols_desc].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    popis_df['IQR'] = popis_df['75%'] - popis_df['25%']
    popis_df['Skew'] = df[cols_desc].skew()
    popis_df['Kurtosis'] = df[cols_desc].kurt()
    popis_df.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_popis.csv'))

    no_goals = df[df['goal'] == 0]
    goals = df[df['goal'] == 1]

    # Prostorová mapa střel
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    nakresli_hriste(ax)
    ax.scatter(no_goals['x_m'], no_goals['y_m'], c='#1f77b4', s=45, alpha=0.60,
               edgecolors='none', label=f'Nezakončeno gólem (n = {len(no_goals)})', zorder=3)
    ax.scatter(goals['x_m'], goals['y_m'], c='#d62728', s=90, alpha=0.90,
               edgecolors='#800000', linewidths=1.2, label=f'Gól (n = {len(goals)})', zorder=4)
    ax.set_title(f'Prostorová mapa střel (n = {len(df)}, gólů: {len(goals)})', fontsize=15,
                 fontweight='bold', pad=12)
    ax.legend(loc='upper left', frameon=True, framealpha=0.92, facecolor='white', edgecolor='#CCCCCC')
    uloz_graf('01_mapa_strel.png')

    # Histogram xG
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(df['xG'], bins=30, kde=True, color='#1f77b4', ax=ax, edgecolor='white', alpha=0.75)
    ax.axvline(df['xG'].mean(), color='#d62728', linestyle='--', linewidth=2,
               label=f'Průměr xG = {df["xG"].mean():.3f}')
    ax.axvline(df['xG'].median(), color='#2ca02c', linestyle='-', linewidth=2,
               label=f'Medián xG = {df["xG"].median():.3f}')
    ax.set_xlabel('Očekávané góly (xG)')
    ax.set_ylabel('Počet střel')
    ax.set_title('Distribuce metriky očekávaných gólů (xG)', fontsize=15, fontweight='bold')
    ax.legend()
    uloz_graf('01b_hist_xG.png')

    # 2D KDE hustota střelby
    pitch_half = VerticalPitch(
        pitch_type='custom', pitch_color='grass', line_color='white',
        stripe_color='#c2d59d', stripe=True, pitch_length=105,
        pitch_width=68, half=True, linewidth=1.5, goal_type='line'
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8))
    pitch_half.draw(ax=axes[0])
    pitch_half.draw(ax=axes[1])

    cmap_no_goals = LinearSegmentedColormap.from_list('c_no_goals', [(0.40, 0.70, 0.90, 0.35), (0.15, 0.45, 0.75, 0.65),
                                                                     (0.04, 0.18, 0.50, 0.92)], N=256)
    cmap_goals = LinearSegmentedColormap.from_list('c_goals', [(1.0, 0.88, 0.20, 0.45), (1.0, 0.50, 0.05, 0.70),
                                                               (0.85, 0.08, 0.08, 0.92)], N=256)

    pitch_half.kdeplot(no_goals['x_m'], no_goals['y_m'], ax=axes[0], cmap=cmap_no_goals, fill=True, levels=14,
                       thresh=0.03, zorder=2)
    axes[0].set_title(
        f'Neproměněné střely (n = {len(no_goals)})\nMedián vzdálenosti: {no_goals["dist_m"].median():.1f} m',
        fontsize=13.5, fontweight='bold', pad=10)

    pitch_half.kdeplot(goals['x_m'], goals['y_m'], ax=axes[1], cmap=cmap_goals, fill=True, levels=14, thresh=0.03,
                       zorder=2)
    axes[1].set_title(f'Vstřelené góly (n = {len(goals)})\nMedián vzdálenosti: {goals["dist_m"].median():.1f} m',
                      fontsize=13.5, fontweight='bold', pad=10)

    fig.suptitle('Srovnání 2D prostorové hustoty střelby (KDE)', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    uloz_graf('01c_kde_hriste.png')


# 2. Diagnostika normality a rozptylů
def analyzuj_normalitu(df):
    print('\nTestování normality a rozptylů...')

    sw_vysledky = {}
    for col in ['dist_m', 'xG', 'angle_deg']:
        data_col = df[col].dropna()
        W_val, p_val = stats.shapiro(data_col)
        sw_vysledky[col] = {'W': W_val, 'p': p_val, 'verdikt': rozhodni_h0(p_val)}

    skupiny_st = [grp['dist_m'].values for _, grp in df.groupby('shotType') if len(grp) > 10]
    L_stat, L_p = stats.levene(*skupiny_st, center='median')

    # Histogram a Q-Q probplot dist_m
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    dist_data = df['dist_m'].dropna()
    mu_d, std_d = dist_data.mean(), dist_data.std()

    ax_h.hist(dist_data, bins=28, density=True, color='#1f77b4', edgecolor='white', alpha=0.75, label='Empirická data')
    x_axis = np.linspace(dist_data.min(), dist_data.max(), 300)
    ax_h.plot(x_axis, stats.norm.pdf(x_axis, mu_d, std_d), 'r-', linewidth=2.5, label='Normální křivka')
    ax_h.set_xlabel('Vzdálenost od branky (m)')
    ax_h.set_ylabel('Hustota pravděpodobnosti')
    ax_h.set_title('Histogram vzdálenosti střel')
    ax_h.legend(loc='upper right')

    w_d, p_d = sw_vysledky['dist_m']['W'], sw_vysledky['dist_m']['p']
    ax_h.text(0.04, 0.94, f'Shapiro-Wilk: p = {formatuj_p(p_d)}', transform=ax_h.transAxes, fontsize=12,
              fontweight='bold', bbox=dict(boxstyle='round', fc='#FDF2F2', ec='#F5C6CB'))

    stats.probplot(dist_data, dist='norm', plot=ax_q)
    ax_q.get_lines()[0].set_markerfacecolor('#1f77b4')
    ax_q.get_lines()[0].set_markeredgecolor('none')
    ax_q.get_lines()[0].set_alpha(0.65)
    ax_q.get_lines()[1].set_color('#d62728')
    ax_q.get_lines()[1].set_linewidth(2.2)
    ax_q.set_title('Normální Q-Q graf (dist_m)')
    ax_q.set_xlabel('Teoretické kvantily')
    ax_q.set_ylabel('Empirické kvantily')

    ax_q.text(0.04, 0.94, f'Leveneho test: p = {formatuj_p(L_p)}', transform=ax_q.transAxes, fontsize=12,
              fontweight='bold', bbox=dict(boxstyle='round', fc='#FDF2F2', ec='#F5C6CB'))
    plt.tight_layout()
    uloz_graf('02_normalita_histogramy_qq.png')

    # Vícerozměrná normalita
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

    fig, (ax_qq_mv, ax_hist_mv) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    ax_qq_mv.scatter(chi2_quantiles, d2_sorted, color='#1f77b4', s=30, alpha=0.70, edgecolors='none')
    max_val = max(chi2_quantiles.max(), d2_sorted.max()) * 1.05
    ax_qq_mv.plot([0, max_val], [0, max_val], 'r--', linewidth=2.4)
    ax_qq_mv.set_xlim(0, chi2_quantiles.max() * 1.05)
    ax_qq_mv.set_ylim(0, d2_sorted.max() * 1.05)
    ax_qq_mv.set_xlabel(f'Teoretické kvantily χ²(df = {p_mv})')
    ax_qq_mv.set_ylabel('Uspořádané Mahalanobisovy vzdálenosti D²')
    ax_qq_mv.set_title('Chí-kvadrát Q-Q graf (D²)')

    crit_95 = stats.chi2.ppf(0.95, df=p_mv)
    crit_99 = stats.chi2.ppf(0.99, df=p_mv)

    ax_hist_mv.hist(d2_mahal, bins=30, density=True, color='#1f77b4', edgecolor='white', alpha=0.75)
    x_grid = np.linspace(0.1, max(d2_mahal), 300)
    ax_hist_mv.plot(x_grid, stats.chi2.pdf(x_grid, df=p_mv), 'r-', linewidth=2.5)
    ax_hist_mv.axvline(crit_95, color='#ff7f0e', linestyle=':', linewidth=2, label=f'95% práh')
    ax_hist_mv.axvline(crit_99, color='#d62728', linestyle='--', linewidth=2, label=f'99% práh')
    ax_hist_mv.set_xlabel('Mahalanobisova vzdálenost D²')
    ax_hist_mv.set_title('Detekce vícerozměrných odlehlých hodnot')
    ax_hist_mv.legend(loc='upper right')

    plt.tight_layout()
    uloz_graf('02b_mahal_chisq_qq.png')

    return {'sw': sw_vysledky, 'L_stat': L_stat, 'L_p': L_p, 'hz_stat': hz_stat, 'hz_p': hz_p}


# 3. Korelační analýza
def analyzuj_korelace(df):
    print('\nPočítám korelační matice a parciální korelace...')

    df_k = df.copy()
    df_k['y_dev'] = np.abs(df_k['y_m'] - 34.0)
    corr_vars = ['dist_m', 'angle_deg', 'y_dev', 'xG', 'minute']
    spearman_mat = df_k[corr_vars].corr(method='spearman')

    r_sp, p_sp = stats.spearmanr(df_k['xG'], df_k['dist_m'])
    part_corr = pg.partial_corr(data=df_k, x='xG', y='angle_deg', covar='dist_m', method='spearman')
    r_part = float(part_corr.iloc[0]['r'])
    p_col = [c for c in part_corr.columns if 'p' in c.lower() and ('val' in c.lower() or 'unc' in c.lower())][0]
    p_part = float(part_corr.iloc[0][p_col])

    # Korelační heatmapa
    fig, ax = plt.subplots(figsize=(10, 7.2))
    sns.heatmap(spearman_mat, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'Spearmanova korelační matice', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    uloz_graf('03_korelacni_heatmapa.png')

    # Pair-plot klíčových proměnných
    pp_data = df[['dist_m', 'angle_deg', 'xG', 'goal']].copy()
    pp_data['Výsledek'] = pp_data['goal'].map({1: 'Gól (1)', 0: 'Bez gólu (0)'})
    g_pair = sns.pairplot(pp_data, vars=['dist_m', 'angle_deg', 'xG'], hue='Výsledek',
                          palette={'Gól (1)': '#d62728', 'Bez gólu (0)': '#1f77b4'}, height=2.4, aspect=1.3,
                          plot_kws={'alpha': 0.60, 's': 32, 'edgecolor': 'none'},
                          diag_kws={'fill': True, 'alpha': 0.45})
    g_pair.savefig(os.path.join(VYSTUPY_DIR, '03b_pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close(g_pair.figure)
    print('Uloženo: 03b_pairplot.png')

    return {'r_sp': r_sp, 'p_sp': p_sp, 'r_part': r_part, 'p_part': p_part}


# 4. Welch ANOVA a MANOVA
def analyzuj_anova_manova(df):
    print('\nTestuji rozdíly typu zakončení (ANOVA, Kruskal, MANOVA)...')

    main_shots = ['RightFoot', 'LeftFoot', 'Head']
    df_shots = df[df['shotType'].isin(main_shots)].copy()

    # Welchova ANOVA a Kruskal-Wallis
    welch_dist = pg.welch_anova(data=df_shots, dv='dist_m', between='shotType')
    F_welch_d = float(welch_dist.iloc[0]['F'])
    p_welch_d = float(welch_dist.iloc[0]['p_unc'] if 'p_unc' in welch_dist.columns else welch_dist.iloc[0]['p-unc'])

    welch_ang = pg.welch_anova(data=df_shots, dv='angle_deg', between='shotType')
    F_welch_a = float(welch_ang.iloc[0]['F'])
    p_welch_a = float(welch_ang.iloc[0]['p_unc'] if 'p_unc' in welch_ang.columns else welch_ang.iloc[0]['p-unc'])

    kw_res = pg.kruskal(data=df_shots, dv='dist_m', between='shotType')
    H_kw = float(kw_res.iloc[0]['H'])
    p_kw = float(kw_res.iloc[0]['p_unc'] if 'p_unc' in kw_res.columns else kw_res.iloc[0]['p-unc'])

    # Boxploty
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    sns.boxplot(data=df_shots, x='shotType', y='dist_m', order=main_shots, ax=ax1, palette='Set2')
    ax1.set_title('Vzdálenost střely dle zakončení', fontweight='bold')
    sns.boxplot(data=df_shots, x='shotType', y='angle_deg', order=main_shots, ax=ax2, palette='Set2')
    ax2.set_title('Střelecký úhel dle zakončení', fontweight='bold')
    plt.tight_layout()
    uloz_graf('04_boxploty_shottype.png')

    # Boxův M test a MANOVA
    box_m_res = pg.box_m(df_shots, dvs=['dist_m', 'angle_deg', 'xG'], group='shotType')
    box_chi2 = float(box_m_res.iloc[0]['Chi2'])
    box_p = float(box_m_res.iloc[0]['pval'] if 'pval' in box_m_res.columns else box_m_res.iloc[0]['p-val'])

    manova = MANOVA.from_formula('dist_m + angle_deg ~ shotType', data=df_shots)
    manova_res = manova.mv_test()
    wilks_val = manova_res.results['shotType']['stat'].loc["Wilks' lambda", 'Value']
    wilks_p = manova_res.results['shotType']['stat'].loc["Wilks' lambda", 'Pr > F']
    pillai_val = manova_res.results['shotType']['stat'].loc["Pillai's trace", 'Value']
    pillai_p = manova_res.results['shotType']['stat'].loc["Pillai's trace", 'Pr > F']

    # Tukey HSD
    tukey_d = pairwise_tukeyhsd(df_shots['dist_m'], df_shots['shotType'], alpha=0.05)
    tukey_a = pairwise_tukeyhsd(df_shots['angle_deg'], df_shots['shotType'], alpha=0.05)

    fig, (ax_td, ax_ta) = plt.subplots(1, 2, figsize=(14.2, 5.8))
    td_df = pd.DataFrame(tukey_d._results_table.data[1:], columns=tukey_d._results_table.data[0])
    pairs_d = [f"{r['group1']} vs {r['group2']}" for _, r in td_df.iterrows()]
    y_pos = np.arange(len(pairs_d))

    ax_td.errorbar(td_df['meandiff'], y_pos,
                   xerr=[td_df['meandiff'] - td_df['lower'], td_df['upper'] - td_df['meandiff']], fmt='o',
                   color='#1f77b4', capsize=6, capthick=2)
    ax_td.axvline(0, color='#d62728', linestyle='--')
    ax_td.set_yticks(y_pos)
    ax_td.set_yticklabels(pairs_d)
    ax_td.set_title('Tukey HSD: Vzdálenost (m)', fontweight='bold')

    ta_df = pd.DataFrame(tukey_a._results_table.data[1:], columns=tukey_a._results_table.data[0])
    pairs_a = [f"{r['group1']} vs {r['group2']}" for _, r in ta_df.iterrows()]
    ax_ta.errorbar(ta_df['meandiff'], y_pos,
                   xerr=[ta_df['meandiff'] - ta_df['lower'], ta_df['upper'] - ta_df['meandiff']], fmt='o',
                   color='#2ca02c', capsize=6, capthick=2)
    ax_ta.axvline(0, color='#d62728', linestyle='--')
    ax_ta.set_yticks(y_pos)
    ax_ta.set_yticklabels(pairs_a)
    ax_ta.set_title('Tukey HSD: Střelecký úhel (°)', fontweight='bold')

    plt.tight_layout()
    uloz_graf('04b_tukey_hsd.png')

    return {'F_welch_d': F_welch_d, 'p_welch_d': p_welch_d, 'F_welch_a': F_welch_a, 'p_welch_a': p_welch_a,
            'H_kw': H_kw, 'p_kw': p_kw, 'box_chi2': box_chi2, 'box_p': box_p,
            'wilks_val': wilks_val, 'wilks_p': wilks_p, 'pillai_val': pillai_val, 'pillai_p': pillai_p}


# 4b. Závislost xG na hráči
def analyzuj_hrace(df):
    print('\nAnalýza kvality šancí dle střelce...')

    kw_p = pg.kruskal(data=df, dv='xG', between='player')
    H_kw_p = float(kw_p.iloc[0]['H'])
    p_kw_p = float(kw_p.iloc[0]['p_unc'] if 'p_unc' in kw_p.columns else kw_p.iloc[0]['p-unc'])

    welch_p = pg.welch_anova(data=df, dv='xG', between='player')
    F_welch_p = float(welch_p.iloc[0]['F'])
    p_welch_p = float(welch_p.iloc[0]['p_unc'] if 'p_unc' in welch_p.columns else welch_p.iloc[0]['p-unc'])

    player_order = df.groupby('player')['xG'].median().sort_values(ascending=False).index.tolist()
    player_stats = df.groupby('player').agg(
        n=('xG', 'count'), goals=('goal', 'sum'), median_xG=('xG', 'median'),
        mean_xG=('xG', 'mean'), mean_dist=('dist_m', 'mean')
    ).loc[player_order].reset_index()
    player_stats['conversion_%'] = (player_stats['goals'] / player_stats['n']) * 100
    player_stats.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_hraci_xG.csv'), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.5), gridspec_kw={'width_ratios': [1.3, 1.0]})
    labels_with_n = [f'{p} (n={df[df["player"] == p].shape[0]})' for p in player_order]
    sns.boxplot(data=df, y='player', x='xG', order=player_order, hue='player', palette='Blues_r', legend=False, ax=ax1,
                showmeans=True)
    ax1.set_yticks(range(len(player_order)))
    ax1.set_yticklabels(labels_with_n, fontsize=12)
    ax1.set_title(f'Distribuce xG dle hráčů', fontsize=13, fontweight='bold')

    sc = ax2.scatter(player_stats['mean_dist'], player_stats['mean_xG'], s=player_stats['n'] * 4.5 + 80,
                     c=player_stats['conversion_%'], cmap='YlOrRd', edgecolors='black')
    cbar = plt.colorbar(sc, ax=ax2, shrink=0.85)
    cbar.set_label('Gólová konverze střelce (%)')
    for _, row in player_stats.iterrows():
        ax2.annotate(row['player'], (row['mean_dist'], row['mean_xG']), textcoords='offset points', xytext=(8, 4),
                     fontsize=10.5)
    ax2.set_xlabel('Průměrná vzdálenost (m)')
    ax2.set_ylabel('Průměrné xG')
    ax2.set_title('Taktické pozice: Vzdálenost vs xG', fontsize=13, fontweight='bold')

    plt.tight_layout()
    uloz_graf('04c_hraci_xG_boxplot.png')

    return {'H_kw_p': H_kw_p, 'p_kw_p': p_kw_p, 'F_welch_p': F_welch_p, 'p_welch_p': p_welch_p}


# 5. PCA a faktorová analýza
def analyzuj_pca_a_fa(df):
    print('\nVýpočet PCA a Faktorové analýzy...')

    df_pca = df.copy()
    df_pca['y_dev'] = np.abs(df_pca['y_m'] - 34.0)
    pca_vars = ['dist_m', 'angle_deg', 'y_dev', 'xG']

    kmo_val = vypocti_kmo(df_pca[pca_vars])
    R_mat = df_pca[pca_vars].corr().values
    n_b, p_b = len(df_pca), len(pca_vars)
    chi2_b = -(n_b - 1 - (2 * p_b + 5) / 6) * np.log(np.linalg.det(R_mat))
    df_b = p_b * (p_b - 1) // 2
    p_val_b = 1.0 - stats.chi2.cdf(chi2_b, df=df_b)

    X_std = StandardScaler().fit_transform(df_pca[pca_vars])
    pca = PCA()
    X_pca = pca.fit_transform(X_std)
    eigen_vals = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_ratio)

    fig, (ax_scr, ax_bip) = plt.subplots(1, 2, figsize=(14.5, 6.2))
    ax_scr.bar(range(1, len(eigen_vals) + 1), eigen_vals, color='#1f77b4', alpha=0.75)
    ax_scr.plot(range(1, len(eigen_vals) + 1), eigen_vals, 'ro-', linewidth=2)
    ax_scr.axhline(1.0, color='#d62728', linestyle='--')
    ax_scr.set_title('Scree plot', fontweight='bold')

    sc = ax_bip.scatter(X_pca[:, 0], X_pca[:, 1], c=df['goal'], cmap='RdBu_r', alpha=0.55, s=32)
    plt.colorbar(sc, ax=ax_bip, shrink=0.85)
    loadings_pca = pca.components_.T[:, :2]
    scale_arrow = 2.5
    for j, var in enumerate(pca_vars):
        lx, ly = loadings_pca[j, 0] * scale_arrow, loadings_pca[j, 1] * scale_arrow
        ax_bip.annotate('', xy=(lx, ly), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.2))
        ax_bip.text(lx * 1.2, ly * 1.2, var, fontsize=12)
    ax_bip.set_title('2D Biplot', fontweight='bold')
    plt.tight_layout()
    uloz_graf('05b_biplot_pca.png')

    fa = FactorAnalysis(n_components=2, rotation='varimax', random_state=42)
    fa.fit(X_std)
    fa_loadings = pd.DataFrame(fa.components_.T, index=pca_vars, columns=['Faktor 1', 'Faktor 2'])
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sns.heatmap(fa_loadings, annot=True, fmt='.3f', cmap='Blues', ax=ax)
    ax.set_title('Faktorová analýza s rotací Varimax', fontweight='bold')
    plt.tight_layout()
    uloz_graf('05c_fa_varimax_loadings.png')

    return {'kmo_val': kmo_val, 'chi2_b': chi2_b, 'p_val_b': p_val_b, 'cum_var_2': cum_var[1]}


# 6. Shlukování
def analyzuj_shluky(df):
    print('\nShlukování (Ward a K-means)...')

    clus_vars = ['dist_m', 'angle_deg', 'xG']
    X_clus = StandardScaler().fit_transform(df[clus_vars])
    link_mat = linkage(X_clus, method='ward', metric='euclidean')

    # Nastavení řezu na 3 shluky (mezi spojením 3->2 a 4->3)
    h_cut_3 = (link_mat[-2, 2] + link_mat[-3, 2]) / 2.0
    ward_raw = fcluster(link_mat, 3, criterion='maxclust')

    df_c = df.copy()
    mean_dists = {cid: df_c[ward_raw == cid]['dist_m'].mean() for cid in np.unique(ward_raw)}
    sorted_cids = sorted(mean_dists, key=mean_dists.get)
    label_map = {old: new for new, old in enumerate(sorted_cids)}
    df_c['cluster'] = [label_map[l] for l in ward_raw]

    k_range = list(range(2, 7))
    sil_scores, inertias = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=15).fit(X_clus)
        sil_scores.append(silhouette_score(X_clus, km.labels_))
        inertias.append(km.inertia_)

    c1, c2, c3 = df_c[df_c['cluster'] == 0], df_c[df_c['cluster'] == 1], df_c[df_c['cluster'] == 2]

    fig, (ax_dend, ax_map) = plt.subplots(1, 2, figsize=(15.2, 6.4))
    dendrogram(link_mat, ax=ax_dend, truncate_mode='lastp', p=20, color_threshold=h_cut_3)
    ax_dend.axhline(h_cut_3, color='#d62728', linestyle='--')
    ax_dend.set_title('Dendrogram hierarchického shlukování', fontweight='bold')

    nakresli_hriste(ax_map)
    ax_map.scatter(c3['x_m'], c3['y_m'], c='#1f77b4', s=45, alpha=0.60, label=f'Dálka (n={len(c3)})')
    ax_map.scatter(c2['x_m'], c2['y_m'], c='#ff7f0e', s=55, alpha=0.75, label=f'Velké vápno (n={len(c2)})')
    ax_map.scatter(c1['x_m'], c1['y_m'], c='#d62728', s=70, alpha=0.90, label=f'Malé vápno (n={len(c1)})')
    ax_map.legend(loc='upper left')
    ax_map.set_title('Taktické zóny zakončení (k = 3)', fontweight='bold')
    plt.tight_layout()
    uloz_graf('06b_mapa_shluku.png')

    return {'opt_k': 3, 'c1_n': len(c1), 'c2_n': len(c2), 'c3_n': len(c3)}


# 7. Logistická regrese a ROC
def analyzuj_logistickou_regresi(df):
    print('\nVýpočet logistické regrese...')

    shot_dummies = pd.get_dummies(df['shotType'], prefix='shotType', drop_first=True, dtype=float)
    X_reg = pd.concat([df[['dist_m', 'angle_deg', 'minute']], shot_dummies], axis=1)
    X_reg_const = sm.add_constant(X_reg)
    y_reg = df['goal']

    logit_mod = sm.Logit(y_reg, X_reg_const).fit(disp=False)
    coefs = logit_mod.params
    p_vals = logit_mod.pvalues
    odds_r = np.exp(coefs)

    y_pred_prob = logit_mod.predict(X_reg_const)
    fpr_mod, tpr_mod, _ = roc_curve(y_reg, y_pred_prob)
    auc_mod = auc(fpr_mod, tpr_mod)
    fpr_xg, tpr_xg, _ = roc_curve(y_reg, df['xG'])
    auc_xg = auc(fpr_xg, tpr_xg)

    mod_univ = sm.Logit(y_reg, sm.add_constant(df['dist_m'])).fit(disp=False)
    dist_seq = np.linspace(2, 42, 200)
    prob_seq = mod_univ.predict(sm.add_constant(dist_seq))

    fig, (ax_sig, ax_roc) = plt.subplots(1, 2, figsize=(14.5, 6.2))
    ax_sig.scatter(df[df['goal'] == 0]['dist_m'], np.zeros(sum(df['goal'] == 0)), c='#1f77b4', alpha=0.45)
    ax_sig.scatter(df[df['goal'] == 1]['dist_m'], np.ones(sum(df['goal'] == 1)), c='#d62728', alpha=0.85)
    ax_sig.plot(dist_seq, prob_seq, color='#d62728', linewidth=3.0)
    ax_sig.set_title('Logistická S-křivka vzdálenosti', fontweight='bold')

    ax_roc.plot(fpr_mod, tpr_mod, color='#d62728', lw=3.0, label=f'Náš model (AUC = {auc_mod:.3f})')
    ax_roc.plot(fpr_xg, tpr_xg, color='#1f77b4', lw=2.4, linestyle='-.', label=f'xG (AUC = {auc_xg:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='#888888', linestyle='--')
    ax_roc.legend()
    ax_roc.set_title('ROC křivka', fontweight='bold')
    plt.tight_layout()
    uloz_graf('07b_logisticka_krivka.png')

    return {'llr_p': logit_mod.llr_pvalue, 'llr_chi2': logit_mod.llr, 'p_vals': p_vals, 'odds_r': odds_r,
            'auc_mod': auc_mod, 'auc_xg': auc_xg}


# 8. Export tabulky hypotéz
def exportuj_tabulku_hypotez(vysledky):
    print('\nGenerování tabulky hypotéz...')

    sw = vysledky['norm']['sw']
    hypotezy = [
        ('1', 'Shapiro-Wilk (dist_m)', 'Vzdálenost dist_m má normální rozdělení', f"W = {sw['dist_m']['W']:.4f}",
         sw['dist_m']['p']),
        ('2', 'Levene (dist_m)', 'Rozptyly vzdálenosti jsou mezi skupinami shodné',
         f"L = {vysledky['norm']['L_stat']:.3f}", vysledky['norm']['L_p']),
        ('3', 'Spearman (dist_m, xG)', 'Korelace mezi vzdáleností a xG je nulová', f"r = {vysledky['kor']['r_sp']:.3f}",
         vysledky['kor']['p_sp']),
        ('4', 'Welch ANOVA (dist_m)', 'Střední vzdálenosti střel jsou shodné',
         f"F = {vysledky['anv']['F_welch_d']:.1f}", vysledky['anv']['p_welch_d']),
        ('5', 'MANOVA', 'Vektory [dist, úhel] jsou shodné', f"Pillai = {vysledky['anv']['pillai_val']:.3f}",
         vysledky['anv']['pillai_p']),
        ('6', 'Bartlett test PCA', 'Korelační matice je jednotková', f"χ² = {vysledky['pca']['chi2_b']:.1f}",
         vysledky['pca']['p_val_b']),
        ('7', 'Logit LLR', 'Všechny koeficienty modelu jsou nulové', f"χ² = {vysledky['logit']['llr_chi2']:.1f}",
         vysledky['logit']['llr_p'])
    ]

    radky_tabulky = [
        {'Metoda': m, 'Nulová hypotéza': h, 'Statistika': s, 'p-hodnota': formatuj_p(p), 'Rozhodnutí': rozhodni_h0(p)}
        for _, m, h, s, p in hypotezy]
    df_hyp_out = pd.DataFrame(radky_tabulky)
    df_hyp_out.to_csv(os.path.join(VYSTUPY_DIR, 'tabulka_hypotezy.csv'), index=False)
    print('Hotovo. Tabulka uložena do tabulka_hypotezy.csv')


# Hlavní spouštěcí blok
if __name__ == '__main__':
    df_clean = pd.read_csv(DATA_PATH)
    print(f'Zahajuji analýzu dat (načteno {len(df_clean)} střel).')

    vysledky = {}
    eda_a_prostorova_mapa(df_clean)
    vysledky['norm'] = analyzuj_normalitu(df_clean)
    vysledky['kor'] = analyzuj_korelace(df_clean)
    vysledky['anv'] = analyzuj_anova_manova(df_clean)
    vysledky['hraci'] = analyzuj_hrace(df_clean)
    vysledky['pca'] = analyzuj_pca_a_fa(df_clean)
    vysledky['clus'] = analyzuj_shluky(df_clean)
    vysledky['logit'] = analyzuj_logistickou_regresi(df_clean)
    exportuj_tabulku_hypotez(vysledky)

    print('\nAnalýza úspěšně dokončena.')