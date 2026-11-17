# -*- coding: utf-8 -*-
"""
=============================================================================
SEMESTRÁLNÍ PROJEKT VMM – STŘELECKÁ ÚSPĚŠNOST (PREMIER LEAGUE)
Skript: Generátor finální prezentace PowerPoint (15 slidů, 16:9, čistý styl)
=============================================================================
Konstrukce prezentace:
  - Slide 1: Titulní snímek (čistý akademický design)
  - Slide 2: Představení datového vzorku a metodiky (textový slide)
  - Slide 3 až 14: Obří vycentrované grafy zachovávající původní poměr stran
  - Slide 15: Souhrnná tabulka VŠECH 14 hypotéz dynamicky načtená ze souboru
              tabulka_hypotezy.csv + 4 praktická doporučení pro trenéry
=============================================================================
"""

import os
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# Cesty
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VYSTUPY_DIR = os.path.join(BASE_DIR, 'vystupy')
PREZ_DIR = os.path.join(BASE_DIR, 'prezentace')
os.makedirs(PREZ_DIR, exist_ok=True)
PPTX_PATH = os.path.join(PREZ_DIR, 'prezentace_strelecka_uspesnost.pptx')
TAB_HYP_PATH = os.path.join(VYSTUPY_DIR, 'tabulka_hypotezy.csv')

# Barevná paleta
BILA          = RGBColor(0xFF, 0xFF, 0xFF)
CERNA         = RGBColor(0x1A, 0x1A, 0x1A)
TMAVA_SEDA    = RGBColor(0x33, 0x33, 0x33)
STRED_SEDA    = RGBColor(0x77, 0x77, 0x77)
SVETLA_SEDA   = RGBColor(0xE0, 0xE0, 0xE0)
MODRA_BOD     = RGBColor(0x1F, 0x77, 0xB4)
CERVENA_ZAM   = RGBColor(0xC5, 0x30, 0x30)
ZELENA_NEZAM  = RGBColor(0x28, 0xA7, 0x45)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.50)
BLANK = prs.slide_layouts[6]
TOTAL_SLIDES = 15


def novy_slide():
    """Vytvoří nový prázdný snímek s bílým pozadím."""
    sl = prs.slides.add_slide(BLANK)
    bg = sl.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BILA
    return sl


def text_pole(slide, text, l, t, w, h,
              font_size=13, bold=False, italic=False,
              fg=TMAVA_SEDA, align=PP_ALIGN.LEFT, wrap=True):
    """Vloží textové pole se specifikovaným stylem."""
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txb.word_wrap = wrap
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(font_size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = fg
    return txb


def linka(slide, l, t, w, tloustka=0.015, barva=SVETLA_SEDA):
    """Vykreslí oddělovací horizontální linku."""
    sh = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(tloustka))
    sh.fill.solid()
    sh.fill.fore_color.rgb = barva
    sh.line.fill.background()
    return sh


def zahlavi(slide, nadpis, podtitulek=None):
    """Vytvoří jednotné záhlaví snímku."""
    text_pole(slide, nadpis, 0.8, 0.28, 11.73, 0.42,
              font_size=20, bold=True, fg=CERNA)
    if podtitulek:
        text_pole(slide, podtitulek, 0.8, 0.68, 11.73, 0.28,
              font_size=11, italic=True, fg=STRED_SEDA)
        linka(slide, 0.8, 0.98, 11.73)
    else:
        linka(slide, 0.8, 0.80, 11.73)


def cislo_slidu(slide, n, total=TOTAL_SLIDES):
    """Vloží číslo snímku do pravého dolního rohu."""
    text_pole(slide, f'{n} / {total}', 12.0, 7.15, 0.8, 0.25,
              font_size=9, fg=STRED_SEDA, align=PP_ALIGN.RIGHT)


def vloz_obri_graf(slide, soubor, w_palce=10.5, t_palce=1.05):
    """Vloží vycentrovaný graf se zachováním poměru stran."""
    cesta = os.path.join(VYSTUPY_DIR, soubor)
    if not os.path.exists(cesta):
        print(f'  [VAROVÁNÍ] Graf nenalezen: {soubor}')
        return None
    l_palce = (13.33 - w_palce) / 2.0
    return slide.shapes.add_picture(cesta, Inches(l_palce), Inches(t_palce), width=Inches(w_palce))


def spodni_box_hypoteza_zaver(slide, test_verdikt, fotbal_zaver, is_zamitnuto=True):
    """Vloží strukturovaný informační box pod graf."""
    l, t, w, h = 0.8, 6.26, 11.73, 0.84

    sh = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor(0xF9, 0xFA, 0xFB)
    sh.line.color.rgb = SVETLA_SEDA
    sh.line.width = Pt(1)

    sh_str = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(0.08), Inches(h))
    sh_str.fill.solid()
    sh_str.fill.fore_color.rgb = MODRA_BOD
    sh_str.line.fill.background()

    txb = slide.shapes.add_textbox(Inches(l + 0.18), Inches(t + 0.04), Inches(w - 0.30), Inches(h - 0.08))
    txb.word_wrap = True
    tf = txb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    r_tag1 = p1.add_run()
    r_tag1.text = 'Statistický test & verdikt: '
    r_tag1.font.bold = True
    r_tag1.font.size = Pt(11.5)
    r_tag1.font.color.rgb = CERNA

    r_val1 = p1.add_run()
    r_val1.text = test_verdikt
    r_val1.font.bold = True
    r_val1.font.size = Pt(11.5)
    r_val1.font.color.rgb = CERVENA_ZAM if is_zamitnuto else ZELENA_NEZAM

    p2 = tf.add_paragraph()
    r_tag2 = p2.add_run()
    r_tag2.text = 'Fotbalový závěr: '
    r_tag2.font.bold = True
    r_tag2.font.size = Pt(11)
    r_tag2.font.color.rgb = CERNA

    r_val2 = p2.add_run()
    r_val2.text = fotbal_zaver
    r_val2.font.size = Pt(11)
    r_val2.font.color.rgb = TMAVA_SEDA


print('=' * 75)
print('GENEROVÁNÍ POWERPOINTOVÉ PREZENTACE (16 SLIDŮ)')
print('=' * 75)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 – TITULNÍ SNÍMEK
# ════════════════════════════════════════════════════════════════════════════
s1 = novy_slide()
text_pole(s1, 'VÍZEROZMĚRNÁ STATISTICKÁ ANALÝZA', 1.0, 1.80, 11.33, 0.40,
          font_size=13, bold=True, fg=MODRA_BOD)
text_pole(s1, 'Faktory střelecké úspěšnosti\nv anglické Premier League', 1.0, 2.30, 11.33, 1.40,
          font_size=32, bold=True, fg=CERNA)
linka(s1, 1.0, 3.85, 4.0, tloustka=0.03, barva=MODRA_BOD)
text_pole(s1,
          'Aplikace vícerozměrných metod na prostorová data zakončení:\n'
          'Multinormalita • Welch ANOVA • Kruskal-Wallis • MANOVA • PCA • Faktorová analýza • K-means • Logistická regrese',
          1.0, 4.05, 11.33, 0.70, font_size=13, italic=True, fg=STRED_SEDA)

text_pole(s1,
          'Předmět:  Vícerozměrné statistické metody (VMM)\n'
          'Instituce: Vysoké učení technické v Brně (VUT)\n'
          'Akademický rok: 2026/2027  •  Dataset: understat.com (Premier League)',
          1.0, 5.20, 11.33, 0.90, font_size=12, fg=TMAVA_SEDA)
cislo_slidu(s1, 1)
print('  [OK] Slide 1: Titulní snímek vytvořen')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 – PŘEDSTAVENÍ DAT A PŘEDZPRACOVÁNÍ
# ════════════════════════════════════════════════════════════════════════════
s2 = novy_slide()
zahlavi(s2, 'Představení datového vzorku a metodického postupu',
        'Základní popis analyzované datové sady z Premier League, ošetření dat a výzkumné cíle')

sh1 = s2.shapes.add_shape(1, Inches(0.8), Inches(1.15), Inches(5.7), Inches(5.6))
sh1.fill.solid(); sh1.fill.fore_color.rgb = RGBColor(0xF9, 0xFA, 0xFB)
sh1.line.color.rgb = SVETLA_SEDA; sh1.line.width = Pt(1)
sh1_str = s2.shapes.add_shape(1, Inches(0.8), Inches(1.15), Inches(0.08), Inches(5.6))
sh1_str.fill.solid(); sh1_str.fill.fore_color.rgb = MODRA_BOD; sh1_str.line.fill.background()

text_pole(s2, 'Datový vzorek & Očištění dat', 1.05, 1.28, 5.3, 0.35, font_size=14, bold=True, fg=CERNA)
text_pole(s2,
    '• Datový zdroj: 1 000 střel z anglické Premier League (understat.com).\n\n'
    '• Vyřazení vlastních gólů (n = 2): Vlastní góly neodpovídají útočnému záměru a mají nestandardní souřadnice.\n\n'
    '• Transparentní řešení chybějících údajů: 263 střel bez souřadnic X/Y bylo separováno a porovnáno; nevykazují selekční zkreslení.\n\n'
    '• Korektní zachování kategorií: Texty "NA" u asistencí a "None" u akcí nebyly smazány, ale převedeny na platné kategorie.\n\n'
    '• Finální vzorek pro geometrickou analýzu: n = 735 střel, z toho 63 gólů (konverze 8,57 %).\n\n'
    '• Zrušení 3-sigma pravidla: Platné góly z dálky nebyly uměle smazány.',
    1.05, 1.70, 5.3, 4.85, font_size=12, fg=TMAVA_SEDA)

sh2 = s2.shapes.add_shape(1, Inches(6.8), Inches(1.15), Inches(5.7), Inches(5.6))
sh2.fill.solid(); sh2.fill.fore_color.rgb = RGBColor(0xF9, 0xFA, 0xFB)
sh2.line.color.rgb = SVETLA_SEDA; sh2.line.width = Pt(1)
sh2_str = s2.shapes.add_shape(1, Inches(6.8), Inches(1.15), Inches(0.08), Inches(5.6))
sh2_str.fill.solid(); sh2_str.fill.fore_color.rgb = MODRA_BOD; sh2_str.line.fill.background()

text_pole(s2, 'Sledované proměnné & Cíle práce', 7.05, 1.28, 5.3, 0.35, font_size=14, bold=True, fg=CERNA)
text_pole(s2,
    '• Reálné rozměry hřiště: 105 × 68 m, šířka branky 7,32 m (y = 30,34 až 37,66 m).\n\n'
    '• dist_m: Eukleidovská vzdálenost od brankové čáry (rozsah 2,8 až 38,1 m, průměr 19,7 m).\n\n'
    '• angle_deg: Střelecký zorný úhel sevřený brankovými tyčemi (rozsah 6,5° až 104,5°, průměr 21,3°).\n\n'
    '• y_dev: Boční odchylka střely od podélné osy hřiště (|y - 34| m).\n\n'
    '• Metrika xG: Použita výhradně jako externí validační benchmark, nikoli jako prediktor v regresi (eliminace cirkularity).\n\n'
    '• Hlavní cíl: Vícerozměrně identifikovat faktory úspěšnosti zakončení a navrhnout praktická trenérská doporučení.',
    7.05, 1.70, 5.3, 4.85, font_size=12, fg=TMAVA_SEDA)

cislo_slidu(s2, 2)
print('  [OK] Slide 2: Představení dat vytvořeno')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 – MAPA STŘEL NA HŘIŠTI
# ════════════════════════════════════════════════════════════════════════════
s3 = novy_slide()
zahlavi(s3, 'Prostorová mapa střel na fotbalovém hřišti (105 × 68 m)',
        'Zobrazení všech analyzovaných střel Premier League s rozlišením vstřelených gólů')
vloz_obri_graf(s3, '01_mapa_strel.png')
spodni_box_hypoteza_zaver(s3,
    'Vzorek: n = 735 střel | Vstřelené góly: 63 (8,57 %) | Reálné hřiště Premier League 105 × 68 m',
    'Góly jsou výrazně lokalizovány v centrálním prostoru velkého vápna; pokusy z velkých vzdáleností vykazují minimální konverzi.',
    is_zamitnuto=False)
cislo_slidu(s3, 3)
print('  [OK] Slide 3: Mapa střel vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 – 2D KDE HUSTOTA: NEPROMĚNĚNÉ STŘELY VS. VSTŘELENÉ GÓLY
# ════════════════════════════════════════════════════════════════════════════
s4 = novy_slide()
zahlavi(s4, 'Srovnání 2D KDE hustoty: Neproměněné střely vs. Vstřelené góly',
        'Jádrový odhad prostorové hustoty: rozptyl 672 neproměněných střel vs. ohnisko 63 gólů (VMM-3/4)')
vloz_obri_graf(s4, '01c_kde_hriste.png')
spodni_box_hypoteza_zaver(s4,
    '2D KDE na útočných polovinách hřiště | Vlevo: 672 neproměněných střel | Vpravo: 63 vstřelených gólů',
    'Porovnání obou polovin jasně demonstruje disparitu: zatímco neproměněné pokusy mají těžiště na hranici vápna (medián 19,8 m), góly jsou téměř výhradně zakončeny z malého vápna (medián 10,8 m).',
    is_zamitnuto=False)
cislo_slidu(s4, 4)
print('  [OK] Slide 4: 2D KDE vytvořeno')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 – DIAGNOSTIKA NORMALITY & ROZPTYLŮ
# ════════════════════════════════════════════════════════════════════════════
s5 = novy_slide()
zahlavi(s5, 'Diagnostika jednorozměrné normality (dist_m) & shody rozptylů',
        'Histogram vzdálenosti s proloženou normální hustotou a empirický Q-Q graf kvantilů')
vloz_obri_graf(s5, '02_normalita_histogramy_qq.png')
spodni_box_hypoteza_zaver(s5,
    'Shapiro-Wilk: W = 0.9833, p < 0.001 -> H₀ ZAMÍTNOUTA | Levene: L = 14.860, p < 0.001 -> H₀ ZAMÍTNOUTA',
    'Vzdálenost střel vykazuje statisticky významné porušení normality i homogenity rozptylů. Je nutné aplikovat robustní Welchovu ANOVA a neparametrické testy.',
    is_zamitnuto=True)
cislo_slidu(s5, 5)
print('  [OK] Slide 5: Jednorozměrná normalita vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 – DIAGNOSTIKA VÍCEROZMĚRNÉ NORMALITY (MAHALANOBIS)
# ════════════════════════════════════════════════════════════════════════════
s6 = novy_slide()
zahlavi(s6, 'Diagnostika vícerozměrné normality: Chí-kvadrát Q-Q graf (VMM-2)',
        'Mahalanobisova vzdálenost D² vektoru [dist_m, angle_deg, xG] vs. teoretické kvantily χ²(3)')
vloz_obri_graf(s6, '02b_mahal_chisq_qq.png')
spodni_box_hypoteza_zaver(s6,
    'Henze-Zirkler test: HZ = 68.281, p < 0.001 -> H₀ ZAMÍTNOUTA | Sdružené rozdělení není vícerozměrně normální',
    'Body v Q-Q grafu systematicky vybočují z teoretické diagonály χ²(3); detekované multivariátní odlehlé hodnoty odpovídají střelám z extrémních pozic.',
    is_zamitnuto=True)
cislo_slidu(s6, 6)
print('  [OK] Slide 6: Vícerozměrná normalita vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 – SPEARMANOVA KORELACE & PARCIÁLNÍ KORELACE
# ════════════════════════════════════════════════════════════════════════════
s7 = novy_slide()
zahlavi(s7, 'Spearmanova korelační analýza & parciální korelace',
        'Pořadová korelační matice a očištění vztahu kvality šance a úhlu o vliv vzdálenosti')
vloz_obri_graf(s7, '03_korelacni_heatmapa.png')
spodni_box_hypoteza_zaver(s7,
    'Spearman r(xG, dist_m) = -0.740, p < 0.001 -> H₀ ZAMÍTNOUTA | Parciální r(xG, úhel | dist) = +0.155, p < 0.001 -> H₀ ZAMÍTNOUTA',
    'Vzdálenost je nejsilnějším negativním faktorem střely. Po fixování vzdálenosti má střelecký úhel nezávislý pozitivní vliv na kvalitu zakončení.',
    is_zamitnuto=True)
cislo_slidu(s7, 7)
print('  [OK] Slide 7: Korelační matice vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 – PAIR-PLOT VZTAHŮ KLÍČOVÝCH PROMĚNNÝCH
# ════════════════════════════════════════════════════════════════════════════
s8 = novy_slide()
zahlavi(s8, 'Párové rozptylové diagramy a marginální hustoty (KDE)',
        'Multivariátní zkoumání separace střel dle výsledku zakončení (Gól vs. Bez gólu)')
vloz_obri_graf(s8, '03b_pairplot.png')
spodni_box_hypoteza_zaver(s8,
    'Párová diskriminace klíčových metrik | Diagonální marginální KDE separace gólů (červená)',
    'Góly jsou jasně koncentrovány v zóně krátké vzdálenosti (< 15 m) a širokého zorného úhlu (> 25°). Mimo tuto oblast je konverze zanedbatelná.',
    is_zamitnuto=False)
cislo_slidu(s8, 8)
print('  [OK] Slide 8: Pair-plot vytvořen')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 – WELCHOVA ANOVA & KRUSKAL-WALLIS
# ════════════════════════════════════════════════════════════════════════════
s9 = novy_slide()
zahlavi(s9, 'Rozdíly geometrie dle typu zakončení (Welchova ANOVA & Kruskal-Wallis)',
        'Robustní parametrické a neparametrické porovnání vzdálenosti a úhlu pro pravou nohu, levou nohu a hlavu')
vloz_obri_graf(s9, '04_boxploty_shottype.png')
spodni_box_hypoteza_zaver(s9,
    'Welch ANOVA: F = 176.87, p < 0.001 -> H₀ ZAMÍTNOUTA | Kruskal-Wallis: H = 88.67, p < 0.001 -> H₀ ZAMÍTNOUTA',
    'Hlavičky se diametrálně liší od střelby nohou – vznikají z podstatně kratší vzdálenosti (průměr ~10 m vs. ~20 m u nohou) a ostřejších úhlů.',
    is_zamitnuto=True)
cislo_slidu(s9, 9)
print('  [OK] Slide 9: Welchova ANOVA vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 – MANOVA & TUKEY HSD
# ════════════════════════════════════════════════════════════════════════════
s10 = novy_slide()
zahlavi(s10, 'Vícerozměrné posouzení sdruženého vektoru (MANOVA & Tukey HSD)',
        'Sdružený vektor geometrie [dist_m, angle_deg], Boxův M test kovariancí a post-hoc 95% intervaly')
vloz_obri_graf(s10, '04b_tukey_hsd.png')
spodni_box_hypoteza_zaver(s10,
    'Boxův M: χ² = 132.4, p < 0.001 | MANOVA Pillai = 0.185, Wilks λ = 0.815, p < 0.001 -> H₀ ZAMÍTNOUTA',
    'Kvůli porušení rovnosti kovariancí preferujeme robustní Pillaiovu stopu. Post-hoc test potvrzuje signifikantní rozdíl hlavy proti oběma nohám (p < 0.001).',
    is_zamitnuto=True)
cislo_slidu(s10, 10)
print('  [OK] Slide 10: MANOVA vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 – ZÁVISLOST KVALITY ŠANCÍ (xG) NA STŘELCI (HRÁČI)
# ════════════════════════════════════════════════════════════════════════════
s11 = novy_slide()
zahlavi(s11, 'Závislost kvality šancí (xG) na střelci: Kruskal-Wallis & Welch ANOVA',
        'Neparametrické a robustní srovnání distribuce xG dle hráčů a taktické mapování pozic střelců')
vloz_obri_graf(s11, '04c_hraci_xG_boxplot.png')
spodni_box_hypoteza_zaver(s11,
    'Kruskal-Wallis: H = 71.26, p < 0.001 -> H₀ ZAMÍTNOUTA | Welch ANOVA: F = 26.90, p < 0.001 -> H₀ ZAMÍTNOUTA',
    'Kvalita šancí silně závisí na herní roli: hrotový útočník (Watkins, medián xG = 0.271) zakončuje z bezprostřední blízkosti, zatímco střední záložníci (Luiz, McGinn, medián xG = 0.025) střílejí z dálky.',
    is_zamitnuto=True)
cislo_slidu(s11, 11)
print('  [OK] Slide 11: Analýza hráčů vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 – REDUKCE DIMENZIONALITY PCA S TESTEM KMO
# ════════════════════════════════════════════════════════════════════════════
s12 = novy_slide()
zahlavi(s12, 'Redukce dimenzionality PCA (Scree plot, 2D Biplot & KMO)',
        'Dekompozice variability bez souřadnicové redundance X/Y, test KMO a projekce do hlavních komponent')
vloz_obri_graf(s12, '05b_biplot_pca.png')
spodni_box_hypoteza_zaver(s12,
    'Bartlett: χ² = 1370.6, p < 0.001 -> H₀ ZAMÍTNOUTA | KMO index = 0.696 | PC1 + PC2 vysvětlují 84,5 % rozptylu',
    'PC1 (66,0 % rozptylu) tvoří osu centrální kvality střely. PC2 (18,5 %) zachycuje boční excentricitu (y_dev). PCA úspěšně komprimuje geometrii.',
    is_zamitnuto=True)
cislo_slidu(s12, 12)
print('  [OK] Slide 12: PCA vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13 – FAKTOROVÁ ANALÝZA S ROTACÍ VARIMAX
# ════════════════════════════════════════════════════════════════════════════
s13 = novy_slide()
zahlavi(s13, 'Faktorová analýza s rotací Varimax (Latentní faktory)',
        'Identifikace skrytých faktorů zakončení ortogonální rotací faktorových zátěží (VMM-9)')
vloz_obri_graf(s13, '05c_fa_varimax_loadings.png')
spodni_box_hypoteza_zaver(s13,
    'Faktorová analýza (2 faktory, rotace Varimax) | Dosažena čistá interpretovatelná ortogonální struktura',
    'Faktor 1 dominuje centrální nebezpečnost (úhel +0.86, dist -0.82, xG +0.67), Faktor 2 představuje boční excentricitu (y_dev +0.74).',
    is_zamitnuto=False)
cislo_slidu(s13, 13)
print('  [OK] Slide 13: Faktorová analýza vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14 – SHLUKOVÁ ANALÝZA: 3 TAKTICKÉ ZÓNY ZAKONČENÍ (WARD k = 3)
# ════════════════════════════════════════════════════════════════════════════
s14 = novy_slide()
zahlavi(s14, 'Shluková analýza: Wardovo hierarchické dělení na 3 taktické zóny',
        'Přesný řez dendrogramem (k = 3), inflexe lokte a prostorová segmentace střeleckých pozic')
vloz_obri_graf(s14, '06b_mapa_shluku.png')
spodni_box_hypoteza_zaver(s14,
    'Řez dendrogramem h = 23.2 dělí strom na 3 větve | Inflexe lokte: pokles WCSS o 41 % z k = 2 na k = 3',
    '3 herní zóny: Shluk 1 (Zlatá zóna < 12 m: konverze 36,4 %), Shluk 2 (Pokutové území 12–18 m: konverze 6,8 %) a Shluk 3 (Dálka > 18 m: konverze 4,0 %).',
    is_zamitnuto=False)
cislo_slidu(s14, 14)
print('  [OK] Slide 14: Shluková analýza vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15 – LOGISTICKÁ REGRESE BEZ DATA LEAKAGE & ROC BENCHMARK
# ════════════════════════════════════════════════════════════════════════════
s15 = novy_slide()
zahlavi(s15, 'Logistická regrese: Sigmoidální křivka šance & srovnání s xG',
        'Multivariabilní model bez cirkulárního xG prediktoru, Waldovy testy a validační ROC diskriminace')
vloz_obri_graf(s15, '07b_logisticka_krivka.png')
spodni_box_hypoteza_zaver(s15,
    'LLR test: χ² = 69.7, p < 0.001 -> H₀ ZAMÍTNOUTA | dist_m: OR = 0.918 (p = 0.016) -> H₀ ZAMÍTNOUTA | angle: OR = 1.037 (p = 0.014) -> H₀ ZAMÍTNOUTA',
    'Každý metr vzdálenosti snižuje šanci na gól o 8,2 % (OR = 0.918). Náš geometrický model dosahuje AUC = 0.776 a překonává surový benchmark xG (AUC = 0.746).',
    is_zamitnuto=True)
cislo_slidu(s15, 15)
print('  [OK] Slide 15: Logistická regrese vytvořena')

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 16 – SOUHRNNÁ TABULKA VŠECH HYPOTÉZ (DYNAMICKY) & DOPORUČENÍ
# ════════════════════════════════════════════════════════════════════════════
s16 = novy_slide()
zahlavi(s16, 'Souhrnná tabulka testovaných hypotéz & doporučení pro trenéry',
        'Dynamický přehled všech 16 verifikovaných statistických hypotéz projektu a klíčová herní doporučení')

# Dynamické načtení z tabulka_hypotezy.csv
if os.path.exists(TAB_HYP_PATH):
    df_hyp_in = pd.read_csv(TAB_HYP_PATH)
else:
    df_hyp_in = pd.DataFrame()

SLOUPCE_X = [0.8, 2.2, 4.3, 8.2, 9.7, 10.9]
SLOUPCE_W = [1.3, 2.0, 3.8, 1.4, 1.1, 1.63]
HLAVICKY  = ['Fáze', 'Metoda', 'Nulová hypotéza (H₀)', 'Statistika', 'p-hodnota', 'Rozhodnutí']

for j, (h_txt, w_col) in enumerate(zip(HLAVICKY, SLOUPCE_W)):
    sh = s16.shapes.add_shape(1, Inches(SLOUPCE_X[j]), Inches(1.02), Inches(w_col), Inches(0.24))
    sh.fill.solid(); sh.fill.fore_color.rgb = CERNA; sh.line.fill.background()
    text_pole(s16, h_txt, SLOUPCE_X[j] + 0.04, 1.04, w_col - 0.08, 0.20,
              font_size=8.5, bold=True, fg=BILA, align=PP_ALIGN.CENTER)

for i, radek in df_hyp_in.iterrows():
    y_pos = 1.28 + (i * 0.22)
    bg_barva = RGBColor(0xFA, 0xFA, 0xFA) if (i % 2 == 0) else BILA
    is_zam = ('ZAMÍTÁME' in str(radek['Rozhodnutí (alpha=0.05)']))

    hodnoty_radku = [
        str(radek['Fáze']),
        str(radek['Metoda']),
        str(radek['Nulová hypotéza (H0)']),
        str(radek['Statistika']),
        str(radek['p-hodnota']),
        str(radek['Rozhodnutí (alpha=0.05)'])
    ]

    for j in range(6):
        val = hodnoty_radku[j]
        w_col = SLOUPCE_W[j]
        sh = s16.shapes.add_shape(1, Inches(SLOUPCE_X[j]), Inches(y_pos), Inches(w_col), Inches(0.21))
        sh.fill.solid(); sh.fill.fore_color.rgb = bg_barva
        sh.line.color.rgb = RGBColor(0xEE, 0xEE, 0xEE); sh.line.width = Pt(0.5)

        is_verdict = (j == 5)
        text_fg = (CERVENA_ZAM if is_zam else ZELENA_NEZAM) if is_verdict else TMAVA_SEDA
        align = PP_ALIGN.CENTER if j in [0, 3, 4, 5] else PP_ALIGN.LEFT
        text_pole(s16, val, SLOUPCE_X[j] + 0.04, y_pos + 0.01, w_col - 0.08, 0.19,
                  font_size=7.5, bold=is_verdict, fg=text_fg, align=align)

# Dolní část: 4 doporučení pro trenéry ve 2 sloupcích
doporuceni = [
    ('1. Priorita zakončení z vápna', 'Střelba ze vzdálenosti < 12 m má mnohonásobně vyšší konverzi než z dálky; klíčem je trpělivá finální přihrávka do pokutového území.'),
    ('2. Maximalizace střeleckého úhlu', 'Širší střelecký zorný úhel statisticky významně zvyšuje šanci na gól (OR = 1.037 na stupeň); útočníci mají preferovat zakončení v ose.'),
    ('3. Taktická profilace střelců', 'Střelbu je nutné svěřovat profilovaným zakončovatelům v pokutovém území (Watkins) a eliminovat neefektivní objem z dálky u záložníků.'),
    ('4. Taktická eliminace střel z dálky', 'Střelba ze vzdálenosti > 22 metrů vykazuje konverzi pod 3 %; v tréninku je nutné eliminovat unáhlené pokusy z dálky.'),
]

for idx_d, (titul_d, text_d) in enumerate(doporuceni):
    col_x = 0.8 if idx_d % 2 == 0 else 6.8
    row_y = 5.02 if idx_d < 2 else 5.88
    w_card = 5.73

    sh_d = s16.shapes.add_shape(1, Inches(col_x), Inches(row_y), Inches(w_card), Inches(0.76))
    sh_d.fill.solid(); sh_d.fill.fore_color.rgb = RGBColor(0xF9, 0xFA, 0xFB)
    sh_d.line.color.rgb = SVETLA_SEDA; sh_d.line.width = Pt(1)

    sh_str = s16.shapes.add_shape(1, Inches(col_x), Inches(row_y), Inches(0.08), Inches(0.76))
    sh_str.fill.solid(); sh_str.fill.fore_color.rgb = MODRA_BOD; sh_str.line.fill.background()

    text_pole(s16, titul_d, col_x + 0.18, row_y + 0.04, w_card - 0.3, 0.22, font_size=10.0, bold=True, fg=CERNA)
    text_pole(s16, text_d, col_x + 0.18, row_y + 0.26, w_card - 0.3, 0.44, font_size=8.5, fg=TMAVA_SEDA)

text_pole(s16, 'Semestrální projekt VMM 2026/27  •  VUT Brno  •  Děkuji za pozornost',
          0.8, 6.95, 11.73, 0.30, font_size=10, fg=STRED_SEDA, align=PP_ALIGN.CENTER)

cislo_slidu(s16, 16)
print('  [OK] Slide 16: Souhrnná tabulka a doporučení vytvořena')

prs.save(PPTX_PATH)
print(f'\n[VÝBORNĚ] Prezentace úspěšně vygenerována: {PPTX_PATH}')
print(f'          Celkový počet slidů: {len(prs.slides)}')
