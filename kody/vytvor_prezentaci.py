# -*- coding: utf-8 -*-
"""
=============================================================================
SEMESTRÁLNÍ PROJEKT VMM – STŘELECKÁ ÚSPĚŠNOST (PREMIER LEAGUE)
Skript 3: Generátor prezentace v PowerPointu (pptx)
=============================================================================
Parametry prezentace:
  - Poměr stran 16:9 widescreen (13.33 × 7.5 palců)
  - Čistě bílé pozadí (#FFFFFF) na všech snímcích
  - Slide 1: Titulní snímek
  - Slide 2: Představení dat a předzpracování (ČISTĚ TEXTOVÝ SLIDE S ODRÁŽKAMI)
  - Slide 3 až 14: JEDEN OBŘÍ VYCENTROVANÝ GRAF (šířka 10.5 palců, výška volná)
                   s textovým boxem s hypotézou a fotbalovým závěrem POD grafem
  - Slide 15: Souhrnná tabulka všech 12 hypotéz a doporučení pro trenéry
=============================================================================
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VYSTUPY_DIR = os.path.join(BASE_DIR, 'vystupy')
PPTX_PATH = os.path.join(BASE_DIR, 'prezentace', 'prezentace_strelecka_uspesnost.pptx')
os.makedirs(os.path.join(BASE_DIR, 'prezentace'), exist_ok=True)

# Barevná paleta
BILA         = RGBColor(0xFF, 0xFF, 0xFF)   # #FFFFFF
CERNA        = RGBColor(0x1A, 0x1A, 0x1A)   # #1A1A1A
TMAVA_SEDA   = RGBColor(0x33, 0x33, 0x33)   # #333333
STRED_SEDA   = RGBColor(0x66, 0x66, 0x66)   # #666666
SVETLA_SEDA  = RGBColor(0xD8, 0xD8, 0xD8)   # #D8D8D8
CERVENA_ZAM  = RGBColor(0xC5, 0x30, 0x30)   # #C53030
ZELENA_NEZAM = RGBColor(0x27, 0x67, 0x49)   # #276749
MODRA_BOD    = RGBColor(0x1F, 0x77, 0xB4)   # #1F77B4

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.50)
BLANK = prs.slide_layouts[6]
TOTAL_SLIDES = 15


def novy_slide():
    """Vytvoří nový prázdný snímek s bílým pozadím (#FFFFFF)."""
    sl = prs.slides.add_slide(BLANK)
    bg = sl.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BILA
    return sl


def text_pole(slide, text, l, t, w, h,
              font_size=13, bold=False, italic=False,
              fg=TMAVA_SEDA, align=PP_ALIGN.LEFT, wrap=True):
    """Vloží textové pole s požadovaným stylem."""
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
    """Vytvoří standardní čisté záhlaví."""
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
    """
    Vloží obří vycentrovaný graf s pevnou šířkou (10.5 palce) bez zadání výšky,
    aby byl striktně zachován původní poměr stran.
    """
    cesta = os.path.join(VYSTUPY_DIR, soubor)
    if not os.path.exists(cesta):
        print(f'  [VAROVÁNÍ] Soubor grafu nenalezen: {soubor}')
        return None
    l_palce = (13.33 - w_palce) / 2.0  # Centrování: 1.415 palce
    return slide.shapes.add_picture(cesta, Inches(l_palce), Inches(t_palce), width=Inches(w_palce))


def spodni_box_hypoteza_zaver(slide, test_verdikt, fotbal_zaver, is_zamitnuto=True):
    """
    Vloží strukturovaný textový blok POD obří graf:
    1. řádek: Statistický test a verdikt hypotézy
    2. řádek: Fotbalový a věcný závěr pro trenéry
    """
    l, t, w, h = 0.8, 6.26, 11.73, 0.84

    # Karta
    sh = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor(0xF9, 0xFA, 0xFB)
    sh.line.color.rgb = SVETLA_SEDA
    sh.line.width = Pt(1)

    # Akcentní proužek
    sh_str = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(0.08), Inches(h))
    sh_str.fill.solid()
    sh_str.fill.fore_color.rgb = MODRA_BOD
    sh_str.line.fill.background()

    txb = slide.shapes.add_textbox(Inches(l + 0.18), Inches(t + 0.04), Inches(w - 0.30), Inches(h - 0.08))
    txb.word_wrap = True
    tf = txb.text_frame
    tf.word_wrap = True

    # 1. Řádek: Statistický verdikt
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

    # 2. Řádek: Fotbalový závěr
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


print('=' * 65)
print('GENEROVÁNÍ POWERPOINT PREZENTACE (16:9, BÍLÉ POZADÍ)')
print('=' * 65)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 – TITULNÍ SNÍMEK
# ════════════════════════════════════════════════════════════════════════════
s1 = novy_slide()

text_pole(s1, 'Střelecká úspěšnost ve fotbale',
          0.8, 1.8, 11.73, 1.1,
          font_size=36, bold=True, fg=CERNA, align=PP_ALIGN.CENTER)

text_pole(s1, 'Vícerozměrná statistická analýza dat střel z anglické Premier League',
          0.8, 2.85, 11.73, 0.6,
          font_size=16, italic=True, fg=STRED_SEDA, align=PP_ALIGN.CENTER)

linka(s1, 3.0, 3.65, 7.33, tloustka=0.02, barva=MODRA_BOD)

text_pole(s1, 'Vysoké učení technické v Brně  •  Vícerozměrné statistické metody (VMM)',
          0.8, 3.95, 11.73, 0.45,
          font_size=14, bold=True, fg=TMAVA_SEDA, align=PP_ALIGN.CENTER)

text_pole(s1, 'Semestrální projekt  •  Akademický rok 2026/27',
          0.8, 4.45, 11.73, 0.4,
          font_size=12, fg=STRED_SEDA, align=PP_ALIGN.CENTER)

text_pole(s1, 'Zdroj dat: shots_outcome_Final.csv (databáze understat.com)',
          0.8, 6.75, 11.73, 0.35,
          font_size=10, fg=STRED_SEDA, align=PP_ALIGN.CENTER)

cislo_slidu(s1, 1)
print('  [OK] Slide 1: Titulní snímek vytvořen')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 – PŘEDSTAVENÍ DAT A PŘEDZPRACOVÁNÍ (ČISTĚ TEXTOVÝ SLIDE S ODRÁŽKAMI)
# ════════════════════════════════════════════════════════════════════════════
s2 = novy_slide()
zahlavi(s2, 'Představení datového vzorku a metodického postupu',
        'Základní popis analyzované datové sady z Premier League, odvození proměnných a vymezení cílů')

# Karta 1: Datový vzorek a čištění
sh1 = s2.shapes.add_shape(1, Inches(0.8), Inches(1.15), Inches(5.7), Inches(5.6))
sh1.fill.solid(); sh1.fill.fore_color.rgb = RGBColor(0xF9, 0xFA, 0xFB)
sh1.line.color.rgb = SVETLA_SEDA; sh1.line.width = Pt(1)
sh1_str = s2.shapes.add_shape(1, Inches(0.8), Inches(1.15), Inches(0.08), Inches(5.6))
sh1_str.fill.solid(); sh1_str.fill.fore_color.rgb = MODRA_BOD; sh1_str.line.fill.background()

text_pole(s2, 'Datový vzorek & Předzpracování', 1.05, 1.28, 5.3, 0.35, font_size=14, bold=True, fg=CERNA)
text_pole(s2,
    '• Datový zdroj: 1 000 střel z anglické Premier League (understat.com).\n\n'
    '• Eliminace nerelevantních střel: Vyřazeny vlastní góly (OwnGoal, n = 2), které nereprezentují střelecký záměr útočníka.\n\n'
    '• Ošetření chybějících souřadnic: Vyřazeno 263 střel bez údajů o poloze; souřadnice neimputujeme průměrem, aby nevznikly artefakty v ose hřiště.\n\n'
    '• Pravidlo 3-sigma: Aplikováno na vzdálenost dist_m (|z| < 3), vyřazena 1 extrémní střela z vlastní poloviny hřiště.\n\n'
    '• Finální analyzovaný vzorek: n = 734 střel, z toho 62 gólů (míra konverze 8,4 %).\n\n'
    '• Cílová proměnná: goal ∈ {0, 1} (1 = Gól, 0 = Zákrok, Mimo, Blok, Tyč).',
    1.05, 1.70, 5.3, 4.85, font_size=12, fg=TMAVA_SEDA)

# Karta 2: Sledované veličiny a cíl projektu
sh2 = s2.shapes.add_shape(1, Inches(6.8), Inches(1.15), Inches(5.7), Inches(5.6))
sh2.fill.solid(); sh2.fill.fore_color.rgb = RGBColor(0xF9, 0xFA, 0xFB)
sh2.line.color.rgb = SVETLA_SEDA; sh2.line.width = Pt(1)
sh2_str = s2.shapes.add_shape(1, Inches(6.8), Inches(1.15), Inches(0.08), Inches(5.6))
sh2_str.fill.solid(); sh2_str.fill.fore_color.rgb = MODRA_BOD; sh2_str.line.fill.background()

text_pole(s2, 'Sledované proměnné & Cíle práce', 7.05, 1.28, 5.3, 0.35, font_size=14, bold=True, fg=CERNA)
text_pole(s2,
    '• X, Y: Normalizované prostorové souřadnice zakončení na jednotném hřišti 100 × 100 % délky a šířky.\n\n'
    '• dist_m: Eukleidovská vzdálenost od středu soupeřovy branky [X=100, Y=50] v metrech (rozsah 2,7 až 41,7 m).\n\n'
    '• angle_deg: Střelecký zorný úhel sevřený brankovými tyčemi (rozsah 4,0° až 121,3°).\n\n'
    '• xG: Expected Goals metrika – pravděpodobnost gólu dle analytického modelu (rozsah 0,01 až 0,76).\n\n'
    '• shotType: Typ zakončení – Pravá noha (n = 377), Levá noha (n = 227), Hlava (n = 101).\n\n'
    '• Výzkumný cíl: Vícerozměrně ověřit vliv prostorové geometrie na úspěšnost zakončení a formulovat praktická doporučení pro trenéry.',
    7.05, 1.70, 5.3, 4.85, font_size=12, fg=TMAVA_SEDA)

cislo_slidu(s2, 2)
print('  [OK] Slide 2: Představení dat vytvořeno (čistě textový slide)')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 – MAPA STŘEL NA CELÉM HŘIŠTI
# ════════════════════════════════════════════════════════════════════════════
s3 = novy_slide()
zahlavi(s3, 'Prostorová mapa střel na celém fotbalovém hřišti (100 × 100)',
        'Zobrazení kompletního souboru střel po aplikaci 3-sigma pravidla na vzdálenost dist_m')
vloz_obri_graf(s3, '01_mapa_strel.png')
spodni_box_hypoteza_zaver(s3,
    'Datový soubor: n = 734 střel | Gól: 62 (8,4 %) | Z-skóre dist_m < 3 vyřadilo 1 odlehlou hodnotu',
    'Góly (červené body) jsou ostře koncentrovány v centrálním prostoru před brankou; zakončení z velké dálky či ostrých úhlů mají minimální konverzi.',
    is_zamitnuto=False)
cislo_slidu(s3, 3)
print('  [OK] Slide 3: Mapa střel vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 – 2D KDE HUSTOTA STŘELBY NA HŘIŠTI
# ════════════════════════════════════════════════════════════════════════════
s4 = novy_slide()
zahlavi(s4, '2D KDE hustota střelby na hřišti (Vrstevnice zakončení)',
        'Plynulý dvourozměrný jádrový odhad hustoty pravděpodobnosti střel (Přednáška VMM-3 / VMM-4)')
vloz_obri_graf(s4, '01c_kde_hriste.png')
spodni_box_hypoteza_zaver(s4,
    '2D jádrový odhad hustoty (Bivariate KDE) | 8 hladin vrstevnic pro góly i minely',
    'Ohnisko úspěšných zakončení vykazuje extrémně kompaktní jádro přímo v ose malého vápna (X > 88, Y ∈ [42, 58]), zatímco minely mají široký rozptyl.',
    is_zamitnuto=False)
cislo_slidu(s4, 4)
print('  [OK] Slide 4: 2D KDE vytvořeno')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 – DIAGNOSTIKA JEDNOROZMĚRNÉ NORMALITY (dist_m)
# ════════════════════════════════════════════════════════════════════════════
s5 = novy_slide()
zahlavi(s5, 'Diagnostika jednorozměrné normality (dist_m) & shody rozptylů',
        'Histogram vzdálenosti s proloženou normální hustotou a jednorozměrný Q-Q diagram kvantilů')
vloz_obri_graf(s5, '02_normalita_histogramy_qq.png')
spodni_box_hypoteza_zaver(s5,
    'Shapiro-Wilk: W = 0.9836, p < 0.001 -> H₀ ZAMÍTNUTA | Levene: W = 10.997, p < 0.001 -> H₀ ZAMÍTNUTA',
    'Vzdálenost střel nevykazuje normální rozdělení a rozptyly mezi skupinami zakončení jsou heteroskedastické. Je nutné využít robustní a neparametrické testy.',
    is_zamitnuto=True)
cislo_slidu(s5, 5)
print('  [OK] Slide 5: Jednorozměrná normalita vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 – DIAGNOSTIKA VÍCEROZMĚRNÉ NORMALITY: CHÍ-KVADRÁT Q-Q (VMM-2)
# ════════════════════════════════════════════════════════════════════════════
s6 = novy_slide()
zahlavi(s6, 'Diagnostika vícerozměrné normality: Chí-kvadrát Q-Q graf (VMM-2)',
        'Mahalanobisova vzdálenost D² vektoru [dist_m, angle_deg, xG] vs. teoretické kvantily χ²(3)')
vloz_obri_graf(s6, '02b_mahal_chisq_qq.png')
spodni_box_hypoteza_zaver(s6,
    'Henze-Zirkler test: HZ = 53.10, p < 0.001 -> H₀ ZAMÍTNUTA | Detekováno 12 vícerozměrných outlierů nad 99% prahem',
    'Sdružené vícerozměrné rozdělení prostorových metrik vykazuje silnou odchylku od multinormality (VMM-2, slidy 46–47); detekované outliery odpovídají kuriózním střelám z půlky.',
    is_zamitnuto=True)
cislo_slidu(s6, 6)
print('  [OK] Slide 6: Vícerozměrná normalita Mahalanobis vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 – SPEARMANOVA KORELACE & PARCIÁLNÍ KORELACE
# ════════════════════════════════════════════════════════════════════════════
s7 = novy_slide()
zahlavi(s7, 'Spearmanova korelační analýza & parciální korelace (pingouin)',
        'Pořadová korelační matice a očištění vztahu xG a střeleckého úhlu o vliv vzdálenosti')
vloz_obri_graf(s7, '03_korelacni_heatmapa.png')
spodni_box_hypoteza_zaver(s7,
    'Spearman r(xG, dist_m) = -0.732, p < 0.001 -> H₀ ZAMÍTNUTA | Parciální r_part(xG, angle_deg | dist_m) = -0.196, p < 0.001',
    'Vzdálenost je dominantním negativním faktorem šance na gól. Po fixování vzdálenosti vykazuje úhel specifický vztah daný střelami z brankové čáry.',
    is_zamitnuto=True)
cislo_slidu(s7, 7)
print('  [OK] Slide 7: Korelační matice a parciální korelace vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 – PAIR-PLOT VZTAHŮ KLÍČOVÝCH PROMĚNNÝCH
# ════════════════════════════════════════════════════════════════════════════
s8 = novy_slide()
zahlavi(s8, 'Párové rozptylové diagramy a marginální hustoty (KDE)',
        'Multivariátní zkoumání separace střel dle výsledku (Gól vs. Nezakončeno)')
vloz_obri_graf(s8, '03b_pairplot.png')
spodni_box_hypoteza_zaver(s8,
    'Párová diskriminace klíčových metrik | Diagonální marginální KDE separace gólů (červená)',
    'Góly jsou jednoznačně koncentrovány v oblasti dist_m < 15 m a současně angle_deg > 30°. Mimo tuto zónu se vyskytují takřka výhradně neúspěšné pokusy.',
    is_zamitnuto=False)
cislo_slidu(s8, 8)
print('  [OK] Slide 8: Pair-plot vytvořen')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 – ANOVA: ROZDÍLY DLE TYPU ZAKONČENÍ
# ════════════════════════════════════════════════════════════════════════════
s9 = novy_slide()
zahlavi(s9, 'Rozdíly geometrie střel dle typu zakončení (Welchova ANOVA)',
        'Porovnání vzdálenosti a úhlu pro pravou nohu, levou nohu a hlavu')
vloz_obri_graf(s9, '04_boxploty_shottype.png')
spodni_box_hypoteza_zaver(s9,
    'Welchova ANOVA dist_m: F = 63.88, p < 0.001 | Welchova ANOVA angle_deg: F = 58.67, p < 0.001 -> H₀ ZAMÍTNUTA',
    'Hlavičkové pokusy vznikají z diametrálně odlišných prostorových pozic než střelba nohou – z poloviční vzdálenosti a podstatně ostřejších úhlů.',
    is_zamitnuto=True)
cislo_slidu(s9, 9)
print('  [OK] Slide 9: ANOVA vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 – MANOVA, BOXŮV M TEST A TUKEY HSD
# ════════════════════════════════════════════════════════════════════════════
s10 = novy_slide()
zahlavi(s10, 'Vícerozměrné posouzení sdruženého vektoru (MANOVA & Tukey HSD)',
        'Sdružený vektor geometrie [dist_m, angle_deg], Boxův M test a post-hoc 95% intervaly')
vloz_obri_graf(s10, '04b_tukey_hsd.png')
spodni_box_hypoteza_zaver(s10,
    'Boxův M: χ² = 125.3, p < 0.001 | MANOVA Pillai = 0.195, Wilks λ = 0.806, p < 0.001 -> H₀ ZAMÍTNUTA',
    'Z důvodu porušení rovnosti kovariancí (Box M) preferujeme Pillaiovu stopu. Hlava se statisticky významně liší od obou nohou (p < 0.001), nohy jsou shodné.',
    is_zamitnuto=True)
cislo_slidu(s10, 10)
print('  [OK] Slide 10: MANOVA & Tukey vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 – REDUKCE DIMENZIONALITY PCA
# ════════════════════════════════════════════════════════════════════════════
s11 = novy_slide()
zahlavi(s11, 'Redukce dimenzionality PCA (Scree plot & 2D Biplot)',
        'Dekompozice variability, Bartlettův test a projekce střel do roviny hlavních komponent')
vloz_obri_graf(s11, '05b_biplot_pca.png')
spodni_box_hypoteza_zaver(s11,
    'Bartlettův test sférickosti: χ² = 2041.4, p < 0.001 -> H₀ ZAMÍTNUTA | První 2 komponenty vysvětlují ~68 % rozptylu',
    'PC1 (48 % rozptylu) představuje dominantní osu kvality střelecké pozice (protichůdné působení vzdálenosti vs. úhlu a xG). PCA efektivně redukuje prostor.',
    is_zamitnuto=True)
cislo_slidu(s11, 11)
print('  [OK] Slide 11: PCA vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 – FAKTOROVÁ ANALÝZA S ROTACÍ VARIMAX
# ════════════════════════════════════════════════════════════════════════════
s12 = novy_slide()
zahlavi(s12, 'Faktorová analýza s rotací Varimax (Latentní faktory)',
        'Identifikace skrytých faktorů zakončení ortogonální rotací rozptylu zátěží (VMM-9)')
vloz_obri_graf(s12, '05c_fa_varimax_loadings.png')
spodni_box_hypoteza_zaver(s12,
    'Faktorová analýza (n_components = 2, rotace Varimax) | Dosažena jednoduchá ortogonální struktura',
    'Rotace Varimax separovala 2 nezávislé faktory: Faktor 1 (Čistá geometrie zakončení – zátěž úhlu 0.97 a xG 0.57) a Faktor 2 (Herní hloubka – zátěž osy X 0.96).',
    is_zamitnuto=False)
cislo_slidu(s12, 12)
print('  [OK] Slide 12: Faktorová analýza vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13 – SHLUKOVÁ ANALÝZA (WARD & K-MEANS)
# ════════════════════════════════════════════════════════════════════════════
s13 = novy_slide()
zahlavi(s13, 'Shluková analýza: Wardovo členění & K-means na hřišti',
        'Hierarchický dendrogram a prostorová klasifikace střel do přirozených klastrů (VMM-11)')
vloz_obri_graf(s13, '06b_mapa_shluku.png')
spodni_box_hypoteza_zaver(s13,
    'Wardovo hierarchické shlukování (cut k = 2) | K-means (k = 2) na standardizovaných proměnných',
    'Algoritmus rozděluje střely na 2 přirozené herní shluky: Shluk 1 (nebezpečné vápno: prům. dist 10 m, xG = 0.34) vs. Shluk 2 (neefektivní dálka: prům. dist 23 m, xG = 0.05).',
    is_zamitnuto=True)
cislo_slidu(s13, 13)
print('  [OK] Slide 13: Shluková analýza vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14 – LOGISTICKÁ REGRESE (SIGMOIDÁLNÍ KŘIVKA & ROC)
# ════════════════════════════════════════════════════════════════════════════
s14 = novy_slide()
zahlavi(s14, 'Logistická regrese: Sigmoidální S-křivka pravděpodobnosti gólu & ROC',
        'Modelování šance na gól se skutečnými body střel, kontrola multikolinearity VIF a diskriminace AUC')
vloz_obri_graf(s14, '07b_logisticka_sigmoida.png')
spodni_box_hypoteza_zaver(s14,
    'Globální LLR test: χ² = 85.3, p < 0.001 -> H₀ ZAMÍTNUTA | AUC = 0.794 | Kontrola VIF < 5.7',
    'S-křivka ukazuje prudký nelineární propad šance: z 10 m je úspěšnost ~25 %, na 25 m padá pod 3 % (OR = 0.922 na metr, p < 0.05). Zakončení nohou má vyšší OR.',
    is_zamitnuto=True)
cislo_slidu(s14, 14)
print('  [OK] Slide 14: Logistická regrese vytvořena')


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15 – SOUHRNNÁ TABULKA VŠECH HYPOTÉZ & DOPORUČENÍ PRO TRENÉRY
# ════════════════════════════════════════════════════════════════════════════
s15 = novy_slide()
zahlavi(s15, 'Souhrnná tabulka testovaných hypotéz & doporučení pro trenéry',
        'Komplexní verifikace 12 statistických hypotéz projektu a klíčová taktická doporučení')

# Horní část: Přehledná tabulka všech 12 hypotéz
tabulka_hyp_data = [
    ('1 – Normalita 1D', 'Shapiro-Wilk (dist_m)', 'Vzdálenost dist_m pochází z normálního rozdělení', 'W = 0.9836', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('1 – Normalita 1D', 'Shapiro-Wilk (xG)', 'Metrika xG pochází z normálního rozdělení', 'W = 0.5433', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('1 – Normalita MV', 'Henze-Zirkler / Chí-kv.', 'Vektor [dist_m, angle, xG] má vícerozměrnou normalitu', 'HZ = 53.10', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('2 – Rozptyly', 'Leveneho test', 'Rozptyly vzdálenosti jsou mezi skupinami zakončení shodné', 'L = 10.997', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('3 – Kovariance', 'Boxův M test', 'Kovarianční matice jsou mezi skupinami zakončení shodné', 'χ² = 125.3', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('4 – Závislost', 'Spearmanův test', 'Korelace mezi xG a dist_m je nulová (r = 0)', 'r = -0.732', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('4 – Parciální kor.', 'Pingouin partial_corr', 'Korelace xG a angle_deg při fixování dist_m je nulová', 'r_p = -0.196', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('5 – Rozdíly ANOVA', 'Welch ANOVA (dist_m)', 'Střední vzdálenosti jsou pro typy zakončení stejné', 'F = 63.88', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('6 – Sdružené MANOVA', 'Pillaiova stopa / Wilks λ', 'Vektory středních hodnot jsou ve skupinách stejné', 'Pillai = 0.195', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('7 – PCA vhodnost', 'Bartlettův test', 'Korelační matice je jednotková (PCA nevhodná)', 'χ² = 2041.4', '< 0.001', 'ZAMÍTÁME H₀', True),
    ('8 – Multikol.', 'Test VIF', 'Prediktory netrpí škodlivou multikolinearitou (VIF < 5)', 'max VIF = 5.66', 'VIF < 10', 'NEZAMÍTÁME H₀', False),
    ('9 – Logit regrese', 'Logit model (dist_m)', 'Vzdálenost střely neovlivňuje šanci na gól (β = 0)', 'OR = 0.922', '< 0.05', 'ZAMÍTÁME H₀', True),
]

SLOUPCE_X = [0.8, 2.2, 4.3, 8.2, 9.7, 10.9]
SLOUPCE_W = [1.3, 2.0, 3.8, 1.4, 1.1, 1.63]
HLAVICKY  = ['Fáze', 'Metoda', 'Nulová hypotéza (H₀)', 'Statistika', 'p-hodnota', 'Rozhodnutí']

for j, (h_txt, w_col) in enumerate(zip(HLAVICKY, SLOUPCE_W)):
    sh = s15.shapes.add_shape(1, Inches(SLOUPCE_X[j]), Inches(1.05), Inches(w_col), Inches(0.32))
    sh.fill.solid(); sh.fill.fore_color.rgb = CERNA; sh.line.fill.background()
    text_pole(s15, h_txt, SLOUPCE_X[j] + 0.05, 1.08, w_col - 0.1, 0.26,
              font_size=9.5, bold=True, fg=BILA, align=PP_ALIGN.CENTER)

for i, radek in enumerate(tabulka_hyp_data):
    y_pos = 1.40 + (i * 0.28)
    bg_barva = RGBColor(0xFA, 0xFA, 0xFA) if (i % 2 == 0) else BILA
    is_zam = radek[6]

    for j in range(6):
        val = radek[j]
        w_col = SLOUPCE_W[j]
        sh = s15.shapes.add_shape(1, Inches(SLOUPCE_X[j]), Inches(y_pos), Inches(w_col), Inches(0.26))
        sh.fill.solid(); sh.fill.fore_color.rgb = bg_barva
        sh.line.color.rgb = RGBColor(0xEE, 0xEE, 0xEE); sh.line.width = Pt(0.5)

        is_verdict = (j == 5)
        text_fg = (CERVENA_ZAM if is_zam else ZELENA_NEZAM) if is_verdict else TMAVA_SEDA
        align = PP_ALIGN.CENTER if j in [0, 3, 4, 5] else PP_ALIGN.LEFT
        text_pole(s15, val, SLOUPCE_X[j] + 0.05, y_pos + 0.02, w_col - 0.1, 0.22,
                  font_size=8.5, bold=is_verdict, fg=text_fg, align=align)

# Dolní část: 4 doporučení pro trenéry ve 2 sloupcích
doporuceni = [
    ('1. Priorita hloubky vápna', 'Střelba z vápna (< 12 m) má 6× vyšší konverzi než z dálky; klíčem je trpělivá finální přihrávka do pokutového území.'),
    ('2. Maximalizace úhlu zakončení', 'Větší střelecký zorný úhel násobně zvyšuje xG; útočníci mají preferovat zakončení z centrální osy před střelbou z ostrých úhlů.'),
    ('3. Specifický nácvik hlaviček', 'Hlavičky vznikají ze zhoršených úhlů a větších vzdáleností; centry musí cíleně směřovat do zóny malého vápna.'),
    ('4. Taktická eliminace střel z dálky', 'Střelba ze vzdálenosti > 20 metrů vykazuje šanci pod 3 %; v tréninku je nutné potlačit ukvapené pokusy ze střední vzdálenosti.'),
]

for idx_d, (titul_d, text_d) in enumerate(doporuceni):
    col_x = 0.8 if idx_d % 2 == 0 else 6.8
    row_y = 5.05 if idx_d < 2 else 5.95
    w_card = 5.73

    sh_d = s15.shapes.add_shape(1, Inches(col_x), Inches(row_y), Inches(w_card), Inches(0.82))
    sh_d.fill.solid(); sh_d.fill.fore_color.rgb = RGBColor(0xF9, 0xFA, 0xFB)
    sh_d.line.color.rgb = SVETLA_SEDA; sh_d.line.width = Pt(1)

    sh_str = s15.shapes.add_shape(1, Inches(col_x), Inches(row_y), Inches(0.08), Inches(0.82))
    sh_str.fill.solid(); sh_str.fill.fore_color.rgb = MODRA_BOD; sh_str.line.fill.background()

    text_pole(s15, titul_d, col_x + 0.18, row_y + 0.06, w_card - 0.3, 0.25, font_size=11, bold=True, fg=CERNA)
    text_pole(s15, text_d, col_x + 0.18, row_y + 0.32, w_card - 0.3, 0.46, font_size=9.5, fg=TMAVA_SEDA)

text_pole(s15, 'Semestrální projekt VMM 2026/27  •  VUT Brno  •  Děkuji za pozornost',
          0.8, 6.95, 11.73, 0.30, font_size=10, fg=STRED_SEDA, align=PP_ALIGN.CENTER)

cislo_slidu(s15, 15)
print('  [OK] Slide 15: Souhrnná tabulka a doporučení vytvořena')

# Uložení prezentace
prs.save(PPTX_PATH)
print(f'\n[VÝBORNĚ] Prezentace úspěšně vygenerována: {PPTX_PATH}')
print(f'          Celkový počet slidů: {len(prs.slides)}')
