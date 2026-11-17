# Semestrální projekt VMM – Střelecká úspěšnost

**Předmět:** Vícerozměrné statistické metody (VMM)  
**VUT Brno** | Akademický rok 2026/27  
**Dataset:** `shots_outcome_Final.csv` – střely z více fotbalových lig (understat.com)

---

## Cíl projektu

Vícerozměrná statistická analýza faktorů střelecké úspěšnosti ve fotbale. Projekt sleduje logický metodický postup od načtení a korektního očištění dat přes diagnostiku předpokladů (jednorozměrná i vícerozměrná normalita, homoskedasticita rozptylů, kovarianční homogenita) až po pokročilé vícerozměrné metody (MANOVA, PCA, faktorová analýza s ortogonální rotací Varimax, shluková analýza a binární logistická regrese).

---

## Struktura složek a výstupů

```
VMM_projekt_03/
├── data/
│   ├── shots_outcome_Final.csv              # Původní surová datová sada střel (1 000 záznamů)
│   └── clean_shots.csv                      # Očištěná data v reálných metrech (735 střel)
├── kody/
│   ├── 01_predzpracovani.py                 # Skript 1: Čištění dat, přepočet na metry, geometrie
│   ├── 02_analyza.py                        # Skript 2: Vícerozměrná analýza a generování grafů/tabulek
│   └── vytvor_prezentaci.py                 # Skript 3: Generátor výsledné PPTX prezentace (15 slidů)
├── materialy/
│   ├── VMM-1.pdf … VMM-11.pdf               # Přednáškové podklady (reference)
│   ├── VMM_Projekty2026.pdf                 # Zadání semestrálního projektu
│   └── *.py, *.ipynb                        # Podklady ze cvičení
├── vystupy/
│   ├── 01_mapa_strel.png                    # Prostorová mapa střel na celém fotbalovém hřišti (105 × 68 m)
│   ├── 01b_hist_xG.png                      # Histogram očekávaných gólů (xG) s pravostrannou asymetrií
│   ├── 01c_kde_hriste.png                   # 2D KDE vrstevnice prostorové hustoty střelby na hřišti (VMM-3/4)
│   ├── 02_normalita_histogramy_qq.png        # Histogram dist_m + normální Q-Q graf (scipy probplot) (1×2)
│   ├── 02b_mahal_chisq_qq.png               # Vícerozměrná normalita: Chí-kvadrát Q-Q graf D² a histogram (VMM-2)
│   ├── 03_korelacni_heatmapa.png            # Plná Spearmanova korelační matice sledovaných veličin
│   ├── 03b_pairplot.png                     # Pair-plot klíčových proměnných s KDE křivkami na diagonále
│   ├── 04_boxploty_shottype.png             # Boxploty vzdálenosti a úhlu dle zakončení (Welchova ANOVA)
│   ├── 04b_tukey_hsd.png                    # Tukeyho HSD post-hoc test (95% intervaly spolehlivosti)
│   ├── 04c_hraci_xG_boxplot.png             # Distribuce xG a taktické pozice klíčových střelců (VMM-5/6)
│   ├── 05b_biplot_pca.png                   # Scree plot + 2D Biplot PCA (osa kvality střelecké pozice)
│   ├── 05c_fa_varimax_loadings.png          # Faktorová analýza FA s rotací Varimax (VMM-9)
│   ├── 06b_mapa_shluku.png                  # Shlukování: Wardův dendrogram a mapa shluků na hřišti (k = 3)
│   ├── 07b_logisticka_krivka.png            # Sigmoidální S-křivka P(gól) vs. dist_m + ROC křivka (1×2)
│   ├── tabulka_popis.csv                    # Popisná statistika sledovaných proměnných (průměr, std, IQR, skew)
│   ├── tabulka_hraci_xG.csv                 # Přehled střelecké efektivity a xG jednotlivých hráčů
│   └── tabulka_hypotezy.csv                 # Souhrnná tabulka testovaných statistických hypotéz
└── prezentace/
    └── prezentace_strelecka_uspesnost.pptx  # Výsledná prezentace (15 slidů, 16:9, čistě bílý styl)
```

---

## Metodický přehled a vazba na přednášky VMM

| Fáze | Přednáška VMM | Aplikovaná metoda | Klíčový výstup / Artefakt |
| :--- | :--- | :--- | :--- |
| **1 – Předzpracování & EDA** | VMM-1, 2 | Ošetření nestandardních textových hodnot (`NA` = bez asistence, `None` = samostatná akce), eliminace vlastních gólů, vyřazení střel bez souřadnic, přepočet na reálné rozměry hřiště $105 \times 68\text{ m}$, výpočet eukleidovské vzdálenosti a zorného úhlu branky ($7{,}32\text{ m}$). | `clean_shots.csv`, `01_mapa_strel.png`, `01b_hist_xG.png` |
| **2 – 2D KDE Hustota** | VMM-3, 4 | Dvourozměrný odhad hustoty jádrovým vyhlazováním (bivariátní KDE vrstevnice na útočné polovině hřiště pro góly i minely). | `01c_kde_hriste.png` |
| **3 – Normalita & Homoskedasticita** | VMM-2, 3 | Shapiro-Wilkův test jednorozměrné normality, empirický Q-Q diagram (`scipy.stats.probplot`), Leveneho test shody rozptylů, Henze-Zirklerův test a Chí-kvadrát Q-Q graf Mahalanobisových vzdáleností $D^2$. | `02_normalita_histogramy_qq.png`, `02b_mahal_chisq_qq.png` |
| **4 – Korelace & Parciální vztahy** | VMM-4 | Spearmanova pořadová korelační matice, parciální korelace (`pingouin.partial_corr`), párové scatter ploty s KDE diagonálou rozlišené podle gólu. | `03_korelacni_heatmapa.png`, `03b_pairplot.png` |
| **5 – ANOVA, MANOVA & Tukey HSD** | VMM-5, 6 | Welchova ANOVA pro heteroskedastická data, Kruskal-Wallisův test, Boxův M test rovnosti kovariančních matic, Wilksova lambda i Pillaiova stopa MANOVA, Tukeyho HSD post-hoc test (95% IS). | `04_boxploty_shottype.png`, `04b_tukey_hsd.png` |
| **6 – Analýza střelců (Hráči)** | VMM-5, 6 | Porovnání distribuce xG a taktických střeleckých pozic mezi hráči (Kruskal-Wallis, Welch ANOVA, scatter vzdálenost vs. xG s konverzním zbarvením). | `04c_hraci_xG_boxplot.png`, `tabulka_hraci_xG.csv` |
| **7 – PCA & Faktorová analýza** | VMM-7, 8, 9 | KMO míra vzorkové adekvátnosti, Bartlettův test sférickosti, Scree plot (Kaiserovo pravidlo), 2D Biplot, Faktorová analýza s ortogonální rotací Varimax pro odhalení skrytých faktorů. | `05b_biplot_pca.png`, `05c_fa_varimax_loadings.png` |
| **8 – Shluková analýza** | VMM-11 | Wardovo aglomerativní hierarchické shlukování (cut $k=3$), K-means shlukování ($k=3$) standardizovaných prostorových proměnných na fotbalovém hřišti. | `06b_mapa_shluku.png` |
| **9 – Logistická regrese & ROC** | VMM-4, 5 | Jednorozměrná sigmoidální křivka šance na gól v závislosti na vzdálenosti, vícerozměrný logit model s typem zakončení, ROC křivka a vyhodnocení diskriminační schopnosti (AUC) ve srovnání s xG. | `07b_logisticka_krivka.png`, `tabulka_hypotezy.csv` |

---

## Struktura prezentace (15 snímků, poměr 16:9)

1. **Slide 1 – Titulní snímek:** Čistý minimalistický design na bílém podkladu (`#FFFFFF`), název práce, autor, předmět VMM, akademický rok.
2. **Slide 2 – Představení datového vzorku a metodického postupu (Čistě textový slide):** Popis vzorku ($n=1\,000$ surových $\to n=735$ vyčištěných střel po vyřazení vlastních gólů a chybějících souřadnic), cíl výzkumu a přehled sledovaných proměnných.
3. **Slide 3 – Prostorová mapa střel na celém fotbalovém hřišti ($105 \times 68\text{ m}$):** Přehled všech 735 střel rozlišených barevně (góly červeně, neproměněné modře) na standardizovaných rozměrech hřiště.
4. **Slide 4 – 2D KDE prostorová hustota střelby na hřišti:** Plynulé vrstevnice jádrového vyhlazení demonstrující kompaktní ohnisko gólů přímo v ose malého vápna oproti širokému rozptylu minel.
5. **Slide 5 – Diagnostika jednorozměrné normality (`dist_m`) a shody rozptylů:** Histogram vzdálenosti s teoretickou křivkou a normální Q-Q graf (`scipy.stats.probplot`), vyhodnocení Shapiro-Wilkova a Leveneho testu.
6. **Slide 6 – Diagnostika vícerozměrné normality (Mahalanobis $D^2$):** Chí-kvadrát Q-Q graf kvantilů Mahalanobisových vzdáleností vektoru `[dist_m, angle_deg, xG]`, Henze-Zirklerův test a detekce odlehlých hodnot.
7. **Slide 7 – Spearmanova korelační analýza a parciální korelace:** Plná pořadová korelační matice a očištění vztahu xG vůči úhlu při zafixované vzdálenosti (`pingouin.partial_corr`).
8. **Slide 8 – Párové rozptylové diagramy a marginální hustoty (Pair-plot):** Multivariátní zobrazení separace gólů a neproměněných pokusů s marginálními KDE křivkami na diagonále.
9. **Slide 9 – Rozdíly geometrie střel dle typu zakončení (Welchova ANOVA):** Boxploty vzdálenosti a úhlu pro pravou nohu, levou nohu a hlavu doplněné Welchovou statistikou.
10. **Slide 10 – Vícerozměrné posouzení sdruženého vektoru (MANOVA & Tukey HSD):** Boxův M test rovnosti kovariancí, Wilksova $\lambda$ i Pillaiova stopa MANOVA a post-hoc párové porovnání s 95% intervaly spolehlivosti.
11. **Slide 11 – Redukce dimenzionality PCA (Scree plot & 2D Biplot):** Bartlettův test sférickosti, scree plot s Kaiserovým pravidlem a 2D biplot s projekcí komponentních zátěží.
12. **Slide 12 – Faktorová analýza s rotací Varimax:** Heatmapa zátěží ortogonálně rotovaných faktorů ($F_1$ = Geometrická kvalita pozice, $F_2$ = Herní hloubka zakončení).
13. **Slide 13 – Shluková analýza (Wardovo členění & K-means na hřišti):** Hierarchický dendrogram a prostorová mapa 3 přirozených taktických zón zakončení ($k=3$: malé vápno, velké vápno, střelba z dálky).
14. **Slide 14 – Logistická regrese (Sigmoidální S-křivka & ROC):** Nelineární model poklesu pravděpodobnosti gólu se vzdáleností, kalibrační body a ROC křivka modelu ($\text{AUC} = 0{,}776$) ve srovnání s xG ($\text{AUC} = 0{,}746$).
15. **Slide 15 – Souhrnná tabulka hypotéz & doporučení pro trenéry:** Přehled ověřených statistických hypotéz a 4 praktická herní doporučení pro trenérskou praxi.

---

## Spuštění projektu

### Požadavky na prostředí
Doporučeno Python 3.10+ s vědeckými knihovnami:
```bash
pip install pandas numpy scipy statsmodels scikit-learn seaborn matplotlib pingouin mplsoccer python-pptx
```

### Postup spuštění ve 3 krocích

#### Krok 1 – Předzpracování a očištění dat
Načte surová data, vyřadí vlastní góly a chybějící souřadnice, přepočte hřiště na reálné metry ($105 \times 68\text{ m}$) a uloží `data/clean_shots.csv`:
```bash
python kody/01_predzpracovani.py
```

#### Krok 2 – Vícerozměrná statistická analýza a generování grafů
Provede kompletní diagnostiku, testování hypotéz a vygeneruje grafické a tabulkové výstupy do složky `vystupy/`:
```bash
python kody/02_analyza.py
```

#### Krok 3 – Vygenerování prezentace
Sestaví 15slidovou prezentaci s vloženými grafy a statistickými verdikty do `prezentace/prezentace_strelecka_uspesnost.pptx`:
```bash
python kody/vytvor_prezentaci.py
```

---

## Hlavní statistické výsledky

1. **Normalita & Homoskedasticita (VMM-2, 3):**
   - Shapiro-Wilkův test prokázal signifikantní odchylku od normálního rozdělení pro vzdálenost `dist_m` ($W = 0{,}983$, $p < 0{,}001$), metriku `xG` ($W = 0{,}543$, $p < 0{,}001$) i střelecký úhel `angle_deg` ($W = 0{,}734$, $p < 0{,}001$).
   - Henze-Zirklerův test zamítl vícerozměrnou normalitu vektoru proměnných ($HZ = 68{,}28$, $p < 0{,}001$).
   - Leveneho test potvrdil heteroskedasticitu rozptylů mezi skupinami zakončení ($L = 14{,}86$, $p < 0{,}001$), což odůvodnilo použití Welchovy ANOVA namísto klasické jednofaktorové ANOVA.

2. **Korelace & Parciální vazby (VMM-4):**
   - Zjištěna silná záporná Spearmanova korelace mezi `xG` a vzdáleností `dist_m` ($r = -0{,}740$, $p < 0{,}001$) a pozitivní vztah s úhlem `angle_deg` ($r = +0{,}644$, $p < 0{,}001$).
   - Po očištění o vliv vzdálenosti zůstává parciální korelace mezi `xG` a střeleckým úhlem statisticky významná ($r_p = +0{,}155$, $p < 0{,}001$).

3. **Srovnání typů zakončení (VMM-5, 6):**
   - Boxův M test potvrdil nerovnost kovariančních matic ($\chi^2 = 132{,}4$, $p < 0{,}001$), proto je v MANOVA preferována robustní Pillaiova stopa.
   - MANOVA prokázala vysoce významný sdružený vliv typu zakončení na geometrii střely (Pillaiova stopa $= 0{,}185$, Wilksova $\lambda = 0{,}815$, obojí $p < 0{,}001$).
   - Tukeyho HSD post-hoc test ukázal, že hlavičkové pokusy vznikají z podstatně kratší vzdálenosti a pod menšími úhly než střely nohou ($p < 0{,}001$). Mezi střelbou pravou a levou nohou nebyl zjištěn statisticky významný geometrický rozdíl.

4. **Redukce dimenzionality PCA & Faktorová analýza (VMM-7, 8, 9):**
   - Bartlettův test sférickosti ($\chi^2 = 1370{,}6$, $p < 0{,}001$) potvrdil vhodnost faktorového zkoumání.
   - První dvě hlavní komponenty PCA kumulativně vysvětlují $84{,}5\,\%$ variability (PC1 $66{,}0\,\%$, PC2 $18{,}5\,\%$); PC1 představuje dominantní osu celkové kvality střelecké pozice.
   - Faktorová analýza s rotací Varimax separovala 2 ortogonální latentní dimenze: $F_1$ (Kvalita střelecké pozice – úhel $+0{,}86$, vzdálenost $-0{,}82$, xG $+0{,}67$) a $F_2$ (Boční odchylka od osy hřiště $+0{,}74$).

5. **Shluková analýza – Herní zóny (VMM-11):**
   - Wardovo hierarchické shlukování a algoritmus K-means ($k = 3$) vymezují 3 odlišné profily zakončení na fotbalovém hřišti:
     - **Shluk 1 (Malé vápno):** Bezprostřední blízkost branky (průměr $8{,}4\text{ m}$), vysoké úhly ($47{,}8^\circ$), $xG = 0{,}40$ a maximální konverze šancí ($36{,}4\,\%$).
     - **Shluk 2 (Velké vápno):** Standardní herní situace v pokutovém území (průměr $15{,}7\text{ m}$), střední pravděpodobnost gólu ($6{,}8\,\%$).
     - **Shluk 3 (Střelba z dálky):** Pokusy ze vzdálenosti $> 20\text{ metrů}$ (průměr $25{,}8\text{ m}$) s minimální efektivitou (konverze $4{,}0\,\%$).

6. **Logistická regrese & Schopnost predikce (VMM-4, 5):**
   - Globální poměr věrohodností LLR test potvrdil statistickou významnost celého logistického modelu ($\chi^2 = 69{,}7$, $p < 0{,}001$).
   - Vzdálenost je klíčovým negativním prediktorem ($\text{OR} = 0{,}918$ na každý metr vzdálenosti, $p = 0{,}016$).
   - Model dosahuje vysoké diskriminační schopnosti ($\text{ROC AUC} = 0{,}776$), která plně snese srovnání a překonává jednorozměrnou predikci xG ($\text{AUC} = 0{,}746$).
