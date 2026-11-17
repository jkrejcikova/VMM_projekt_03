# Semestrální projekt VMM – Vícerozměrná analýza střelecké úspěšnosti v Premier League

**Předmět:** Vícerozměrné statistické metody (VMM)  
**Instituce:** Vysoké učení technické v Brně (VUT) | Fakulta elektrotechniky a komunikačních technologií  
**Akademický rok:** 2026/2027  
**Zdrojová data:** `shots_outcome_Final.csv` – 1 000 střel z anglické Premier League (understat.com)  
**Finální analyzovaný vzorek:** $n = 735$ střel s kompletní prostorovou geometrií, 63 gólů (konverze $8,57\,\%$)

---

## 1. Úvod a cíl práce

Cílem projektu je komplexní vícerozměrná statistická analýza faktorů determinujících úspěšnost fotbalového zakončení. Práce sleduje rigorózní metodologický postup od transparentního předzpracování a kontroly předpokladů (normalita, homoskedasticita, homogenita kovariancí, multikolinearita) přes vícerozměrné srovnávací a redukční metody (Welchova ANOVA, Kruskal-Wallis, MANOVA, PCA, faktorová analýza s rotací Varimax, shluková analýza) až po multivariabilní prediktivní modelování (binární logistická regrese bez datového úniku).

Každá aplikovaná metoda v projektu má své striktní matematické opodstatnění, navazuje na přednáškový cyklus VMM a je doplněna věcnou interpretací pro sportovní analytiku a trenérskou praxi.

---

## 2. Architektura projektu a souborová struktura

```
VMM_projekt_03/
├── data/
│   ├── shots_outcome_Final.csv        # Původní surová data Premier League (1 000 záznamů)
│   ├── clean_shots.csv                # Očištěný analyzovaný vzorek (735 střel, 63 gólů)
│   ├── shots_missing_xy.csv           # Transparentně vyřazené střely bez souřadnic (263 střel)
│   ├── srovnani_chybejicich.csv       # Komparační tabulka pro ověření neutrality vyřazení
│   └── shrnuti_predzpracovani.txt     # Auditní protokol čištění dat
├── kody/
│   ├── 01_predzpracovani.py           # Čištění dat, eliminace chyb, výpočet geometrie na 105 × 68 m
│   ├── 02_analyza.py                  # Hlavní analytický skript (Fáze 1–8, dynamické testování hypotéz)
│   └── vytvor_prezentaci.py            # Generátor PPTX prezentace (15 slidů, 16:9, čistě bílý styl)
├── materialy/
│   ├── VMM-1.pdf … VMM-11.pdf         # Přednáškové podklady z předmětu VMM
│   └── VMM_Projekty2026.pdf           # Zadání semestrálního projektu
├── vystupy/
│   ├── 01_mapa_strel.png              # Prostorová mapa střel na fotbalovém hřišti (105 × 68 m)
│   ├── 01b_hist_xG.png                # Histogram očekávaných gólů (xG) s pravostrannou asymetrií
│   ├── 01c_kde_hriste.png             # 2D KDE vrstevnice prostorové hustoty střelby na hřišti
│   ├── 02_normalita_histogramy_qq.png  # Histogram dist_m + normální Q-Q graf (1×2)
│   ├── 02b_mahal_chisq_qq.png         # Mahalanobis D² vs. teoretické kvantily χ²(3) + outliery
│   ├── 03_korelacni_heatmapa.png      # Spearmanova korelační matice s parciální korelací
│   ├── 03b_pairplot.png               # Pair-plot klíčových proměnných s marginální KDE dle gólu
│   ├── 04_boxploty_shottype.png       # Boxploty vzdálenosti a úhlu dle typu zakončení
│   ├── 04b_tukey_hsd.png              # Tukeyho HSD post-hoc test (95% IS) a Wilksova lambda MANOVA
│   ├── 04c_hraci_xG_boxplot.png       # Dvoupanelový graf: Boxploty xG dle hráčů a taktická mapa pozic
│   ├── 05_scree_plot.png              # Scree plot PCA s Kaiserovým pravidlem (λ = 1)
│   ├── 05b_biplot_pca.png             # Scree plot + 2D Biplot (osa kvality a boční excentricita)
│   ├── 05c_fa_varimax_loadings.png    # Faktorová analýza s ortogonální rotací Varimax
│   ├── 06_dendrogram.png              # Validace počtu shluků (Elbow metoda & Silhouette score)
│   ├── 06b_mapa_shluku.png            # Wardův dendrogram (řez na k=3) a prostorová mapa 3 zón
│   ├── 07b_logisticka_krivka.png      # S-křivka pravděpodobnosti gólu a ROC srovnání s benchmarkem xG
│   ├── tabulka_popis.csv              # Popisná statistika sledovaných proměnných
│   ├── tabulka_hraci_xG.csv           # Podrobná statistika střelců (n, góly, konverze, medián a průměr xG)
│   ├── tabulka_hypotezy.csv           # Souhrnná tabulka VŠECH 16 dynamicky testovaných hypotéz
│   ├── tabulka_logit_koeficienty.csv  # Odhady koeficientů, OR, 95% CI a p-hodnoty logistického modelu
│   └── tabulka_vif.csv                # Diagnostická tabulka multikolinearity prediktorů (VIF)
└── prezentace/
    └── prezentace_strelecka_uspesnost.pptx # Výsledná prezentace k obhajobě (16 slidů, 16:9)
```

---

## 3. Detailní metodologický průvodce: Použité testy, motivace a výsledky

Tato sekce detailně objasňuje každou statistickou metodu v logické posloupnosti projektu: **proč byla použita**, **co přesně testuje ($H_0$ vs. $H_1$)**, **jaký výsledek byl spočten** a **jaký je věcný závěr**.

---

### Fáze 0: Příprava dat a prostorová geometrie (`01_predzpracovani.py`)

1. **Eliminace vlastních gólů ($n = 2$):** Vlastní góly (`OwnGoal`) nereprezentují střelecký záměr útočníka a mají nelogické souřadnice u vlastní branky ($X \approx 0$). Byly vyřazeny.
2. **Ošetření chybějících hodnot X/Y ($n = 263$):** Namísto netransparentního smazání či chybného imputování průměrem (které by vytvořilo umělý shluk bodů uprostřed hřiště) byly tyto záznamy odděleny do `shots_missing_xy.csv`. Byla sestavena komparační tabulka `srovnani_chybejicich.csv`, která prokázala, že vyřazené střely mají srovnatelnou konverzi ($10,3\,\%$ vs. $8,6\,\%$, $p = 0,41$) i průměrné $xG$ ($0,096$ vs. $0,090$), čímž bylo vyloučeno selekční zkreslení.
3. **Zachování věcných kategorií:** Řetězec `'NA'` u asistencí znamená *„bez asistence“* a `'None'` u předchozí akce znamená *„samostatná akce“*. Skript zajistil, že je pandas neinterpretoval jako chybějící `NaN`.
4. **Reálná geometrie hřiště 105 × 68 m:** Souřadnice byly převedeny na metry s brankou o šířce $7,32\text{ m}$ v ose $y = 34\text{ m}$ (tyče na $y \in [30,34; 37,66]\text{ m}$).
   - **Vzdálenost (`dist_m`):** Eukleidovská vzdálenost od středu brankové čáry $[105, 34]$.
   - **Střelecký zorný úhel (`angle_deg`):** Skutečný úhel sevřený brankovými tyčemi odvozený z kosinové věty:
     $$\cos(\theta) = \frac{d_1^2 + d_2^2 - w^2}{2 d_1 d_2}$$
     kde $d_1, d_2$ jsou vzdálenosti k tyčím a $w = 7,32\text{ m}$.
5. **Zrušení naivního 3-sigma pravidla:** Platné góly z dálky nebyly uměle smazány, data reprezentují reálné herní situace.
6. **Komparativní odhad prostorové hustoty (2D KDE na dvou půlkách hřiště):** V grafu `01c_kde_hriste.png` je vykreslen dvourozměrný jádrový odhad hustoty pravděpodobnosti (KDE) na dvou útočných polovinách hřiště vedle sebe (1×2 subplots). Levá polovina zobrazuje prostorový rozptyl neproměněných střel ($n = 672$, modrý gradient s vrcholem na hranici vápna), zatímco pravá polovina zachycuje prostorovou koncentraci vstřelených gólů ($n = 63$, žluto-červený termální gradient lokalizovaný v malém vápně do 10 m od branky).

---

### Fáze 1: Diagnostika jednorozměrné normality (VMM-3)

#### Shapiro-Wilkův test normality (pro `dist_m` a `angle_deg`)
- **Proč byl použit:** Většina klasických parametrických testů (Studentův t-test, Fisherova ANOVA, Pearsonova korelace) striktně předpokládá normální rozdělení zkoumaných veličin. Shapiro-Wilkův test je v literatuře považován za nejrobustnější test normality pro vzorky do tisíců pozorování.
- **Co zkoumá:**
  - $H_0$: Náhodná veličina pochází z normálního rozdělení $N(\mu, \sigma^2)$.
  - $H_1$: Rozdělení náhodné veličiny se statisticky významně liší od normálního.
- **Dosažené výsledky:**
  - `dist_m`: $W = 0,9833$, $p = 1,91 \times 10^{-7} \rightarrow$ **ZAMÍTÁME $H_0$** (šikmost $= 0,38$, mírná pravostranná asymetrie).
  - `angle_deg`: $W = 0,7344$, $p = 1,28 \times 10^{-32} \rightarrow$ **ZAMÍTÁME $H_0$** (šikmost $= 2,05$, výrazná asymetrie).
  - `xG`: $W = 0,5430$, $p = 7,24 \times 10^{-40} \rightarrow$ **ZAMÍTÁME $H_0$** (extrémní pravostranné sešikmení).
- **Věcný závěr:** Prostorové metriky střelby nejsou normálně rozdělené. Nelze spoléhat na klasické parametrické metody bez ověření jejich robustnosti, a je nutné upřednostnit neparametrické a robustní postupy.

---

### Fáze 2: Diagnostika vícerozměrné normality (VMM-2)

#### Henze-Zirklerův test & Chí-kvadrát Q-Q graf Mahalanobisových vzdáleností
- **Proč byl použit:** Jednorozměrná normalita jednotlivých proměnných je nutnou, ale nikoli postačující podmínkou vícerozměrné normality vektoru $\mathbf{X} = [\text{dist\_m}, \text{angle\_deg}, xG]^T$. Vícerozměrná normalita je klíčovým předpokladem pro klasickou MANOVA a lineární diskriminační analýzu.
- **Co zkoumá:**
  - $H_0$: Vektor proměnných má sdružené vícerozměrné normální rozdělení $N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.
  - $H_1$: Sdružené rozdělení není vícerozměrně normální.
  - Využívá se vlastnost, že při platnosti multinormality mají Mahalanobisovy vzdálenosti $D^2 = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$ teoretické rozdělení $\chi^2(p)$, kde $p = 3$.
- **Dosažené výsledky:**
  - Henze-Zirklerova statistika: $HZ = 68,281$, $p = 1,28 \times 10^{-188} \rightarrow$ **ZAMÍTÁME $H_0$**.
  - Empirický $\chi^2$ Q-Q diagram v grafu `02b_mahal_chisq_qq.png` vykazuje výrazný nelineární ohyb a detekuje multivariátní odlehlé hodnoty nad $99\%$ prahem $\chi^2(3) = 11,34$.
- **Věcný závěr:** Sdružené rozdělení prostorových metrik vykazuje silnou odchylku od multinormality. V navazující MANOVA je proto nezbytné interpretovat robustní **Pillaiovu stopu** namísto citlivé Wilksovy lambdy.

---

### Fáze 3: Diagnostika homoskedasticity a rovnosti kovariancí (VMM-3, VMM-5)

#### 1. Leveneho test homogenity rozptylů
- **Proč byl použit:** Testuje předpoklad homoskedasticity (shody rozptylů) mezi skupinami zakončení (`RightFoot`, `LeftFoot`, `Head`). Porušení homoskedasticity způsobuje selhání klasické Fisherovy jednofaktorové ANOVA (vede k falešně pozitivním výsledkům).
- **Co zkoumá:**
  - $H_0$: Rozptyly vzdálenosti jsou ve všech skupinách zakončení shodné ($\sigma_1^2 = \sigma_2^2 = \sigma_3^2$).
  - $H_1$: Alespoň jedna skupina vykazuje statisticky významně odlišný rozptyl.
- **Dosažené výsledky:**
  - $L = 14,860$, $p = 2,12 \times 10^{-9} \rightarrow$ **ZAMÍTÁME $H_0$**.
- **Věcný závěr:** Rozptyly vzdáleností jsou mezi typy zakončení heteroskedastické. Hlavičky mají výrazně menší variabilitu než střely nohou. **Klasická Fisherova ANOVA je nepoužitelná, je nutné nasadit Welchovu robustní ANOVA.**

#### 2. Boxův M test rovnosti kovariančních matic
- **Proč byl použit:** Jedná se o vícerozměrné zobecnění Leveneho testu pro MANOVA. Zkoumá, zda jsou sdružené kovarianční matice vektoru $[\text{dist\_m}, \text{angle\_deg}, xG]$ mezi skupinami zakončení shodné.
- **Co zkoumá:**
  - $H_0: \boldsymbol{\Sigma}_{\text{RightFoot}} = \boldsymbol{\Sigma}_{\text{LeftFoot}} = \boldsymbol{\Sigma}_{\text{Head}}$.
  - $H_1$: Kovarianční matice se mezi skupinami liší.
- **Dosažené výsledky:**
  - $\chi^2 = 132,38$, $p = 2,05 \times 10^{-22} \rightarrow$ **ZAMÍTÁME $H_0$**.
- **Věcný závěr:** Kovarianční struktury skupin jsou odlišné. Toto zjištění striktně vylučuje použití Wilksovy lambdy jako primární statistiky v MANOVA a vyžaduje použití robustní **Pillaiovy stopy**.

---

### Fáze 4: Analýza závislostí a parciální vazby (VMM-4)

#### 1. Spearmanův pořadový korelační test
- **Proč byl použit:** Vzhledem k prokázanému porušení normality proměnných nelze použít parametrický Pearsonův korelační koeficient. Spearmanovo $r_s$ hodnotí monotónní vztahy a je robustní vůči odlehlým hodnotám a nelinearitě.
- **Co zkoumá:**
  - $H_0$: Monotónní závislost mezi vzdáleností `dist_m` a metrikou `xG` je nulová ($r_s = 0$).
  - $H_1$: Mezi vzdáleností a xG existuje statisticky významná monotónní závislost ($r_s \neq 0$).
- **Dosažené výsledky:**
  - $r_s = -0,740$, $p = 1,93 \times 10^{-128} \rightarrow$ **ZAMÍTÁME $H_0$**.
- **Věcný závěr:** Mezi vzdáleností od branky a kvalitou střelecké pozice existuje velmi silná záporná vazba. Každé přiblížení k brance exponenciálně zvyšuje kvalitu šance.

#### 2. Test parciální korelace (kontrola vlivu vzdálenosti)
- **Proč byl použit:** Surová korelace mezi úhlem `angle_deg` a `xG` činí $+0,436$. Vzdálenost i úhel jsou však vázány geometrií hřiště (čím blíže k brance, tím větší je zorný úhel). Parciální korelace očišťuje vztah mezi úhlem a xG o matoucí vliv vzdálenosti (`dist_m`).
- **Co zkoumá:**
  - $H_0$: Parciální korelace mezi `xG` a `angle_deg` při konstantní vzdálenosti `dist_m` je nulová ($r_p = 0$).
  - $H_1$: Parciální korelace je statisticky významná ($r_p \neq 0$).
- **Dosažené výsledky:**
  - $r_p = +0,155$, $p = 2,52 \times 10^{-5} \rightarrow$ **ZAMÍTÁME $H_0$**.
- **Věcný závěr:** I když zafixujeme vzdálenost střely (např. přesně na 16 metrech), střelecký úhel má stále statisticky významný pozitivní vliv na kvalitu šance. Střela z centrální osy je podstatně nebezpečnější než střela ze stejné vzdálenosti z ostrého úhlu.

---

### Fáze 5: Porovnání skupin zakončení (VMM-5, VMM-6)

#### 1. Welchova robustní ANOVA
- **Proč byla použita:** V datech byla Leveneho testem prokázána silná heteroskedasticita rozptylů. Welchova ANOVA modifikuje stupně volnosti pomocí Satterthwaiteovy aproximace a poskytuje korektní $F$-test i při nestejných rozptylech a nevyvážených četnostech skupin.
- **Co zkoumá:**
  - $H_0$: Střední vzdálenosti střel jsou pro všechny typy zakončení stejné ($\mu_{\text{Pravá}} = \mu_{\text{Levá}} = \mu_{\text{Hlava}}$).
  - $H_1$: Alespoň jeden typ zakončení vzniká z jiné střední vzdálenosti.
- **Dosažené výsledky:**
  - Vzdálenost `dist_m`: $F = 176,87$, $p = 1,64 \times 10^{-44} \rightarrow$ **ZAMÍTÁME $H_0$**.
  - Úhel `angle_deg`: $F = 50,09$, $p = 8,03 \times 10^{-17} \rightarrow$ **ZAMÍTÁME $H_0$**.
- **Věcný závěr:** Typ zakončení zásadně determinuje prostorovou geometrii střely.

#### 2. Kruskal-Wallisův neparametrický test
- **Proč byl použit:** Slouží jako neparametrický protějšek jednofaktorové ANOVA založený na pořadích. Nevyžaduje normální rozdělení ani shodu rozptylů a byl zařazen jako robustní pojistka k Welchově ANOVA.
- **Co zkoumá:**
  - $H_0$: Distribuce vzdáleností střel jsou mezi typy zakončení shodné.
  - $H_1$: Alespoň jedna skupina vykazuje odlišné rozdělení (stochastickou dominanci).
- **Dosažené výsledky:**
  - $H = 88,67$, $p = 5,55 \times 10^{-20} \rightarrow$ **ZAMÍTÁME $H_0$**.
- **Věcný závěr:** Neparametrický test v plné shodě s Welchovou ANOVA potvrzuje, že prostorová pozice střelby závisí na typu tělesného zakončení.

#### 3. Vícerozměrná analýza rozptylu MANOVA (Pillaiova stopa & Wilksova $\lambda$)
- **Proč byla použita:** Namísto izolovaného testování vzdálenosti a úhlu hodnotí MANOVA sdružený dvourozměrný geometrický vektor $\mathbf{Y} = [\text{dist\_m}, \text{angle\_deg}]^T$, čímž eliminuje inflaci chyby I. druhu a zohledňuje vzájemnou kovarianci.
- **Co zkoumá:**
  - $H_0$: Vektory středních hodnot geometrie jsou ve všech skupinách zakončení totožné ($\boldsymbol{\mu}_1 = \boldsymbol{\mu}_2 = \boldsymbol{\mu}_3$).
  - $H_1$: Alespoň jedna skupina má statisticky odlišný vektor středních hodnot.
- **Dosažené výsledky:**
  - Pillaiova stopa: $\text{Pillai} = 0,1851$, $p = 2,12 \times 10^{-24} \rightarrow$ **ZAMÍTÁME $H_0$**.
  - Wilksova lambda: $\lambda = 0,8154$, $p = 1,24 \times 10^{-25} \rightarrow$ **ZAMÍTÁME $H_0$**.
- **Věcný závěr:** Typ zakončení vyvolává vysoce signifikantní sdružený posun v prostoru $[dist, angle]$. Post-hoc testy (Tukey HSD) odhalily, že hlavičky vznikají ze specifických pozic (průměrná vzdálenost $9,8\text{ m}$ oproti $\sim 20\text{ m}$ u střelby nohou, $p < 0,001$), zatímco mezi pravou a levou nohou nebyl zjištěn signifikantní prostorový rozdíl ($p = 0,22$).

---

### Fáze 5b: Závislost kvality šancí ($xG$) na střelci (Hráčské role)

#### 1. Kruskal-Wallisův test a Welchova ANOVA kvality šancí dle střelců
- **Proč byla použita:** Fotbalový výzkumný dotaz směřuje k tomu, zda je metrika očekávaných gólů ($xG$) nezávislá na střelci, nebo zda se jednotliví hráči dostávají do systematicky odlišných příležitostí. Vzhledem k silné asymetrii $xG$, heteroskedasticitě rozptylů a nevyváženým vzorkům střel (od $n = 15$ po $n = 262$) byl nasazen neparametrický Kruskal-Wallisův test doplněný o robustní Welchovu ANOVA.
- **Co zkoumá:**
  - $H_0$: Distribuce kvality šancí $xG$ je mezi všemi střelci týmu shodná (všichni hráči zakončují ze situací s ekvivalentním průměrným ohrožením branky).
  - $H_1$: Alespoň jeden hráč zakončuje z kvalitativně významně odlišných pozic ($xG$).
- **Dosažené výsledky:**
  - Kruskal-Wallisův test: $H = 71,26$, $p = 7,04 \times 10^{-11} \rightarrow$ **ZAMÍTÁME $H_0$**.
  - Welchova robustní ANOVA: $F = 26,90$, $p = 3,06 \times 10^{-27} \rightarrow$ **ZAMÍTÁME $H_0$**.
  - Výsledky byly detailně uloženy do `tabulka_hraci_xG.csv` a vizualizovány v grafu `04c_hraci_xG_boxplot.png`.
- **Věcný a taktický závěr:**
  - $xG$ **výrazně a signifikantně závisí na konkrétním střelci**, což je dáno taktickou herní rolí a pozičním úkolem:
    - **Hrotový útočník (Ollie Watkins):** Vykazuje extrémně vysokou kvalitu šancí (průměr $xG = 0,279$, medián $0,271$, průměrná vzdálenost $11,1\text{ m}$). Zakončuje takřka výhradně z malého vápna a dosahuje nejvyšší konverze ($35,3\,\%$).
    - **Křídelní a podhrotoví hráči (Jack Grealish, Bertrand Traoré, Trézéguet):** Střední pásmo kvality (medián $xG \approx 0,048$ až $0,053$, průměrná vzdálenost $15$ až $17\text{ m}$).
    - **Střední a defenzivní záložníci (Douglas Luiz, John McGinn, Ross Barkley):** Velmi nízká kvalita zakončení (medián $xG \approx 0,025$ až $0,040$, průměrná vzdálenost $21$ až $23\text{ m}$). Tito hráči produkují vysoký objem střel z dálky s minimální nadějí na skórování ($2$ až $5\,\%$).

---

### Fáze 6: Redukce dimenzionality – PCA a Faktorová analýza (VMM-7, 8, 9)

#### 1. Testy vhodnosti dat pro faktorování: KMO index & Bartlettův test
- **Metodická inovace projektu:** Ze vstupních proměnných byly odstraněny duplicitní souřadnice $X$ a $Y$, protože `dist_m` a `angle_deg` jsou jejich deterministickými transformacemi. Místo nich byla zařazena boční odchylka od osy hřiště $y_{\text{dev}} = |y - 34|\,\text{m}$. Vektor tvoří: `['dist_m', 'angle_deg', 'y_dev', 'xG']`.
- **Kaiser-Meyer-Olkin (KMO) index:**
  - Měří podíl rozptylu proměnných, který by mohl být způsoben společnými latentními faktory.
  - Výsledek: **$\text{KMO} = 0,696$** (hodnota $\approx 0,70$ potvrzuje dobrou adekvátnost datového výběru).
- **Bartlettův test sférickosti:**
  - $H_0$: Korelační matice je jednotková $\mathbf{R} = \mathbf{I}$ (proměnné jsou vzájemně nekorelované a faktorování nemá smysl).
  - $H_1$: Korelační matice se statisticky významně liší od jednotkové.
  - Výsledek: $\chi^2 = 1370,6$, $df = 6$, $p < 0,001 \rightarrow$ **ZAMÍTÁME $H_0$**.
- **Věcný závěr:** Množina prostorových proměnných vykazuje dostatečnou společnou variabilitu pro smysluplnou dekompozici PCA a faktorovou analýzu.

#### 2. Analýza hlavních komponent (PCA)
- **Kaiserovo pravidlo & Scree plot:** Vlastní čísla klesají: $\lambda_1 = 2,64$, $\lambda_2 = 0,74$, $\lambda_3 = 0,44$, $\lambda_4 = 0,18$. První dvě komponenty vysvětlují **$84,5\,\%$ celkového rozptylu** (PC1 = $66,0\,\%$, PC2 = $18,5\,\%$).
- **Interpretace komponent (2D Biplot):**
  - **PC1 (Osa centrální kvality pozice):** Silná záporná zátěž vzdálenosti ($-0,52$) a kladná zátěž úhlu ($+0,57$) a $xG$ ($+0,50$). Reprezentuje posun z dálky přímo před branku.
  - **PC2 (Osa boční excentricity):** Dominuje boční odchylka $y_{\text{dev}}$ ($+0,88$). Zachycuje střelbu z křídelních prostor.

#### 3. Faktorová analýza s ortogonální rotací Varimax
- **Proč byla použita:** PCA pouze maximalizuje vysvětlený rozptyl. Faktorová analýza s rotací Varimax usiluje o tzv. *jednoduchou strukturu* (Thurstone), kde má každá proměnná vysokou zátěž na jednom faktoru a téměř nulovou na druhém, což umožňuje identifikaci skrytých herních fenoménů.
- **Identifikované latentní faktory:**
  - **Faktor 1 (Centrální nebezpečnost zakončení):** Zátěž úhlu $+0,864$, zátěž $xG +0,673$ a záporná zátěž vzdálenosti $-0,818$.
  - **Faktor 2 (Boční excentricita zakončení):** Zátěž boční odchylky $y_{\text{dev}} +0,742$.
- **Věcný závěr:** Variabilita zakončení v Premier League je řízena dvěma ortogonálními silami: centrální nebezpečností (jak blízko v ose hráč je) a křídelní odchylkou (jak moc střílí z úhlu).

---

### Fáze 7: Shluková analýza – Segmentace střel do 3 zón (VMM-11)

#### 1. Matematické odůvodnění řezu na $k = 3$ shluky
- Hierarchické aglomerativní shlukování Wardovou metodou s eukleidovskou metrikou spojuje data na těchto klíčových výškách:
  - Spojení 2 shluků do 1: výška $h = 47,01$
  - Spojení 3 shluků do 2: výška $h = 29,60$
  - Spojení 4 shluků do 3: výška $h = 16,84$
- **Korektní řez dendrogramem:** Řez byl umístěn přesně do středu intervalu mezi 3 a 4 shluky na výšku **$h = 23,22$**, kde protíná přesně **3 větve stromu**.
- **Metoda lokte (Elbow Method):** Přechod z $k = 2$ na $k = 3$ přináší dramatický pokles vnitroshlukového součtu čtverců (WCSS / Inertia) z $1070$ na $631$, což představuje **pokles variability o $41,0\,\%$**. Za bodem $k = 3$ je pokles již pozvolný – $k = 3$ představuje zřetelnou inflexi metody lokte.
- **Silhouette score:** Pro $k = 3$ dosahuje koeficient siluety stabilní hodnoty **$0,484$**.

#### 2. Taktická charakteristika a věcný smysl 3 shluků
Původní dělení na $k = 2$ shluky bylo analyticky triviální (pouze rozdělení na „blízko vs. daleko“). Dělení na $k = 3$ shluky odkrývá **3 fundamentální taktické zóny moderního fotbalu**:

| Shluk | Název zóny | Počet střel ($n$) | Prům. vzdálenost | Prům. úhel | Prům. $xG$ | Vstřelené góly | Konverze střel |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Zlatá zóna (Malé vápno)** | $77$ ($10,5\,\%$) | **$8,4\text{ m}$** | **$47,8^\circ$** | $0,397$ | **$28$ gólů** | **$36,4\,\%$** |
| **2** | **Pokutové území (2. vlna)** | $308$ ($41,9\,\%$) | **$15,7\text{ m}$** | **$22,0^\circ$** | $0,081$ | **$21$ gólů** | **$6,8\,\%$** |
| **3** | **Dálková střelba (Mimo vápno)** | $350$ ($47,6\,\%$) | **$25,8\text{ m}$** | **$14,8^\circ$** | $0,029$ | **$14$ gólů** | **$4,0\,\%$** |

- **Věcný závěr:** Neřízené shlukování odhalilo obrovský paradox: Ačkoliv Zlatá zóna tvoří pouhých $10,5\,\%$ všech střel, padá z ní **$44,4\,\%$ všech gólů** celého datasetu. Naopak dálková střelba spálí téměř polovinu pokusů týmu ($47,6\,\%$) s bídnou konverzí $4\,\%$. Jde o matematický důkaz, proč moderní velkokluby v tréninku eliminují střelbu zpoza vápna ve prospěch dohrávání akcí do Zlaté zóny.

---

### Fáze 8: Multivariabilní logistická regrese a ROC analýza (VMM-4, 5)

#### 1. Metodická eliminace datového úniku (Data Leakage)
V modelech logistické regrese nesmí metrika $xG$ vystupovat jako prediktor, protože je sama prediktivním modelem odvozeným z geometrie. Model odhaduje logit šance na gól:
$$\ln\left(\frac{P(\text{gól})}{1 - P(\text{gól})}\right) = \beta_0 + \beta_1 \cdot \text{dist\_m} + \beta_2 \cdot \text{angle\_deg} + \beta_3 \cdot \text{minute} + \sum \beta_j \cdot \text{shotType}_j$$
Referenční kategorií pro dummy kódování je zakončení hlavou (`Head`).

#### 2. Likelihood Ratio Test (LLR $\chi^2$) celkového modelu
- **Proč byl použit:** Testuje celkovou statistickou významnost regresního modelu porovnáním logaritmu věrohodnosti plného modelu ($LL$) s modelem obsahujícím pouze konstantu ($LL_0$).
- **Co zkoumá:**
  - $H_0: \beta_1 = \beta_2 = \dots = \beta_k = 0$ (žádný prediktor nemá vliv na šanci na gól).
  - $H_1$: Alespoň jeden regresní koeficient je nenulový.
- **Dosažené výsledky:**
  - $\chi^2 = 69,74$, $df = 6$, $p = 4,62 \times 10^{-13} \rightarrow$ **ZAMÍTÁME $H_0$**.
- **Věcný závěr:** Model jako celek statisticky vysoce významně predikuje pravděpodobnost gólu.

#### 3. Waldovy testy regresních koeficientů a poměry šancí (Odds Ratios)
- **Vzdálenost (`dist_m`):**
  - $\beta = -0,0855$, $z = -2,416$, **$p = 0,0157 \rightarrow$ ZAMÍTÁME $H_0$**.
  - **Odds Ratio: $\text{OR} = 0,918$** ($95\%$ CI: $[0,856; 0,984]$).
  - **Interpretace:** Každý další metr vzdálenosti od soupeřovy branky snižuje poměr šancí na vstřelení gólu o **$8,2\,\%$** ($1 - 0,918 = 0,082$).
- **Střelecký úhel (`angle_deg`):**
  - $\beta = +0,0368$, $z = 2,447$, **$p = 0,0144 \rightarrow$ ZAMÍTÁME $H_0$**.
  - **Odds Ratio: $\text{OR} = 1,037$** ($95\%$ CI: $[1,007; 1,068]$).
  - **Interpretace:** Každý další stupeň zorného úhlu zvyšuje šanci na gól o **$3,7\,\%$**.
- **Typ zakončení (Noha vs. Hlava):**
  - Pravá noha: $\text{OR} = 10,18$ ($p = 0,003$); Levá noha: $\text{OR} = 7,17$ ($p = 0,013$).
  - Zakončení nohou má při stejné geometrické pozici řádově vyšší šanci na gól než zakončení hlavou.

#### 4. Diagnostika multikolinearity (VIF)
Všechny hodnoty variance inflation factor v modelu jsou hluboko pod kritickou mezí $5,0$ (`dist_m`: $1,78$; `angle_deg`: $1,75$; `minute`: $1,01$; dummy zakončení: $3,52$ až $4,10$). Model netrpí škodlivou multikolinearitou.

#### 5. ROC křivka a srovnání s benchmarkem xG
Náš multivariabilní geometrický model dosahuje plochy pod ROC křivkou **$\text{AUC} = 0,776$**, čímž prokazuje vysokou diskriminační schopnost a překonává surový externí benchmark samotné metriky $xG$ ($\text{AUC} = 0,746$).

---

## 4. Souhrnná tabulka všech 14 testovaných statistických hypotéz

Všechna rozhodnutí o nulových hypotézách jsou v kódu vyhodnocována **striktně dynamicky** na hladině významnosti $\alpha = 0,05$:

| Fáze | Metoda | Nulová hypotéza ($H_0$) | Testová statistika | $p$-hodnota | Rozhodnutí ($\alpha = 0,05$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 – Normalita** | Shapiro-Wilkův test (`dist_m`) | Vzdálenost má jednorozměrné normální rozdělení | $W = 0,9833$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **1 – Normalita** | Shapiro-Wilkův test (`angle_deg`) | Střelecký úhel má jednorozměrné normální rozdělení | $W = 0,7344$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **1 – Normalita MV** | Henze-Zirklerův test | Vektor $[dist, angle, xG]$ má sdružené normální rozdělení | $HZ = 68,281$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **2 – Homoskedasticita** | Leveneho test rozptylů | Rozptyly vzdálenosti jsou mezi skupinami shodné | $L = 14,860$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **2 – Kovariance** | Boxův M test kovariancí | Kovarianční matice jsou ve skupinách shodné | $\chi^2 = 132,4$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **3 – Závislost** | Spearmanův pořadový test | Monotónní závislost `dist_m` a `xG` je nulová ($r_s = 0$) | $r_s = -0,740$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **3 – Parciální kor.** | Pingouin parciální korelace | Korelace `xG` a úhlu při kontrole vzdálenosti je nulová | $r_p = +0,155$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **4 – Skupiny Welch** | Welchova robustní ANOVA (`dist_m`) | Střední vzdálenosti střel jsou pro typy zakončení shodné | $F = 176,9$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **4 – Skupiny Neparam.** | Kruskal-Wallisův test (`dist_m`) | Distribuce vzdáleností jsou mezi skupinami shodné | $H = 88,7$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **4 – Sdružené MANOVA** | Pillaiova stopa / Wilks $\lambda$ | Sdružené vektory středních hodnot geometrie jsou shodné | $\text{Pillai} = 0,185$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **5 – Vhodnost PCA** | Bartlettův test sférickosti | Korelační matice je jednotková (proměnné nekorelované) | $\chi^2 = 1370,6$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **6 – Celkový Logit** | Likelihood Ratio Test (LLR) | Všechny regresní koeficienty modelu jsou nulové | $\chi^2 = 69,7$ | $< 0,001$ | **ZAMÍTÁME $H_0$** |
| **6 – Vzdálenost Logit** | Waldův test koeficientu (`dist_m`) | Vzdálenost neovlivňuje logit šance na gól ($\beta_{\text{dist}} = 0$) | $\text{OR} = 0,918$ | $0,0157$ | **ZAMÍTÁME $H_0$** |
| **6 – Úhel Logit** | Waldův test koeficientu (`angle_deg`) | Střelecký úhel neovlivňuje logit šance na gól ($\beta_{\text{angle}} = 0$) | $\text{OR} = 1,037$ | $0,0144$ | **ZAMÍTÁME $H_0$** |

---

## 5. Praktická doporučení pro trenéry a taktické závěry

1. **Priorita Zlaté zóny před brankou:** Zakončení z prostoru malého vápna ($< 12\text{ m}$) dosahuje konverze **$36,4\,\%$**, zatímco mimo vápno padá pod **$4\,\%$**. Herní nácvik musí preferovat finální přihrávky pod sebe (*cutbacks*) z brankové čáry do Zlaté zóny.
2. **Maximalizace střeleckého zorného úhlu:** Každý stupeň úhlu zvyšuje šanci na gól o **$3,7\,\%$**. Útočníci mají být vedeni k zakončení z centrální osy hřiště a vyvarovat se unáhlené střelbě z ostrých úhlů u postranních čar.
3. **Taktické omezení dálkové střelby:** Střely ze vzdálenosti $> 20\text{ m}$ tvoří téměř polovinu pokusů týmu, ale přinášejí minimální užitek. V tréninku je nutné hráče motivovat k trpělivé mezihře a neplýtvat držením míče střelbou z nepřipravených pozic.
4. **Specifický nácvik centrovaných míčů na hlavičky:** Hlavičkové pokusy mají při stejné geometrii podstatně nižší konverzi než střela nohou ($\text{OR} \approx 0,10$). Centry musí cílit výhradně do bezprostřední blízkosti branky (5–8 metrů), kde je šance hlavičkujícího hráče reálná.

---

## 6. Návod na spuštění a reprodukovatelnost

### 1. Požadavky na prostředí
Doporučeno prostředí Python 3.10+ s instalovanými balíčky:
```bash
pip install pandas numpy scipy statsmodels scikit-learn seaborn matplotlib mplsoccer pingouin python-pptx
```

### 2. Krok 1 – Očištění a předzpracování dat
```bash
cd kody
python 01_predzpracovani.py
```
*Výstup:* Vytvoří očištěný soubor `data/clean_shots.csv` a protokoly o chybějících datech.

### 3. Krok 2 – Spuštění vícerozměrné statistické analýzy
```bash
python 02_analyza.py
```
*Výstup:* Spustí všech 8 fází analýzy, vyhodnotí 14 hypotéz a vygeneruje grafy a CSV tabulky do složky `vystupy/`.

### 4. Krok 3 – Vygenerování prezentace
```bash
python vytvor_prezentaci.py
```
*Výstup:* Vygeneruje finální PowerPointovou prezentaci do `prezentace/prezentace_strelecka_uspesnost.pptx` (15 slidů, poměr stran 16:9).
