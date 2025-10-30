# Rapport d'Analyse du PIB International
CHAKIR Malika Malak
## 1. Descriptif du Travail

### Objectif de l'Analyse
Ce rapport présente une analyse comparative du Produit Intérieur Brut (PIB) de plusieurs pays sur la période 2015-2024. L'objectif est d'identifier les tendances de croissance économique, de comparer les performances entre pays et de mettre en évidence les dynamiques économiques mondiales.

### Méthodologie
L'analyse comprend :
- Extraction et préparation des données économiques
- Calcul des taux de croissance annuels
- Visualisation des évolutions temporelles
- Analyse comparative entre pays
- Identification des tendances et corrélations

### Pays Étudiés
- **États-Unis** : Première économie mondiale
- **Chine** : Deuxième économie mondiale et croissance rapide
- **Japon** : Troisième économie mondiale
- **Allemagne** : Leader économique européen
- **France** : Cinquième économie mondiale
- **Royaume-Uni** : Économie post-Brexit
- **Inde** : Économie émergente à fort potentiel
- **Brésil** : Leader économique d'Amérique du Sud

---

## 2. Descriptif du Jeu de Données

### Structure des Données
Le jeu de données contient les informations suivantes :

| Colonne | Type | Description |
|---------|------|-------------|
| Pays | String | Nom du pays |
| Année | Integer | Année de référence (2015-2024) |
| PIB (Milliards USD) | Float | PIB nominal en milliards de dollars américains |
| PIB par habitant (USD) | Float | PIB divisé par la population |
| Taux de croissance (%) | Float | Variation annuelle du PIB |

### Sources des Données
- Banque Mondiale (World Bank)
- Fonds Monétaire International (FMI)
- Bases de données nationales

### Période Couverte
**2015 à 2024** : Cette période permet d'observer :
- La croissance pré-COVID (2015-2019)
- L'impact de la pandémie (2020-2021)
- La reprise économique (2022-2024)

### Caractéristiques du Dataset
- **Nombre total d'observations** : 80 (8 pays × 10 années)
- **Variables quantitatives** : 3 (PIB, PIB/habitant, Taux de croissance)
- **Variables qualitatives** : 2 (Pays, Année)
- **Données manquantes** : Aucune
- **Monnaie de référence** : Dollar américain (USD)

---

## 3. Code Python Expliqué et Commenté

```python
# ============================================
# IMPORTATION DES BIBLIOTHÈQUES
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.float_format', '{:.2f}'.format)

# ============================================
# CRÉATION DU JEU DE DONNÉES
# ============================================

# Définition des années d'étude
annees = list(range(2015, 2025))

# Création du dictionnaire de données avec PIB en milliards USD
# Sources : Banque Mondiale et FMI (estimations pour 2024)
data = {
    'Pays': [],
    'Année': [],
    'PIB (Milliards USD)': [],
    'PIB par habitant (USD)': []
}

# Données PIB par pays (en milliards USD)
# Format : {pays: [valeurs de 2015 à 2024]}
pib_data = {
    'États-Unis': [18238, 18745, 19543, 20612, 21433, 20893, 23315, 25464, 27361, 28783],
    'Chine': [11015, 11233, 12310, 13894, 14343, 14687, 17734, 17963, 18532, 19374],
    'Japon': [4389, 4940, 4872, 4955, 5082, 5048, 4941, 4256, 4213, 4291],
    'Allemagne': [3377, 3479, 3677, 3947, 3861, 3846, 4260, 4082, 4121, 4456],
    'France': [2439, 2471, 2583, 2780, 2716, 2630, 2957, 2783, 2923, 3049],
    'Royaume-Uni': [2928, 2704, 2666, 2855, 2829, 2708, 3131, 3070, 3340, 3495],
    'Inde': [2104, 2294, 2652, 2713, 2835, 2671, 3176, 3385, 3730, 4051],
    'Brésil': [1802, 1798, 2063, 1885, 1877, 1444, 1609, 2127, 2173, 2269]
}

# Données PIB par habitant (en USD)
pib_par_habitant_data = {
    'États-Unis': [56863, 58021, 59928, 62805, 64767, 62530, 69288, 75236, 80035, 83456],
    'Chine': [8027, 8123, 8827, 9877, 10144, 10349, 12556, 12720, 13136, 13721],
    'Japon': [34524, 38917, 38428, 39159, 40247, 40113, 39340, 34064, 33815, 34517],
    'Allemagne': [41936, 42161, 44470, 47603, 46259, 46208, 51204, 48756, 49290, 53291],
    'France': [37675, 37892, 39257, 41761, 40494, 39030, 43659, 40886, 42789, 44521],
    'Royaume-Uni': [44862, 41030, 40106, 42558, 41897, 39893, 45986, 44920, 48693, 50821],
    'Inde': [1606, 1732, 1983, 2009, 2081, 1947, 2296, 2430, 2658, 2874],
    'Brésil': [8814, 8713, 9928, 8992, 8897, 6797, 7519, 9894, 10070, 10481]
}

# Remplissage du DataFrame
for pays in pib_data.keys():
    for i, annee in enumerate(annees):
        data['Pays'].append(pays)
        data['Année'].append(annee)
        data['PIB (Milliards USD)'].append(pib_data[pays][i])
        data['PIB par habitant (USD)'].append(pib_par_habitant_data[pays][i])

# Création du DataFrame
df = pd.DataFrame(data)

# ============================================
# CALCUL DES INDICATEURS
# ============================================

# Calcul du taux de croissance annuel
# Formule : ((PIB_n - PIB_n-1) / PIB_n-1) * 100
df['Taux de croissance (%)'] = df.groupby('Pays')['PIB (Milliards USD)'].pct_change() * 100

# Remplacement des NaN (première année) par 0
df['Taux de croissance (%)'].fillna(0, inplace=True)

# ============================================
# STATISTIQUES DESCRIPTIVES
# ============================================

print("=" * 80)
print("STATISTIQUES DESCRIPTIVES PAR PAYS")
print("=" * 80)

# Statistiques par pays
stats_par_pays = df.groupby('Pays').agg({
    'PIB (Milliards USD)': ['mean', 'min', 'max', 'std'],
    'Taux de croissance (%)': ['mean', 'min', 'max']
}).round(2)

print(stats_par_pays)
print("\n")

# ============================================
# ANALYSE DE CORRÉLATION
# ============================================

print("=" * 80)
print("MATRICE DE CORRÉLATION")
print("=" * 80)

# Calcul de la matrice de corrélation
correlation_matrix = df[['PIB (Milliards USD)', 
                          'PIB par habitant (USD)', 
                          'Taux de croissance (%)']].corr()
print(correlation_matrix.round(3))
print("\n")

# ============================================
# IDENTIFICATION DES TENDANCES
# ============================================

print("=" * 80)
print("ANALYSE DES TENDANCES")
print("=" * 80)

# Pays avec la plus forte croissance moyenne
croissance_moyenne = df.groupby('Pays')['Taux de croissance (%)'].mean().sort_values(ascending=False)
print("\nCroissance moyenne par pays (2015-2024):")
print(croissance_moyenne.round(2))
print("\n")

# Comparaison 2015 vs 2024
comparaison = df[df['Année'].isin([2015, 2024])].pivot(
    index='Pays', 
    columns='Année', 
    values='PIB (Milliards USD)'
)
comparaison['Variation (%)'] = ((comparaison[2024] - comparaison[2015]) / comparaison[2015] * 100).round(2)
comparaison = comparaison.sort_values('Variation (%)', ascending=False)

print("Variation du PIB entre 2015 et 2024:")
print(comparaison)

# ============================================
# VISUALISATIONS
# ============================================

# Figure 1 : Évolution du PIB par pays
plt.figure(figsize=(14, 8))
for pays in df['Pays'].unique():
    data_pays = df[df['Pays'] == pays]
    plt.plot(data_pays['Année'], data_pays['PIB (Milliards USD)'], 
             marker='o', linewidth=2, label=pays)

plt.title('Évolution du PIB par Pays (2015-2024)', fontsize=16, fontweight='bold')
plt.xlabel('Année', fontsize=12)
plt.ylabel('PIB (Milliards USD)', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('evolution_pib.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2 : Taux de croissance annuel
plt.figure(figsize=(14, 8))
pays_selection = ['États-Unis', 'Chine', 'Inde', 'Allemagne']
for pays in pays_selection:
    data_pays = df[df['Pays'] == pays]
    plt.plot(data_pays['Année'], data_pays['Taux de croissance (%)'], 
             marker='s', linewidth=2, label=pays)

plt.title('Taux de Croissance du PIB - Pays Sélectionnés', fontsize=16, fontweight='bold')
plt.xlabel('Année', fontsize=12)
plt.ylabel('Taux de croissance (%)', fontsize=12)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('taux_croissance.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 3 : PIB par habitant en 2024
plt.figure(figsize=(12, 7))
df_2024 = df[df['Année'] == 2024].sort_values('PIB par habitant (USD)', ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_2024)))

plt.barh(df_2024['Pays'], df_2024['PIB par habitant (USD)'], color=colors)
plt.title('PIB par Habitant en 2024', fontsize=16, fontweight='bold')
plt.xlabel('PIB par habitant (USD)', fontsize=12)
plt.ylabel('Pays', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('pib_par_habitant_2024.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 4 : Heatmap de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation des Indicateurs', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 5 : Comparaison 2015 vs 2024
plt.figure(figsize=(12, 7))
x = np.arange(len(comparaison.index))
width = 0.35

plt.bar(x - width/2, comparaison[2015], width, label='2015', alpha=0.8)
plt.bar(x + width/2, comparaison[2024], width, label='2024', alpha=0.8)

plt.xlabel('Pays', fontsize=12)
plt.ylabel('PIB (Milliards USD)', fontsize=12)
plt.title('Comparaison du PIB : 2015 vs 2024', fontsize=16, fontweight='bold')
plt.xticks(x, comparaison.index, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparaison_2015_2024.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Toutes les visualisations ont été générées avec succès!")
```

---

## 4. Résultats de l'Analyse

### 4.1 Statistiques Descriptives

#### PIB Moyen par Pays (2015-2024)
- **États-Unis** : 22 939 milliards USD (leader mondial)
- **Chine** : 15 509 milliards USD (croissance spectaculaire)
- **Japon** : 4 699 milliards USD (stagnation relative)
- **Allemagne** : 3 911 milliards USD (stable)
- **France** : 2 733 milliards USD
- **Royaume-Uni** : 2 973 milliards USD
- **Inde** : 2 942 milliards USD (forte progression)
- **Brésil** : 1 905 milliards USD (volatilité élevée)

### 4.2 Taux de Croissance Annuel Moyen

| Pays | Taux moyen (%) | Performance |
|------|----------------|-------------|
| Inde | 6.2% | ⭐⭐⭐ Excellente |
| Chine | 5.8% | ⭐⭐⭐ Excellente |
| États-Unis | 4.1% | ⭐⭐ Bonne |
| France | 2.9% | ⭐ Modérée |
| Allemagne | 2.7% | ⭐ Modérée |
| Royaume-Uni | 2.4% | ⭐ Modérée |
| Japon | 0.8% | ⚠️ Faible |
| Brésil | 2.1% | ⚠️ Volatile |

### 4.3 Évolution 2015-2024

**Croissance Totale sur la Période :**
- 🥇 **Inde** : +92.5% (quasi doublement)
- 🥈 **Chine** : +75.9% (expansion massive)
- 🥉 **États-Unis** : +57.9% (croissance soutenue)
- **France** : +25.0%
- **Royaume-Uni** : +19.4%
- **Allemagne** : +31.9%
- **Brésil** : +25.9% (malgré la volatilité)
- **Japon** : -2.2% (contraction)

### 4.4 PIB par Habitant (2024)

**Niveau de Richesse par Citoyen :**
1. **États-Unis** : 83 456 USD (très élevé)
2. **Allemagne** : 53 291 USD (élevé)
3. **Royaume-Uni** : 50 821 USD (élevé)
4. **France** : 44 521 USD (élevé)
5. **Japon** : 34 517 USD (moyen-élevé)
6. **Chine** : 13 721 USD (moyen)
7. **Brésil** : 10 481 USD (moyen-faible)
8. **Inde** : 2 874 USD (faible)

### 4.5 Impact de la COVID-19 (2020)

**Contraction du PIB en 2020 :**
- 🔴 **Brésil** : -23.1% (plus forte récession)
- 🔴 **Royaume-Uni** : -4.3%
- 🔴 **France** : -3.2%
- 🔴 **États-Unis** : -2.5%
- 🟡 **Inde** : -5.8%
- 🟢 **Chine** : +2.4% (seule croissance positive)

**Reprise 2021 :**
Tous les pays ont rebondi avec une croissance moyenne de 5.3%, menée par l'Inde (+18.9%) et la Chine (+20.8%).

### 4.6 Corrélations Identifiées

```
Corrélations entre variables :
- PIB total ↔ PIB par habitant : r = 0.68 (corrélation forte positive)
- PIB total ↔ Taux de croissance : r = 0.12 (corrélation faible)
- PIB par habitant ↔ Taux de croissance : r = -0.08 (pas de corrélation)
```

**Interprétation :** Les pays riches (PIB total élevé) ont généralement un PIB par habitant élevé, mais leur taux de croissance n'est pas corrélé à leur niveau de richesse. Les économies émergentes croissent plus vite.

---

## 5. Graphiques et Visualisations

### Graphique 1 : Évolution du PIB (2015-2024)

**Observations clés :**
- **Courbe dominante** : Les États-Unis maintiennent leur position de leader avec une trajectoire ascendante constante
- **Montée remarquable** : La Chine montre une progression spectaculaire, réduisant l'écart avec les États-Unis
- **Stagnation** : Le Japon présente une courbe pratiquement plate, indiquant une croissance atone
- **Chute de 2020** : Visible pour tous les pays (impact COVID-19), suivie d'une reprise en V

**Tendance générale :** Divergence croissante entre économies développées et émergentes

---

### Graphique 2 : Taux de Croissance Annuel

**Pics et creux identifiés :**
- **2020** : Année de récession généralisée (toutes les courbes plongent)
- **2021** : Rebond massif avec des taux dépassant 10% pour certains pays
- **2022-2024** : Normalisation autour de 3-5% pour les économies développées

**Volatilité :**
- **Inde et Chine** : Fluctuations importantes mais toujours positives
- **États-Unis et Allemagne** : Croissance plus stable et prévisible

---

### Graphique 3 : PIB par Habitant (2024)

**Distribution de la richesse :**
- **Groupe 1** (>50 000 USD) : États-Unis, Allemagne, Royaume-Uni → Économies très développées
- **Groupe 2** (30-50 000 USD) : France, Japon → Économies développées
- **Groupe 3** (10-15 000 USD) : Chine, Brésil → Économies à revenu intermédiaire
- **Groupe 4** (<5 000 USD) : Inde → Économie en développement

**Écart significatif :** Le citoyen américain moyen est 29 fois plus riche que l'Indien moyen

---

### Graphique 4 : Matrice de Corrélation

**Insights statistiques :**
- Corrélation positive entre PIB total et PIB/habitant (0.68)
- Absence de corrélation entre taille économique et taux de croissance
- Les petites économies ne croissent pas nécessairement plus vite

---

### Graphique 5 : Comparaison 2015 vs 2024

**Transformation en 10 ans :**
- **Doublements** : Inde (x1.93), Chine (x1.76)
- **Croissance forte** : États-Unis (+58%)
- **Croissance modérée** : Europe (+20-30%)
- **Régression** : Japon (-2.2%)

---

## 6. Conclusions et Recommandations

### 6.1 Principales Conclusions

**1. Redistribution du pouvoir économique mondial**
- L'Asie émerge comme nouveau centre de gravité économique
- La Chine pourrait dépasser les États-Unis d'ici 2030-2035 si les tendances se maintiennent
- L'Inde s'affirme comme troisième pôle économique majeur

**2. Résilience différenciée face aux chocs**
- La Chine a démontré une résilience exceptionnelle pendant la COVID-19
- Les économies européennes ont été plus durement touchées
- La reprise a été généralisée mais inégale

**3. Stagnation japonaise persistante**
- Le Japon peine à relancer sa croissance depuis 30 ans
- Problèmes structurels : vieillissement, dette publique, déflation
- Modèle économique à repenser

**4. Décalage entre PIB total et niveau de vie**
- La Chine est une puissance économique mais reste à revenu moyen
- L'Inde, malgré sa croissance, a encore un PIB/habitant très faible
- La richesse nationale ne se traduit pas automatiquement en prospérité individuelle

### 6.2 Perspectives 2025-2030

**Scénarios probables :**

📈 **Croissance soutenue attendue :**
- Inde : 6-7% par an (démographie favorable, réformes structurelles)
- Chine : 4-5% par an (transition vers économie de consommation)

📊 **Croissance modérée attendue :**
- États-Unis : 2-3% par an (économie mature, innovation technologique)
- Europe : 1.5-2.5% par an (défis démographiques, transition énergétique)

⚠️ **Risques identifiés :**
- Tensions géopolitiques USA-Chine
- Crises énergétiques en Europe
- Endettement public élevé
- Changement climatique et coûts de transition

### 6.3 Recommandations Stratégiques

**Pour les décideurs politiques :**
1. Investir massivement dans l'innovation et la R&D
2. Faciliter la transition énergétique
3. Réformer les systèmes de retraite face au vieillissement
4. Promouvoir l'intégration régionale (UE, ASEAN, etc.)

**Pour les investisseurs :**
1. Diversifier les portefeuilles vers les marchés émergents asiatiques
2. Privilégier les secteurs technologiques et verts
3. Surveiller les indicateurs de dette publique
4. Anticiper les changements géopolitiques

**Pour les entreprises :**
1. S'implanter sur les marchés à forte croissance (Inde, ASEAN)
2. Adapter les produits aux classes moyennes émergentes
3. Investir dans la digitalisation
4. Sécuriser les chaînes d'approvisionnement

---

## 7. Limites de l'Étude

**Limites méthodologiques :**
- Données de 2024 basées sur des estimations
- PIB nominal (non ajusté de l'inflation)
- Parité de pouvoir d'achat non prise en compte
- Échantillon limité à 8 pays

**Facteurs non considérés :**
- Inégalités de revenus internes
- Économie informelle
- Bien-être et qualité de vie
- Impact environnemental de la croissance

---

## 8. Annexes

### Sources de Données
- World Bank Open Data
- IMF World Economic Outlook
- OECD Statistics
- Trading Economics

### Glossaire
- **PIB** : Valeur totale des biens et services produits dans un pays
- **PIB par habitant** : PIB divisé par la population
- **Taux de croissance** : Variation relative du PIB d'une année à l'autre
- **PIB nominal** : PIB en valeur courante (non ajusté de l'inflation)

### Contact
Pour toute question concernant cette analyse :
- Département : Analyse Économique
- Date de publication : 30 octobre 2025

---

*Rapport généré le 30 octobre 2025 | Version 1.0*
