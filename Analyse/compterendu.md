# ============================================================================
# ANALYSE DE SENTIMENT SUR LES DONNÉES REDDIT
# Rapport Complet - Classification Multi-classe avec 10 Modèles
# ============================================================================

# =============================================================================
# 1. INSTALLATION ET IMPORTATION DES BIBLIOTHÈQUES
# =============================================================================

# Installation des packages nécessaires
!pip install -q wordcloud imbalanced-learn xgboost

# Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# Sklearn - Prétraitement et Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Sklearn - Modèles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Sklearn - Métriques
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Imbalanced-learn
from imblearn.over_sampling import SMOTE

# Scipy
from scipy.sparse import hstack

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")

print("✓ Toutes les bibliothèques ont été importées avec succès!")

# =============================================================================
# 2. TÉLÉCHARGEMENT ET CHARGEMENT DU DATASET
# =============================================================================

print("\n" + "="*80)
print("TÉLÉCHARGEMENT DU DATASET")
print("="*80)

# Téléchargement du dataset depuis Kaggle
# Note: Assurez-vous d'avoir configuré votre API Kaggle au préalable
# Instructions: https://www.kaggle.com/docs/api

!mkdir -p ~/.kaggle
# Uploadez votre kaggle.json ou utilisez la commande ci-dessous
# !cp /content/kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# Téléchargement du dataset
!kaggle datasets download -d alyahmedts13/reddit-sentiment-analysis-dataset-for-nlp-projects

# Décompression
!unzip -q reddit-sentiment-analysis-dataset-for-nlp-projects.zip

# Chargement des données
df = pd.read_csv('Dataset.csv')

print(f"\n✓ Dataset chargé avec succès!")
print(f"Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")

# =============================================================================
# 3. EXPLORATION INITIALE DES DONNÉES
# =============================================================================

print("\n" + "="*80)
print("EXPLORATION INITIALE DES DONNÉES")
print("="*80)

# Aperçu du dataset
print("\n--- Aperçu des premières lignes ---")
display(df.head())

# Informations sur les colonnes
print("\n--- Informations sur le dataset ---")
print(df.info())

# Statistiques descriptives
print("\n--- Statistiques descriptives ---")
display(df.describe())

# Noms des colonnes
print("\n--- Colonnes disponibles ---")
print(df.columns.tolist())

# =============================================================================
# 4. PRÉPARATION DES DONNÉES
# =============================================================================

print("\n" + "="*80)
print("PRÉPARATION DES DONNÉES")
print("="*80)

# 4.1 Vérification des valeurs manquantes
print("\n--- Vérification des valeurs manquantes ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if df.isnull().sum().sum() > 0:
    print(f"\n⚠ Suppression de {df.isnull().sum().sum()} valeurs manquantes...")
    df = df.dropna()
    print(f"✓ Lignes restantes: {df.shape[0]}")
else:
    print("✓ Aucune valeur manquante détectée")

# 4.2 Vérification et suppression des doublons
print("\n--- Vérification des doublons ---")
duplicates = df.duplicated().sum()
print(f"Nombre de doublons: {duplicates}")

if duplicates > 0:
    print(f"⚠ Suppression de {duplicates} doublons...")
    df = df.drop_duplicates()
    print(f"✓ Lignes restantes: {df.shape[0]}")
else:
    print("✓ Aucun doublon détecté")

# 4.3 Encodage de la variable cible
print("\n--- Encodage de la variable cible ---")
print("Distribution avant encodage:")
print(df['category'].value_counts())

# Encodage: -1 -> 0 (Négatif), 0 -> 1 (Neutre), 1 -> 2 (Positif)
df['sentiment'] = df['category'].map({-1: 0, 0: 1, 1: 2})

print("\nDistribution après encodage:")
print(df['sentiment'].value_counts())
print("\nMapping: 0=Négatif, 1=Neutre, 2=Positif")

# =============================================================================
# 5. ANALYSE EXPLORATOIRE ET VISUALISATIONS
# =============================================================================

print("\n" + "="*80)
print("ANALYSE EXPLORATOIRE ET VISUALISATIONS")
print("="*80)

# 5.1 Distribution des sentiments
print("\n--- 5.1 Distribution des sentiments ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Graphique en barres
sentiment_counts = df['sentiment'].value_counts().sort_index()
colors = ['#e74c3c', '#f39c12', '#2ecc71']
ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
ax1.set_ylabel('Nombre de commentaires', fontsize=12, fontweight='bold')
ax1.set_title('Distribution des Sentiments', fontsize=14, fontweight='bold')
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(['Négatif', 'Neutre', 'Positif'])
ax1.grid(axis='y', alpha=0.3)

# Diagramme circulaire
sentiment_pct = df['sentiment'].value_counts(normalize=True) * 100
labels = ['Négatif', 'Neutre', 'Positif']
ax2.pie(sentiment_pct.values, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Répartition en Pourcentages', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nPourcentages des sentiments:")
for i, label in enumerate(labels):
    print(f"{label}: {sentiment_pct[i]:.2f}%")

# 5.2 Longueur des commentaires
print("\n--- 5.2 Analyse de la longueur des commentaires ---")

df['comment_length'] = df['clean_comment'].apply(len)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogramme
axes[0].hist(df['comment_length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Longueur (caractères)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Fréquence', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution de la Longueur des Commentaires', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Boxplot par sentiment
df_plot = df[['sentiment', 'comment_length']].copy()
df_plot['sentiment_label'] = df_plot['sentiment'].map({0: 'Négatif', 1: 'Neutre', 2: 'Positif'})
sns.boxplot(x='sentiment_label', y='comment_length', data=df_plot, palette=colors, ax=axes[1])
axes[1].set_xlabel('Sentiment', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Longueur (caractères)', fontsize=11, fontweight='bold')
axes[1].set_title('Longueur par Sentiment', fontsize=12, fontweight='bold')

# Violin plot
sns.violinplot(x='sentiment_label', y='comment_length', data=df_plot, palette=colors, ax=axes[2])
axes[2].set_xlabel('Sentiment', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Longueur (caractères)', fontsize=11, fontweight='bold')
axes[2].set_title('Distribution de la Longueur (Violin Plot)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Statistiques
print("\nStatistiques de longueur par sentiment:")
display(df.groupby('sentiment')['comment_length'].describe())

# 5.3 Nuages de mots (Word Clouds)
print("\n--- 5.3 Nuages de mots par sentiment ---")

def create_wordcloud(text, title, color):
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap=color,
                         max_words=100).generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Nuages de mots par sentiment
sentiments_config = [
    (0, 'Négatif', 'Reds'),
    (1, 'Neutre', 'Oranges'),
    (2, 'Positif', 'Greens')
]

for sentiment_val, label, colormap in sentiments_config:
    text = ' '.join(df[df['sentiment'] == sentiment_val]['clean_comment'].astype(str))
    create_wordcloud(text, f'Nuage de Mots - Sentiment {label}', colormap)

# 5.4 Mots les plus fréquents par sentiment
print("\n--- 5.4 Mots les plus fréquents par sentiment ---")

def get_top_words(text_series, n=20):
    words = ' '.join(text_series.astype(str)).lower()
    words = re.findall(r'\b[a-z]+\b', words)
    return Counter(words).most_common(n)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (sentiment_val, label, color) in enumerate(sentiments_config):
    top_words = get_top_words(df[df['sentiment'] == sentiment_val]['clean_comment'])
    words, counts = zip(*top_words)
    
    axes[idx].barh(words, counts, color=color.replace('s', ''))
    axes[idx].set_xlabel('Fréquence', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'Top 20 Mots - {label}', fontsize=13, fontweight='bold')
    axes[idx].invert_yaxis()
    axes[idx].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# 5.5 Nombre de mots par commentaire
print("\n--- 5.5 Nombre de mots par commentaire ---")

df['word_count'] = df['clean_comment'].astype(str).apply(lambda x: len(x.split()))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Distribution
ax1.hist(df['word_count'], bins=50, color='teal', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Nombre de mots', fontsize=11, fontweight='bold')
ax1.set_ylabel('Fréquence', fontsize=11, fontweight='bold')
ax1.set_title('Distribution du Nombre de Mots', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Par sentiment
df_plot = df[['sentiment', 'word_count']].copy()
df_plot['sentiment_label'] = df_plot['sentiment'].map({0: 'Négatif', 1: 'Neutre', 2: 'Positif'})
sns.boxplot(x='sentiment_label', y='word_count', data=df_plot, palette=colors, ax=ax2)
ax2.set_xlabel('Sentiment', fontsize=11, fontweight='bold')
ax2.set_ylabel('Nombre de mots', fontsize=11, fontweight='bold')
ax2.set_title('Nombre de Mots par Sentiment', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

# 6.1 Extraction de features textuelles
print("\n--- Extraction de features textuelles ---")

# Nombre de caractères uniques
df['unique_chars'] = df['clean_comment'].astype(str).apply(lambda x: len(set(x)))

# Nombre de mots en majuscules
df['upper_words'] = df['clean_comment'].astype(str).apply(
    lambda x: len([w for w in x.split() if w.isupper() and len(w) > 1])
)

# Ponctuation
df['exclamation_count'] = df['clean_comment'].astype(str).apply(lambda x: x.count('!'))
df['question_count'] = df['clean_comment'].astype(str).apply(lambda x: x.count('?'))

# Ratio majuscules/minuscules
df['upper_ratio'] = df['clean_comment'].astype(str).apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
)

print("✓ Features créées:")
print("  - word_count: Nombre de mots")
print("  - unique_chars: Nombre de caractères uniques")
print("  - upper_words: Nombre de mots en majuscules")
print("  - exclamation_count: Nombre de points d'exclamation")
print("  - question_count: Nombre de points d'interrogation")
print("  - upper_ratio: Ratio de lettres majuscules")

print("\nAperçu des nouvelles features:")
display(df[['word_count', 'unique_chars', 'upper_words', 
            'exclamation_count', 'question_count', 'upper_ratio']].head(10))

# 6.2 Vectorisation TF-IDF
print("\n--- Vectorisation TF-IDF ---")

# Vectorisation avec TF-IDF
print("Application de TF-IDF (max 1000 features, unigrammes et bigrammes)...")
tfidf = TfidfVectorizer(max_features=1000, 
                       stop_words='english', 
                       ngram_range=(1, 2),
                       min_df=2,
                       max_df=0.9)

X_text = tfidf.fit_transform(df['clean_comment'].astype(str))
print(f"✓ Shape des features textuelles: {X_text.shape}")

# Features numériques
numeric_features = df[['word_count', 'unique_chars', 'upper_words', 
                       'exclamation_count', 'question_count', 'upper_ratio']].values
print(f"✓ Shape des features numériques: {numeric_features.shape}")

# Combinaison des features
X_combined = hstack([X_text, numeric_features])
print(f"✓ Shape totale des features: {X_combined.shape}")

# Variable cible
y = df['sentiment']
print(f"✓ Shape de la variable cible: {y.shape}")

# 6.3 Visualisation de la corrélation des features numériques
print("\n--- Corrélation des features numériques ---")

numeric_df = df[['word_count', 'comment_length', 'unique_chars', 
                 'upper_words', 'exclamation_count', 'question_count', 
                 'upper_ratio', 'sentiment']]

plt.figure(figsize=(10, 8))
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation des Features Numériques', 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# =============================================================================
# 7. GESTION DU DÉSÉQUILIBRE DES CLASSES
# =============================================================================

print("\n" + "="*80)
print("GESTION DU DÉSÉQUILIBRE DES CLASSES")
print("="*80)

# 7.1 Vérification du déséquilibre
print("\n--- Analyse du déséquilibre ---")
class_distribution = y.value_counts().sort_index()
print("Distribution des classes:")
for i, count in enumerate(class_distribution):
    label = ['Négatif', 'Neutre', 'Positif'][i]
    print(f"  {label} ({i}): {count} ({count/len(y)*100:.2f}%)")

ratio = class_distribution.max() / class_distribution.min()
print(f"\nRatio max/min: {ratio:.2f}")

if ratio > 1.5:
    print("⚠ Déséquilibre détecté - Application de SMOTE recommandée")
else:
    print("✓ Classes relativement équilibrées")

# 7.2 Application de SMOTE
print("\n--- Application de SMOTE ---")

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_combined, y)

print(f"Échantillons avant SMOTE: {X_combined.shape[0]}")
print(f"Échantillons après SMOTE: {X_balanced.shape[0]}")

print("\nDistribution après SMOTE:")
balanced_dist = pd.Series(y_balanced).value_counts().sort_index()
for i, count in enumerate(balanced_dist):
    label = ['Négatif', 'Neutre', 'Positif'][i]
    print(f"  {label} ({i}): {count} ({count/len(y_balanced)*100:.2f}%)")

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Avant SMOTE
class_distribution.plot(kind='bar', ax=ax1, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('Distribution AVANT SMOTE', fontsize=13, fontweight='bold')
ax1.set_xlabel('Sentiment', fontsize=11, fontweight='bold')
ax1.set_ylabel('Nombre d\'échantillons', fontsize=11, fontweight='bold')
ax1.set_xticklabels(['Négatif', 'Neutre', 'Positif'], rotation=0)
ax1.grid(axis='y', alpha=0.3)

# Après SMOTE
balanced_dist.plot(kind='bar', ax=ax2, color=colors, alpha=0.7, edgecolor='black')
ax2.set_title('Distribution APRÈS SMOTE', fontsize=13, fontweight='bold')
ax2.set_xlabel('Sentiment', fontsize=11, fontweight='bold')
ax2.set_ylabel('Nombre d\'échantillons', fontsize=11, fontweight='bold')
ax2.set_xticklabels(['Négatif', 'Neutre', 'Positif'], rotation=0)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Mise à jour des variables
X = X_balanced
y = y_balanced

print("\n✓ Données équilibrées et prêtes pour la modélisation")

# =============================================================================
# 8. SÉPARATION TRAIN/TEST
# =============================================================================

print("\n" + "="*80)
print("SÉPARATION DES DONNÉES")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Taille du set d'entraînement: {X_train.shape[0]} échantillons")
print(f"Taille du set de test: {X_test.shape[0]} échantillons")
print(f"Ratio train/test: {X_train.shape[0]/X_test.shape[0]:.2f}")

# =============================================================================
# 9. MODÉLISATION
# =============================================================================

print("\n" + "="*80)
print("ENTRAÎNEMENT DES MODÈLES")
print("="*80)

# Dictionnaire pour stocker les résultats
results = {}
models_dict = {}
predictions = {}

# Fonction utilitaire pour afficher la matrice de confusion
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Négatif', 'Neutre', 'Positif'],
                yticklabels=['Négatif', 'Neutre', 'Positif'],
                cbar_kws={'label': 'Nombre de prédictions'})
    plt.title(f'Matrice de Confusion - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie Classe', fontsize=12, fontweight='bold')
    plt.xlabel('Classe Prédite', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 9.1 Logistic Regression
print("\n" + "-"*80)
print("9.1 LOGISTIC REGRESSION")
print("-"*80)
print("""
Description: Modèle linéaire qui estime les probabilités d'appartenance aux classes
en appliquant la fonction sigmoïde à une combinaison linéaire des features.

Avantages: Simple, rapide, interprétable, efficace pour les données linéairement séparables
Inconvénients: Suppose une relation linéaire, limité pour les patterns complexes
""")

lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
print("Entraînement en cours...")
lr_model.fit(X_train, y_train)
y_lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_lr_pred)
results['Logistic Regression'] = lr_accuracy
models_dict['Logistic Regression'] = lr_model
predictions['Logistic Regression'] = y_lr_pred

print(f"\n✓ Accuracy: {lr_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_lr_pred, 
                          target_names=['Négatif', 'Neutre', 'Positif'],
                          digits=4))
plot_confusion_matrix(y_test, y_lr_pred, 'Logistic Regression')

# 9.2 Decision Tree
print("\n" + "-"*80)
print("9.2 DECISION TREE")
print("-"*80)
print("""
Description: Arbre qui divise récursivement l'espace des features selon des règles
de décision simples pour créer des régions homogènes.

Avantages: Facile à interpréter, gère les non-linéarités, pas de normalisation nécessaire
Inconvénients: Tendance au surapprentissage, instable aux petites variations
""")

dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=20)
print("Entraînement en cours...")
dt_model.fit(X_train, y_train)
y_dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_dt_pred)
results['Decision Tree'] = dt_accuracy
models_dict['Decision Tree'] = dt_model
predictions['Decision Tree'] = y_dt_pred

print(f"\n✓ Accuracy: {dt_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_dt_pred,
                          target_names=['Négatif', 'Neutre', 'Positif'],
                          digits=4))
plot_confusion_matrix(y_test, y_dt_pred, 'Decision Tree')

# 9.3 K-Nearest Neighbors
print("\n" + "-"*80)
print("9.3 K-NEAREST NEIGHBORS (KNN)")
print("-"*80)
print("""
Description: Classe une observation en se basant sur les k observations les plus proches
dans l'espace des features (apprentissage paresseux).

Avantages: Simple, intuitif, pas d'hypothèse sur la distribution des données
Inconvénients: Coûteux en calcul, sensible à l'échelle et aux dimensions
""")

knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, n_jobs=-1)
print("Entraînement en cours...")
knn_model.fit(X_train, y_train)
y_knn_pred = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_knn_pred)
results['KNN'] = knn_accuracy
models_dict['KNN'] = knn_model
predictions['KNN'] = y_knn_pred

print(f"\n✓ Accuracy: {knn_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_knn_pred,
                          target_names=['Négatif', 'Neutre', 'Positif'],
                          digits=4))
plot_confusion_matrix(y_test, y_knn_pred, 'K-Nearest Neighbors')

# 9.4 Gaussian Naive Bayes
print("\n" + "-"*80)
print("9.4 GAUSSIAN NAIVE BAYES")
print("-"*80)
print("""
Description: Applique le théorème de Bayes en supposant que les features suivent
une distribution normale et sont conditionnellement indépendantes.

Avantages: Très rapide, efficace avec peu de données, fonctionne bien en haute dimension
Inconvénients: Hypothèse d'indépendance forte, suppose distribution gaussienne
""")

# Conversion en dense array (nécessaire pour GaussianNB)
X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

gnb_model = GaussianNB()
print("Entraînement en cours...")
gnb_model.fit(X_train_dense, y_train)
y_gnb_pred = gnb_model.predict(X_test_dense)

gnb_accuracy = accuracy_score(y_test, y_gnb_pred)
results['Gaussian NB'] = gnb_accuracy
models_dict['Gaussian NB'] = gnb_model
predictions['Gaussian NB'] = y_gnb_pred

print(f"\n✓ Accuracy: {gnb_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_gnb_pred,
                          target_names=['Négatif', 'Neutre', 'Positif'],
                          digits=4))
plot_confusion_matrix(y_test, y_gnb_pred, 'Gaussian Naive Bayes')

# 9.5 Multinomial Naive Bayes
print("\n" + "-"*80)
print("9.5 MULTINOMIAL NAIVE BAYES")
print("-"*80)
print("""
Description: Variante de Naive Bayes adaptée pour les features discrètes (comptages),
particulièrement efficace pour la classification de texte.

Avantages: Très rapide, excellent pour le texte, gère bien les grands vocabulaires
Inconvénients: Hypothèse d'indépendance, nécessite des features positives
""")

mnb_model = MultinomialNB()
print("Entraînement en cours...")
mnb_model.fit(X_train, y_train)
y_mnb_pred = mnb_model.predict(X_test)

mnb_accuracy = accuracy_score(y_test, y_mnb_pred)
results['Multinomial NB'] = mnb_accuracy
models_dict['Multinomial NB'] = mnb_model
predictions['Multinomial NB'] = y_mnb_pred

print(f"\n✓ Accuracy: {mnb_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_mnb_pred,
                          target_names=['Négatif', 'Neutre', 'Positif'],
                          digits=4))
plot_confusion_matrix(y_test, y_mnb_pred, 'Multinomial Naive Bayes')

# 9.6 Support Vector Classifier
print("\n" + "-"*80)
print("9.6 SUPPORT VECTOR CLASSIFIER (SVC)")
print("-"*80)
print("""
Description: Trouve l'hyperplan optimal qui maximise la marge entre les classes.
Utilise le kernel trick pour gérer les relations non-linéaires.

Avantages: Efficace en haute dimension, robuste, flexible avec différents kernels
Inconvénients: Coûteux pour de grands datasets, difficile à interpréter
""")

svc_model = SVC(kernel='rbf',
