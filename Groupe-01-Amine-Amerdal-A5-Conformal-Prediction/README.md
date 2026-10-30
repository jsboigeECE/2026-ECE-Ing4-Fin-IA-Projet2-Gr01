# Conformal Prediction for Financial Risk Management (Sujet A5)

## Présentation du Projet
Ce projet implémente une approche moderne de la quantification de l'incertitude dans la prédiction de rendements financiers. Contrairement aux modèles de Machine Learning traditionnels qui fournissent des prédictions "ponctuelles", nous utilisons la **Conformal Prediction (CP)** pour générer des **intervalles de confiance statistiquement garantis**.

**Objectif :** Remplacer les mesures de risque classiques (souvent basées sur des hypothèses de normalité erronées) par une méthode "Distribution-Free" capable de s'adapter à la volatilité réelle du marché.

## Méthodologie : Split Conformal Prediction
Pour garantir la robustesse du modèle, nous suivons un protocole rigoureux :
1. **Entraînement :** Un modèle (Random Forest ou Gradient Boosting) apprend les patterns historiques du S&P 500.
2. **Calibration :** Utilisation d'un set de données indépendant pour calculer les scores de non-conformité $s_i = |y_i - \hat{y}_i|$.
3. **Inférence avec Garantie :** Calcul du quantile de calibration pour définir l'intervalle $[ \hat{y} - q, \hat{y} + q ]$.

## Indicateurs de Performance
* **Coverage Error (Taux de Couverture) :** Le pourcentage de données réelles tombant effectivement dans l'intervalle (cible : $1 - \alpha$).
* **Average Width (Efficacité) :** La largeur moyenne de l'intervalle. Un modèle précis est un modèle qui garantit la couverture avec l'intervalle le plus fin possible.

##  Structure du Repository
* `main.py` : Pipeline complet (Data -> Model -> Conformal -> Plots).
* `conformal_utils.py` : Logique mathématique de calcul des scores.
* `Theory_Guide.md` : Documentation technique approfondie (Remplacement présentation).
