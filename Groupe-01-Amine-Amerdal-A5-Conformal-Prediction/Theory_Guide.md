Conformal Prediction & Adaptive Inference pour le Risk Management

Auteur : Amerdal Amine ING4 GROUPE1


## 1. Introduction et Problématique
Dans le domaine de la finance de marché, la gestion des risques repose traditionnellement sur la **Value-at-Risk (VaR)**. Cependant, les méthodes classiques (paramétriques ou historiques) souffrent de deux faiblesses majeures :
1.  Elles supposent souvent une distribution normale des rendements, ignorant les "queues épaisses" (fat tails) et les crises.
2.  Elles manquent de garanties statistiques formelles sur les données hors-échantillon.

Ce projet implémente la **Conformal Prediction (CP)**, un cadre de travail "Distribution-Free" qui permet de transformer n'importe quel algorithme d'IA en un estimateur d'incertitude avec une **couverture garantie à $1 - \alpha$**.

## 2. Fondements Théoriques

### 2.1 La Conformal Prediction (CP)
La CP ne cherche pas à prédire une valeur unique $\hat{y}$, mais un ensemble de prédiction $C(x)$ tel que :
$$P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$$

Pour ce faire, nous utilisons un **score de non-conformité** $s_i$, qui mesure le degré "d'étrangeté" d'une nouvelle observation par rapport au modèle entraîné. Dans ce projet, nous utilisons le résidu absolu :
$$s_i = |y_i - \hat{y}_i|$$



### 2.2 Adaptive Conformal Inference (ACI) - Niveau "Excellent"
Le problème des séries temporelles financières est la **non-stationnarité** (la volatilité de 2024 n'est pas celle de 2021). La méthode *Split Conformal* classique échoue si les données ne sont pas IID (indépendantes et identiquement distribuées).

L'**ACI** (Gibbs & Candès, 2021) corrige cela en ajustant dynamiquement le quantile de confiance $q$ à chaque pas de temps $t$. La règle de mise à jour est :
$$q_{t+1} = q_t + \gamma (\alpha - \text{err}_t)$$
Où :
* $\gamma$ est le taux d'apprentissage (learning rate).
* $\text{err}_t = 1$ si la valeur réelle est sortie de l'intervalle, $0$ sinon.

Si le modèle se trompe, l'ACI augmente la prudence en élargissant l'intervalle.

## 3. Implémentation Technique

### 3.1 Pipeline de Données
* **Source** : Yahoo Finance (Ticker : `^GSPC` - S&P 500).
* **Features** : "Lag features" (rendements des 5 jours précédents) pour capturer l'auto-corrélation.
* **Algorithme** : Gradient Boosting Regressor (performant pour capturer les non-linéarités).

### 3.2 Architecture du Code
Le projet est modularisé pour respecter les standards d'ingénierie :
* `conformal_utils.py` : Contient la classe `AdaptiveConformalInference` gérant la logique de mise à jour du quantile.
* `main.py` : Gère le flux complet, de l'acquisition des données à la visualisation finale.

## 4. Analyse des Résultats et Comparaison

### 4.1 Performance de Couverture
Le modèle atteint une couverture réelle proche de la cible ($1 - \alpha = 95\%$). Contrairement aux méthodes bayésiennes qui nécessitent un choix de "Prior" complexe, la CP garantit ce taux sans aucune hypothèse sur la loi des rendements.

### 4.2 Analyse des Périodes de Crise
Pendant les pics de volatilité, on observe que :
1.  Les résidus augmentent brusquement.
2.  L'algorithme **ACI** réagit immédiatement en augmentant le quantile de sécurité.
3.  L'intervalle de risque s'élargit, protégeant ainsi le portefeuille d'une sous-estimation du risque.



## 5. Conclusion
L'application de la Conformal Prediction au Risk Management permet de passer d'un modèle de "prédiction aveugle" à un système de **pilotage du risque dynamique**. Cette approche est particulièrement robuste pour les actifs volatils (Crypto-actifs, indices en période de crise) car elle s'auto-calibre en continu.

## 6. Analyse Comparative : Pourquoi la CP ?

Pour justifier le choix de la Conformal Prediction, nous l'avons comparée aux méthodes standards de l'industrie.

### 6.1 CP vs Méthodes Bayésiennes
* **Méthodes Bayésiennes** : Elles nécessitent de définir un "Prior" (une croyance a priori sur la distribution). Si le Prior est faux (ce qui arrive souvent lors d'un krach boursier), les intervalles de confiance sont totalement erronés.
* **Conformal Prediction** : Elle est **"Distribution-Free"**. Elle ne suppose rien sur la forme des données. Sa validité est prouvée mathématiquement même si le modèle de base est médiocre.

### 6.2 CP vs Quantile Regression
La Quantile Regression tente d'estimer directement les quantiles (ex: VaR 95%). Cependant :
1. Elle n'offre aucune garantie de couverture sur les données de test (out-of-sample).
2. Elle est sujette à l' "Overfitting".
**La CP**, en revanche, utilise un ensemble de calibration indépendant, ce qui garantit que le taux d'erreur sur les nouvelles données sera exactement de $\alpha$.

## 7. Extension : Application au Portefeuille (CPPS)
L'objectif "Excellent" mentionne le **CPPS (Conformal Predictive Portfolio Selection)**. 
Dans ce cadre, la CP ne sert plus seulement à prédire un actif, mais à sélectionner les actifs dont l'incertitude est la plus faible. 

**Logique d'implémentation :**
1. Nous générons des intervalles CP pour $N$ actifs.
2. Nous calculons le ratio Rendement/Incertitude (similaire au ratio de Sharpe, mais en remplaçant la Volatilité par la largeur de l'intervalle CP).
3. On alloue plus de poids aux actifs ayant les intervalles les plus "fins" et stables. Cela permet une gestion de portefeuille beaucoup plus robuste en période de crise.
