import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from conformal_utils import AdaptiveConformalInference

# CONFIGURATION
TICKER = "^GSPC" # S&P 500
START_DATE = "2018-01-01" 
END_DATE = "2024-01-01"
ALPHA = 0.05 # On veut 95% de couverture 

# RÉCUPÉRATION & PRÉPARATION 
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
df['Returns'] = df['Close'].pct_change()
for i in range(1, 6): # Lag features (5 jours passés)
    df[f'Lag_{i}'] = df['Returns'].shift(i)
df = df.dropna()

X = df[[f'Lag_{i}' for i in range(1, 6)]].values
y = df['Returns'].values

# Split Chronologique 
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# MODÈLE DE BASE 
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# ADAPTIVE CONFORMAL INFERENCE 
aci = AdaptiveConformalInference(alpha=ALPHA, gamma=0.02)
predictions = model.predict(X_test)

# On utilise les résidus du train comme base initiale de scores
initial_scores = np.abs(y_train - model.predict(X_train))

lower_bounds, upper_bounds = [], []
current_scores = list(initial_scores)

for i in range(len(y_test)):
    # Calcul de l'intervalle avec le quantile dynamique
    low, high = aci.get_interval(predictions[i], current_scores, aci.q)
    
    # Vérification 
    error = (y_test[i] < low) or (y_test[i] > high)
    
    # Mise à jour de l'algo ACI 
    aci.update_quantile(error)
    
    # Sauvegarde
    lower_bounds.append(low)
    upper_bounds.append(high)
    # On ajoute le nouveau score pour rester "up-to-date" 
    current_scores.append(np.abs(y_test[i] - predictions[i]))
    if len(current_scores) > 500: current_scores.pop(0) # Fenêtre glissante

# ÉVALUATION 
coverage = np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds))
print(f"--- RÉSULTATS ---")
print(f"Couverture cible : {1-ALPHA:.2%}")
print(f"Couverture réelle obtenue : {coverage:.2%}")

# VISUALISATION 
plt.figure(figsize=(15, 7))
plt.plot(y_test[-200:], label="Rendement Réel", color='black', lw=1, alpha=0.6)
plt.fill_between(range(200), np.array(lower_bounds)[-200:], np.array(upper_bounds)[-200:], 
                 color='red', alpha=0.2, label=f'Intervalle de Risque (ACI {1-ALPHA:.0%})')
plt.title(f"Risk Management via ACI sur {TICKER} (Focus Volatilité)")
plt.legend()
plt.savefig("resultats_conformal.png") 
plt.show()
