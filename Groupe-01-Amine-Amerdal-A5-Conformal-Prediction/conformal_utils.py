import numpy as np

class AdaptiveConformalInference:
    """
    ImplémenteAdaptive Conformal Inference pour les séries temporelles.
    """
    def __init__(self, alpha=0.05, gamma=0.01):
        self.alpha = alpha   # Taux d'erreur cible 
        self.gamma = gamma   # Vitesse d'apprentissage de l'adaptation
        self.q = alpha       # Quantile initial 
        self.history_q = []

    def update_quantile(self, error_occurred):
        """
        Ajuste le quantile dynamiquement. 
        Si une erreur (outlier) est détectée, on augmente la prudence.
        """
        # Formule ACI : q_{t+1} = q_t + gamma * (alpha - err_t)
        # err_t = 1 si la valeur réelle est hors intervalle, 0 sinon.
        self.q = self.q + self.gamma * (self.alpha - float(error_occurred))
        # On contraint q entre 0 et 1
        self.q = np.clip(self.q, 0, 1)
        return self.q

    def get_interval(self, prediction, scores, current_q):
        """Calcule l'intervalle basé sur le quantile actuel des scores."""
        q_val = np.quantile(scores, 1 - current_q, method='higher')
        return prediction - q_val, prediction + q_val
