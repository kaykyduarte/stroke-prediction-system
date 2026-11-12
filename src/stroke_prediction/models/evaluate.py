import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    precision_score,
    precision_recall_curve,
)


def calcular_metricas(y_real, probabilidades, limiar):
    """Calcula metricas para um limiar especifico."""
    y_pred = (probabilidades >= limiar).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y_real,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "Limiar": limiar,
        "Accuracy": accuracy_score(y_real, y_pred),
        "Recall": recall_score(y_real, y_pred, zero_division=0),
        "Precision": precision_score(y_real, y_pred, zero_division=0),
        "VN": int(vn),
        "FP": int(fp),
        "FN": int(fn),
        "VP": int(vp),
    }


def selecionar_limiar(y_real, probabilidades, recall_minimo=0.80):
    """seleciona a maior precision entre os limiares com recall suficiente."""
    if (y_real == 1).sum() == 0:
        return None

    precisao, recall, limiares = precision_recall_curve(
        y_real,
        probabilidades,
    )

    candidatos = pd.DataFrame({
        "Limiar": limiares,
        "Precision": precisao[:-1],
        "Recall": recall[:-1]
    })

    candidatos = candidatos[candidatos["Recall"] >= recall_minimo]

    if candidatos.empty:
        return None

    melhor = candidatos.sort_values(
        ["Precision", "Recall", "Limiar"],
        ascending=[False, False, False],
    ).iloc[0]

    return float(melhor["Limiar"])