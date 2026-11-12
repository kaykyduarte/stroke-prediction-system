from sklearn.metrics import average_precision_score

from src.stroke_prediction.models.evaluate import calcular_metricas


def avaliar_final(pipeline, X_test, y_test, limiar):
    """Avalia um pipeline ja treinado usando um limiar previamente definido."""

    indice_classe_1 = list(pipeline.classes_).index(1)

    probabilidades = pipeline.predict_proba(
        X_test
    )[:, indice_classe_1]

    resultados = calcular_metricas(
        y_test,
        probabilidades,
        limiar,
    )

    resultados["Average Precision"] = average_precision_score(
        y_test,
        probabilidades,
    )

    resultados["Matriz de confusao"] = [
        [resultados["VN"], resultados["FP"]],
        [resultados["FN"], resultados["VP"]],
    ]

    return resultados