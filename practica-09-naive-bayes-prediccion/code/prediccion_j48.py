"""
Predicción de riesgo crediticio con el árbol de decisión J48 (C4.5).
Reglas extraídas del modelo entrenado en Weka sobre loan_risk_dataset.arff.

Ejecución:
    python3 prediccion_j48.py
"""


def predecir_credito(age: str, income: str, loan_history: str) -> str:
    """Devuelve 'risky' o 'safe' según el árbol J48.

    age          : middle-aged | senior | young
    income       : high | low | medium
    loan_history : average | good | poor
    """
    if loan_history == "good":
        return "safe"
    if loan_history == "poor":
        return "risky"
    # loan_history == 'average'
    if income == "low":
        return "risky"
    # income high o medium
    if age == "young":
        return "risky"
    return "safe"


def main():
    print("=== Evaluación de Riesgo Crediticio (árbol J48) ===")
    age = input("→ Edad (middle-aged / senior / young): ").strip().lower()
    income = input("→ Ingreso (high / low / medium): ").strip().lower()
    hist = input("→ Historial crediticio (average / good / poor): ").strip().lower()

    decision = predecir_credito(age, income, hist)
    etiqueta = "SEGURO (otorgar crédito)" if decision == "safe" else "RIESGOSO (no otorgar)"
    print(f"\nDecisión del modelo: {decision} → {etiqueta}")


if __name__ == "__main__":
    main()
