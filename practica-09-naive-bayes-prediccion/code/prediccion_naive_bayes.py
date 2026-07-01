"""
Predicción de riesgo crediticio con Naive Bayes.
Tablas de probabilidad reproducidas de Weka (estimador de Laplace, +1 por celda)
sobre loan_risk_dataset.arff (1000 instancias).

Ejecución:
    python3 prediccion_naive_bayes.py
"""

# Probabilidades a priori
P = {"risky": 497 / 1000, "safe": 503 / 1000}

# Conteos con corrección de Laplace (+1) y totales por clase
totales = {"risky": 500.0, "safe": 506.0}

probs = {
    "risky": {
        "Age": {"middle-aged": 156 / 500, "senior": 139 / 500, "young": 205 / 500},
        "Income": {"high": 142 / 500, "low": 234 / 500, "medium": 124 / 500},
        "Loan_History": {"average": 178 / 500, "good": 1 / 500, "poor": 321 / 500},
    },
    "safe": {
        "Age": {"middle-aged": 192 / 506, "senior": 201 / 506, "young": 113 / 506},
        "Income": {"high": 205 / 506, "low": 127 / 506, "medium": 174 / 506},
        "Loan_History": {"average": 149 / 506, "good": 356 / 506, "poor": 1 / 506},
    },
}


def predecir_credito(age: str, income: str, loan_history: str):
    def score(clase: str) -> float:
        return (
            P[clase]
            * probs[clase]["Age"][age]
            * probs[clase]["Income"][income]
            * probs[clase]["Loan_History"][loan_history]
        )

    s_risky, s_safe = score("risky"), score("safe")
    total = s_risky + s_safe
    p_risky, p_safe = s_risky / total, s_safe / total
    decision = "risky" if p_risky > p_safe else "safe"
    return decision, p_risky, p_safe


def main():
    print("=== Evaluación de Riesgo Crediticio (Naive Bayes) ===")
    age = input("→ Edad (middle-aged / senior / young): ").strip().lower()
    income = input("→ Ingreso (high / low / medium): ").strip().lower()
    hist = input("→ Historial crediticio (average / good / poor): ").strip().lower()

    decision, p_risky, p_safe = predecir_credito(age, income, hist)
    print(f"\nProbabilidad RISKY: {p_risky:.4f}")
    print(f"Probabilidad SAFE : {p_safe:.4f}")
    etiqueta = "SEGURO (otorgar crédito)" if decision == "safe" else "RIESGOSO (no otorgar)"
    print(f"Decisión del modelo: {decision} → {etiqueta}")


if __name__ == "__main__":
    main()
