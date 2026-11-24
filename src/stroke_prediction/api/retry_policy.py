import math


def calcular_atraso(
        tentativa: int,
        atraso_inicial: float = 1.0,
        atraso_maximo: float = 30.0,
) -> float:
    if tentativa < 1:
        raise ValueError("A tentativa deve começar em 1.")

    if not math.isfinite(atraso_inicial) or not math.isfinite(atraso_maximo):
        raise ValueError("Os atrasos devem ser finitos.")

    if atraso_inicial <= 0 or atraso_maximo <= 0:
        raise ValueError("Os atrasos devem ser maiores que zero.")

    if atraso_maximo < atraso_inicial:
        raise ValueError(
            "O atraso máximo deve ser maior ou igual ao inicial."
        )

    return min(
        atraso_inicial * 2 ** (tentativa - 1),
        atraso_maximo,
    )