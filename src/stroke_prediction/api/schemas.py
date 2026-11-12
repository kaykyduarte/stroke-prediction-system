from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrokeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female", "Other"]
    age: float = Field(ge=0, le=125)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal[
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked",
    ]
    Residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float = Field(gt=0)
    bmi: float | None = Field(gt=0)
    smoking_status: Literal[
    "formerly smoked",
    "never smoked",
    "smokes",
    "Unknown",
]


class StrokeOutput(BaseModel):
    score: float = Field(ge=0, le=1)
    classificacao: Literal[0, 1]
    versao_artefato: str