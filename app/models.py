from pydantic import BaseModel
from typing import Dict


class SampleData(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    timestamp: int


class SensorInput(BaseModel):
    sensors: Dict[str, Dict[str, SampleData]]
