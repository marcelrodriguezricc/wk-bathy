from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class AOI:
    name: str
    filename: str
    lat: str
    lon: str
    crm_link: str
    crm_local: str = "null"
    crm_date: str = "null"
    bbox_lat: float = 0.25
    bbox_lon: float = 0.25
    date_offset: int = 0
    shoreline_buffer: int = 0

    data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    selected_dates: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class params:
    num_days: int = 20