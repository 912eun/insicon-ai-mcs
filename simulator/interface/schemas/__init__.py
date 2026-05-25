from dataclasses import dataclass
from typing import Literal


@dataclass
class TransportRequest:
    lot_id: str
    from_equipment: str
    to_equipment: str
    priority: Literal["hot", "normal"]
    request_time: float


@dataclass
class DispatchCommand:
    request_id: str
    amr_id: str
    pickup: str
    dropoff: str


@dataclass
class AMRStatus:
    amr_id: str
    position: tuple[float, float]
    status: Literal["idle", "moving", "loading", "unloading"]
    current_task: str | None
