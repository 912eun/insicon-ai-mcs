from abc import ABC, abstractmethod
from simulator.interface.schemas import TransportRequest, DispatchCommand, AMRStatus


class BaseDispatcher(ABC):
    """4인 RL 전략이 공통으로 구현해야 하는 인터페이스."""

    @abstractmethod
    def select_amr(
        self,
        request: TransportRequest,
        amrs: list[AMRStatus],
    ) -> DispatchCommand:
        """반송 요청에 대해 어떤 AMR을 배정할지 결정."""
        ...

    @abstractmethod
    def update(self, obs: dict, reward: float, done: bool) -> None:
        """한 스텝 후 에이전트 업데이트 (학습 루프에서 호출)."""
        ...
