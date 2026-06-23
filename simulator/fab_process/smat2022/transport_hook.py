"""
AMRTransportHook — PySCFabSim 공정 완료 이벤트 → AMR 반송 요청 변환.

동작 원리:
    PySCFabSim의 on_step_done()이 호출될 때마다
    (from_machine_family, to_machine_family) 쌍을
    SMAT2022 포트 이름으로 변환해 FABSimulator 큐에 주입한다.

    매핑 경로:
        SMT2020 STNFAM (예: "WE_FE_84")
            == SMAT2022 port_locations의 'tool' 필드
            → FABSimulator.tool_to_ports["WE_FE_84"] = ["WE0101", "WE0102", ...]
            → 그 중 랜덤 1개 선택

사용법:
    import sys; sys.path.insert(0, 'external/PySCFabSim')
    from simulation.read import read_all
    from simulation.file_instance import FileInstance
    from simulator.amr.simulator import FABSimulator
    from simulator.fab_process.smat2022.transport_hook import AMRTransportHook

    fab_sim = FABSimulator(grid_map, port_locations, auto_dispatch=True)
    hook    = AMRTransportHook(fab_sim)
    files   = read_all('external/PySCFabSim/datasets/SMT2020_LVHM')
    pyscfab = FileInstance(files, run_to=3600*24*30, lot_for_machine=True,
                           plugins=[hook])
"""

import sys
import os

_pyscfabsim_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "external", "PySCFabSim")
)
if _pyscfabsim_path not in sys.path:
    sys.path.insert(0, _pyscfabsim_path)

from simulation.plugins.interface import IPlugin

from simulator.amr.simulator import FABSimulator


class AMRTransportHook(IPlugin):
    """
    PySCFabSim IPlugin 구현체.

    Lot이 한 공정 단계를 마칠 때마다:
        step.family (직전 기계 타입)  → from_port
        lot.actual_step.family (다음)  → to_port
    로 변환해 FABSimulator에 반송 요청을 주입한다.
    """

    def __init__(self, fab_sim: FABSimulator):
        self.fab_sim = fab_sim

    def on_step_done(self, instance, lot, step):
        """
        lot이 step(이전 단계)을 완료하고 lot.actual_step(다음 단계)으로
        이동할 시점에 호출된다.

        step이 None → lot 최초 투입 (공장 입구에서 첫 장비로 이동)
        lot.actual_step이 None → 전체 공정 완료 (반송 불필요)
        """
        if lot.actual_step is None:
            return

        from_family = step.family if step is not None else "STOCKER"
        to_family   = lot.actual_step.family

        from_port = self.fab_sim._resolve_port(from_family)
        to_port   = self.fab_sim._resolve_port(to_family)

        if from_port is None or to_port is None or from_port == to_port:
            return

        # lot.priority = 데이터셋 PRIOR tier. SMT2020_LVHM 분포: 10=98%, 20=1.8%, 30=0.1%.
        # PRIOR>0(이전 기준)은 모든 lot이 10 이상이라 100% HOT으로 붕괴됨 → 우선순위 신호 소멸.
        # PRIOR>10(즉 20·30 tier)을 HOT으로 정의 → ~2% HOT, 실제 fab 핫랏 비율과 일치.
        priority = "HOT" if getattr(lot, "priority", 0) > 10 else "NORMAL"

        if getattr(instance, "external_transport", False):
            # closed-loop: 배송 완료 시 이 lot을 다음 장비에 release하도록 콜백 연결.
            # 요청이 실제 생성된 경우에만 lot을 hold(_awaiting_transport=True).
            req = self.fab_sim.add_request(
                from_port, to_port, priority,
                on_complete=lambda r, _inst=instance, _lot=lot: _inst.free_up_lots_after_transport(_lot),
            )
            if req is not None:
                lot._awaiting_transport = True
        else:
            self.fab_sim.add_request(from_port, to_port, priority)
