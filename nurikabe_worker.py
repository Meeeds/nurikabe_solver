import queue
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from nurikabe_model import NurikabeModel
from nurikabe_rules import NurikabeSolver
from nurikabe_rules_v2 import NurikabeSolverV2

@dataclass
class WorkerCommand:
    kind: str
    payload: Optional[Dict[str, Any]] = None

@dataclass
class WorkerResult:
    kind: str
    payload: Dict[str, Any]

class SolverWorker:
    def __init__(self, solver_class=NurikabeSolver) -> None:
        self._cmd_q: "queue.Queue[WorkerCommand]" = queue.Queue()
        self._res_q: "queue.Queue[WorkerResult]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="SolverWorker", daemon=True)

        self._model = NurikabeModel()
        self._solver_class = solver_class
        self._solver = self._solver_class(self._model)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self._cmd_q.put_nowait(WorkerCommand(kind="stop"))
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def send(self, cmd: WorkerCommand) -> None:
        self._cmd_q.put(cmd)

    def try_recv(self) -> Optional[WorkerResult]:
        try:
            return self._res_q.get_nowait()
        except queue.Empty:
            return None

    def _emit_state(self, kind: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"state": self._model.snapshot()}
        if extra:
            payload.update(extra)
        self._res_q.put(WorkerResult(kind=kind, payload=payload))

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                cmd = self._cmd_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if cmd.kind == "stop":
                return

            if cmd.kind == "sync_state":
                state = (cmd.payload or {}).get("state")
                if isinstance(state, dict):
                    try:
                        self._model.restore(state)
                        self._solver = self._solver_class(self._model)
                        self._emit_state("synced")
                    except Exception as e:
                        self._res_q.put(WorkerResult(kind="error", payload={"message": f"Restore failed: {e}"}))
                continue

            if cmd.kind == "step":
                try:
                    step_res = self._solver.step()
                    extra = {
                        "step_result": {
                            "changed_cells": list(step_res.changed_cells),
                            "message": step_res.message,
                            "rule": step_res.rule,
                        }
                    }
                    self._emit_state("stepped", extra=extra)
                except Exception as e:
                    self._res_q.put(WorkerResult(kind="error", payload={"message": f"Step failed: {e}"}))
                continue

            if cmd.kind == "next_cell":
                try:
                    final_step_res = None
                    newly_certain_cells = []
                    
                    while True:
                        # Capture certainty before step
                        pre_certain = [[(self._model.is_land_certain(r, c) or self._model.is_sea_certain(r, c))
                                        for c in range(self._model.cols)]
                                       for r in range(self._model.rows)]
                        
                        step_res = self._solver.step()
                        
                        if step_res.rule == "None":
                            final_step_res = step_res
                            break
                        
                        # Find all cells that became certain in this step
                        for r, c in step_res.changed_cells:
                            if (self._model.is_land_certain(r, c) or self._model.is_sea_certain(r, c)) and not pre_certain[r][c]:
                                print(f"Cell ({r}, {c}) became certain due to rule: {step_res.rule}")
                                newly_certain_cells.append((r, c))
                        
                        if newly_certain_cells:
                            final_step_res = step_res
                            break
                    
                    extra = {
                        "step_result": {
                            "changed_cells": newly_certain_cells,
                            "message": final_step_res.message,
                            "rule": final_step_res.rule,
                        }
                    }
                    self._emit_state("stepped", extra=extra)
                except Exception as e:
                    self._res_q.put(WorkerResult(kind="error", payload={"message": f"Next Cell failed: {e}"}))
                continue

            if cmd.kind == "reset":
                try:
                    s = (cmd.payload or {}).get("state")
                    if isinstance(s, dict):
                        self._model.restore(s)
                        self._solver = self._solver_class(self._model)
                        self._emit_state("reset_done")
                    else:
                        self._res_q.put(WorkerResult(kind="error", payload={"message": "Reset missing state."}))
                except Exception as e:
                    self._res_q.put(WorkerResult(kind="error", payload={"message": f"Reset failed: {e}"}))
                continue

            if cmd.kind == "load_grid":
                try:
                    grid = (cmd.payload or {}).get("grid")
                    if isinstance(grid, list) and grid and isinstance(grid[0], list):
                        self._model.load_grid([[int(v) for v in row] for row in grid])
                        self._solver = self._solver_class(self._model)
                        self._emit_state("loaded")
                    else:
                        self._res_q.put(WorkerResult(kind="error", payload={"message": "Load missing/invalid grid."}))
                except Exception as e:
                    self._res_q.put(WorkerResult(kind="error", payload={"message": f"Load failed: {e}"}))
                continue
