"""학습 job 하나만 실행되도록 관리하는 상태 머신.

동시 실행을 막는 확인과 표시를 **같은 lock 안에서** 하므로, 동시에 들어온 두 요청이
둘 다 통과하는 일이 없습니다.

Stream을 읽을 때 pipe마다 thread를 하나씩 씁니다. ``communicate()``는 process가 끝날
때까지 막혀서 실시간 log를 못 보고, stdout을 다 읽고 stderr를 읽는 순차 방식은 반대쪽
pipe의 OS 버퍼(보통 64KB)가 차는 순간 교착합니다. 학습은 stderr를 반드시 채웁니다.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..errors import JobConflictError, JobNotFoundError
from ..masking import sanitize_line
from ..paths import REPOSITORY_ROOT
from ..progress import (
    ProgressState,
    consume_line,
    seed_epochs,
    snapshot,
    take_quiet_change,
)
from ..train_config import config_relative_path, read_runtime_config
from .. import state_sync, team_sync
from . import runner, store
from .model import (
    ACTIVE_STATUSES,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    JobRecord,
    utc_now_text,
)


__all__ = ["JobManager", "get_manager"]


def _unwrap_stage(value: Any) -> dict[str, Any]:
    """``--only train`` 결과는 stage 이름으로 한 겹 감싸여 옵니다.

    ``main_pipeline``이 ``{"train": {...}}`` 형태로 돌려주므로 한 겹 벗겨 냅니다.
    """

    if not isinstance(value, dict):
        return {}
    stage = value.get("train")
    if isinstance(stage, dict):
        return dict(stage)
    return dict(value)


def _lost_message(record: JobRecord) -> str:
    """서버가 다시 떴을 때 그 학습에 무슨 일이 있었는지 사실대로 적습니다.

    POSIX에서 학습은 별도 session으로 뜨므로 서버가 죽어도 함께 죽지 않습니다.
    그때 "상태를 잃었습니다"만 보여 주면 학습이 끝난 줄 알고 process를 죽이거나
    같은 GPU에 새 학습을 얹습니다. checkpoint는 학습이 다 끝난 뒤 한 번에
    저장되므로, 그렇게 죽이면 그때까지 학습한 것이 전부 사라집니다.
    """

    pid = record.process_id
    if pid and runner.process_alive(pid):
        return (
            f"서버가 다시 시작되어 이 화면은 실행 상태를 잃었지만, 학습 process"
            f"(PID {pid})는 아직 돌고 있습니다. 결과는 학습이 끝나면 그대로 저장됩니다."
            " 지금 그 process를 죽이면 여기까지 학습한 것이 모두 사라집니다."
            " 로그는 여기에 다시 이어지지 않습니다."
        )
    return "서버가 다시 시작되어 실행 상태를 잃었습니다."


def _lost_runtime_message() -> str:
    """이 기계가 아니라 사라진 런타임에서 돌던 학습에 붙이는 안내입니다.

    같은 기계에서 서버만 죽은 경우와 달리, 여기서는 그 학습 process를 찾을 수
    없습니다. 없는 process를 찾아보라고 안내하면 시간만 씁니다.
    """

    return (
        "이 학습을 시작한 런타임이 사라졌습니다. 그 process는 남아 있지 않습니다."
        " epoch마다 저장한 checkpoint가 S3에 있으므로 거기서 이어서 학습할 수 있습니다."
    )


def _up_to_last_checkpoint(record: JobRecord) -> list[dict[str, Any]]:
    """이 기록의 epoch 중 checkpoint로 남아 있는 데까지만 돌려줍니다.

    train은 `epoch % checkpoint_every == 0`일 때 저장합니다(마지막 epoch는 주기와
    무관하지만, 끝까지 간 학습은 이어서 할 대상이 아닙니다). 그래서 15 epoch까지
    돌고 끊긴 주기 5짜리 학습은 10에서 다시 출발합니다.
    """

    entries = [
        entry
        for entry in (record.progress.get("epochs") or [])
        if isinstance(entry, dict) and isinstance(entry.get("epoch"), int)
    ]
    every = record.settings.get("checkpoint_every")
    if not isinstance(every, int) or isinstance(every, bool) or every < 1:
        return entries
    last = max((entry["epoch"] for entry in entries), default=0)
    cutoff = last - (last % every)
    return [entry for entry in entries if entry["epoch"] <= cutoff]


class JobManager:
    """이 서버가 실행한 학습 job들의 유일한 소유자."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 대기열에서 다음 항목을 꺼내 시작하는 구간 전체를 한 번에 하나만 지나갑니다.
        # `_lock`과 달리 `start()`가 끝날 때까지 놓지 않으므로 순서가 뒤집히지 않습니다.
        self._start_lock = threading.Lock()
        self._records: dict[str, JobRecord] = {}
        self._active_job_id: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._cancel_requested = False
        self._progress: ProgressState | None = None
        self._sequence = 0
        self._log_handle: Any = None
        self._stdout_chunks: list[str] = []
        self._loaded = False
        # 아직 시작하지 않은 대기열입니다. 앞에서부터 하나씩 꺼내 씁니다.
        self._queue: list[dict[str, Any]] = []
        # Browser login token은 queue JSON이나 API 응답에 남기지 않고 process가 살아
        # 있는 동안에만 entry_id로 찾습니다.
        self._queue_access_tokens: dict[str, str] = {}
        # 멈춘 대기열은 앞 학습이 끝나도 다음을 시작하지 않습니다. 사람이 중지를
        # 눌렀는데 곧바로 다음 학습이 뜨면 멈춘 것이 아니기 때문입니다.
        self._queue_paused = True
        # GPU 자리를 잡는 문. 학습 시작과 자동 평가 시작이 이 문을 함께 씁니다.
        # 직렬에서 둘이 서로를 못 보고 동시에 출발하면 8GB 카드에서 둘 다 out of
        # memory로 잃습니다. 잠금 순서는 언제나 gate → evaluation runner → manager.
        self._gpu_gate = threading.Lock()
        # 학습이 끝날 때마다 깨어나 평가를 이어 도는 thread. 처음 필요할 때 만듭니다.
        self._evaluation_wakeup = threading.Event()
        self._evaluation_thread: threading.Thread | None = None
        # **이 서버가 방금 끝낸** 학습만 자동 평가합니다. 디스크에 남은 옛 성공 기록을
        # 훑으면, 서버를 다시 켤 때마다 몇 십 건이 줄줄이 GPU를 잡습니다.
        self._evaluation_pending: list[str] = []

    # ------------------------------------------------------------------ 조회

    def load(self) -> None:
        """디스크에 남은 기록을 읽어 옵니다. 서버 시작 때 한 번 부릅니다."""

        with self._lock:
            if self._loaded:
                return
            # 런타임이 바뀌었으면 이 디스크에는 아무 기록도 없습니다. 자기 칸에 남은
            # 사본을 먼저 되살려야 그 학습이 아래에서 interrupted가 되고, 화면에
            # "이어서 학습"이 붙습니다.
            elsewhere = state_sync.restore()
            for record in store.load_all_records():
                if record.status in ACTIVE_STATUSES:
                    # 이 서버는 저 학습을 더 이상 관리할 수 없습니다. log pipe도 다시
                    # 이을 수 없습니다. 그대로 두면 유령 job이 영원히 남아 새 학습을
                    # 막습니다.
                    record.status = STATUS_INTERRUPTED
                    record.finished_at = record.finished_at or utc_now_text()
                    record.message = (
                        _lost_runtime_message()
                        if record.job_id in elsewhere
                        else _lost_message(record)
                    )
                    try:
                        team_sync.get_team_sync().enqueue_update(record)
                        store.save_record(record)
                    except OSError:
                        pass
                self._records[record.job_id] = record
            saved = store.load_queue()
            self._queue = saved["entries"]
            # 목록은 살리되 시작은 사람이 시킵니다. 서버가 뜨자마자 GPU 학습이
            # 저절로 시작되면 곤란합니다.
            self._queue_paused = saved["paused"]
            self._loaded = True

    def list_jobs(self) -> list[JobRecord]:
        self.load()
        with self._lock:
            records = list(self._records.values())
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records

    def get(self, job_id: str) -> JobRecord:
        self.load()
        with self._lock:
            record = self._records.get(job_id)
            if record is not None:
                return record
        return store.load_record(job_id)

    def active_job(self) -> JobRecord | None:
        with self._lock:
            if self._active_job_id is None:
                return None
            return self._records.get(self._active_job_id)

    def logs(self, job_id: str, *, after: int = 0, limit: int = 500) -> dict[str, Any]:
        self.get(job_id)  # 없는 job이면 여기서 404가 납니다.
        return store.read_logs(job_id, after=after, limit=limit)

    # ------------------------------------------------------------------ 대기열

    def queue_entries(self) -> list[dict[str, Any]]:
        """아직 시작하지 않은 항목을 기다리는 순서대로 돌려줍니다."""

        self.load()
        with self._lock:
            return [dict(item) for item in self._queue]

    def queue_paused(self) -> bool:
        self.load()
        with self._lock:
            return self._queue_paused

    def _save_queue(self) -> None:
        """lock을 쥔 채로 부릅니다."""

        store.save_queue({"entries": [dict(item) for item in self._queue]})

    def enqueue(
        self,
        config_id: str,
        *,
        access_token: str | None = None,
        resumed_from_job_id: str | None = None,
    ) -> JobRecord | None:
        """설정 하나를 대기열에 넣습니다. 비어 있으면 곧바로 시작합니다.

        시작한 경우에만 `JobRecord`를 돌려줍니다. 뒤에 줄을 선 경우는 `None`입니다.

        `resumed_from_job_id`는 이어서 학습할 때 앞선 job의 id입니다. 항목과 함께
        저장해 두므로, 다른 학습 뒤에 줄을 서 있다가 나중에 시작해도 앞선 손실
        곡선을 잃지 않습니다.
        """

        self.load()
        # 없는 설정은 여기서 걸러야, 자는 동안 조용히 건너뛰는 항목이 생기지 않습니다.
        config = read_runtime_config(config_id)
        run_id = str((config.get("train") or {}).get("run_id") or "")
        entry_id = uuid4().hex
        with self._lock:
            self._queue.append(
                {
                    "entry_id": entry_id,
                    "config_id": config_id,
                    "run_id": run_id,
                    "queued_at": utc_now_text(),
                    **(
                        {"resumed_from_job_id": resumed_from_job_id}
                        if resumed_from_job_id
                        else {}
                    ),
                }
            )
            if access_token:
                self._queue_access_tokens[entry_id] = access_token
            self._queue_paused = False
            self._save_queue()
        return self._start_next()

    def remove_from_queue(self, entry_id: str) -> None:
        """아직 시작하지 않은 항목 하나를 뺍니다. 실행 중인 학습은 대상이 아닙니다."""

        self.load()
        with self._lock:
            removed_ids = {
                item["entry_id"] for item in self._queue if item["entry_id"] == entry_id
            }
            remaining = [item for item in self._queue if item["entry_id"] != entry_id]
            if len(remaining) == len(self._queue):
                raise JobNotFoundError("대기열에서 그 항목을 찾을 수 없습니다.")
            self._queue = remaining
            for removed_id in removed_ids:
                self._queue_access_tokens.pop(removed_id, None)
            self._save_queue()

    def clear_queue(self) -> None:
        """기다리는 항목을 모두 비웁니다. 실행 중인 학습은 그대로 둡니다."""

        self.load()
        with self._lock:
            self._queue = []
            self._queue_access_tokens.clear()
            self._save_queue()

    def resume_queue(self, *, access_token: str | None = None) -> JobRecord | None:
        """멈춰 있던 대기열을 다시 돌립니다.

        받은 token을 기다리는 항목 **전체에 덮어씁니다.** 두 가지를 한꺼번에 처리해야
        하기 때문입니다.

        하나는 서버 재시작입니다. memory에만 두던 token이 사라지므로 항목에 token이
        아예 없습니다. 첫 항목에만 붙이면 그것이 끝난 뒤 다음 항목에서 또 멈춰, 밤새
        돌리라고 만든 목록을 사람이 하나씩 눌러 깨워야 합니다.

        다른 하나는 **만료**입니다. Cognito access token의 수명은 기본 한 시간이라
        앞 학습이 그보다 길면 다음 항목이 쥔 token은 이미 죽어 있습니다. AppSync가
        401로 거절하면 대기열은 멈추면서 그 만료된 token을 도로 항목에 붙여 둡니다.
        여기서 비어 있을 때만 채우면 죽은 token이 살아남아, 사람이 새로 로그인해
        다시 돌리기를 눌러도 같은 401이 반복됩니다. 서버를 다시 띄우기 전에는
        복구할 방법이 없어집니다.

        덮어써도 되는 이유는 다시 돌리기를 누른 사람이 곧 그 학습들을 시작한
        사람이기 때문입니다. 팀 기록에도 그 사람으로 남는 것이 맞습니다.
        """

        self.load()
        with self._lock:
            if access_token:
                for item in self._queue:
                    self._queue_access_tokens[item["entry_id"]] = access_token
            self._queue_paused = False
        return self._start_next()

    def _start_next(self) -> JobRecord | None:
        """비어 있고 멈추지 않았으면 다음 항목을 시작합니다.

        `start()`가 lock을 다시 잡으므로 lock을 쥐지 않은 상태에서 불러야 합니다.

        꺼내기와 시작 사이를 `_start_lock`으로 묶습니다. 항목을 꺼내는 동안만 `_lock`을
        쥐면, 앞 학습이 끝난 thread가 두 번째를 꺼내 아직 `start()`에 닿기 전에, 세 번째를
        넣는 thread가 "지금 도는 학습이 없다"를 보고 세 번째를 먼저 시작할 수 있습니다.
        진 쪽은 `JobConflictError`를 받고 자기 항목을 맨 앞으로 되돌리므로 아무것도
        잃지는 않지만, 순서가 뒤집힙니다. 밤새 돌리려고 정해 둔 차례가 그대로 지켜지지
        않으면 대기열을 쓰는 이유가 없습니다.

        `_start_lock`을 먼저 잡고 그 안에서 `_lock`을 잡습니다. `enqueue`·`resume_queue`·
        job thread 모두 `_lock`을 놓은 뒤에 이 함수를 부르므로 반대 순서는 생기지 않습니다.
        """

        with self._start_lock:
            return self._start_next_locked()

    def _start_next_locked(self) -> JobRecord | None:
        """`_start_lock`을 쥔 채로만 부릅니다."""

        while True:
            with self._lock:
                if self._active_job_id is not None or self._queue_paused or not self._queue:
                    return None
                entry = self._queue.pop(0)
                access_token = self._queue_access_tokens.pop(entry["entry_id"], None)
                self._save_queue()
            try:
                return self.start(
                    entry["config_id"],
                    access_token=access_token,
                    resumed_from_job_id=entry.get("resumed_from_job_id"),
                )
            except JobConflictError:
                # 그 사이 다른 학습이 시작됐습니다. 이 항목을 되돌리고 물러납니다.
                with self._lock:
                    self._queue.insert(0, entry)
                    if access_token:
                        self._queue_access_tokens[entry["entry_id"]] = access_token
                    self._save_queue()
                return None
            except Exception:
                # 팀 기록 인증이나 일시적인 network 오류도 여기로 옵니다. 항목과 login
                # token을 지운 채 성공처럼 돌아가면 사용자는 재개 요청을 되살릴 수 없으므로
                # 원래 자리에 복원하고 명시적으로 멈춘 뒤 API가 실제 오류를 응답하게 합니다.
                with self._lock:
                    self._queue.insert(0, entry)
                    if access_token:
                        self._queue_access_tokens[entry["entry_id"]] = access_token
                    self._queue_paused = True
                    self._save_queue()
                raise

    # -------------------------------------------------------------- 자동 평가

    def wake_evaluation(self) -> None:
        """평가할 것이 생겼다고 알립니다. thread가 없으면 여기서 띄웁니다.

        학습이 끝날 때마다, 그리고 설정이 바뀔 때마다 불립니다. 실제로 무엇을 언제
        돌릴지는 thread가 정합니다. 줄이 비어 있거나 사람이 평가 방식을 아직 고르지
        않았으면 thread 자체를 띄우지 않습니다 — 할 일이 없는데 깨어 있는 thread를
        두면, 나중에 설정을 저장할 때 이 함수가 다시 불려 그때 띄우면 됩니다.
        """

        from .. import settings as web_settings

        if web_settings.evaluation_mode() is None:
            return
        with self._lock:
            if not self._evaluation_pending:
                return
            if self._evaluation_thread is None or not self._evaluation_thread.is_alive():
                self._evaluation_thread = threading.Thread(
                    target=self._evaluation_loop, name="auto-evaluate", daemon=True
                )
                self._evaluation_thread.start()
        self._evaluation_wakeup.set()

    def _next_unevaluated(self) -> JobRecord | None:
        """줄 서 있는 것 중 아직 평가하지 않은 첫 학습입니다.

        평가를 이미 한 번 돌린 기록은 건너뜁니다. 실패한 평가도 다시 집지 않습니다 —
        같은 이유로 계속 실패할 텐데 그때마다 GPU를 잡으면 대기열이 통째로 막힙니다.
        사람이 화면에서 다시 누를 수 있습니다.
        """

        with self._lock:
            while self._evaluation_pending:
                record = self._records.get(self._evaluation_pending.pop(0))
                if (
                    record is not None
                    and record.status == STATUS_SUCCEEDED
                    and not record.evaluation
                ):
                    return record
        return None

    def _requeue_evaluation(self, job_id: str) -> None:
        """꺼냈지만 시작하지 못한 학습을 줄 맨 앞에 되돌립니다."""

        with self._lock:
            self._evaluation_pending.insert(0, job_id)

    def _evaluation_loop(self) -> None:
        """학습이 끝나면 평가를 이어 돕니다.

        ``serial``이면 도는 학습이 하나도 없을 때까지 기다립니다. 평가가 VRAM을 약
        1.8GB 더 쓰기 때문에, 8GB 카드에서 학습과 겹치면 둘 다 out of memory로
        잃습니다. ``parallel``이면 곧바로 시작합니다.

        기록을 꺼내는 것부터 시작까지 EvaluationRunner의 lock을 쥡니다. `/evaluate`
        route가 같은 이유로 그렇게 합니다 — 놓고 있으면 그 틈에 DELETE가 성공하고,
        이미 지워진 record로 시작한 평가가 끝나면서 사람이 지운 기록을 다시 저장해
        되살립니다. lock은 언제나 **runner 먼저, manager 나중** 순서로 잡습니다.
        DELETE도 그 순서라 서로 엇갈리지 않습니다.

        ponytail: 평가가 끝났는지는 5초마다 상태를 물어봅니다. 평가 하나가 몇 분
        걸리므로 이 정도 지연은 보이지 않습니다. 더 촘촘해야 하면 EvaluationRunner에
        완료 callback을 답니다.
        """

        from .. import settings as web_settings
        from ..errors import JobConflictError as _Conflict
        from ..evaluation import get_evaluation_runner

        runner_ = get_evaluation_runner()
        while True:
            self._evaluation_wakeup.wait()
            self._evaluation_wakeup.clear()
            while True:
                mode = web_settings.evaluation_mode()
                # 설정 화면에서 한 번 고르기 전까지는 자동으로 아무것도 돌리지
                # 않습니다. 서버를 올렸다는 이유만으로 GPU가 몇 시간씩 돌면 안 됩니다.
                if mode is None:
                    break
                if not self._start_one_evaluation(runner_, _Conflict, serial=mode == "serial"):
                    break
            # 직렬에서는 평가가 도는 동안 대기열이 멈춰 서 있었습니다. 다 끝났으니
            # 다시 밀어 줍니다. 여기서 깨우지 않으면 밤새 돌리려던 대기열이 첫
            # 평가에서 멈춘 채 아침을 맞습니다.
            try:
                self._start_next()
            except Exception:
                # background thread에는 오류를 응답할 caller가 없습니다.
                pass

    def _start_one_evaluation(
        self, runner_: Any, conflict: type[Exception], *, serial: bool
    ) -> bool:
        """줄에서 하나를 꺼내 평가를 시작하고 끝날 때까지 기다립니다.

        더 볼 것이 있으면 True입니다. 시작할 것이 없거나 더 봐서는 안 되는 상태면
        False를 돌려 바깥 반복을 멈춥니다.

        **도는 학습이 있는지 확인하는 것과 평가를 시작하는 것을 같은 문 안에서**
        합니다. 갈라 두면 학습 시작 쪽과 서로를 못 보고 둘 다 출발해, 직렬로 두었는데
        8GB 카드에서 둘 다 out of memory로 잃습니다. 잠금 순서는 gate → runner.
        """

        with self._gpu_gate, runner_.locked():
            if serial and self.active_job() is not None:
                return False
            record = self._next_unevaluated()
            if record is None:
                return False
            try:
                runner_.start(record, {})
            except conflict:
                # 사람이 화면에서 먼저 눌렀습니다. 꺼낸 것을 되돌리고, 그 평가가
                # 끝나기를 기다렸다 다시 집습니다. 되돌리지 않으면 이 학습은
                # 영영 자동 평가되지 않습니다.
                self._requeue_evaluation(record.job_id)
            except Exception:
                # 평가 하나가 시작조차 못 했다고 server를 죽이지 않습니다.
                # 되돌리지 않는 것은 같은 이유로 계속 실패하며 도는 것을 막기
                # 위해서입니다. 사람이 화면에서 다시 누를 수 있습니다.
                return False
        # 한 번에 하나만 돌 수 있으므로 끝날 때까지 기다렸다 다음을 집습니다.
        # lock은 여기서 이미 놓았습니다 — 쥔 채로 기다리면 사람이 화면에서 누른
        # 평가와 기록 삭제가 평가가 끝날 때까지 통째로 막힙니다.
        while runner_.status().get("status") == "running":
            time.sleep(5)
        return True

    # ------------------------------------------------------------------ 실행

    def _refuse_while_evaluating(self) -> None:
        """직렬로 두었으면 평가가 도는 동안 학습을 시작하지 않습니다.

        직렬을 고른 이유가 바로 이것입니다 — 8GB 카드에서 학습과 평가가 겹치면 둘
        다 out of memory로 잃습니다. 평가를 시작할 때만 학습을 확인하고 반대쪽을
        비워 두면, 평가가 도는 사이에 대기열의 다음 학습이나 사람이 누른 시작이
        그대로 들어와 약속이 깨집니다.

        **`_gpu_gate`를 쥔 채로 불러야 합니다.** 확인과 시작이 갈라져 있으면 두
        thread가 각자 "상대는 없다"를 보고 둘 다 진행합니다. 좁은 틈이라고 두었더니
        그 틈이 곧 둘 다 out of memory로 잃는 경로였습니다.
        """

        from .. import settings as web_settings
        from ..evaluation import get_evaluation_runner

        if web_settings.evaluation_mode() != "serial":
            return
        if get_evaluation_runner().status().get("status") != STATUS_RUNNING:
            return
        raise JobConflictError(
            "평가가 실행 중입니다. 평가 실행을 '직렬'로 두었으므로 평가가 끝난 뒤에"
            " 시작할 수 있습니다. 설정에서 '학습과 함께'로 바꾸면 겹쳐서 돌릴 수 있습니다."
        )

    def start(
        self,
        config_id: str,
        *,
        access_token: str | None = None,
        resumed_from_job_id: str | None = None,
    ) -> JobRecord:
        """저장된 설정으로 학습을 시작합니다. 이미 실행 중이면 거부합니다."""

        # GPU 자리를 잡는 구간 전체를 한 번에 하나만 지나갑니다. 자동 평가도 같은
        # 문을 지나므로, 둘이 서로를 못 보고 동시에 출발하는 일이 없습니다.
        with self._gpu_gate:
            record = self._start_locked(config_id, access_token, resumed_from_job_id)
        return self._spawn(record, config_id)

    def _start_locked(
        self,
        config_id: str,
        access_token: str | None,
        resumed_from_job_id: str | None = None,
    ) -> JobRecord:
        """`_gpu_gate`를 쥔 채로만 부릅니다. 기록을 만들고 자리를 잡습니다."""

        self._refuse_while_evaluating()
        self.load()
        config = read_runtime_config(config_id)
        settings = dict(config.get("train") or {})
        data_inputs = dict((config.get("inputs") or {}).get("data") or {})
        run_id = str(settings.get("run_id") or "")

        with self._lock:
            if self._active_job_id is not None:
                active = self._records.get(self._active_job_id)
                name = active.run_id if active else self._active_job_id
                raise JobConflictError(
                    f"이미 '{name}' 학습이 실행 중입니다. 한 번에 하나만 실행할 수 있습니다."
                )

            job_id = uuid4().hex
            cloud_run_id = team_sync.get_team_sync().create_run(
                access_token=access_token,
                local_job_id=job_id,
                run_id=run_id,
                settings=settings,
                data_inputs=data_inputs,
            )
            record = JobRecord(
                job_id=job_id,
                config_id=config_id,
                run_id=run_id,
                status=STATUS_RUNNING,
                started_at=utc_now_text(),
                settings=settings,
                data_inputs=data_inputs,
                cloud_run_id=cloud_run_id,
            )
            self._records[job_id] = record
            self._active_job_id = job_id
            self._cancel_requested = False
            self._progress = ProgressState()
            self._seed_resumed_epochs(self._progress, resumed_from_job_id)
            self._sequence = 0
            self._stdout_chunks = []
            record.progress = snapshot(self._progress)
            team_sync.get_team_sync().enqueue_update(record)
        return record

    def _seed_resumed_epochs(self, state: ProgressState, job_id: str | None) -> None:
        """이어서 학습한 실행의 손실 그래프에 앞선 실행의 epoch를 미리 채웁니다.

        `_lock`을 쥔 채로 부릅니다. 앞선 epoch는 checkpoint 안에만 있고 진행 log로는
        오지 않으므로, 채우지 않으면 11 epoch부터 이어서 한 실행의 그래프가 11에서
        시작해 그전 곡선이 사라진 것처럼 보입니다. 앞선 기록이 이 PC에 없으면
        (예: 다른 Colab runtime에서 돈 학습, 또는 사람이 그 기록을 지운 경우)
        지어내지 않고 그냥 둡니다.

        train은 epoch마다 완료를 알리지만 checkpoint는 `checkpoint_every`마다
        저장하므로, 기록의 마지막 epoch가 이어서 학습이 실제로 출발하는 자리보다
        앞설 수 있습니다. 같은 주기로 미리 잘라 냅니다. train이 첫 `epoch_started`를
        보내기 전에도 화면이 없는 epoch를 보여 주면 안 되기 때문입니다.
        """

        record = self._records.get(job_id) if job_id else None
        if record is not None:
            seed_epochs(state, _up_to_last_checkpoint(record))

    def _spawn(self, record: JobRecord, config_id: str) -> JobRecord:
        """자리를 잡은 기록으로 실제 process를 띄웁니다. gate 밖에서 합니다."""

        job_id = record.job_id
        store.save_record(record)
        thread = threading.Thread(
            target=self._run, args=(job_id, config_id), name=f"train-job-{job_id[:8]}", daemon=True
        )
        thread.start()
        if record.cloud_run_id:
            threading.Thread(
                target=self._heartbeat,
                args=(job_id,),
                name=f"team-heartbeat-{job_id[:8]}",
                daemon=True,
            ).start()
        return record

    def _heartbeat(self, job_id: str) -> None:
        """긴 epoch 중에도 다른 팀원에게 이 PC가 살아 있음을 알립니다."""

        while True:
            time.sleep(30)
            with self._lock:
                record = self._records.get(job_id)
                if record is None or record.status not in ACTIVE_STATUSES:
                    return
                team_sync.get_team_sync().enqueue_update(record)
                try:
                    store.save_record(record)
                except OSError:
                    pass

    def _open_log(self, job_id: str) -> None:
        directory = store.job_directory(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        self._log_handle = (directory / "log.jsonl").open("a", encoding="utf-8", newline="\n")

    def _close_log(self) -> None:
        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def _emit(self, job_id: str, stream_name: str, level: str, text: str) -> None:
        """Log 한 줄을 기록합니다. 반드시 lock을 쥔 채로 부릅니다."""

        self._sequence += 1
        entry = {
            "seq": self._sequence,
            "stream": stream_name,
            "level": level,
            "text": sanitize_line(text),
            "ts": utc_now_text(),
        }
        record = self._records.get(job_id)
        if record is not None:
            record.log_lines = self._sequence
            team_sync.get_team_sync().enqueue_log(record, entry)
        if self._log_handle is None:
            return
        try:
            self._log_handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            self._log_handle.flush()
        except (OSError, ValueError):
            pass  # log를 못 써도 학습은 계속됩니다.

    def _read_stream(self, job_id: str, pipe: Any, stream_name: str) -> None:
        """Pipe 하나를 끝까지 읽습니다. thread 하나가 pipe 하나만 담당합니다."""

        try:
            for line in pipe:
                with self._lock:
                    if stream_name == "stdout":
                        # stdout은 최종 결과 JSON 문서 하나입니다. 줄 단위로 log에 남기면
                        # 조각난 JSON이 흘러넘치고 '"status": "error"' 같은 줄이 오류로
                        # 잘못 분류됩니다. 모아 두었다가 파싱하고, 파싱에 실패했을 때만
                        # 원문을 log로 남깁니다.
                        self._stdout_chunks.append(line)
                        continue
                    state = self._progress
                    entry = consume_line(state, line) if state is not None else None
                    # batch 진행은 log를 만들지 않으므로, log 줄만 보고 갱신하면 긴
                    # epoch 동안 화면이 멈춰 있습니다. 표시가 남지 않도록 log 줄이
                    # 있을 때도 반드시 물어봅니다.
                    quiet = take_quiet_change(state) if state is not None else False
                    if entry is not None:
                        self._emit(job_id, stream_name, entry["level"], entry["text"])
                    if entry is not None or quiet:
                        record = self._records.get(job_id)
                        if record is not None and state is not None:
                            record.progress = snapshot(state)
                            team_sync.get_team_sync().enqueue_update(record)
        except (OSError, ValueError):
            pass  # process가 죽으면서 pipe가 닫히는 것은 정상입니다.
        finally:
            try:
                pipe.close()
            except (OSError, ValueError):
                pass

    def _run(self, job_id: str, config_id: str) -> None:
        """Job thread. 예외가 나도 반드시 최종 상태를 남깁니다."""

        process: subprocess.Popen[str] | None = None
        try:
            argv = runner.build_argv(config_relative_path(config_id))
            with self._lock:
                if self._cancel_requested:
                    # 취소가 spawn보다 먼저 도착했습니다. process를 띄우지 않습니다.
                    self._cancel_before_spawn(job_id)
                    return
                self._open_log(job_id)
                self._emit(job_id, "system", "info", "학습 process를 시작합니다.")
            process = runner.spawn(
                argv, cwd=Path(REPOSITORY_ROOT), env=runner.child_environment()
            )
            with self._lock:
                # 서버가 죽었다가 다시 떴을 때 이 학습이 아직 살아 있는지 알아보려면
                # PID가 디스크에 남아 있어야 합니다. ``_process``보다 먼저 기록해야
                # process가 떴다고 본 쪽이 PID 없는 기록을 읽는 일이 없습니다.
                record = self._records.get(job_id)
                if record is not None:
                    record.process_id = process.pid
                    try:
                        store.save_record(record)
                    except OSError:
                        pass
                self._process = process
                # spawn과 취소 사이의 아주 좁은 틈으로 취소가 들어올 수 있습니다.
                cancel_now = self._cancel_requested
            if cancel_now:
                runner.terminate_tree(process)

            readers = [
                threading.Thread(
                    target=self._read_stream, args=(job_id, process.stdout, "stdout"), daemon=True
                ),
                threading.Thread(
                    target=self._read_stream, args=(job_id, process.stderr, "stderr"), daemon=True
                ),
            ]
            for reader in readers:
                reader.start()
            exit_code = process.wait()
            for reader in readers:
                reader.join(timeout=30)
            self._finalize(job_id, exit_code)
        except Exception as error:  # spawn 실패 등
            self._finalize_error(job_id, error)
        finally:
            cancelled = False
            with self._lock:
                self._close_log()
                self._process = None
                if self._active_job_id == job_id:
                    self._active_job_id = None
                cancelled = self._cancel_requested
                if cancelled:
                    # 중지를 눌렀는데 다음 학습이 곧바로 뜨면 멈춘 것이 아닙니다.
                    self._queue_paused = True
            # 실패해도 다음으로 넘어갑니다. 자는 동안 하나가 OOM으로 죽었다고
            # 나머지가 안 돌면 밤을 통째로 버립니다.
            if not cancelled:
                try:
                    self._start_next()
                except Exception:
                    # background thread에는 오류를 응답할 caller가 없습니다. _start_next가
                    # 항목을 복원하고 queue를 멈췄으므로 사람이 상태를 보고 재개할 수 있습니다.
                    pass
            # 성공한 학습만 평가할 것이 있습니다. 취소·실패에는 쓸 checkpoint가 없습니다.
            with self._lock:
                finished = self._records.get(job_id)
                if finished is not None and finished.status == STATUS_SUCCEEDED:
                    self._evaluation_pending.append(job_id)
            self.wake_evaluation()

    # ------------------------------------------------------------------ 종료

    def _cancel_before_spawn(self, job_id: str) -> None:
        """Process를 띄우기 전에 취소가 도착했을 때의 마무리. lock을 쥔 채로 부릅니다."""

        record = self._records.get(job_id)
        if record is None:
            return
        record.status = STATUS_CANCELLED
        record.finished_at = utc_now_text()
        record.message = "학습을 시작하기 전에 취소했습니다."
        team_sync.get_team_sync().enqueue_update(record)
        try:
            store.save_record(record)
        except OSError:
            pass

    def _parse_result(self) -> dict[str, Any] | None:
        text = "".join(self._stdout_chunks).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            if start < 0:
                return None
            try:
                parsed, _ = json.JSONDecoder().raw_decode(text[start:])
            except ValueError:
                return None
        return parsed if isinstance(parsed, dict) else None

    def _finalize(self, job_id: str, exit_code: int) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            cancelled = self._cancel_requested
            result = self._parse_result()
            state = self._progress
            if state is not None:
                record.progress = snapshot(state)

            record.exit_code = exit_code
            record.finished_at = utc_now_text()

            if cancelled:
                record.status = STATUS_CANCELLED
                record.message = "사용자 요청으로 학습을 중지했습니다."
                record.orphan_note = (
                    "강제 종료라 train이 남긴 임시 directory가 정리되지 않았을 수 있습니다."
                )
            elif result is None:
                record.status = STATUS_FAILED
                record.message = "학습 결과 JSON을 해석하지 못했습니다."
                raw = "".join(self._stdout_chunks).strip()
                if raw:
                    for line in raw.splitlines()[:20]:
                        self._emit(job_id, "stdout", "error", line)
            else:
                status = result.get("status")
                message = result.get("message")
                artifacts = result.get("artifacts")
                summary = result.get("summary")
                # --only train 결과는 stage 이름으로 한 겹 감싸여 옵니다.
                record.artifacts = _unwrap_stage(artifacts)
                record.summary = _unwrap_stage(summary)
                record.message = sanitize_line(str(message)) if message else None

                if exit_code == 0 and status == "ok":
                    record.status = STATUS_SUCCEEDED
                else:
                    record.status = STATUS_FAILED
                    if not record.message:
                        record.message = f"학습이 실패했습니다(exit code {exit_code})."

            self._emit(
                job_id,
                "system",
                "info" if record.status == STATUS_SUCCEEDED else "error",
                f"학습이 {record.status} 상태로 끝났습니다(exit code {exit_code}).",
            )
            team_sync.get_team_sync().enqueue_update(record)
            # 상태가 종료로 바뀌는 순간에는 이미 디스크에 남아 있어야 합니다. 저장을
            # lock 밖으로 빼면, 종료 상태를 본 쪽이 아직 기록되지 않은 job을 읽습니다.
            try:
                store.save_record(record)
            except OSError:
                pass

    def _finalize_error(self, job_id: str, error: Exception) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            record.status = STATUS_FAILED
            record.finished_at = utc_now_text()
            record.message = sanitize_line(
                f"학습 process를 시작하지 못했습니다: {type(error).__name__}"
            )
            self._emit(job_id, "system", "error", record.message)
            team_sync.get_team_sync().enqueue_update(record)
            try:
                store.save_record(record)
            except OSError:
                pass

    def cancel(self, job_id: str) -> JobRecord:
        """실행 중인 학습을 중지합니다."""

        self.load()
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError("학습 기록을 찾을 수 없습니다.")
            if not record.is_active():
                raise JobConflictError("이미 끝난 학습은 중지할 수 없습니다.")
            # 신호를 보내기 전에 표시해야, 뒤따르는 비정상 exit code를 실패가 아니라
            # 취소로 해석할 수 있습니다.
            self._cancel_requested = True
            process = self._process

        # 최종 상태는 항상 job thread가 정합니다. 아직 process가 없으면 그 thread가
        # spawn 직전에 이 표시를 보고 아예 띄우지 않습니다.
        if process is not None:
            runner.terminate_tree(process)
            threading.Thread(
                target=self._escalate, args=(process,), name="train-cancel-escalate", daemon=True
            ).start()
        return record

    def delete(self, job_id: str) -> None:
        """이 GUI가 들고 있던 학습 기록 하나를 지웁니다.

        **학습 산출물은 지우지 않습니다.** checkpoint, 학습 결과 폴더, registry에
        등록된 실험, 팀에 공유된 기록은 모두 다른 곳이 만든 것이라 남습니다.
        지우는 것은 ``artifacts/web/jobs/<job_id>/`` 하나뿐입니다.

        아직 끝나지 않은 학습은 거절합니다. 기록을 없애면 그 process를 중지할
        방법도, 끝났을 때 결과를 적을 곳도 사라지기 때문입니다.
        """

        self.load()
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                # 형식이 틀린 id는 여기서 걸리고, 맞더라도 없으면 store가 알려 줍니다.
                store.delete_record(job_id)
                return
            if record.is_active() or job_id == self._active_job_id:
                raise JobConflictError("실행 중이거나 대기 중인 학습의 기록은 지울 수 없습니다.")
            store.delete_record(job_id)
            self._records.pop(job_id, None)

    def _escalate(self, process: subprocess.Popen[str]) -> None:
        """정중한 종료가 통하지 않으면 강제로 끝냅니다."""

        try:
            process.wait(timeout=runner.TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            runner.kill_tree(process)
        except Exception:
            pass


_MANAGER: JobManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> JobManager:
    """서버 전체가 공유하는 manager 하나를 돌려줍니다."""

    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER
