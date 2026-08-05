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
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..errors import JobConflictError, JobNotFoundError
from ..masking import sanitize_line
from ..paths import REPOSITORY_ROOT
from ..progress import ProgressState, consume_line, snapshot
from ..train_config import config_relative_path, read_runtime_config
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


class JobManager:
    """이 서버가 실행한 학습 job들의 유일한 소유자."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, JobRecord] = {}
        self._active_job_id: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._cancel_requested = False
        self._progress: ProgressState | None = None
        self._sequence = 0
        self._log_handle: Any = None
        self._stdout_chunks: list[str] = []
        self._loaded = False

    # ------------------------------------------------------------------ 조회

    def load(self) -> None:
        """디스크에 남은 기록을 읽어 옵니다. 서버 시작 때 한 번 부릅니다."""

        with self._lock:
            if self._loaded:
                return
            for record in store.load_all_records():
                if record.status in ACTIVE_STATUSES:
                    # 서버가 죽는 동안 OS process는 사라졌습니다. 그대로 두면 유령 job이
                    # 영원히 남아 새 학습을 막습니다.
                    record.status = STATUS_INTERRUPTED
                    record.finished_at = record.finished_at or utc_now_text()
                    record.message = "서버가 다시 시작되어 실행 상태를 잃었습니다."
                    try:
                        store.save_record(record)
                    except OSError:
                        pass
                self._records[record.job_id] = record
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

    # ------------------------------------------------------------------ 실행

    def start(self, config_id: str) -> JobRecord:
        """저장된 설정으로 학습을 시작합니다. 이미 실행 중이면 거부합니다."""

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
            record = JobRecord(
                job_id=job_id,
                config_id=config_id,
                run_id=run_id,
                status=STATUS_RUNNING,
                started_at=utc_now_text(),
                settings=settings,
                data_inputs=data_inputs,
            )
            self._records[job_id] = record
            self._active_job_id = job_id
            self._cancel_requested = False
            self._progress = ProgressState()
            self._sequence = 0
            self._stdout_chunks = []
            record.progress = snapshot(self._progress)

        store.save_record(record)
        thread = threading.Thread(
            target=self._run, args=(job_id, config_id), name=f"train-job-{job_id[:8]}", daemon=True
        )
        thread.start()
        return record

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
                    if entry is not None:
                        self._emit(job_id, stream_name, entry["level"], entry["text"])
                        record = self._records.get(job_id)
                        if record is not None and state is not None:
                            record.progress = snapshot(state)
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
            with self._lock:
                self._close_log()
                self._process = None
                if self._active_job_id == job_id:
                    self._active_job_id = None

    # ------------------------------------------------------------------ 종료

    def _cancel_before_spawn(self, job_id: str) -> None:
        """Process를 띄우기 전에 취소가 도착했을 때의 마무리. lock을 쥔 채로 부릅니다."""

        record = self._records.get(job_id)
        if record is None:
            return
        record.status = STATUS_CANCELLED
        record.finished_at = utc_now_text()
        record.message = "학습을 시작하기 전에 취소했습니다."
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
