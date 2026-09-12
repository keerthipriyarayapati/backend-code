"""
Pipeline Telemetry Tracker & Execution Performance Monitor.
Measures true elapsed timings, agent lifecycle events, and memory footprint.
"""

import os
import time
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from telemetry.schemas import AgentExecutionLog, PipelineTelemetry
from database.repositories import (
    log_agent_execution,
    save_telemetry_metrics,
    get_telemetry_by_application
)

logger = logging.getLogger("PipelineTelemetryTracker")


def get_current_memory_mb() -> float:
    """Returns the current process resident set size (RSS) memory in megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


class PipelineTelemetryTracker:
    """Tracks end-to-end and granular agent execution metrics for a loan application."""

    def __init__(self, application_id: str, db: Optional[Session] = None):
        self.application_id = application_id
        self.db = db
        self.pipeline_start_time = time.perf_counter()
        self.agent_logs: List[AgentExecutionLog] = []
        self.agent_durations: Dict[str, float] = {}
        self.specialized_durations: Dict[str, float] = {
            "ocr_duration_ms": 0.0,
            "validation_duration_ms": 0.0,
            "cross_doc_duration_ms": 0.0,
            "risk_duration_ms": 0.0,
            "eligibility_duration_ms": 0.0,
            "decision_graph_duration_ms": 0.0,
            "report_duration_ms": 0.0
        }

    @contextmanager
    def track_agent(
        self,
        agent_name: str,
        input_summary: Optional[Dict[str, Any]] = None
    ):
        """Context manager measuring execution time and status of an individual agent."""
        t0 = time.perf_counter()
        dt_start = datetime.utcnow()
        status = "SUCCESS"
        error_msg = None
        output_holder: Dict[str, Any] = {}

        try:
            yield output_holder
        except Exception as ex:
            status = "FAILED"
            error_msg = str(ex)
            raise ex
        finally:
            t1 = time.perf_counter()
            dt_end = datetime.utcnow()
            duration_ms = round((t1 - t0) * 1000, 2)
            self.agent_durations[agent_name] = duration_ms

            # Map to specialized durations if applicable
            if agent_name in ["agent_1", "ocr"]:
                self.specialized_durations["ocr_duration_ms"] = duration_ms
            elif agent_name == "agent_3":
                self.specialized_durations["validation_duration_ms"] = duration_ms
            elif agent_name == "agent_4":
                self.specialized_durations["cross_doc_duration_ms"] = duration_ms
            elif agent_name == "agent_5":
                self.specialized_durations["risk_duration_ms"] = duration_ms
            elif agent_name in ["eligibility_engine", "eligibility"]:
                self.specialized_durations["eligibility_duration_ms"] = duration_ms
            elif agent_name in ["decision_graph", "graph"]:
                self.specialized_durations["decision_graph_duration_ms"] = duration_ms
            elif agent_name == "agent_6":
                self.specialized_durations["report_duration_ms"] = duration_ms

            log_entry = AgentExecutionLog(
                agent_name=agent_name,
                started_at=dt_start.isoformat(),
                completed_at=dt_end.isoformat(),
                duration_ms=duration_ms,
                status=status,
                input_summary=input_summary,
                output_summary=output_holder.get("summary"),
                error_message=error_msg
            )
            self.agent_logs.append(log_entry)

            # Persist individual agent log
            if self.db:
                try:
                    log_agent_execution(
                        db=self.db,
                        application_id=self.application_id,
                        agent_name=agent_name,
                        started_at=dt_start,
                        completed_at=dt_end,
                        duration_ms=duration_ms,
                        status=status,
                        input_summary=input_summary,
                        output_summary=output_holder.get("summary"),
                        error_message=error_msg
                    )
                except Exception as e:
                    logger.warning(f"Could not persist log for {agent_name}: {e}")

    def record_stage_duration(self, stage_name: str, duration_ms: float):
        """Records granular sub-stage duration directly."""
        key = f"{stage_name}_duration_ms"
        if key in self.specialized_durations:
            self.specialized_durations[key] = duration_ms
        self.agent_durations[stage_name] = duration_ms

    def finalize(
        self,
        documents_count: int = 0,
        fields_count: int = 0,
        citations_count: int = 0
    ) -> PipelineTelemetry:
        """Finalizes total execution metrics and persists summary record."""
        total_duration = round((time.perf_counter() - self.pipeline_start_time) * 1000, 2)
        mem_mb = get_current_memory_mb()

        telemetry = PipelineTelemetry(
            application_id=self.application_id,
            total_pipeline_duration_ms=total_duration,
            agent_durations=self.agent_durations,
            ocr_duration_ms=self.specialized_durations.get("ocr_duration_ms", 0.0),
            validation_duration_ms=self.specialized_durations.get("validation_duration_ms", 0.0),
            cross_doc_duration_ms=self.specialized_durations.get("cross_doc_duration_ms", 0.0),
            risk_duration_ms=self.specialized_durations.get("risk_duration_ms", 0.0),
            eligibility_duration_ms=self.specialized_durations.get("eligibility_duration_ms", 0.0),
            decision_graph_duration_ms=self.specialized_durations.get("decision_graph_duration_ms", 0.0),
            report_duration_ms=self.specialized_durations.get("report_duration_ms", 0.0),
            documents_count=documents_count,
            fields_count=fields_count,
            citations_count=citations_count,
            memory_mb=mem_mb,
            created_at=datetime.utcnow().isoformat(),
            agent_logs=self.agent_logs
        )

        if self.db:
            try:
                save_telemetry_metrics(
                    db=self.db,
                    application_id=self.application_id,
                    metrics=telemetry.dict()
                )
                logger.info(f"Persisted telemetry for {self.application_id}: total={total_duration}ms, memory={mem_mb}MB")
            except Exception as e:
                logger.error(f"Failed to persist telemetry metrics: {e}", exc_info=True)

        return telemetry
