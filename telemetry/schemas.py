"""
Pydantic Schemas for Telemetry & Execution Performance Tracking.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class AgentExecutionLog(BaseModel):
    """Execution telemetry record for an individual pipeline agent or sub-engine."""
    agent_name: str = Field(..., description="Agent or engine name e.g. agent_1, eligibility_engine")
    started_at: str = Field(..., description="ISO start timestamp")
    completed_at: Optional[str] = Field(None, description="ISO completion timestamp")
    duration_ms: float = Field(default=0.0, description="Measured execution duration in milliseconds")
    status: str = Field(default="SUCCESS", description="SUCCESS, WARNING, or FAILED")
    input_summary: Optional[Dict[str, Any]] = Field(None, description="High-level masked summary of inputs")
    output_summary: Optional[Dict[str, Any]] = Field(None, description="High-level masked summary of outputs")
    error_message: Optional[str] = Field(None, description="Exception details if failed")


class PipelineTelemetry(BaseModel):
    """End-to-end execution telemetry and resource utilization for an application run."""
    application_id: str = Field(..., description="Loan application ID")
    total_pipeline_duration_ms: float = Field(default=0.0, description="Total wall-clock pipeline duration in ms")
    agent_durations: Dict[str, float] = Field(default_factory=dict, description="Per-agent duration breakdown in ms")
    ocr_duration_ms: float = Field(default=0.0)
    validation_duration_ms: float = Field(default=0.0)
    cross_doc_duration_ms: float = Field(default=0.0)
    risk_duration_ms: float = Field(default=0.0)
    eligibility_duration_ms: float = Field(default=0.0)
    decision_graph_duration_ms: float = Field(default=0.0)
    report_duration_ms: float = Field(default=0.0)
    documents_count: int = Field(default=0)
    fields_count: int = Field(default=0)
    citations_count: int = Field(default=0)
    memory_mb: float = Field(default=0.0, description="Process RSS memory footprint in Megabytes")
    created_at: Optional[str] = Field(None)
    agent_logs: List[AgentExecutionLog] = Field(default_factory=list)
