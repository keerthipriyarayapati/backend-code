"""
Pydantic Schemas for Evidence-Based Decision Graph.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Single stage node in the loan processing decision graph."""
    node_id: str = Field(..., description="Unique node ID e.g. 'agent_1_classify'")
    stage_name: str = Field(..., description="Stage identifier e.g. 'STAGE_4_CLASSIFY'")
    label: str = Field(..., description="Human-readable display title")
    status: str = Field(..., description="PASS, FAIL, WARNING, or INFO")
    agent_name: Optional[str] = Field(None, description="Responsible agent or engine")
    description: Optional[str] = Field(None, description="Summary of evaluation at this node")
    evidence_count: int = Field(default=0, description="Number of supporting evidence items")
    citation_count: int = Field(default=0, description="Number of policy/field citations")
    node_metadata: Dict[str, Any] = Field(default_factory=dict, description="Detailed stage metrics and findings")


class GraphEdge(BaseModel):
    """Directed connection between decision graph stages."""
    source: str = Field(..., description="Origin node ID")
    target: str = Field(..., description="Destination node ID")
    label: Optional[str] = Field(None, description="Relationship or condition label")
    type: str = Field(default="DIRECTED", description="Edge type")


class DecisionGraphData(BaseModel):
    """Complete 11-stage decision graph topology."""
    graph_id: str = Field(..., description="Unique graph ID")
    application_id: str = Field(..., description="Loan application ID")
    total_nodes: int = Field(default=11)
    total_edges: int = Field(default=10)
    passed_nodes: int = Field(default=0)
    failed_nodes: int = Field(default=0)
    warning_nodes: int = Field(default=0)
    info_nodes: int = Field(default=0)
    created_at: Optional[str] = Field(None)
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
