from typing import Any, Generic, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field, validator

# --- Generic JSON-RPC Models ---
T = TypeVar("T")
E = TypeVar("E")  # Error data type


class JSONRPCRequest(BaseModel, Generic[T]):
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version, must be '2.0'")
    method: str = Field(..., description="Method name to be invoked")
    params: Optional[T] = Field(default=None, description="Parameters for the method")
    id: Optional[Union[str, int, None]] = Field(
        default=None, description="Request identifier (string, number, or null)"
    )

    @validator("jsonrpc")
    def check_jsonrpc_version(cls, v: str) -> str:
        if v != "2.0":
            raise ValueError('jsonrpc version must be "2.0"')
        return v


class JSONRPCErrorObject(BaseModel, Generic[E]):
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[E] = Field(default=None, description="Additional error data")


class JSONRPCResponse(BaseModel, Generic[T, E]):
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version, must be '2.0'")
    result: Optional[T] = Field(
        default=None, description="Result of the method invocation (present on success)"
    )
    error: Optional[JSONRPCErrorObject[E]] = Field(
        default=None, description="Error object (present on failure)"
    )
    id: Union[str, int, None] = Field(
        ..., description="Response identifier, mirrors request id"
    )

    @validator("jsonrpc")
    def check_jsonrpc_version(cls, v: str) -> str:
        if v != "2.0":
            raise ValueError('jsonrpc version must be "2.0"')
        return v

    @validator("error", always=True)
    def check_result_error_conditions(
        cls, error_value: Optional[JSONRPCErrorObject[E]], values: dict[str, Any]
    ) -> Optional[JSONRPCErrorObject[E]]:
        """
        Validates that a JSON-RPC response contains either a result or an error, but not both.
        
        Raises:
            ValueError: If both "result" and "error" are present, or if neither is present.
        
        Returns:
            The error object if present; otherwise, None.
        """
        result_value = values.get("result")

        if result_value is not None and error_value is not None:
            raise ValueError(
                'Both "result" and "error" cannot be present in a JSONRPCResponse'
            )

        if result_value is None and error_value is None:
            raise ValueError(
                'Either "result" or "error" must be present in a JSONRPCResponse'
            )

        return error_value


# --- MCP Specific Schemas based on claude_mcp_config.json and typical MCP interactions ---


# Params for "initialize" method
class MCPInitializeClientInfo(BaseModel):
    client_name: Optional[str] = Field(
        default=None, description="Name of the client application"
    )
    client_version: Optional[str] = Field(
        default=None, description="Version of the client application"
    )
    supported_mcp_versions: Optional[list[str]] = Field(
        default_factory=list,
        description="MCP versions supported by the client, defaults to empty list",
    )


class MCPInitializeParams(BaseModel):
    process_id: Optional[int] = Field(default=None, description="Client's process ID")
    client_info: MCPInitializeClientInfo = Field(
        default_factory=MCPInitializeClientInfo,
        description="Information about the client",
    )


class MCPInitializeResult(BaseModel):
    server_name: str = Field(
        default="NexusMind-MCP", description="Name of the MCP server"
    )
    server_version: str = Field(
        default="0.1.0", description="Version of the MCP server"
    )
    mcp_version: str = Field(
        default="0.1.0", description="MCP version implemented by the server"
    )


# Params for "asr_got.query" method
class MCPQueryContext(BaseModel):
    conversation_id: Optional[str] = Field(default=None)
    history: Optional[list[dict[str, Any]]] = Field(default=None)
    user_preferences: Optional[dict[str, Any]] = Field(default=None)


class MCPQueryOperationalParams(BaseModel):
    include_reasoning_trace: bool = Field(default=True)
    include_graph_state: bool = Field(default=True)
    max_nodes_in_response_graph: Optional[int] = Field(default=50, ge=0)
    output_detail_level: Optional[str] = Field(
        default="summary", examples=["summary", "detailed"]
    )


class MCPASRGoTQueryParams(BaseModel):
    query: str
    context: Optional[MCPQueryContext] = Field(default_factory=MCPQueryContext)
    parameters: Optional[MCPQueryOperationalParams] = Field(
        default_factory=MCPQueryOperationalParams
    )
    session_id: Optional[str] = Field(default=None)


# Result for "asr_got.query" method
class GraphNodeSchema(BaseModel):
    node_id: str = Field(..., examples=["n0"])
    label: str = Field(..., examples=["Task Understanding"])
    type: str = Field(..., examples=["root"])
    confidence: Optional[list[float]] = Field(
        default=None, examples=[0.9, 0.85, 0.92, 0.88]
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeSchema(BaseModel):
    edge_id: str = Field(..., examples=["e_n0_dim1"])
    source: str = Field(..., examples=["n0"])
    target: str = Field(..., examples=["dim1"])
    edge_type: str = Field(..., examples=["decomposition"])
    confidence: Optional[float] = Field(default=None, examples=[0.9])
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphHyperedgeSchema(BaseModel):
    edge_id: str
    nodes: list[str]
    confidence: Optional[float] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphStateSchema(BaseModel):
    nodes: list[GraphNodeSchema] = Field(default_factory=list)
    edges: list[GraphEdgeSchema] = Field(default_factory=list)
    hyperedges: list[GraphHyperedgeSchema] = Field(default_factory=list)
    layers: Optional[dict[str, list[str]]] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPASRGoTQueryResult(BaseModel):
    answer: str
    reasoning_trace_summary: Optional[str] = Field(default=None)
    graph_state_full: Optional[GraphStateSchema] = Field(default=None)
    confidence_vector: Optional[list[float]] = Field(
        default=None, examples=[0.7, 0.6, 0.8, 0.75]
    )
    execution_time_ms: Optional[int] = Field(default=None)
    session_id: Optional[str] = Field(default=None)


# Example for a hypothetical "got/processQuery" method
class GoTQueryInput(BaseModel):
    query: str = Field(
        ..., description="The natural language query or problem statement"
    )
    config_override: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for NexusMind parameters for this query",
    )
    session_id: Optional[str] = Field(
        default=None, description="Optional session ID to continue or manage a session"
    )


class GoTQueryThoughtStep(BaseModel):
    stage_name: str
    summary: str


class GoTQueryProgressParams(BaseModel):
    session_id: str
    stage: str
    status: str
    message: Optional[str] = Field(default=None)
    progress_percentage: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    intermediate_results: Optional[list[GoTQueryThoughtStep]] = Field(default=None)


class GoTQueryProgressNotification(JSONRPCRequest[GoTQueryProgressParams]):
    method: str = "got/queryProgress"


class GoTQueryFinalResult(BaseModel):
    session_id: str
    final_answer: str
    confidence_vector: Optional[list[float]] = Field(default=None)
    supporting_evidence_ids: Optional[list[str]] = Field(default=None)
    full_graph_summary: Optional[dict[str, Any]] = Field(default=None)


# --- Standard MCP Notification/Request structures ---
class SetTraceParams(BaseModel):
    value: Literal["off", "messages", "verbose"]


class SetTraceNotification(JSONRPCRequest[SetTraceParams]):
    method: str = "$/setTrace"


class LogTraceParams(BaseModel):
    message: str
    verbose: Optional[str] = Field(default=None)


class LogTraceNotification(JSONRPCRequest[LogTraceParams]):
    method: str = "$/logTrace"


# --- Shutdown and Exit ---
class ShutdownParams(BaseModel):
    pass


class ShutdownResult(BaseModel):
    pass


class ExitParams(BaseModel):
    pass


def create_jsonrpc_error(
    request_id: Optional[Union[str, int, None]], 
    code: int, 
    message: str, 
    data: Any = None
) -> JSONRPCResponse:
    """
    Create a JSON-RPC error response.
    
    Args:
        request_id: The ID of the request that generated this error
        code: Error code
        message: Error message
        data: Optional additional error data
        
    Returns:
        A JSON-RPC response with the error object
    """
    error_obj = JSONRPCErrorObject(code=code, message=message, data=data)
    return JSONRPCResponse(id=request_id, error=error_obj, result=None)
