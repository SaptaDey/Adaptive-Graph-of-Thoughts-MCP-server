# src/adaptive_graph_of_thoughts/validation.py
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import re

class QueryValidation(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, regex=r'^[a-zA-Z0-9_-]+$')
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('query')
    def validate_query_content(cls, v):
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE | re.DOTALL):
                raise ValueError(f"Query contains potentially dangerous content")
        
        return v.strip()
    
    @validator('parameters')
    def validate_parameters(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Parameters must be a dictionary")
        
        # Limit parameter size
        if len(str(v)) > 50000:
            raise ValueError("Parameters too large")
        
        return v

class Neo4jQueryValidation(BaseModel):
    label: str = Field(..., regex=r'^[a-zA-Z][a-zA-Z0-9_]*$')
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('label')
    def validate_label(cls, v):
        allowed_labels = {'User', 'Document', 'Hypothesis', 'Evidence', 'Session'}
        if v not in allowed_labels:
            raise ValueError(f"Label '{v}' not in allowed list: {allowed_labels}")
        return v
    
    @validator('properties')
    def validate_properties(cls, v):
        # Validate property names
        for key in v.keys():
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid property name: {key}")
        
        # Limit property values
        for key, value in v.items():
            if isinstance(value, str) and len(value) > 10000:
                raise ValueError(f"Property '{key}' value too long")
        
        return v
