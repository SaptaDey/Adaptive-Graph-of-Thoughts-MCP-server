# MCP Integration Guide

## Overview

The Adaptive Graph of Thoughts MCP server provides advanced scientific reasoning capabilities through the Model Context Protocol (MCP). This guide covers integration with various MCP clients.

## Available Tools

### 1. scientific_reasoning_query
Perform advanced scientific reasoning using graph-based analysis.

**Parameters:**
- `query` (required): Scientific question to analyze
- `include_reasoning_trace` (optional): Include detailed reasoning steps
- `include_graph_state` (optional): Include graph visualization data
- `max_nodes_in_response_graph` (optional): Limit graph size in response
- `output_detail_level` (optional): "summary" or "detailed"
- `session_id` (optional): Session tracking identifier

**Example:**
```json
{
  "query": "What is the relationship between microbiome diversity and cancer progression?",
  "include_reasoning_trace": true,
  "include_graph_state": true
}
