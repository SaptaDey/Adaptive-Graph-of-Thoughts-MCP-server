# Neo4j Indexing and Constraints Strategy for NexusMind

This document outlines the recommended Neo4j indexing and constraint strategy to ensure optimal performance for the NexusMind application. These commands are typically run manually by a database administrator or as part of an initial database setup script.

## Rationale

Indexes are crucial for speeding up query performance, especially for `MATCH` operations with `WHERE` clauses on specific properties. Constraints (like uniqueness) also implicitly create indexes and ensure data integrity.

The following recommendations are based on common query patterns observed in the NexusMind stages.

## Uniqueness Constraints

It's vital that node IDs are unique. The `id` property is used universally for node lookups.

```cypher
// Ensures every node with the :Node label has a unique 'id' property.
// This also creates an index on :Node(id).
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;
```

*Note: All nodes created by the application currently receive the `:Node` label in addition to a more specific type label (e.g., `:HYPOTHESIS`, `:EVIDENCE`). This constraint effectively covers all application-managed nodes.*

## Node Property Indexes

These indexes will speed up filtering and lookups based on common properties.

```cypher
// Index on the 'type' property for nodes.
// Useful for queries that filter by n.type (e.g., pruning, some audit checks).
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.type);

// Index on 'metadata_impact_score' for nodes.
// Used in pruning and potentially subgraph extraction.
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.metadata_impact_score);

// Index for overall confidence. Choose one of the following based on availability/usage:
// If 'confidence_overall_avg' is reliably populated:
// CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.confidence_overall_avg);
// As a common proxy if the above is not set:
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.confidence_empirical_support);

// Index on 'metadata_layer_id' for nodes.
// Can be used in subgraph extraction or other layer-specific queries.
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.metadata_layer_id);

// Index on 'metadata_is_knowledge_gap' for nodes.
// Used in subgraph extraction and reflection audits.
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.metadata_is_knowledge_gap);
```

## Label-Specific Property Indexes

While the `:Node` indexes cover many cases, indexes on properties within specific types can be beneficial if those types are frequently queried with those property filters.

```cypher
// Index on 'metadata_query_context' for :ROOT nodes.
// Speeds up the initial root node lookup in Stage 1.
CREATE INDEX IF NOT EXISTS FOR (r:ROOT) ON (r.metadata_query_context);
```

## Relationship Property Indexes (Optional)

Indexing relationship properties is less common than node properties but can be beneficial for queries that scan many relationships based on a property.

```cypher
// Optional: Index on the 'confidence' property of relationships.
// May improve performance of edge pruning in Stage 5 if the graph has a vast number of relationships.
// CREATE INDEX IF NOT EXISTS FOR ()-[r]-() ON (r.confidence);
```

## Applying Indexes

These Cypher commands can be executed directly in the Neo4j Browser or via a Cypher shell. It's generally safe to run `CREATE INDEX IF NOT EXISTS` or `CREATE CONSTRAINT IF NOT EXISTS` multiple times; they will only create the index/constraint if it doesn't already exist.

Building indexes can take time on large databases. It's recommended to apply them during a maintenance window if the database is already heavily populated. For new databases, apply them after the initial schema setup.
