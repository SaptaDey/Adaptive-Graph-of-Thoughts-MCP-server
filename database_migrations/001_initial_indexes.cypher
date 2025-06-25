CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.type);
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.metadata_impact_score);
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.confidence_empirical_support);
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.metadata_layer_id);
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.metadata_is_knowledge_gap);
CREATE INDEX IF NOT EXISTS FOR (r:ROOT) ON (r.metadata_query_context);
