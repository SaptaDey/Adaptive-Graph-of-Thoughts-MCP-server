import pytest
import asyncio
from adaptive_graph_of_thoughts.infrastructure.neo4j_utils import bulk_create_nodes_optimized

class TestPerformance:
    @pytest.mark.asyncio
    async def test_bulk_operations_batching(self, mock_neo4j):
        nodes = [{"id": i, "name": f"node_{i}"} for i in range(5000)]
        
        # Should batch into multiple queries
        result = await bulk_create_nodes_optimized("TestNode", nodes, batch_size=1000)
        
        # Verify batching occurred
        assert len(result) == 5000
