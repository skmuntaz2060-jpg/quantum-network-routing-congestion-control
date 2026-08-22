import pytest
import networkx as nx
from quantum_routing.benchmarking.metrics import Benchmark

def test_benchmark_init():
    graph = nx.Graph()
    bench = Benchmark(graph)
    assert bench.graph == graph
