import pandas as pd
import networkx as nx

class Benchmark:
    """
    Calculates performance metrics from simulation results.
    """

    def __init__(self, network_graph: nx.Graph):
        """
        Initialize the benchmark engine.
        
        Args:
            network_graph (nx.Graph): The network topology graph.
        """
        self.graph = network_graph

    def calculate_throughput(self, routes: dict, demands: list) -> float:
        """
        Calculates total network throughput based on assigned routes and capacities.
        
        Args:
            routes (dict): The routing assignments.
            demands (list): The original traffic demands.
            
        Returns:
            float: Total throughput achieved.
        """
        pass

    def calculate_delay(self, routes: dict) -> pd.DataFrame:
        """
        Calculates end-to-end delay for the assigned routes.
        
        Args:
            routes (dict): The routing assignments.
            
        Returns:
            pd.DataFrame: Delay metrics for each demand.
        """
        pass

    def detect_congestion(self, routes: dict, demands: list) -> int:
        """
        Counts the number of congested links (where traffic > capacity).
        
        Args:
            routes (dict): The routing assignments.
            demands (list): The original traffic demands.
            
        Returns:
            int: Number of congested links.
        """
        pass
