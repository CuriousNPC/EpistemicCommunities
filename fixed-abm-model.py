import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
import random
from enum import Enum

class UtilityType(str, Enum):
    TRUTH_SEEKER = "TRUTH_SEEKER"
    FREE_RIDER = "FREE_RIDER"
    MIXED = "MIXED"

@dataclass
class AgentState:
    """Track the state of each agent"""
    utility_type: UtilityType
    private_belief: float  # Signal received from nature
    expressed_belief: Optional[float]  # What they express to others (if anything)
    alpha: float  # Truth-seeking weight (for mixed type)
    beta: float  # Cost sensitivity (for mixed type)
    last_update_tick: int
    neighbor_influences: Dict[int, float]
    is_participating: bool  # Track participation decision

class InformationDynamicsABM:
    def __init__(
            self,
            n_agents: int,
            true_state: float,
            p: float,  # Probability of true signal from nature
            cost: float,  # Cost of participation
            demographic_distribution: Optional[Dict[UtilityType, float]] = None,
            network: Optional[nx.Graph] = None,
            network_density: float = 0.1,
            update_sensitivity: float = 0.5,
            k_star: int = 10,
            update_lag: int = 1
    ):
        if cost < 0:
            raise ValueError("Cost must be non-negative")
        if update_sensitivity <= 0:
            raise ValueError("Update sensitivity must be positive")

        self.n_agents = n_agents
        self.true_state = true_state
        self.p = p
        self.cost = cost
        self.network_density = network_density
        self.update_sensitivity = update_sensitivity
        self.k_star = k_star
        self.update_lag = update_lag
        self.current_tick = 0

        # Set default demographic distribution if none provided
        self.demographic_distribution = demographic_distribution or {
            UtilityType.TRUTH_SEEKER: 0.3,
            UtilityType.FREE_RIDER: 0.3,
            UtilityType.MIXED: 0.4
        }

        # Initialize network
        self.network = network if network is not None else nx.erdos_renyi_graph(
            self.n_agents, self.network_density)

        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Make initial participation decisions
        self._update_all_participation_decisions()

        # Track history
        self.history = defaultdict(list)
        self.type_history = defaultdict(lambda: defaultdict(list))

    def _initialize_agents(self) -> Dict[int, AgentState]:
        """Initialize agents with utility types and initial beliefs"""
        agents = {}
        
        # Distribute utility types
        utility_types = []
        for u_type, proportion in self.demographic_distribution.items():
            count = int(proportion * self.n_agents)
            utility_types.extend([u_type] * count)

        # Adjust for rounding
        while len(utility_types) < self.n_agents:
            utility_types.append(UtilityType.MIXED)
        random.shuffle(utility_types)

        # Initialize each agent
        for i in range(self.n_agents):
            # Get signal from nature
            received_true = random.random() < self.p
            private_belief = 1.0 if received_true else 0.0

            if utility_types[i] == UtilityType.TRUTH_SEEKER:
                alpha, beta = 1.0, 0.0
            elif utility_types[i] == UtilityType.FREE_RIDER:
                alpha, beta = 0.0, 1.0
            else:  # Mixed type
                alpha = random.random()
                beta = random.random()

            agents[i] = AgentState(
                utility_type=utility_types[i],
                private_belief=private_belief,
                expressed_belief=None,  # Initially no expression
                alpha=alpha,
                beta=beta,
                last_update_tick=0,
                neighbor_influences={},
                is_participating=False  # Will be set in _update_all_participation_decisions
            )

        return agents

    def _decide_participation(self, agent_id: int) -> bool:
        """Determine if agent participates based on utility function"""
        agent = self.agents[agent_id]

        if agent.utility_type == UtilityType.TRUTH_SEEKER:
            return True
        elif agent.utility_type == UtilityType.FREE_RIDER:
            return False
        else:  # Mixed type
            # Expected utility calculation from formal model
            expected_utility = agent.alpha * self.p - agent.beta * self.cost
            return expected_utility > 0

    def _update_all_participation_decisions(self):
        """Update participation decisions for all agents based on current cost"""
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            should_participate = self._decide_participation(agent_id)
            
            # Update participation status
            agent.is_participating = should_participate
            
            # Update expressed belief based on participation decision
            agent.expressed_belief = agent.private_belief if should_participate else None

    def get_opinion_diversity(self) -> float:
        """Calculate opinion diversity metric"""
        expressed = [a.expressed_belief for a in self.agents.values() 
                    if a.expressed_belief is not None]
        if not expressed:
            return 0.0
        return np.std(expressed) if len(expressed) > 1 else 0.0

    def step(self):
        """Execute one time step of the simulation"""
        self.current_tick += 1

        # Only update neighbor influences based on network connections
        for agent_id, agent in self.agents.items():
            if (self.current_tick - agent.last_update_tick) >= self.update_lag:
                neighbors = list(self.network.neighbors(agent_id))
                if neighbors:
                    agent.neighbor_influences = {
                        n: self.update_sensitivity
                        for n in neighbors
                        if self.agents[n].expressed_belief is not None
                    }
                agent.last_update_tick = self.current_tick

        self._record_state()

    def update_cost(self, new_cost: float):
        """Update cost and recalculate participation decisions"""
        if new_cost < 0:
            raise ValueError("Cost must be non-negative")
        
        self.cost = new_cost
        self._update_all_participation_decisions()

    def _record_state(self):
        """Record state for analysis"""
        # Count active participants by type
        type_counts = defaultdict(lambda: {'active': 0, 'total': 0})
        expressed_beliefs = []

        for agent in self.agents.values():
            type_counts[agent.utility_type]['total'] += 1
            if agent.expressed_belief is not None:
                type_counts[agent.utility_type]['active'] += 1
                expressed_beliefs.append(agent.expressed_belief)

        # Record overall metrics
        self.history['tick'].append(self.current_tick)
        self.history['active_participants'].append(len(expressed_beliefs))
        self.history['consensus'] = self.get_consensus()

        # Record type-specific metrics
        for u_type in UtilityType:
            self.type_history[u_type]['active'].append(type_counts[u_type]['active'])
            self.type_history[u_type]['participation_rate'].append(
                type_counts[u_type]['active'] / type_counts[u_type]['total']
                if type_counts[u_type]['total'] > 0 else 0
            )

    def get_consensus(self) -> Optional[float]:
        """Calculate consensus from expressed beliefs only"""
        expressed = [a.expressed_belief for a in self.agents.values()
                    if a.expressed_belief is not None]

        if len(expressed) >= self.k_star:
            return float(np.mean(expressed) > 0.5)
        return None

    def run(self, n_steps: int = 1) -> Dict:
        """Run simulation and return results"""
        for _ in range(n_steps):
            self.step()

        return {
            'history': dict(self.history),
            'type_history': {k: dict(v) for k, v in self.type_history.items()},
            'final_consensus': self.get_consensus(),
            'participation_by_type': {
                u_type: len([a for a in self.agents.values()
                           if a.utility_type == u_type and
                           a.expressed_belief is not None])
                for u_type in UtilityType
            }
        }