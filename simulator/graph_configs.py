"""
Graph workload registry -- the GNN counterpart to `model_configs.py`.

A GNN layer is `H' = sigma( A_hat @ (H @ W) )`, which is two operations with
opposite characters:

  * **Combine** `H @ W` -- a dense GEMM with `W` shared across every node.
    Shape `(num_nodes, F_in) x (F_in, F_out)`.  This is structurally identical
    to a transformer FFN and needs nothing new from the simulator.
  * **Aggregate** `A_hat @ .` -- a sparse gather-accumulate over the edge list.
    Cost scales with `num_edges`, not `num_nodes^2`, and the access pattern is
    irregular.  The simulator has no sparse operand, so this is priced by a
    closed-form model in `analysis/gnn/gnn_run.py` rather than simulated.

**Combine-first ordering.**  `A_hat @ (H @ W)` and `(A_hat @ H) @ W` are
mathematically identical and cost wildly different amounts.  Combining first
costs `N*F_in*F_out + E*F_out`; aggregating first costs `E*F_in + N*F_in*F_out`.
Since `F_in > F_out` on the first layer of every configuration below (1433 -> 16
for Cora), combine-first is cheaper by roughly `F_in/F_out`, and it is what
every real implementation does.  `layer_shapes()` returns that ordering.

**Edge counts are DIRECTED** -- the number of non-zeros in `A_hat`, which is
what a gather actually pays for.  An undirected edge {u,v} contributes two.
Self-loops (GCN's `A + I`) are counted where the standard preprocessing adds
them; `num_edges` below is stated per dataset with its convention in `source`.

**These are the published dataset statistics, not values read from a loader.**
They are the standard figures for each benchmark and are right to the digit as
commonly reported, but a run that needs exactness should check them against the
actual dataset object.  Nothing in the analysis depends on the last digit --
the results are driven by average degree and feature width, which are robust to
a few thousand edges either way.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class GraphConfig:
    """One graph workload: the graph, and the GNN evaluated over it."""
    name: str
    num_nodes: int
    num_edges: int          # directed non-zeros in A_hat (see module docstring)
    feat_dim: int           # input feature width
    num_classes: int        # output width of the last layer
    hidden_dim: int         # width of every internal layer
    num_layers: int = 2     # GCN/SAGE convention
    source: str = ''

    @property
    def avg_degree(self) -> float:
        return self.num_edges / self.num_nodes

    @property
    def density(self) -> float:
        """Fraction of the dense adjacency that is non-zero."""
        return self.num_edges / (self.num_nodes ** 2)

    def layer_shapes(self) -> List[Tuple[int, int]]:
        """(F_in, F_out) per layer, combine-first ordering.

        Layer 0 takes the raw features; the last layer emits class logits;
        everything between is `hidden_dim -> hidden_dim`.
        """
        widths = ([self.feat_dim]
                  + [self.hidden_dim] * (self.num_layers - 1)
                  + [self.num_classes])
        return list(zip(widths[:-1], widths[1:]))


GRAPH_REGISTRY: Dict[str, GraphConfig] = {}


def _register(g: GraphConfig):
    GRAPH_REGISTRY[g.name] = g


# ---- Planetoid citation networks (small, sparse, wide raw features) --------
# The classic GCN benchmarks.  hidden_dim = 16 is Kipf & Welling's setting and
# is deliberately kept, because a 16-wide FP16 feature row is 32 B -- half a
# DDR5 burst.  That is the regime the gather model is most sensitive in.

_register(GraphConfig(
    'Cora', num_nodes=2708, num_edges=10556, feat_dim=1433, num_classes=7,
    hidden_dim=16, source='Planetoid; 5278 undirected edges x2'))

_register(GraphConfig(
    'CiteSeer', num_nodes=3327, num_edges=9104, feat_dim=3703, num_classes=6,
    hidden_dim=16, source='Planetoid; 4552 undirected edges x2'))

_register(GraphConfig(
    'PubMed', num_nodes=19717, num_edges=88648, feat_dim=500, num_classes=3,
    hidden_dim=16, source='Planetoid; 44324 undirected edges x2'))

# ---- Large-scale graphs (deep enough that the feature matrix does not fit) --

_register(GraphConfig(
    'ogbn-arxiv', num_nodes=169343, num_edges=2332486, feat_dim=128,
    num_classes=40, hidden_dim=256,
    source='OGB; 1166243 directed citations, symmetrized'))

_register(GraphConfig(
    'Reddit', num_nodes=232965, num_edges=114615892, feat_dim=602,
    num_classes=41, hidden_dim=256,
    source='GraphSAGE; directed non-zeros'))

_register(GraphConfig(
    'ogbn-products', num_nodes=2449029, num_edges=123718280, feat_dim=100,
    num_classes=47, hidden_dim=256,
    source='OGB; 61859140 undirected edges x2'))


def get_graph_config(name: str) -> GraphConfig:
    """Look up a graph workload, with the valid names in the error."""
    try:
        return GRAPH_REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown graph {name!r}; "
                       f"known: {', '.join(GRAPH_REGISTRY)}") from None


def list_graphs() -> List[str]:
    return list(GRAPH_REGISTRY)
