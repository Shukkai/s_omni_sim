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

**Degree distributions are SYNTHESISED, not measured.**  `avg_degree` alone is
not enough to price aggregation, because a per-node cost that is affine in
degree with a large constant term behaves very differently on a node of degree
1 than on the mean.  Real graphs here are heavy-tailed, so this module fits a
**discrete power law** `p(d) ~ d^-gamma` on `[min_degree, max_degree]` and
solves for the one free parameter `gamma` so that the distribution's mean
reproduces the published `avg_degree` exactly.  `max_degree` is the commonly
reported maximum for each dataset; `gamma` is *derived here*, not taken from a
paper.  So a bucket count is a model output, never a dataset fact, and
`degree_distribution()` says so in its docstring.  Two graphs with the same
mean and different maxima get different distributions, which is the point.

**These are the published dataset statistics, not values read from a loader.**
They are the standard figures for each benchmark and are right to the digit as
commonly reported, but a run that needs exactness should check them against the
actual dataset object.  Nothing in the analysis depends on the last digit --
the results are driven by average degree and feature width, which are robust to
a few thousand edges either way.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _powerlaw_mean(gamma: float, d_min: int, d_max: int) -> float:
    """Mean of the discrete power law `p(d) ~ d^-gamma` on `[d_min, d_max]`."""
    num = den = 0.0
    for d in range(d_min, d_max + 1):
        w = d ** -gamma
        num += d * w
        den += w
    return num / den


def fit_powerlaw_exponent(mean: float, d_min: int, d_max: int,
                          tol: float = 1e-9, iters: int = 200) -> float:
    """Solve `_powerlaw_mean(gamma, d_min, d_max) == mean` for gamma.

    The mean is strictly decreasing in gamma (more weight on small degrees), so
    a bisection on `[-4, 24]` is exact to floating point and needs no library.
    gamma < 0 is allowed and simply means a distribution skewed toward the
    *high* degrees; it never occurs for the registered graphs, but refusing it
    would silently clamp instead of failing loudly.

    Raises ValueError if `mean` is outside `(d_min, d_max)`, which is the only
    range any distribution on that support can produce.
    """
    if not d_min <= mean <= d_max:
        raise ValueError(f"mean {mean} outside [{d_min}, {d_max}]")
    lo, hi = -4.0, 24.0
    if _powerlaw_mean(lo, d_min, d_max) < mean:
        raise ValueError(f"mean {mean} too high for d_max={d_max}")
    if _powerlaw_mean(hi, d_min, d_max) > mean:
        raise ValueError(f"mean {mean} too low for d_min={d_min}")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _powerlaw_mean(mid, d_min, d_max) > mean:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


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

    # --- Degree distribution (a MODEL -- see the module docstring) ----------
    # `max_degree` is the commonly reported maximum in-degree for the dataset.
    # `min_degree` is 1 for every registered graph (isolated nodes are dropped
    # by the standard loaders, and GCN's self-loop would give even an isolated
    # node degree 1).  Setting `min_degree == max_degree` makes the graph
    # regular, which is what the reduction check in `gnn_run.py` uses.
    min_degree: int = 1
    max_degree: int = 0     # 0 = no distribution registered
    degree_source: str = ''

    @property
    def avg_degree(self) -> float:
        return self.num_edges / self.num_nodes

    # ---- Degree distribution ----------------------------------------------

    @property
    def degree_exponent(self) -> Optional[float]:
        """Fitted power-law exponent, or None for a regular graph.

        **Derived, not published.**  It is whatever value makes the discrete
        power law on `[min_degree, max_degree]` have mean `avg_degree`.  It is
        reported so the model can be sanity-checked against the literature's
        2.0-3.0 range for citation and social graphs, not because any paper
        states it for these datasets.
        """
        if self.max_degree <= 0:
            raise ValueError(f"{self.name}: no max_degree registered")
        if self.min_degree == self.max_degree:
            return None
        key = (self.min_degree, self.max_degree, self.avg_degree)
        if key not in _EXPONENT_CACHE:
            _EXPONENT_CACHE[key] = fit_powerlaw_exponent(
                self.avg_degree, self.min_degree, self.max_degree)
        return _EXPONENT_CACHE[key]

    def degree_distribution(self) -> List[Tuple[int, float]]:
        """`[(degree, expected_node_count), ...]`, a SYNTHESISED heavy tail.

        Not read from any dataset.  A discrete power law `p(d) ~ d^-gamma` on
        `[min_degree, max_degree]`, with `gamma` fitted so the mean reproduces
        the published `avg_degree`, multiplied by `num_nodes`.

        **The counts are deliberately fractional.**  They are expected node
        counts, not a realisation.  Rounding them to integers is the obvious
        thing to do and it is wrong here: the tail degrees each have an
        expected count well below 1 while carrying a large share of the edges,
        so integer rounding drops them and loses up to **34%** of `num_edges`
        on ogbn-arxiv (measured while building this; `degree_edge_error()` is
        what watches for it).  Keeping them fractional makes both
        `sum(count) == num_nodes` and `sum(degree * count) == num_edges` hold
        to floating point, which is the only way a distribution-aware total is
        comparable to the mean-degree one at all.

        A regular graph (`min_degree == max_degree`) collapses to a single
        bucket `[(d, num_nodes)]`, so every sum over the distribution reduces
        exactly to the mean-degree calculation.  That is the reduction the
        pre-flight checks.

        **What it is not.**  No community structure, no degree correlation
        between the endpoints of an edge, no in/out-degree split.  Anything
        depending on *which* neighbour a node has -- locality, partition
        quality, cache reuse -- cannot be asked of this model, only things
        depending on how many.
        """
        if self.max_degree <= 0:
            raise ValueError(f"{self.name}: no max_degree registered")
        key = (self.min_degree, self.max_degree, self.avg_degree,
               self.num_nodes)
        if key in _DISTRIBUTION_CACHE:
            return _DISTRIBUTION_CACHE[key]

        if self.min_degree == self.max_degree:
            dist = [(self.min_degree, float(self.num_nodes))]
            _DISTRIBUTION_CACHE[key] = dist
            return dist

        gamma = self.degree_exponent
        degrees = range(self.min_degree, self.max_degree + 1)
        weights = [d ** -gamma for d in degrees]
        total_w = sum(weights)
        dist = [(d, self.num_nodes * w / total_w)
                for d, w in zip(degrees, weights)]
        _DISTRIBUTION_CACHE[key] = dist
        return dist

    def degree_edge_error(self) -> float:
        """Relative error in `num_edges` implied by the distribution.

        Zero to floating point by construction -- `gamma` is fitted to the mean
        and the counts are not rounded.  It is computed rather than asserted
        because it is the one number that says the fit converged.
        """
        edges = sum(d * c for d, c in self.degree_distribution())
        return (edges - self.num_edges) / self.num_edges

    def degree_head_share(self, frac: float = 0.1) -> float:
        """Share of edges owned by the top `frac` of nodes by degree.

        The one-number summary of how far the distribution is from its mean,
        and the reason a mean-degree cycle count can be right in total while
        being wrong about every individual node.
        """
        dist = sorted(self.degree_distribution(), reverse=True)
        budget = self.num_nodes * frac
        edges = 0.0
        for d, c in dist:
            take = min(c, budget)
            edges += d * take
            budget -= take
            if budget <= 0:
                break
        return edges / self.num_edges

    # ---- Shapes ------------------------------------------------------------

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


_EXPONENT_CACHE: Dict[tuple, float] = {}
_DISTRIBUTION_CACHE: Dict[tuple, List[Tuple[int, float]]] = {}

GRAPH_REGISTRY: Dict[str, GraphConfig] = {}


def _register(g: GraphConfig):
    GRAPH_REGISTRY[g.name] = g


# ---- Planetoid citation networks (small, sparse, wide raw features) --------
# The classic GCN benchmarks.  hidden_dim = 16 is Kipf & Welling's setting and
# is deliberately kept, because a 16-wide FP16 feature row is 32 B -- half a
# DDR5 burst.  That is the regime the gather model is most sensitive in.

_register(GraphConfig(
    'Cora', num_nodes=2708, num_edges=10556, feat_dim=1433, num_classes=7,
    hidden_dim=16, source='Planetoid; 5278 undirected edges x2',
    max_degree=168, degree_source='commonly reported Cora max degree'))

_register(GraphConfig(
    'CiteSeer', num_nodes=3327, num_edges=9104, feat_dim=3703, num_classes=6,
    hidden_dim=16, source='Planetoid; 4552 undirected edges x2',
    max_degree=99, degree_source='commonly reported CiteSeer max degree'))

_register(GraphConfig(
    'PubMed', num_nodes=19717, num_edges=88648, feat_dim=500, num_classes=3,
    hidden_dim=16, source='Planetoid; 44324 undirected edges x2',
    max_degree=171, degree_source='commonly reported PubMed max degree'))

# ---- Large-scale graphs (deep enough that the feature matrix does not fit) --

_register(GraphConfig(
    'ogbn-arxiv', num_nodes=169343, num_edges=2332486, feat_dim=128,
    num_classes=40, hidden_dim=256,
    source='OGB; 1166243 directed citations, symmetrized',
    max_degree=13161, degree_source='OGB ogbn-arxiv max degree'))

_register(GraphConfig(
    'Reddit', num_nodes=232965, num_edges=114615892, feat_dim=602,
    num_classes=41, hidden_dim=256,
    source='GraphSAGE; directed non-zeros',
    max_degree=21657, degree_source='GraphSAGE Reddit max degree'))

_register(GraphConfig(
    'ogbn-products', num_nodes=2449029, num_edges=123718280, feat_dim=100,
    num_classes=47, hidden_dim=256,
    source='OGB; 61859140 undirected edges x2',
    max_degree=17481, degree_source='OGB ogbn-products max degree'))


def get_graph_config(name: str) -> GraphConfig:
    """Look up a graph workload, with the valid names in the error."""
    try:
        return GRAPH_REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown graph {name!r}; "
                       f"known: {', '.join(GRAPH_REGISTRY)}") from None


def list_graphs() -> List[str]:
    return list(GRAPH_REGISTRY)
