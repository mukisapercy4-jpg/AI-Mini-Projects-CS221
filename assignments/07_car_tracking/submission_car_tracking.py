"""
Submission for Assignment 7: Car Tracking (HMM & Particle Filtering) & Logic
Author: Muzira Joshua Kwizera (Student No. 2500703484)
"""

import collections
import math
import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


def gaussian_pdf(mean: float, std: float, x: float) -> float:
    """Normal density N(x; mean, std^2)."""
    var = std * std
    return math.exp(-((x - mean) ** 2) / (2.0 * var)) / math.sqrt(2.0 * math.pi * var)


class ExactInference:
    """
    Exact Forward HMM Inference over a discrete (rows x cols) grid.
    """

    def __init__(self, rows: int, cols: int, transition_probs: Dict[Tuple[int, int], Dict[Tuple[int, int], float]], sensor_std: float = 1.0):
        """Initialize this object's state and configuration."""
        self.rows = rows
        self.cols = cols
        self.transition_probs = transition_probs
        self.sensor_std = sensor_std
        uniform = 1.0 / (rows * cols)
        self.belief: Dict[Tuple[int, int], float] = {
            (r, c): uniform for r in range(rows) for c in range(cols)
        }

    def _normalize(self) -> None:
        """Normalize state probabilities, using a uniform fallback if all weights vanish."""
        total = sum(self.belief.values())
        if total == 0:
            uniform = 1.0 / (self.rows * self.cols)
            for k in self.belief:
                self.belief[k] = uniform
        else:
            for k in self.belief:
                self.belief[k] /= total

    def observe(self, obs_r: int, obs_c: int, observed_dist: float) -> None:
        """Bayes update: b'(s) proportional to b(s) * P(observed_dist | s)."""
        for r, c in self.belief:
            true_dist = math.hypot(r - obs_r, c - obs_c)
            likelihood = gaussian_pdf(true_dist, self.sensor_std, observed_dist)
            self.belief[(r, c)] *= likelihood
        self._normalize()

    def elapse_time(self) -> None:
        """Forward transition update: b'(s') = sum_s b(s) * P(s' | s)."""
        new_belief: Dict[Tuple[int, int], float] = collections.defaultdict(float)
        for s, prob in self.belief.items():
            if prob == 0:
                continue
            for nxt, t_prob in self.transition_probs[s].items():
                new_belief[nxt] += prob * t_prob
        self.belief = {
            (r, c): new_belief.get((r, c), 0.0)
            for r in range(self.rows)
            for c in range(self.cols)
        }
        self._normalize()

    def get_belief(self) -> Dict[Tuple[int, int], float]:
        """Return a copy of the current normalized state distribution."""
        return dict(self.belief)

    def map_estimate(self) -> Tuple[int, int]:
        """Return the most probable state under the current belief."""
        return max(self.belief, key=self.belief.get)


class ParticleFilter:
    """
    Sequential Monte Carlo (Particle Filtering) for Car Tracking.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        transition_probs: Dict[Tuple[int, int], Dict[Tuple[int, int], float]],
        sensor_std: float = 1.0,
        num_particles: int = 500,
        seed: int = 42,
    ):
        """Initialize this object's state and configuration."""
        self.rows = rows
        self.cols = cols
        self.transition_probs = transition_probs
        self.sensor_std = sensor_std
        self.num_particles = num_particles
        self.rng = random.Random(seed)
        self.particles: Dict[Tuple[int, int], int] = collections.Counter(
            (self.rng.randrange(rows), self.rng.randrange(cols))
            for _ in range(num_particles)
        )

    def observe(self, obs_r: int, obs_c: int, observed_dist: float) -> None:
        """Weight particle counts by sensor likelihood and resample."""
        weighted: Dict[Tuple[int, int], float] = {}
        for (r, c), count in self.particles.items():
            true_dist = math.hypot(r - obs_r, c - obs_c)
            weighted[(r, c)] = count * gaussian_pdf(true_dist, self.sensor_std, observed_dist)

        total = sum(weighted.values())
        if total == 0:
            self.particles = collections.Counter(
                (self.rng.randrange(self.rows), self.rng.randrange(self.cols))
                for _ in range(self.num_particles)
            )
            return

        # Systematic resampling
        states = list(weighted.keys())
        probs = [weighted[s] / total for s in states]
        cdf = []
        running = 0.0
        for p in probs:
            running += p
            cdf.append(running)
        cdf[-1] = 1.0

        step = 1.0 / self.num_particles
        u0 = self.rng.random() * step
        new_particles: collections.Counter = collections.Counter()
        idx = 0
        for i in range(self.num_particles):
            u = u0 + i * step
            while u > cdf[idx]:
                idx += 1
            new_particles[states[idx]] += 1
        self.particles = new_particles

    def elapse_time(self) -> None:
        """Propagate each particle forward according to the transition model."""
        new_particles: collections.Counter = collections.Counter()
        for state, count in self.particles.items():
            trans = self.transition_probs[state]
            nxt_states = list(trans.keys())
            nxt_probs = list(trans.values())
            chosen = self.rng.choices(nxt_states, weights=nxt_probs, k=count)
            new_particles.update(chosen)
        self.particles = new_particles

    def get_belief(self) -> Dict[Tuple[int, int], float]:
        """Return a copy of the current normalized state distribution."""
        total = sum(self.particles.values())
        return {s: cnt / total for s, cnt in self.particles.items()}

    def map_estimate(self) -> Tuple[int, int]:
        """Return the most probable state under the current belief."""
        return max(self.particles, key=self.particles.get)


# =============================================================================
# PART 2: PROPOSITIONAL LOGIC, CNF TRANSFORMATION & RESOLUTION
# =============================================================================

class Formula:
    """Define the base type for propositional logic formulas."""
    pass


class Atom(Formula):
    """Represent a propositional variable by its name."""
    def __init__(self, name: str):
        """Initialize this object's state and configuration."""
        self.name = name

    def __repr__(self) -> str:
        """Support the   repr   operation used by this algorithm."""
        return self.name

    def __eq__(self, other) -> bool:
        """Support the   eq   operation used by this algorithm."""
        return isinstance(other, Atom) and self.name == other.name

    def __hash__(self) -> int:
        """Support the   hash   operation used by this algorithm."""
        return hash(("Atom", self.name))


class Not(Formula):
    """Represent logical negation of a formula."""
    def __init__(self, arg: Formula):
        """Initialize this object's state and configuration."""
        self.arg = arg

    def __repr__(self) -> str:
        """Support the   repr   operation used by this algorithm."""
        return f"~{self.arg}"


class And(Formula):
    """Represent the conjunction of two formulas."""
    def __init__(self, left: Formula, right: Formula):
        """Initialize this object's state and configuration."""
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        """Support the   repr   operation used by this algorithm."""
        return f"({self.left} & {self.right})"


class Or(Formula):
    """Represent the disjunction of two formulas."""
    def __init__(self, left: Formula, right: Formula):
        """Initialize this object's state and configuration."""
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        """Support the   repr   operation used by this algorithm."""
        return f"({self.left} | {self.right})"


class Implies(Formula):
    """Represent an implication between two formulas."""
    def __init__(self, left: Formula, right: Formula):
        """Initialize this object's state and configuration."""
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        """Support the   repr   operation used by this algorithm."""
        return f"({self.left} -> {self.right})"


class Equiv(Formula):
    """Represent logical equivalence between two formulas."""
    def __init__(self, left: Formula, right: Formula):
        """Initialize this object's state and configuration."""
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        """Support the   repr   operation used by this algorithm."""
        return f"({self.left} <-> {self.right})"


def _eliminate_implications(f: Formula) -> Formula:
    """Rewrite implication and equivalence operators using NOT and OR."""
    if isinstance(f, Atom):
        return f
    if isinstance(f, Not):
        return Not(_eliminate_implications(f.arg))
    if isinstance(f, And):
        return And(_eliminate_implications(f.left), _eliminate_implications(f.right))
    if isinstance(f, Or):
        return Or(_eliminate_implications(f.left), _eliminate_implications(f.right))
    if isinstance(f, Implies):
        return Or(Not(_eliminate_implications(f.left)), _eliminate_implications(f.right))
    if isinstance(f, Equiv):
        l = _eliminate_implications(f.left)
        r = _eliminate_implications(f.right)
        return And(Or(Not(l), r), Or(Not(r), l))
    raise TypeError(f"Unknown formula: {f}")


def _push_negations(f: Formula) -> Formula:
    """Move negations inward until they apply only to atoms."""
    if isinstance(f, Atom):
        return f
    if isinstance(f, Not):
        inner = f.arg
        if isinstance(inner, Atom):
            return f
        if isinstance(inner, Not):
            return _push_negations(inner.arg)
        if isinstance(inner, And):
            return Or(_push_negations(Not(inner.left)), _push_negations(Not(inner.right)))
        if isinstance(inner, Or):
            return And(_push_negations(Not(inner.left)), _push_negations(Not(inner.right)))
    if isinstance(f, And):
        return And(_push_negations(f.left), _push_negations(f.right))
    if isinstance(f, Or):
        return Or(_push_negations(f.left), _push_negations(f.right))
    raise TypeError(f"Unexpected formula in NNF: {f}")


def _distribute_or(a: Formula, b: Formula) -> Formula:
    """Distribute OR over AND to build conjunctive normal form."""
    if isinstance(a, And):
        return And(_distribute_or(a.left, b), _distribute_or(a.right, b))
    if isinstance(b, And):
        return And(_distribute_or(a, b.left), _distribute_or(a, b.right))
    return Or(a, b)


def _to_cnf_ast(f: Formula) -> Formula:
    """Recursively convert a negation-normal formula tree to CNF."""
    if isinstance(f, (Atom, Not)):
        return f
    if isinstance(f, And):
        return And(_to_cnf_ast(f.left), _to_cnf_ast(f.right))
    if isinstance(f, Or):
        return _distribute_or(_to_cnf_ast(f.left), _to_cnf_ast(f.right))
    raise TypeError(f"Unexpected formula in CNF: {f}")


def _extract_clauses(f: Formula) -> Set[FrozenSet[str]]:
    """Extract a CNF formula as a set of literal clauses."""
    if isinstance(f, And):
        return _extract_clauses(f.left) | _extract_clauses(f.right)

    def _literals(sub: Formula) -> Set[str]:
        """Flatten a disjunction into its set of clause literals."""
        if isinstance(sub, Or):
            return _literals(sub.left) | _literals(sub.right)
        if isinstance(sub, Atom):
            return {sub.name}
        if isinstance(sub, Not) and isinstance(sub.arg, Atom):
            return {f"~{sub.arg.name}"}
        raise TypeError(f"Non-literal in clause: {sub}")

    return {frozenset(_literals(f))}


def to_cnf(f: Formula) -> Set[FrozenSet[str]]:
    """Converts a propositional Formula into a set of CNF clauses (frozensets of literals)."""
    nnf = _push_negations(_eliminate_implications(f))
    cnf_ast = _to_cnf_ast(nnf)
    return _extract_clauses(cnf_ast)


def _negate_literal(lit: str) -> str:
    """Return the complementary literal."""
    return lit[1:] if lit.startswith("~") else f"~{lit}"


def resolution_entails(kb: List[Formula], query: Formula) -> bool:
    """
    Proves KB |= query via Resolution Refutation:
    Checks whether KB & ~query derives the empty clause {}.
    """
    clauses: Set[FrozenSet[str]] = set()
    for f in kb:
        clauses |= to_cnf(f)
    clauses |= to_cnf(Not(query))

    new_clauses: Set[FrozenSet[str]] = set()
    while True:
        clause_list = list(clauses)
        n = len(clause_list)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = clause_list[i], clause_list[j]
                for lit in c1:
                    comp = _negate_literal(lit)
                    if comp in c2:
                        resolvent = (c1 - {lit}) | (c2 - {comp})
                        if not resolvent:
                            return True
                        new_clauses.add(resolvent)
        if new_clauses.issubset(clauses):
            return False
        clauses |= new_clauses
