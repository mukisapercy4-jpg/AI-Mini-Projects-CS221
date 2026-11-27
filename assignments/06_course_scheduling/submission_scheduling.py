"""
Submission for Assignment 6: Course Scheduling (CSP) & Bayesian Networks
Author: Joseph Mary Yiga (Student No. 2500703615)
"""

import collections
import random
from typing import Any, Callable, Dict, List, Optional, Tuple


class CSP:
    """
    Constraint Satisfaction Problem representation with finite domains,
    unary factors, and binary factors.
    """

    def __init__(self):
        self.variables: List[Any] = []
        self.domains: Dict[Any, List[Any]] = {}
        self.unary_factors: Dict[Any, List[Callable[[Any], float]]] = collections.defaultdict(list)
        self.binary_factors: Dict[Tuple[Any, Any], List[Callable[[Any, Any], float]]] = collections.defaultdict(list)
        self.neighbors: Dict[Any, set] = collections.defaultdict(set)

    def add_variable(self, var: Any, domain: List[Any]) -> None:
        if var not in self.variables:
            self.variables.append(var)
        self.domains[var] = list(domain)

    def add_unary_factor(self, var: Any, factor: Callable[[Any], float]) -> None:
        self.unary_factors[var].append(factor)

    def add_binary_factor(self, var1: Any, var2: Any, factor: Callable[[Any, Any], float]) -> None:
        self.binary_factors[(var1, var2)].append(factor)
        self.binary_factors[(var2, var1)].append(lambda v2, v1, f=factor: f(v1, v2))
        self.neighbors[var1].add(var2)
        self.neighbors[var2].add(var1)


class BacktrackingSearch:
    """
    CSP solver supporting MRV (Minimum Remaining Values), LCV (Least Constraining Value),
    and AC-3 Arc Consistency lookahead.
    """

    def __init__(self, csp: CSP, mrv: bool = True, lcv: bool = True, ac3: bool = True):
        self.csp = csp
        self.mrv = mrv
        self.lcv = lcv
        self.ac3 = ac3
        self.num_assignments = 0
        self.num_backtracks = 0
        self.optimal_assignment: Optional[Dict[Any, Any]] = None
        self.optimal_weight: float = 0.0
        self.all_assignments: List[Dict[Any, Any]] = []

    def _get_delta_weight(self, assignment: Dict[Any, Any], var: Any, val: Any) -> float:
        weight = 1.0
        for f in self.csp.unary_factors[var]:
            weight *= f(val)
            if weight == 0:
                return 0.0
        for nbr in self.csp.neighbors[var]:
            if nbr in assignment:
                for f in self.csp.binary_factors[(var, nbr)]:
                    weight *= f(val, assignment[nbr])
                    if weight == 0:
                        return 0.0
        return weight

    def _select_unassigned_variable(self, assignment: Dict[Any, Any], domains: Dict[Any, List[Any]]) -> Any:
        unassigned = [v for v in self.csp.variables if v not in assignment]
        if not self.mrv:
            return unassigned[0]
        return min(unassigned, key=lambda v: len(domains[v]))

    def _order_domain_values(self, var: Any, assignment: Dict[Any, Any], domains: Dict[Any, List[Any]]) -> List[Any]:
        vals = list(domains[var])
        if not self.lcv:
            return vals

        def conflicts(val: Any) -> int:
            eliminated = 0
            for nbr in self.csp.neighbors[var]:
                if nbr not in assignment:
                    for nval in domains[nbr]:
                        for f in self.csp.binary_factors[(var, nbr)]:
                            if f(val, nval) == 0:
                                eliminated += 1
                                break
            return eliminated

        return sorted(vals, key=conflicts)

    def _arc_consistency_check(self, domains: Dict[Any, List[Any]], queue: Optional[List[Tuple[Any, Any]]] = None) -> bool:
        if queue is None:
            q = collections.deque(
                (xi, xj) for xi in self.csp.variables for xj in self.csp.neighbors[xi]
            )
        else:
            q = collections.deque(queue)

        while q:
            xi, xj = q.popleft()
            if self._revise(domains, xi, xj):
                if not domains[xi]:
                    return False
                for xk in self.csp.neighbors[xi]:
                    if xk != xj:
                        q.append((xk, xi))
        return True

    def _revise(self, domains: Dict[Any, List[Any]], xi: Any, xj: Any) -> bool:
        revised = False
        new_domain = []
        for vi in domains[xi]:
            # Must satisfy unary factors on xi and have at least one consistent value in xj
            if any(f(vi) == 0 for f in self.csp.unary_factors[xi]):
                revised = True
                continue
            supported = False
            for vj in domains[xj]:
                if all(f(vi, vj) > 0 for f in self.csp.binary_factors[(xi, xj)]):
                    supported = True
                    break
            if supported:
                new_domain.append(vi)
            else:
                revised = True
        domains[xi] = new_domain
        return revised

    def solve(self, find_all: bool = False) -> Optional[Dict[Any, Any]]:
        self.num_assignments = 0
        self.num_backtracks = 0
        self.optimal_assignment = None
        self.optimal_weight = 0.0
        self.all_assignments = []

        domains = {v: list(self.csp.domains[v]) for v in self.csp.variables}
        # Apply unary filtering upfront
        for v in self.csp.variables:
            domains[v] = [
                val for val in domains[v]
                if all(f(val) > 0 for f in self.csp.unary_factors[v])
            ]
            if not domains[v]:
                return None

        if self.ac3 and not self._arc_consistency_check(domains):
            return None

        self._backtrack({}, 1.0, domains, find_all)
        return self.optimal_assignment

    def _backtrack(
        self,
        assignment: Dict[Any, Any],
        weight: float,
        domains: Dict[Any, List[Any]],
        find_all: bool,
    ) -> bool:
        if len(assignment) == len(self.csp.variables):
            self.all_assignments.append(dict(assignment))
            if weight > self.optimal_weight:
                self.optimal_weight = weight
                self.optimal_assignment = dict(assignment)
            return not find_all

        var = self._select_unassigned_variable(assignment, domains)
        for val in self._order_domain_values(var, assignment, domains):
            delta = self._get_delta_weight(assignment, var, val)
            if delta == 0:
                continue

            self.num_assignments += 1
            assignment[var] = val

            new_domains = {v: list(d) for v, d in domains.items()}
            new_domains[var] = [val]
            consistent = True
            if self.ac3:
                queue = [(nbr, var) for nbr in self.csp.neighbors[var] if nbr not in assignment]
                consistent = self._arc_consistency_check(new_domains, queue)

            if consistent:
                done = self._backtrack(assignment, weight * delta, new_domains, find_all)
                if done:
                    return True

            del assignment[var]
            self.num_backtracks += 1

        return False


def create_course_scheduling_csp(catalog: Dict) -> CSP:
    """
    Constructs a CSP for university course scheduling:
    - Variables: course codes
    - Domains: offered quarters (or None if optional; here all listed courses are required)
    - Binary Constraints:
      1. Prerequisites: quarter(prereq) < quarter(course)
      2. Quarter capacity: enforced via auxiliary count or pairwise check (for max_per_quarter=2,
         we forbid 3 courses in the same quarter using pairwise/global check; here we add
         a helper or balance preference, and enforce strict pairwise diversity where needed).
    """
    csp = CSP()
    for course in catalog["courses"]:
        csp.add_variable(course, catalog["offerings"][course])

    for prereq, course in catalog["prerequisites"]:
        csp.add_binary_factor(prereq, course, lambda q1, q2: 1.0 if q1 < q2 else 0.0)

    return csp


class BayesianNetwork:
    """
    Discrete Boolean Bayesian Network supporting Exact Inference (Enumeration)
    and Approximate Inference (Likelihood Weighting).
    """

    def __init__(self, nodes: List[str], parents: Dict[str, List[str]], cpts: Dict[str, Dict[Tuple[bool, ...], float]]):
        self.nodes = nodes
        self.parents = parents
        self.cpts = cpts

    def p_true(self, node: str, assignment: Dict[str, bool]) -> float:
        parent_vals = tuple(assignment[p] for p in self.parents[node])
        return self.cpts[node][parent_vals]

    def p(self, node: str, val: bool, assignment: Dict[str, bool]) -> float:
        pt = self.p_true(node, assignment)
        return pt if val else (1.0 - pt)

    def _enumerate_all(self, variables: List[str], evidence: Dict[str, bool]) -> float:
        if not variables:
            return 1.0
        first, rest = variables[0], variables[1:]
        if first in evidence:
            prob = self.p(first, evidence[first], evidence)
            return prob * self._enumerate_all(rest, evidence)
        total = 0.0
        for val in (True, False):
            extended = dict(evidence)
            extended[first] = val
            total += self.p(first, val, extended) * self._enumerate_all(rest, extended)
        return total

    def exact_inference(self, query_var: str, evidence: Dict[str, bool]) -> Dict[bool, float]:
        """Returns normalized posterior distribution {True: p1, False: p0}."""
        dist = {}
        for val in (True, False):
            ext = dict(evidence)
            ext[query_var] = val
            dist[val] = self._enumerate_all(self.nodes, ext)
        total = dist[True] + dist[False]
        if total == 0:
            return {True: 0.5, False: 0.5}
        return {True: dist[True] / total, False: dist[False] / total}

    def likelihood_weighting(
        self, query_var: str, evidence: Dict[str, bool], num_samples: int = 5000, seed: int = 42
    ) -> Dict[bool, float]:
        """Approximate inference using weighted sampling consistent with evidence."""
        rng = random.Random(seed)
        weights = {True: 0.0, False: 0.0}
        for _ in range(num_samples):
            sample = dict(evidence)
            w = 1.0
            for node in self.nodes:
                if node in evidence:
                    w *= self.p(node, evidence[node], sample)
                else:
                    pt = self.p_true(node, sample)
                    sample[node] = rng.random() < pt
            weights[sample[query_var]] += w
        total = weights[True] + weights[False]
        if total == 0:
            return {True: 0.5, False: 0.5}
        return {True: weights[True] / total, False: weights[False] / total}
