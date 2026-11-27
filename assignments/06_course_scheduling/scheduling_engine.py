"""
Course Scheduling & Bayesian Network Engine — Assignment 6
Provides sample university course catalogs and Bayesian network definitions.
"""

from typing import Dict, List, Tuple


def get_sample_course_catalog() -> Dict:
    """
    Returns a realistic Makerere-inspired CS curriculum scheduling problem:
    - quarters: list of available terms
    - courses: list of course codes
    - prerequisites: list of (prereq, course) meaning prereq must be taken strictly before course
    - offerings: dict mapping course -> list of quarters when it is offered
    - max_per_quarter: maximum courses allowed per quarter
    """
    quarters = [1, 2, 3, 4]
    courses = ["CS101", "CS102", "MATH101", "CS201", "CS202", "AI301"]
    prerequisites = [
        ("CS101", "CS102"),
        ("CS102", "CS201"),
        ("MATH101", "CS202"),
        ("CS201", "AI301"),
        ("CS202", "AI301"),
    ]
    offerings = {
        "CS101": [1, 2],
        "MATH101": [1, 2],
        "CS102": [2, 3],
        "CS201": [3, 4],
        "CS202": [2, 3, 4],
        "AI301": [4],
    }
    return {
        "quarters": quarters,
        "courses": courses,
        "prerequisites": prerequisites,
        "offerings": offerings,
        "max_per_quarter": 2,
    }


def get_alarm_bayesian_network() -> Dict:
    """
    Classic Pearl Alarm Bayesian Network:
      Burglary (B), Earthquake (E) -> Alarm (A) -> JohnCalls (J), MaryCalls (M)
    Returns node ordering and Conditional Probability Tables (CPTs).
    """
    return {
        "nodes": ["B", "E", "A", "J", "M"],
        "parents": {
            "B": [],
            "E": [],
            "A": ["B", "E"],
            "J": ["A"],
            "M": ["A"],
        },
        "cpts": {
            "B": {(): 0.001},
            "E": {(): 0.002},
            "A": {
                (True, True): 0.95,
                (True, False): 0.94,
                (False, True): 0.29,
                (False, False): 0.001,
            },
            "J": {(True,): 0.90, (False,): 0.05},
            "M": {(True,): 0.70, (False,): 0.01},
        },
    }
