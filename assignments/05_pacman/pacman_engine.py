"""
Self-Contained Pac-Man Environment Engine
Supports simulation of multi-agent Pac-Man games for Stanford CS221 Assignment 5.
"""

import copy
import random
from typing import List, Tuple, Dict, Any, Optional

class Directions:
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    STOP = 'Stop'

    VECTORS = {
        NORTH: (0, 1),
        SOUTH: (0, -1),
        EAST: (1, 0),
        WEST: (-1, 0),
        STOP: (0, 0),
    }

def manhattanDistance(xy1: Tuple[int, int], xy2: Tuple[int, int]) -> int:
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class GhostState:
    def __init__(self, position: Tuple[int, int], scaredTimer: int = 0):
        self.position = position
        self.scaredTimer = scaredTimer

    def getPosition(self) -> Tuple[int, int]:
        return self.position

class FoodGrid:
    def __init__(self, width: int, height: int, initial_food: List[Tuple[int, int]]):
        self.width = width
        self.height = height
        self.data = [[False for _ in range(height)] for _ in range(width)]
        for x, y in initial_food:
            if 0 <= x < width and 0 <= y < height:
                self.data[x][y] = True

    def __getitem__(self, idx):
        x, y = idx
        return self.data[x][y]

    def count(self) -> int:
        return sum(sum(1 for cell in col if cell) for col in self.data)

    def asList(self) -> List[Tuple[int, int]]:
        result = []
        for x in range(self.width):
            for y in range(self.height):
                if self.data[x][y]:
                    result.append((x, y))
        return result

    def copy(self):
        new_grid = FoodGrid(self.width, self.height, [])
        new_grid.data = [col[:] for col in self.data]
        return new_grid

class PacmanGameState:
    def __init__(
        self,
        width: int = 9,
        height: int = 7,
        walls: Optional[List[Tuple[int, int]]] = None,
        pacman_pos: Tuple[int, int] = (1, 1),
        ghost_positions: Optional[List[Tuple[int, int]]] = None,
        food_positions: Optional[List[Tuple[int, int]]] = None,
        capsules: Optional[List[Tuple[int, int]]] = None,
        score: float = 0.0,
        scaredTimer: int = 0
    ):
        self.width = width
        self.height = height
        self.walls = set(walls) if walls else self._default_walls(width, height)
        self.pacman_pos = pacman_pos
        self.ghost_states = [GhostState(pos, scaredTimer) for pos in (ghost_positions or [(width - 2, height - 2)])]
        self.food = FoodGrid(width, height, food_positions if food_positions is not None else self._default_food())
        self.capsules = list(capsules) if capsules is not None else [(width - 2, 1)]
        self.score = score
        self._is_win = False
        self._is_lose = False

    def _default_walls(self, width: int, height: int) -> set:
        walls = set()
        for x in range(width):
            walls.add((x, 0))
            walls.add((x, height - 1))
        for y in range(height):
            walls.add((0, y))
            walls.add((width - 1, y))
        # Inner obstacle wall
        walls.add((width // 2, height // 2))
        return walls

    def _default_food(self) -> List[Tuple[int, int]]:
        food = []
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                if (x, y) not in self.walls and (x, y) != self.pacman_pos:
                    food.append((x, y))
        return food

    def getPacmanPosition(self) -> Tuple[int, int]:
        return self.pacman_pos

    def getGhostStates(self) -> List[GhostState]:
        return self.ghost_states

    def getGhostPosition(self, agentIndex: int) -> Tuple[int, int]:
        return self.ghost_states[agentIndex - 1].getPosition()

    def getFood(self) -> FoodGrid:
        return self.food

    def getCapsules(self) -> List[Tuple[int, int]]:
        return self.capsules

    def getScore(self) -> float:
        return self.score

    def getNumAgents(self) -> int:
        return 1 + len(self.ghost_states)

    def isWin(self) -> bool:
        return self._is_win or (self.food.count() == 0 and not self._is_lose)

    def isLose(self) -> bool:
        return self._is_lose

    def getLegalActions(self, agentIndex: int = 0) -> List[str]:
        if self.isWin() or self.isLose():
            return []

        if agentIndex == 0:
            x, y = self.pacman_pos
        else:
            x, y = self.ghost_states[agentIndex - 1].getPosition()

        legal = []
        for action, (dx, dy) in Directions.VECTORS.items():
            if action == Directions.STOP and agentIndex != 0:
                continue
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) not in self.walls:
                legal.append(action)
        return legal

    def generateSuccessor(self, agentIndex: int, action: str) -> 'PacmanGameState':
        if self.isWin() or self.isLose():
            return self

        next_state = PacmanGameState(
            width=self.width,
            height=self.height,
            walls=list(self.walls),
            pacman_pos=self.pacman_pos,
            ghost_positions=[g.getPosition() for g in self.ghost_states],
            food_positions=self.food.asList(),
            capsules=self.capsules,
            score=self.score - 1.0,  # Step penalty
            scaredTimer=self.ghost_states[0].scaredTimer
        )
        # Deep copy ghost timers
        for i, g in enumerate(self.ghost_states):
            next_state.ghost_states[i].scaredTimer = max(0, g.scaredTimer - (1 if agentIndex == 0 else 0))

        dx, dy = Directions.VECTORS.get(action, (0, 0))

        if agentIndex == 0:
            # Pacman Move
            new_pos = (self.pacman_pos[0] + dx, self.pacman_pos[1] + dy)
            next_state.pacman_pos = new_pos

            # Check capsule
            if new_pos in next_state.capsules:
                next_state.capsules.remove(new_pos)
                next_state.score += 50.0
                for g in next_state.ghost_states:
                    g.scaredTimer = 15

            # Check food
            if next_state.food[new_pos[0], new_pos[1]]:
                next_state.food.data[new_pos[0]][new_pos[1]] = False
                next_state.score += 10.0
                if next_state.food.count() == 0:
                    next_state.score += 500.0
                    next_state._is_win = True

            # Collision check
            for g in next_state.ghost_states:
                if g.getPosition() == new_pos:
                    if g.scaredTimer > 0:
                        next_state.score += 200.0
                        g.position = (self.width - 2, self.height - 2)
                        g.scaredTimer = 0
                    else:
                        next_state.score -= 500.0
                        next_state._is_lose = True

        else:
            # Ghost Move
            ghost_idx = agentIndex - 1
            old_pos = self.ghost_states[ghost_idx].getPosition()
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
            next_state.ghost_states[ghost_idx].position = new_pos

            # Collision check with Pac-Man
            if new_pos == next_state.pacman_pos:
                if next_state.ghost_states[ghost_idx].scaredTimer > 0:
                    next_state.score += 200.0
                    next_state.ghost_states[ghost_idx].position = (self.width - 2, self.height - 2)
                    next_state.ghost_states[ghost_idx].scaredTimer = 0
                else:
                    next_state.score -= 500.0
                    next_state._is_lose = True

        return next_state

    def generatePacmanSuccessor(self, action: str) -> 'PacmanGameState':
        return self.generateSuccessor(0, action)

    def render(self) -> str:
        grid = [[' ' for _ in range(self.height)] for _ in range(self.width)]
        for x, y in self.walls:
            grid[x][y] = '%'
        for x, y in self.food.asList():
            grid[x][y] = '.'
        for x, y in self.capsules:
            grid[x][y] = 'o'
        for g in self.ghost_states:
            gx, gy = g.getPosition()
            grid[gx][gy] = 'S' if g.scaredTimer > 0 else 'G'
        px, py = self.pacman_pos
        grid[px][py] = 'P'

        lines = []
        for y in range(self.height - 1, -1, -1):
            lines.append("".join(grid[x][y] for x in range(self.width)))
        return "\n".join(lines)
