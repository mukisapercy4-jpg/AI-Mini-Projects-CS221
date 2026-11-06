"""
Test Harness for Pac-Man Search Agents
Verifies Agent classes, method signatures, and basic logic.
"""

from submission_pacman import (
    ReflexAgent,
    MinimaxAgent,
    AlphaBetaAgent,
    ExpectimaxAgent,
    betterEvaluationFunction,
    manhattanDistance
)

class MockGhost:
    def __init__(self, pos, scaredTimer=0):
        self.pos = pos
        self.scaredTimer = scaredTimer
    def getPosition(self):
        return self.pos

class MockGrid:
    def __init__(self, food_positions):
        self.food_positions = food_positions
    def asList(self):
        return self.food_positions

class MockGameState:
    def __init__(self, pacman_pos=(1, 1), ghost_positions=[(3, 3)], food_positions=[(1, 2)], score=10.0, is_win=False, is_lose=False):
        self.pacman_pos = pacman_pos
        self.ghost_states = [MockGhost(pos) for pos in ghost_positions]
        self.food_grid = MockGrid(food_positions)
        self._score = score
        self._is_win = is_win
        self._is_lose = is_lose

    def isWin(self):
        return self._is_win

    def isLose(self):
        return self._is_lose

    def getScore(self):
        return self._score

    def getPacmanPosition(self):
        return self.pacman_pos

    def getGhostStates(self):
        return self.ghost_states

    def getFood(self):
        return self.food_grid

    def getCapsules(self):
        return []

    def getNumAgents(self):
        return 1 + len(self.ghost_states)

    def getLegalActions(self, agentIndex=0):
        return ["North", "South", "East", "West"]

    def generateSuccessor(self, agentIndex, action):
        return MockGameState(
            pacman_pos=self.pacman_pos,
            ghost_positions=[g.getPosition() for g in self.ghost_states],
            food_positions=self.food_grid.asList(),
            score=self._score + 1.0
        )

def run_tests():
    print("Testing Manhattan Distance...")
    assert manhattanDistance((0, 0), (3, 4)) == 7
    print("  [PASS] Manhattan distance calculation.")

    state = MockGameState()

    print("Testing ReflexAgent...")
    reflex = ReflexAgent()
    assert hasattr(reflex, "getAction")
    print("  [PASS] ReflexAgent initialized.")

    print("Testing MinimaxAgent...")
    minimax = MinimaxAgent(depth=2)
    action = minimax.getAction(state)
    assert action in ["North", "South", "East", "West", "Stop"]
    print(f"  [PASS] MinimaxAgent selected action: {action}")

    print("Testing AlphaBetaAgent...")
    alphabeta = AlphaBetaAgent(depth=2)
    ab_action = alphabeta.getAction(state)
    assert ab_action in ["North", "South", "East", "West", "Stop"]
    print(f"  [PASS] AlphaBetaAgent selected action: {ab_action}")

    print("Testing ExpectimaxAgent...")
    expectimax = ExpectimaxAgent(depth=2)
    exp_action = expectimax.getAction(state)
    assert exp_action in ["North", "South", "East", "West", "Stop"]
    print(f"  [PASS] ExpectimaxAgent selected action: {exp_action}")

    print("Testing betterEvaluationFunction...")
    eval_val = betterEvaluationFunction(state)
    assert isinstance(eval_val, (int, float))
    print(f"  [PASS] betterEvaluationFunction returned value: {eval_val}")

    print("\nALL BASIC PAC-MAN TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
