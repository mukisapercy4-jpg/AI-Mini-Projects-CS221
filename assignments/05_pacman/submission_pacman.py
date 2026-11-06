"""
Pac-Man Multi-Agent Adversarial Search Implementation
Assigned to: Mukisa Percy & Jill
Course: Makerere University AI Mini-Projects (CS221 Aligned)

This module implements multi-agent adversarial search algorithms for Pac-Man:
- ReflexAgent: State-action evaluation
- MinimaxAgent: Multi-agent adversarial minimax tree search
- AlphaBetaAgent: Minimax search with alpha-beta pruning
- ExpectimaxAgent: Multi-agent expectimax search for probabilistic ghost models
- betterEvaluationFunction: Linear feature combination for state evaluations
"""

import random
from typing import List, Tuple, Dict, Any, Optional

# =============================================================================
# ABSTRACT BASE CLASSES & DATA TYPES
# =============================================================================

class Agent:
    """
    An agent must define a getAction method.
    """
    def __init__(self, index: int = 0):
        self.index = index

    def getAction(self, gameState: Any) -> str:
        raise NotImplementedError


class MultiAgentSearchAgent(Agent):
    """
    Common base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """
    def __init__(self, evalFn: str = 'scoreEvaluationFunction', depth: int = 2):
        super().__init__(index=0)
        self.evaluationFunction = evalFn
        self.depth = int(depth)


# =============================================================================
# 1. REFLEX AGENT
# =============================================================================

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining its
    alternatives via a state evaluation function.
    """
    def getAction(self, gameState: Any) -> str:
        """
        Collects legal moves and chooses the action with highest evaluation score.
        """
        legalMoves = gameState.getLegalActions(0)
        if not legalMoves:
            return "Stop"

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: Any, action: str) -> float:
        """
        Evaluates the action in currentGameState.
        Combines food distance and ghost danger penalty.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        # Ghost penalty
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0:
                if dist <= 1:
                    return -float('inf')
                score -= 2.0 / (dist + 0.1)
            else:
                score += 5.0 / (dist + 0.1)

        # Food incentive
        foodList = newFood.asList() if hasattr(newFood, 'asList') else []
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 10.0 / (minFoodDist + 1.0)
        else:
            score += 100.0

        return score


# =============================================================================
# 2. MINIMAX AGENT
# =============================================================================

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Multi-agent Minimax search agent.
    Pac-Man is agent index 0 (MAX).
    Ghosts are agents 1, 2, ... (MIN).
    """

    def getAction(self, gameState: Any) -> str:
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def value(state: Any, agentIndex: int, currentDepth: int) -> Tuple[float, Optional[str]]:
            # Terminal states check
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.getEval(state), None

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

            if agentIndex == 0:
                return maxValue(state, agentIndex, currentDepth, nextAgent, nextDepth)
            else:
                return minValue(state, agentIndex, currentDepth, nextAgent, nextDepth)

        def maxValue(state, agentIndex, currentDepth, nextAgent, nextDepth):
            bestScore = -float('inf')
            bestAction = None
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.getEval(state), None

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = value(successor, nextAgent, nextDepth)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            return bestScore, bestAction

        def minValue(state, agentIndex, currentDepth, nextAgent, nextDepth):
            bestScore = float('inf')
            bestAction = None
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.getEval(state), None

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = value(successor, nextAgent, nextDepth)
                if score < bestScore:
                    bestScore = score
                    bestAction = action
            return bestScore, bestAction

        _, chosenAction = value(gameState, 0, 0)
        return chosenAction or "Stop"

    def getEval(self, state: Any) -> float:
        if callable(self.evaluationFunction):
            return self.evaluationFunction(state)
        return state.getScore()


# =============================================================================
# 3. ALPHA-BETA PRUNING AGENT
# =============================================================================

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with Alpha-Beta pruning across multiple adversary ghosts.
    """

    def getAction(self, gameState: Any) -> str:
        """
        Returns the minimax action using alpha-beta pruning.
        """
        def value(state: Any, agentIndex: int, currentDepth: int, alpha: float, beta: float) -> Tuple[float, Optional[str]]:
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.getEval(state), None

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

            if agentIndex == 0:
                return maxValue(state, agentIndex, currentDepth, nextAgent, nextDepth, alpha, beta)
            else:
                return minValue(state, agentIndex, currentDepth, nextAgent, nextDepth, alpha, beta)

        def maxValue(state, agentIndex, currentDepth, nextAgent, nextDepth, alpha, beta):
            bestScore = -float('inf')
            bestAction = None
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.getEval(state), None

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = value(successor, nextAgent, nextDepth, alpha, beta)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                if bestScore > beta:
                    return bestScore, bestAction
                alpha = max(alpha, bestScore)
            return bestScore, bestAction

        def minValue(state, agentIndex, currentDepth, nextAgent, nextDepth, alpha, beta):
            bestScore = float('inf')
            bestAction = None
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.getEval(state), None

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = value(successor, nextAgent, nextDepth, alpha, beta)
                if score < bestScore:
                    bestScore = score
                    bestAction = action
                if bestScore < alpha:
                    return bestScore, bestAction
                beta = min(beta, bestScore)
            return bestScore, bestAction

        alpha = -float('inf')
        beta = float('inf')
        _, chosenAction = value(gameState, 0, 0, alpha, beta)
        return chosenAction or "Stop"

    def getEval(self, state: Any) -> float:
        if callable(self.evaluationFunction):
            return self.evaluationFunction(state)
        return state.getScore()


# =============================================================================
# 4. EXPECTIMAX AGENT
# =============================================================================

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent modeling adversaries as choosing actions uniformly at random.
    """

    def getAction(self, gameState: Any) -> str:
        """
        Returns the expectimax action from current gameState.
        """
        def value(state: Any, agentIndex: int, currentDepth: int) -> Tuple[float, Optional[str]]:
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.getEval(state), None

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = currentDepth + 1 if nextAgent == 0 else currentDepth

            if agentIndex == 0:
                return maxValue(state, agentIndex, currentDepth, nextAgent, nextDepth)
            else:
                return expValue(state, agentIndex, currentDepth, nextAgent, nextDepth)

        def maxValue(state, agentIndex, currentDepth, nextAgent, nextDepth):
            bestScore = -float('inf')
            bestAction = None
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.getEval(state), None

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = value(successor, nextAgent, nextDepth)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            return bestScore, bestAction

        def expValue(state, agentIndex, currentDepth, nextAgent, nextDepth):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.getEval(state), None

            totalScore = 0.0
            prob = 1.0 / len(legalActions)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = value(successor, nextAgent, nextDepth)
                totalScore += prob * score
            return totalScore, None

        _, chosenAction = value(gameState, 0, 0)
        return chosenAction or "Stop"

    def getEval(self, state: Any) -> float:
        if callable(self.evaluationFunction):
            return self.evaluationFunction(state)
        return state.getScore()


# =============================================================================
# 5. BETTER EVALUATION FUNCTION
# =============================================================================

def betterEvaluationFunction(currentGameState: Any) -> float:
    """
    Extreme evaluation function for Pac-Man game states.
    Features:
    - Current game score
    - Distance to closest food pellet
    - Number of remaining food pellets
    - Distance to active and scared ghosts
    - Number of remaining power capsules
    """
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')

    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList() if hasattr(foodGrid, 'asList') else []
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules() if hasattr(currentGameState, 'getCapsules') else []

    # 1. Food proximity feature
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, f) for f in foodList)
        score += 2.0 / minFoodDist
        score -= 4.0 * len(foodList)

    # 2. Capsule proximity & remaining count
    if capsules:
        minCapDist = min(manhattanDistance(pacmanPos, c) for c in capsules)
        score += 3.0 / minCapDist
        score -= 20.0 * len(capsules)

    # 3. Ghost distance consideration
    for ghost in ghostStates:
        gDist = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            # Ghost is vulnerable: prioritize hunting it
            score += 10.0 / (gDist + 0.1)
        else:
            # Active ghost: maintain safe distance
            if gDist <= 1:
                return -float('inf')
            score -= 3.0 / (gDist + 0.1)

    return score


def manhattanDistance(xy1: Tuple[int, int], xy2: Tuple[int, int]) -> int:
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
