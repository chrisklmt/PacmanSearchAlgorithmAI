# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import search
from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getAvailableActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateNextState(agentIndex, action):
        Returns the nextState game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        numAgents = gameState.getNumAgents()
        def minimax(state, agentIndex, depth):
            # Τερματικές καταστάσεις: win, lose, βάθος reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # Υπολογισμός επόμενου agent και βάθους
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            legalActions = state.getAvailableActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            # Pacman: max
            if agentIndex == 0:
                bestValue = float('-inf')
                for action in legalActions:
                    succ = state.generateNextState(agentIndex, action)
                    val = minimax(succ, nextAgent, nextDepth)
                    bestValue = max(bestValue, val)
                return bestValue
            else:
                worstValue = float('inf')
                for action in legalActions:
                    succ = state.generateNextState(agentIndex, action)
                    val = minimax(succ, nextAgent, nextDepth)
                    worstValue = min(worstValue, val)
                return worstValue
        # Η καλύτερη κίνηση για το Pacman (όπου agentIndex=0)
        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in gameState.getAvailableActions(0):
            succ = gameState.generateNextState(0, action)
            score = minimax(succ, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        numAgents = gameState.getNumAgents()
        def alphabeta(state, agentIndex, depth, alpha, beta):
            # Τερματικές καταστάσεις
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # Προετοιμασία επόμενου agent και βάθους
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            actions = state.getAvailableActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            # Pacman (max node)
            if agentIndex == 0:
                value = float('-inf')
                for action in actions:
                    succ = state.generateNextState(agentIndex, action)
                    value = max(value, alphabeta(succ, nextAgent, nextDepth, alpha, beta))
                    # prune
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            # Ghost (min node)
            else:
                value = float('inf')
                for action in actions:
                    succ = state.generateNextState(agentIndex, action)
                    value = min(value, alphabeta(succ, nextAgent, nextDepth, alpha, beta))
                    # prune
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
        # Εκκίνηση του alpha-beta για κάθε κίνηση Pacman
        alpha = float('-inf')
        beta = float('inf')
        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in gameState.getAvailableActions(0):
            succ = gameState.generateNextState(0, action)
            score = alphabeta(succ, 1, 0, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            # Ενημερώνεται ο alpha
            alpha = max(alpha, bestScore)
        return bestAction
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        numAgents = gameState.getNumAgents()
        def expectimax(state, agentIndex, depth):
            # Τερματικές καταστάσεις
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # Επόμενος agent και βάθος
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            actions = state.getAvailableActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            # Pacman: max node
            if agentIndex == 0:
                bestVal = float('-inf')
                for action in actions:
                    succ = state.generateNextState(agentIndex, action)
                    val = expectimax(succ, nextAgent, nextDepth)
                    bestVal = max(bestVal, val)
                return bestVal
            # Κόμβος τύχης φαντάσματος:υπολογίζεται ο μέσος όρος των ενεργειών
            else:
                total = 0
                for action in actions:
                    succ = state.generateNextState(agentIndex, action)
                    total += expectimax(succ, nextAgent, nextDepth)
                return total / len(actions)
        # Εκκίνηση του expectimax για Pacman (όπου agentIndex=0)
        bestScore = float('-inf')
        bestAction = Directions.STOP
        for action in gameState.getAvailableActions(0):
            succ = gameState.generateNextState(0, action)
            score = expectimax(succ, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()
    # Ποινή για τον αριθμό των υπολειπόμενων τροφίμων
    foodCount = len(foodList)
    foodCountScore = -20 * foodCount
    # Επιβράβευση εγγύτητας στο πλησιέστερο τρόφιμο
    if foodList:
        dFood = min(manhattanDistance(pos, food) for food in foodList)
        foodProximityScore = 10.0 / (dFood + 1)
    else:
        foodProximityScore = 0
    # Χαρακτηριστικά φαντασμάτων
    ghostScore = 0
    for gs in ghostStates:
        dist = manhattanDistance(pos, gs.getPosition())
        # Καταδίωξη τρομαγμένων φαντασμάτων
        if gs.scaredTimer > 0:
            ghostScore += 20.0 / (dist + 1)
        # Αποφυγή ενεργών φαντασμάτων
        else:
            if dist < 2:
                ghostScore -= 500
            else:
                ghostScore -= 5.0 / (dist)
    # Χαρακτηριστικά καψουλών
    capScore = 0
    capCount = len(capsules)
    if capsules:
        dCap = min(manhattanDistance(pos, c) for c in capsules)
        capScore += 10.0 / (dCap + 1)
    # Ποινή υπολοίπων καψουλών
    capScore -= 20 * capCount
    return score + foodCountScore + foodProximityScore + ghostScore + capScore

# Abbreviation
better = betterEvaluationFunction
