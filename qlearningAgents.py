from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random
import util
import math
import numpy


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - getQValue
        - getAction
        - getValue
        - getPolicy
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.gamma (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions
          for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.Q = {}


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        if not state in self.Q:
            return 0.0
        else:
            if not action in self.Q[state]:
                return 0.0
            else:
                return self.Q[state][action]


    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        max_val = None
        max_act = None
        for action in self.getLegalActions(state):
            if action == Directions.STOP:
                continue
            val = self.getQValue(state, action)
            if val is None or val > max_val:
                max_val = val
                max_act = action

        if max_val is None:
            return 0.0, None
        else:
            return max_val, max_act

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        val, action = self.getValue(state)
        return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        if util.flipCoin(self.epsilon):
            return random.choice(self.getLegalActions(state))
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        next_max_val, next_max_act = self.getValue(nextState)
        increment_q = reward + self.gamma * next_max_val

        if not state in self.Q:
            self.Q[state] = {}
        if not action in self.Q[state]:
            self.Q[state][action] = increment_q
        else:
            self.Q[state][action] *= 1 - self.alpha
            self.Q[state][action] += self.alpha*increment_q


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    # def simplify_state(self, state):
    #   agents = state.data.agentStates
    #   pacman = agents[0].configuration.getPosition()
    #   ghost = None
    #   if len(agents) > 1:
    #       ghost = agents[1].configuration.getPosition()
    #   proximity = "N"
    #
    #   if ghost != None:
    #       diffx = pacman[0] - ghost[0]
    #       diffy = pacman[1] - ghost[1]
    #       if (diffx == 0 and diffy == -1):
    #           proximity = "U"
    #       elif(diffx == 0 and diffy == 1):
    #           proximity = "D"
    #       elif (diffy == 0 and diffx == -1):
    #           proximity = "R"
    #       elif(diffy == 0 and diffx == 1):
    #           proximity = "L"
    #
    #   return pacman, proximity, str(state.data.food)

    def biais(self, state):
        return 1.0

    def ghost_east(self, state):
        agents = state.data.agentStates
        pacman_pos = agents[0].configuration.getPosition()
        for i in range(1, len(agents)):
            ghost_pos = agents[i].configuration.getPosition()
            diff_x = ghost_pos[0] - pacman_pos[0]
            diff_y = ghost_pos[1] - pacman_pos[1]
            if diff_x == 0 and diff_y == 1:
                return diff_y
        return 0.0

    def ghost_west(self, state):
        agents = state.data.agentStates
        pacman_pos = agents[0].configuration.getPosition()
        for i in range(1, len(agents)):
            ghost_pos = agents[i].configuration.getPosition()
            diff_x = ghost_pos[0] - pacman_pos[0]
            diff_y = ghost_pos[1] - pacman_pos[1]
            if diff_x == 0 and diff_y == -1:
                return 1.0
        return 0.0

    def ghost_north(self, state):
        agents = state.data.agentStates
        pacman_pos = agents[0].configuration.getPosition()
        for i in range(1, len(agents)):
            ghost_pos = agents[i].configuration.getPosition()
            diff_x = ghost_pos[0] - pacman_pos[0]
            diff_y = ghost_pos[1] - pacman_pos[1]
            if diff_y == 0 and diff_x == 1:
                return 1.0
        return 0.0

    def ghost_south(self, state):
        agents = state.data.agentStates
        pacman_pos = agents[0].configuration.getPosition()
        for i in range(1, len(agents)):
            ghost_pos = agents[i].configuration.getPosition()
            diff_x = ghost_pos[0] - pacman_pos[0]
            diff_y = ghost_pos[1] - pacman_pos[1]
            if diff_y == 0 and diff_x == -1:
                return 1.0
        return 0.0

    def no_wall_east(self, state):
        agents = state.data.agentStates
        walls = state.getWalls()
        x, y = agents[0].configuration.getPosition()
        if walls[x+1][y]:
            return 0.0
        return 1.0

    def no_wall_west(self, state):
        agents = state.data.agentStates
        walls = state.getWalls()
        x, y = agents[0].configuration.getPosition()
        if walls[x-1][y]:
            return 0.0
        return 1.0

    def no_wall_north(self, state):
        agents = state.data.agentStates
        walls = state.getWalls()
        x, y = agents[0].configuration.getPosition()
        if walls[x][y+1]:
            return 0.0
        return 1.0

    def no_wall_south(self, state):
        agents = state.data.agentStates
        walls = state.getWalls()
        x, y = agents[0].configuration.getPosition()
        if walls[x][y-1]:
            return 0.0
        return 1.0

    def closestFood(self, state):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        pos = state.data.agentStates[0].configuration.getPosition()
        food = state.getFood()
        walls = state.getWalls()
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                self.closest_food = (pos_x, pos_y)
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        self.closest_food = None
        return None

    def close_dot_west(self, state):
        if self.closest_food is None:
            return 0.0
        agents = state.data.agentStates
        x, y = agents[0].configuration.getPosition()
        if self.closest_food[0] < x:
            return 1/float(x - self.closest_food[0])
        return 0.0

    def close_dot_east(self, state):
        if self.closest_food is None:
            return 0.0
        agents = state.data.agentStates
        x, y = agents[0].configuration.getPosition()
        if self.closest_food[0] > x:
            return 1/float(self.closest_food[0] - x)
        return 0.0

    def close_dot_north(self, state):
        if self.closest_food is None:
            return 0.0
        agents = state.data.agentStates
        x, y = agents[0].configuration.getPosition()
        if self.closest_food[1] > y:
            return 1/float(self.closest_food[1] - y)
        return 0.0

    def close_dot_south(self, state):
        if self.closest_food is None:
            return 0.0
        agents = state.data.agentStates
        x, y = agents[0].configuration.getPosition()
        if self.closest_food[1] < y:
            return 1/float(y - self.closest_food[1])
        return 0.0

    def posx(self, state):
        agents = state.data.agentStates
        x, y = agents[0].configuration.getPosition()
        return x / float(state.data.layout.width)

    def posy(self, state):
        agents = state.data.agentStates
        x, y = agents[0].configuration.getPosition()
        return y / float(state.data.layout.height)

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        #self.epsilon = 0.05
        #self.gamma = 0.8
        #self.alpha = 0.2

        # You might want to initialize weights here.
        self.all_directions = [Directions.NORTH,
                               Directions.SOUTH,
                               Directions.EAST,
                               Directions.WEST,
                               Directions.STOP]
        self.nb_states = 0
        self.closest_food = None
        self.featureQ = [self.ghost_west, self.ghost_east, self.ghost_north, self.ghost_south,
                         self.no_wall_west, self.no_wall_east, self.no_wall_north, self.no_wall_south,
                         #self.posx, self.posy, self.biais,
                         self.close_dot_west, self.close_dot_east, self.close_dot_north, self.close_dot_south]
        self.w = {}
        for action in self.all_directions:
            temp = []
            for i in range(0, len(self.featureQ)):
                temp.append(1.0)
            self.w[action] = temp


    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        result = 0
        for i in range(0, len(self.featureQ)):
            result += self.w[action][i] * self.featureQ[i](state)
        return result

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        self.closestFood(state)

        next_max_val, next_max_act = self.getValue(nextState)
        max_val, max_act = self.getValue(state)
        increment_q = reward + self.gamma * next_max_val

        for i in range(0, len(self.featureQ)):
            #self.w[action][i] *= 1 - self.alpha
            self.w[action][i] += self.alpha * (increment_q - max_val) * self.featureQ[i](state)

            # if not simple_state in self.Q:
            #     self.Q[simple_state] = {}
            # if not action in self.Q[simple_state]:
            #     self.Q[simple_state][action] = increment_q
            # else:
            #     self.Q[simple_state][action] *= 1 - self.alpha
            #     self.Q[simple_state][action] += self.alpha*increment_q

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print "w: ", self.w


class PerceptronQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        numpy.set_printoptions(precision=2, suppress=True)

        # You might want to initialize weights here.
        self.all_directions = [Directions.NORTH,
                               Directions.SOUTH,
                               Directions.EAST,
                               Directions.WEST,
                               Directions.STOP]
        self.alpha = 0.01
        self.areaRad = 5
        self.biais = 1.0
        self.features = ['food', 'ghost', 'wall']
        self.out_of_map = {'ghost': 0, 'wall': 0, 'food': 0}
        self.w = []
        for feature in self.features:
            temp = []
            for i in range(0, 2 * self.areaRad + 1):
                temp.append([])
                for j in range(0, 2 * self.areaRad + 1):
                    temp[i].append(1.0)
            self.w.append(temp)

        # Try to load the weights
        self.load_weights()

    def is_out_of_map(self, x, y, i, j, width, height):
        # out of map?
        if x + i - self.areaRad - 1 < 0:
            return True
        elif x + i - self.areaRad - 1 >= width:
            return True
        elif y + j - self.areaRad - 1 < 0:
            return True
        elif y + j - self.areaRad - 1 >= height:
            return True

    def process_state(self, state, action):
        # The origin is at the bottom, to the left
        # The bottom left corner is (1, 1)
        x, y = state.getPacmanPosition()
        # Make coordinates start at (1, 1)
        x += 1
        y += 1

        # Food
        food = state.getFood()
        food_state = [[0 for col in range(2 * self.areaRad + 1)] for row in range(2 * self.areaRad + 1)]
        for i in range(0, 2 * self.areaRad + 1):
            for j in range(0, 2 * self.areaRad + 1):
                if self.is_out_of_map(x, y, i, j, state.data.layout.width, state.data.layout.height):
                    food_state[i][j] = self.out_of_map['food']
                elif food[x + i - self.areaRad - 1][y + j - self.areaRad - 1]:
                    food_state[i][j] = 1


        # Ghost OK
        agents = state.data.agentStates
        # The agent's coordinates should be converted as well
        ghost_state = [[0 for col in range(2 * self.areaRad + 1)] for row in range(2 * self.areaRad + 1)]
        for i in range(0, 2 * self.areaRad + 1):
            for j in range(0, 2 * self.areaRad + 1):
                if self.is_out_of_map(x, y, i, j, state.data.layout.width, state.data.layout.height):
                    ghost_state[i][j] = self.out_of_map['ghost']
                else:
                    for a in range(1, state.getNumAgents()):
                        if agents[a].configuration.getPosition() == \
                                (x + i - self.areaRad - 1, y + j - self.areaRad - 1):
                            ghost_state[i][j] = 1
                            break

        # Wall
        walls = state.getWalls()
        wall_state = [[0 for col in range(2 * self.areaRad + 1)] for row in range(2 * self.areaRad + 1)]
        for i in range(0, 2 * self.areaRad + 1):
            for j in range(0, 2 * self.areaRad + 1):
                if self.is_out_of_map(x, y, i, j, state.data.layout.width, state.data.layout.height):
                    wall_state[i][j] = self.out_of_map['wall']
                elif walls[x + i - self.areaRad - 1][y + j - self.areaRad - 1]:
                    wall_state[i][j] = 1

        uni_state = []
        for feature in self.features:
            if feature == 'ghost':
                uni_state.append(ghost_state)
            elif feature == 'food':
                uni_state.append(food_state)
            elif feature == 'wall':
                uni_state.append(wall_state)

        # Rotate it if needed because of the action
        for l in range(0, len(self.features)):
            layer = uni_state[l]
            if action == Directions.NORTH:
                pass
            elif action == Directions.WEST:
                uni_state[l] = zip(*layer[::-1])
            elif action == Directions.SOUTH:
                layer2 = zip(*layer[::-1])
                uni_state[l] = zip(*layer2[::-1])
            elif action == Directions.EAST:
                layer2 = zip(*layer[::-1])
                layer = zip(*layer2[::-1])
                uni_state[l] = zip(*layer[::-1])
        return uni_state

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        #if action == Directions.STOP:
        #   return -sys.maxint
        uni_state = self.process_state(state, action)
        result = self.biais
        for i in range(0, len(self.features)):
            for j in range(0, 2 * self.areaRad + 1):
                for k in range(0, 2 * self.areaRad + 1):
                    result += self.w[i][j][k] * uni_state[i][j][k]
        return result / float((2 * self.areaRad + 1) * (2 * self.areaRad + 1) * len(self.features) + 1)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        next_max_val, next_max_act = self.getValue(nextState)
        max_val = self.getQValue(state, action)
        difference = reward + self.gamma * next_max_val - max_val
        #print reward, " + ", self.gamma * next_max_val, " - ", max_val, " = ", difference
        uni_state = self.process_state(state, action)

        #Update biais
        #self.biais *= 1 - self.alpha
        self.biais += self.alpha * difference

        # Update weights
        for i in range(0, len(self.features)):
            for j in range(0, 2 * self.areaRad + 1):
                for k in range(0, 2 * self.areaRad + 1):
                    #self.w[i][j][k] *= 1 - self.alpha
                    self.w[i][j][k] += self.alpha * difference * uni_state[i][j][k]

        # Apply weight constraints
        #for i in range(0, len(self.features)):
        #    sum_feature = math.fabs(sum(map(sum, self.w[i])))
        #    self.w[i] = map(lambda row: map(lambda weight: weight / sum_feature, row), self.w[i])

    def save_weights(self):
        for i in range(0, len(self.features)):
            f = open("weights" + str(self.areaRad) + "." + str(self.features[i]), "w")
            f.write("# weights\n")
            numpy.savetxt(f, numpy.array(self.w[i]).T)
        f = open("weights" + str(self.areaRad) + ".biais", "w")
        f.write("# weights\n")
        f.write(str(self.biais))

    def load_weights(self):
        print "Loading weights from:"
        for i in range(0, len(self.features)):
            file_name = "weights" + str(self.areaRad) + "." + str(self.features[i])
            if not os.path.exists(file_name):
                break
            print file_name
            self.w[i] = numpy.loadtxt(file_name, unpack=True)
        file_name = "weights" + str(self.areaRad) + ".biais"
        if not os.path.exists(file_name):
            return
        print file_name
        self.biais = numpy.loadtxt(file_name, unpack=True)
        print "biais: ", self.biais

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            self.save_weights()
            print "biais ", self.biais
            for i in range(0, len(self.features)):
                print "\n"
                print self.features[i]
                print (numpy.array(self.w[i]))
