import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    LearningAgentState = namedtuple('LearningAgentState', ['next_waypoint', 'light', 'oncoming', 'right', 'left', 'deadline'])

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q_values = {}
        self.last_action = None


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        present_state = self.LearningAgentState(next_waypoint=self.next_waypoint, light=inputs['light'], oncoming=inputs['oncoming'], right=inputs['right'], left=inputs['left'], deadline=deadline)

        self.state = present_state

        old_q, action = self.choose_action(present_state)

        self.last_action = action
        # Execute action and get reward
        reward = self.env.act(self, action)

        # Sense new state
        new_inputs = self.env.sense(self)

        new_state = self.LearningAgentState(next_waypoint=self.next_waypoint, light=new_inputs['light'], oncoming=new_inputs['oncoming'], right=new_inputs['right'], left=new_inputs['left'], deadline=deadline)

        new_Q = self.learned_val(reward, new_state, old_q)

        self.Q_values[(present_state, action)] = new_Q

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, oldQ = {}, newQ = {},".format(deadline, inputs, action, reward, old_q, new_Q)  # [debug]

    def learned_val(self, reward, new_state, old_q):
        learning_rate = 0.4
        discount_factor = 0.25

        new_q = old_q + (learning_rate * (reward + (discount_factor * self.maxQ(new_state) - old_q)))

        return new_q

    def maxQ(self, state):
        q = [self.getQ(state, a) for a in Environment.valid_actions]
        maxQ = max(q)

        return maxQ

    def choose_action(self, state):
        q = [self.getQ(state, a) for a in Environment.valid_actions]
        maxQ = max(q)

        if random.random() < 1:
            best = [i for i in range(len(Environment.valid_actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = Environment.valid_actions[i]

        return maxQ, action

    def getQ(self, state, a):
        return self.Q_values.get((state, a), 1)

class TrainedAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    LearningAgentState = namedtuple('LearningAgentState', ['next_waypoint', 'light', 'oncoming', 'right', 'left', 'deadline'])

    def __init__(self, env, policy):
        super(TrainedAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.policy = policy
        self.last_action = None
        self.penalties = 0
        self.actionsAvail = 0
        self.actionsTaken = 0


    def reset(self, destination=None):
        self.planner.route_to(destination)

        print "penalties incurred = {}".format(self.penalties)
        print 'actions available = {}'.format(self.actionsAvail)
        print 'actions taken = {}'.format(self.actionsTaken)

        self.actionsAvail += self.env.get_deadline(self)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)


        S = self.LearningAgentState(next_waypoint=self.next_waypoint, light=inputs['light'], oncoming=inputs['oncoming'], right=inputs['right'], left=inputs['left'], deadline=deadline)

        self.state = S

        old_q, action = self.choose_action(S)

        self.last_action = action

        # Execute action and get reward
        reward = self.env.act(self, action)

        if reward < 0:
            self.penalties += 1

        self.actionsTaken += 1


    def choose_action(self, state):
        q = [self.getQ(state, a) for a in Environment.valid_actions]
        maxQ = max(q)

        i = q.index(maxQ)

        action = Environment.valid_actions[i]

        return maxQ, action

    def getQ(self, state, a):
        if (state, a) in self.policy:
            return self.policy[(state, a)]
        else:
            return 1

def run():
    """Run the agent for a finite number of trials."""


    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

    policy = a.Q_values

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(TrainedAgent, policy)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=25)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
