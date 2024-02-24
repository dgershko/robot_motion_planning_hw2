import numpy as np
from RRTTree import RRTTree
from MapEnvironment import MapEnvironment
from shapely.geometry import Point, LineString, Polygon
import time

class RRTPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env # type: MapEnvironment
        self.tree = RRTTree(self.planning_env)
        self.tree.add_vertex(self.planning_env.start)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.step_size = 1.5

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []
        while not self.tree.is_goal_exists(self.planning_env.goal):
            rand_state = self.get_random_sample()
            near_idx, near_state = self.tree.get_nearest_state(rand_state)
            new_state = self.extend(near_state, rand_state)
            if self.planning_env.edge_validity_checker(near_state, new_state):
                new_idx = self.tree.add_vertex(new_state)
                self.tree.add_edge(near_idx, new_idx, np.linalg.norm(np.array(near_state) - np.array(new_state)))
        

        
        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.time()-start_time))

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps.
        @param plan A given plan for the robot.
        '''
        return LineString(plan).length

    def extend(self, near_state, rand_state):
        '''
        Compute and return a new position for the sampled one.
        @param near_state The nearest position to the sampled position.
        @param rand_state The sampled position.
        '''
        if self.ext_mode == 'E1':
            return rand_state
        elif self.ext_mode == 'E2':
            direction = rand_state - near_state
            distance = np.linalg.norm(direction)
            if distance <= self.step_size:
                return rand_state
            unit_direction_vector = direction / distance
            return near_state + unit_direction_vector * self.step_size


    def get_random_sample(self):
        if np.random.uniform(0, 1) <= self.goal_prob:
            return self.planning_env.goal
        x = np.random.uniform(*self.planning_env.xlimit)
        y = np.random.uniform(*self.planning_env.ylimit)
        while not self.planning_env.state_validity_checker((x, y)):
            x = np.random.uniform(*self.planning_env.xlimit)
            y = np.random.uniform(*self.planning_env.ylimit)
        return (x, y)