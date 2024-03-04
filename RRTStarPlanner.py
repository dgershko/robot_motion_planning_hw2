import numpy as np
from Tree import RRTree
from MapEnvironment import MapEnvironment
from shapely.geometry import Point, LineString, Polygon
import time

class RRTStarPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob, k, time_limit = None):

        # set environment and search tree
        self.planning_env = planning_env # type: MapEnvironment
        self.tree = RRTree(tuple(self.planning_env.start))

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.step_size = 10
        self.k = k
        if self.k == -1:
            self.k = None
            print("setting k to logn mode")
        self.time_limit = time_limit
        if time_limit is None:
            self.time_limit = np.inf
        

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        start_time = time.perf_counter()
        goal_state = tuple(self.planning_env.goal)

        # initialize an empty plan.
        plan = []
        while time.perf_counter() - start_time < self.time_limit:
            rand_state = self.get_random_sample()
            near_state = self.tree.get_nearest_state(rand_state)
            new_state = self.extend(near_state, rand_state)
            if self.planning_env.edge_validity_checker(near_state, new_state):
                self.tree.insert_state(new_state, near_state)

                if self.k is None:
                    near_states = self.tree.get_knn_states(new_state, int(np.log(len(self.tree.vertices))))
                else:
                    near_states = self.tree.get_knn_states(new_state, self.k)
                for state in near_states:
                    self.rewire(state, new_state)
                for state in near_states:
                    self.rewire(new_state, state)
                
                if np.array_equal(new_state, goal_state):
                    if self.time_limit == np.inf:
                        break

        if goal_state not in self.tree.vertices.keys():
            return None
        plan, cost = self.tree.path_to_state(goal_state)
        
        # print total path cost and time
        print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
        print('Total time: {:.2f}'.format(time.perf_counter()-start_time))

        return np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps.
        @param plan A given plan for the robot.
        '''
        return LineString(plan).length


    def rewire(self, potential_parent, child):
        if self.planning_env.edge_validity_checker(potential_parent, child):
            cost = np.linalg.norm(np.array(potential_parent) - np.array(child))
            if self.tree.cost_to_state(potential_parent) + cost < self.tree.cost_to_state(child):
                self.tree.set_parent_for_state(child, potential_parent)


    def extend(self, near_state, rand_state):
        '''
        Compute and return a new position for the sampled one.
        @param near_state The nearest position to the sampled position.
        @param rand_state The sampled position.
        '''
        if self.ext_mode == 'E1':
            return rand_state
        elif self.ext_mode == 'E2':
            direction = np.array(rand_state) - np.array(near_state)
            distance = np.linalg.norm(direction)
            if distance <= self.step_size:
                return rand_state
            unit_direction_vector = direction / distance
            return tuple(near_state + unit_direction_vector * self.step_size)


    def get_random_sample(self):
        if np.random.uniform(0, 1) <= self.goal_prob:
            return tuple(self.planning_env.goal)
        x = np.random.uniform(*self.planning_env.xlimit)
        y = np.random.uniform(*self.planning_env.ylimit)
        while not self.planning_env.state_validity_checker((x, y)):
            x = np.random.uniform(*self.planning_env.xlimit)
            y = np.random.uniform(*self.planning_env.ylimit)
        return (x, y)
