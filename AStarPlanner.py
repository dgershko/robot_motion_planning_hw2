import numpy as np
import heapq
from MapEnvironment import MapEnvironment
from shapely.geometry import LineString
from time import perf_counter

class AStarPlanner(object):    
    def __init__(self, planning_env, epsilon=1):
        self.planning_env = planning_env # type: MapEnvironment
        # self.epsilon = epsilon
        self.epsilon = 1

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = set() 
        self.valid_nodes = set()
        self.invalid_nodes = set()

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        start_time = perf_counter()
        plan = []
        start_position = tuple(self.planning_env.start)
        goal_position = tuple(self.planning_env.goal)
        frontier = Frontier()
        came_from = {}
        g_score = {start_position: 0}
        f_score = {start_position: self.heuristic(start_position)}
        frontier.insert(start_position, f_score[start_position])

        while frontier:
            current = frontier.pop()
            if current == goal_position:
                while current != start_position:
                    plan.insert(0, current)
                    current = came_from[current]
                plan.insert(0, current)
                print('Total cost of path: {:.2f}'.format(LineString(plan).length))
                print('Total time: {:.2f}'.format(perf_counter()-start_time))
                return np.array(plan)
            
            self.expanded_nodes.add(current)
            for neighbor in self.get_neighbors(current):
                tentative_score = g_score[current] + np.linalg.norm(np.array(current) - np.array(neighbor))
                if tentative_score < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_score
                    f_score[neighbor] = tentative_score + self.epsilon * self.heuristic(neighbor)
                    if neighbor not in frontier:
                        frontier.insert(neighbor, f_score[neighbor])
        return np.array([])

    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        '''
        return self.expanded_nodes

    def heuristic(self, pos):
        return np.linalg.norm(pos - self.planning_env.goal)

    def get_neighbors(self, pos):
        """
        Return list of valid neighbors for the given position.
        Uses a set of checked nodes to avoid checking the same node multiple times.
        This is done with a set of nodes that were tested to be valid, and a set that was tested to be in collision.
        """
        x, y = pos
        deltas = [-1, 0, 1]
        neighbors = [(x + dx, y + dy) for dx in deltas for dy in deltas]
        neighbors.remove(pos)
        valid_neighbors = []
        for neighbor in neighbors:
            if neighbor in self.valid_nodes:
                valid_neighbors.append(neighbor)
            elif neighbor in self.invalid_nodes:
                continue
            else:
                if self.planning_env.state_validity_checker(neighbor):
                    self.valid_nodes.add(neighbor)
                    valid_neighbors.append(neighbor)
                else:
                    self.invalid_nodes.add(neighbor)
        return valid_neighbors


class Frontier:
    def __init__(self):
        self.pq = []
        self.set = set()
    
    def pop(self):
        _, element = heapq.heappop(self.pq)
        self.set.remove(element)
        return element

    def insert(self, element, priority):
        heapq.heappush(self.pq, (priority, element))
        self.set.add(element)

    def __contains__(self, element):
        return element in self.set

    def __bool__(self):
        return bool(self.set)
