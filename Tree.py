import numpy as np

class RRTree():
    def __init__(self, root_state):
        self.root_state = root_state
        self.root_node = RRNode(root_state, 0, None)
        self.vertices = {root_state: self.root_node}

    def insert_state(self, state: tuple[float, float], parent_state: tuple[float, float]):
        cost = np.linalg.norm(np.array(state) - np.array(parent_state))
        parent_node = self.vertices[parent_state]
        state = tuple(state)
        self.vertices[state] = RRNode(state, cost, parent_node)
    
    def numpy_cdist(self, XA, XB):
        return np.sqrt(np.sum((XA[:, None, :] - XB[None, :, :]) ** 2, axis=-1))
    
    def get_nearest_state(self, state):
        state = np.array(state)
        vertices = list(self.vertices.keys())
        distances = self.numpy_cdist(np.array([state]), np.array(vertices))
        nearest_vertex_index = np.argmin(distances)
        return vertices[nearest_vertex_index]
        # return min(self.vertices.keys(), key=lambda x: np.linalg.norm(np.array(x) - state))

    def cost_to_state(self, state):
        return self.vertices[state].total_cost

    def get_state_parent(self, state):
        try:
            return self.vertices[state].parent.state
        except:
            return None
    
    def path_to_state(self, state):
        path = []
        cost = self.vertices[state].total_cost
        current_state = self.vertices[state]
        while current_state.parent:
            path.append(current_state.state)
            current_state = current_state.parent
        path.append(current_state.state)
        path.reverse()
        return path, cost

    def get_knn_states(self, state, k):
        states = list(self.vertices.keys())
        states.remove(state)
        if not states:
            return []
        if k >= len(states):
            return states
        state_arr = np.array(state)
        states_arr = np.array(states)
        distances = np.linalg.norm(states_arr - state_arr, axis=1)
        partitioned_states = np.array(states)[np.argpartition(distances, k)][:k] # type: list
        partitioned_states = list([tuple(state) for state in partitioned_states])
        return partitioned_states

    def get_edges_as_states(self):
        return [(self.vertices[state].state, self.vertices[state].parent.state) for state in self.vertices.keys() if self.vertices[state].parent]

    def set_parent_for_state(self, state, new_parent):
        new_parent_node = self.vertices[new_parent]
        state_node = self.vertices[state]
        state_node.parent = new_parent_node
        state_node.cost = np.linalg.norm(np.array(state) - np.array(new_parent))
    


class RRNode():
    def __init__(self, state: tuple[float, float], cost, parent_node: "RRNode | None" = None):
        self.cost = cost
        self.parent = parent_node # type: RRNode | None
        self.state = state

    @property
    def total_cost(self):
        if self.parent:
            return self.cost + self.parent.total_cost
        return 0
