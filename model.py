import numpy as np
import numpy.random as npr


class MallowsPermutation(object):

    def __init__(self, n, q, seed=None):
        '''
        Create a random Mallows permutation with parameters 'n' and 'q'.
        '''
        self.n = n
        self.q = q
        self.seed = seed
        npr.seed(self.seed)

        self.params = 'n={};q={}'.format(self.n,self.q)

        self.__permutation__()

    def __permutation__(self):
        '''
        Generates the permutation.
        It first creates a vector 'G' such that 0 <= G[i] <= n-1-i and G[i] ~ Geometric(q).
        This vector is then used to create the permutation by choosing the image of i as the G[i]-th available value.
        '''
        if self.q == 0:
            self.permutation = np.arange(self.n)
        elif self.q_is_inf():
            self.permutation = np.arange(self.n-1, -1, -1)

        else:
            U = npr.rand(self.n)
            N = np.arange(self.n, 0, -1)

            if self.q == 1:
                G = np.floor(N*U).astype(int)
            else:
                G = np.floor(np.log(1 - U*(1 - self.q**N))/np.log(self.q)).astype(int)

            available_values = np.arange(self.n)
            self.permutation = np.zeros(self.n)
            for i in range(self.n):
                self.permutation[i] = available_values[G[i]]
                available_values = np.delete(available_values, G[i])

            self.permutation = np.argsort(self.permutation).astype(int)

    def __getitem__(self, key):
        '''
        Calling 'MallowsPermutation()[i]' returns the image of i through the permutation.
        '''
        return self.permutation[key]

    def __iter__(self):
        '''
        Iterating through 'MallowsPermutation()' goes through the images of 0, 1, 2, ...
        '''
        return iter(self.permutation)

    def q_is_inf(self):
        '''
        Checks if 'q' is infinity.
        '''
        return self.q > max(self.n**2, 1e10)


class MallowsTree(MallowsPermutation):

    def __init__(self, n, q, seed=None):
        '''
        Creates a random Mallows tree from a Mallows permutation.
        '''
        super().__init__(n, q, seed)
        self.__tree__()

    def __tree__(self):
        '''
        This function creates the tree corresponding to the permutation.
        It explores the entries of the permutation and insert them one after the other in the tree.
        The parameter 'self.int_to_node' is a dictionary mapping the entries to their nodes.
        The parameter 'self.node_to_int' is a dictionary mapping the nodes to their entries.
        The parameter 'self.E' is a list of edges of the tree.
        '''
        self.int_to_node = {}
        self.node_to_int = {}
        self.E = []

        for v in self:
            node_v = ''
            parent = -1
            while node_v in self.node_to_int:
                parent = self.node_to_int[node_v]
                if v > parent:
                    node_v += '1'
                else:
                    node_v += '0'
            self.int_to_node[v] = node_v
            self.node_to_int[node_v] = v
            if parent >= 0:
                self.E.append((parent, v))

    def __call__(self, key):
        '''
        Calling 'MallowsTree()(i)' returns the node of i.
        '''
        return self.int_to_node[key]

    def height(self):
        '''
        Computes the height of the tree.
        '''
        return max([len(m) for m in self.int_to_node.values()])

    def depth(self, v):
        '''
        Computes the depth of 'v'.
        '''
        return len(self.int_to_node[v])

    def parent(self, v):
        '''
        Finds the parent of 'v' in the tree.
        '''
        node = self.int_to_node[v]
        if node:
            return self.node_to_int[node[:-1]]
        else:
            raise Exception('The root has no parent')

    def path_from_root(self, v, edge_list=False):
        '''
        Computes the path from the root to the node 'v'.
        Either as a list of nodes, or as a list of edges.
        '''
        path = [self.node_to_int['']]
        node_v = self.int_to_node[v]
        for i in range(len(node_v)):
            path.append(self.node_to_int[node_v[:i+1]])

        if edge_list:
            edge_path = []
            if len(path) > 1:
                u = path[0]
                for v in path[1:]:
                    edge_path.append((u,v))
                    u = v
            return edge_path

        else:
            return path

    @staticmethod
    def c_star():
        '''
        Computes c*, solution to clog(2e/c)=1 with c>=2.
        This corresponds to the ratio height of random binary search tree over log(n).
        '''
        return 4.311070407001