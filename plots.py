import os
import os.path as osp
from shutil import rmtree

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from PIL import Image

from model import MallowsTree


COLOURS = {
    'tree': 'darkgreen',
    'node': 'gold',
    'edge': 'forestgreen',
    'move': 'crimson',
}
TIMES = {
    'basis': 100,
    'tree': 200,
    'final': 3000,
}


class TreePlots(object):

    def __init__(self, images_dir='images', gifs_dir='gifs', colours=COLOURS, times=TIMES, max_speed=0.5):
        '''
        Initialize the 'TreePlots' instance.
        Creates two directories, to save images and gifs.
        Sets the colour theme of the figures and the time fo the frames for the gifs.
        Set the maximal moving speed of the tree for the gifs.
        '''
        self.images_dir = images_dir
        if not osp.exists(self.images_dir):
            os.makedirs(self.images_dir)
        self.gifs_dir = gifs_dir
        if not osp.exists(self.gifs_dir):
            os.makedirs(self.gifs_dir)

        self.colours = colours
        self.times = times
        self.max_speed = max_speed

    @staticmethod
    def compute_pos(MT):
        '''
        Given a tree, outputs the position of the nodes.
        '''
        pos = np.zeros((MT.n, 2))
        id_n = np.arange(MT.n)
        pos[:,0] = id_n
        pos[:,1] = -np.array([MT.depth(v) for v in id_n])

        return pos

    @staticmethod
    def _normalize_pos(POS, V):
        '''
        Normalize the input positions to fit the nodes of 'V' into [-1,1]^2.
        '''
        pos = POS.copy()
        pos[V,0] = np.argsort(np.argsort(pos[V,0]))

        max_X = np.max(pos[V,0])
        min_X = np.min(pos[V,0])
        max_Y = np.max(pos[V,1])
        min_Y = np.min(pos[V,1])

        r = max(len(V), 1)
        rX = max_X - min_X + 3/r
        rY = max_Y - min_Y + 3/r
        rY = max(rY, MallowsTree.c_star()*np.log(rX))

        pos[:,0] = (2*pos[:,0] - max_X - min_X)/rX
        pos[:,1] = (2*pos[:,1] - max_Y - min_Y)/rY

        return pos

    def tree(self,
             MT,
             pos=None,
             r=None,
             q=None,
             V_plot=None,
             V_move=[],
             E_plot=None,
             E_move=[],
             move=None,
             plot_numbers=False,
             plot_q=False,
             name='tree'):
        '''
        Plots the tree 'MT' with a lot of customizing arguments:
        - 'pos' for the positions of the nodes.
        - 'r' for the inverse radius of the nodes and edges.
        - 'q' for the represented value of q.
        - 'V_plot' for the nodes to be plotted.
        - 'V_move' for the moving node(s).
        - 'E_plot' for the edges to be plotted.
        - 'E_move' for the moving edge(s).
        - 'move' for the value of the moving node.
        - 'plot_numbers' if the numbers of the permutation have to be plotted.
        - 'plot_q' if the value 'q' has to be plotted.
        - 'name' for the name of the image.
        '''

        # Initialization of the variables
        if E_plot is None:
            E_plot = MT.E
        if V_plot is None:
            V_plot = list(MT)
        if pos is None:
            pos = self.compute_pos(MT)
            pos = self._normalize_pos(pos, V_plot)
        if r is None:
            r = max(len(V_plot), 1)
        if q is None:
            plot_q = False

        E_plot = [(u,v) for (u,v) in E_plot if (u in V_plot) & (v in V_plot)]
        V_move = [v for v in V_move if v in V_plot]

        tree_colour = self.colours['tree']
        node_colour = self.colours['node']
        edge_colour = self.colours['edge']
        move_colour = self.colours['move']

        X = pos[:,0]
        Y = pos[:,1]

        length = 1.1
        if plot_numbers or plot_q:
            length += 0.1

        # Start plotting the figure
        plt.figure(figsize=(9,9))
        plt.axis(xmin=-length,xmax=length,ymin=-length,ymax=length)

        # PLotting the edges
        for u,v in MT.E:
            if (u,v) in E_plot:
                plt.plot([X[u],X[v]], [Y[u],Y[v]], tree_colour, linewidth=min(60,300/r))
        for u,v in MT.E:
            if (u,v) in E_plot:
                if (u,v) in E_move:
                    plt.plot([X[u],X[v]], [Y[u],Y[v]], move_colour, linewidth=min(40,200/r))
                else:
                    plt.plot([X[u],X[v]], [Y[u],Y[v]], edge_colour, linewidth=min(20,100/r))

        # PLotting the nodes
        plt.plot(X[V_plot], Y[V_plot], color=tree_colour, marker='.', markersize=min(240,1200/r), linewidth=0, zorder=1)
        plt.plot(X[V_plot], Y[V_plot], color=node_colour, marker='.', markersize=min(140,700/r), linewidth=0, zorder=2)
        plt.plot(X[V_move], Y[V_move], color=move_colour, marker='.', markersize=min(210,1050/r), linewidth=0, zorder=4)

        # Representation of the node values and corresponding permutation
        if plot_numbers:
            # PLots the number on the nodes
            for v in V_plot:
                if v < 9:
                    plt.text(
                        X[v] - min(0.065, 0.325/r),
                        Y[v] - min(0.065, 0.325/r),
                        v+1,
                        fontsize=min(50, 250/r),
                        fontweight='demibold',
                        zorder=3
                    )
                else:
                    plt.text(
                        X[v] - min(0.13, 0.65/r),
                        Y[v] - min(0.065, 0.325/r),
                        v+1,
                        fontsize=min(50, 250/r),
                        fontweight='demibold',
                        zorder=3
                    )
            # Plots the number on the moving node(s)
            if move is not None:
                for v in V_move:
                    if move < 9:
                        plt.text(
                            X[v] - min(0.09, 0.45/r),
                            Y[v] - min(0.09, 0.45/r),
                            move+1,
                            fontsize=min(70, 350/r),
                            fontweight='demibold',
                            zorder=5
                        )
                    else:
                        plt.text(
                            X[v] - min(0.18, 0.9/r),
                            Y[v] - min(0.09, 0.45/r),
                            move+1,
                            fontsize=min(70, 350/r),
                            fontweight='demibold',
                            zorder=5
                        )

            # Plots the current state of the permutation
            text = '\u03C3 = ('
            for v in V_plot:
                text += '{}, '.format(v+1)
            if move is not None:
                if move not in V_plot:
                    text += '{}, '.format(move+1)
            text = text[:-2]
            if len(text.split(',')) == MT.n:
                text += ')'
            else:
                text += ','
            plt.text(-1.15, -1.15, text, fontsize=15, fontweight='demibold')
            # Adds a title to the figure
            title = 'Mallows tree with {}'.format(MT.params.replace(';', ' and '))
            plt.text(-1.15, 1.1, title, fontsize=20, fontweight='demibold')

        # Representation of the value q of the permutation
        elif plot_q:
            DOT_COLOUR = 'black'
            LINE_COLOUR = 'grey'

            if q <= 1:
                q_x = -1 + q**2
            else:
                q_x = 1 - 1/q**2
            q = '{:.2f}'.format(q)

            # Plots an axis for the value of q
            plt.plot([-1,1], [-1.13]*2, LINE_COLOUR, linewidth=2)
            plt.plot([-1]*2, [-1.12,-1.14], LINE_COLOUR, linewidth=2)
            plt.plot([0]*2, [-1.12,-1.14], LINE_COLOUR, linewidth=2)
            plt.plot([1]*2, [-1.12,-1.14], LINE_COLOUR, linewidth=2)
            plt.text(-1.01, -1.19, 0, color=LINE_COLOUR, fontsize=10, fontweight='demibold')
            plt.text(-0.01, -1.19, 1, color=LINE_COLOUR, fontsize=10, fontweight='demibold')
            plt.text(0.98, -1.19, 'inf', color=LINE_COLOUR, fontsize=10, fontweight='demibold')
            # Puts q at the corresponding position on the axis
            plt.text(-1.15, -1.15, 'q = ', color=DOT_COLOUR, fontsize=15, fontweight='demibold')
            plt.plot(q_x, -1.13, DOT_COLOUR, marker='.', markersize=20)
            plt.text(q_x+0.01, -1.1, q, color=DOT_COLOUR, fontsize=10, fontweight='demibold')
            # Adds a title to the figure
            plt.text(-1.15, 1.1, 'Mallows trees with n={}'.format(MT.n), fontsize=20, fontweight='demibold')

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.savefig(osp.join(self.images_dir, name + '.png'))
        plt.close()

    def _construction_gif(self, MT, frames_dir='frames', plot_numbers=False):
        '''
        Saves the frames for the construction gif.
        '''
        if osp.exists(osp.join(self.images_dir, frames_dir)):
            rmtree(osp.join(self.images_dir, frames_dir))
        os.makedirs(osp.join(self.images_dir, frames_dir))

        time_basis = self.times['basis']
        time_final = self.times['final']

        def plot(pos,
                 r,
                 index,
                 duration=time_basis,
                 V_plot=None,
                 V_move=[],
                 E_move=[],
                 move=None):
            '''
            This function is used to simplify repetitions in the code.
            '''
            self.tree(MT=MT,
                      pos=pos,
                      r=r,
                      V_plot=V_plot,
                      V_move=V_move,
                      E_move=E_move,
                      move=move,
                      plot_numbers=plot_numbers,
                      name=osp.join(frames_dir, '{}-{}'.format(index, duration)))

            return index + 1

        index = 0
        V = [MT[0]]
        r = 1
        POS = self.compute_pos(MT)
        pos = self._normalize_pos(POS, V)

        index = plot(pos, r, index, V_plot=V)

        for v in MT[1:]:
            edge_path = MT.path_from_root(v, edge_list=True)
            step = 1/(3*MT.depth(v) + 2)
            parent = MT.parent(v)

            #Npos = next position
            Npos = self._normalize_pos(POS, V + [v])
            diff_pos = Npos - pos

            for x,y in edge_path:
                pos += step*diff_pos
                r += step
                index = plot(pos, r, index, V_plot=V, V_move=[x], move=v)

                if y == v: # when we reach the end of the path
                    V.append(v)
                    pos[v,:] = pos[parent,:]
                    diff_pos[v,:] = (Npos[v,:] - pos[v,:])/(4*step)
                    x = y

                pos += step*diff_pos
                r += step
                index = plot(pos, r, index, V_plot=V, V_move=[x], E_move=[(x,y)], move=v)

                pos += step*diff_pos
                r += step
                index = plot(pos, r, index, V_plot=V, V_move=[y], E_move=[(x,y)], move=v)

            pos += step*diff_pos
            r += step
            index = plot(pos, r, index, V_plot=V, V_move=[v], move=v)

            pos += step*diff_pos
            r += step
            index = plot(pos, r, index, V_plot=V)
        
        index = plot(pos, r, index, duration=time_final, V_plot=V)

    def _evolution_gif(self, n, q_step=0.1, seed=None, frames_dir='frames', plot_q=False):
        '''
        Saves the frames for the evolution gif.
        '''
        if seed is None:
            seed = npr.randint(1e6)

        if osp.exists(osp.join(self.images_dir, frames_dir)):
            rmtree(osp.join(self.images_dir, frames_dir))
        os.makedirs(osp.join(self.images_dir, frames_dir))

        time_basis = self.times['basis']
        time_tree = self.times['tree']

        def plot(MT,
                 pos,
                 index,
                 q,
                 duration=time_basis,
                 E_plot=None):
            '''
            This function is used to simplify repetitions in the code.
            '''
            self.tree(MT=MT,
                     pos=pos,
                     r=n,
                     q=q,
                     E_plot=E_plot,
                     plot_q=plot_q,
                     name=osp.join(frames_dir, '{}-{}'.format(index, duration)))

            return index + 1

        q_list = np.arange(q_step,1,q_step)**.5
        q_list = list(q_list) + [1] + list(1/q_list[::-1]) + [np.inf]
        V = list(range(n))
        index = 0
        
        q0 = 0
        MT = MallowsTree(n, q0, seed)
        pos = self.compute_pos(MT)
        pos =  self._normalize_pos(pos, V)
        index = plot(MT, pos, index, q0, duration=time_tree)

        for q in q_list:
            # NMT = next Mallows tree
            NMT = MallowsTree(n, q, seed)
            # Npos = next position
            Npos = self.compute_pos(NMT)
            Npos = self._normalize_pos(Npos, V)
            diff_pos = Npos - pos
            if NMT.q_is_inf(): # to avoid being absorbed by q=inf
                diff_q = 1/(1/q0 - 1/q)
            else:
                diff_q = q - q0

            if np.any(diff_pos):
                n_steps = int(np.ceil(np.max(diff_pos)/self.max_speed))
                step = 1/(2*(n_steps + 1.))

                for i in range(n_steps):
                    pos += step*diff_pos
                    q0 += step*diff_q
                    index = plot(MT, pos, index, q0)

                pos += step*diff_pos
                q0 += step*diff_q
                index = plot(MT, pos, index, q0, E_plot=NMT.E)

                MT = NMT
                for i in range(n_steps):
                    pos += step*diff_pos
                    q0 += step*diff_q
                    index = plot(MT, pos, index, q0)

                pos = Npos
                q0 = q
                index = plot(MT, pos, index, q0, duration=time_tree)

    def _images_to_gif(self, frames_dir='frames', name='mallows-tree', symmetric=False):
        '''
        Transforms a list of frames into a gif.
        '''
        frame_list = os.listdir(osp.join(self.images_dir, frames_dir))
        frame_list.sort(key=lambda x:float(x.split('-')[0]))

        frames = []
        duration = []        
        for frame in frame_list:
            frames.append(Image.open(osp.join(self.images_dir, frames_dir, frame)))
            duration.append(int(frame.split('-')[1].split('.')[0]))

        if symmetric:
            frames = frames + frames[::-1][1:-1]
            duration = duration + duration[::-1][1:-1]

        frames[0].save(osp.join(self.gifs_dir, name + '.gif'),
                       save_all=True,
                       append_images=frames[1:],
                       duration=duration,
                       optimize=True,
                       loop=0)

    def construction(self, MT, plot_numbers=False):
        '''
        Creates the construction gif.
        It represents the Mallows tree 'MT' by recursively inserting the values of the permutation.
        - 'plot_numbers' says if the permutation and the numbers have to be plotted.
        '''
        frames_dir = 'frames ({})'.format(MT.params)
        name = 'construction ({})'.format(MT.params)

        self._construction_gif(MT, frames_dir=frames_dir, plot_numbers=plot_numbers)
        self._images_to_gif(name=name, frames_dir=frames_dir, symmetric=False)

    def evolution(self, n, seed=None, q_step=0.1, plot_q=False):
        '''
        Creates the evolution gif.
        It represents the evolution of Mallows tree with size 'n' and changing parameters.
        - 'seed' is used for the coupling of the evolution.
        - 'q_step' correspondonds to the evolving steps of q.
        - 'plot_q' says if the evolving value of q vae to be plotted.
        '''
        frames_dir = 'frames ({})'.format(n)
        name = 'evolution ({})'.format(n)

        self._evolution_gif(n, q_step=q_step, frames_dir=frames_dir, seed=seed, plot_q=plot_q)
        self._images_to_gif(name=name, frames_dir=frames_dir, symmetric=True)