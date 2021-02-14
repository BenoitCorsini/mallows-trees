import argparse

from model import MallowsTree
from plots import TreePlots


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--q', type=float, default=1)
    # The figures to be saved
    parser.add_argument('--save_tree', type=int, default=1)
    parser.add_argument('--save_construction', type=int, default=1)
    parser.add_argument('--save_evolution', type=int, default=1)
    # The information on the figures
    parser.add_argument('--plot_numbers', type=int, default=1)
    parser.add_argument('--plot_q', type=int, default=1)
    # Extra parameters
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--q_step', type=float, default=0.1)
    args = parser.parse_args()

    Plots = TreePlots(colours=COLOURS, times=TIMES)
    MT = MallowsTree(args.n, args.q)

    if args.save_tree:
        Plots.tree(MT, plot_numbers=args.plot_numbers, plot_q=args.plot_q)

    if args.save_construction:
        Plots.construction(MT, plot_numbers=args.plot_numbers)

    if args.save_evolution:
        Plots.evolution(args.n, plot_q=args.plot_q, seed=args.seed, q_step=args.q_step)