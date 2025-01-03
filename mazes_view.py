import os
import numpy as np
import matplotlib.pyplot as plt

def load_mazes(directory):
    maze_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    mazes = [np.load(maze_file, allow_pickle=True) for maze_file in maze_files]
    return mazes

def print_mazes(mazes):
    for i, maze in enumerate(mazes):
        plt.figure(figsize=(5, 5))
        plt.imshow(maze, cmap='gray_r')
        plt.title(f'Maze {i}')
        plt.show()

if __name__ == '__main__':
    directory = 'mazes'
    mazes = load_mazes(directory)
    print_mazes(mazes)