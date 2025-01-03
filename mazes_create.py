import os
import numpy as np
import random

def generate_maze(size):
    maze = np.ones((size, size), dtype=int)
    start_pos = (0, 0)
    goal_pos = (size - 1, size - 1)
    maze[start_pos] = 0
    maze[goal_pos] = 0

    def carve_passages_from(cx, cy):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for direction in directions:
            nx, ny = cx + direction[0], cy + direction[1]
            if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == 1:
                if (nx, ny) == goal_pos or (nx, ny) == start_pos:
                    continue
                if sum(0 <= nx + dx < size and 0 <= ny + dy < size and maze[nx + dx, ny + dy] == 0 for dx, dy in directions) <= 1:
                    maze[nx, ny] = 0
                    carve_passages_from(nx, ny)

    carve_passages_from(0, 0)
    return maze

def save_mazes(num_mazes, size, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(num_mazes):
        maze = generate_maze(size)
        np.save(os.path.join(directory, f'maze_{i}.npy'), maze)

if __name__ == '__main__':
    num_mazes = 10
    size = 10
    directory = 'mazes'
    save_mazes(num_mazes, size, directory)