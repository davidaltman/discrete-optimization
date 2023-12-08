#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
import typing
import copy

Item = namedtuple("Item", ['index', 'value', 'weight'])

def dp(capacity: int, values: list, weights: list) -> list:
    """use dynamic programming and K solution"""
    K = np.zeros((capacity+1,len(values)+1))
    density =[v/w for v,w in zip(values,weights)]
    sorted_idx = list(np.argsort(weights))
    sorted_idx = list(range(len(weights)))
    values = [values[i] for i in sorted_idx]
    weights = [weights[i] for i in sorted_idx]
    print(f'density: {density} sorted_idx: {sorted_idx}')
    print(f'values: {values} weights: {weights}')
    w = weights
    for r in range(capacity+1):
        for c in range(1,len(values)+1):
            if r >= weights[c-1]:
                K[r][c] = max(values[c-1] + K[r-w[c-1]][c-1], K[r][c-1])
                            
            else:
                K[r][c] = K[r][c-1]
    
    row = len(K)-1
    sol = [0]*(len(values))
    # backtrack through K now
    cols = len(K[0])
    done = False
    for c in range(cols-1,0,-1):
        if K[row][c] == K[row][c-1]:
            sol[c-1] = 0
        else:
            row -= weights[c-1]
            sol[c-1] = 1
    reorder_solution = [0]*len(sol)
    for i in range(len(sol)):
        reorder_solution[sorted_idx[i]] = sol[i]
    check_solution = np.sum([i*v for i,v in zip(sol,values)])
    assert(check_solution==int(K[-1][-1]))
    return int(K[-1][-1]), reorder_solution


def dfs(capacity: int, j: int, values: list, weights: list, memo: list) -> int:
    """recursive solution for maximizing value in backpack"""
    k = capacity
    v = values
    w = weights
    if j < 0:
        return 0
    elif memo[k][j] != -1:
        return memo[k][j]
    elif w[j] <= k:
        memo[k][j] = max(dfs(k,j-1,v,w,memo), v[j] + dfs(k-w[j],j-1,v,w,memo))
        return memo[k][j]
    else:
        memo[k][j] = dfs(k,j-1,v,w,memo)
        return memo[k][j]

def read_values(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []
    w = [] # weights
    v = [] # values
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
        v.append(int(parts[0]))
        w.append(int(parts[1]))
    assert(len(items) == item_count)
    return v, w, capacity

def solve_it_greedy(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    
    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)
    print(items)
    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    v, w, capacity = read_values(input_data)
    max_value, indexes = dp(capacity,v,w)
    print(f'max_value: {max_value} indexes: {indexes}')
    is_optimal = 0
    tot_weight = sum([indexes[i]*w[i] for i in range(len(w))])
    print(f'sum weights: {(tot_weight)}')
    if capacity == tot_weight:
        is_optimal = 1
    output_data = str(max_value) + ' ' + str(is_optimal) + '\n'
    output_data += ' '.join(map(str, indexes))
    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        v, w, capacity = read_values(input_data)
        print(f'capacity: {capacity} values: {v} weights: {w}')

        ans  = solve_it(input_data)
        print(ans)
        exit(0)

        memo = np.ones((capacity+1, len(v)))*-1
        total_value = dfs(capacity, len(v)-1,v,w,memo)
        max_value, indexes = dp(capacity, len(v)-1,v,w)
        print(f'max_value: {max_value} indexes: {indexes}')
        # double check my work
        assert total_value == max_value

    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

