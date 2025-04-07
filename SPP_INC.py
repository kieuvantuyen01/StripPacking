import math
import fileinput
import matplotlib.pyplot as plt
import timeit
import pandas as pd
import signal

from pysat.formula import CNF
from pysat.solvers import Glucose42

start = timeit.default_timer() # start clock

# Read file
def read_file_instance(n_instance):
    s = ''
    filepath = "inputs/ins-{}.txt".format(n_instance)
    for line in fileinput.input(files=filepath):
        s += line
    return s.splitlines()

def display_solution(strip, rectangles, pos_circuits):
    # define Matplotlib figure and axis
    fig, ax = plt.subplots()
    plt.title(f"Strip Packing Solution (Width: {strip[0]}, Height: {strip[1]})")

    if len(pos_circuits) > 0:
        for i in range(len(rectangles)):
            rect = plt.Rectangle(pos_circuits[i], *rectangles[i], edgecolor="#333", 
                                facecolor="lightblue", alpha=0.6)
            ax.add_patch(rect)
            rx, ry = pos_circuits[i]
            cx, cy = rx + rectangles[i][0]/2, ry + rectangles[i][1]/2
            ax.annotate(str(i), (cx, cy), color='black', ha='center', va='center')

    ax.set_xlim(0, strip[0])
    ax.set_ylim(0, strip[1] + 1)
    ax.set_xticks(range(strip[0] + 1))
    ax.set_yticks(range(strip[1] + 1))
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def positive_range(end):
    if end < 0:
        return []
    return range(end)

def SPP_Incremental(rectangles, strip_width, lower_bound, upper_bound, timeout=1800):
    """
    Solve 2SPP using incremental SAT solving as described in the paper.
    Returns the optimal height and the positions of rectangles.
    """
    n_rectangles = len(rectangles)
    
    # Initialize the CNF formula and variables
    cnf = CNF()
    variables = {}
    counter = 1
    
    # Start a timer for the timeout
    start_time = timeit.default_timer()
    
    # Find max height and width for symmetry breaking
    max_height = max([rectangle[1] for rectangle in rectangles])
    max_width = max([rectangle[0] for rectangle in rectangles])
    
    # Create variables for rectangle positions and relations
    # lr (left-right) and ud (up-down) variables
    for i in range(n_rectangles):
        for j in range(n_rectangles):
            if i != j:
                variables[f"lr{i+1},{j+1}"] = counter  # lri,rj
                counter += 1
                variables[f"ud{i+1},{j+1}"] = counter  # udi,rj
                counter += 1
        
        # Position variables with order encoding
        for e in positive_range(strip_width - rectangles[i][0] + 2):
            variables[f"px{i+1},{e}"] = counter  # pxi,e
            counter += 1
            
        for f in positive_range(upper_bound - rectangles[i][1] + 2):
            variables[f"py{i+1},{f}"] = counter  # pyi,f
            counter += 1
    
    # Height variables - ph_h means "can pack with height ≤ h"
    for h in range(lower_bound, upper_bound + 1):
        variables[f"ph_{h}"] = counter
        counter += 1
    
    # Add order encoding axiom clauses
    for i in range(n_rectangles):
        # ¬pxi,e ∨ pxi,e+1
        for e in range(strip_width - rectangles[i][0] + 1):
            cnf.append([-variables[f"px{i+1},{e}"], variables[f"px{i+1},{e+1}"]])
        
        # ¬pyi,f ∨ pyi,f+1
        for f in range(upper_bound - rectangles[i][1] + 1):
            cnf.append([-variables[f"py{i+1},{f}"], variables[f"py{i+1},{f+1}"]])
    
    # Add height variable ordering constraints (formula 7 in the paper)
    # If ph_o is true, then ph_{o+1} must also be true
    for h in range(lower_bound, upper_bound):
        cnf.append([-variables[f"ph_{h}"], variables[f"ph_{h+1}"]])
    
    # Define the non-overlapping constraints function
    def add_non_overlapping(i, j, h1, h2, v1, v2):
        i_width = rectangles[i][0]
        i_height = rectangles[i][1]
        j_width = rectangles[j][0]
        j_height = rectangles[j][1]
        
        # lri,j ∨ lrj,i ∨ udi,j ∨ udj,i (formula 4 in the paper)
        four_literal = []
        if h1: four_literal.append(variables[f"lr{i+1},{j+1}"])
        if h2: four_literal.append(variables[f"lr{j+1},{i+1}"])
        if v1: four_literal.append(variables[f"ud{i+1},{j+1}"])
        if v2: four_literal.append(variables[f"ud{j+1},{i+1}"])
        cnf.append(four_literal)
        
        # First type constraints (formula 5 in the paper)
        # ¬lri,j ∨ ¬pxj,e
        if h1:
            for e in range(i_width):
                if f"px{j+1},{e}" in variables:
                    cnf.append([-variables[f"lr{i+1},{j+1}"], -variables[f"px{j+1},{e}"]])
        
        # ¬lrj,i ∨ ¬pxi,e
        if h2:
            for e in range(j_width):
                if f"px{i+1},{e}" in variables:
                    cnf.append([-variables[f"lr{j+1},{i+1}"], -variables[f"px{i+1},{e}"]])
        
        # ¬udi,j ∨ ¬pyj,f
        if v1:
            for f in range(i_height):
                if f"py{j+1},{f}" in variables:
                    cnf.append([-variables[f"ud{i+1},{j+1}"], -variables[f"py{j+1},{f}"]])
        
        # ¬udj,i ∨ ¬pyi,f
        if v2:
            for f in range(j_height):
                if f"py{i+1},{f}" in variables:
                    cnf.append([-variables[f"ud{j+1},{i+1}"], -variables[f"py{i+1},{f}"]])
        
        # Second type constraints (formula 5 continued)
        # ¬lri,j ∨ pxi,e ∨ ¬pxj,e+wi
        if h1:
            for e in positive_range(strip_width - i_width):
                if f"px{j+1},{e+i_width}" in variables:
                    cnf.append([-variables[f"lr{i+1},{j+1}"], 
                              variables[f"px{i+1},{e}"], 
                              -variables[f"px{j+1},{e+i_width}"]])
        
        # ¬lrj,i ∨ pxj,e ∨ ¬pxi,e+wj
        if h2:
            for e in positive_range(strip_width - j_width):
                if f"px{i+1},{e+j_width}" in variables:
                    cnf.append([-variables[f"lr{j+1},{i+1}"], 
                              variables[f"px{j+1},{e}"], 
                              -variables[f"px{i+1},{e+j_width}"]])
        
        # ¬udi,j ∨ pyi,f ∨ ¬pyj,f+hi
        if v1:
            for f in positive_range(upper_bound - i_height):
                if f"py{j+1},{f+i_height}" in variables:
                    cnf.append([-variables[f"ud{i+1},{j+1}"], 
                              variables[f"py{i+1},{f}"], 
                              -variables[f"py{j+1},{f+i_height}"]])
        
        # ¬udj,i ∨ pyj,f ∨ ¬pyi,f+hj
        if v2:
            for f in positive_range(upper_bound - j_height):
                if f"py{i+1},{f+j_height}" in variables:
                    cnf.append([-variables[f"ud{j+1},{i+1}"], 
                              variables[f"py{j+1},{f}"], 
                              -variables[f"py{i+1},{f+j_height}"]])
    
    # Add non-overlapping constraints for all pairs
    for i in range(n_rectangles):
        for j in range(i + 1, n_rectangles):
            # Large-rectangles horizontal
            if rectangles[i][0] + rectangles[j][0] > strip_width:
                add_non_overlapping(i, j, False, False, True, True)
            # Large rectangles vertical
            elif rectangles[i][1] + rectangles[j][1] > upper_bound:
                add_non_overlapping(i, j, True, True, False, False)
            # Same-sized rectangles
            elif rectangles[i] == rectangles[j]:
                add_non_overlapping(i, j, True, False, True, True)
            # Largest width rectangle symmetry breaking
            elif rectangles[i][0] == max_width and rectangles[j][0] > (strip_width - max_width) / 2:
                add_non_overlapping(i, j, False, True, True, True)
            # Largest height rectangle symmetry breaking
            elif rectangles[i][1] == max_height and rectangles[j][1] > (upper_bound - max_height) / 2:
                add_non_overlapping(i, j, True, True, False, True)
            # Normal rectangles
            else:
                add_non_overlapping(i, j, True, True, True, True)
    
    # Domain encoding (rectangles must stay inside strip)
    for i in range(n_rectangles):
        # px(i, W-wi) - right edge must be inside strip width
        cnf.append([variables[f"px{i+1},{strip_width - rectangles[i][0]}"]])
    
    # Height constraints (formula 6 in paper)
    # For each height h, if ph_h is true, all rectangles must be below h
    for h in range(lower_bound, upper_bound + 1):
        for i in range(n_rectangles):
            # If ph_h is true, rectangle i must have its top edge at or below h
            if h >= rectangles[i][1]:
                cnf.append([-variables[f"ph_{h}"], variables[f"py{i+1},{h - rectangles[i][1]}"]])
    
    # Initialize the incremental SAT solver with the CNF formula
    with Glucose42(bootstrap_with=cnf) as solver:
        optimal_height = upper_bound
        positions = None
        
        # For model reuse (as described in the paper)
        best_model = None
        
        # Binary search with incremental solving
        current_lb = lower_bound
        current_ub = upper_bound
        
        while current_lb <= current_ub:
            # Check timeout
            if timeit.default_timer() - start_time > timeout:
                print(f"Timeout after {timeout} seconds")
                break
            
            mid = (current_lb + current_ub) // 2
            print(f"Trying height: {mid} (lower={current_lb}, upper={current_ub})")
            
            # Set up assumptions for this iteration - test if we can pack with height ≤ mid
            assumptions = [variables[f"ph_{mid}"]]
            
            # If we have a previous model, we can use it to help guide the solver
            if best_model is not None:
                # Set phase saving based on previous model (model reuse technique)
                # (Note: Glucose4 in PySAT might not directly support this feature, but the concept is here)
                pass
            
            # Solve with assumptions
            is_sat = solver.solve(assumptions=assumptions)
            
            if is_sat:
                # We found a solution with height ≤ mid
                optimal_height = mid
                
                # Save the model for reuse in future iterations
                best_model = solver.get_model()
                
                # Extract positions from the model
                positions = [[0, 0] for _ in range(n_rectangles)]
                model_vars = {abs(v): v > 0 for v in best_model}
                
                for i in range(n_rectangles):
                    # Find x position (first position where px is true)
                    for e in range(strip_width - rectangles[i][0] + 1):
                        var = variables.get(f"px{i+1},{e}", None)
                        if var is None:
                            continue
                        
                        is_true = model_vars.get(var, False)
                        prev_var = variables.get(f"px{i+1},{e-1}", None)
                        prev_is_true = model_vars.get(prev_var, False) if prev_var is not None else False
                        
                        if is_true and (e == 0 or not prev_is_true):
                            positions[i][0] = e
                            break
                    
                    # Find y position (first position where py is true)
                    for f in range(upper_bound - rectangles[i][1] + 1):
                        var = variables.get(f"py{i+1},{f}", None)
                        if var is None:
                            continue
                        
                        is_true = model_vars.get(var, False)
                        prev_var = variables.get(f"py{i+1},{f-1}", None)
                        prev_is_true = model_vars.get(prev_var, False) if prev_var is not None else False
                        
                        if is_true and (f == 0 or not prev_is_true):
                            positions[i][1] = f
                            break
                
                # Update search range - try lower height
                current_ub = mid - 1
                
                # This is the key part: Add the learned clause permanently
                # (In PySAT we just continue, but the solver keeps the learned clauses)
            
            else:
                # No solution with height ≤ mid
                # Update search range - try higher height
                current_lb = mid + 1
        
        return optimal_height, positions

# Main program
if __name__ == "__main__":
    # Default instance
    instance_number = 30
    
    # Read data
    input_data = read_file_instance(instance_number)
    width = int(input_data[0])
    n_rec = int(input_data[1])
    rectangles = [[int(val) for val in i.split()] for i in input_data[-n_rec:]]
    
    # Calculate bounds
    heights = [rectangle[1] for rectangle in rectangles]
    area = math.ceil(sum([rectangle[0] * rectangle[1] for rectangle in rectangles]) / width)
    upper_bound = sum(heights)
    lower_bound = max(area, max(heights), max([rectangle[0] for rectangle in rectangles]))
    
    print(f"Solving 2D Strip Packing for instance {instance_number}")
    print(f"Width: {width}")
    print(f"Number of rectangles: {n_rec}")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    
    # Solve with incremental SAT
    optimal_height, optimal_pos = SPP_Incremental(rectangles, width, lower_bound, upper_bound)
    
    stop = timeit.default_timer()
    print(f"Solve time: {stop - start:.2f} seconds")
    
    if optimal_height is not None:
        print(f"Optimal strip height: {optimal_height}")
        print("Rectangle positions (x, y):")
        for i, (x_pos, y_pos) in enumerate(optimal_pos):
            print(f"Rectangle {i}: ({x_pos}, {y_pos}) width={rectangles[i][0]}, height={rectangles[i][1]}")

        strip = [width, optimal_height]
        display_solution(strip, rectangles, optimal_pos)
    else:
        print("Failed to find a solution.")