import math
import fileinput
import matplotlib.pyplot as plt
import timeit
import pandas as pd
import signal

from pysat.formula import CNF
from pysat.solvers import Glucose4

start = timeit.default_timer() # start clock

# Read file
def read_file_instance(n_instance):
    s = ''
    filepath = "inputs/ins-{}.txt".format(n_instance)
    for line in fileinput.input(files=filepath):
        s += line
    return s.splitlines()

def display_solution(strip, rectangles, pos_circuits, rotations):
    # define Matplotlib figure and axis
    fig, ax = plt.subplots()
    plt.title(f"Strip Packing Solution (Width: {strip[0]}, Height: {strip[1]})")

    if len(pos_circuits) > 0:
        for i in range(len(rectangles)):
            # Use rotated dimensions if needed
            w = rectangles[i][1] if rotations[i] else rectangles[i][0]
            h = rectangles[i][0] if rotations[i] else rectangles[i][1]
            
            rect = plt.Rectangle(pos_circuits[i], w, h, edgecolor="#333", 
                                facecolor="lightblue", alpha=0.6)
            ax.add_patch(rect)
            rx, ry = pos_circuits[i]
            cx, cy = rx + w/2, ry + h/2
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

def get_dimensions(rectangles, i, rotated=False):
    """Helper function to get width and height based on rotation state"""
    if not rotated:
        return rectangles[i][0], rectangles[i][1]  # original width, height
    else:
        return rectangles[i][1], rectangles[i][0]  # rotated width, height (swapped)

def SPP_Incremental_Rotation(rectangles, strip_width, lower_bound, upper_bound, timeout=1800):
    """
    Solve 2SPP with rotation using incremental SAT solving.
    Returns the optimal height, the positions of rectangles, and rotation status.
    """
    n_rectangles = len(rectangles)
    
    # Initialize the CNF formula and variables
    cnf = CNF()
    variables = {}
    counter = 1
    
    # Start a timer for the timeout
    start_time = timeit.default_timer()
    
    # Create variables for rectangle positions, relations, and rotation
    # Rotation variables - r_i is TRUE if rectangle i is rotated
    for i in range(n_rectangles):
        variables[f"r{i+1}"] = counter
        counter += 1
    
    # lr (left-right) and ud (up-down) variables
    for i in range(n_rectangles):
        for j in range(n_rectangles):
            if i != j:
                variables[f"lr{i+1},{j+1}"] = counter  # lri,rj
                counter += 1
                variables[f"ud{i+1},{j+1}"] = counter  # udi,rj
                counter += 1
    
    # Position variables - need to consider both orientations for each rectangle
    for i in range(n_rectangles):
        # Calculate maximum width for both orientations
        max_width_i = max(rectangles[i][0], rectangles[i][1])
        
        # Position variables for x-axis
        for e in positive_range(strip_width - min(rectangles[i][0], rectangles[i][1]) + 2):
            variables[f"px{i+1},{e}"] = counter  # pxi,e
            counter += 1
        
        # Position variables for y-axis
        for f in positive_range(upper_bound - min(rectangles[i][0], rectangles[i][1]) + 2):
            variables[f"py{i+1},{f}"] = counter  # pyi,f
            counter += 1
    
    # Height variables - ph_h means "can pack with height ≤ h"
    for h in range(lower_bound, upper_bound + 1):
        variables[f"ph_{h}"] = counter
        counter += 1
    
    # Add order encoding axiom clauses
    for i in range(n_rectangles):
        # Need to consider minimum width regardless of rotation
        min_width_i = min(rectangles[i][0], rectangles[i][1])
        min_height_i = min(rectangles[i][0], rectangles[i][1])
        
        # ¬pxi,e ∨ pxi,e+1
        for e in range(strip_width - min_width_i + 1):
            cnf.append([-variables[f"px{i+1},{e}"], variables[f"px{i+1},{e+1}"]])
        
        # ¬pyi,f ∨ pyi,f+1
        for f in range(upper_bound - min_height_i + 1):
            cnf.append([-variables[f"py{i+1},{f}"], variables[f"py{i+1},{f+1}"]])
    
    # Add height variable ordering constraints
    for h in range(lower_bound, upper_bound):
        cnf.append([-variables[f"ph_{h}"], variables[f"ph_{h+1}"]])
    
    # Define the non-overlapping constraints function for both rotation possibilities
    def add_non_overlapping(i, j, h1, h2, v1, v2):
        # We need to handle all 4 rotation combinations
        # 1. Both normal orientation
        # 2. i rotated, j normal
        # 3. i normal, j rotated
        # 4. Both rotated
        
        # To simplify, we'll add these constraints as four separate cases
        # Case 1: Both normal
        add_non_overlapping_fixed(i, j, h1, h2, v1, v2, False, False)
        
        # Case 2: i rotated, j normal
        add_non_overlapping_fixed(i, j, h1, h2, v1, v2, True, False)
        
        # Case 3: i normal, j rotated
        add_non_overlapping_fixed(i, j, h1, h2, v1, v2, False, True)
        
        # Case 4: Both rotated
        add_non_overlapping_fixed(i, j, h1, h2, v1, v2, True, True)
    
    def add_non_overlapping_fixed(i, j, h1, h2, v1, v2, i_rotated, j_rotated):
        # Get dimensions based on rotation status
        i_width, i_height = get_dimensions(rectangles, i, i_rotated)
        j_width, j_height = get_dimensions(rectangles, j, j_rotated)
        
        # Build literals for rotations
        i_rotation = variables[f"r{i+1}"] if i_rotated else -variables[f"r{i+1}"]
        j_rotation = variables[f"r{j+1}"] if j_rotated else -variables[f"r{j+1}"]
        
        # lri,j ∨ lrj,i ∨ udi,j ∨ udj,i with rotation conditions
        four_literal = []
        if h1: four_literal.append(variables[f"lr{i+1},{j+1}"])
        if h2: four_literal.append(variables[f"lr{j+1},{i+1}"])
        if v1: four_literal.append(variables[f"ud{i+1},{j+1}"])
        if v2: four_literal.append(variables[f"ud{j+1},{i+1}"])
        
        # Add rotation conditions to the core constraint
        cnf.append(four_literal + [i_rotation, j_rotation])
        
        # First type constraints conditioned on rotation
        # ¬lri,j ∨ ¬pxj,e ∨ ¬ri ∨ ¬rj
        if h1:
            for e in range(i_width):
                if f"px{j+1},{e}" in variables:
                    cnf.append([-variables[f"lr{i+1},{j+1}"], -variables[f"px{j+1},{e}"], 
                               -i_rotation, -j_rotation])
        
        # ¬lrj,i ∨ ¬pxi,e ∨ ¬ri ∨ ¬rj
        if h2:
            for e in range(j_width):
                if f"px{i+1},{e}" in variables:
                    cnf.append([-variables[f"lr{j+1},{i+1}"], -variables[f"px{i+1},{e}"], 
                               -i_rotation, -j_rotation])
        
        # ¬udi,j ∨ ¬pyj,f ∨ ¬ri ∨ ¬rj
        if v1:
            for f in range(i_height):
                if f"py{j+1},{f}" in variables:
                    cnf.append([-variables[f"ud{i+1},{j+1}"], -variables[f"py{j+1},{f}"], 
                               -i_rotation, -j_rotation])
        
        # ¬udj,i ∨ ¬pyi,f ∨ ¬ri ∨ ¬rj
        if v2:
            for f in range(j_height):
                if f"py{i+1},{f}" in variables:
                    cnf.append([-variables[f"ud{j+1},{i+1}"], -variables[f"py{i+1},{f}"], 
                               -i_rotation, -j_rotation])
        
        # Second type constraints conditioned on rotation
        # ¬lri,j ∨ pxi,e ∨ ¬pxj,e+wi ∨ ¬ri ∨ ¬rj
        if h1:
            for e in positive_range(strip_width - i_width):
                if f"px{j+1},{e+i_width}" in variables and f"px{i+1},{e}" in variables:
                    cnf.append([-variables[f"lr{i+1},{j+1}"], 
                              variables[f"px{i+1},{e}"], 
                              -variables[f"px{j+1},{e+i_width}"],
                              -i_rotation, -j_rotation])
        
        # ¬lrj,i ∨ pxj,e ∨ ¬pxi,e+wj ∨ ¬ri ∨ ¬rj
        if h2:
            for e in positive_range(strip_width - j_width):
                if f"px{i+1},{e+j_width}" in variables and f"px{j+1},{e}" in variables:
                    cnf.append([-variables[f"lr{j+1},{i+1}"], 
                              variables[f"px{j+1},{e}"], 
                              -variables[f"px{i+1},{e+j_width}"],
                              -i_rotation, -j_rotation])
        
        # ¬udi,j ∨ pyi,f ∨ ¬pyj,f+hi ∨ ¬ri ∨ ¬rj
        if v1:
            for f in positive_range(upper_bound - i_height):
                if f"py{j+1},{f+i_height}" in variables and f"py{i+1},{f}" in variables:
                    cnf.append([-variables[f"ud{i+1},{j+1}"], 
                              variables[f"py{i+1},{f}"], 
                              -variables[f"py{j+1},{f+i_height}"],
                              -i_rotation, -j_rotation])
        
        # ¬udj,i ∨ pyj,f ∨ ¬pyi,f+hj ∨ ¬ri ∨ ¬rj
        if v2:
            for f in positive_range(upper_bound - j_height):
                if f"py{i+1},{f+j_height}" in variables and f"py{j+1},{f}" in variables:
                    cnf.append([-variables[f"ud{j+1},{i+1}"], 
                              variables[f"py{j+1},{f}"], 
                              -variables[f"py{i+1},{f+j_height}"],
                              -i_rotation, -j_rotation])
    
    # Add non-overlapping constraints for all pairs
    for i in range(n_rectangles):
        for j in range(i + 1, n_rectangles):
            # For rotation, we need to check all possible combinations of dimensions
            min_i_dim = min(rectangles[i][0], rectangles[i][1])
            min_j_dim = min(rectangles[j][0], rectangles[j][1])
            
            # Large-rectangles horizontal (no matter the rotation)
            if min_i_dim + min_j_dim > strip_width:
                add_non_overlapping(i, j, False, False, True, True)
            # Large rectangles vertical (no matter the rotation)
            elif min_i_dim + min_j_dim > upper_bound:
                add_non_overlapping(i, j, True, True, False, False)
            # Normal rectangles
            else:
                add_non_overlapping(i, j, True, True, True, True)
    
    # Domain encoding (rectangles must stay inside strip)
    for i in range(n_rectangles):
        # Original orientation
        if rectangles[i][0] <= strip_width:
            # If not rotated, the right edge must be inside strip width
            cnf.append([-variables[f"r{i+1}"], variables[f"px{i+1},{strip_width - rectangles[i][0]}"]])
        else:
            # If rectangle is too wide to fit in normal orientation, force rotation
            cnf.append([variables[f"r{i+1}"]])
        
        # Rotated orientation
        if rectangles[i][1] <= strip_width:
            # If rotated, the right edge (now width=height) must be inside strip width
            cnf.append([variables[f"r{i+1}"], variables[f"px{i+1},{strip_width - rectangles[i][1]}"]])
        else:
            # If rectangle is too wide to fit in rotated orientation, forbid rotation
            cnf.append([-variables[f"r{i+1}"]])
    
    # Height constraints with rotation consideration
    for h in range(lower_bound, upper_bound + 1):
        for i in range(n_rectangles):
            # Normal orientation: if ph_h is true, rectangle i must have its top edge at or below h
            if h >= rectangles[i][1]:
                cnf.append([-variables[f"ph_{h}"], variables[f"r{i+1}"], 
                           variables[f"py{i+1},{h - rectangles[i][1]}"]])
            
            # Rotated orientation: top edge is determined by the original width
            if h >= rectangles[i][0]:
                cnf.append([-variables[f"ph_{h}"], -variables[f"r{i+1}"], 
                           variables[f"py{i+1},{h - rectangles[i][0]}"]])
    
    # Initialize the incremental SAT solver with the CNF formula
    with Glucose4(bootstrap_with=cnf) as solver:
        optimal_height = upper_bound
        positions = None
        rotations = [False] * n_rectangles
        
        # For model reuse
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
            
            # Solve with assumptions
            is_sat = solver.solve(assumptions=assumptions)
            
            if is_sat:
                # We found a solution with height ≤ mid
                optimal_height = mid
                
                # Save the model for reuse in future iterations
                best_model = solver.get_model()
                
                # Extract positions and rotations from the model
                positions = [[0, 0] for _ in range(n_rectangles)]
                model_vars = {abs(v): v > 0 for v in best_model}
                
                # Extract rotation variables
                for i in range(n_rectangles):
                    rotations[i] = model_vars.get(variables[f"r{i+1}"], False)
                
                # Extract positions
                for i in range(n_rectangles):
                    # Current rectangle dimensions based on rotation
                    width_i = rectangles[i][1] if rotations[i] else rectangles[i][0]
                    height_i = rectangles[i][0] if rotations[i] else rectangles[i][1]
                    
                    # Find x position (first position where px is true)
                    for e in range(strip_width - width_i + 1):
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
                    for f in range(upper_bound - height_i + 1):
                        var = variables.get(f"py{i+1},{f}", None)
                        if var is None:
                            continue
                        
                        is_true = model_vars.get(var, False)
                        prev_var = variables.get(f"py{i+1},{f-1}", None)
                        prev_is_true = model_vars.get(prev_var, False) if prev_var is not None else False
                        
                        if is_true and (f == 0 or not prev_is_true):
                            positions[i][1] = f
                            break
                
                # Verify solution
                actual_max_height = 0
                for i in range(n_rectangles):
                    height_i = rectangles[i][0] if rotations[i] else rectangles[i][1]
                    top_edge = positions[i][1] + height_i
                    actual_max_height = max(actual_max_height, top_edge)
                
                if actual_max_height > optimal_height:
                    print(f"WARNING: Actual height ({actual_max_height}) exceeds solution height ({optimal_height})")
                    optimal_height = actual_max_height
                
                # Update search range - try lower height
                current_ub = mid - 1
            
            else:
                # No solution with height ≤ mid
                # Update search range - try higher height
                current_lb = mid + 1
        
        return optimal_height, positions, rotations

# Main program
if __name__ == "__main__":
    # Default instance
    instance_number = 10
    
    # Read data
    input_data = read_file_instance(instance_number)
    width = int(input_data[0])
    n_rec = int(input_data[1])
    rectangles = [[int(val) for val in i.split()] for i in input_data[-n_rec:]]
    
    # Calculate bounds
    max_rectangle_dimension = max([max(rect[0], rect[1]) for rect in rectangles])
    sum_areas = sum([rect[0] * rect[1] for rect in rectangles])
    
    # For rotation, lower bound needs to consider minimum height after possible rotation
    lower_bound = max(
        math.ceil(sum_areas / width),  # Area bound
        max([min(rect[0], rect[1]) for rect in rectangles])  # Minimum height with rotation allowed
    )
    
    # Upper bound is more conservative with rotation
    upper_bound = min(
        sum([min(rect[0], rect[1]) for rect in rectangles]),  # Sum of minimum heights
        width * max_rectangle_dimension  # Worst case: tallest rectangle takes entire width
    )
    
    print(f"Solving 2D Strip Packing with rotation for instance {instance_number}")
    print(f"Width: {width}")
    print(f"Number of rectangles: {n_rec}")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    
    # Solve with incremental SAT including rotation
    optimal_height, optimal_pos, rotations = SPP_Incremental_Rotation(
        rectangles, width, lower_bound, upper_bound)
    
    stop = timeit.default_timer()
    print(f"Solve time: {stop - start:.2f} seconds")
    
    if optimal_height is not None:
        print(f"Optimal strip height: {optimal_height}")
        print("Rectangle positions (x, y) and rotation status:")
        for i, (x_pos, y_pos) in enumerate(optimal_pos):
            rot_status = "Rotated" if rotations[i] else "Not Rotated"
            width_i = rectangles[i][1] if rotations[i] else rectangles[i][0]
            height_i = rectangles[i][0] if rotations[i] else rectangles[i][1]
            print(f"Rectangle {i}: ({x_pos}, {y_pos}) width={width_i}, height={height_i}, {rot_status}")

        strip = [width, optimal_height]
        display_solution(strip, rectangles, optimal_pos, rotations)
    else:
        print("Failed to find a solution.")