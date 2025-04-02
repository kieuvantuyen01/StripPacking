import math
import tempfile
import subprocess
import os
import fileinput
import matplotlib.pyplot as plt
import timeit
import sys
import pandas as pd

start = timeit.default_timer()  # start clock

# read file
def read_file_instance(n_instance):
    s = ''
    filepath = "inputs/ins-{}.txt".format(n_instance)
    for line in fileinput.input(files=filepath):
        s += line
    return s.splitlines()

def display_solution(strip, rectangles, pos_circuits):
    fig, ax = plt.subplots()
    plt.title(f"Strip Packing Solution (Width: {strip[0]}, Height: {strip[1]})")

    if len(pos_circuits) > 0:
        for i in range(len(rectangles)):
            rect = plt.Rectangle(pos_circuits[i], rectangles[i][0], rectangles[i][1], 
                               edgecolor="#333", facecolor="lightblue", alpha=0.6)
            ax.add_patch(rect)
            rx, ry = pos_circuits[i]
            cx, cy = rx + rectangles[i][0]/2, ry + rectangles[i][1]/2
            ax.annotate(str(i), (cx, cy), color='black', ha='center', va='center')

    ax.set_xlim(0, strip[0])
    ax.set_ylim(0, strip[1] + 1)
    ax.set_xticks(range(strip[0] + 1))
    ax.set_yticks(range(int(strip[1]) + 2))
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def positive_range(end):
    if end < 0:
        return []
    return range(end)

def SPP_MaxSAT(width, rectangles, lower_bound, upper_bound):
    """Solve 2SPP using Max-SAT approach with open-wbo"""
    n_rectangles = len(rectangles)
    variables = {}
    counter = 1
    
    # Create a temporary file for the Max-SAT formula
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.wcnf') as file:
        wcnf_file = file.name
        
        # Define variables for rectangle positions and relations
        for i in range(n_rectangles):
            for j in range(n_rectangles):
                if i != j:
                    variables[f"lr{i + 1},{j + 1}"] = counter  # lri,rj
                    counter += 1
                    variables[f"ud{i + 1},{j + 1}"] = counter  # uri,rj
                    counter += 1
            for e in range(width - rectangles[i][0] + 2):  # Position variables for x-axis
                variables[f"px{i + 1},{e}"] = counter  # pxi,e
                counter += 1
            for h in range(upper_bound - rectangles[i][1] + 2):  # Position variables for y-axis
                variables[f"py{i + 1},{h}"] = counter  # pyi,h
                counter += 1
        
        # Height constraint variables - ph_h means "can pack with height ≤ h"
        for h in range(lower_bound, upper_bound + 1):
            variables[f"ph_{h}"] = counter
            counter += 1
            
        # Prepare hard clauses (basic packing constraints)
        hard_clauses = []
        
        # Order encoding axioms
        for i in range(n_rectangles):
            for e in range(width - rectangles[i][0] + 1):
                hard_clauses.append([-variables[f"px{i + 1},{e}"], variables[f"px{i + 1},{e + 1}"]])
            for h in range(upper_bound - rectangles[i][1] + 1):
                hard_clauses.append([-variables[f"py{i + 1},{h}"], variables[f"py{i + 1},{h + 1}"]])
        
        # Height variable ordering - this enforces that ph_h implies ph_{h+1}
        for h in range(lower_bound, upper_bound):
            hard_clauses.append([-variables[f"ph_{h}"], variables[f"ph_{h+1}"]])
        
        # Non-overlapping constraints function 
        def non_overlapping(i, j, h1, h2, v1, v2):
            i_width = rectangles[i][0]
            i_height = rectangles[i][1]
            j_width = rectangles[j][0]
            j_height = rectangles[j][1]
        
            # lri,j v lrj,i v udi,j v udj,i
            four_literal = []
            if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
            if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
            if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
            if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])
            hard_clauses.append(four_literal)
        
            # First type of constraints - prevent rectangle j's left edge from being in first i_width positions
            if h1:
                for e in range(i_width):  # FIXED: Use i_width instead of min(width...)
                    if f"px{j + 1},{e}" in variables:
                        hard_clauses.append([-variables[f"lr{i + 1},{j + 1}"], -variables[f"px{j + 1},{e}"]])
            
            # First type for h2 - prevent rectangle i's left edge from being in first j_width positions
            if h2:
                for e in range(j_width):  # FIXED: Use j_width instead of min(width...)
                    if f"px{i + 1},{e}" in variables:
                        hard_clauses.append([-variables[f"lr{j + 1},{i + 1}"], -variables[f"px{i + 1},{e}"]])
        
            # First type for v1 - prevent rectangle j's bottom edge from being in first i_height positions
            if v1:
                for y_pos in range(i_height):  # FIXED: Use i_height instead of min(upper_bound...)
                    if f"py{j + 1},{y_pos}" in variables:
                        hard_clauses.append([-variables[f"ud{i + 1},{j + 1}"], -variables[f"py{j + 1},{y_pos}"]])
            
            # First type for v2 - prevent rectangle i's bottom edge from being in first j_height positions
            if v2:
                for y_pos in range(j_height):  # FIXED: Use j_height instead of min(upper_bound...)
                    if f"py{i + 1},{y_pos}" in variables:
                        hard_clauses.append([-variables[f"ud{j + 1},{i + 1}"], -variables[f"py{i + 1},{y_pos}"]])
        
            # Second type of constraints - enforce relative positions when certain relations hold
            
            # For h1: if rectangle i is to the left of j, then i's left edge at e implies j can't be at e+i_width
            if h1:
                for e in positive_range(width - i_width):  # FIXED: Use width - i_width
                    if f"px{i + 1},{e}" in variables and f"px{j + 1},{e + i_width}" in variables:
                        hard_clauses.append([-variables[f"lr{i + 1},{j + 1}"],
                                       variables[f"px{i + 1},{e}"], 
                                       -variables[f"px{j + 1},{e + i_width}"]])
            
            # For h2: if rectangle j is to the left of i, then j's left edge at e implies i can't be at e+j_width
            if h2:
                for e in positive_range(width - j_width):  # FIXED: Use width - j_width
                    if f"px{j + 1},{e}" in variables and f"px{i + 1},{e + j_width}" in variables:
                        hard_clauses.append([-variables[f"lr{j + 1},{i + 1}"],
                                       variables[f"px{j + 1},{e}"], 
                                       -variables[f"px{i + 1},{e + j_width}"]])
        
            # For v1: if rectangle i is below j, then i's bottom edge at f implies j can't be at f+i_height
            if v1:
                for y_pos in positive_range(upper_bound - i_height):  # FIXED: Use height - i_height
                    if f"py{i + 1},{y_pos}" in variables and f"py{j + 1},{y_pos + i_height}" in variables:
                        hard_clauses.append([-variables[f"ud{i + 1},{j + 1}"],
                                       variables[f"py{i + 1},{y_pos}"], 
                                       -variables[f"py{j + 1},{y_pos + i_height}"]])
            
            # For v2: if rectangle j is below i, then j's bottom edge at f implies i can't be at f+j_height
            if v2:
                for y_pos in positive_range(upper_bound - j_height):  # FIXED: Use height - j_height
                    if f"py{j + 1},{y_pos}" in variables and f"py{i + 1},{y_pos + j_height}" in variables:
                        hard_clauses.append([-variables[f"ud{j + 1},{i + 1}"],
                                       variables[f"py{j + 1},{y_pos}"], 
                                       -variables[f"py{i + 1},{y_pos + j_height}"]])
                
        # Add non-overlapping constraints for all pairs of rectangles
        for i in range(n_rectangles):
            for j in range(i + 1, n_rectangles):
                # Large-rectangles horizontal
                if rectangles[i][0] + rectangles[j][0] > width:
                    non_overlapping(i, j, False, False, True, True)
                # Large rectangles vertical
                elif rectangles[i][1] + rectangles[j][1] > upper_bound:
                    non_overlapping(i, j, True, True, False, False)
                # Same-sized rectangles
                elif rectangles[i] == rectangles[j]:
                    non_overlapping(i, j, True, False, True, True)
                # Normal rectangles
                else:
                    non_overlapping(i, j, True, True, True, True)
                
        # Domain encoding to ensure rectangles are placed within strip boundaries
        for i in range(n_rectangles):
            # Right edge within strip width
            hard_clauses.append([variables[f"px{i + 1},{width - rectangles[i][0]}"]])
            
            # Top edge within strip height - integrated with height constraint variables
            for h in range(lower_bound, upper_bound + 1):
                if h >= rectangles[i][1]:  # Rectangle must fit below height h
                    hard_clauses.append([-variables[f"ph_{h}"], variables[f"py{i + 1},{h - rectangles[i][1]}"]])
        
        # Prepare soft clauses with weights for height minimization
        soft_clauses = []
        
        # Use increasing weights to prefer FALSE for larger heights
        for h in range(lower_bound, upper_bound + 1):
            weight = 1
            soft_clauses.append((1, [variables[f"ph_{h}"]]))  # We want ph_h to be FALSE when possible
        
        # Require at least one ph_h to be true (ensures a solution exists)
        all_ph_vars = [variables[f"ph_{h}"] for h in range(lower_bound, upper_bound + 1)]
        hard_clauses.append(all_ph_vars)
        
        # Write WCNF header: p wcnf num_variables num_clauses top_weight
        top_weight = 2  # Any value > max soft clause weight
        file.write(f"p wcnf {counter - 1} {len(hard_clauses) + len(soft_clauses)} {top_weight}\n")
        
        # Write hard clauses with top weight
        for clause in hard_clauses:
            file.write(f"{top_weight} {' '.join(map(str, clause))} 0\n")
        
        # Write soft clauses with their weights
        for weight, clause in soft_clauses:
            file.write(f"{weight} {' '.join(map(str, clause))} 0\n")
    
    # Call open-wbo solver
    try:
        print(f"Running open-wbo on {wcnf_file}...")
        result = subprocess.run(
            ["open-wbo", wcnf_file], 
            capture_output=True, 
            text=True
        )
        
        output = result.stdout
        
        # Parse the output to find the model
        optimal_height = upper_bound
        positions = [[0, 0] for _ in range(n_rectangles)]
        
        if "OPTIMUM FOUND" in output:
            print("Optimal solution found!")
            # Extract the model line (starts with "v ")
            for line in output.split('\n'):
                if line.startswith('v '):
                    model_str = line[2:].strip()
                    model = list(map(int, model_str.split()))
                    
                    # Process the model to extract positions
                    true_vars = set()
                    for var in model:
                        if var > 0:  # Only positive literals are true
                            true_vars.add(var)
                    
                    # Extract height variables and find minimum height where ph_h is true
                    ph_true_heights = []
                    for h in range(lower_bound, upper_bound + 1):
                        if variables[f"ph_{h}"] in true_vars:
                            ph_true_heights.append(h)
                    
                    if ph_true_heights:
                        optimal_height = min(ph_true_heights)
                    else:
                        print("WARNING: No ph_h variables are true! This shouldn't happen with our constraints.")
                    
                    # Diagnostic output
                    print(f"Heights where ph_h is true: {sorted(ph_true_heights)}")
                    
                    # Extract positions - Find the exact transition point for each rectangle
                    for i in range(n_rectangles):
                        # Find x position (first position where px is true)
                        found_x = False
                        for e in range(width - rectangles[i][0] + 1):
                            if f"px{i + 1},{e}" in variables and variables[f"px{i + 1},{e}"] in true_vars:
                                if e == 0 or variables[f"px{i + 1},{e-1}"] not in true_vars:
                                    positions[i][0] = e
                                    found_x = True
                                    break
                        if not found_x:
                            print(f"WARNING: Could not determine x-position for rectangle {i}!")
                        
                        # Find y position (first position where py is true)
                        found_y = False
                        for y_pos in range(upper_bound - rectangles[i][1] + 1):
                            if f"py{i + 1},{y_pos}" in variables and variables[f"py{i + 1},{y_pos}"] in true_vars:
                                if y_pos == 0 or variables[f"py{i + 1},{y_pos-1}"] not in true_vars:
                                    positions[i][1] = y_pos
                                    found_y = True
                                    break
                        if not found_y:
                            print(f"WARNING: Could not determine y-position for rectangle {i}!")
                    
                    # CRITICAL: Verify that all rectangles fit within the optimal height
                    actual_max_height = 0
                    for i in range(n_rectangles):
                        top_edge = positions[i][1] + rectangles[i][1]
                        actual_max_height = max(actual_max_height, top_edge)
                        
                        # Individual rectangle check
                        if top_edge > optimal_height:
                            print(f"WARNING: Rectangle {i} extends to height {top_edge}, "
                                  f"exceeding stated optimal height {optimal_height}!")
                    
                    # Overall check
                    if actual_max_height != optimal_height:
                        print(f"WARNING: Actual packing height ({actual_max_height}) differs from "
                              f"theoretical optimal ({optimal_height})!")
                        
                        # Use the actual maximum height to ensure valid display
                        optimal_height = actual_max_height
                    else:
                        print(f"Verification successful: All rectangles fit within optimal height {optimal_height}.")
                    
                    break
        else:
            print("No optimal solution found.")
            print(f"Solver output: {output}")
        
        # Clean up the temporary file
        os.unlink(wcnf_file)
        return optimal_height, positions
    
    except Exception as e:
        print(f"Error running Max-SAT solver: {e}")
        if os.path.exists(wcnf_file):
            os.unlink(wcnf_file)
        return None, None

# Main program
if __name__ == "__main__":
    # Default instance number
    instance_number = 30
    
    # Check if instance number is provided as command-line argument
    if len(sys.argv) > 1:
        instance_number = int(sys.argv[1])
    
    # Read data from file
    input_data = read_file_instance(instance_number)
    width = int(input_data[0])
    n_rec = int(input_data[1])
    rectangles = [[int(val) for val in i.split()] for i in input_data[-n_rec:]]
    
    # Calculate bounds
    heights = [int(rectangle[1]) for rectangle in rectangles]
    area = math.floor(sum([int(rectangle[0] * rectangle[1]) for rectangle in rectangles]) / width)
    upper_bound = sum(heights)
    lower_bound = max(math.ceil(sum([int(rectangle[0] * rectangle[1]) for rectangle in rectangles]) / width), 
                      max(heights))
    
    print(f"Solving Strip Packing with MaxSAT (No Rotation) for instance {instance_number}")
    print(f"Width: {width}")
    print(f"Number of rectangles: {n_rec}")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    
    # Solve with MaxSAT
    optimal_height, optimal_pos = SPP_MaxSAT(width, rectangles, lower_bound, upper_bound)
    
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