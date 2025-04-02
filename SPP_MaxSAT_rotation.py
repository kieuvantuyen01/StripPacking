import math
import tempfile
import subprocess
import os
import fileinput
import matplotlib.pyplot as plt
import timeit
import sys

start = timeit.default_timer()  # start clock

# read file
def read_file_instance(n_instance):
    s = ''
    filepath = "inputs/ins-{}.txt".format(n_instance)
    for line in fileinput.input(files=filepath):
        s += line
    return s.splitlines()

def display_solution(strip, rectangles, pos_circuits, rotations):
    fig, ax = plt.subplots()
    plt.title(f"Strip Packing Solution (Width: {strip[0]}, Height: {strip[1]})")

    if len(pos_circuits) > 0:
        for i in range(len(rectangles)):
            w = rectangles[i][1] if rotations[i] else rectangles[i][0]
            h = rectangles[i][0] if rotations[i] else rectangles[i][1]
            rect = plt.Rectangle(pos_circuits[i], w, h, 
                               edgecolor="#333", facecolor="lightblue", alpha=0.6)
            ax.add_patch(rect)
            rx, ry = pos_circuits[i]
            cx, cy = rx + w/2, ry + h/2
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
            for e in range(width):
                variables[f"px{i + 1},{e}"] = counter  # pxi,e
                counter += 1
            for h in range(upper_bound):
                variables[f"py{i + 1},{h}"] = counter  # pyi,h
                counter += 1
            variables[f"r{i + 1}"] = counter  # rotation
            counter += 1
        
        # Height constraint variables
        for h in range(lower_bound, upper_bound + 1):
            variables[f"ph_{h}"] = counter
            counter += 1
            
        # Prepare hard clauses (basic packing constraints)
        hard_clauses = []
        
        # Order encoding axioms
        for i in range(n_rectangles):
            for e in range(width - 1):
                hard_clauses.append([-variables[f"px{i + 1},{e}"], variables[f"px{i + 1},{e + 1}"]])
            for h in range(upper_bound - 1):
                hard_clauses.append([-variables[f"py{i + 1},{h}"], variables[f"py{i + 1},{h + 1}"]])
        
        # Height variable ordering - this enforces that ph_h implies ph_{h+1}
        for h in range(lower_bound, upper_bound):
            hard_clauses.append([-variables[f"ph_{h}"], variables[f"ph_{h+1}"]])
        
        # Non-overlapping constraints function 
        def add_non_overlapping(rotated, i, j, h1, h2, v1, v2):
            # Get dimensions based on rotation
            if not rotated:
                i_width = rectangles[i][0]
                i_height = rectangles[i][1]
                j_width = rectangles[j][0]
                j_height = rectangles[j][1]
                i_rotation = variables[f"r{i + 1}"]
                j_rotation = variables[f"r{j + 1}"]
            else:
                i_width = rectangles[i][1]
                i_height = rectangles[i][0]
                j_width = rectangles[j][1]
                j_height = rectangles[j][0]
                i_rotation = -variables[f"r{i + 1}"]
                j_rotation = -variables[f"r{j + 1}"]

            # lri,j v lrj,i v udi,j v udj,i
            four_literal = []
            if h1: four_literal.append(variables[f"lr{i + 1},{j + 1}"])
            if h2: four_literal.append(variables[f"lr{j + 1},{i + 1}"])
            if v1: four_literal.append(variables[f"ud{i + 1},{j + 1}"])
            if v2: four_literal.append(variables[f"ud{j + 1},{i + 1}"])

            hard_clauses.append(four_literal + [i_rotation])
            hard_clauses.append(four_literal + [j_rotation])

            # Add constraints only if they're necessary
            if h1:
                for e in range(min(width, i_width)):
                    hard_clauses.append([i_rotation, -variables[f"lr{i + 1},{j + 1}"], -variables[f"px{j + 1},{e}"]])
            
                for e in positive_range(width - i_width):
                    hard_clauses.append([i_rotation, -variables[f"lr{i + 1},{j + 1}"],
                                variables[f"px{i + 1},{e}"], -variables[f"px{j + 1},{e + i_width}"]])
            
            if h2:
                for e in range(min(width, j_width)):
                    hard_clauses.append([j_rotation, -variables[f"lr{j + 1},{i + 1}"], -variables[f"px{i + 1},{e}"]])
                
                for e in positive_range(width - j_width):
                    hard_clauses.append([j_rotation, -variables[f"lr{j + 1},{i + 1}"],
                                variables[f"px{j + 1},{e}"], -variables[f"px{i + 1},{e + j_width}"]])

            if v1:
                for y_pos in range(min(upper_bound, i_height)):
                    hard_clauses.append([i_rotation, -variables[f"ud{i + 1},{j + 1}"], -variables[f"py{j + 1},{y_pos}"]])
                
                for y_pos in positive_range(upper_bound - i_height):
                    hard_clauses.append([i_rotation, -variables[f"ud{i + 1},{j + 1}"],
                                variables[f"py{i + 1},{y_pos}"], -variables[f"py{j + 1},{y_pos + i_height}"]])
            
            if v2:
                for y_pos in range(min(upper_bound, j_height)):
                    hard_clauses.append([j_rotation, -variables[f"ud{j + 1},{i + 1}"], -variables[f"py{i + 1},{y_pos}"]])
                
                for y_pos in positive_range(upper_bound - j_height):
                    hard_clauses.append([j_rotation, -variables[f"ud{j + 1},{i + 1}"],
                                variables[f"py{j + 1},{y_pos}"], -variables[f"py{i + 1},{y_pos + j_height}"]])
                
        # Add non-overlapping constraints for all pairs of rectangles
        for i in range(n_rectangles):
            for j in range(i + 1, n_rectangles):
                # Large-rectangles horizontal
                if min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > width:
                    add_non_overlapping(False, i, j, False, False, True, True)
                    add_non_overlapping(True, i, j, False, False, True, True)
                # Large rectangles vertical
                elif min(rectangles[i][0], rectangles[i][1]) + min(rectangles[j][0], rectangles[j][1]) > upper_bound:
                    add_non_overlapping(False, i, j, True, True, False, False)
                    add_non_overlapping(True, i, j, True, True, False, False)
                # Normal rectangles
                else:
                    add_non_overlapping(False, i, j, True, True, True, True)
                    add_non_overlapping(True, i, j, True, True, True, True)
                
        # Domain encoding to ensure every rectangle stays inside strip's boundary
        for i in range(n_rectangles):
            if rectangles[i][0] > width:
                hard_clauses.append([variables[f"r{i + 1}"]])
            else:
                for e in range(width - rectangles[i][0], width):
                    hard_clauses.append([variables[f"r{i + 1}"], variables[f"px{i + 1},{e}"]])
            
            if rectangles[i][1] > upper_bound:
                hard_clauses.append([variables[f"r{i + 1}"]])
            else:
                for y_pos in range(upper_bound - rectangles[i][1], upper_bound):
                    hard_clauses.append([variables[f"r{i + 1}"], variables[f"py{i + 1},{y_pos}"]])

            # Rotated
            if rectangles[i][1] > width:
                hard_clauses.append([-variables[f"r{i + 1}"]])
            else:
                for e in range(width - rectangles[i][1], width):
                    hard_clauses.append([-variables[f"r{i + 1}"], variables[f"px{i + 1},{e}"]])
            
            if rectangles[i][0] > upper_bound:
                hard_clauses.append([-variables[f"r{i + 1}"]])
            else:
                for y_pos in range(upper_bound - rectangles[i][0], upper_bound):
                    hard_clauses.append([-variables[f"r{i + 1}"], variables[f"py{i + 1},{y_pos}"]])
        
        # Height-related constraints - a rectangle must fit below height h when ph_h is true
        for h in range(lower_bound, upper_bound + 1):
            for i in range(n_rectangles):
                # Normal orientation
                rect_height = rectangles[i][1]
                if h >= rect_height:
                    hard_clauses.append([-variables[f"ph_{h}"], variables[f"r{i + 1}"], 
                                       variables[f"py{i + 1},{h - rect_height}"]])
                
                # Rotated orientation
                rotated_height = rectangles[i][0]
                if h >= rotated_height:
                    hard_clauses.append([-variables[f"ph_{h}"], -variables[f"r{i + 1}"], 
                                       variables[f"py{i + 1},{h - rotated_height}"]])
        
        # Prepare soft clauses with weights
        soft_clauses = []
        
        # THE KEY FIX: Use different soft clauses that directly minimize height
        # For each height, add a soft clause to minimize the height
        # We want ph_lower_bound to be true and ph_upper_bound to be false
        
        # 1. Add a soft clause for each height directly, with unit weights
        for h in range(lower_bound, upper_bound + 1):
            soft_clauses.append((1, [variables[f"ph_{h}"]]))  # We want ph_h to be TRUE
        
        # 2. Require at least one ph_h to be true with a hard clause (min feasible height)
        all_ph_vars = [variables[f"ph_{h}"] for h in range(lower_bound, upper_bound + 1)]
        hard_clauses.append(all_ph_vars)
        
        # Write WCNF header: p wcnf num_variables num_clauses top_weight
        top_weight = 2  # Any value > max soft clause weight (which is 1)
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
        rotations = [False for _ in range(n_rectangles)]
        
        if "OPTIMUM FOUND" in output:
            print("Optimal solution found!")
            # Extract the model line (starts with "v ")
            for line in output.split('\n'):
                if line.startswith('v '):
                    model_str = line[2:].strip()
                    model = list(map(int, model_str.split()))
                    
                    # Process the model to extract positions and rotations
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
                    
                    # Extract rotation variables
                    for i in range(n_rectangles):
                        if variables[f"r{i + 1}"] in true_vars:
                            rotations[i] = True
                    
                    # Extract positions - Find the exact transition point for each rectangle
                    for i in range(n_rectangles):
                        # Find x position (first position where px is true)
                        found_x = False
                        for e in range(width):
                            if variables[f"px{i + 1},{e}"] in true_vars:
                                if e == 0 or variables[f"px{i + 1},{e-1}"] not in true_vars:
                                    positions[i][0] = e
                                    found_x = True
                                    break
                        if not found_x:
                            print(f"WARNING: Could not determine x-position for rectangle {i}!")
                        
                        # Find y position (first position where py is true)
                        found_y = False
                        for y_pos in range(upper_bound):
                            if variables[f"py{i + 1},{y_pos}"] in true_vars:
                                if y_pos == 0 or variables[f"py{i + 1},{y_pos-1}"] not in true_vars:
                                    positions[i][1] = y_pos
                                    found_y = True
                                    break
                        if not found_y:
                            print(f"WARNING: Could not determine y-position for rectangle {i}!")
                    
                    # CRITICAL: Verify that all rectangles fit within the optimal height
                    actual_max_height = 0
                    for i in range(n_rectangles):
                        rect_height = rectangles[i][0] if rotations[i] else rectangles[i][1]
                        top_edge = positions[i][1] + rect_height
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
        return optimal_height, positions, rotations
    
    except Exception as e:
        print(f"Error running Max-SAT solver: {e}")
        if os.path.exists(wcnf_file):
            os.unlink(wcnf_file)
        return None, None, None

# Main program
if __name__ == "__main__":
    # Default instance number
    instance_number = 15
    
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
    widths = [int(rectangle[0]) for rectangle in rectangles]
    upper_bound = sum(heights)
    lower_bound = max(math.ceil(sum([int(rectangle[0] * rectangle[1]) for rectangle in rectangles]) / width), 
                      max(max(heights), max(widths)))
    
    print(f"Solving 2D Strip Packing with MaxSAT for instance {instance_number}")
    print(f"Width: {width}")
    print(f"Number of rectangles: {n_rec}")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    
    # Solve with MaxSAT
    optimal_height, optimal_pos, optimal_rot = SPP_MaxSAT(width, rectangles, lower_bound, upper_bound)
    
    stop = timeit.default_timer()
    print(f"Solve time: {stop - start:.2f} seconds")
    
    if optimal_height is not None:
        print(f"Optimal strip height: {optimal_height}")
        print("Rectangle positions (x, y) and rotation status:")
        for i, (x_pos, y_pos) in enumerate(optimal_pos):
            rot_status = "Rotated" if optimal_rot[i] else "Not Rotated"
            effective_w = rectangles[i][1] if optimal_rot[i] else rectangles[i][0]
            effective_h = rectangles[i][0] if optimal_rot[i] else rectangles[i][1]
            print(f"Rectangle {i}: ({x_pos}, {y_pos}) "
                f"width={effective_w}, height={effective_h}, {rot_status}")

        strip = [width, optimal_height]
        display_solution(strip, rectangles, optimal_pos, optimal_rot)
    else:
        print("Failed to find a solution.")