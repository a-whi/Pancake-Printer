"""
Created by: Alex W
Last edited: 07/11/24

Document Purpose: To convert the image paths into G-Code.

1. Read text file created from image processing
2. Convert to G-Code, and save
"""
def converter():
    # Open the text file and read it
    txtfile = open('./paths/XY_path_shocked_Pika.png.txt', "r")
    lines = txtfile.readlines()

#####
    # Define pan dimensions
    MAX_X = 200  # Replace with your max X dimension in mm
    MAX_Y = 150  # Replace with your max Y dimension in mm
#####

    # Find max X & Y values in the XY path txt file and scale to within the pan dimesions
    x_values = []
    y_values = []
    for line in lines:
        line = line.strip()
        if line:
            x, y = map(float, line.split())
            x_values.append(x)
            y_values.append(y)

    # Find the bounding box of the image
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    # Calculate the width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Calculate the scaling factor to fit within MAX_X and MAX_Y
    scale_factor = min(MAX_X / width, MAX_Y / height)


    #Convert the XY coords to G-Code
    with open(f'./paths/gcode_path_shocked_Pika.cnc', "w") as pathtxt:
        pathtxt.write('G21 G17 G90\n')  # G21-units(mm), G17-XY plane, G90-Absolute mode
        for i in range(len(lines)):
            if i == 0:
                next_line = lines[i + 1].strip()
                x, y = next_line.split()
                # Apply scaling and offset to fit within boundaries
                x = (float(x) - min_x) * scale_factor
                y = (float(y) - min_y) * scale_factor
                pathtxt.write(f'G00 X{x:.2f} Y{y:.2f}\n')

            line = lines[i].strip()

            # If new paths reposition printer head
            if line == "":
                # Move to the new start point on the next non-blank line
                if i + 1 < len(lines):      #For the last line this may need to be i+1 <len(lines)-1 , not sure yet
                    next_line = lines[i + 1].strip()
                    if next_line:
                        x, y = next_line.split()
                        # Apply scaling and offset to fit within boundaries
                        x = (float(x) - min_x) * scale_factor
                        y = (float(y) - min_y) * scale_factor
                        pathtxt.write(f'G00 X{x:.2f} Y{y:.2f}\n')
                continue
            else:
                x, y = line.split()
                # Apply scaling and offset
                x = (float(x) - min_x) * scale_factor
                y = (float(y) - min_y) * scale_factor
                pathtxt.write(f'G01 X{x:.2f} Y{y:.2f}\n')

        pathtxt.write('G28\n') 


converter()
#####
# If we need curves in the images
    # gcode = []
    # for i in range(len(path) - 1):
    #     p1 = path[i]
    #     p2 = path[i+1]
        
    #     # If the points are close enough, treat it as a straight line
    #     if is_straight_line(p1, p2):
    #         gcode.append(f"G01 X{p2[0]} Y{p2[1]}")
    #     else:
    #         # If the points form a curve, find the center and radius for an arc
    #         arc = fit_arc(p1, p2, path[i+2] if i+2 < len(path) else None) 
    #         gcode.append(f"G02 X{arc['end'][0]} Y{arc['end'][1]} I{arc['center'][0]} J{arc['center'][1]}")
    # return gcode
#####
"""
# Notes for G Code

G21 = mm (units)

G00 = Move to start position
G28 = Return home

G01 = Move in straight line
G02 = Circular Interpolation Clockwise
G03 = Circular Interpolation Counterclockwise

# To set the workspace
G17 = XY plane
G18 = XZ plane
G19 = YZ plane

G90 = Absolute mode (Moves to exact coords) <--- what we want
G91 = Relative mode (Continuously increments the coords)

Bonus stuff:
M104 = Start extruder heating
M109 = Wait until extruder reaches T0
M140 = Start bed heating
M190 = Wait until bed reaches T0
M106 = Set fan speed
"""