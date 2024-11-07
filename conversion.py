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
    print(lines)
    print(len(lines))
    print('HEllo')

    with open(f'./paths/gcode_path_shocked_Pika.cnc', "w") as pathtxt:
        for i in range(len(lines)):
            if i =='\n': #if there is a blank line
                # pathtxt.write
                continue

            else:
                for x, y in lines[i]:
                # Write each coordinate as a string in "x y" format, followed by a newline
                    pathtxt.write(f"{x} {y}\n")


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

converter()





# Notes for G Code
"""
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