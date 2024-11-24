"""
Created by: Alex W
Created: 27/10/24

Document Purpose: To take an image and process it to be used by the printer.

Two images will be produced.
1. The edges of the image (a silhouette). This will be used for the image outline.
2. Black and white image. The printer will print the black parts within the image 
   then the white. 
"""
import cv2
import numpy as np
from shapely.geometry import LineString, Polygon, MultiLineString
from shapely.ops import unary_union

# Image file name, located in 'images' folder.
# image_name = 'shocked_Pika.png'
image_name = 'square.jpg'

def main():
    
    img_processor()
    print('img_processor Status: SUCCESS')

    # Load the images
    edges_image = cv2.imread(f'./processed/edges_{image_name}', cv2.IMREAD_GRAYSCALE)

    # Get contours for edges and greyscale images
    edge_contours = get_outline(edges_image)  # This will get the outer most edges (the images outline)
    print('Outline Status: SUCCESS')
    inside_edges_contours = get_inside_contours(edges_image)
    print('Detailing Edge Status: SUCCESS')

    # Convert contours to paths for the printer
    edge_path = contours_to_paths(edge_contours)
    print('contours_to_paths Status: SUCCESS')
    inside_paths = contours_to_paths(inside_edges_contours)
    print('contours_to_paths Status: SUCCESS')


    print(edge_path)
    print('AAAAAAAAAAA')
    print(inside_paths)
######
    output_image = cv2.cvtColor(edges_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color display
    output_image = draw_paths(output_image, edge_path, color=(0, 0, 255)) 
    cv2.imshow('edge path', output_image)
    cv2.waitKey(0)  # Wait for keypress to close the window

    output_image = draw_paths(output_image, inside_paths, color=(0, 255, 0)) 
    cv2.imshow('inside path', output_image)
    cv2.waitKey(0)  # Wait for keypress to close the window
#####

    # print('AAAAAAAAAA')
    # print(len(inside_paths))
    # print(len(inside_paths))

    # # Fill the image
    # fill_paths = generate_fill_paths(edge_path, inside_paths, spacing=5.0, angle=0, pattern="zigzag")
    # # Can inside paths have the edge path as well? cause currently it does
    # fill_paths = optimise_fill_paths(fill_paths, start_point=(0,0)) # Should we change starting point

    # cv2.imshow("Buffered Contour", fill_paths)
    # cv2.waitKey(0)


#### Testing visualiser
    # # Draw paths on images for visualization
    # output_image = cv2.cvtColor(edges_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color display

    # output_image = draw_paths(output_image, edge_path, color=(0, 0, 255))  # Draw edges in red (This is only needed as the outline for the printer)
    # output_image = draw_paths(output_image, inside_paths, color=(0, 255, 0))   # Draws all contours
    # output_image = draw_paths(output_image, fill_paths, color=(255, 0, 0))   # Draws all contours

    # # output_image = draw_paths(output_image, grey_paths, color=(0, 255, 0))   # Draw grey paths in green

    # # Display and save the final image with drawn paths
    # cv2.imshow('Paths Drawn', output_image)
    # cv2.waitKey(0)  # Wait for keypress to close the window
    # cv2.imwrite(f'./processed/final_paths_{image_name}', output_image)

# ### IDK what this part is, dont think it works
#     # Step 1: Create buffered mask around the outer contour
#     buffered_mask = create_buffered_contour(grey_image, inside_edges_contours, buffer_size=10)

#     # Step 2: Create fill paths within the buffered area
#     fill_paths = create_fill_paths(grey_image, buffered_mask, line_spacing=15)

#     # Display results
#     cv2.imshow("Buffered Contour", buffered_mask)
#     cv2.waitKey(0)
#     cv2.imshow("Fill Paths", fill_paths)
#     cv2.waitKey(0)

    cv2.destroyAllWindows()

####################################################################################################

def img_processor():
    """
    This function takes an image and applies various transformation to it.
    1. Flips the image: The image will be flipped as when printed onto the pan its image will be flipped
    2. Converts the image to grey: This helps with finding the edges and identiying the empty space between the edges
    3. Padding & Morphological closing: Helps identiy edges
    
    Next we find the edges of the image and also the white spaces inbetween these edges.
    All images are saved to the 'processed' folder where they can be accessed later for other functions
    """

    print(f'Processing {image_name} ...')

    img = cv2.imread(f'./images/{image_name}', cv2.IMREAD_COLOR)    # Open image file
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)  # Wait until a key is pressed

    img = cv2.flip(img, 1)  # Flip horizontally
    # Convert BGR image to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pad the image for morphological closing, but crop it out before edge detection
    padded_image = cv2.copyMakeBorder(grey, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    
    # Apply morphological closing with a larger kernel to close gaps
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(padded_image, cv2.MORPH_CLOSE, kernel)

    # Remove the padding
    refined_image = closed[10:-10, 10:-10]

    edges = cv2.Canny(refined_image, threshold1=100, threshold2=200)
    cv2.imwrite(f'./processed/edges_{image_name}', edges)
    cv2.imshow('Refined_edges', edges)
    cv2.waitKey(0)  # Wait until a key is pressed

    # Create a greyscale thresholded layer for shading
    _, thresholded_grey = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'./processed/thresholded_grey_{image_name}', thresholded_grey)
    cv2.imshow('Thresholded Grey', thresholded_grey)
    cv2.waitKey(0)  # Wait until a key is pressed

####################################################################################################

"""
Now we turn the image edges into a path that can be printed
"""

def get_outline(image):
    """
    This will return the outer most contour whuch will act as the outline for the image and stop
    anything being printed outside of this line.
    """
    
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are no contours found, return an empty list
    if not contours:
        return []

    # Get the image center
    image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])

    # Find the contour with the maximum distance from the center
    max_distance = 0
    outermost_contour = None
    for cnt in contours:
        # Calculate the maximum distance of points in the contour from the center
        distances = [np.linalg.norm(point[0] - image_center) for point in cnt]
        max_contour_distance = max(distances)
        
        # Update the outermost contour if this one has a greater max distance
        if max_contour_distance > max_distance:
            max_distance = max_contour_distance
            outermost_contour = cnt

    # Approximate the outermost contour to reduce point density (smooths the lines)
    outline = cv2.approxPolyDP(outermost_contour, epsilon=1.5, closed=True)

    return [outline] 

# def get_inside_contours(image):
#     """
#     Function finds all contours within the image.
#     """
#     # Use RETR_TREE to get all contours, including nested ones
#     contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # Approximate contours to reduce point density (Smooths the lines)
#     approx_contours = [cv2.approxPolyDP(cnt, epsilon=1.5, closed=True) for cnt in contours]

#     return approx_contours
def get_inside_contours(image):
    """
    Function finds all contours within the image except the outermost contour.
    """
    # Use RETR_TREE to get all contours, including nested ones
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the image center
    image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])

    # Find the outermost contour by identifying the contour with the maximum distance from the center
    max_distance = 0
    outermost_contour = None
    for cnt in contours:
        distances = [np.linalg.norm(point[0] - image_center) for point in cnt]
        max_contour_distance = max(distances)
        if max_contour_distance > max_distance:
            max_distance = max_contour_distance
            outermost_contour = cnt

    # Filter out the outermost contour
    inside_contours = [cnt for cnt in contours if cnt is not outermost_contour]

    # Approximate contours to reduce point density (smooths the lines)
    approx_contours = [cv2.approxPolyDP(cnt, epsilon=1.5, closed=True) for cnt in inside_contours]

    return approx_contours


def contours_to_paths(contours):
    """
    Function converts contours to a path of X & Y values
    """
    paths = []
    for cnt in contours:
        path = []
        for point in cnt:
            x, y = point[0]  # Extract x, y coordinates
            path.append((x, y))
        paths.append(path)

    # Save paths in a text file
    with open(f'./paths/XY_path_{image_name}.txt', "w") as pathtxt:
        # Loop over each path (contour) in paths
        for path in paths:
            # Loop over each coordinate pair in the current path
            for x, y in path:
                # Write each coordinate as a string in "x y" format, followed by a newline
                pathtxt.write(f"{x} {y}\n")
            # Optionally, add a newline between contours to separate paths visually
            pathtxt.write("\n")
    
    return paths

# # Unsure if needed
# def export_paths(paths, layer_name):
#     """
#     Function to convert contours to G-code or printer-compatible paths
#     # Example output format for printer (can be modified to G-code)
#     """
#     print(f"\nLayer: {layer_name}")
#     for path in paths:
#         print("New Path:")
#         for x, y in path:
#             print(f"Move to ({x}, {y})")
####################################################################################################
"""Fill object with paths"""

def generate_fill_paths(outline_path, detail_paths, spacing=5.0, angle=0, pattern="zigzag"):
    """
    Generate fill paths for an object with outline and detail paths.
    
    Parameters:
    outline_path: List of (x,y) points defining the outer boundary
    detail_paths: List of Lists of (x,y) points defining internal details
    spacing: Distance between fill lines
    angle: Angle of fill lines in degrees
    pattern: "lines" or "zigzag"
    
    Returns:
    List of fill paths as (x,y) point pairs
    """
    # Convert the outline_path format for Shapely
    # Unpack the (x,y) tuples into separate coordinates
    outline_coords = [(x, y) for x, y in outline_path[0]]  # Note the [0] since outline_path is a list of path
    # Make sure the polygon is closed (first and last points match)
    if outline_coords[0] != outline_coords[-1]:
        outline_coords.append(outline_coords[0])
    # # Convert outline to Shapely polygon
    # outline_polygon = Polygon(outline_path)
        # Convert outline to Shapely polygon
    try:
        outline_polygon = Polygon(outline_coords)
    except Exception as e:
        print("Error creating polygon:", e)
        print("Outline coordinates:", outline_coords)
        return []
    
    # # Convert detail paths to Shapely linestrings
    # detail_lines = [LineString(path) for path in detail_paths]
    detail_lines = []
    for detail_path in detail_paths:
        try:
            # Convert each detail path to correct format
            detail_coords = [(x, y) for x, y in detail_path]
            detail_lines.append(LineString(detail_coords))
        except Exception as e:
            print(f"Error creating detail line: {e}")
            continue
    
    # Get bounding box
    minx, miny, maxx, maxy = outline_polygon.bounds
    
    # Calculate rotated bounding box for angled lines
    theta = np.radians(angle)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    
    # Generate parallel lines across the bounding box
    line_segments = []
    current_y = miny
    
    while current_y <= maxy:
        # Create line at current_y
        line_start = np.array([minx, current_y])
        line_end = np.array([maxx, current_y])
        
        # Rotate line if angle specified
        if angle != 0:
            line_start = rot_matrix @ line_start
            line_end = rot_matrix @ line_end
        
        line = LineString([tuple(line_start), tuple(line_end)])
        
        # Intersect with outline polygon
        if line.intersects(outline_polygon):
            intersection = line.intersection(outline_polygon)
            
            # Handle multiple intersection segments
            if isinstance(intersection, MultiLineString):
                segments = list(intersection.geoms)
            else:
                segments = [intersection]
            
            # For each segment, check if it intersects detail paths
            for segment in segments:
                valid_segment = True
                for detail in detail_lines:
                    if segment.intersects(detail):
                        valid_segment = False
                        break
                
                if valid_segment:
                    line_segments.append(list(segment.coords))
        
        current_y += spacing

    # Convert to zigzag pattern if requested
    if pattern == "zigzag":
        zigzag_paths = []
        for i in range(0, len(line_segments) - 1, 2):
            # Get current and next line segment
            current_line = line_segments[i]
            if i + 1 < len(line_segments):
                next_line = line_segments[i + 1]
                # Reverse every other line to create continuous path
                next_line = next_line[::-1]
                # Combine into single zigzag path
                zigzag_path = current_line + next_line
                zigzag_paths.append(zigzag_path)
            else:
                # Handle odd number of lines
                zigzag_paths.append(current_line)
        return zigzag_paths
    
    return line_segments

def optimise_fill_paths(fill_paths, start_point=(0,0)):
    """
    Optimise the order of fill paths to minimize travel distance.
    
    Parameters:
    fill_paths: List of fill paths
    start_point: Starting point for optimisation
    
    Returns:
    Optimised list of fill paths
    """
    if not fill_paths:
        return []
    
    remaining_paths = fill_paths.copy()
    optimised_paths = []
    current_point = start_point
    
    while remaining_paths:
        # Find closest path to current point
        min_dist = float('inf')
        closest_path = None
        closest_path_idx = None
        should_reverse = False
        
        for i, path in enumerate(remaining_paths):
            # Check distance to path start
            dist_to_start = np.sqrt((path[0][0] - current_point[0])**2 + 
                                  (path[0][1] - current_point[1])**2)
            if dist_to_start < min_dist:
                min_dist = dist_to_start
                closest_path = path
                closest_path_idx = i
                should_reverse = False
            
            # Check distance to path end
            dist_to_end = np.sqrt((path[-1][0] - current_point[0])**2 + 
                                (path[-1][1] - current_point[1])**2)
            if dist_to_end < min_dist:
                min_dist = dist_to_end
                closest_path = path
                closest_path_idx = i
                should_reverse = True
        
        # Add closest path to optimized paths
        if should_reverse:
            closest_path = closest_path[::-1]
        optimised_paths.append(closest_path)
        current_point = closest_path[-1]
        
        # Remove path from remaining paths
        remaining_paths.pop(closest_path_idx)
    
    return optimised_paths


####################################################################################################
"""ALL TESTING STUFF"""
def draw_paths(image, paths, color=(0, 255, 0), thickness=2):
    """
    Function used for testing to draw paths on an image, so I can see everything is working correctly
    """
    for path in paths:
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            cv2.line(image, start_point, end_point, color, thickness)
    return image

# def create_buffered_contour(grey_image, contours, buffer_size=10):
#     # Create a mask with the contour filled
#     contour_mask = np.zeros_like(grey_image)
#     cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)

#     # Create a buffered zone around the contour using dilation
#     kernel = np.ones((buffer_size, buffer_size), np.uint8)
#     buffered_mask = cv2.dilate(contour_mask, kernel, iterations=1)

#     return buffered_mask

# def create_fill_paths(image, buffered_mask, line_spacing=15):
#     # Mask where filling can occur (inside the outer contour, minus the buffered zone)
#     fill_area = cv2.bitwise_and(buffered_mask, buffered_mask, mask=255 - buffered_mask)

#     # Create spaced lines within the fill area
#     fill_image = np.zeros_like(image)
#     for y in range(0, fill_image.shape[0], line_spacing):
#         for x in range(fill_image.shape[1]):
#             if fill_area[y, x] == 255:  # If within the allowed fill area
#                 fill_image[y, x] = (255, 255, 255)  # Mark path line
#     return fill_image


main()