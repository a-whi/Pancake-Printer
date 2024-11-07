"""
Created by: Alex W
Last edited: 27/10/24

Document Purpose: To take an image and process it to be used by the printer.

Two images will be produced.
1. The edges of the image (a silhouette). This will be used for the image outline.
2. Black and white image. The printer will print the black parts within the image 
   then the white. 
"""
import cv2
import numpy as np

# Image file name, located in 'images' folder.
image_name = 'shocked_Pika.png'
# image_name = 'me_outline.png'

def main():
    
    img_processor()

    # Load the images
    edges_image = cv2.imread(f'./processed/edges_{image_name}', cv2.IMREAD_GRAYSCALE)
    grey_image = cv2.imread(f'./processed/thresholded_grey_{image_name}', cv2.IMREAD_GRAYSCALE)

    # Get contours for edges and greyscale images
    edge_contours = get_outline(edges_image)  # This will get the outer most edges (the images outline)
    inside_edges_contours = get_inside_contours(edges_image)
    # grey_contours = get_contours(grey_image)

    # Convert contours to paths for the printer
    edge_path = contours_to_paths(edge_contours)
    inside_paths = contours_to_paths(inside_edges_contours)
    # grey_paths = contours_to_paths(grey_contours)

    # Draw paths on images for visualization
    output_image = cv2.cvtColor(edges_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color display

    output_image = draw_paths(output_image, edge_path, color=(0, 0, 255))  # Draw edges in red (This is only needed as the outline for the printer)
    output_image = draw_paths(output_image, inside_paths, color=(0, 255, 0))   # Draws all contours

    # output_image = draw_paths(output_image, grey_paths, color=(0, 255, 0))   # Draw grey paths in green

    # Display and save the final image with drawn paths
    cv2.imshow('Paths Drawn', output_image)
    cv2.waitKey(0)  # Wait for keypress to close the window
    cv2.imwrite(f'./processed/final_paths_{image_name}', output_image)

    # Step 1: Create buffered mask around the outer contour
    buffered_mask = create_buffered_contour(grey_image, inside_edges_contours, buffer_size=10)

    # Step 2: Create fill paths within the buffered area
    fill_paths = create_fill_paths(grey_image, buffered_mask, line_spacing=15)

    # Display results
    cv2.imshow("Buffered Contour", buffered_mask)
    cv2.waitKey(0)
    cv2.imshow("Fill Paths", fill_paths)
    cv2.waitKey(0)

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

    # Pad the image to avoid detecting border as a contour
    padded_image = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    
    # Apply morphological closing with a larger kernel to close gaps
    kernel = np.ones((5, 5), np.uint8)
    refined_edges = cv2.morphologyEx(padded_image, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(image=refined_edges, threshold1=100, threshold2=200)
    cv2.imwrite(f'./processed/edges_{image_name}', edges)
    cv2.imshow('Refined_edges', edges)
    cv2.waitKey(0)  # Wait until a key is pressed

    # Create a greyscale thresholded layer for shading
    _, thresholded_grey = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'./processed/thresholded_grey_{image_name}', thresholded_grey)
    cv2.imshow('Thresholded Grey', thresholded_grey)
    cv2.waitKey(0)  # Wait until a key is pressed

    print('SUCCESS')

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

def get_inside_contours(image):
    """
    Function finds all contours within the image.
    """
    # Use RETR_TREE to get all contours, including nested ones
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Approximate contours to reduce point density (Smooths the lines)
    approx_contours = [cv2.approxPolyDP(cnt, epsilon=1.5, closed=True) for cnt in contours]
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

def create_buffered_contour(grey_image, contours, buffer_size=10):
    # Create a mask with the contour filled
    contour_mask = np.zeros_like(grey_image)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Create a buffered zone around the contour using dilation
    kernel = np.ones((buffer_size, buffer_size), np.uint8)
    buffered_mask = cv2.dilate(contour_mask, kernel, iterations=1)

    return buffered_mask

def create_fill_paths(image, buffered_mask, line_spacing=15):
    # Mask where filling can occur (inside the outer contour, minus the buffered zone)
    fill_area = cv2.bitwise_and(buffered_mask, buffered_mask, mask=255 - buffered_mask)

    # Create spaced lines within the fill area
    fill_image = np.zeros_like(image)
    for y in range(0, fill_image.shape[0], line_spacing):
        for x in range(fill_image.shape[1]):
            if fill_area[y, x] == 255:  # If within the allowed fill area
                fill_image[y, x] = (255, 255, 255)  # Mark path line
    return fill_image


main()