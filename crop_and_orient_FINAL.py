# Name: crop_and_orient_FINAL.py
# Author: Gaspard Clerc
# Date: 23.08.2025

# Importation of the relevant module:

import cv2
import os
import numpy as np

#%%
def white_balance(img):
    # helper function to correct perform white balance correctur on the image
    result = img.copy().astype(np.float32)

    # Compute the average of each channel
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Scale channels to make their averages equal to gray average
    result[:, :, 0] *= avg_gray / avg_b
    result[:, :, 1] *= avg_gray / avg_g
    result[:, :, 2] *= avg_gray / avg_r

    # Clip and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


#%%
def crop_and_orient_card(image_path, output_path, thres, debug=False):
    # function to identify the contour of the card, crop it, correct the distortion
    # the value thres, can be adjusted in case the crop is not following the contour of the card
    # the debug function allow to visualize or not the results. To be use in the case the crop is not performing well
    
    """
    Function to identify the contour of the card, crop it and correct the distortion
    Save the image in the output_path

    Args:
        image_path (str): Path to the input images.
        output_path (str): Path to the folder where the processed images will be saved.
        thres (float): Can be adjusted to improve the contour detection
        debug (binary): Allow to visualize or not the intermediary program steps.
    """    
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if image is None:
        print(f"Error: Could not load image from path: {image_path}")
        return
    
    original_image = image.copy()   # Keep a copy for contour visualization
    image = white_balance(image)
    error = 0                       #error will be = 1 if error is detected
    #transform the image in hsv_color format to better recognize the contour
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    
    #apply mask to isolate card colour
    lower_val = np.array([0, 0, 0])
    upper_val = np.array([40, 255, 255])
    mask = cv2.inRange(hsv_image, lower_val, upper_val)


    # Apply GaussianBlur to reduce noise and detail
    blurred = cv2.GaussianBlur(mask, (95, 95), 0)
    # Apply adaptive thresholding for better edge detection (optional)
    _, thresh = cv2.threshold(blurred, thres, 255, cv2.THRESH_BINARY)
    # Apply Canny edge detection
    edges = cv2.Canny(thresh, 30, 150)
    
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort the contours by area in descending order and keep the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    # create empty contour to store all the found vector
    card_contour = None
    
    # Loop over the contours to find a rectangular contour
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the approximated contour has four points, we assume it's the card
        if len(approx) == 4:
            card_contour = approx
            break
    
    if debug:
        # Visualize the contours
        debug_image = original_image.copy()
        cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)  # Draw all contours
        
        if card_contour is not None:
            # Highlight the detected card contour in blue
            cv2.drawContours(debug_image, [card_contour], -1, (255, 0, 0), 5)
        
        # Display the image with contours for debugging
        imC = cv2.resize(debug_image, (960, 540))
        cv2.imshow("Contours", imC)
        imE = cv2.resize(edges, (960, 540))
        cv2.imshow("Edges", imE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if card_contour is None:
        print("Could not find a rectangular contour that looks like a card.")
        error = 1
        return error

    # Reorder the points of the contour to top-left, top-right, bottom-right, bottom-left
    def reorder_points(pts):
        pts = pts.reshape((4, 2))
        rect = np.zeros((4, 2), dtype="float32")
        
        # The top-left point will have the smallest sum, bottom-right will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # The top-right will have the smallest difference, bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    rect = reorder_points(card_contour)
    
    # Compute the width and height of the card
    widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Ensure the card's long side is horizontal
    if maxHeight > maxWidth:
        # If the height is greater than the width, we swap the points
        rect = np.array([rect[1], rect[2], rect[3], rect[0]])

    # Set a fixed aspect ratio (width:height ~ 1.586)
    aspect_ratio = 1.586
    fixed_width = 800  #  choose any fixed width
    fixed_height = int(fixed_width / aspect_ratio)

    # Define the destination points for the perspective transform with the fixed aspect ratio
    dst = np.array([
        [0, 0],
        [fixed_width - 1, 0],
        [fixed_width - 1, fixed_height - 1],
        [0, fixed_height - 1]
    ], dtype="float32")
    
    # Apply the perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (fixed_width, fixed_height))
    
    # resize the image, Save the cropped and oriented image
    target_size = 500
    resized_img = cv2.resize(warped, (target_size, target_size))
    cv2.imwrite(output_path, resized_img)
    print(f"Credit card image saved to {output_path}")
    #if no mistake was detected return error = 0
    return error
#%%
def batch_process_images(input_folder, output_folder):
    """
    Goes through each file in the input folder, applies the
    your_processing_script, and saves the result in the output folder.

    Args:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where the processed images will be saved.
    """
    counter = 1
    err = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        name, ext = os.path.splitext(filename)
        output_filename = f"{counter}{ext.lower()}"  # You can customize the output filename
        output_path = os.path.join(output_folder, output_filename)
        error = crop_and_orient_card(input_path, output_path, 20, debug=False)
        err = err + error
        counter += 1

    return counter, err

#%%
if __name__ == "__main__":
    #input_directory = "C:/Users/GaspardClerc/OneDrive - SWS AG/Dokumente/Python/Dendro/IMAGES/CARD_04_HR"
    input_directory = "C:/Users/GaspardClerc/SWS AG/SharePoint - Data/03_Operations/Innovation/Innovation_Pipeline/Stage_1/Dendro/120"
    output_directory = "C:/Users/GaspardClerc/OneDrive - SWS AG/Dokumente/Python/Dendro/IMAGES/C120"



    counter, err = batch_process_images(input_directory, output_directory)

    print(f"Batch processing complete.{10-err}/10")
