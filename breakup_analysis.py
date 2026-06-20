import cv2
import numpy as np
import matplotlib.pyplot as plt
from breakup_analyzer import preprocess_data
import math


def iris_detection(data,min_treshold= 0,max_treshold=35,min_area=2500,max_area= 6000):
    """
    function that takes a preprocessed image and return dark spots detected within a 
    small region using countours. It is crutial to parameter tune so the model best fits
    your dataset. 

    PARAMETERS:
    min_treshold: min trshold for canny detection
    max_treshold: max treshold for canny detection

    min_area: min area for countour to return
    max_area: max area for countour to return
    """
    green = [img[:, :, 1] for img in data]
        #extract green channel

    equalized_images = []

    for im in green:
        #equalize images
        im_eq = cv2.equalizeHist(im)
        equalized_images.append(im_eq)



    edges_masked_list = []


    for im_eq in equalized_images:
        
        blurred = cv2.GaussianBlur(im_eq, (7, 7), 0)
            #Gaussian blur to remove noise and random low values


        _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            #optimal treshold using OTSU like we learned in class
        
        kernel = np.ones((5, 5), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            #open and close like we learned in class 8 part 1

        
        dark_only = cv2.bitwise_and(im_eq, dark_mask)
            #simplify the exercise

    
        edges_masked = cv2.Canny(dark_only, min_treshold, max_treshold)
            #one last canny to improve results
        edges_masked_list.append(edges_masked)

    mask = []

    for edges in edges_masked_list:
        #find contours of the detected edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        filtered_mask = np.zeros_like(edges)
            #create an empty mask to copy over contour area

        for cnt in contours:
            #draw contours within the given area
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                #make sure we have the right sizes
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1) 
                    #fill in for the contours with the right requirements



        mask.append(filtered_mask)



    return mask


if __name__ == "__main__":
    """
    plot
    """
    masked_images, mean, std = preprocess_data(20)

    mask = iris_detection(masked_images)

    lst = []

    for i in range(len(mask)):
            mask_bool = mask[i] == 255

            result = masked_images[i].copy()
            result[mask_bool] = 255
            lst.append(result)

    num_images = len(lst)
    cols = 4 
    rows = math.ceil(num_images / cols)

    plt.figure(figsize=(16, 8))
    for i, img in enumerate(lst):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Aligned Image {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    #NOTE: I had to run in terminal to make importing of pictures work, instead of in VSCODE
