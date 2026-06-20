import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_data(max_pictures=20):
    """
    helper function to load data, loads in both color and grey scale since we are working with both
    in other functions. 
    """
    pic_path_lst = []

    dataset_path = "dataset"
    dir_list = os.listdir(dataset_path)

    count = 0
    for folder in dir_list:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            #ignore DS files
            continue

        for filename in sorted(os.listdir(folder_path)):
            if filename.startswith("."):
                continue

            if count >= max_pictures:
                #break out of but folder
                break

            pic_path_lst.append(os.path.join(folder_path, filename))
            count += 1

        if count >= max_pictures:
            #break out of dataset folder
            break

    

    grey_lst = []
    color_lst = []

    for path in pic_path_lst:
        #save all the paths in both colors
        img_grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        grey_lst.append(img_grey)
        color_lst.append(img_color)

    return grey_lst, color_lst


def grayscale_custom_transformation(img, x, y):
    """
    function used in assignment 3 - preprocessing

    https://numpy.org/doc/2.3/reference/generated/numpy.clip.html
    """

    a = np.percentile(img, x)
    b = np.percentile(img, 100-y)
    im = np.clip(img, a, b)
        #found this powerful function on link above

    im = (im - a) / (b - a) * 255
        #formula from lecuture slides
    im = np.uint8(im)
        #type error

    
    return im

def circle(im):

    im_eq = cv2.equalizeHist(im)
    blurred = cv2.GaussianBlur(im_eq, (7,7), 0)
    blurred1 = grayscale_custom_transformation(blurred,10,0)


    circles = cv2.HoughCircles(
            blurred1,                  
            cv2.HOUGH_GRADIENT,      
            dp=2,                  
            minDist=500,              
            param1=100,                
            param2=50,                
            minRadius=300,            
            maxRadius=600             
        )
    output = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  
        #need colour to draw circle
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            r = int(r * 0.69)  
                #shrink by 31%
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)  
                #green circle
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  
                #red dot


    return (x,y), r

def align_to_mean(datapoints, images_lst, radiuses):
    """
    second part of preprocessing, find a mean center so we can align 
    all iris'es
    """
    mean_point = np.mean(datapoints, axis=0)
    align_lst = []

    for img, ref_point, r in zip(images_lst, datapoints, radiuses):
        dx = mean_point[0] - ref_point[0]
        dy = mean_point[1] - ref_point[1]
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
            #build a tranlation matrix so we can move the images to where they belong, 3x2 so nothing else changes.
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        aligned_color = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)

        
        cv2.circle(aligned_color, (int(mean_point[0]), int(mean_point[1])), int(r), (0, 255, 0), 2)
            #draw cirle for vizualtion.

        align_lst.append(aligned_color)



    return align_lst,mean_point
        #we will need mean points for later potensially



def mask_circle(img, center, radius):
    """
    Sets all pixels outside a circle to fill_outside.
    Works for grayscale or color images I'm not sure which one I'm going to use yet. 
    """

    white = 255

    if len(img.shape) == 2:
        #grayscale
        result = np.full_like(img, white)
        mask = np.zeros_like(img, dtype=np.uint8)
            #dtype to avoid crashes
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
        result[mask == 255] = img[mask == 255]

        iris_pixels = img[mask == 255]
        mean_val = np.mean(iris_pixels)
        std_val = np.std(iris_pixels)

    else:
        #color img
        result = np.full_like(img, white)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
            #dtype to avoid crashes
        cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
        for c in range(3):
            #clip for all dimentions 
            result[:,:,c][mask==255] = img[:,:,c][mask==255]

        iris_pixels = img[mask == 255]
        mean_val = np.mean(iris_pixels)
        std_val = np.std(iris_pixels)

    



    return result,mean_val,std_val



def preprocess_data(max_pictures = 20):
    """
    funtion that puts all the preprocessing functions into one. 
    """

    grey_data,color_data = load_data(max_pictures)
    datapoints = []
    radiuses = []

    for img in grey_data:
        output,r = circle(img)
        datapoints.append(output)
        radiuses.append(r)

    align_lst,mean_point = align_to_mean(datapoints,grey_data,radiuses)
    radiuses = np.array(radiuses)
        #turn into np array to get mean, this got better results.
    masked_img_lst = []
    mean_lst = []
    std_lst = []

    for _ in range(len(color_data)):
        masked_image,mean,std = mask_circle(color_data[_], mean_point, radiuses.mean())
        masked_img_lst.append(masked_image)
        mean_lst.append(mean)
        std_lst.append(std)

    return masked_img_lst,mean_lst,std_lst




if __name__ == "__main__":

    
    masked_img_lst,_,_ = preprocess_data(8)

    plt.figure(figsize=(12, 6))
    for i, img in enumerate(masked_img_lst):
        plt.subplot(2, 4, i + 1) 
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Image {i+1}')
    plt.tight_layout()
    plt.show()

    #NOTE: I had to run in terminal to make importing of pictures work, instead of in VSCODE
