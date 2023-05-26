# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import cv2
from project import disparity


# Example usage
def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except IOError:
        print(f"Error: Could not read the file '{file_path}'.")





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # Read the array from file
    # array = np.loadtxt("example/disp_right.txt", delimiter=',')
    #
    # # Convert the array to an image
    # imager = (array / np.max(array)) * 255
    # imager = imager.astype(np.uint8)
    #
    # # Read the array from file
    # array = np.loadtxt('example/disp_left.txt', delimiter=',')
    #
    # # Convert the array to an image
    # imagel = (array / np.max(array)) * 255
    # imagel = imagel.astype(np.uint8)
    # # Read the array from file
    # array = np.loadtxt('example/depth_right.txt', delimiter=',')
    #
    # # Convert the array to an image
    # imagedr = (array / np.max(array)) * 255
    # imagedr = imagedr.astype(np.uint8)
    # # Read the array from file
    # array = np.loadtxt('example/depth_left.txt', delimiter=',')
    #
    # # Convert the array to an image
    # imagedl = (array / np.max(array)) * 255
    # imagedl = imagedl.astype(np.uint8)
    #
    # # Save the image to disk
    #
    # cv2.imwrite("set_1/dis_left.jpg", imagel)
    # cv2.imwrite("set_1/dis_right.jpg", imager)
    # cv2.imwrite("set_1/depth_left.jpg", imagedl )
    # cv2.imwrite("set_1/depth_right.jpg", imagedr *255)


    # set1
    imageR = cv2.imread("set_1/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("set_1/im_left.jpg", cv2.IMREAD_GRAYSCALE)

    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('set_1/max_disp.txt'),kernel_size=41,path="set_1/")

    # set2
    imageR = cv2.imread("set_2/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("set_2/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('set_2/max_disp.txt'),kernel_size=42,path="set_2/")
    print(0)

    #set3
    imageR = cv2.imread("set_3/im_right.jpg" ,cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("set_3/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('set_3/max_disp.txt'),kernel_size=25,path="set_3/")



    # set4

    imageR = cv2.imread("set_4/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("set_4/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('set_4/max_disp.txt'),kernel_size=45,path="set_4/")



    # set5
    imageR = cv2.imread("set_5/im_right.jpg",cv2.IMREAD_GRAYSCALE)
    imageL = cv2.imread("set_5/im_left.jpg", cv2.IMREAD_GRAYSCALE)
    #example k=5 kernel=25
    #set 1 k=5\7 kernel=41
    disparity(imageL,imageR,k=5, file_content = read_file('set_5/max_disp.txt'),kernel_size=30,path="set_5/")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
