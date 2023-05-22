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


imageR = cv2.imread("Data/set_1/im_right.jpg",cv2.IMREAD_GRAYSCALE)
imageL = cv2.imread("Data/set_1/im_right.jpg", cv2.IMREAD_GRAYSCALE)

k=5
disparity(imageL,imageR,k, file_content = read_file('Data/set_1/max_disp.txt'))


     # matrix = np.array([[1, 2, 3, 4],
    #                    [5, 6, 7, 8],
    #                    [9, 10, 11, 12]])
    #
    # k = 3

    #
    #
    # print(k_values)
    # #
    # # הגדרת המטריצה
    # matrix =  np.array([[688.000061035156, 0,511.5],
    #                    [0, 688.000061035156, 217.5]])
    # matrix[:,:]*=2
    # print(matrix)
    #
    # # כפל איבר-איבר בין התמונה ובין המטריצה
    # effective_matrix = image.astype(np.float32)
    # for c in range(3):
    #     d=c//2
    #     x=c%2
    #     effective_matrix[:, :, c] *= matrix[d, c]
    #     print(d,x,c)
    #
    # # הצגת התוצאה
    # effective_matrix = effective_matrix.astype(np.uint8)
    # cv2.imshow('Result', effective_matrix)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
