import numpy as np
import cv2
import matplotlib.pyplot as plt
def reproject_image_to_3D(image, depth_map, intrinsics, extrinsics):
    # Get the shape of the image
    height, width, _ = image.shape

    # Create pixel coordinates
    u, v = np.meshgrid(range(width), range(height))
    image_coords = np.stack([u, v, np.ones_like(u)], axis=2)

    # Flatten image coordinates
    image_coords_flat = image_coords.reshape(-1, 3)

    # Reshape depth map
    depth_map_flat = depth_map.reshape(-1)

    # Intrinsic camera matrix
    K = intrinsics

    # Inverse of intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Transform image coordinates to normalized coordinates
    normalized_coords = np.dot(K_inv, image_coords_flat.T).T

    # Apply depth information
    camera_coords = normalized_coords * depth_map_flat[:, np.newaxis]

    # Homogeneous coordinates
    homogeneous_coords = np.concatenate((camera_coords, np.ones((camera_coords.shape[0], 1))), axis=1)

    # Extrinsic matrix
    extrinsics_4x4 = np.concatenate((extrinsics, np.array([[0, 0, 0, 1]])), axis=0)

    # Apply extrinsic transformation
    transformed_coords = np.dot(homogeneous_coords, extrinsics_4x4.T)

    return transformed_coords
def reproject_to_3d(image, K, depth_map, ext):
    # Get image dimensions
    height, width = image.shape[:2]

    # Create array to store 3D points
    points_3d = np.zeros((height, width, 3), dtype=np.float32)

    # Calculate inverse camera matrix
    inv_camera_matrix = np.linalg.inv(K)

    # Reproject each pixel to 3D
    for y in range(height):
        for x in range(width):
            # Get depth value for the pixel
            depth = depth_map[y, x]

            # Calculate 3D coordinates using the camera matrix and depth value
            pixel = np.array([[x, y, 1]], dtype=np.float32)
            pixel_3d = depth * np.dot(inv_camera_matrix, pixel.T)
            points_3d[y, x] = pixel_3d[:, 0]

    return points_3d
def project_to_camera_plane(points_3d, camera_matrix):
    # Get image dimensions
    height, width = points_3d.shape[:2]

    # Create array to store projected 2D points
    points_2d = np.zeros((height, width, 2), dtype=np.float32)

    # Project 3D points to camera plane
    for y in range(height):
        for x in range(width):
            # Get 3D coordinates of the point
            point_3d = points_3d[y, x]

            # Convert 3D point to homogeneous coordinates
            point_3d_homogeneous = np.append(point_3d, 1)

            # Project 3D point to 2D using camera matrix
            point_2d_homogeneous = np.dot(camera_matrix, point_3d_homogeneous)
            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
            # Store the projected 2D point
            points_2d[y, x] = point_2d

    return points_2d


# def calculate_baseline(left_disparity_map, right_disparity_map, left_intrinsics):
#     # Get the focal length from the intrinsics matrix
#     focal_length = left_intrinsics[0, 0]
#
#     # Calculate the baseline
#     baseline = focal_length * np.abs(left_intrinsics[0, 2] - right_intrinsics[0, 2]) / np.mean(
#         left_disparity_map - right_disparity_map)
#
#     return baseline


## Main
############################################################################################
# Load the image, camera matrix, and depth map
image = cv2.imread('Data/example/im_left.jpg')
K = np.array([[576, 0, 511.5], [0, 576, 217.5], [0, 0, 1]], dtype=np.float32)
depth_map = np.loadtxt('Data/example/depth_left.txt',delimiter=',')
# Initialize an array of 11 translation values along the baseline
baseline = 0.1  # Baseline distance in meters
translation_values = np.linspace(0, 0.1, 11)

for i in range (11):
    # Call the function to reproject image coordinates to 3D space
    ext=np.array([[1, 0, 0,-0.01*i], [0, 1, 0,0], [0, 0, 1,0]], dtype=np.float32)
    camera_matrix =np.dot(K ,ext )
    points_3d = reproject_to_3d(image, K, depth_map,ext)
    # Call the function to project 3D points to the camera plane
    points_2d = project_to_camera_plane(points_3d, camera_matrix)

    # Create a blank image with the same size as the original image
    reprojected_image = np.zeros_like(image)

    # Copy pixel values from the original image to the reprojected image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if not(np.isnan(points_2d[y, x]).any()):
                # Get the projected 2D coordinates of the point
                point_2d = points_2d[y, x]

                # Round the 2D coordinates to the nearest pixel
                point_2d_rounded = np.round(point_2d).astype(int)
                if point_2d_rounded[1]< image.shape[0] and point_2d_rounded[1] >=0 and point_2d_rounded[0]< image.shape[1] and point_2d_rounded[0] >= 0:
                    # Copy the RGB pixel value from the original image to the reprojected image
                    reprojected_image[point_2d_rounded[1], point_2d_rounded[0]] = image[y, x]

    # Display the reprojected image
    plt.imshow(reprojected_image)
    plt.show()
    #image=reprojected_image
    cv2.imwrite(f"s{i+1}.jpg", reprojected_image)
    print(i+1,"/",11)
