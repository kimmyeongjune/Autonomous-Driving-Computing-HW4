import sys

import numpy as np
import cv2
from matplotlib import pyplot as plt
from m2bk import *

dataset_handler = DatasetHandler()

image = dataset_handler.images[30]

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image, cmap='gray')
plt.show()

image_rgb = dataset_handler.images_rgb[30]

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image_rgb)
plt.show()

i = 30
depth = dataset_handler.depth_maps[i]

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(depth, cmap='jet')
plt.show()

dataset_handler.k

print(dataset_handler.num_frames)

i = 30
image = dataset_handler.images[i]

plt.figure(figsize=(8, 6), dpi=100)
plt.imshow(image, cmap='gray')
plt.show()

def extract_features(image):
    """
    Find keypoints and descriptors for the image

    Arguments:
    image -- a grayscale image

    Returns:
    kp -- list of the extracted keypoints (features) in an image
    des -- list of the keypoint descriptors in an image
    """
    ### START CODE HERE ### 
    img = image
    # Initiate ORB detector
    orb = cv.ORB_create()
    # Find the keypoints and descriptors with ORB
    kp, des = orb.detectAndCompute(image, None)

    ### END CODE HERE ###
    return kp, des

i = 1
image = dataset_handler.images[i]
kp, des = extract_features(image)
print("Number of features detected in frame {0}: {1}\n".format(i, len(kp)))

print("Coordinates of the first keypoint in frame {0}: {1}".format(i, str(kp[0].pt)))

def visualize_features(image, kp):
    """
    Visualize extracted features in the image

    Arguments:
    image -- a grayscale image
    kp -- list of the extracted keypoints

    Returns:
    """
    display = cv2.drawKeypoints(image, kp, None)
    plt.imshow(display)
    plt.show()
    
i = 50
image = dataset_handler.images_rgb[i]

visualize_features(image, kp)

def extract_features_dataset(images, extract_features_function):
    """
    Find keypoints and descriptors for each image in the dataset

    Arguments:
    images -- a list of grayscale images
    extract_features_function -- a function which finds features (keypoints and descriptors) for an image

    Returns:
    kp_list -- a list of keypoints for each image in images
    des_list -- a list of descriptors for each image in images
    
    """
    kp_list = []
    des_list = []
    
    for image in images:
        kp, des = extract_features_function(image)
        kp_list.append(kp)
        des_list.append(des)

    
    return kp_list, des_list


images = dataset_handler.images
kp_list, des_list = extract_features_dataset(images, extract_features)

i = 5
print("Number of features detected in frame {0}: {1}".format(i, len(kp_list[i])))
print("Coordinates of the first keypoint in frame {0}: {1}\n".format(i, str(kp_list[i][0].pt)))

# Remember that the length of the returned by dataset_handler lists should be the same as the length of the image array
print("Length of images array: {0}".format(len(images)))

def match_features(des1, des2):
    """
    Match features from two images

    Arguments:
    des1 -- list of the keypoint descriptors in the first image
    des2 -- list of the keypoint descriptors in the second image

    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor
    """
    ### START CODE HERE ###
    # Create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck= True)

    # Match descriptors.
    match = bf.match(des1, des2)

    # Sort them in the order of their distance.
    match = sorted(match, key = lambda x:x.distance)

    ### END CODE HERE ###
    
    return match

i = 0 
des1 = des_list[i]
des2 = des_list[i+1]

match = match_features(des1, des2)
print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(match)))

# Optional
def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches

    Arguments:
    match -- list of matched features from two images
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_match -- list of good matches, satisfying the distance threshold
    """
    filtered_match = []
    
    ### START CODE HERE ###
    for data in match:
        if data.distance <= dist_threshold:
            filtered_match.append(data)
    
    ### END CODE HERE ###

    return filtered_match

# Optional
i = 0 
des1 = des_list[i]
des2 = des_list[i+1]
match = match_features(des1, des2)

dist_threshold = 10
filtered_match = filter_matches_distance(match, dist_threshold)

print("Number of features matched in frames {0} and {1} after filtering by distance: {2}".format(i, i+1, len(filtered_match)))

def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    plt.show()
    
    
n = 20
filtering = False

i = 0 
image1 = dataset_handler.images[i]
image2 = dataset_handler.images[i+1]

kp1 = kp_list[i]
kp2 = kp_list[i+1]

des1 = des_list[i]
des2 = des_list[i+1]

match = match_features(des1, des2)
if filtering:
    dist_threshold = 10
    match = filter_matches_distance(match, dist_threshold)

image_matches = visualize_matches(image1, kp1, image2, kp2, match[:n]) 

def match_features_dataset(des_list, match_features):
    """
    Match features for each subsequent image pair in the dataset

    Arguments:
    des_list -- a list of descriptors for each image in the dataset
    match_features -- a function which maches features between a pair of images

    Returns:
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
               
    """
    matches = []
    
    ### START CODE HERE ###
    for i in range(len(des_list) - 1):
        match = match_features(des_list[i], des_list[i + 1])
        matches.append(match)

    
    ### END CODE HERE ###
    
    return matches

matches = match_features_dataset(des_list, match_features)

i = 5
print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(matches[i])))

def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    """
    Filter matched features by distance for each subsequent image pair in the dataset

    Arguments:
    filter_matches_distance -- a function which filters matched features from two images by distance between the best matches
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    dist_threshold -- maximum allowed relative distance between the best matches, (0.0, 1.0) 

    Returns:
    filtered_matches -- list of good matches for each subsequent image pair in the dataset. 
                        Each matches[i] is a list of good matches, satisfying the distance threshold
               
    """
    filtered_matches = []
    
    ### START CODE HERE ###
    for match in matches:
        filtered_match = filter_matches_distance(match, dist_threshold)
        filtered_matches.append(filtered_match)
    ### END CODE HERE ###
    
    return filtered_matches

dist_threshold = 10

filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)

if len(filtered_matches) > 0:
    
    # Make sure that this variable is set to True if you want to use filtered matches further in your assignment
    is_main_filtered_m = False
    if is_main_filtered_m: 
        matches = filtered_matches

    i = 0
    print("Number of filtered matches in frames {0} and {1}: {2}".format(i, i+1, len(filtered_matches[i])))
    

def estimate_motion(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 
    
    Optional arguments:
    depth1 -- a depth map of the first frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
    image2_points -- a list of selected match coordinates in the second image. image1_points[i] = [u, v], where u and v are 
                     coordinates of the i-th match in the image coordinate system
               
    """
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    
    ### START CODE HERE ###
    for m in match:
        image1_points.append(kp1[m.queryIdx].pt)
        image2_points.append(kp2[m.trainIdx].pt)

    image1_points = np.array(image1_points, dtype=np.float32)
    image2_points = np.array(image2_points, dtype=np.float32)
    if len(image1_points) < 8:
        raise ValueError("Not enough points to compute Essential Matrix.")
    
    # Estimate the Essential matrix
    E, mask = cv2.findEssentialMat(image1_points, image2_points, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise ValueError("Essential Matrix computation failed.")
    # Decompose the Essential matrix to obtain rotation and translation
    image1_points = image1_points[mask.ravel() == 1]
    image2_points = image2_points[mask.ravel() == 1]
    if len(image1_points) < 8:
        raise ValueError("Not enough points to compute Essential Matrix.")
    ret, rmat, tvec, mask_pose = cv2.recoverPose(E, image1_points, image2_points, k)
    
    ### END CODE HERE ###
    
    return rmat, tvec, image1_points, image2_points

i = 0
match = matches[i]
kp1 = kp_list[i]
kp2 = kp_list[i+1]
k = dataset_handler.k
depth = dataset_handler.depth_maps[i]

rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth1=depth)

print("Estimated rotation:\n {0}".format(rmat))
print("Estimated translation:\n {0}".format(tvec))

def estimate_motion_depth(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames using depth information.

    Arguments:
    match -- list of matched features from the pair of images
    kp1 -- list of the keypoints in the first image
    kp2 -- list of the keypoints in the second image
    k -- camera calibration matrix 
    depth1 -- a depth map of the first frame. Used to compute 3D points for PnP estimation

    Returns:
    rmat -- recovered 3x3 rotation numpy matrix
    tvec -- recovered 3x1 translation numpy vector
    image1_points -- a list of selected match coordinates in the first image
    image2_points -- a list of selected match coordinates in the second image
    """
    if depth1 is None:
        raise ValueError("Depth map (depth1) is required for this function.")

    image1_points = []
    image2_points = []
    object_points = []

    # 매칭된 점들에서 3D 포인트와 2D 이미지 좌표 추출
    for m in match:
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt

        # 깊이 값 확인
        if int(v1) >= depth1.shape[0] or int(u1) >= depth1.shape[1]:
            continue  # 좌표가 depth1 범위를 벗어나면 건너뜀
        s = depth1[int(v1), int(u1)]
        
        # 깊이 값이 유효한 범위 내에 있는 경우만 처리
        if 0 < s < 1000:  # 깊이 값이 0보다 크고 1000 이하인 경우
            # 3D 포인트 계산 (픽셀 좌표를 카메라 좌표로 변환)
            X = (u1 - k[0, 2]) * s / k[0, 0]
            Y = (v1 - k[1, 2]) * s / k[1, 1]
            Z = s
            object_points.append([X, Y, Z])

            # 2D 이미지 좌표 저장
            image1_points.append([u1, v1])
            image2_points.append([u2, v2])

    object_points = np.array(object_points, dtype=np.float32)
    image1_points = np.array(image1_points, dtype=np.float32)
    image2_points = np.array(image2_points, dtype=np.float32)

    # PnP 알고리즘을 사용하여 모션 추정
    if len(object_points) > 0:
        _, rvec, tvec, _ = cv2.solvePnPRansac(object_points, image2_points, k, None)
        rmat, _ = cv2.Rodrigues(rvec)
    else:
        # 깊이 값이 없는 경우, Essential Matrix 기반으로 회전 및 이동 추정
        E, mask = cv2.findEssentialMat(image1_points, image2_points, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, rmat, tvec, _ = cv2.recoverPose(E, image1_points, image2_points, k)

    return rmat, tvec, image1_points, image2_points
i = 0
match = matches[i]
kp1 = kp_list[i]
kp2 = kp_list[i+1]
k = dataset_handler.k
depth = dataset_handler.depth_maps[i]

rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth1=depth)

print("Estimated rotation:\n {0}".format(rmat))
print("Estimated translation:\n {0}".format(tvec))

i=30
image1  = dataset_handler.images_rgb[i]
image2 = dataset_handler.images_rgb[i + 1]

image_move = visualize_camera_movement(image1, image1_points, image2, image2_points)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)
plt.show()

image_move = visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=True)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)
plt.show()
# These visualizations might be helpful for understanding the quality of image points selected for the camera motion estimation

def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    """
    Estimate complete camera trajectory from subsequent image pairs

    Arguments:
    estimate_motion -- a function which estimates camera motion from a pair of subsequent image frames
    matches -- list of matches for each subsequent image pair in the dataset. 
               Each matches[i] is a list of matched features from images i and i + 1
    des_list -- a list of keypoints for each image in the dataset
    k -- camera calibration matrix 
    
    Optional arguments:
    depth_maps -- a list of depth maps for each frame. This argument is not needed if you use Essential Matrix Decomposition

    Returns:
    trajectory -- a 3xlen numpy array of the camera locations, where len is the lenght of the list of images and   
                  trajectory[:, i] is a 3x1 numpy vector, such as:
                  
                  trajectory[:, i][0] - is X coordinate of the i-th location
                  trajectory[:, i][1] - is Y coordinate of the i-th location
                  trajectory[:, i][2] - is Z coordinate of the i-th location
                  
                  * Consider that the origin of your trajectory cordinate system is located at the camera position 
                  when the first image (the one with index 0) was taken. The first camera location (index = 0) is geven 
                  at the initialization of this function

    """
    trajectory = [np.zeros((3, 1))]  # Start with the origin point as a list element
    current_position = np.zeros((3, 1))  # Initial position at the origin
    
    ### START CODE HERE ###
    for i in range(len(matches)):
        # estimate the rmat, tvec
        rmat, tvec, im1_p, im2_p = estimate_motion_depth(matches[i], kp_list[i], kp_list[i+1], k, depth_maps[i])
        # Update current position by applying the rotation and translation
        current_position += tvec
        n_position = np.dot(rmat, current_position)
        
        # Append the new position to the trajectory list
        trajectory.append(n_position)
    
    # Convert trajectory to a numpy array of shape (3, len)
    trajectory = np.hstack(trajectory)
        
        
    ### END CODE HERE ###
    
    return trajectory

depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)

i = 1
print("Camera location in point {0} is: \n {1}\n".format(i, trajectory[:, [i]]))

# Remember that the length of the returned by trajectory should be the same as the length of the image array
print("Length of trajectory: {0}".format(trajectory.shape[1]))

# Note: Make sure to uncomment the below line if you modified the original data in any ways
#dataset_handler = DatasetHandler()


# Part 1. Features Extraction
images = dataset_handler.images
kp_list, des_list = extract_features_dataset(images, extract_features)


# Part II. Feature Matching
matches = match_features_dataset(des_list, match_features)

# Set to True if you want to use filtered matches or False otherwise
is_main_filtered_m = True
if is_main_filtered_m:
    dist_threshold = 100
    filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)
    matches = filtered_matches

    
# Part III. Trajectory Estimation
depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps)


#!!! Make sure you don't modify the output in any way
# Print Submission Info
print("Trajectory X:\n {0}".format(trajectory[0,:].reshape((1,-1))))
print("Trajectory Y:\n {0}".format(trajectory[1,:].reshape((1,-1))))
print("Trajectory Z:\n {0}".format(trajectory[2,:].reshape((1,-1))))


visualize_trajectory(trajectory)

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

locX, locY, locZ = [], [], []

# Set up the figure
fig = plt.figure(figsize=(8, 6), dpi=100)
gspec = gridspec.GridSpec(3, 3)
ZY_plt = plt.subplot(gspec[0, 1:])
YX_plt = plt.subplot(gspec[1:, 0])
traj_main_plt = plt.subplot(gspec[1:, 1:])
D3_plt = plt.subplot(gspec[0, 0], projection='3d')

# Determine plot limits
max_value = np.max(trajectory)
min_value = np.min(trajectory)
maxY, minY = max_value, min_value

def animate(i):
    current_pos = trajectory[:, i]
    print(f"Frame {i}: {current_pos}")  # 프레임 번호와 현재 포지션 출력
    locX.append(current_pos[0])
    locY.append(current_pos[1])
    locZ.append(current_pos[2])

    traj_main_plt.clear()
    ZY_plt.clear()
    YX_plt.clear()
    D3_plt.clear()

    # Main trajectory plot (Z, X)
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=1.06)
    traj_main_plt.plot(locZ, locX, ".-", label="Trajectory", zorder=1, linewidth=1, markersize=4)
    traj_main_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    traj_main_plt.set_xlim([min_value, max_value])
    traj_main_plt.set_ylim([min_value, max_value])
    traj_main_plt.set_xlabel("Z")
    traj_main_plt.legend(loc=1, title="Legend", borderaxespad=0., fontsize="medium", frameon=True)

    # ZY plot
    ZY_plt.plot(locZ, locY, ".-", linewidth=1, markersize=4, zorder=0)
    ZY_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    ZY_plt.set_xlim([min_value, max_value])
    ZY_plt.set_ylim([minY, maxY])
    ZY_plt.set_ylabel("Y")
    ZY_plt.axes.xaxis.set_ticklabels([])

    # YX plot
    YX_plt.plot(locY, locX, ".-", linewidth=1, markersize=4, zorder=0)
    YX_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([min_value, max_value])
    YX_plt.set_xlabel("Y")
    YX_plt.set_ylabel("X")

    # 3D plot
    D3_plt.plot3D(locX, locZ, locY, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(min_value, max_value)
    D3_plt.set_ylim3d(min_value, max_value)
    D3_plt.set_zlim3d(min_value, max_value)
    D3_plt.set_xlabel("X", labelpad=0)
    D3_plt.set_ylabel("Z", labelpad=0)
    D3_plt.set_zlabel("Y", labelpad=-2)
    D3_plt.view_init(45, azim=30)

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=trajectory.shape[1], interval=100, repeat=True)
plt.tight_layout()
plt.show()