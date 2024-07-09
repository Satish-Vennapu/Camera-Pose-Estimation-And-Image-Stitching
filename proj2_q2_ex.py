import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

plt.rcParams['figure.figsize'] = [8, 8]

# function to Read the image and convert to gray
def read_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
    img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

# function to resize the stitched image to original size 
def resize_img(image1, image2):
    height1, width1, channel1 =image1.shape
    height2, width2, channel2 =image2.shape
    if height1 != height2 or width1 != width2:
        image1_resized =cv2.resize(image1, (width2, height2))
    else:
        image1_resized = image1.copy()
    return image1_resized

def resize_gray(image1, image2):
    height, width = image2.shape[:2]
    resized_image = cv2.resize(image1, (width, height), interpolation= cv2.INTER_LINEAR)
    return resized_image

# function for SIFT algorithm
def SIFT(img):
    siftDetector= cv2.SIFT_create() 
    keypoint, descriptor = siftDetector.detectAndCompute(img, None)
    return keypoint, descriptor

# function for plotting the keypoints on the image
def plot_sift(gray, rgb, keypoint):
    tmp = rgb.copy()
    img = cv2.drawKeypoints(gray, keypoint, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

# Function for matching the keypoints in the images
def matcher(keypoint1, descriptor1, img1, keypoint2, descriptor2, img2, threshold):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1,descriptor2, k=2)

    # selecting the better one
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(keypoint1[pair[0].queryIdx].pt + keypoint2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches

#function for plotting the keypoints with lines on the images
def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) 
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.5)

    plt.show()

# Function for finding the homography
def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] 
    return H

# Function for selecting random point
def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

# Function for catching the errors
def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        # set index 2 to 1 and slice the index 0, 1
        estimate_p2[i] = (temp/temp[2])[0:2] 
    
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2
    return errors

# Function for ransac algorithm
def ransac(matches, threshold, iters):
    num_best_inliers = 0
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        #  for avoiding division by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers or matches found: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

# Function for stitching the images
def stitch_img(left, right, H):
    print("Stitching the images together wait!! ...")
    
    # Converting to double and applying normalization to avoid noise
    left = cv2.normalize(left.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    right = cv2.normalize(right.astype('float'), None, 
                            0.0, 1.0, cv2.NORM_MINMAX)   
    
    # For the left image
    height_l, width_l, channel_l = left.shape
    corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Getting new height and width
    height_new = int(round(abs(y_min) + height_l))
    width_new = int(round(abs(x_min) + width_l))
    size = (width_new, height_new)

    # For the right image
    warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
    height_r, width_r, channel_r = right.shape
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)

    warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)

    # Creating black pixel
    black = np.zeros(3)  
    # Procedure for Stitching and storing results in warped_r and warped_l
    for i in range(warped_r.shape[0]):
        for j in range(warped_r.shape[1]):
            pixel_l = warped_l[i, j, :]
            pixel_r = warped_r[i, j, :]
            
            if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_l
            elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = pixel_r
            elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
                warped_l[i, j, :] = (pixel_l + pixel_r) / 2
            else:
                pass
                  
    stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
    return stitch_image

##############################################################################################################
left_gray, left_origin, left_rgb = read_image("image_1.jpg")
mid1_gray, mid1_origin, mid1_rgb = read_image("image_2.jpg")
mid2_gray, mid2_origin, mid2_rgb = read_image("image_3.jpg")
right_gray, right_origin, right_rgb = read_image("image_4.jpg")

keypoint1, descriptor1 = SIFT(left_gray)
keypoint2, descriptor2 = SIFT(mid1_gray)
keypoint3, descriptor3 = SIFT(mid2_gray)
keypoint4, descriptor4 = SIFT(right_gray)

keypoint1_img = plot_sift(left_gray, left_rgb, keypoint1)
keypoint2_img = plot_sift(mid1_gray, mid1_rgb, keypoint2)
keypoint3_img = plot_sift(mid2_gray, mid2_rgb, keypoint3)
keypoint4_img = plot_sift(right_gray, right_rgb, keypoint4)
total_keypoint = np.concatenate((keypoint1_img, keypoint2_img), axis=1)
plt.imshow(total_keypoint)

matches1 = matcher(keypoint1, descriptor1, left_rgb, keypoint2, descriptor2, mid1_rgb, 0.5)

total_img = np.concatenate((left_rgb, mid1_rgb), axis=1)
plot_matches(matches1, total_img)

inliers, H = ransac(matches1, 0.5, 2000)
plot_matches(inliers, total_img)

stitch1 = stitch_img(left_rgb, mid1_rgb, H)
plt.imshow(stitch1)

stitch1_resized = resize_img(stitch1, mid2_rgb)
plt.imshow(stitch1_resized)
plt.axis('off')

plt.savefig('First_stitched_resized.png', dpi = 200, bbox_inches = 'tight', pad_inches =0)
############################################################################################################
left_gray, left_origin, left_rgb = read_image('First_stitched_resized.png')
plt.imshow(left_rgb)
left_rgb = resize_img(left_rgb, mid2_rgb)
left_gray = resize_gray(left_gray, mid2_gray)
left_origin = resize_img(left_origin,mid2_origin)

keypoint1, descriptor1 = SIFT(left_gray)
keypoint3, descriptor3 = SIFT(mid2_gray)
keypoint1_img = plot_sift(left_gray, left_rgb, keypoint1)
total_keypoint = np.concatenate((keypoint1_img, keypoint2_img), axis=1)
plt.imshow(total_keypoint)

matches2 = matcher(keypoint1, descriptor1, left_rgb, keypoint3, descriptor3, mid2_rgb, 0.5)
total_img = np.concatenate((left_rgb, mid2_rgb), axis=1)
plot_matches(matches2, total_img)

inliers, H = ransac(matches2, 0.5, 2000)
plot_matches(inliers, total_img)

stitch2 = stitch_img(left_rgb, mid2_rgb, H)
plt.imshow(stitch2)

stitch2_resized = resize_img(stitch2, right_rgb)
plt.imshow(stitch2_resized)
plt.axis('off')

plt.savefig('Second_stitched_resized.png', dpi = 200, bbox_inches = 'tight', pad_inches =0)

##################################################################################################################

left_gray, left_origin, left_rgb = read_image('Second_stitched_resized.png')
plt.imshow(left_rgb)
left_rgb = resize_img(left_rgb, right_rgb)
left_gray = resize_gray(left_gray, right_gray)
left_origin = resize_img(left_origin,right_origin)

keypoint1, descriptor1 = SIFT(left_gray)
keypoint4, descriptor4 = SIFT(right_gray)
keypoint1_img = plot_sift(left_gray, left_rgb, keypoint1)
total_keypoint = np.concatenate((keypoint1_img, keypoint4_img), axis=1)
plt.imshow(total_keypoint)

matches3 = matcher(keypoint1, descriptor1, left_rgb, keypoint4, descriptor4, right_rgb, 0.5)
total_img = np.concatenate((left_rgb, right_rgb), axis=1)
plot_matches(matches3, total_img)

inliers, H = ransac(matches3, 0.5, 2000)
plot_matches(inliers, total_img)

# Final stitched Panorama
stitch3 = stitch_img(left_rgb, right_rgb, H)
plt.imshow(stitch3)
plt.show()

# Cropping the Panorama
x, y = 1300, 10
final_img = stitch3[x:, y:]
plt.imshow(final_img)
plt.show()