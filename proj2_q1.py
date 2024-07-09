import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cap = cv.VideoCapture('project2.avi')

while cap.isOpened():
    
    # reading the frame
    ret, frame = cap.read()
    if not ret:
        break
    # converting the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Canny edge detection algorithm
    edges = cv.Canny(gray, 200, 255,apertureSize=3)

    imgplot = plt.imshow(edges,interpolation='Nearest',cmap='gray')
    plt.show()
    print(edges)
    rect= edges[0:720,:]
    plt.imshow(rect,interpolation='Nearest',cmap='gray')
    plt.show()

    mask = np.zeros(edges.shape[:2], np.uint8)
    mask[0:720,:] = 255
    # Applying bitwise_and operator
    masked_img = cv.bitwise_and(edges,edges,mask = mask)

    plt.figure()
    plt.imshow(cv.cvtColor(masked_img, cv.COLOR_BGR2RGB))

    # Creating a Gaussian blur filter and appling to the edges
    kernel_size = (5, 5)
    sigma = 1.0
    blur = cv.GaussianBlur(masked_img, kernel_size, sigma)
    imgplot = plt.imshow(blur,interpolation='Nearest',cmap='gray')
    plt.show()

    # Pixels distance
    rho_resolution = 1  
    #Rotation  in radians
    theta_resolution = np.pi / 180  

    # Hough space accumulator
    height, width = edges.shape
    max_distance = int(np.sqrt(height**2 + width**2))
    rho_range = np.arange(-max_distance, max_distance + 1, rho_resolution)
    theta_range = np.arange(-np.pi/2, np.pi/2 + theta_resolution, theta_resolution)
    hough_space = np.zeros((len(rho_range), len(theta_range)), dtype=np.uint64)

    hough_space_1 = np.zeros((4, 4), dtype=np.uint64)


    # Performing Hough transform on the frames
    for y, x in np.argwhere(edges):
        for i, theta in enumerate(theta_range):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            j = int(rho / rho_resolution) + len(rho_range) // 2
            hough_space[j, i] += 1

    threshold=np.max(hough_space)

    # Finding the lines in the Hough space
    lines = []
    line_cord=[]
    lines1=[]
    print(hough_space)


    for j, i in np.argwhere(hough_space >= 0.7*threshold):
        initial_rho=j
        initial_theta=i
        rho = (j - len(rho_range) // 2) * rho_resolution
        theta = theta_range[i]

        t1 = np.cos(theta)
        t2 = np.sin(theta)
       
        x0 = t1 * rho
        y0 = t2 * rho
        x1 = int(x0 + max_distance * (-t2))
        y1 = int(y0 + max_distance * (t1))
        x2 = int(x0 - max_distance * (-t2))
        y2 = int(y0 - max_distance * (t1))
        lines.append((rho, theta, x1, y1, x2, y2))
        line_cord.append((initial_rho,initial_theta))


    # Drawing lines 
    img_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    for rho, theta, x1, y1, x2, y2 in lines:
        cv.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # results
    cv.imshow('Edge image', edges)
    cv.imshow('Hough space', cv.resize(np.uint8(hough_space / hough_space.max() * 255), None, fx=4, fy=4))
    cv.imshow('Detected lines', img_color)

    cv.waitKey(0)
    cv.destroyAllWindows()

    vert=[]

    def find_vertex(line1,line2):
        x11=line1[2]
        y11=line1[3]
        x12=line1[4]
        y12=line1[5]
        x21=line2[2]
        y21=line2[3]
        x22=line2[4]
        y22=line2[5]
        m1=(y12-y11)/(x12-x11)
        c1=y11-(m1*x11)
        m2=(y22-y21)/(x22-x21)
        c2=y21-(m2*x21)
        x=(c2-c1)/(m1-m2)
        y=m1*x11+c1
        return x,y

    for i in range(len(lines)):
        for j in range(i+1,len(lines)-i):
            x,y=find_vertex(lines[i],lines[j])
            vert.append((x,y))
    final_vert=[]
    for x,y in vert:
        if(x in range(1000,2000) and y in range(1100,1150)):
            final_vert.append((x,y))

    print(vert)
    verti=[]
    plt.scatter(*zip(*vert))
    plt.show()

    #multipling the intrinsic values of the camera like the caliberation matrix 
    # extrinsic values like world coordinates in mm
    world_coord=([0,0],[2160,0],[2160,2790],[0,2790])
    if(len(vert)>=4):
        for count in range(4):
            verti.append((vert[count]))
    else:
       continue

    # Function for converting to cartesian coordinates
    def Convert_to_slope_intercept_form(lines):
        new_lines=[]
        for i in range(len(lines)):
            y2=lines[i][5]
            y1=lines[i][3]
            x2=lines[i][4]
            x1=lines[i][2]
            m=(y2-y1)/(x2-x1)
            c=y1-m*x1
            new_lines.append((m,c))
        return new_lines
    new_lines=Convert_to_slope_intercept_form(lines)   
    print(new_lines)

    def find_vertex1(line1,line2):
        c1=line1[1]
        c2=line2[1]
        m1=line1[0]
        m2=line2[0]
        x=(c1-c2)/(m2-m1)
        y=m1*x+c1
        return x,y

    vertices=[]

    for i in range(len(new_lines)):
            for j in range(i+1,len(new_lines)):
                if abs(new_lines[i][0]-new_lines[j][0])<0.8 :
                    continue
                x,y=find_vertex1(new_lines[i],new_lines[j])
                vertices.append([x,y])
    print(vertices)

    # Function for computing homography
    def compute_homography(src_points, dst_points):
        A = []
        for src, dst in zip(src_points, dst_points):
            x, y = src
            xp, yp = dst
            A.append([x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp])
            A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp])
        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1, :].reshape((3, 3))
        return H / H[2, 2]
    
    # Function for computing homography with ransac
    def compute_homography_ransac(src_points, dst_points, num_iterations=500, threshold=5):
        best_H = None
        best_num_inliers = 0

        for i in range(num_iterations):
            # Choosing random 4 points
            indices = np.random.choice(len(src_points), size=4, replace=False)
            src_subset = src_points[indices]
            dst_subset = dst_points[indices]

            # Computing the homography matrix
            H = compute_homography(src_subset, dst_subset)

            # Applying the homography to all points 
            src_homogeneous = np.concatenate((src_points, np.ones((len(src_points), 1))), axis=1)
            dst_homogeneous = np.dot(H, src_homogeneous.T).T
            dst_normalized = dst_homogeneous[:, :2] / dst_homogeneous[:, 2:]

            # Counting the number of inliers
            distances = np.linalg.norm(dst_normalized - dst_points, axis=1)
            num_inliers = np.sum(distances < threshold)

            # Updating the best homography matrix if necessary
            if num_inliers > best_num_inliers:
                best_H = H
                best_num_inliers = num_inliers
       
        # Refining the homography matrix using all inlier points
        inlier_indices = np.where(distances < threshold)[0]
        inlier_src_points = src_points[inlier_indices]
        inlier_dst_points = dst_points[inlier_indices]
        best_H = compute_homography(inlier_src_points, inlier_dst_points)

        return best_H


    def cross_prod(a, b):
        result = [a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]]

        return result


    world_coord=np.array([[0,0],[2160,0],[2160,2790],[0,2790]])
  
    K=np.array([[1.38E+03,0,9.46E+02],[0,1.38E+03,5.27E+02],[0,0,1]])
    npvertices=np.array([[vertices[0][0],vertices[0][1],1],
                        [vertices[1][0],vertices[1][1],1],
                        [vertices[2][0],vertices[2][1],1],
                        [vertices[3][0],vertices[3][1],1]])
    
 
    npvertices=npvertices.T
    vertex_cam=K@npvertices
    vertex_cam=vertex_cam.T
    cam_coord=vertex_cam[:,0:2]
    print(cam_coord)

    H=compute_homography_ransac(world_coord,cam_coord)
    print (H)

    r1 = [H[0][0], H[1][0], H[2][0]]
    r2 = [H[0][1], H[1][1], H[2][1]]
    r3 = [H[0][2], H[1][2], H[2][2]]
    s = math.sqrt((H[0][0])**2 + (H[1][0])**2 + (H[2][0])**2)
    r1 = [a/s for a in r1]
    r2 = [a/s for a in r2]
    r3 = cross_prod(r1, r2)
    R = [r1, r2, r3]
    print("\n")
    print("Homography matrix: ")
    print (R)
    print("\n")
    pitch = math.atan2(-R[2][0], math.sqrt(R[2][1]**2 + R[2][2]**2))
    yaw = math.atan2(R[2][1], R[2][2])
    roll = math.atan2(R[1][0], R[0][0])
    null_space = np.linalg.svd(H)[2][-1, :]

    # Applying Normalization to the null space to obtain the translation vector
    t = null_space / np.linalg.norm(null_space)
    print("\n")
    print(f"Pitch={pitch}\nYaw={yaw}\nRoll={roll}\nTranslation={t}")
    print("\n")
    cap.release()
    cv.destroyAllWindows()