# Camera-Pose-Estimation-And-Image-Stitching
From scratch code to estimate Camera Pose And generate panorama images without OpenCV libraries

# Overview

This repository/project explores building image processing algorithms like Hough Transform, and Homography from scratch. The Hough transform is a technique used in computer vision and image processing to detect geometric shapes, primarily lines or curves, in an image. The Hough transform works by identifying points in an image that belong to the same geometric shape, even if those points are not connected in the image. It achieves this by mapping the points in the image space to a parameter space, where each geometric shape is represented by a curve or surface. 

Specifically, the Hough transform technique in this project is utilized to detect the corners of a paper within a sequence of video frames. Following the identification of paper edges, the Hough transform computes lines in the Hough space or parameter space, utilizing the polar equation of a line. Subsequently, the corners of the paper are determined by identifying intersection points where the maximum number of lines intersect.    

Another task achieved through this project is Image Stitching through the Homography technique. Homography is a transformation used in computer vision and image processing to map points from one plane to another. In this specific context of image stitching, homography is utilized to align multiple images taken from different viewpoints into a single panoramic image. The process involves finding the homography matrix that describes the geometric transformation between two images, allowing for the seamless merging of their contents.

## Instructions for Problem1.py : 
1. Open the Problem1.py file from the code Folder using VSCode or any Python IDE. 
2. Before running the file please make sure the "project2.avi" file is in the same folder as this script.
3. Install necessary libraries such as OpenCV, NumPy, and Matplotlib.
4. The Video frame of the corner of the paper and edges are shown. The Final Plots will be shown on the screen once the video is finished running.

## Instructions for Problem2.py: 
1. Open the Problem2.py  file from the code Folder using VSCode or any Python IDE. 
2. Before running the file please make sure the image files are in the same folder as this script.
3. Install necessary libraries such as OpenCV, NumPy, and Matplotlib.
4. The plot of various images is shown on the console.
