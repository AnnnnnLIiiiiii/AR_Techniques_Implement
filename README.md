# AR_Techniques_Implement
Using Python and OpenCV to implement some statndard AR related techniques

This project will focus on detecting a custom AR Tag (a form of fiducial marker), that is used for obtaining a point of reference in the real world, such as in augmented reality applications.

There are two aspects to using an AR Tag - detection and tracking. Both of them are implemented in this project. The detection stage involves finding the AR Tag from a given image sequence, while the tracking stage involves keeping the tag in “view” throughout the sequence and performing image processing operations based on the tag’s orientation and position (a.k.a. the pose). The end goal is to implement these techniques on videos.

Detection:

Using a custom AR Tag image (ref_marker.png) to be used as reference. This tag encodes both the orientation as well as the ID of the tag.


Encoding Scheme:

In order to properly use the tag, it is necessary to understand how the data is encoded in the tag. Consider the refrence marker:

• The tag can be decomposed into an 8 × 8 grid of squares, which includes a padding of 2 squares width along the borders. This allows easy detection of the tag when placed on white background.

• The inner 4 × 4 grid (i.e. after removing the padding) has the orientation depicted by a white square in the lower-right corner. This represents the upright position of the tag.

• Fianlly, the inner-most 2 × 2 grid (i.e. after removing the padding and the orientation grids) encodes the binary representation of the tag’s ID, which is ordered in the clockwise direction from least significant bit to most significant. So, the top-left square is the least significant bit, and the bottom-left square is the most significant bit.

In this project, cv2.goodFeaturedetection based on Shi-Tomasi method is used as the corner detector algorithm 


Tracking

Once you have the four corners of the tag, we can perform homography estimation on this in order to perform some image processing operations, such as superimposing an image over the tag. The image Lena.png file is used as the template image.

• The first step is to compute the homography between the corners of the template and the four corners of the tag.

• Then, transform the template image onto the tag, such that the tag is “replaced” with the template.


Placing a virtual cube on the tag

Augmented reality applications generally place 3D objects onto the real world, which maintain three dimensional properties such as rotation and scaling as you move around the “object” with your camera. In this part of the project, you will attempt to implement a simple version of the same, by placing a 3D cube on the tag. This is the process of “projecting” a 3D shape onto a 2D image. The “cube” is a simple structure made up of 8 points and lines joining them. It is also an easy shape to implement as practice.

• First, compute the homography between the world coordinates (the reference AR tag) and the image plane (the tag in the image sequence).

• Build the projection matrix from the camera calibration matrix provided and the homography matrix.

• Assuming that the virtual cube is sitting on “top” of the marker, and that the Z axis is negative in the upwards direction, we will be able to obtain the coordinates of the other four corners of the cube.

• This allows us to now transform all the corners of the cube onto the image plane using the projection matrix
