# Traffic Lane Tracking

This is a project from the Udacity Self-Driving Car Nanodegree that builds a pipeline to detect and track traffic lanes on the highway

<a href="http://www.youtube.com/watch?feature=player_embedded&v=wRnFrW5-yrg
" target="_blank"><img src="http://img.youtube.com/vi/wRnFrW5-yrg/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

## Camera Calibration
Before diving into lane detection, we have to calibrate the camera to undistort the images.
A set of chessboard images were provided in order to perform camera calibration.

For each chessboard image, the corners (the point where any four squares on the chessboard share) were extracted in terms of pixel positions using opencv function
```corners = cv2.findChessboardCorners```

A list of world coordinates of the corners were also created, called object points. The camera matrix can then be calculated from the corners and objpoints using opencv function ```cv2.calibrateCamera```.

The Camera class in Camera.py provided a nice interface to calibrate the camera.

```
# Directory that contains chessboard images
cal_dir = '../camera_cal/'

# Create a camera object
camera = Camera()

# Calibrate the camera
camera.Calibrate(cal_dir, (9, 6), filetype='jpg')

# Undistort an image
distorted_img = cv2.imread('test_img.jpg')
undistorted_img = camera.UndistortImage(distorted_img)
```
![alt text][undistort_chessboard]

# Detection Pipeline
After the camera has been calibrated, the pipeline for finding traffic lane on an image is as follow:
- Undistort the image using the camera matrix
- Transform the view of the road into bird's eye view perspective
- Find lanes
- Fit polynomial
- Measure curvature radius
- Transform back to original view for visualization

## Undistort Image
After camera calibration, the input image first has to be undistorted
```
undistorted_img = camera.UndistortImage(distorted_img)
```
![alt text][undistort_img]


## Bird's Eye View Transform
Since we want to measure the curvature of the lane, we have to analyze the image from bird's eye view perspective to get accurate measurement.
There are also a lot of other stuffs in the image beside traffic lanes (trees, other cars, the sky), we want the bird's eye view image to only have the view of the street.

This involves picking 4 points that cover the current lane form the original image and 4 points in the output image in which where we want the lane to be. The output image would look like the following
An assumption I made while picking the 4 points is that the camera is always in the center of the lanes.
![alt text][bev_tf]

## Find Lanes
This step extracts the lanes from the bird's eye view image. Since HSV image provides channels with characteristics that are invariant to lighting condition, I converted RGB to HSV for better robustness.

### Gradient on V channel
The V channel provides color intensity regardless of lighting, therefore I computed the gradient using Sobel operations on this channel to detect edges. I thresholded the magnitude and direction of the gradient to isolate the traffic lanes as much as possible.

### Gradient S channel
Similar to the V channel, I thresholded the magnitude and direction of the sobel gradient on this channel

### Threshold on raw H channel
The H channel provides the color value range, since the lanes are either white or yellow, I thresholded the h channel to keep pixel that are in this range.

### Gradient on gray image
During experimenting, I observed that gray image provides some pretty good responses on finding lanes. I thresholded the magnitude of the gradient on the gray image

At this point we have 4 binary channels that find lanes, I added these channels together. For each pixel in the summation, if the pixel passed 2 out of 4 filters, the pixel is kept.

The binary summation output image was returned.

![alt text][fine_lane]

## Fit Polynomial
Once the binary image has been found, I performed a sliding window search to find points correspond to the left and right lane. I first created a histogram accross the x axis of the images. The I kept the 2 position with the most responses.
These are the starting points of the traffic lanes. I find all the points that correspond to the left and right lane. I then fit two second order polynomial lines on these two set of points.


## Measure Curvature
The curvature at a given point on the polynomial can be calculated using the formula:

```
f(y) = Ay^2 + By + C
R_curve = ((1 + (2Ay + B)^2) ^ (3/2)) / abs(2A)
```

I measured the curvature at the point closest to the car:
```
y = frame.shape[0]
```

I first converted form pixel to real word space (in meters). Then I calculated the curve using the above formula.

[undistort_chessboard]: https://raw.github.com/tkkhuu/TrafficLaneDetection/master/README_files/undistort_chessboard.png "Undistort chessboard"
[undistort_img]: https://raw.github.com/tkkhuu/TrafficLaneDetection/master/README_files/undistort_img.png "Undistort scene"
[bev_tf]: https://raw.github.com/tkkhuu/TrafficLaneDetection/master/README_files/bev_tf.png "Birds Eye View transform"
[fine_lane]: https://raw.github.com/tkkhuu/TrafficLaneDetection/master/README_files/fine_lane.png "Lane Detection"


