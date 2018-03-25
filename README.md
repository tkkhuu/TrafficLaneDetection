# Traffic Lane Tracking

This is a project from the Udacity Self-Driving Car Nanodegree that builds a pipeline to detect and track traffic lanes on the highway

<a href="http://www.youtube.com/watch?feature=player_embedded&v=wRnFrW5-yrg
" target="_blank"><img src="http://img.youtube.com/vi/wRnFrW5-yrg/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>

## Camera Calibration
Before diving into lane detection, we have to calibrate the camera to undistort the images.
A set of chessboard images were provided in order to perform camera calibration.

For each chessboard image, the corners (the point where any four squares on the chessboard share) were extracted in terms of pixel positions using opencv function

```
corners = cv2.findChessboardCorners(gray_img, (n_col, n_row), None)
```

A list of world coordinates of the corners were also created, called object points. The camera matrix can then be calculated from the corners and objpoints.

```
cv2.calibrateCamera(objpoints, imgpoints
```

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



