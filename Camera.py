import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Camera():
    def __init__(self, cam_mtx=None, dist_coeff=None):
        self._cam_mtx = cam_mtx
        self._dist_coeff = dist_coeff
        
    def Calibrate(self, chessboard_images_dir, n_corners, filetype='jpg'):
        
        # Appending back slash '/' if needed
        filepath = chessboard_images_dir if (chessboard_images_dir.endswith('/')) else (chessboard_images_dir + '/')
        filepath += '*.' + filetype
        
        images_path = glob.glob(filepath)
        
        objpoints = []
        imgpoints = []
        
        n_row = n_corners[0]
        n_col = n_corners[1]
        
        objp = np.zeros((n_row*n_col, 3), np.float32)
        objp[:,:2] = np.mgrid[0:n_col, 0:n_row].T.reshape(-1, 2)
        
        for image_path in images_path:
            
            img = cv2.imread(image_path)
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray_img, (n_col, n_row), None)
            
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
        
        ret, self._cam_mtx, self._dist_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)
    
    def UndistortImage(self, distorted_img):
        if (self._cam_mtx is None or self._dist_coeff is None):
            raise ValueError('Camera has not been calibrated, please calibrate camera by using Calibrate() before undistorting images')
        else:
            return cv2.undistort(distorted_img, self._cam_mtx, self._dist_coeff, None, self._cam_mtx)

def main():
    
    camera = Camera()
    
    camera.Calibrate('../camera_cal/', (9, 6), filetype='jpg')
    
    distorted_img = cv2.imread('../camera_cal/calibration3.jpg')
    
    undistorted_img = camera.UndistortImage(distorted_img)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(distorted_img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(undistorted_img)
    ax2.set_title('Undistorted Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    
if __name__ == '__main__': main()