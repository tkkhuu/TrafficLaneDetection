import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Camera import Camera

class LaneDetection():
    def __init__(self, camera):
        self._camera = camera
        self._left_fit = [] # Left lane 2nd order polynomial list of coeffs
        self._right_fit = [] # Right lane 2nd order polynomial list of coeffs
        self._first_frame = True
        self._M_perspective = None
        self._inv_M_perspective = None
        
    def _WarpImageTF(self, img, rec_corners, bev_corners):
        img_size = (img.shape[1], img.shape[0])
        x_center = img.shape[1] // 2
    
        src = np.float32(rec_corners)
        
        dst = np.float32(bev_corners)
        
        self._M_perspective = cv2.getPerspectiveTransform(src, dst)
        self._inv_M_perspective = cv2.getPerspectiveTransform(dst, src)
        warped_img = cv2.warpPerspective(img, self._M_perspective, img_size)
        return warped_img
    
    def _FindLaneEdgesBinaryMap(self, rgb_img,
                             gray_thresh=(20, 255, None, None),
                             v_thresh=(2, 20, -0.8, 0.8),
                             h_thresh=(18, 50, None, None),
                             s_thresh=(100, 255, -0.8, 0.8)):
        
        # Convert RGB images to HSV format and grayscale
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV).astype(np.float)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float)
        
        h_channel = hsv_img[:,:,0]
        s_channel = hsv_img[:,:,1]
        v_channel = hsv_img[:,:,2]
    
        # Compute gradient on gray_img
        sobelx_gray = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)                         # Take gradient in x
        abs_sobelx_gray = np.absolute(sobelx_gray)                                  # Get the absolute value
        scaled_sobelx_gray = np.uint8(255*abs_sobelx_gray/np.max(abs_sobelx_gray))  # Gradient scaled magnitude

        gray_binary = np.zeros_like(scaled_sobelx_gray)
        gray_binary[(scaled_sobelx_gray >= gray_thresh[0]) & (scaled_sobelx_gray <= gray_thresh[1])] = 1
        
        # Compute gradient on v_channel
        v_channel = hsv_img[:,:,2]
        v_sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0)                       # Take gradient in x
        v_sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1)                       # Take gradient in y
        v_abs_sobelx = np.absolute(v_sobelx)
        v_abs_sobely = np.absolute(v_sobely)
        v_abs_sobelxy = np.sqrt(v_sobelx**2 + v_sobely**2)
        v_direction_grad = np.arctan2(v_abs_sobely, v_abs_sobelx)               # Gradient direction
        v_scaled_sobelxy = np.uint8(255*v_abs_sobelxy / np.max(v_abs_sobelxy))  # Gradient scaled magnitude
        
        v_binary = np.zeros_like(v_direction_grad)
        v_binary[(v_scaled_sobelxy >= v_thresh[0]) & (v_scaled_sobelxy <= v_thresh[1])
                & (v_direction_grad >= v_thresh[2]) & (v_direction_grad <= v_thresh[3])] = 1
        
        # Compute binary on h_channel
        h_channel = hsv_img[:,:,0]
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
        
        # Compute binary on s_channel
        s_sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        s_sobely = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1)
        s_abs_sobelx = np.absolute(s_sobelx)
        s_abs_sobely = np.absolute(s_sobely)
        s_abs_sobelxy = np.sqrt(s_sobelx**2 + s_sobely**2)
        s_direction_grad = np.arctan2(s_abs_sobely, s_abs_sobelx)
        s_scaled_sobelxy = np.uint8(255*s_abs_sobelxy / np.max(s_abs_sobelxy))
        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
                & (s_direction_grad >= s_thresh[2]) & (s_direction_grad <= s_thresh[3])] = 1
        
        binary = gray_binary + v_binary + h_binary + s_binary
    
        binary[binary < 2] = 0
        binary[binary >= 2] = 1
        
        return binary
    
    def _DetectLanes(self, rgb_img, rec_corners=None, bev_corners=None):
        img_size = (rgb_img.shape[1], rgb_img.shape[0])
        x_center = rgb_img.shape[1] // 2
        
        if (rec_corners is None):
            rec_bot_margin = 400
            rec_top_margin = 50
            rec_top_height = 450
            rec_bot_height = 40
            
            rec_corners = [(x_center - rec_bot_margin, img_size[1] - rec_bot_height),
                           (x_center + rec_bot_margin, img_size[1] - rec_bot_height),
                           (x_center - rec_top_margin, rec_top_height),
                           (x_center + rec_top_margin, rec_top_height)]
        
        if (bev_corners is None):
            bev_bot_margin = 400
            bev_top_height = 0
            bev_bot_height = 40
            
            bev_corners = [(x_center - bev_bot_margin, img_size[1] - bev_bot_height),
                           (x_center + bev_bot_margin, img_size[1] - bev_bot_height),
                           (x_center - bev_bot_margin, bev_top_height),
                           (x_center + bev_bot_margin, bev_top_height)]
        
        undistorted_frame = self._camera.UndistortImage(rgb_img)
        bev_img = self._WarpImageTF(undistorted_frame, rec_corners, bev_corners)
        binary_img = self._FindLaneEdgesBinaryMap(bev_img)
        
        return binary_img
    
    def _TrackLanes(self, frame):
        img_size = (frame.shape[1], frame.shape[0])
        
        binary_image = self._DetectLanes(frame)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Set the width of the windows +/- margin
        margin = 100
        
        if self._first_frame:
            
            self._first_frame = False
            
            histogram = np.sum(binary_image[binary_image.shape[0]//2:,:], axis=0)
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            
            nwindows = 9
            
            # Set height of windows
            window_height = np.int(binary_image.shape[0]/nwindows)
            
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []
            
            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_image.shape[0] - (window+1)*window_height
                win_y_high = binary_image.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
               
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                    
            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            
            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 
            
            # Fit a second order polynomial to each
            self._left_fit = np.polyfit(lefty, leftx, 2)
            self._right_fit = np.polyfit(righty, rightx, 2)
            
        else:
            left_lane_inds = ((nonzerox > (self._left_fit[0]*(nonzeroy**2) + self._left_fit[1]*nonzeroy + 
            self._left_fit[2] - margin)) & (nonzerox < (self._left_fit[0]*(nonzeroy**2) + 
            self._left_fit[1]*nonzeroy + self._left_fit[2] + margin))) 
            
            right_lane_inds = ((nonzerox > (self._right_fit[0]*(nonzeroy**2) + self._right_fit[1]*nonzeroy + 
            self._right_fit[2] - margin)) & (nonzerox < (self._right_fit[0]*(nonzeroy**2) + 
            self._right_fit[1]*nonzeroy + self._right_fit[2] + margin)))
            
            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            
            # Fit a second order polynomial to each
            self._left_fit = np.polyfit(lefty, leftx, 2)
            self._right_fit = np.polyfit(righty, rightx, 2)
            
        ploty = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0] )
        left_fitx = self._left_fit[0]*ploty**2 + self._left_fit[1]*ploty + self._left_fit[2]
        right_fitx = self._right_fit[0]*ploty**2 + self._right_fit[1]*ploty + self._right_fit[2]
    
        left_pts = np.array(list(zip(left_fitx, ploty)), dtype=np.int32)
        right_pts = np.array(list(zip(right_fitx, ploty)), dtype=np.int32)

        out_img = np.zeros(frame.shape)

        for i in range(frame.shape[0]):
            out_img[i, left_pts[i, 0]:right_pts[i, 0], 1] = 255

        unwarped_img = cv2.warpPerspective(out_img, self._inv_M_perspective, img_size)
        result = cv2.addWeighted(np.uint8(frame), 1, np.uint8(unwarped_img), 0.3, 0)
        
        y_eval = np.max(ploty)
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        cv2.putText(result, 'Left curvature radius: {}'.format(left_curverad), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(result, 'Right curvature radius: {}'.format(right_curverad), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        
        return result
    
    def TrackLanesImg(self, img, viz=False):
        
        self._first_frame = True
        
        result = self._TrackLanes(img)
        
        if viz:
            plt.imshow(result)
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show()
        return result
      
    def TrackLanesVideo(self, video_in_path, video_out_path):
        self._first_frame = True
        video_in = VideoFileClip(video_in_path)
        processed_video_in = video_in.fl_image(self._TrackLanes)
        processed_video_in.write_videofile(video_out_path, audio=False)
        
def main():
    camera = Camera()
    
    camera.Calibrate('../camera_cal/', (9, 6), filetype='jpg')
    
    lane_detection = LaneDetection(camera)
    
    test_img = mpimg.imread('../test_images/test2.jpg')
    
    out_img = lane_detection.TrackLanesImg(test_img, False)
    
    lane_detection.TrackLanesVideo('../project_video.mp4', './output_video.mp4')
    
if __name__ == '__main__': main()
        
        