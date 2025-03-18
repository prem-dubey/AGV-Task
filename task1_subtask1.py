import cv2
import numpy as np

def lucas_kanade_optical_flow(video_path):
    """
    Tracks motion in a video using the Lucas-Kanade Optical Flow algorithm  .

    Parameters :
        video_path : Path to the input video file 

    Returns:
        Displays the video and draws lines where movements is detected using optical flow 
    """
    # Parameters for Shi-Tomasi corner detection 
    feature_params = dict(maxCorners=100, qualityLevel=0.35, minDistance=5, blockSize=6)
    '''
    Shi-Tomasi corner detection uses goodFeaturesToTrack() function from cv2 

    Parameters : 
         
        image: The input grayscale image for feature detection.
        maxCorners : maximum no of corner it detects .
        qualityLevel : Minimum accepted quality of a corner 
        minDistance: Minimum distance between detected corners to avoid clustering.
        mask: Optional region of interest (ROI) where corners should be detected.
        blockSize: Window size for computing gradients in corner detection.

    returns : an array of detected corner points in the form of a NumPy array      
    '''

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    '''
    parameter : 
        prevImg : previous grayscale image 
        nextImg: next grayscale image
         prevPts: Array of points (keypoints) to track from prevImg
        nextPts: Output array of tracked points in nextImg.
        winSize: Size of the search window at each pyramid level .
        maxLevel : No of pyramid levels for high movement tracking 
        criteria: Stopping criteria ((type, max_iter, epsilon) .
    Returns :
         nextPts: A NumPy array of shape (N, 1, 2) containing the new (x, y) positions of the tracked points.
         status: A NumPy array of shape (N, 1), where 1 means the point was successfully tracked, and 0 means tracking failed.
         err: A NumPy array of shape (N, 1) containing the tracking error for each point (lower is better).
    '''

    # Read the video
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:

        # Read and update the frame regulary for major movements 
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        

        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # making it grayScale 

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0,0), 3) # drawing line from a,b to c,d 
            
            frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1) # drawing circle for good points 

        # Displaying the result
        img = cv2.add(frame, mask)
        cv2.imshow('Optical Flow', img)

        # break if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        # Updating the previous frame and points 
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

# Usage
video_path = 'video2.mp4'
lucas_kanade_optical_flow(video_path)
