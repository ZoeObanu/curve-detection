import cv2
import numpy as np
 
# define a video capture object 
vid = cv2.VideoCapture(0)
  
while(True): 
      
    # Capture the video frame by frame 
    ret, frame = vid.read()

    #convert to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #blur image
    blur_gray = cv2.GaussianBlur(gray,(5, 5),0)

    #apply canny edge
    edges = cv2.Canny(blur_gray, 50, 150)

    # code for mask taken from stack overflow
    # create a mask
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[100:600, 300:1000] = 255

    # compute the bitwise AND using the mask
    masked_img = cv2.bitwise_and(edges,edges,mask = mask)

    # From stack overflow
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours is not None:
        frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 10)

        # from stack overflow
        # read image and invert so lines are white on black background
        img = 255-edges
        
        # do some eroding of img
        kernel = np.ones((20,20), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)

        # do distance transform
        dist = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)

        # set up cross for tophat skeletonization
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        skeleton = cv2.morphologyEx(dist, cv2.MORPH_TOPHAT, kernel)

        masked_img2 = cv2.bitwise_and(skeleton,skeleton,mask = mask)
        

    # from geeks for geeks
    cv2.rectangle(frame, (300,100), (1000,600), (0,0,0), 5)

    # from tutorialKart
    scale_percent = 300 # percent of original size
    width = int(masked_img2.shape[1] * scale_percent / 100)
    height = int(masked_img2.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(masked_img2, dim, interpolation = cv2.INTER_AREA)

    result = frame.copy()
    result[masked_img2!=0] = (0,0,255)

        
    cv2.imshow("Detected Circle", result)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()
