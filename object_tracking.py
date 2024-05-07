import cv2
import numpy as np
import argparse

def load_video(video_path:str):
    """
    Function to load image
    
    Parameter(s):    video_path:str

    Returns:       video: video to process
    """
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print("Error: Could not open video capture device.")
        exit()
    return vid

def load_object_image(img_path:str):
    """
    Function to load image
    
    Parameter(s):    img_path:str

    Returns:       object_img: image with the object of interest
    """
    object_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if object_img is None:
        print(f"Error: Could not load image {img_path}")
        exit()
    return object_img


def detect_features(image):
    """
    Function to detect features 
    
    Parameter(s):    image: loaded image

    Returns:        keypoints: keypoints of the image
                    descriptors:descriptores of the image
                    orb: orb create
                    feature_params: parameters of shi-tomasi
    """
    orb = cv2.ORB_create()
    feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)
    corners = cv2.goodFeaturesToTrack(image, mask=None, **feature_params)
    keypoints = [cv2.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size=7) for c in corners] if corners is not None else []
    keypoints, descriptors = orb.compute(image, keypoints)
    return keypoints, descriptors, orb, feature_params

def parse_user_data()->argparse:
    """
    Function to input the user data 
    
    Parameter(s):    None

    Returns:       args(argparse): argparse object with the user info
    """
    parser = argparse.ArgumentParser(description='Feature matching between two images.')
    parser.add_argument('--img_obj', required=True,
                        help='Input image for feature matching')
    parser.add_argument('--video',required=True,
                        help='Video sequence path')
    args = parser.parse_args()
    return args


def process_and_display(img_path:str,video_path:str)->None:
    """
    Function to process and display 
    
    Parameter(s):    img_path: path to the image 
                    video_path:path to the image

    Returns:        None
    """
    object_img = load_object_image(img_path)
    vid = load_video(video_path)
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])
    kp1,des1,orb,feature_params = detect_features(object_img)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    line_position = 300
    crossing_count_ltr = 0
    crossing_count_rtl = 0
    previous_position = None
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = vid.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert the frame from BGR to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            hsv = cv2.erode(hsv, kernel, iterations=2)
            hsv = cv2.dilate(hsv, kernel, iterations=2)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            cv2.imshow("hsv",mask)
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    roi = frame[y:y+h, x:x+w]
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    corners = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
                    
                    if corners is not None and len(corners) > 0:
                        keypoints_roi = [cv2.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size=7) for c in corners]
                        kp2, des2 = orb.compute(roi_gray, keypoints_roi)

                        if des2 is not None:
                            matches = bf.match(des1, des2)
                            matches = sorted(matches, key=lambda x: x.distance)

                            img_matches = cv2.drawMatches(object_img, kp1, roi, kp2, matches[:10], None, flags=2)

                            cv2.imshow('Matches', img_matches)



                            points = np.zeros((len(matches), 2), dtype=np.float32)
                            for i, match in enumerate(matches):
                                points[i, :] = kp2[match.trainIdx].pt

                            if len(points) > 0:
                                x2, y2, w2, h2 = cv2.boundingRect(points)
                                cv2.rectangle(roi, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)

                                centroid_x = int(x + x2 + w2 / 2)
                                centroid_y = int(y + y2 + h2 / 2)
                                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
                                
                                if previous_position is not None:
                                    # Check if crossing from left to right
                                    if previous_position < line_position <= centroid_x:
                                        crossing_count_ltr += 1
                                    # Check if crossing from right to left
                                    elif previous_position > line_position >= centroid_x:
                                        crossing_count_rtl += 1

                                previous_position = centroid_x

            cv2.line(frame, (line_position, 0), (line_position, frame.shape[0]), (0, 0, 255), 2)

            # Display the crossing counts
            cv2.putText(frame, f"LtR Crossings: {crossing_count_ltr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"RtL Crossings: {crossing_count_rtl}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)                

            # Display the resulting frame
            cv2.imshow('Original Video', frame)
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
    # When everything done, release the capture
        vid.release()
        cv2.destroyAllWindows()

def pipeline():
    args = parse_user_data()
    img_path = args.img_obj
    vid_path = args.video
    process_and_display(img_path,vid_path)

if __name__ == "__main__":
    pipeline()