import cv2 
import numpy as np 
import argparse

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
                # default="DICT_ARUCO_ORIGINAL",
                default = "DICT_6X6_100",
                help="type of ArUCo tag to detect")
ap.add_argument("-v", "--video", type=str,
                default="merged.mp4",
                help="relative path of video file")
args = vars(ap.parse_args())

aruco_type = args["type"]
video_path = args["video"]

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters()  # Updated line
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)


# YOUR CAMERA CALIBRATION
camera_matrix = np.array([[682.2733199, 0, 271.96919478],
                        [ 0, 683.32544778, 221.84412458],
                        [ 0, 0, 1]])

dist_coeffs = np.array([-0.51105737, -0.05309581,  0.01634051,  0.00750281,  1.93236235])

# open the camera
cap = cv2.VideoCapture(0)  # choose appropriate camera

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# get the optimal camera matrix for undistortion
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# precompute the undistortion map
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)

fourcc = cv2.VideoWriter_fourcc(*'H264')  # codec for MP4

# CHANGE values based on your camera frame size
c_top = 20
c_bottom = 40
c_left = 80
c_right = 120
out = cv2.VideoWriter(video_path, fourcc, 30.0, (w - (c_left + c_right), h - (c_top + c_bottom)))  # ensure resolution matches (w, h)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Undistort the frame
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    # crop the frame
    cropped_frame = undistorted_frame[c_top:-c_bottom, c_left:-c_right] 

    (corners, ids, rejected) = detector.detectMarkers(cropped_frame)

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
		# flatten the ArUco IDs list
        ids = ids.flatten()
        
        for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (in top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

			# draw the bounding box of the ArUCo detection
            cv2.line(cropped_frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(cropped_frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(cropped_frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(cropped_frame, bottomLeft, topLeft, (0, 255, 0), 2)

            # compute and draw the center (x, y) coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(cropped_frame, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the frame
            cv2.putText(cropped_frame, str(markerID),
                (topLeft[0], topLeft[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

            # draw point to indicate front of robot
            fX = int((topLeft[0] + topRight[0]) / 2.0)
            fY = int((topLeft[1] + topRight[1]) / 2.0)
            cv2.circle(frame, (fX, fY), 4, (255, 0, 0), -1)


    # save the video
    out.write(cropped_frame)

    # display the cropped frame
    cv2.imshow('Cropped Undistorted Livestream', cropped_frame)

    # 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video {video_path} Saved")