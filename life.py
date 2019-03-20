import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3
EAR_AVG = 0

COUNTER = 0
TOTAL = 0

def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])


    C = dist.euclidean(eye[0], eye[3])


    ear = (A + B) / (2 * C)
    return ear


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()

            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])

            left_eye = landmarks[LEFT_EYE_POINTS]

            right_eye = landmarks[RIGHT_EYE_POINTS]

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1) # (image, [contour], all_contours, color, thickness)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            ear_left = eye_aspect_ratio(left_eye)

            ear_right = eye_aspect_ratio(right_eye)

            ear_avg = (ear_left + ear_right) / 2.0

            if ear_avg < EYE_AR_THRESH:
                COUNTER += 1
                print("fake")
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    print("real")
                COUNTER = 0


        cv2.imshow("Winks Found", frame)
        key = cv2.waitKey(1) & 0xFF

        if key is ord('q'):
            break

print(TOTAL)
cap.release()

cv2.destroyAllWindows()
