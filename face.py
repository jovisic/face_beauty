import dlib
import numpy as np
from imutils import face_utils
import argparse
import imutils
import cv2


def rect_to_bb(rect):

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):

    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)


    return coords

DEFAULT_PREDICTOR = "predictor/shape_predictor_68_face_landmarks.dat"
DEFAULT_IMAGE = "pics/smigic.jpg"

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=False,
	help="Shape predictor path", default=DEFAULT_PREDICTOR)

ap.add_argument("-i", "--image", required=False,
	help="Image path", default=DEFAULT_IMAGE)

args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


rects = detector(gray, 1)

for (i, rect) in enumerate(rects):

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Output", image)
cv2.waitKey(0)

