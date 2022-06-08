import numpy as np
import cv2 as cv
import glob
import random as rnd
import math

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 300, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*7,3), np.float32)
objp = np.zeros((6 * 8, 3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
objp[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('resources/*.jpg')
for fname in images:
    img = cv.imread(fname)
    img = cv.resize(img, (800, 800))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = cv.blur(gray, (3, 3))
    # gray = cv.dilate(gray, (3, 3))
    # cv.imshow('gray', gray)
    # cv.waitKey(100)
    # Find the chess board corners
    # ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    ret, corners = cv.findChessboardCorners(gray, (6, 8), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # print(corners)
        # Draw and display the corners
        # cv.drawChessboardCorners(img, (7,6), corners2, ret)
        img = cv.drawChessboardCorners(img, (6, 8), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

print(data)

kernel = np.ones((5, 5), np.uint8)
img = cv.imread('resources2/1.jpg')
canny = cv.Canny(img, 100, 200)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img_gray, (3, 3), 0)
sobelx = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
erosion = cv.erode(img, kernel, iterations=1)
dilation = cv.dilate(img, kernel, iterations=1)
filtered_with_edge_detector = cv.imread('resources2/1.jpg')
filtered_with_edge_detector = cv.resize(filtered_with_edge_detector, (800, 800))
filtered_with_edge_detector = cv.cvtColor(filtered_with_edge_detector, cv.COLOR_BGR2GRAY)
filtered_with_edge_detector = cv.GaussianBlur(filtered_with_edge_detector, (3, 3), 0)
filtered_with_edge_detector = cv.Canny(filtered_with_edge_detector, 100, 200)
# filtered_with_edge_detector = cv.Sobel(src=filtered_with_edge_detector, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
# filtered_with_edge_detector = cv.erode(filtered_with_edge_detector, kernel, iterations=1)
# filtered_with_edge_detector = cv.dilate(filtered_with_edge_detector, kernel, iterations=1)


def goodFeaturesToTrack(img_gray, val):
    maxCorners = max(val, 1)
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04
    copy = np.copy(img_gray)
    corners = cv.goodFeaturesToTrack(img_gray, maxCorners, qualityLevel, minDistance, None,
                                     blockSize=blockSize, gradientSize=gradientSize,
                                     useHarrisDetector=useHarrisDetector, k=k)
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (int(corners[i, 0, 0]), int(corners[i, 0, 1])), radius,
                  (rnd.randint(0, 256), rnd.randint(0, 256), rnd.randint(0, 256)), cv.FILLED)
    return copy


shi_tomasi = goodFeaturesToTrack(img_gray, 23)
img_gray = np.float32(img_gray)
harris = cv.cornerHarris(img_gray, 2, 3, 0.04)
harris = cv.dilate(harris, None)
harris_ret, harris = cv.threshold(harris, 0.01 * harris.max(), 255, 0)
harris = np.uint8(harris)
filtered_with_corner_detector = cv.imread('resources2/1.jpg')
filtered_with_corner_detector = cv.resize(filtered_with_corner_detector, (800, 800))
filtered_with_corner_detector = cv.cvtColor(filtered_with_corner_detector, cv.COLOR_BGR2GRAY)
filtered_with_corner_detector = goodFeaturesToTrack(filtered_with_corner_detector, 23)
filtered_with_corner_detector = np.float32(filtered_with_corner_detector)
filtered_with_corner_detector = cv.cornerHarris(filtered_with_corner_detector, 2, 3, 0.04)
filtered_with_corner_detector = cv.dilate(filtered_with_corner_detector, None)
filtered_with_corner_detector_ret, filtered_with_corner_detector = cv.threshold(filtered_with_corner_detector, 0.01 * harris.max(), 255, 0)
filtered_with_corner_detector = np.uint8(filtered_with_corner_detector)


distance = -1
a = [-1, -1]
b = [-1, -1]


def clickCb(event, x, y, flags, param):
    # time.sleep(1)
    global distance, a, b
    if event == cv.EVENT_LBUTTONDOWN:
        a = (x, y)
    if event == cv.EVENT_RBUTTONDOWN:
        b = (x, y)
    if a != [-1, -1] and b != [-1, -1]:
        distance = math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
        return


cv.imshow('filtered with edge detector', filtered_with_edge_detector)
cv.setMouseCallback('filtered with edge detector', clickCb)
# cv.imshow('canny', canny)
# cv.setMouseCallback('canny', clickCb)
# cv.imshow('blur', blur)
# cv.setMouseCallback('blur', clickCb)
# cv.imshow('sobelx', sobelx)
# cv.setMouseCallback('sobelx', clickCb)
# cv.imshow('sobely', sobely)
# cv.setMouseCallback('sobely', clickCb)
# cv.imshow('sobelxy', sobelxy)
# cv.setMouseCallback('sobelxy', clickCb)
# cv.imshow('erosion', erosion)
# cv.setMouseCallback('erosion', clickCb)
# cv.imshow('dilation', dilation)
# cv.setMouseCallback('dilation', clickCb)
# cv.imshow('shi_tomasi', shi_tomasi)
# cv.setMouseCallback('shi_tomasi', clickCb)
# cv.imshow('harris', harris)
# cv.setMouseCallback('harris', clickCb)
while distance == -1:
    # time.sleep(10)
    cv.imshow('filtered with edge detector', filtered_with_edge_detector)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # continue
print('dimension in px == ' + str(distance))
# print('enter dimension in cm: ')
# cm = input()
# print('cm == ' + cm)
# cell_size = 1  # mm
obj_size = 25  # mm
f_x = data.get('camera_matrix')[0][0]
d = f_x * obj_size / distance
print(d)
distance = -1
a = [-1, -1]
b = [-1, -1]
cv.imshow('filtered with corner detector', filtered_with_corner_detector)
cv.setMouseCallback('filtered with corner detector', clickCb)
while distance == -1:
    cv.imshow('filtered with corner detector', filtered_with_corner_detector)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # continue
print('dimension in px == ' + str(distance))
f_x = data.get('camera_matrix')[0][0]
d = f_x * obj_size / distance
print(d)
cv.waitKey(0)

kernel = np.ones((5, 5), np.uint8)
img = cv.imread('resources2/2.jpg')
canny = cv.Canny(img, 100, 200)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img_gray, (3, 3), 0)
sobelx = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
erosion = cv.erode(img, kernel, iterations=1)
dilation = cv.dilate(img, kernel, iterations=1)
filtered_with_edge_detector = cv.imread('resources2/2.jpg')
filtered_with_edge_detector = cv.resize(filtered_with_edge_detector, (800, 800))
filtered_with_edge_detector = cv.cvtColor(filtered_with_edge_detector, cv.COLOR_BGR2GRAY)
filtered_with_edge_detector = cv.GaussianBlur(filtered_with_edge_detector, (3, 3), 0)
filtered_with_edge_detector = cv.Canny(filtered_with_edge_detector, 100, 200)
# filtered_with_edge_detector = cv.Sobel(src=filtered_with_edge_detector, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
# filtered_with_edge_detector = cv.erode(filtered_with_edge_detector, kernel, iterations=1)
# filtered_with_edge_detector = cv.dilate(filtered_with_edge_detector, kernel, iterations=1)


def goodFeaturesToTrack(img_gray, val):
    maxCorners = max(val, 1)
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04
    copy = np.copy(img_gray)
    corners = cv.goodFeaturesToTrack(img_gray, maxCorners, qualityLevel, minDistance, None,
                                     blockSize=blockSize, gradientSize=gradientSize,
                                     useHarrisDetector=useHarrisDetector, k=k)
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (int(corners[i, 0, 0]), int(corners[i, 0, 1])), radius,
                  (rnd.randint(0, 256), rnd.randint(0, 256), rnd.randint(0, 256)), cv.FILLED)
    return copy


shi_tomasi = goodFeaturesToTrack(img_gray, 23)
img_gray = np.float32(img_gray)
harris = cv.cornerHarris(img_gray, 2, 3, 0.04)
harris = cv.dilate(harris, None)
harris_ret, harris = cv.threshold(harris, 0.01 * harris.max(), 255, 0)
harris = np.uint8(harris)
filtered_with_corner_detector = cv.imread('resources2/2.jpg')
filtered_with_corner_detector = cv.resize(filtered_with_corner_detector, (800, 800))
filtered_with_corner_detector = cv.cvtColor(filtered_with_corner_detector, cv.COLOR_BGR2GRAY)
filtered_with_corner_detector = goodFeaturesToTrack(filtered_with_corner_detector, 23)
filtered_with_corner_detector = np.float32(filtered_with_corner_detector)
filtered_with_corner_detector = cv.cornerHarris(filtered_with_corner_detector, 2, 3, 0.04)
filtered_with_corner_detector = cv.dilate(filtered_with_corner_detector, None)
filtered_with_corner_detector_ret, filtered_with_corner_detector = cv.threshold(filtered_with_corner_detector, 0.01 * harris.max(), 255, 0)
filtered_with_corner_detector = np.uint8(filtered_with_corner_detector)


distance = -1
a = [-1, -1]
b = [-1, -1]


def clickCb(event, x, y, flags, param):
    # time.sleep(1)
    global distance, a, b
    if event == cv.EVENT_LBUTTONDOWN:
        a = (x, y)
    if event == cv.EVENT_RBUTTONDOWN:
        b = (x, y)
    if a != [-1, -1] and b != [-1, -1]:
        distance = math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
        return


cv.imshow('filtered with edge detector', filtered_with_edge_detector)
cv.setMouseCallback('filtered with edge detector', clickCb)
# cv.imshow('canny', canny)
# cv.setMouseCallback('canny', clickCb)
# cv.imshow('blur', blur)
# cv.setMouseCallback('blur', clickCb)
# cv.imshow('sobelx', sobelx)
# cv.setMouseCallback('sobelx', clickCb)
# cv.imshow('sobely', sobely)
# cv.setMouseCallback('sobely', clickCb)
# cv.imshow('sobelxy', sobelxy)
# cv.setMouseCallback('sobelxy', clickCb)
# cv.imshow('erosion', erosion)
# cv.setMouseCallback('erosion', clickCb)
# cv.imshow('dilation', dilation)
# cv.setMouseCallback('dilation', clickCb)
# cv.imshow('shi_tomasi', shi_tomasi)
# cv.setMouseCallback('shi_tomasi', clickCb)
# cv.imshow('harris', harris)
# cv.setMouseCallback('harris', clickCb)
while distance == -1:
    # time.sleep(10)
    cv.imshow('filtered with edge detector', filtered_with_edge_detector)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # continue
print('dimension in px == ' + str(distance))
# print('enter dimension in cm: ')
# cm = input()
# print('cm == ' + cm)
# cell_size = 1  # mm
obj_size = 560  # mm
f_x = data.get('camera_matrix')[0][0]
d = f_x * obj_size / distance
print(d)
distance = -1
a = [-1, -1]
b = [-1, -1]
cv.imshow('filtered with corner detector', filtered_with_corner_detector)
cv.setMouseCallback('filtered with corner detector', clickCb)
while distance == -1:
    cv.imshow('filtered with corner detector', filtered_with_corner_detector)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # continue
print('dimension in px == ' + str(distance))
f_x = data.get('camera_matrix')[0][0]
d = f_x * obj_size / distance
print(d)
cv.waitKey(0)
