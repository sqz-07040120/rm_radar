from math import sqrt , pow
import cv2
import numpy as np
tag_size = 72.5
tag_size_half = 36.25
img='CalibrateCamera/aaaaaa0.png'
# img = 'CalibrateCamera/calibresult3.jpg'
cameraMatrix = np.array([[1511.75000103212, 0.00000000e+00, 1001.28782882040],
                            [0.00000000e+00, 1511.53856178096, 458.372511183584],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)
distCoeffs = np.array(
    [[0.0768086100327664, 0.00292814654724461, 0.00158621071696517, -0.000565571534844693]], # ok
    # [[0.0787117281910156, -0.00185798044270703, 0.00168704529471195, -0.000715144029872121]],  # ok
    dtype=np.float64)
objPoints = np.array([[-tag_size_half, -tag_size_half, 0],
                      [tag_size_half, -tag_size_half, 0],
                      [tag_size_half, tag_size_half, 0],
                      [-tag_size_half, tag_size_half, 0]], dtype=np.float64)
objectPoints = [[-tag_size_half, tag_size_half, 1],
                      [-tag_size_half, -tag_size_half, 1],
                      [tag_size_half, -tag_size_half, 1],
                      [tag_size_half, tag_size_half, 1]]
objPoints = np.array([[-tag_size_half, tag_size_half, 1],
                      [-tag_size_half, -tag_size_half, 1],
                      [tag_size_half, -tag_size_half, 1],
                      [tag_size_half, tag_size_half, 1]], dtype=np.float64)
objectPoints = [[-10, 5, 1],
                      [10, 5, 1],
                      [10, -5, 1],
                      [-10, -5, 1]]
objPoints = np.array([[-10, 5, 1],
                      [10, 5, 1],
                      [10, -5, 1],
                      [-10, -5, 1]], dtype=np.float64)
points = [[432, 92], [566, 90], [580, 186], [450, 182]]

leftedge_to_Rbase = 2924.2475017359507

Bbase_to_Rbase = 10176.381306041108

leftedge_to_Radar = 5906.97995350662

base_objpoints = np.array([[-101,790.5,0],[342.5,790.5,0],[101,790.5,0],[-342.5,-790.5,0]], dtype=np.float64)
# points = [[126, 112], [148, 254], [428, 262], [430, 72]]
def coordinate(objPoints,imgpoints):
    cameraMatrix = np.array([[1511.73984750674, 0.00000000e+00, 1001.30135091120],
                            [-0.0187641102844033, 1511.52937964640, 458.355771433958],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64)

    distCoeffs = np.array([[0.105136422472789, -0.155623814703081, 0.00148579254813449, 0.00161806604998984, -0.0450618830121726]], dtype=np.float64)

    imgPoints = np.array(imgpoints, dtype=np.float64)
    retval, R, T = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    return T
# imgPoints = np.array(points, dtype=np.float64)
# cameraMatrix = K
# distCoeffs = np.array(
#     [[0.105136422472789, -0.155623814703081, 0.00148579254813449, 0.00161806604998984, -0.0450618830121726]],
#     dtype=np.float64)
# retval,R,T  = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
# print(sqrt(pow(T[0],2)+pow(T[1],2)+pow(T[2],2)))
# print(T)
# one = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
#
# img=cv2.imread(img,cv2.IMREAD_COLOR)
# cv2.polylines(img, np.array([points]), 1, (0, 255, 0), 5, 8, 0)
# cv2.imshow('AreaCompeCali',img)
# key = cv2.waitKey(10000)
#
# Z = []
# R,jacobian = cv2.Rodrigues(R) #from R-vector to R-matrix
# Rr = np.matrix.transpose(R)
# for i in range(len(objectPoints)):
#     Rr = []
#     for j in range(len(T)):
#         Rr.append(np.append(R[j],T[j]))
#     Rr = np.array(Rr)
#     # print(Rr)
#     en_para = np.concatenate((Rr,np.array([[0,0,0,1]],dtype=np.float64)), axis=0)
#     print(en_para)
#     K_all = np.dot(np.dot(K,one),en_para)
#       opencv官网solvepnp详解部分公式
#     world = np.dot(np.linalg.pinv(K_all),np.array(objectPoints, dtype=np.float64)[i])
#     print("world"+str(world))
#     print("ZZZZ",np.dot(np.linalg.pinv(R*(-1)),T))
#     Z.append(sqrt(pow(world[0],2)+pow(world[1],2)+pow(world[2],2)))
# print(Z)

