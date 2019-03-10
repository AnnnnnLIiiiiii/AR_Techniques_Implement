import cv2
import numpy as np
import math


def sort_points(array_of_points):
    pi = 3.1415926
    point_numbers = len(array_of_points)
    center = np.array([0, 0])
    points = []

    for i in range(point_numbers):
        points.append([array_of_points[i][0], array_of_points[i][1]])
        center[0] = points[i][0] + center[0]
        center[1] = points[i][1] + center[1]

    center = center/point_numbers
    # vec = [1,0]
    angle = []

    for i in range(point_numbers):
        vec_center_other = [points[i][0] - center[0], points[i][1] - center[1]]
        angle1 = math.acos(vec_center_other[0]/(vec_center_other[0]**2 + vec_center_other[1]**2)**0.5)
        angle2 = math.asin(vec_center_other[1]/(vec_center_other[0]**2 + vec_center_other[1]**2)**0.5)

        if angle2>0:
            angle.append(180*angle1/pi)
        else:
            angle.append(180*angle1/pi+2*(180-180*angle1/pi))

    for i in range(point_numbers, 0, -1):
        for j in range(i-1):
            if angle[j+1] < angle[j]:
                angle[j], angle[j+1] = angle[j+1], angle[j]
                points[j], points[j+1] = points[j+1], points[j]

    center = np.array(center)

    return points, center.astype(int)


def vector(p1, p2):
    return [p2[0]-p1[0], p2[1]-p1[1]]


def whether_inside_four(target_point, sorted_point):
    target_point = np.array(target_point)
    sorted_point = np.array(sorted_point)
    p1 = sorted_point[0]
    p2 = sorted_point[1]
    p3 = sorted_point[2]
    p4 = sorted_point[3]
    vec_12 = vector(p1, p2)
    vec_34 = vector(p3, p4)
    vec_23 = vector(p2, p3)
    vec_41 = vector(p4, p1)
    vec_target1 = vector(p1, target_point)
    vec_target2 = vector(p2, target_point)
    vec_target3 = vector(p3, target_point)
    vec_target4 = vector(p4, target_point)

    if np.dot(np.cross(vec_12, vec_target1), np.cross(vec_34, vec_target3)) >= 0 and np.dot(np.cross(vec_23, vec_target2), np.cross(vec_41, vec_target4)) >= 0:
        return 1
    else:
        return 0


def point_inside(sorted_point):
    sorted_point = np.array(sorted_point)
    max = np.amax(sorted_point, axis=0)
    min = np.amin(sorted_point, axis=0)
    p1 = sorted_point[0]
    p2 = sorted_point[1]
    p3 = sorted_point[2]
    p4 = sorted_point[3]
    center = (p1+p2+p3+p4)/4
    mid_12 = (p1+p2)/2
    mid_23 = (p2+p3)/2
    mid_34 = (p3+p4)/2
    mid_41 = (p1+p4)/2
    area_1 = np.array([p1, mid_12, center, mid_41])
    area_2 = np.array([mid_12, p2, mid_23, center])
    area_3 = np.array([center, mid_23, p3, mid_34])
    area_4 = np.array([mid_41, center, mid_34, p4])
    point_inside_1 = []
    point_inside_2 = []
    point_inside_3 = []
    point_inside_4 = []

    for i in range(min[0], max[0]+1):
        for j in range(min[1], max[1]+1):
            if whether_inside_four([i, j], area_1) == 1:
                point_inside_1.append([i, j])
            if whether_inside_four([i, j], area_2) == 1:
                point_inside_2.append([i, j])
            if whether_inside_four([i, j], area_3) == 1:
                point_inside_3.append([i, j])
            if whether_inside_four([i, j], area_4) == 1:
                point_inside_4.append([i, j])

    return [point_inside_1, point_inside_2, point_inside_3, point_inside_4]


def return_last_contour(old_contour, new_contour, threshold):
    area_old = cv2.contourArea(old_contour[1])
    area_new = cv2.contourArea(new_contour[1])
    gap = (area_old - area_new)**2
    if gap > threshold:
        return old_contour, 0
    else:
        return new_contour, 1


def distance(p1, p2):
    dis = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    return dis


def related_point(old_corner_point, new_corner_point, critical_index):

    list_num = [0, 1, 2, 3]
    result = []
    for i in list_num:
        for j in list_num:
            for k in list_num:
                for f in list_num:
                    if (i != j and i != k and i != f and j != k and j != f and k != f):
                        result.append([i, j, k, f])

    distance_list = []
    for i in range(len(result)):
        dist_total = 0
        for j in range(4):
            dist_total = dist_total + distance(new_corner_point[j], old_corner_point[result[i][j]])
        distance_list.append(dist_total)
    min_index = np.argmin(distance_list)
    point_list = result[min_index]
    answer_index = point_list.index(critical_index)
    return new_corner_point[answer_index]


def homography(_4_corners_list, ref_corners):
    #ref_corners = np.array([[512,512], [0,512], [0,0], [512,0]])
    A = np.zeros((8,9))
    for i in range(0, 8):
        if i%2 == 0:
            xc = _4_corners_list[int(i/2),0]
            xw = ref_corners[int(i/2), 0]
            yw = ref_corners[int(i/2),1]
            A[i, 0:3] = xw,yw,1
            A[i, 3:6] = 0,0,0
            A[i, 6:9] = -xc * xw, -xc * yw, -xc
        else:
            yc = _4_corners_list[int((i-1)/2), 1]
            A[i, 0:3] = 0, 0, 0
            A[i, 3:6] = xw, yw, 1
            A[i, 6:9] = -yc * xw, -yc * yw, -yc
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    v1 = vh[8,:]
    v1 = v1 / v1[8]
    H_mat = v1.reshape(3, 3)
    H_mat /= H_mat[2][2]
    return H_mat


def proj_mat(H_m, K_m):
    # H_m and K_m have to be numpy arrays.
    Kinv_H = np.matmul(np.linalg.inv(K_m), H_m)
    lam = 2 / (np.linalg.norm(Kinv_H[:, 0]) + np.linalg.norm(Kinv_H[:, 1]))
    if np.linalg.det(Kinv_H) < 0:
        B = -lam * Kinv_H
    else:
        B = lam * Kinv_H
    r1 = B[:, 0]
    r2 = B[:, 1]
    r3 = np.cross(r1, r2)
    t = B[:, 2]
    P_m = np.matmul(K_m, np.array([r1, r2, r3, t]).T)
    return P_m


def draw_box(_8_pts_aft_P, draw_frame):
    for i in range(4):
        xi, yi = int(round(_8_pts_aft_P[i][0])), int(round(_8_pts_aft_P[i][1]))
        xi_top, yi_top = int(round(_8_pts_aft_P[i + 4][0])), int(round(_8_pts_aft_P[i + 4][1]))
        if i < 3:
            xi_next, yi_next = int(round(_8_pts_aft_P[i + 1][0])), int(round(_8_pts_aft_P[i + 1][1]))
            xi_top_next, yi_top_next = int(round(_8_pts_aft_P[i + 5][0])), int(round(_8_pts_aft_P[i + 5][1]))
        else:
            xi_next, yi_next = int(round(_8_pts_aft_P[0][0])), int(round(_8_pts_aft_P[0][1]))
            xi_top_next, yi_top_next = int(round(_8_pts_aft_P[4][0])), int(round(_8_pts_aft_P[4][1]))
        cv2.line(draw_frame, (xi, yi), (xi_next, yi_next), (255, 0, 0), 2)
        cv2.line(draw_frame, (xi, yi), (xi_top, yi_top), (255, 0, 0), 2)
        cv2.line(draw_frame, (xi_top, yi_top), (xi_top_next, yi_top_next), (255, 0, 0), 2)


def _3D_to_2D(Homo, step, _frame, cube_2D_old=0):
    K = [[1406.08415449821, 2.20679787308599, 1014.13643417416],
         [0, 1417.99930662800, 566.347754321696],
         [0, 0, 1]]
    _P = proj_mat(Homo, K)

    cube_in_3D = np.array([[200, 200, 0], [200, 0, 0], [0, 0, 0], [0, 200, 0],
                           [200, 200, -200], [200, 0, -200],
                           [0, 0, -200], [0, 200, -200]]).T
    one = np.ones((1, len(cube_in_3D[0])))
    cube_in_3D_1 = np.concatenate((cube_in_3D, one), axis=0)

    cube_in_2D = np.matmul(_P, cube_in_3D_1)
    cube_in_2D = cube_in_2D / cube_in_2D[2, :]
    cube_in_2D = np.array([[int(cube_in_2D[0][i]), int(cube_in_2D[1][i])] for i in range(len(cube_in_2D[0]))])
    if step < 1:
        cube_2D_old = cube_in_2D.copy()
    else:
        cube_in_2D[4:8][:] = (cube_in_2D[4:8][:] + cube_2D_old[4:8][:]) / 2
        cube_2D_old = cube_in_2D.copy()
    draw_box(cube_in_2D, _frame)
    return cube_2D_old


# Video Reader
cap = cv2.VideoCapture("../ReferenceInput/Tag0.mp4")
success, frame = cap.read()

# Image Reader
lena = cv2.imread("../ReferenceInput/Lena.png")
x_lena = len(lena[0])
y_lena = len(lena)

old_corners_sorted = np.zeros((4, 2))
count = 1
while success:
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    frame_cube = frame.copy()
    frame_clean = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_mask = gray_frame.copy()
    _, gray_frame_threshold = cv2.threshold(gray_frame, 230, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(gray_frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if count > 1:
        contours, contours_state = return_last_contour(old_contours, contours, 3200000)
    old_contours = contours
    cv2.drawContours(frame_clean, contours, 1, (0, 0, 0), -1)
    cv2.drawContours(frame, contours, 1, (0, 255, 0), 1)
    cv2.drawContours(frame_mask, contours, 1, 0, -1)
    frame_hollow = gray_frame - frame_mask

    epsilon = 0.1 * cv2.arcLength(contours[1], True)
    corners = cv2.approxPolyDP(contours[1], epsilon, True)
    corner_points = []
    if len(corners) == 4:
        for i in range(len(corners)):
            corner_points.append([corners[i][0][0], corners[i][0][1]])
    else:
        corner_points = old_corners_sorted
    corners_sorted, center = sort_points(corner_points)
    cv2.circle(frame, (center[0], center[1]), 3, (0, 255, 0), -1)
    cv2.imwrite("frame%d.jpg" % count, frame)

    if count > 1:
        critical_point = related_point(old_corners_sorted, corners_sorted, critical_index)
    old_corners_sorted = corners_sorted
    for i in range(len(corner_points)):
        x, y = corner_points[i][0], corner_points[i][1]
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
    if count == 1:
        points_inside_xy = point_inside(corners_sorted)
        pixel_value_total = []
        for i in range(4):
            pixel_value = 0
            for j in range(len(points_inside_xy[i])):
                pixel_value = pixel_value + gray_frame_threshold[points_inside_xy[i][j][1]][points_inside_xy[i][j][0]]
            pixel_value_total.append(pixel_value)
        pixel_value_total = np.array(pixel_value_total)
        target = int(np.argmax(pixel_value_total))
        critical_point = [corners_sorted[target][0], corners_sorted[target][1]]
    critical_index = corners_sorted.index(critical_point)
    cv2.circle(frame, (critical_point[0], critical_point[1]), 3, (255, 0, 0), -1)

    # WarpPerspective
    corners_append = np.concatenate((corners_sorted, corners_sorted), axis=0)
    corners_homo = corners_append[critical_index:critical_index+4]
    lenaCorners = np.array([[x_lena, y_lena], [0, y_lena], [0, 0], [x_lena, 0]])  # [x, y]
    H_mat = homography(lenaCorners, corners_homo)

    x_max, y_max = np.amax(corners_homo, axis=0)
    x_min, y_min = np.amin(corners_homo, axis=0)

    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            if frame_hollow[i][j] != 0:
                pointLena = np.matmul(H_mat, np.array([j, i, 1]))
                pointLena /= pointLena[2]
                if pointLena[0] < 0: pointLena[0] = 0
                if pointLena[0] >= x_lena: pointLena[0] = x_lena-1
                if pointLena[1] < 0: pointLena[1] = 0
                if pointLena[1] >= y_lena: pointLena[1] = y_lena-1

                frame_clean[i][j] = lena[int(pointLena[1])][int(pointLena[0])]

    paste_width = 200
    paste_height = paste_width
    paste_size = np.array([[paste_width, paste_height], [0, paste_height], [0, 0], [paste_width, 0]])
    paste_size, _ = sort_points(paste_size)
    paste_size = np.array(paste_size)
    H = homography(corners_homo, paste_size)
    if count == 1:
        old_2d = _3D_to_2D(H, count-1, frame_cube)

    else:
        old_2d = _3D_to_2D(H, count-1, frame_cube, old_2d)

    # Video showing
    cv2.imshow("Frame", frame)
    cv2.imshow("Frame_Lena", frame_clean)
    cv2.imshow("Frame_Cube", frame_cube)

    count = count + 1

    # Control Panel
    delay = 1
    if delay >= 0 and cv2.waitKey(delay) == 32:
        cv2.waitKey(0)
    if delay >= 0 and cv2.waitKey(delay) == 27:
        break

    success, frame = cap.read()

cap.release()
cv2.destroyAllWindows()