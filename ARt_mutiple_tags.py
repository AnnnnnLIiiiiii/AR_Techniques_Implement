import cv2
import numpy as np
import math

np.set_printoptions(threshold=np.inf)


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


def return_last_contour(old_contour, new_contour, old_hierarchy, hierarchy, threshold):
    area_old = cv2.contourArea(old_contour[1])
    area_new = cv2.contourArea(new_contour[1])
    gap = (area_old - area_new)**2
    if gap > threshold:
        return old_contour, old_hierarchy
    else:
        return new_contour, hierarchy


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


def related_list(old_lists, new_lists):
    list_num = [0, 1, 2]
    result = []
    for i in list_num:
        for j in list_num:
            for k in list_num:
                if i != j and i != k and j != k:
                    result.append([i, j, k])
    old_lists = np.array(old_lists)
    new_lists = np.array(new_lists)
    center_old = np.zeros((3, 2))
    center_new = np.zeros((3, 2))
    for i in range(len(old_lists)):
        for j in range(len(old_lists[0])):
            center_old[i] += old_lists[i][j]
            center_new[i] += new_lists[i][j]
    center_new /= 4
    center_old /= 4

    distance_list = []
    for i in range(len(result)):
        dist_total = 0
        for j in range(3):
            dist_total = dist_total + distance(center_new[j], center_old[result[i][j]])
        distance_list.append(dist_total)
    min_index = np.argmin(distance_list)
    answer = result[min_index]
    first = answer.index(0)
    second = answer.index(1)
    third = answer.index(2)

    return new_lists[first].tolist(), new_lists[second].tolist(), new_lists[third].tolist()


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


def warptransform(corners_sorted, critical_index, frame_hollow, frame_clean):
    corners_append = np.concatenate((corners_sorted, corners_sorted), axis=0)
    corners_homo = corners_append[critical_index:critical_index + 4]
    lenaCorners = np.array([[511, 511], [0, 511], [0, 0], [511, 0]])  # [x, y]
    H_mat = homography(lenaCorners, corners_homo)

    x_max, y_max = np.amax(corners_homo, axis=0)
    x_min, y_min = np.amin(corners_homo, axis=0)

    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            if frame_hollow[i][j] != 0:
                pointLena = np.matmul(H_mat, np.array([j, i, 1]))
                pointLena /= pointLena[2]
                if pointLena[0] < 0: pointLena[0] = 0
                if pointLena[0] >= 511: pointLena[0] = 511 - 1
                if pointLena[1] < 0: pointLena[1] = 0
                if pointLena[1] >= 511: pointLena[1] = 511 - 1

                frame_clean[i][j] = lena[int(pointLena[1])][int(pointLena[0])]

    return corners_homo


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
cap = cv2.VideoCapture("../ReferenceInput/multipleTags.mp4")
success, frame = cap.read()

# Image Reader
lena = cv2.imread("../ReferenceInput/Lena.png")
x_lena = len(lena[0])
y_lena = len(lena)

old_corners_sorted_1 = np.zeros((4, 2))
old_corners_sorted_2 = np.zeros((4, 2))
old_corners_sorted_3 = np.zeros((4, 2))
count = 1
while success:
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    frame_clean = frame.copy()
    frame_cube = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_mask = gray_frame.copy()
    _, gray_frame_threshold = cv2.threshold(gray_frame, 230, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(gray_frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if count > 1:
        contours, hierarchy = return_last_contour(old_contours, contours, old_hierarchy, hierarchy, 2400000)
    old_contours = contours
    old_hierarchy = hierarchy
    num_contours = len(contours)
    corners = []
    for i in range(num_contours):
        if hierarchy[0][i][-1] != -1 and hierarchy[0][i][-2] != -1 and hierarchy[0][hierarchy[0][i][-1]][-1] == -1:
            epsilon = 0.1 * cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            corners.append(approx)
            # draw it in red color
            cv2.drawContours(frame_mask, [approx], -1, 0, -1)
    corner_points_1 = []
    corner_points_2 = []
    corner_points_3 = []
    if len(corners) == 3 and len(corners[0]) == len(corners[1]) == len(corners[2]) == 4:
        for i in range(len(corners[0])):
            corner_points_1.append([corners[0][i][0][0], corners[0][i][0][1]])
            corner_points_2.append([corners[1][i][0][0], corners[1][i][0][1]])
            corner_points_3.append([corners[2][i][0][0], corners[2][i][0][1]])
    else:
        corner_points_1 = old_corners_sorted_1
        corner_points_2 = old_corners_sorted_2
        corner_points_3 = old_corners_sorted_3
    corner_points = [corner_points_1, corner_points_2, corner_points_3]
    if count > 1:
        corner_points_1, corner_points_2, corner_points_3 = related_list(old_corner_points, corner_points)

    corners_sorted_1, center_1 = sort_points(corner_points_1)
    corners_sorted_2, center_2 = sort_points(corner_points_2)
    corners_sorted_3, center_3 = sort_points(corner_points_3)

    if count > 1:
        critical_point_1 = related_point(old_corners_sorted_1, corners_sorted_1, critical_index_1)
        critical_point_2 = related_point(old_corners_sorted_2, corners_sorted_2, critical_index_2)
        critical_point_3 = related_point(old_corners_sorted_3, corners_sorted_3, critical_index_3)
    old_corners_sorted_1 = corners_sorted_1
    old_corners_sorted_2 = corners_sorted_2
    old_corners_sorted_3 = corners_sorted_3
    old_corner_points = [old_corners_sorted_1, old_corners_sorted_2, old_corners_sorted_3]

    if count == 1:
        points_inside_xy_1 = point_inside(corners_sorted_1)
        points_inside_xy_2 = point_inside(corners_sorted_2)
        points_inside_xy_3 = point_inside(corners_sorted_3)
        pixel_value_total_1 = []
        pixel_value_total_2 = []
        pixel_value_total_3 = []
        for i in range(4):
            pixel_value_1 = 0
            pixel_value_2 = 0
            pixel_value_3 = 0
            for j in range(len(points_inside_xy_1[i])):
                pixel_value_1 = pixel_value_1 + gray_frame_threshold[points_inside_xy_1[i][j][1]][points_inside_xy_1[i][j][0]]
            for j in range(len(points_inside_xy_2[i])):
                pixel_value_2 = pixel_value_2 + gray_frame_threshold[points_inside_xy_2[i][j][1]][points_inside_xy_2[i][j][0]]
            for j in range(len(points_inside_xy_3[i])):
                pixel_value_3 = pixel_value_3 + gray_frame_threshold[points_inside_xy_3[i][j][1]][points_inside_xy_3[i][j][0]]
            pixel_value_total_1.append(pixel_value_1)
            pixel_value_total_2.append(pixel_value_2)
            pixel_value_total_3.append(pixel_value_3)
        pixel_value_total_1 = np.array(pixel_value_total_1)
        pixel_value_total_2 = np.array(pixel_value_total_2)
        pixel_value_total_3 = np.array(pixel_value_total_3)
        target_1 = int(np.argmax(pixel_value_total_1))
        target_2 = int(np.argmax(pixel_value_total_2))
        target_3 = int(np.argmax(pixel_value_total_3))
        critical_point_1 = [corners_sorted_1[target_1][0], corners_sorted_1[target_1][1]]
        critical_point_2 = [corners_sorted_2[target_2][0], corners_sorted_2[target_2][1]]
        critical_point_3 = [corners_sorted_3[target_3][0], corners_sorted_3[target_3][1]]
    critical_index_1 = corners_sorted_1.index(critical_point_1)
    critical_index_2 = corners_sorted_2.index(critical_point_2)
    critical_index_3 = corners_sorted_3.index(critical_point_3)
    cv2.circle(frame, (critical_point_1[0], critical_point_1[1]), 4, (255, 0, 0), -1)
    cv2.circle(frame, (critical_point_2[0], critical_point_2[1]), 4, (0, 255, 0), -1)
    cv2.circle(frame, (critical_point_3[0], critical_point_3[1]), 4, (0, 0, 255), -1)

    # Plot all contours
    cv2.drawContours(frame, contours[0], -1, (0, 255, 0), 1)
    cv2.drawContours(frame_clean, contours, 1, (0, 0, 0), -1)
    frame_hollow = gray_frame - frame_mask

    # WarpPerspective
    corners_homo_1 = warptransform(corners_sorted_1, critical_index_1, frame_hollow, frame_clean)
    corners_homo_2 = warptransform(corners_sorted_2, critical_index_2, frame_hollow, frame_clean)
    corners_homo_3 = warptransform(corners_sorted_3, critical_index_3, frame_hollow, frame_clean)

    # Projection
    cube_width = 200
    cube_height = cube_width
    cube_corners = np.array([[cube_width, cube_height], [0, cube_height], [0, 0], [cube_width, 0]])
    cube_size, _ = sort_points(cube_corners)
    cube_size = np.array(cube_size)
    H_1 = homography(corners_homo_1, cube_size)
    H_2 = homography(corners_homo_2, cube_size)
    H_3 = homography(corners_homo_3, cube_size)
    if count == 1:
        old_2d_1 = _3D_to_2D(H_1, count-1, frame_cube)
        old_2d_2 = _3D_to_2D(H_2, count - 1, frame_cube)
        old_2d_3 = _3D_to_2D(H_3, count - 1, frame_cube)
    else:
        old_2d_1 = _3D_to_2D(H_1, count-1, frame_cube, old_2d_1)
        old_2d_2 = _3D_to_2D(H_2, count - 1, frame_cube, old_2d_2)
        old_2d_3 = _3D_to_2D(H_3, count - 1, frame_cube, old_2d_3)

    # Video showing
    cv2.imshow("Frame", frame)
    cv2.imshow("Frame_cube", frame_cube)
    cv2.imshow("Frame_clean", frame_clean)

    # Control Panel
    delay = 1
    if delay >= 0 and cv2.waitKey(delay) == 32:
        cv2.waitKey(0)
    if delay >= 0 and cv2.waitKey(delay) == 27:
        break
    count = count + 1
    success, frame = cap.read()

cap.release()
cv2.destroyAllWindows()