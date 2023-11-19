import cv2
import numpy as np

cam = cv2.VideoCapture('Input.mp4')

# citesc primul frame pentru a afla dimensiunile acestuia
ret, first_frame = cam.read()

if not ret:
    print("Nu am putut citi primul frame.")
    cam.release()
    cv2.destroyAllWindows()
    exit()

init_dimensions = first_frame.shape[:2]
initial_width = init_dimensions[1]
initial_height = init_dimensions[0]

# imi creez noile dimensiuni a.s. sa pot rula 12 ferestre diferite pe ecran
small_dimensions = tuple((x // 3) - 45 for x in init_dimensions)
small_width = small_dimensions[1]
small_height = small_dimensions[0]

# coordinates of the trapezoid
upper_left = (int(0.45 * small_width), int(0.75 * small_height))
upper_right = (int(0.55 * small_width), int(0.75 * small_height))
lower_left = (int(0 * small_width), int(1 * small_height))
lower_right = (int(1 * small_width), int(1 * small_height))
trapezoid_points = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

# coordinates of the top-down frame
upper_left = (int(0 * small_width), int(0 * small_height))
upper_right = (int(1 * small_width), int(0 * small_height))
lower_left = (int(0 * small_width), int(1 * small_height))
lower_right = (int(1 * small_width), int(1 * small_height))
top_down_points = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

# Sobel matrix
sobel_vertical = np.float32([[-1, -4, -1],
                             [0, 0, 0],
                             [+1, +4, +1]])
sobel_horizontal = np.transpose(sobel_vertical)

while True:
    ret, frame = cam.read()

    if not ret:
        break

    # //*  ex 1  *\\
    #cv2.imshow('Original', frame)

    # //*  ex 2  *\\ - smaller frame
    resized_frame = cv2.resize(frame, (small_width, small_height))
    cv2.imshow('Small', resized_frame)

    # //*  ex 3  *\\ - grayscale frame
    grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('GrayScale', grayscale_frame)

    # //*  ex 4  *\\
    trapezoid_frame = np.zeros((small_height, small_width), dtype=np.uint8)
    cv2.fillConvexPoly(trapezoid_frame, trapezoid_points, 1)
    road_frame = cv2.multiply(grayscale_frame, trapezoid_frame)
    cv2.imshow('Trapezoid', trapezoid_frame * 255)
    cv2.imshow('RoadFrame', road_frame)

    # //*  ex 5  *\\
    trapezoid_points_flt = np.float32(trapezoid_points)
    top_down_points_flt = np.float32(top_down_points)
    matrix = cv2.getPerspectiveTransform(trapezoid_points_flt, top_down_points_flt)
    top_down_frame = cv2.warpPerspective(road_frame, matrix, (small_width, small_height))
    cv2.imshow("Top-Down", top_down_frame)

    # //*  ex 6  *\\
    blurred_frame = cv2.blur(top_down_frame, ksize=(7, 7))
    cv2.imshow("Blur", blurred_frame)

    # # # PANA AICI AM PREZENTAT PRIMA DATA # # #

    # //*  ex 7  *\\ -- sobel
    blurred_frame = np.float32(blurred_frame)
    blurred_frame_copy = blurred_frame.copy()
    sobel_vertical_frame = cv2.filter2D(blurred_frame, -1, sobel_vertical)
    sobel_horizontal_frame = cv2.filter2D(blurred_frame_copy, -1, sobel_horizontal)
    # cv2.imshow("Sobel Vertical", cv2.convertScaleAbs(sobel_vertical_frame))
    # cv2.imshow("Sobel Horizontal", cv2.convertScaleAbs(sobel_horizontal_frame))

    sobel_frame = np.sqrt((sobel_horizontal_frame ** 2 + sobel_vertical_frame ** 2))
    cv2.imshow("Sobel", cv2.convertScaleAbs(sobel_frame))

    # //*  ex 8  *\\ -- binarizare
    ret, bin_frame = cv2.threshold(sobel_frame, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binarized", bin_frame)

    # //*  ex 9  *\\ --
    copy_of_bin_frame = bin_frame.copy()

    five_percentage_of_columns = np.int32(np.float32(small_width) * 0.05)
    copy_of_bin_frame[:, :five_percentage_of_columns] = (
        np.zeros_like(copy_of_bin_frame[:, :five_percentage_of_columns], dtype=np.uint8))
    copy_of_bin_frame[:, -five_percentage_of_columns:] = (
        np.zeros_like(copy_of_bin_frame[:, -five_percentage_of_columns:], dtype=np.uint8))
    cv2.imshow("Binarized - 5%", copy_of_bin_frame)

    # slicing in 2 halves
    left_half_frame = copy_of_bin_frame[:, :small_width // 2]
    right_half_frame = copy_of_bin_frame[:, small_width // 2:]
    # cv2.imshow("Left Frame", left_half_frame)
    # cv2.imshow("Right Frame", right_half_frame)

    # extracting white points  ----(y, x)----
    left_points = np.argwhere(left_half_frame == 255)
    right_points = np.argwhere(right_half_frame == 255)

    # separating the x and y coordinates in different arrays
    left_x_points = left_points[:, 1]
    left_y_points = left_points[:, 0]
    right_x_points = right_points[:, 1] + (small_width // 2)
    right_y_points = right_points[:, 0]

    # //*  ex 10  *\\
    # finding the line that best passes trough points list --- (b and a) ---
    left_line_coord = np.polynomial.polynomial.polyfit(left_x_points, left_y_points, deg=1)
    right_line_coord = np.polynomial.polynomial.polyfit(right_x_points, right_y_points, deg=1)

    # y = ax + b
    # top y = 0 ==> x = -b / a
    # bottom y = h ==> x = (h-b) / a
    left_top_y = np.min(left_line_coord)
    left_top_x = np.int32((-left_line_coord[0] / left_line_coord[1]))

    left_bottom_y = small_height
    left_bottom_x = np.int32(((small_height - left_line_coord[0]) / left_line_coord[1]))

    right_top_y = np.min(right_line_coord)
    right_top_x = np.int32((-right_line_coord[0] / right_line_coord[1]))

    right_bottom_y = small_height
    right_bottom_x = np.int32(((small_height - right_line_coord[0]) / right_line_coord[1]))

    left_top = np.int32(left_top_x), np.int32(left_top_y)
    left_bottom = np.int32(left_bottom_x), np.int32(left_bottom_y)

    right_top = np.int32(right_top_x), np.int32(right_top_y)
    right_bottom = np.int32(right_bottom_x), np.int32(right_bottom_y)

    cv2.line(copy_of_bin_frame, left_top, left_bottom, (200, 0, 0), 5)
    cv2.line(copy_of_bin_frame, right_top, right_bottom, (200, 0, 0), 5)

    cv2.imshow("Lines", copy_of_bin_frame)

    # //*  ex 11  *\\
    white_left_line_frame = np.zeros(small_dimensions, dtype=np.uint8)
    cv2.line(white_left_line_frame, left_top, left_bottom, (255, 0, 0), 15)
    matrix = cv2.getPerspectiveTransform(top_down_points_flt, trapezoid_points_flt)
    white_left_line_frame = cv2.warpPerspective(white_left_line_frame, matrix,
                                                (small_width, small_height))
    white_left_line_frame = cv2.resize(white_left_line_frame, (initial_width, initial_height))

    white_right_line_frame = np.zeros(small_dimensions, dtype=np.uint8)
    cv2.line(white_right_line_frame, right_top, right_bottom, (255, 0, 0), 15)
    matrix = cv2.getPerspectiveTransform(top_down_points_flt, trapezoid_points_flt)
    white_right_line_frame = cv2.warpPerspective(white_right_line_frame, matrix,
                                                 (small_width, small_height))
    white_right_line_frame = cv2.resize(white_right_line_frame, (initial_width, initial_height))

    left_points = np.argwhere(white_left_line_frame == 255)
    right_points = np.argwhere(white_right_line_frame == 255)

    final = frame.copy()

    for point in left_points:
        x, y = point[1], point[0]
        # set the pixel to red
        final[y, x] = (50, 50, 250)

    for point in right_points:
        x, y = point[1], point[0]
        # set the pixel to green
        final[y, x] = (250, 50, 250)

    cv2.imshow("Final", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
