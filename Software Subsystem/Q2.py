import cv2
import numpy as np
import matplotlib.pyplot as plt

def count_grey_pixels(img_gray, y, x_start, x_end, grey_range):
    """Count pixels within a specific grey range in a horizontal line segment."""
    line = img_gray[y, x_start:x_end]
    return np.sum((line >= grey_range[0]) & (line <= grey_range[1]))

def optimize_roi_top(img_gray, base_top_y, base_left_x, base_right_x, roi_height, grey_range):
    """Find optimal top line for ROI based on road surface color."""
    height, width = img_gray.shape
    base_width = base_right_x - base_left_x
    initial_top_width = int(base_width * 0.3)
    half_top_width = initial_top_width // 2
    base_x_center = (base_left_x + base_right_x) // 2

    best_y = base_top_y - roi_height
    max_grey_count = -1

    # Search vertically to find best road surface line
    for y in range(max(best_y - 20, 0), min(best_y + 20, height)):
        x_start = max(base_x_center - half_top_width, 0)
        x_end = min(base_x_center + half_top_width, width)
        grey_count = count_grey_pixels(img_gray, y, x_start, x_end, grey_range)
        if grey_count > max_grey_count:
            max_grey_count = grey_count
            best_y = y

    left_x = base_x_center - half_top_width
    right_x = base_x_center + half_top_width

    # Expand line horizontally to cover full road width
    while left_x > 0 and grey_range[0] <= img_gray[best_y, left_x-1] <= grey_range[1]:
        left_x -= 1

    while right_x < width-1 and grey_range[0] <= img_gray[best_y, right_x+1] <= grey_range[1]:
        right_x += 1

    # Add small margin for stability
    extra_len = int(0.02 * (right_x - left_x))
    left_x = max(0, left_x - extra_len)
    right_x = min(width-1, right_x + extra_len)

    return best_y, left_x, right_x

def process_road_image(image_path,
                       sky_brightness_threshold=150,
                       min_brightness_diff=30,
                       grey_range=(50, 200)):

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    blue_channel = img[:, :, 0]

    grid_rows = 30
    row_height = height // grid_rows

    blue_brightness = [np.mean(blue_channel[i*row_height:(i+1)*row_height, :]) for i in range(grid_rows)]

    upper_section = blue_brightness[:grid_rows//3]
    lower_section = blue_brightness[-grid_rows//3:]

    max_brightness = np.max(upper_section)
    bottom_brightness = np.mean(lower_section)

    brightness_diff = max_brightness - bottom_brightness

    use_sky_detection = max_brightness > sky_brightness_threshold and brightness_diff > min_brightness_diff

    roi_mask = np.zeros((height, width), dtype=np.uint8)

    roi_debug_img = img_rgb.copy()

    if use_sky_detection:
        detection_method = "Sky-based ROI"

        blue_gradient = np.gradient(blue_brightness)
        steep_falls = np.where(blue_gradient < -5)[0]

        horizon_row = (np.min(steep_falls) + np.max(steep_falls)) // 2 if len(steep_falls) > 0 else grid_rows // 4

        horizon_y = horizon_row * row_height
        roi_top_y = min(horizon_y + row_height, height - 1)

        roi_vertices = np.array([[
            (int(width*0.35), roi_top_y),
            (int(width*0.05), height),
            (int(width*0.95), height),
            (int(width*0.65), roi_top_y)]], dtype=np.int32)

        cv2.fillPoly(roi_mask, [roi_vertices], 255)

    else:
        detection_method = "Stacked ROIs (No clear sky detected)"

        roi_height = height // 5

        bottom_base_y = height - 1
        bottom_base_left = int(width * 0.05)
        bottom_base_right = int(width * 0.95)

        # Build ROI from bottom up, layer by layer
        bottom_best_y, bottom_left, bottom_right = optimize_roi_top(
            img_gray, bottom_base_y, bottom_base_left, bottom_base_right,
            roi_height, grey_range)

        bottom_roi_vertices = np.array([[
            (bottom_left, bottom_best_y),
            (bottom_base_left, bottom_base_y),
            (bottom_base_right, bottom_base_y),
            (bottom_right, bottom_best_y)]], dtype=np.int32)

        middle_best_y, middle_left, middle_right = optimize_roi_top(
            img_gray, bottom_best_y, bottom_left, bottom_right,
            roi_height, grey_range)

        middle_roi_vertices = np.array([[
            (middle_left, middle_best_y),
            (bottom_left, bottom_best_y),
            (bottom_right, bottom_best_y),
            (middle_right, middle_best_y)]], dtype=np.int32)

        top_best_y, top_left, top_right = optimize_roi_top(
            img_gray, middle_best_y, middle_left, middle_right,
            roi_height, grey_range)

        top_roi_vertices = np.array([[
            (top_left, top_best_y),
            (middle_left, middle_best_y),
            (middle_right, middle_best_y),
            (top_right, top_best_y)]], dtype=np.int32)

        cv2.fillPoly(roi_mask, [bottom_roi_vertices, middle_roi_vertices, top_roi_vertices], 255)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Define color ranges for yellow and white lines
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([40, 255, 255])
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])

    # Create masks for yellow and white colors
    yellow_mask = cv2.inRange(img_hsv, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(img_hsv, white_lower, white_upper)

    # Apply ROI mask to color masks
    yellow_mask = cv2.bitwise_and(yellow_mask, roi_mask)
    white_mask = cv2.bitwise_and(white_mask, roi_mask)

    kernel_dilate = np.ones((5, 5), np.uint8)  # You can adjust kernel size as needed
    white_mask = cv2.dilate(white_mask, kernel_dilate, iterations=1)
    yellow_mask = cv2.dilate(yellow_mask, kernel_dilate, iterations=1)
    
    
    # Create highlighted image
    highlighted_img = img_rgb.copy()
    highlighted_img[white_mask > 0] = [0, 255, 255]   # Cyan for white lanes
    highlighted_img[yellow_mask > 0] = [255, 255, 0]  # Yellow for yellow lanes
    plt.figure(figsize=(10, 8))

    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    cv2.imwrite('original.jpg',img)
    plt.subplot(222)
    plt.imshow(highlighted_img)
    plt.title('Highlighted Lanes')
    plt.axis('off')
    cv2.imwrite('highlighted.jpg',cv2.cvtColor(highlighted_img, cv2.COLOR_RGB2BGR))
    plt.subplot(223)
    plt.imshow(white_mask, cmap='gray')
    plt.title('White Lane Mask')
    plt.axis('off')
    cv2.imwrite('white.jpg',white_mask)
    plt.subplot(224)
    plt.imshow(yellow_mask, cmap='gray')
    plt.title('Yellow Lane Mask')
    plt.axis('off')
    cv2.imwrite('yellow.jpg',yellow_mask)
    plt.tight_layout()
    plt.show()

        
    return {
        'original': img_rgb,
        'highlighted': highlighted_img,
        'white_mask': white_mask,
        'yellow_mask': yellow_mask,
        'detection_method': detection_method
    }

# Example usage
if __name__ == "__main__":
    image_path = 'Road1.jpg'  
    process_road_image(image_path)