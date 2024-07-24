import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML, Image

def display_images(image_list, columns=2, rows=5, cmap=None):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            image_list: List of np.arrays compatible with plt.imshow.
            columns (Default = 2): Number of columns in the figure.
            rows (Default = 5): Number of rows in the figure.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(10, 11))
    for i, img in enumerate(image_list):
        plt.subplot(rows, columns, i+1)
        cmap = 'gray' if len(img.shape) == 2 else cmap
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

# Reading in the test images
input_images = [plt.imread(img) for img in glob.glob('test_images/*.jpg')]
# display_images(input_images)

def rgb_color_selection(img):
    """
    Apply color selection to RGB images to blackout everything except for white and yellow lane lines.
        Parameters:
            img: An np.array compatible with plt.imshow.
    """
    # White color mask
    lower_white = np.uint8([200, 200, 200])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(img, lower_white, upper_white)

    # Yellow color mask
    lower_yellow = np.uint8([175, 175, 0])
    upper_yellow = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)

    # Combine white and yellow masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    result_image = cv2.bitwise_and(img, img, mask=combined_mask)

    return result_image

# display_images(list(map(rgb_color_selection, input_images)))

def to_hsv(img):
    """
    Convert RGB images to HSV.
        Parameters:
            img: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# display_images(list(map(to_hsv, input_images)))

def hsv_color_selection(img):
    """
    Apply color selection to the HSV images to blackout everything except for white and yellow lane lines.
        Parameters:
            img: An np.array compatible with plt.imshow.
    """
    # Convert the input image to HSV
    hsv_img = to_hsv(img)

    # White color mask
    lower_white = np.uint8([0, 0, 210])
    upper_white = np.uint8([255, 30, 255])
    white_mask = cv2.inRange(hsv_img, lower_white, upper_white)

    # Yellow color mask
    lower_yellow = np.uint8([18, 80, 80])
    upper_yellow = np.uint8([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    # Combine white and yellow masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    result_image = cv2.bitwise_and(img, img, mask=combined_mask)

    return result_image

# display_images(list(map(hsv_color_selection, input_images)))

def to_hsl(img):
    """
    Convert RGB images to HSL.
        Parameters:
            img: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

# display_images(list(map(to_hsl, input_images)))

def hsl_color_selection(img):
    """
    Apply color selection to the HSL images to blackout everything except for white and yellow lane lines.
        Parameters:
            img: An np.array compatible with plt.imshow.
    """
    # Convert the input image to HSL
    hsl_img = to_hsl(img)

    # White color mask
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hsl_img, lower_white, upper_white)

    # Yellow color mask
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(hsl_img, lower_yellow, upper_yellow)

    # Combine white and yellow masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    result_image = cv2.bitwise_and(img, img, mask=combined_mask)

    return result_image

# display_images(list(map(hsl_color_selection, input_images)))

color_selected_imgs = list(map(hsl_color_selection, input_images))

def to_gray(img):
    """
    Convert images to gray scale.
        Parameters:
            img: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gray_imgs = list(map(to_gray, color_selected_imgs))
# display_images(gray_imgs)

def gaussian_blur(img, kernel_size=13):
    """
    Apply Gaussian filter to the input image.
        Parameters:
            img: An np.array compatible with plt.imshow.
            kernel_size (Default = 13): The size of the Gaussian kernel will affect the performance of the detector.
            It must be an odd number (3, 5, 7, ...).
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

blurred_imgs = list(map(gaussian_blur, gray_imgs))
# display_images(blurred_imgs)

def canny_edge_detection(img, low_thresh=50, high_thresh=150):
    """
    Apply Canny Edge Detection algorithm to the input image.
        Parameters:
            img: An np.array compatible with plt.imshow.
            low_thresh (Default = 50).
            high_thresh (Default = 150).
    """
    return cv2.Canny(img, low_thresh, high_thresh)

edge_detected_imgs = list(map(canny_edge_detection, blurred_imgs))
# display_images(edge_detected_imgs)

def region_of_interest(img):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            img: An np.array compatible with plt.imshow.
    """
    mask = np.zeros_like(img)
    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # We could have used fixed numbers as the vertices of the polygon,
    # but they will not be applicable to images with different dimensions.
    rows, cols = img.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

masked_imgs = list(map(region_of_interest, edge_detected_imgs))
# display_images(masked_imgs)

def hough_lines_detection(img):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            img: The output of a Canny transform.
    """
    rho = 1              # Distance resolution of the accumulator in pixels.
    theta = np.pi/180    # Angle resolution of the accumulator in radians.
    threshold = 20       # Only lines that are greater than threshold will be returned.
    min_line_len = 20    # Line segments shorter than that are rejected.
    max_line_gap = 300   # Maximum allowed gap between points on the same line to link them
    return cv2.HoughLinesP(img, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=min_line_len, maxLineGap=max_line_gap)

hough_lines = list(map(hough_lines_detection, masked_imgs))

def draw_lines_on_image(img, lines, color=[255, 0, 0], thickness=2):
    """
    Draw lines onto the input image.
        Parameters:
            img: An np.array compatible with plt.imshow.
            lines: The lines we want to draw.
            color (Default = red): Line color.
            thickness (Default = 2): Line thickness.
    """
    img_copy = np.copy(img)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

line_images = []
for img, lines in zip(input_images, hough_lines):
    line_images.append(draw_lines_on_image(img, lines))
display_images(line_images)

def process_pipeline(img):
    """
    Combine all the steps above to process the input images.
        Parameters:
            img: An np.array compatible with plt.imshow.
    """
    color_selected = hsl_color_selection(img)
    gray_img = to_gray(color_selected)
    smoothed_img = gaussian_blur(gray_img)
    edges = canny_edge_detection(smoothed_img)
    region_selected = region_of_interest(edges)
    lines = hough_lines_detection(region_selected)
    return draw_lines_on_image(img, lines)

# display_images(list(map(process_pipeline, input_images)))

def process_video_frame(frame):
    """
    Process the input images and return the one with detected lines.
        Parameters:
            frame: An np.array compatible with plt.imshow.
    """
    return process_pipeline(frame)

# Create output directory if it doesn't exist
output_directory = 'test_videos_output'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Output a sample video
output_video = os.path.join(output_directory, 'solidWhiteRight.mp4')
input_video = VideoFileClip('test_videos/solidWhiteRight.mp4')
processed_video = input_video.fl_image(process_video_frame)  # NOTE: this function expects color images!!
processed_video.write_videofile(output_video, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output_video))
