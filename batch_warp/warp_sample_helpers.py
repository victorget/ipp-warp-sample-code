import cv2
import numpy as np
import random
import time
import csv

IMAGE_SIZES = {
    'FHD': [1920, 1080],    # 2MP video
    '2K': [2560, 1440],     # 4MP video
    '4K': [3840, 2160],     # 8MP video
    '8K': [7680, 4320],     # 33MP video
    '12MP': [3024, 4032],   # Typical phone camera
    '50MP': [6144, 8192],   # High-end phone camera
    '200MP': [12240, 16320] # High-end camera ultra resolution
}

# Default configuration
DEFAULT_NUM_THREADS = 128
DEFAULT_MP_SIZE = '12MP'
DEFAULT_BATCH_SIZE = DEFAULT_NUM_THREADS

def generate_text_page(id, img_width, img_height):

    """
    Generate a grayscale image with text lines.

    Args:
        id: Page number
        img_width: Width of the image
        img_height: Height of the image

    Returns:
        Grayscale image with text
    """
    # Create a white grayscale image for the text page
    page_img = np.ones((img_height, img_width), dtype=np.uint8) * 255

    # Calculate font and spacing to fit specified number of lines
    top_margin = 30
    bottom_margin = 30
    available_height = img_height - top_margin - bottom_margin


    # Generate text lines
    num_lines=30
    text_lines = []
    for i in range(num_lines):
        if i == 0:
            text_lines.append(f"Line 1: Perspective Transform Demo, Page {id}")
        elif i == 1:
            text_lines.append("Line 2: This is a sample text document")
        elif i < 8:
            templates = [
                "with multiple lines of content.",
                "Each line will be warped using",
                "a perspective transformation.",
                "The corners are randomly offset",
                "to create a realistic effect.",
                "This simulates a photographed page."
            ]
            text_lines.append(f"Line {i+1}: {templates[(i-2) % len(templates)]}")
        else:
            text_lines.append(f"Line {i+1}: Sample text content for line {i+1}.")

    # Draw text lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_spacing = available_height / num_lines
    font_scale = (line_spacing * 0.6) / 30.0  # 30 pixels is approximate height at scale 1.0
    font_thickness = max(1, int(font_scale * 2))
    text_color = 0  # Black text
    y_position = top_margin + line_spacing * 0.8  # Start at first baseline
    for line in text_lines:
        cv2.putText(page_img, line, (30, int(y_position)), font, font_scale,
                    text_color, font_thickness, cv2.LINE_AA)
        y_position += line_spacing

    return page_img


def generate_warped_batch(
    batch_size,
    img_width,
    img_height,
    np_type=np.uint8,
    seed=None
):
    """
    Generate a batch of warped text images with different perspective transformations.

    Args:
        batch_size: Number of warped images to generate
        img_width: Width of the source image
        img_height: Height of the source image
        np_type: Numpy data type for the output images
        seed: Random seed for reproducibility (optional)

    Returns:
        warped_images: Array of warped images (batch_size, H, W)
        warp_points: Array of destination points for each warp (batch_size, 4, 2)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Random offset range: 30% of image dimensions
    offset_range_x = int(img_width * 0.3)
    offset_range_y = int(img_height * 0.3)

    # Define source points (corners of the original page)
    src_points = np.float32([
        [0, 0],                    # Top-left
        [img_width - 1, 0],        # Top-right
        [img_width - 1, img_height - 1],  # Bottom-right
        [0, img_height - 1]        # Bottom-left
    ])

    # Prepare arrays to store results
    warped_images = np.zeros((batch_size, img_height, img_width), dtype=np_type)
    warp_points = np.zeros((batch_size, 4, 2), dtype=np.float32)

    # Generate batch_size warped images
    for idx in range(batch_size):
        # Generate random offsets for each corner
        offsets = [
            [random.randint(-offset_range_x, offset_range_x),
             random.randint(-offset_range_y, offset_range_y)],  # Top-left
            [random.randint(-offset_range_x, offset_range_x),
             random.randint(-offset_range_y, offset_range_y)],  # Top-right
            [random.randint(-offset_range_x, offset_range_x),
             random.randint(-offset_range_y, offset_range_y)],  # Bottom-right
            [random.randint(-offset_range_x, offset_range_x),
             random.randint(-offset_range_y, offset_range_y)]   # Bottom-left
        ]

        # Calculate destination points with offsets
        dst_points_temp = np.float32([
            [offsets[0][0], offsets[0][1]],  # Top-left
            [img_width + offsets[1][0], offsets[1][1]],  # Top-right
            [img_width + offsets[2][0], img_height + offsets[2][1]],  # Bottom-right
            [offsets[3][0], img_height + offsets[3][1]]   # Bottom-left
        ])

        # Find bounding box of warped points
        min_x = np.floor(np.min(dst_points_temp[:, 0]))
        max_x = np.ceil(np.max(dst_points_temp[:, 0]))
        min_y = np.floor(np.min(dst_points_temp[:, 1]))
        max_y = np.ceil(np.max(dst_points_temp[:, 1]))

        # Adjust destination points to fit within canvas with 5% margin
        margin_factor = 0.95
        available_width = img_width * margin_factor
        available_height = img_height * margin_factor

        warped_width = max_x - min_x
        warped_height = max_y - min_y

        scale_x = available_width / warped_width if warped_width > available_width else 1.0
        scale_y = available_height / warped_height if warped_height > available_height else 1.0
        scale = min(scale_x, scale_y)

        # Apply scaling and centering
        center_offset_x = (img_width - warped_width * scale) / 2
        center_offset_y = (img_height - warped_height * scale) / 2

        dst_points = np.float32([
            [(dst_points_temp[i][0] - min_x) * scale + center_offset_x,
             (dst_points_temp[i][1] - min_y) * scale + center_offset_y]
            for i in range(4)
        ])

        # Store warp points
        warp_points[idx] = dst_points

        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Generate the source text page for this image
        source_image = generate_text_page(idx, img_width, img_height)

        # Apply perspective warp
        warped = cv2.warpPerspective(source_image, matrix, (img_width, img_height),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=128)

        warped_images[idx] = warped.astype(np_type)

    return warped_images, warp_points

def measured_run(func, *args, num_runs=10, **kwargs):
    """Utility function to time any function call multiple times and return the median time"""
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)

    # Return median time
    times.sort()
    return times[len(times) // 2]

def dump_output(dewarped_images, max_images=5):
    """Save dewarped images to disk"""
    if dewarped_images is not None:
        for i in range(dewarped_images.shape[0]):
            if i < max_images:
                filename = f'dewarped_text_{i:03d}.png'
                cv2.imwrite(filename, dewarped_images[i])
                print(f"Dewarped image {i+1} saved: {filename}")

def dump_results_to_csv(results, filename='warp_benchmark_results.csv'):
    """
    Write benchmark results to a CSV file.

    Args:
        results: List of dictionaries with keys: img_size, data_type, num_threads, times_ms
        filename: Output CSV filename (default: 'warp_benchmark_results.csv')
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['img_size', 'data_type', 'num_threads', 'times_ms', 'fps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results written to {filename}")
