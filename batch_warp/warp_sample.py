import math
import cv2
import numpy as np
import csv
import argparse

from warp_sample_helpers import generate_warped_batch
from warp_sample_helpers import measured_run
from warp_sample_helpers import dump_output
from warp_sample_helpers import IMAGE_SIZES, DEFAULT_BATCH_SIZE
from batch_warp import batch_warp_transform as batch_warp_transform_ipp

def affine_transform(image, quad):
    h, w = image.shape
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1]])
    M = cv2.getAffineTransform(quad[:3], dst)
    return cv2.warpAffine(image, M, (w, h))

def perspective_transform(image, quad):
    h, w = image.shape
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, H, (w, h))

def batch_warp_transform_ocv(warped_images, dewarped_images, warp_points, affine=False, num_threads=1):
    cv2.setNumThreads(num_threads)
    if(affine):
        for i, (image, point) in enumerate(zip(warped_images, warp_points)):
            dewarped_images[i] = affine_transform(image, point)
    else:
        for i, (image, point) in enumerate(zip(warped_images, warp_points)):
            dewarped_images[i] = perspective_transform(image, point)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Benchmark warp transformations using OpenCV and IPP')
parser.add_argument('--threads', type=int, nargs='+', default=[1,2,3,4,6,8,16,24,32],
                    help='List of thread counts to test (default: 1 2 3 4 6 8 16 24 32)')
parser.add_argument('--sizes', type=str, nargs='+', default=["FHD", "2K", "4K", "8K", "12MP", "50MP", "200MP"],
                    choices=list(IMAGE_SIZES.keys()),
                    help='List of image sizes to test (default: all sizes)')
parser.add_argument('--types', type=str, nargs='+', default=["uint8", "float32"],
                    choices=["uint8", "float32"],
                    help='List of data types to test (default: uint8 float32)')
parser.add_argument('--affine', type=str, nargs='+', default=["True", "False"],
                    choices=["True", "False"],
                    help='Transform types to test: True for affine, False for perspective (default: True False)')
parser.add_argument('--lib', type=str, nargs='+', default=["ocv", "ipp"],
                    choices=["ocv", "ipp"],
                    help='Libraries to test: ocv for OpenCV, ipp for IPP (default: ocv ipp)')
parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                    help=f'Batch size for processing (default: {DEFAULT_BATCH_SIZE})')
parser.add_argument('--output', type=str, default='warp_benchmark_results.csv',
                    help='Output CSV filename (default: warp_benchmark_results.csv)')
parser.add_argument('--dump-images', type=int, metavar='N',
                    help='Dump N output images for visual inspection')

args = parser.parse_args()

# Convert data type strings to numpy types
data_type_map = {"uint8": np.uint8, "float32": np.float32}
data_types = [data_type_map[dt] for dt in args.types]

# Convert affine strings to booleans
affine_modes = [True if a == "True" else False for a in args.affine]

# Build library list based on args
lib_map = {"ocv": batch_warp_transform_ocv, "ipp": batch_warp_transform_ipp}
selected_libs = [(lib_map[lib], lib) for lib in args.lib]

# main program
batch_size = args.batch_size
thread_counts = args.threads
img_sizes = args.sizes

# Open CSV file and write header
csv_filename = args.output

print(f"\nBatch size: {batch_size}")
print(f"Thread counts: {thread_counts}")
print(f"Image sizes: {img_sizes}")
print(f"Data types: {args.types}")
print(f"Libraries: {args.lib}")
print(f"Transform types: {['Affine' if a else 'Perspective' for a in affine_modes]}")
print(f"Results will be written to {csv_filename}")

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['lib','transform', 'type', 'size', 'n_threads', 'fps', 'time(ms)', 'speedup', 'efficiency'])

for affine in affine_modes:
    transform_name = "Affine" if affine else "Perspective"
    for batch_warp_transform, lib_name in selected_libs:
        lib_display_name = "OpenCV" if lib_name == "ocv" else "IPP"
        for data_type in data_types:
            data_type_name = "8u" if data_type == np.uint8 else "32f"
            for img_size in img_sizes:
                w, h = IMAGE_SIZES[img_size]
                print(f"\nWarp {transform_name} {data_type_name} {lib_display_name} {img_size} ({w}x{h}) based on threads number:")

                warped_images, warp_points = generate_warped_batch(batch_size=batch_size, img_width=w, img_height=h, np_type=data_type, seed=42)
                dewarped_images = np.zeros((batch_size, h, w), dtype=data_type)
                base_time = measured_run(batch_warp_transform, warped_images, dewarped_images, warp_points, affine, num_threads=1)

                for num_threads in thread_counts:
                    if num_threads == 1:
                        time_ms = base_time
                    else:
                        time_ms = measured_run(batch_warp_transform, warped_images, dewarped_images, warp_points, affine, num_threads=num_threads)
                    fps = batch_size * 1000 / time_ms
                    print(f"{num_threads}: {time_ms:.3f} ms ({fps:.2f} fps)")
                    speedup = base_time / time_ms
                    efficiency = speedup / num_threads
                    # Write immediately to CSV file
                    with open(csv_filename, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        truncated_fps = math.floor(fps * 10) / 10
                        truncated_time = math.floor(time_ms * 1000) / 1000
                        csv_writer.writerow([lib_display_name, transform_name, data_type_name, img_size, num_threads, truncated_fps, truncated_time, f"{speedup:.2f}", f"{efficiency:.2f}"])

# Dump output images if requested
if args.dump_images:
    print(f"\nDumping {args.dump_images} output images...")
    dump_output(dewarped_images, max_images=args.dump_images)
