# Batch Warp Transformation Performance sample based on Intel® Integrated Performance Primitives (Intel® IPP) and OpenCV libraries

This project provides a performance benchmark for comparing warp affine and perspective transformations between OpenCV and Intel® Integrated Performance Primitives (Intel® IPP) libraries. It includes a high-performance Cython-based implementation that leverages Intel® IPP for optimal performance on Intel CPUs.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building the Project](#building-the-project)
- [Running the Benchmark](#running-the-benchmark)
- [Command Line Options](#command-line-options)
- [Usage Examples](#usage-examples)
- [Example Output](#example-output)
- [Project Structure](#project-structure)

## Prerequisites

### Required Software

1. **Intel® Integrated Performance Primitives**
   - Version 2026.0.0 or later
   - Available as part of Intel® oneAPI Base Toolkit or standalone download
   - Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp-download.html

2. **Intel® oneAPI DPC++/C++ Compiler**
   - Part of Intel® oneAPI Toolkit 2026.0.0 or later
   - Used for optimal compilation with OpenMP support, available as part of Intel® oneAPI Base Toolkit
   - Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html

3. **Python**
   - Python 3.9 development package or later, cython(`cython`)
   - Install python development package (`sudo apt install python3-dev`) and Cython (`pip install Cython`)

4. **OpenCV**
   - OpenCV 4.13 python package (`opencv-python`)
   - Install using `pip3 install opencv-python`

5. **CMake**
   - Version 3.15 or later

### System Requirements

- **OS:** Linux (tested on Ubuntu 24.04)
- **CPU:** Intel CPU with AVX-512 support (tested on Intel® Xeon® 6 Processors with P-cores)
- **Memory:** 32GB+ RAM (depending on batch sizes)

## Building the Project

### Step 1: Source Intel® oneAPI Environment

Before building, you need to set up the Intel® oneAPI environment:

```bash
source /opt/intel/oneapi/setvars.sh
```

Or if IPP is installed in a custom location:

```bash
export IPPROOT=/path/to/ipp
```

### Step 2: Build the Cython Extension

Navigate to the project directory and run the build script:

```bash
./_build.sh
```

The build script will:
1. Check for required dependencies
2. Configure the project using CMake
3. Compile the Cython extension with Intel® IPP
4. Create the `batch_warp` Python module

#### Manual Build (Alternative)

If you prefer to build manually:

```bash
mkdir -p build
cd build
cmake ..
make
cd ..
```

### Build Output

After a successful build, you should see:
- `batch_warp.cpython-39-x86_64-linux-gnu.so` (or similar, depending on your Python version)
- This shared library provides the IPP-accelerated warp functions

## Running the Benchmark

### Quick Start

Run the benchmark with default settings:

```bash
python3 warp_sample.py
```

This will:
- Test both OpenCV and IPP implementations
- Use the default thread-count list configured in `warp_sample.py`
- Test all image sizes: FHD, 2K, 4K, 8K, 12MP, 50MP, 200MP
- Test both data types: uint8 and float32
- Test both affine and perspective transformations by default
- Output results to `warp_benchmark_results.csv`

## Command Line Options

```
usage: warp_sample.py [-h] [--threads THREADS [THREADS ...]]
                      [--sizes {FHD,2K,4K,8K,12MP,50MP,200MP} [{FHD,2K,4K,8K,12MP,50MP,200MP} ...]]
                      [--types {uint8,float32} [{uint8,float32} ...]]
                      [--affine {True,False} [{True,False} ...]]
                      [--lib {ocv,ipp} [{ocv,ipp} ...]]
                      [--batch-size BATCH_SIZE]
                      [--output OUTPUT] [--dump-images N]

Benchmark warp transformations using OpenCV and IPP

optional arguments:
  -h, --help            show this help message and exit

  --threads THREADS [THREADS ...]
                        List of thread counts to test
                        Default: 1, 2, 3, 4, 6, 8, 16, 24, 32

  --sizes {FHD,2K,4K,8K,12MP,50MP,200MP} [{FHD,2K,4K,8K,12MP,50MP,200MP} ...]
                        List of image sizes to test (default: all sizes)

                        Available sizes:
                        - FHD: 1920x1080
                        - 2K: 2560x1440
                        - 4K: 3840x2160
                        - 8K: 7680x4320
                        - 12MP: 3024x4032
                        - 50MP: 6144x8192
                        - 200MP: 12240x16320

  --types {uint8,float32} [{uint8,float32} ...]
                        List of data types to test
                        Default: uint8 float32

  --affine {True,False} [{True,False} ...]
                        Transform types to test: True for affine, False for perspective
                        Default: True False (tests both)

  --lib {ocv,ipp} [{ocv,ipp} ...]
                        Libraries to test: ocv for OpenCV, ipp for IPP
                        Default: ocv ipp (tests both)

  --batch-size BATCH_SIZE
                        Batch size for processing (default: 128)

  --output OUTPUT       Output CSV filename (default: warp_benchmark_results.csv)

  --dump-images N       Dump N output images for visual inspection
```

## Usage Examples

### Example 1: Test Specific Thread Counts

Test only 4, 8, and 16 threads:

```bash
python3 warp_sample.py --threads 4 8 16
```

### Example 2: Test Only FHD Resolution

Test only Full HD (1920x1080) images:

```bash
python3 warp_sample.py --sizes FHD
```

### Example 3: Test Only uint8 Data Type

Test only 8-bit unsigned integer images:

```bash
python3 warp_sample.py --types uint8
```

### Example 4: Test Only IPP Library

Test only the Intel® IPP implementation:

```bash
python3 warp_sample.py --lib ipp
```

### Example 5: Test Only OpenCV Library

Test only the OpenCV implementation:

```bash
python3 warp_sample.py --lib ocv
```

### Example 6: Test Only Affine Transformation

Test only affine transformation:

```bash
python3 warp_sample.py --affine True
```

### Example 7: Test Only Perspective Transformation

Test only perspective transformation:

```bash
python3 warp_sample.py --affine False
```

### Example 8: Quick Test Configuration

Run a quick test with minimal configurations:

```bash
python3 warp_sample.py --threads 4 8 --sizes FHD 4K --types uint8 --lib ipp --affine False
```

### Example 9: High-Resolution Test

Test only high-resolution images:

```bash
python3 warp_sample.py --sizes 50MP 200MP --threads 16 32
```

### Example 10: Custom Batch Size

Test with a larger batch size:

```bash
python3 warp_sample.py --batch-size 256 --sizes FHD 2K
```

### Example 11: Save Results to Custom File

Save benchmark results to a specific CSV file:

```bash
python3 warp_sample.py --output my_benchmark_results.csv
```

### Example 12: Dump Output Images

Run benchmark and save the first 10 dewarped images for visual inspection:

```bash
python3 warp_sample.py --sizes FHD --dump-images 10
```

### Example 13: Production Benchmark

Run a comprehensive benchmark for production analysis:

```bash
python3 warp_sample.py --threads 8 16 32 \
                       --sizes FHD 4K 8K \
                       --types uint8 \
                       --lib ipp \
                       --affine False \
                       --batch-size 256 \
                       --output production_benchmark.csv
```

### Example 14: Compare Affine vs Perspective for Both Libraries

Compare both transform types across both libraries:

```bash
python3 warp_sample.py --affine True False --lib ocv ipp --sizes FHD --threads 8 16
```

### Example 15: IPP-Only Full Test

Test IPP across all configurations:

```bash
python3 warp_sample.py --lib ipp --affine True False --types uint8 float32
```

## Example Output

### Console Output

```
Batch size: 128
Thread counts: [1, 2, 3, 4, 6, 8, 16, 24, 32]
Image sizes: ['FHD', '2K', '4K', '8K', '12MP', '50MP', '200MP']
Data types: ['uint8', 'float32']
Libraries: ['ocv', 'ipp']
Transform types: ['Affine', 'Perspective']
Results will be written to warp_benchmark_results.csv

Warp Affine 8u OpenCV FHD (1920x1080) based on threads number:
1: 712.787 ms (179.58 fps)
4: 239.310 ms (534.87 fps)
8: 195.843 ms (653.59 fps)
16: 110.458 ms (1158.81 fps)
32: 70.701 ms (1810.45 fps)

Warp Affine 8u IPP FHD (1920x1080) based on threads number:
1: 312.456 ms (409.73 fps)
4: 89.123 ms (1436.15 fps)
8: 52.678 ms (2431.24 fps)
16: 31.234 ms (4099.61 fps)
32: 24.567 ms (5210.37 fps)
...
```

### CSV Output Format

The benchmark generates a CSV file with the following columns (the numbers are just an example):

```csv
lib,transform,type,size,n_threads,fps,time(ms),speedup,efficiency
IPP,Perspective,32f,FHD,1,276.5,347.189,1.00,1.00
IPP,Perspective,32f,FHD,2,542.4,176.966,1.96,0.98
IPP,Perspective,32f,FHD,3,790.2,121.486,2.86,0.95
...
```

**Column Descriptions:**
- `lib`: Library used (OpenCV or IPP)
- `type`: Data type (8u for uint8, 32f for float32)
- `size`: Image resolution (FHD, 2K, 4K, etc.)
- `n_threads`: Number of threads used
- `fps`: Frames per second (throughput)
- `time(ms)`: Processing time in milliseconds for the entire batch
- `speedup`: Speedup of multithreaded computing compared to single-threaded
- `efficiency`: Efficiency of multithreaded computing

## Performance Tips

1. **Thread Count Selection:**
   - For best performance, use thread counts that match your CPU's physical core count
   - Avoid over-subscription (more threads than physical cores)

2. **NUMA Considerations:**
   - For systems with multiple NUMA nodes, pin execution to a single NUMA node
   - Use `numactl` for NUMA-aware execution

3. **CPU Frequency:**
   - Ensure CPU is running at maximum frequency (disable power-saving modes)
   - Check with: `cat /proc/cpuinfo | grep MHz`

4. **Batch Size:**
   - Larger batch sizes reduce per-image overhead
   - Balance batch size with available memory

## Troubleshooting

### Build Issues

**Problem:** "Intel® IPP not found"
```
Solution: Source the oneAPI environment:
  source /opt/intel/oneapi/setvars.sh
```

**Problem:** "Cython not found"
```
Solution: Install Cython:
  pip install Cython
```

**Problem:** "unrecognized command-line option '-qopenmp-link=static'"
```
Solution: The CMakeLists.txt has been updated to use direct static linking.
         Ensure you're using the latest version of the project.
```

### Runtime Issues

**Problem:** "ImportError: No module named 'batch_warp'"
```
Solution: Ensure the .so file is built and in the same directory as warp_sample.py
         or in your PYTHONPATH.
```

**Problem:** Low performance
```
Solution:
  1. Verify CPU frequency is not throttled
  2. Check NUMA node assignment
  3. Ensure IPP library is properly loaded
  4. Verify you're using the IPP implementation, not just OpenCV
```

## References

- [Intel® IPP](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html)
- [OpenCV Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)

## License

See LICENSE file for details.
