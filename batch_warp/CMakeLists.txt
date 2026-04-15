cmake_minimum_required(VERSION 3.15)
project(batch_warp_perspective C)

# Build system for batch_warp Python extension using Cython based on Intel(R) Integrated Performance Primitives (Intel(R) IPP)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Find Intel(R) IPP
# You may need to set IPP_ROOT or source setvars.sh before running cmake
if(DEFINED ENV{IPPROOT})
    set(IPP_ROOT $ENV{IPPROOT})
endif()

if(IPP_ROOT)
    message(STATUS "Using IPP from: ${IPP_ROOT}")
    set(IPP_INCLUDE_DIR ${IPP_ROOT}/include)

    # Find IPP static libraries for static linking
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(IPP_LIB_DIR ${IPP_ROOT}/lib/intel64)
        set(IPP_STATIC_LIBS libippi.lib libipps.lib libippvm.lib libippcore.lib)
    else()
        set(IPP_LIB_DIR ${IPP_ROOT}/lib/intel64)
        # Use static libraries (.a files)
        set(IPP_STATIC_LIBS libippi.a libipps.a libippvm.a libippcore.a)
    endif()

    # Check if static libraries exist
    foreach(lib ${IPP_STATIC_LIBS})
        find_library(IPP_${lib}_STATIC ${lib} PATHS ${IPP_LIB_DIR} NO_DEFAULT_PATH)
        if(NOT IPP_${lib}_STATIC)
            --forcemessage(FATAL_ERROR "Could not find required Intel(R) IPP static library: ${lib} in ${IPP_LIB_DIR}")
        else()
            message(STATUS "Found Intel(R) IPP static library: ${IPP_${lib}_STATIC}")
            list(APPEND IPP_LIBRARIES ${IPP_${lib}_STATIC})
        endif()
    endforeach()
endif()

if(NOT IPP_INCLUDE_DIR)
    message(FATAL_ERROR "Intel(R) IPP not found. Please set IPPROOT environment variable or source setvars.sh")
endif()

message(STATUS "Intel(R) Include Directory: ${IPP_INCLUDE_DIR}")
message(STATUS "Intel(R) Libraries: ${IPP_LIBRARIES}")

# Check for Cython
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import Cython; print(Cython.__version__)"
    OUTPUT_VARIABLE CYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE CYTHON_NOT_FOUND
)

if(CYTHON_NOT_FOUND)
    message(FATAL_ERROR "Cython not found. Install with: pip install Cython")
endif()

message(STATUS "Cython Version: ${CYTHON_VERSION}")

# Cythonize the .pyx file
set(CYTHON_PYX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/batch_warp.pyx)
set(CYTHON_C_FILE ${CMAKE_CURRENT_BINARY_DIR}/batch_warp.c)

add_custom_command(
    OUTPUT ${CYTHON_C_FILE}
    COMMAND ${Python3_EXECUTABLE} -m cython ${CYTHON_PYX_FILE} -o ${CYTHON_C_FILE}
    DEPENDS ${CYTHON_PYX_FILE}
    COMMENT "Cythonizing batch_warp.pyx"
)

# Resolve the compiler used for batch_warp_core.c explicitly so failures are
# diagnosed during CMake configuration instead of later during the build.
if(CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
    set(BATCH_WARP_CORE_COMPILER ${CMAKE_C_COMPILER})
else()
    find_program(BATCH_WARP_CORE_COMPILER NAMES icx)
    if(NOT BATCH_WARP_CORE_COMPILER)
        message(FATAL_ERROR
            "Intel(R) oneAPI DPC++/C++ Compiler is required to build batch_warp_core.c, but it was not found. "
            "Configure CMake with a proper toolchain/compiler or ensure 'icx' is available on PATH.")
    endif()
endif()

# Keep the custom-command OpenMP flag centralized for consistency.
set(BATCH_WARP_CORE_OPENMP_FLAG -qopenmp)

# Force Intel(R) oneAPI DPC++/C++ Compiler compiler for batch_warp_core.c only
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/batch_warp_core.o
    COMMAND ${BATCH_WARP_CORE_COMPILER} -O3 ${BATCH_WARP_CORE_OPENMP_FLAG} -fPIC -c ${CMAKE_CURRENT_SOURCE_DIR}/batch_warp_core.c
            -o ${CMAKE_CURRENT_BINARY_DIR}/batch_warp_core.o
            -I${IPP_INCLUDE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/batch_warp_core.c
    COMMENT "Compiling batch_warp_core.c with ${BATCH_WARP_CORE_COMPILER}"
)

# Create the Python extension module with pre-compiled object
add_library(batch_warp MODULE
    ${CYTHON_C_FILE}
    ${CMAKE_CURRENT_BINARY_DIR}/batch_warp_core.o
)

# Remove the 'lib' prefix on Unix-like systems
set_target_properties(batch_warp PROPERTIES PREFIX "")

# Set the extension suffix based on platform
if(WIN32)
    set_target_properties(batch_warp PROPERTIES SUFFIX ".pyd")
else()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYTHON_EXT_SUFFIX)
        set_target_properties(batch_warp PROPERTIES SUFFIX "${PYTHON_EXT_SUFFIX}")
    else()
        set_target_properties(batch_warp PROPERTIES SUFFIX ".so")
    endif()
endif()

# Include directories
target_include_directories(batch_warp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${IPP_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

# Suppress deprecated NumPy API warning
target_compile_definitions(batch_warp PRIVATE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

# Link libraries statically
# Force static linking by using -Wl,--whole-archive for Intel(R) IPP libraries
if(UNIX AND NOT APPLE)
    # Find Intel OpenMP static library
    find_library(INTEL_OMP_LIB libiomp5.a
        PATHS /opt/intel/oneapi/compiler/latest/lib
              /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin
              $ENV{ONEAPI_ROOT}/compiler/latest/lib
              $ENV{ONEAPI_ROOT}/compiler/latest/linux/compiler/lib/intel64_lin
        NO_DEFAULT_PATH
    )

    if(INTEL_OMP_LIB)
        message(STATUS "Found Intel OpenMP static library: ${INTEL_OMP_LIB}")
    else()
        message(WARNING "Intel OpenMP static library (libiomp5.a) not found, using dynamic linking")
    endif()

    target_link_libraries(batch_warp PRIVATE
        -Wl,--whole-archive
        ${IPP_LIBRARIES}
        -Wl,--no-whole-archive
        ${INTEL_OMP_LIB}
        ${Python3_LIBRARIES}
        -ldl  # Required for static Intel(R) IPP linkage on Linux
        -Wl,--allow-multiple-definition  # Allow multiple symbol definitions
    )
else()
    target_link_libraries(batch_warp PRIVATE
        ${IPP_LIBRARIES}
        ${Python3_LIBRARIES}
    )
endif()

# Compiler options for OpenMP
target_compile_options(batch_warp PRIVATE -fopenmp)

# Link with OpenMP (removed since we're linking libiomp5.a directly)
# target_link_options(batch_warp PRIVATE -fopenmp)

# Set output directory
set_target_properties(batch_warp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Installation (optional)
install(TARGETS batch_warp
    LIBRARY DESTINATION ${Python3_SITELIB}
    RUNTIME DESTINATION ${Python3_SITELIB}
)

# Print build information
message(STATUS "Python3 Executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 Include Dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 NumPy Include Dirs: ${Python3_NumPy_INCLUDE_DIRS}")
message(STATUS "Python3 Libraries: ${Python3_LIBRARIES}")
message(STATUS "Python Extension Suffix: ${PYTHON_EXT_SUFFIX}")
