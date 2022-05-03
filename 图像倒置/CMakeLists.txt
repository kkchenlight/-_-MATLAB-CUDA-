cmake_minimum_required(VERSION 3.5)
project(test)
find_package(CUDA REQUIRED)
include_directories("/home/aistudio/opencv/install/include/opencv4/")
link_directories("/home/aistudio/opencv/install/lib")

cuda_add_executable(test main.cu)
target_link_libraries(test opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc)
