cmake_minimum_required(VERSION 3.19)
project(opencv_tutorial)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(opencv_tutorial main.cpp quickdemo.h quickdemo.cpp)
target_link_libraries(opencv_tutorial ${OpenCV_LIBS})