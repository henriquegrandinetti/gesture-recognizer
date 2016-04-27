cmake_minimum_required(VERSION 2.8)

project(gestureRecognizer)

find_package(OpenCV REQUIRED)

add_executable(gestureRecognizer main.cpp)

target_link_libraries(gestureRecognizer ${OpenCV_LIBS})
