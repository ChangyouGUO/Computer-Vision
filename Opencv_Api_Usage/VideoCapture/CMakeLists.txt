cmake_minimum_required(VERSION 3.5.1)

project(videocapture)

SET (CMAKE_CXX_FLAGS "-std=c++11 -O3")

set (OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(videocapture video_operation.cpp)

target_link_libraries(videocapture 
${OpenCV_LIBS}
)