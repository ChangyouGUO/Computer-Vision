cmake_minimum_required(VERSION 2.8.12)
project("PedDetection")

### ------ Opencv ------ ###
set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

### ------- set output dir ------- ###
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/buildDebug/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/buildRelease/bin)


include_directories(${OpenCV_INCLUDE_DIRS})

### ------ Build Sub-Modules ------ ###
add_subdirectory(deps)
add_subdirectory(src)