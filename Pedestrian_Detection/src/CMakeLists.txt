set(TARGET_NAME "PedDetection")

### ------ build type ------ ###
# SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
    SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
endif()

### ------ Module Building Settings ------ ###
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source)
file(GLOB PedDetection_all_srcs LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")

### ------ Module Compile ------ ###
add_executable(${TARGET_NAME} ${PedDetection_all_srcs})
target_link_libraries(${TARGET_NAME} 
                      ${OpenCV_LIBS})



