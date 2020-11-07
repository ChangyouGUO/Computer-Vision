set(TARGET_NAME "PedDetection")

### ------ build type ------ ###
# SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "Release")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -std=c++11 -O0 -Wall -g -ggdb")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
    SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -std=c++11 -O3 -Wall")
endif()


set(THIRDPARTY_DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")

### ------ Module Building Settings ------ ###
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source/)
include_directories(${THIRDPARTY_DEPS_DIR}/include/)


### ------ Module Compile ------ ###
set(PREDICT_LIST main.cpp hog.cpp)
add_executable(${TARGET_NAME} ${PREDICT_LIST})

set(PEDESTRIAN_TRAIN "ped_train")
set(TRAIN_LIST ped_train.cpp)
add_executable(${PEDESTRIAN_TRAIN}  ${TRAIN_LIST})

target_link_libraries(${TARGET_NAME} 
${OpenCV_LIBS}
)

target_link_libraries(${PEDESTRIAN_TRAIN} 
${OpenCV_LIBS}
${LibSVM}
)
