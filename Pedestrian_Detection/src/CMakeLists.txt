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
include_directories(${THIRDPARTY_DEPS_DIR}/LibSVM/)


### ------ Module Compile ------ ###
# detection
set(PREDICT_LIST "source/ped_predict.cpp" "source/hog.cpp")
add_executable(${TARGET_NAME} ${PREDICT_LIST})

target_link_libraries(${TARGET_NAME} 
${OpenCV_LIBS}
LibSVM
)

# ped train
set(PEDESTRIAN_TRAIN "ped_train")
set(TRAIN_LIST "source/ped_train.cpp")
add_executable(${PEDESTRIAN_TRAIN}  ${TRAIN_LIST})

target_link_libraries(${PEDESTRIAN_TRAIN} 
${OpenCV_LIBS}
LibSVM
)

# ped eval 
set(PEDESTRIAN_EVAL "ped_eval")
set(EVAL_LIST "source/ped_eval.cpp")
add_executable(${PEDESTRIAN_EVAL}  ${EVAL_LIST})

target_link_libraries(${PEDESTRIAN_EVAL} 
${OpenCV_LIBS}
LibSVM
)