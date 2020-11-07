cmake_minimum_required(VERSION 2.8)

project(LibSVM)

SET( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

include_directories(${CMAKE_SOURCE_DIR}/deps/include/LibSVM/)

add_library("svm" STATIC svm.cpp)