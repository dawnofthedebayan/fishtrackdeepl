cmake_minimum_required(VERSION 3.9)
project( VideoCapture )
find_package( OpenCV REQUIRED )

#include_directories("/home/fmthhadmin/mehmood/pylon5/include")
#link_directories("/home/fmthhadmin/mehmood/pylon5/bin")
#link_directories("/home/fmthhadmin/mehmood/pylon5/lib64")
#add_executable( VideoCapture video_capture.cpp )
#target_link_libraries( VideoCapture ${OpenCV_LIBS} pylonutility pylonbase )
set(PYLON_ROOT "/home/fmthhadmin/Debayan/pylon5/pylon5/")

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PYLON_ROOT}/include")

#GLEW
#find_package(OpenGL REQUIRED)
#include_directories(${OPENGL_INCLUDE_DIR})

#set(CORELIBS ${QT_LIBRARY} ${OPENGL_LIBRARY} )


#CUDA
enable_language(CUDA)
find_package(CUDA 10.1 REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA include: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA lib: ${CUDA_LIBRARIES}")
endif()
#find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

include_directories(${CUDA_INCLUDE_DIRS})

#SET(CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)
#SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
#link_libraries("/usr/local/cuda-10.1/lib64")



#############################################

# Construct path to CUDA driver API lib (not provided by FindCUDA)
get_filename_component(CUDA_LIB_DIR ${CUDA_cudart_static_LIBRARY} DIRECTORY)
find_library(CUDA_LIB NAMES cuda HINTS ${CUDA_LIB_DIR})


#############################################


#find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
#add_executable(binary_linking_to_cudart AppEncCuda.cpp)
#target_include_directories(binary_linking_to_cudart PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#target_link_libraries(binary_linking_to_cudart ${CUDART_LIBRARY})

#Video Codec
set(VIDEO_CODEC_ROOT "/home/fmthhadmin/Debayan/videocodec_nvidia/")
include_directories("/usr/local/lib/pkgconfig")
include_directories("${VIDEO_CODEC_ROOT}/include")
include_directories("${VIDEO_CODEC_ROOT}/Samples/NvCodec")
include_directories("/usr/local/cuda-10.0/samples/common/inc")
include_directories("/home/fmthhadmin/Debayan/ZebraTrack/Samples/Utils/")
include_directories("/usr/include/x86_64-linux-gnu/libavformat/")
#################################################################




##################################################################
EXEC_PROGRAM("${PYLON_ROOT}/bin/pylon-config" ARGS --cflags OUTPUT_VARIABLE CPPFLAGS)
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CPPFLAGS}" CACHE STRING "CPPFLAGS")
SET(CPPFLAGS "${CPPFLAGS}" CACHE STRING "CPPFLAGS")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CPPFLAGS}")

EXEC_PROGRAM("${PYLON_ROOT}/bin/pylon-config" ARGS --libs-rpath OUTPUT_VARIABLE LDFLAGS)
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LDFLAGS}" CACHE STRING "LDFLAGS")
SET(LDFLAGS "${LDFLAGS}" CACHE STRING "LDFLAGS")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LDFLAGS}")

EXEC_PROGRAM("${PYLON_ROOT}/bin/pylon-config" ARGS --libs OUTPUT_VARIABLE LDLIBS)
SET(LDLIBS "${LDLIBS}" CACHE STRING "LDLIBS")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

#set(SOURCE_FILES AppEncCuda.cpp NvEncoder.cpp NvEncoderCuda.cpp)


file(GLOB SOURCE_FILES
        "*.h"
        "*.cpp"
        )

add_executable(VideoCapture ${SOURCE_FILES})
target_include_directories(VideoCapture PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#target_link_libraries(VideoCapture ${CUDART_LIBRARY})
link_directories(${VIDEO_CODEC_ROOT}/Lib/linux/stubs/x86_64)
target_link_libraries(VideoCapture ${OpenCV_LIBS} ${LDLIBS} ${CUDA_LIBRARIES} ${CUDA_LIB})