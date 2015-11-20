cmake_minimum_required(VERSION 3.3)
project(HappyBankING)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_PREFIX_PATH "/usr/local/opt/opencv3/share/OpenCV")

set(OpenCV_INCLUDE_DIRS "/usr/local/opt/opencv3/include")
set(OpenCV_LIBS "/usr/local/opt/opencv3/lib")

set(SOURCE_FILES smiledetect.cpp)

add_executable(HappyBankING ${SOURCE_FILES})

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${LIBCURL_CFLAGS})

target_link_libraries(HappyBankING ${OpenCV_LIBS} curl)
