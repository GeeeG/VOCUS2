# CMakeLists.txt for VOCUS 2 saliency system (Frintrop et al. CVPR 2015)
#
# see http://www.cmake.org/ for a documentation about CMake
#
# compile your programm using the following commands:
# $ cmake CMakeLists.txt
# $ make

cmake_minimum_required(VERSION 2.8)

project(VOCUS2 CXX)

# choose if you want to compile with or without debugging flags,
# uncomment only one of the following two commands at a time
#set(CMAKE_BUILD_TYPE release)
set(CMAKE_BUILD_TYPE debug)

# add a compiler flag for the C++2011 standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -fopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS serialization REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

ADD_LIBRARY( vocus2l STATIC
       src/VOCUS2.cpp    
       )

add_executable(vocus2 src/main.cpp src/VOCUS2.cpp)
target_link_libraries(vocus2 ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(vocus2l ${OpenCV_LIBS} ${Boost_LIBRARIES})

