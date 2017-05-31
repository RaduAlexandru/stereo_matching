cmake_minimum_required(VERSION 2.8.3)
project(stereo_test)


# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")




set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/deps ${CMAKE_SOURCE_DIR}/deps/imgui  )		# Header folder
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DAPC_CAPTURE_DEBUG=1")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(MPI REQUIRED)
find_package(GLFW REQUIRED)
#find_package(GLM REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)


include_directories(${PROJECT_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLFW_INCLUDE_DIR})
#include_directories(${GLM_INCLUDE_DIRS})
include_directories( ${CMAKE_SOURCE_DIR}/include/stereo_test/elas_incl)
#add_definitions(${GLM_DEFINITIONS})


file(GLOB IMGUI_SRC ${CMAKE_SOURCE_DIR}/deps/imgui/*.cpp)
file(GLOB SRC ${CMAKE_SOURCE_DIR}/src/*.c*)
file(GLOB ELAS_SRC ${CMAKE_SOURCE_DIR}/src/elas_src/*.c*)

message (${IMGUI_SRC})
message (${CMAKE_SOURCE_DIR}/include/stereo_test/elas_incl)


add_executable(stereo_test ${SRC} ${IMGUI_SRC} ${ELAS_SRC} deps/glad/glad.c  )
target_link_libraries(stereo_test  ${Boost_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_gl_LIBRARY} ${GLFW_LIBRARIES})
