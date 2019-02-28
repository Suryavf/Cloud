# CMake entry point
cmake_minimum_required (VERSION 3.0)
project (cloud LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Options
# -------

# Force the use of GLEW - if found - to handle OpenGL extensions,
# otherwise a python script will generate functions loaders.
set(USE_GLEW ON)


# Dependencies
# ------------

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(GLEW REQUIRED)
if(${GLEW_FOUND})
    add_definitions(-DUSE_GLEW)
endif()

find_package(CUDA REQUIRED)

# Main target build
# -----------------

cuda_add_executable( cloud
                     main.cpp 
                     sphere.cpp
                     sphere.h
                     common/shader.cpp
                     common/shader.hpp
                     common/controls.cpp
                     common/controls.hpp
                     common/texture.cpp
                     common/texture.hpp
                     common/objloader.cpp
                     common/objloader.hpp
                     #vertexshader.glsl
                     #fragmentshader.glsl
                     )


if(CMAKE_COMPILER_IS_GNUCXX)
  set(CXX_FLAGS "-g -O3 -Wall -std=c++11 -stdlib=libc++ -lGLEW ")
elseif(MSVC)
  set(CXX_FLAGS "")
elseif(${CMAKE_CXX_COMPILER})
  message(WARNING "untested compiler")
else()
  message(FATAL_ERROR "no compiler was found")
endif()


set_target_properties(cloud PROPERTIES
                      LINKER_LANGUAGE CXX)


include_directories("./common"
                    ${GLFW_INCLUDE_DIRS}
                    ${OPENGL_INCLUDE_DIRS}
                    ${GLEW_INCLUDE_DIRS}
                    ${GLUT_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS})


target_link_libraries(cloud 
                     ${GLFW_STATIC_LIBRARIES}
                     ${OPENGL_LIBRARIES}
                     ${GLEW_LIBRARIES}
                     ${GLUT_LIBRARY})                    
