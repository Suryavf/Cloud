# CMake entry point
cmake_minimum_required (VERSION 3.0)
project (cloud)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

add_executable( cloud
                main.cpp 

                common/shader.cpp
                common/shader.hpp
                common/controls.cpp
                common/controls.hpp
                common/texture.cpp
                common/texture.hpp
                common/objloader.cpp
                common/objloader.hpp

                vertexshader.glsl
                fragmentshader.glsl
)

target_link_libraries(cloud ${GLFW_LIBRARIES})
target_link_libraries(cloud ${GLFW_STATIC_LIBRARIES})

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
target_link_libraries(cloud ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )