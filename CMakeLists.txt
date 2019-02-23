# CMake entry point
cmake_minimum_required (VERSION 3.0)
project (cloud)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

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

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
target_link_libraries(cloud ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )