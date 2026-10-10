cmake_minimum_required(VERSION 3.22)
project(cuda_sdf_viewer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(glm REQUIRED)

add_executable(cuda_sdf_viewer
    src/main.cpp
    src/player.cpp
    src/gl_renderer.cpp
    src/world_manager.cu
    src/cuda_renderer.cu
)

target_include_directories(cuda_sdf_viewer PRIVATE
    src
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(cuda_sdf_viewer PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
    glm::glm
)
