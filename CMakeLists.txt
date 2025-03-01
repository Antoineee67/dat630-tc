cmake_minimum_required(VERSION 3.20.0)
project(treecode LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_executable(treecode)

file(GLOB SOURCES
        "src/*.c"
        "src/*.cpp"
        "src/*.cu"
)

target_sources(treecode PRIVATE ${SOURCES})

target_include_directories(treecode PRIVATE
        "src"
        "/usr/local/cuda/targets/x86_64-linux/include"

)



target_link_libraries(treecode PRIVATE
        m
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
)



target_compile_definitions(treecode PRIVATE
        LINUX
)