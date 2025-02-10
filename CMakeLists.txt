cmake_minimum_required(VERSION 3.20.0)
project(treecode)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(treecode)

file(GLOB SOURCES
    "src/*.c"
)

target_sources(treecode PRIVATE ${SOURCES})

target_include_directories(treecode PRIVATE
    "src"
)

target_link_libraries(treecode PRIVATE
    m
    OpenMP::OpenMP_C
)

target_compile_definitions(treecode PRIVATE
    LINUX
)