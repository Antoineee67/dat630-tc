cmake_minimum_required(VERSION 3.22.1)
project(treecode)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)

add_executable(treecode)

file(GLOB SOURCES
        "src/*.c"
)

target_sources(treecode PRIVATE ${SOURCES})

target_include_directories(treecode PRIVATE
        "src"
        ${MPI_INCLUDE_PATH}
)

target_link_libraries(treecode PRIVATE
        m
        MPI::MPI_C
)

target_compile_definitions(treecode PRIVATE
        LINUX
)

#### MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)
