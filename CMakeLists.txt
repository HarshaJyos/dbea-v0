cmake_minimum_required(VERSION 3.14)
project(DBEA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Fetch nlohmann/json for save/load
include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(dbea_main ${SOURCES})

target_link_libraries(dbea_main PRIVATE nlohmann_json::nlohmann_json)