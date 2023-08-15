# dependencies

find_package(Threads REQUIRED)

# gptneox

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(gptneox)
endif()
