# gptneox

set(TARGET gptneox)

add_library(${TARGET}
    ../arch-util.h
    gptneox.h
    gptneox.cpp
    )

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PRIVATE
    ggml
    )

if (BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET} PRIVATE GPTNEOX_SHARED GPTNEOX_BUILD)
endif()

# common-gptneox

set(TARGET common-gptneox)

add_library(${TARGET} OBJECT
    common-gptneox.h
    common-gptneox.cpp
    )

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PRIVATE gptneox)


# main-gptneox

set(TARGET main-gptneox)
add_executable(${TARGET} main-gptneox.cpp)
target_link_libraries(${TARGET} PRIVATE common-gptneox gptneox ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()


# quantize-gptneox

set(TARGET quantize-gptneox)
add_executable(${TARGET} quantize-gptneox.cpp)
target_link_libraries(${TARGET} PRIVATE gptneox ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()


# update-gptneox

set(TARGET update-gptneox)
add_executable(${TARGET} update-gptneox.cpp)
target_link_libraries(${TARGET} PRIVATE gptneox ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()
