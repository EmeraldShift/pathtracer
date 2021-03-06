SET(pwd ${CMAKE_CURRENT_LIST_DIR})

UNSET(src)

# Uncomment the following lines to explicitly set files to compile from
# CMakeLists.sources
#IF (EXISTS CMakeLists.sources)
#      FILE(STRINGS CMakeLists.sources src)
#ENDIF()

#AUX_SOURCE_DIRECTORY(${pwd} src)
#AUX_SOURCE_DIRECTORY(${pwd}/fileio src)
#AUX_SOURCE_DIRECTORY(${pwd}/parser src)
#AUX_SOURCE_DIRECTORY(${pwd}/scene src)
#AUX_SOURCE_DIRECTORY(${pwd}/SceneObjects src)
#AUX_SOURCE_DIRECTORY(${pwd}/ui src)
#AUX_SOURCE_DIRECTORY(${pwd}/win32 src)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
file(GLOB_RECURSE src
		"*.h"
		"*.cu")

add_executable(ray ${src})
set_property(TARGET ray PROPERTY CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "ray added, files ${src}")

target_link_libraries(ray ${OPENGL_gl_LIBRARY})


#SET(FLTK_SKIP_FLUID TRUE)
#FIND_PACKAGE(FLTK REQUIRED)
#SET_PROPERTY(TARGET ray APPEND PROPERTY INCLUDE_DIRECTORIES ${FLTK_INCLUDE_DIR})
#target_link_libraries(ray ${FLTK_LIBRARIES})
FIND_PACKAGE(JPEG REQUIRED)
target_link_libraries(ray ${JPEG_LIBRARIES})
FIND_PACKAGE(PNG REQUIRED)
target_link_libraries(ray ${PNG_LIBRARIES})
FIND_PACKAGE(ZLIB REQUIRED)
target_link_libraries(ray ${ZLIB_LIBRARIES})
target_link_libraries(ray ${OPENGL_glu_LIBRARY})
# set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ray Threads::Threads)
