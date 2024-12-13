cmake_minimum_required(VERSION 3.16)
project(binfuse)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if (MSVC)
  list(APPEND PROJECT_COMPILE_OPTIONS /W3)
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " /fsanitize=address")
else() # *nix
  list(APPEND PROJECT_COMPILE_OPTIONS -Wall -Wextra -Wpedantic -Wshadow -Wextra-semi
    -Wmissing-noreturn -Wconversion -Wsign-conversion -Wno-ignored-attributes)

  if (NOT MINGW) # sanitizers are not working under mingw
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address,undefined,leak")
  endif()
endif()

add_subdirectory(ext/xor_singleheader)

add_subdirectory(ext/mio)

add_library(binfuse INTERFACE)
target_include_directories(binfuse INTERFACE include)
target_compile_options(binfuse INTERFACE ${PROJECT_COMPILE_OPTIONS})
target_compile_features(binfuse INTERFACE cxx_std_20)
target_link_libraries(binfuse INTERFACE xor_singleheader mio)

option(BINFUSE_BENCH "Build BINFUSE benchmark" OFF)
if(BINFUSE_BENCH)
  add_executable(binfuse_bench_large bench/large.cpp)
  target_compile_options(binfuse_bench_large PRIVATE ${PROJECT_COMPILE_OPTIONS})
  target_compile_features(binfuse_bench_large PRIVATE cxx_std_20)
  target_link_libraries(binfuse_bench_large PRIVATE binfuse)
endif()

# testing

option(BINFUSE_TEST "Enable BINFUSE tests" ON)
if(BINFUSE_TEST)
  get_property(all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
  enable_testing()
  add_subdirectory(test)
else(BINFUSE_TEST)
  message(STATUS "BINFUSE Tests are disabled. Set BINFUSE_TEST to ON to run tests.")
endif(BINFUSE_TEST)

# make compile commands available for IDEs etc
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_BINARY_DIR}/compile_commands.json
  ${CMAKE_SOURCE_DIR}/compile_commands.json
)
