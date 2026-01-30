
cmake_minimum_required(VERSION 3.16)
project(sensei_depth1_gtp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sensei_depth1_gtp sensei_depth1_gtp_main.cpp)

target_include_directories(sensei_depth1_gtp PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/..
)

target_sources(sensei_depth1_gtp PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/../board/board.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../board/get_moves.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../board/get_flip.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../board/bitpattern.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../utils/files.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../utils/misc.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../evaluatedepthone/pattern.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../evaluatedepthone/pattern_evaluator.cpp
)
