cmake_minimum_required(VERSION 3.10)
project(MLP)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置Eigen库路径
set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Eigen)

# 包含头文件目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN_INCLUDE_DIR}
)

# 源文件
set(SOURCES
    src/mlp.cpp
    src/activation.cpp
    src/loss.cpp
)

# 创建静态库
add_library(mlp_lib STATIC ${SOURCES})

# 创建测试可执行文件（Loss 测试）
add_executable(mlp_test_loss tests/test_loss.cpp)
target_link_libraries(mlp_test_loss mlp_lib)

# 编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mlp_lib PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(mlp_test_loss PRIVATE -Wall -Wextra -Wpedantic)
endif()


# 打印配置信息
message(STATUS "Eigen include directory: ${EIGEN_INCLUDE_DIR}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")