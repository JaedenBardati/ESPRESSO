cmake_minimum_required(VERSION 3.15...4.0)
project(ESPRESSO VERSION 0.0.1
                DESCRIPTION "Extremely SimPle RadiativE Simulation using Stochastic Output: A GPU-based Monte Carlo Radiative Transfer Code"
                LANGUAGES CXX)


set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++-15")


find_package(OpenMP REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(espresso espresso.cpp)

target_compile_features(espresso PRIVATE cxx_std_20)
target_compile_options(espresso PRIVATE
  -O3
  -march=native
  -ffast-math
  $<$<BOOL:${OpenMP_CXX_FOUND}>:-fopenmp>
)

target_link_libraries(espresso PRIVATE
  Kokkos::kokkos
  OpenMP::OpenMP_CXX
)


# MAKE WITH cmake .. -DKokkos_ROOT=/usr/local/kokkos-openmp/lib/cmake/Kokkos -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15