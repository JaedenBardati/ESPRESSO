cmake_minimum_required(VERSION 3.15...4.0)
project(ESPRESSO VERSION 0.0.1
                DESCRIPTION "Extremely SimPle RadiativE Simulation using Stochastic Output: A CPU and GPU supported Monte Carlo Radiative Transfer Code."
                LANGUAGES CXX)


find_package(Kokkos REQUIRED)
find_package(OpenMP QUIET)

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