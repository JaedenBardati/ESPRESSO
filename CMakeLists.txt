cmake_minimum_required(VERSION 3.15...4.0)
project(ESPRESSO VERSION 0.0.1
                DESCRIPTION "Extremely SimPle RadiativE Simulation using Stochastic Output: A CPU and GPU supported Monte Carlo Radiative Transfer Code."
                LANGUAGES CXX)

# find or fetch Kokkos
find_package(Kokkos QUIET)
if (NOT Kokkos_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/archive/refs/tags/4.5.01.zip
  )
  FetchContent_MakeAvailable(Kokkos)
endif()

# find OpenMP if available
find_package(OpenMP)

# compile structure
add_executable(espresso espresso.cpp)
target_compile_features(espresso PRIVATE cxx_std_20)

# compile options
if(MSVC)
  target_compile_options(espresso PRIVATE /O2 /openmp)
else()
  target_compile_options(espresso PRIVATE
    -O3
    -march=native
    -ffast-math
    $<$<BOOL:${OpenMP_CXX_FOUND}>:-fopenmp>
  )
endif()

# link everything
target_link_libraries(espresso PRIVATE
  Kokkos::kokkos
  OpenMP::OpenMP_CXX
)


# FOR OPENMP, build with something like
#   cmake .. -DKokkos_ENABLE_OPENMP=ON -DKokkos_ROOT=/usr/local/kokkos-openmp/lib/cmake/Kokkos -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 && make
# ON MAC (and no openmp), you can also build with something like
#   cmake .. -DKokkos_ENABLE_THREADS=ON && make
# FOR GPU in Linux (gcc), build with something like
#   cmake .. -DKokkos_ENABLE_CUDA=ON && make
# FOR GPU on Windows (need msvc), build with something like
#   cmake .. -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_AMPERE86=ON -DCMAKE_CUDA_ARCHITECTURES=86 && nmake
