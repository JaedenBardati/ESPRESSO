cmake_minimum_required(VERSION 3.15...4.0)
project(ESPRESSO VERSION 0.0.1
                DESCRIPTION "Extremely SimPle RadiativE Simulation using Stochastic Output: A CPU and GPU supported Monte Carlo Radiative Transfer Code."
                LANGUAGES CXX)

# find or fetch Kokkos
find_package(Kokkos QUIET)
if (NOT Kokkos_FOUND)
  message(STATUS "Kokkos not found: Fetching library.")
  include(FetchContent)
  FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/archive/refs/tags/4.5.01.zip
  )
  FetchContent_MakeAvailable(Kokkos)
else()
  message(STATUS "Kokkos found: Using existing library.")
endif()

# find OpenMP if available
find_package(OpenMP QUIET)

# compile structure
add_executable(espresso espresso.cpp)
target_compile_features(espresso PRIVATE cxx_std_20)

# compile options
if(MSVC)
  message(STATUS "Compiling for Microsoft Visual Studio support.")
  target_compile_options(espresso PRIVATE /O2)
  if(OpenMP_CXX_FOUND)
    target_compile_options(espresso PRIVATE /openmp)
  endif()
else()
  message(STATUS "Compiling for GCC/Clang support.")
  target_compile_options(espresso PRIVATE
    -O3
    -march=native
    -ffast-math
  )
  if(OpenMP_CXX_FOUND)
    target_compile_options(espresso PRIVATE -fopenmp)
  endif()
endif()

# link everything
target_link_libraries(espresso PRIVATE
  Kokkos::kokkos
)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: Compiling with OpenMP support.")
  target_link_libraries(espresso PRIVATE OpenMP::OpenMP_CXX)
endif()


# FOR OPENMP, build with something like
#   cmake .. -DKokkos_ENABLE_OPENMP=ON -DKokkos_ROOT=/usr/local/kokkos-openmp/lib/cmake/Kokkos -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 && make
# WITHOUT OPENMP (e.g. default mac clang), you can also build (a generally slower version) with something like
#   cmake .. -DKokkos_ENABLE_THREADS=ON && make
# FOR GPU in Linux (gcc), build with something like
#   cmake .. -DKokkos_ENABLE_CUDA=ON && make
# FOR GPU on Windows (need msvc), build with something like
#   cmake .. -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_AMPERE86=ON -DCMAKE_CUDA_ARCHITECTURES=86 && nmake
