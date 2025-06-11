cmake_minimum_required(VERSION 3.15...4.0)
project(ESPRESSO VERSION 0.0.1
                DESCRIPTION "Extremely SimPle Radiative TransfEr using Stochastic Sampling Operations: A CPU and GPU parallelized Monte Carlo Radiative Transfer Code."
                LANGUAGES CXX)

# find or fetch Kokkos (required)
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

# set compile structure
add_executable(espresso espresso.cpp)

# set compile options
target_compile_features(espresso PRIVATE cxx_std_20)
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

# link libaries
target_link_libraries(espresso PRIVATE
  Kokkos::kokkos
)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: Compiling with OpenMP support.")
  target_link_libraries(espresso PRIVATE OpenMP::OpenMP_CXX)
endif()


# --- BUILD BASICS --- #
# For Threads, build with something like:
#   mkdir build_threads && cd build_threads
#   cmake .. -DKokkos_ENABLE_THREADS=ON
#   make
# For OpenMP, build with:
#   mkdir build_openmp && cd build_openmp
#   cmake .. -DKokkos_ENABLE_OPENMP=ON 
#   make
# FOR CUDA, build with:
#   mkdir build_cuda && build_cuda
#   cmake .. -DKokkos_ENABLE_CUDA=ON
#   make
# If absolutely wanted, you can run on CPU in serial with -Kokkos_ENABLE_SERIAL=ON instead.
# In theory, this should also work as above for HIP (AMD GPUs) and SYNCL (Intel GPUs), but has not been tested.
# When using CUDA, you need to install the CUDA Toolkit including the CUDA compiler nvcc.
# When using MSVC, use nmake instead of make.
#
# --- BUILD OPTIONS --- #
# To specify a preferred c++ compiler use the tag: 
#   -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15
# To specify an existing Kokkos install (to avoid fetching) add the cmake tag pointing to build location:
#   -DKokkos_ROOT=/usr/local/kokkos-openmp/lib/cmake/Kokkos
# It may help to specify CPU or GPU architecture with something like:
#   -DKokkos_ARCH_AMPERE86=ON -DCMAKE_CUDA_ARCHITECTURES=86
# When using MSVC, you may need to add the flag:
#   -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON
#
# See the Kokkos documentation for more tags: https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html
#
# --- OTHER COMMENTS --- #
# Serial and Threads should generally always work.
# For linux, just use gcc and everything should be nice and easy.
# For mac, clang by default does not support OpenMP. To fix this, you can install OpenMP (brew install libomp) and either 1) add the compiler 
#   flag "-Xclang" or 2) brew install gcc or a newer version of llvm and point to the working compiler as mentioned above.
# For windows, you can run in WSL2 Ubuntu fairly easily as though it was linux, but that is typically slow. Note CUDA does not currently support any 
# other WSL2 linux distro. For faster speeds, you can run on windows directly with MSVC 2022 and cmake in its custom cmd terminal, but use nmake. GPU 
# support is tricker on windows and you might need to play with the build options a bit, including specifying architecture as above.
#
