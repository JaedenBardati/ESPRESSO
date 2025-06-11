# ESPRESSO
"Extremely SimPle Radiative TransfEr using Stochastic Sampling Operations". A CPU and GPU parallelized Monte Carlo Radiative Transfer Code. 

Uses Kokkos to be versatile across platforms. Tested in the following environments:
- Mac M3: Clang with Threads multi-threading
- Mac M3: G++15 with OpenMP multi-threading
- WSL2 Ubuntu x64: G++13 with OpenMP multi-threading
- WSL2 Ubuntu x64 + Ampere GPU: G++13 with CUDA GPU-acceleration
- Windows x64: MSVC with OpenMP multi-threading
- Windows x64 + Ampere GPU: MSVC with CUDA GPU-acceleration

To run, build with cmake - see `CMakeLists.txt`.
