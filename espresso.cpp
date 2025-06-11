/*
 Simple Monte Carlo radiative transfer with C++20 for Ay190 at Caltech.
 
 Author: Jaeden Bardati
 Date: May 29, 2025
 Compiler call: g++-15 -fopenmp -O3 -march=native -ffast-math -std=c++20 espresso.cpp -o espresso
*/
#define DEBUG 1

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <concepts>
#include <type_traits>
using namespace std;

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

constexpr double PI = 3.1415926535897932;
constexpr double TWO_PI = 6.283185307179586;


// CONCEPTS
template<typename T>
concept FloatLike = std::is_floating_point_v<T>;

template<typename T>
concept IntLike = std::is_integral_v<T>;

template<typename T>
concept UnsignedIntLike = std::is_unsigned_v<T>;


// FILE OUTPUT
template <typename T>
int dump_vector_textfile(string filename, vector<T> x, vector<T> y, vector<T> z, string delim=" ") {
    /* Dumps a couple, equal length, vectors of floats into a textfile */
    ofstream outputFile;
    outputFile.open(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }
    
    for (int i=0; i < x.size(); ++i) {
        outputFile << x[i] << delim << y[i] << delim << z[i] << std::endl;
    }
    
    outputFile.close();
    return 0;
}


// DENSITY
template <typename Tf>
KOKKOS_INLINE_FUNCTION
Tf density(Tf x, Tf y, Tf z) {
    Tf eps = 0.001;
    Tf mag = 10.0;
    Tf norm = Kokkos::sqrt(x * x + y * y + z * z);
    return eps + Kokkos::max(mag * (1.0 - norm), 0.0);
}


// PHOTON PACKETS
template <typename Tf, typename Tu>
struct PhotonPackets {
    const unsigned int dim;
    Kokkos::View<Tu> num;
    Kokkos::View<Tf*> locations;
    Kokkos::View<Tf*> directions;
    Kokkos::View<Tf*> weights;
    Kokkos::View<bool*> isalive;

    PhotonPackets(unsigned int dim, Tu allocated_space): dim(dim), locations("locations", dim * allocated_space), directions("directions", dim * allocated_space), weights("weights", allocated_space), isalive("isalive", allocated_space) {
        num = Kokkos::View<Tu>("num");
        Kokkos::deep_copy(num, 0);
    }

    struct HostMirror {
        Kokkos::View<Tf*, Kokkos::HostSpace> locations;
        Kokkos::View<Tf*, Kokkos::HostSpace> directions;
        Kokkos::View<Tf*, Kokkos::HostSpace> weights;
        Kokkos::View<bool*, Kokkos::HostSpace> isalive;
        Kokkos::View<Tu, Kokkos::HostSpace> num;
    };
    
    HostMirror mirror_to_host_space() {
        HostMirror host;
        host.locations = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), locations);
        host.directions = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), directions);
        host.weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), weights);
        host.isalive = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), isalive);
        host.num = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), num);
        return host;
    }
};


// MAIN
int main(int argc, char** argv) {
    /* ** SIMULATION OPTIONS ** */
    const unsigned int DIM = 3; // dimension of the simulation
    const unsigned int RANDOM_SEED = 0;
    
    using UINT = unsigned int; // define precision of simulation
    using FLOAT = float; // define precision of simulation
    
    const FLOAT DOMAIN_XMIN = -3.0;
    const FLOAT DOMAIN_XMAX = 3.0;
    const FLOAT DOMAIN_YMIN = -3.0;
    const FLOAT DOMAIN_YMAX = 3.0;
    const FLOAT DOMAIN_ZMIN = -3.0;
    const FLOAT DOMAIN_ZMAX = 3.0;
    
    const FLOAT int_nbins = 4.0;
    const FLOAT max_time = 10.0;
    
    FLOAT kappa_a = 0.5;  // absorption opacity
    FLOAT kappa_s = 0.5;  // scattering opacity
    FLOAT speed_of_light = 1.0;
    FLOAT frac_of_mfpovc = 0.1; // moves 10% the way through one max optical depth
    FLOAT luminosity_normalization = 1.0;
    
    FLOAT kappa = kappa_a + kappa_s;
    FLOAT dt = frac_of_mfpovc*1.0/(kappa*(10.01)*speed_of_light); // timestep
    
    UINT NPACKETS = 1e6*dt/0.05;
    UINT allocated_space = NPACKETS*max_time/dt; // very conservative allocation: will definitely not be more than this (assumes no interactions)
    
    const FLOAT XINIT = 2.0;
    const FLOAT YINIT = 0.0;
    const FLOAT ZINIT = 0.0;
    
    const bool photon_dump = false;
    const bool zslice_dump = true;
    
    /* ****** SIMULATION ****** */
    Kokkos::initialize(argc, argv);
    {
        // precompute quantities
        FLOAT inv_int_nbins = 1.0/int_nbins;
        FLOAT P_abs = kappa_a/kappa;
        FLOAT new_packet_energy = luminosity_normalization*dt/NPACKETS;
        
        // make random pool
        Kokkos::Random_XorShift64_Pool<> rand_pool(RANDOM_SEED);
        
        // reserve space for photon packets
        PhotonPackets<FLOAT, UINT> packets(DIM, allocated_space);
    
        
        cout << "Running simulation ..." << endl;
        auto start = chrono::high_resolution_clock::now();
        
        // start up the host-device sync npacket variables
        UINT host_packets_num = 0;
        Kokkos::deep_copy(packets.num, host_packets_num);
        
        FLOAT time = 0.0;
        while (time < max_time) {
            #if DEBUG >= 1
            cout << "time = " << fixed << setprecision(6) << time << "; nphotons=" << host_packets_num << endl;
            #endif
            
            // spawn source packets randomly
            #if DEBUG >= 2
            cout << " spawning..." << endl;
            #endif
            
            Kokkos::parallel_for("spawn", NPACKETS, KOKKOS_LAMBDA(UINT i) {
                auto rand_gen = rand_pool.get_state();
                FLOAT phi_i = rand_gen.frand(0.0, TWO_PI);
                FLOAT costheta_i = rand_gen.frand(-1.0, 1.0);
                rand_pool.free_state(rand_gen);
                
                FLOAT sintheta_i = Kokkos::sqrt(1.0-costheta_i*costheta_i);
                
                FLOAT dir_x = sintheta_i*Kokkos::cos(phi_i);
                FLOAT dir_y = sintheta_i*Kokkos::sin(phi_i);
                FLOAT dir_z = costheta_i;
                FLOAT weight = new_packet_energy;
                
                UINT idx = packets.num() + i;  // start indexing after current packets
                packets.locations(DIM*idx+0) = XINIT;
                packets.locations(DIM*idx+1) = YINIT;
                packets.locations(DIM*idx+2) = ZINIT;
                packets.directions(DIM*idx+0) = dir_x;
                packets.directions(DIM*idx+1) = dir_y;
                packets.directions(DIM*idx+2) = dir_z;
                packets.weights(idx) = weight;
                packets.isalive(idx) = true;
                
                rand_pool.free_state(rand_gen);
            });
            host_packets_num += NPACKETS;
            Kokkos::deep_copy(packets.num, host_packets_num);
            
            // main integration
            #if DEBUG >= 2
            cout << " integrating..." << endl;
            #endif
            FLOAT cdt = speed_of_light*dt;
            Kokkos::parallel_for("integrate", host_packets_num, KOKKOS_LAMBDA(UINT i) {
                // move all packets
                FLOAT old_x = packets.locations(3*i);
                FLOAT old_y = packets.locations(3*i+1);
                FLOAT old_z = packets.locations(3*i+2);
                
                FLOAT dx = cdt*packets.directions(3*i);
                FLOAT dy = cdt*packets.directions(3*i+1);
                FLOAT dz = cdt*packets.directions(3*i+2);
                
                packets.locations(3*i) = old_x + dx;
                packets.locations(3*i+1) = old_y + dy;
                packets.locations(3*i+2) = old_z + dz;
                
                // delete packet if outside of domain
                if (packets.locations(3*i) > DOMAIN_XMAX ||
                    DOMAIN_XMIN > packets.locations(3*i) ||
                    packets.locations(3*i+1) > DOMAIN_YMAX ||
                    DOMAIN_YMIN > packets.locations(3*i+1) ||
                    packets.locations(3*i+2) > DOMAIN_ZMAX ||
                    DOMAIN_ZMIN > packets.locations(3*i+2)) {
                    packets.isalive(i) = false;
                    return;
                }
                
                // integrate with trapezoidal rule to get optical depth travelled
                FLOAT dtau = 0.5 * (density(old_x, old_y, old_z) + density(packets.locations(3*i), packets.locations(3*i+1), packets.locations(3*i+2)));
                for (UINT j=1; j < int_nbins-1; ++j) {
                    dtau  = dtau + density(old_x+j*dx*inv_int_nbins, old_y+j*dy*inv_int_nbins, old_z+j*dz*inv_int_nbins);
                }
                dtau *= cdt * kappa;  // kappa can go outside of integral atm since it is not dependent on location
                
                // handle interaction if relevant
                auto rand_gen = rand_pool.get_state();
                FLOAT r1 = rand_gen.frand(0.0, 1.0);
                FLOAT r2 = rand_gen.frand(0.0, 1.0);
                rand_pool.free_state(rand_gen);
                
                if (r1 < 1 - Kokkos::exp(-dtau)) {
                    // interaction occurs
                    if (r2 < P_abs) {
                        // delete if absorbed
                        packets.isalive(i) = false;
                        return;
                    } else {
                        // otherwise, isotropic scattering interaction
                        FLOAT phi_i = rand_gen.frand(0.0, TWO_PI);
                        FLOAT costheta_i = rand_gen.frand(-1.0, 1.0);
                        FLOAT sintheta_i = Kokkos::sqrt(1.0-costheta_i*costheta_i);
                        
                        packets.directions(DIM*i+0) = sintheta_i*Kokkos::cos(phi_i);
                        packets.directions(DIM*i+1) = sintheta_i*Kokkos::sin(phi_i);
                        packets.directions(DIM*i+2) = costheta_i;
                    }
                }
            });
            
            
            // "delete" packets (really just ignore those that are irrelevant)
            #if DEBUG >= 2
            cout << " deleting..." << endl;
            #endif
            
            // use prefix sum method for array compaction
            Kokkos::parallel_scan("scan", host_packets_num, KOKKOS_LAMBDA(const UINT i, UINT &update, const bool final) {
                bool val = packets.isalive(i);
                
                if (final && val) {
                    UINT j = update;
                    for (UINT d = 0; d < DIM; ++d) {
                        packets.locations(DIM*j+d) = packets.locations(DIM*i+d);
                        packets.directions(DIM*j+d) = packets.directions(DIM*i+d);
                    }
                    packets.weights(j) = packets.weights(i);
                    packets.isalive(j) = 1;
                }

                update += val ? 1 : 0;

                // store final count in new_size after last iteration
                if (final && i == host_packets_num - 1) {
                    packets.num() = update;
                }
            });
            host_packets_num = packets.num();  // update number of packets host side
            
            time += dt;
        }
        
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Simulation took " << duration.count()/1000.0 << " seconds." << endl;
    
        /* ***************************** */
        
        // copy views to host mirrors
        auto host_packets = packets.mirror_to_host_space();
        
        // dump photon information
        if (photon_dump) {
            string filename = "photon.txt";
            ofstream outputFile;
            outputFile.open(filename);
            if (!outputFile.is_open()) {
                std::cerr << "Error opening file: " << filename << std::endl;
                return 1;
            }
            
            for (UINT i=0; i < host_packets.num(); ++i) {
                outputFile << host_packets.locations[3*i] << " " << host_packets.locations[3*i+1] << " " << host_packets.locations[3*i+2] << " " << host_packets.directions[3*i] << " " << host_packets.directions[3*i+1] << " " << host_packets.directions[3*i+2] << " " << host_packets.weights(i) << std::endl;
            }
            outputFile.close();
        }
        
        // dump energy density z slice
        if (zslice_dump){
            const FLOAT L = 0.1;
            const UINT xbins = (DOMAIN_XMAX - DOMAIN_XMIN)/L;
            const UINT ybins = (DOMAIN_YMAX - DOMAIN_YMIN)/L;
            const UINT zbins = 1;
            vector<FLOAT> energy_density(xbins*ybins*zbins);
            
            #pragma omp parallel for
            for (UINT i=0; i < host_packets.num(); ++i) {
                if (host_packets.locations[3*i+2] < -0.5*L || 0.5*L < host_packets.locations[3*i+2]) {
                    continue;
                }
                
                UINT x_idx = (host_packets.locations[3*i] - DOMAIN_XMIN)/L;
                UINT y_idx = (host_packets.locations[3*i+1] - DOMAIN_YMIN)/L;
                energy_density[x_idx + y_idx*ybins] += host_packets.weights[i]/(L*L*L);
            }
            
            vector<FLOAT> xs(xbins*ybins*zbins);
            vector<FLOAT> ys(xbins*ybins*zbins);
            
            #pragma omp parallel for
            for (UINT x_idx=0; x_idx < xbins; ++x_idx) {
                for (UINT y_idx=0; y_idx < ybins; ++y_idx) {
                    xs[x_idx + y_idx*ybins] = (x_idx+0.5)*(DOMAIN_XMAX - DOMAIN_XMIN)/xbins + DOMAIN_XMIN;
                    ys[x_idx + y_idx*ybins] = (y_idx+0.5)*(DOMAIN_YMAX - DOMAIN_YMIN)/ybins + DOMAIN_YMIN;
                }
            }
            dump_vector_textfile("energy_density_zslice.txt", xs, ys, energy_density);
        }
        
        Kokkos::printf("Terminating program.\n");
    }
    Kokkos::finalize();
    return 0;
}


