//Copyright (C) 2015, NRC "Kurchatov institute", http://www.nrcki.ru/e/engl.html, Moscow, Russia
//Author: Vladislav Neverov, vs-never@hotmail.com, neverov_vs@nrcki.ru
//
//This file is part of XaNSoNS.
//
//XaNSoNS is free software: you can redistribute it and / or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//XaNSoNS is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program. If not, see <http://www.gnu.org/licenses/>.

//Contains host and device code for the CUDA version of XaNSoNS

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "typedefs.h"
#ifdef UseCUDA

#include "config.h"
#include "block.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//Calculates rotational matrix (from CalcFunctions.cpp)
void calcRotMatrix(vect3d <double> * const RM0, vect3d <double> * const RM1, vect3d <double> * const RM2, const vect3d <double> euler, const unsigned int convention);

//some float4 and float 3 functions (float4 used as float3)
inline __device__ __host__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ __host__ float dot(float3 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ __host__ float dot(float4 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __host__ __device__ float3 operator+(float3 a, float3 b){ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ float3 operator-(float3 a, float3 b){ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __host__ __device__ float3 operator*(float3 a, float b){ return make_float3(a.x * b, a.y * b, a.z * b); }
inline __device__ float length(float3 v){ return sqrtf(dot(v, v)); }

//the following functions are used to calculate 2D diffraction patterns
//all the 2D arrays are flattened

/**
	Resets the 2D scattering intensity array

	@param *I   Intensity array
	@param Nq   Size of the scattering vector magnitude mesh (number of rows in the 2D intensity array)
	@param Nfi  Size of the scattering vector polar angle mesh (number of columns in the 2D intensity array)
*/
__global__ void zeroInt2DKernel(float * const I, const unsigned int Nq, const unsigned int Nfi);

/**
	Resets the 2D scattering amplitude arrays (real and imaginary parts)

	@param *Ar  Real part of the 2D scattering amplitude array
	@param *Ai  Imaginary part of the 2D scattering amplitude array
	@param Nq   Size of the scattering vector magnitude mesh (number of rows in the 2D intensity array)
	@param Nfi  Size of the scattering vector polar angle mesh (number of columns in the 2D intensity array)
*/
__global__ void zeroAmp2DKernel(float * const Ar, float * const Ai, const unsigned int Nq, const unsigned int Nfi);

/**
	Computes the 2D scattering intensity using the scattering amplitude
	
	@param *I   Intensity array
	@param *Ar  Real part of the 2D scattering amplitude array
	@param *Ai  Imaginary part of the 2D scattering amplitude array
	@param Nq   Size of the scattering vector magnitude mesh (number of rows in the 2D amplitude array)
	@param Nfi  Size of the scattering vector polar angle mesh (number of columns in the 2D amplitude array)
*/
__global__ void Sum2DKernel(float * const I, const float * const Ar, const float * const Ai, const unsigned int Nq, const unsigned int Nfi);

/**
	Multiplies the 2D scattering intensity by a normalizing factor

	@param *I   Intensity array
	@param Nq   Size of the scattering vector magnitude mesh (number of rows in the 2D intensity array)
	@param Nfi  Size of the scattering vector polar angle mesh (number of columns in the 2D intensity array)
	@param norm Normalizing factor
*/
__global__ void Norm2DKernel(float * const I, const unsigned int Nq, const unsigned int Nfi, const float norm);

/**
	Computes the polarization factor and multiplies the 2D scattering intensity by this factor

	@param *I     Intensity array
	@param Nq     Size of the scattering vector magnitude mesh (number of rows in the 2D intensity array)
	@param Nfi    Size of the scattering vector polar angle mesh (number of columns in the 2D intensity array)
	@param *q     Scattering vector magnitude array
	@param lambda Wavelength of the source
*/
template <unsigned int BlockSize2D> __global__ void PolarFactor2DKernel(float * const I, const unsigned int Nq, const unsigned int Nfi, const float * const q, const float lambda);

/**
	Computes the real and imaginary parts of the 2D x-ray (source == xray) or neutron (source == neutron) scattering amplitude in the polar coordinates(q, q_fi) of the reciprocal space

	@param source xray or neutron
	@param *Ar    Real part of the 2D scattering amplitude array
	@param *Ai	  Imaginary part of the 2D scattering amplitude array
	@param *q     Scattering vector magnitude array
	@param Nq     Size of the scattering vector magnitude mesh (number of rows in the 2D amplitude array)
	@param Nfi    Size of the scattering vector polar angle mesh (number of columns in the 2D amplitude array)
	@param CS[]   Transposed rotational matrix. Defines the orientation of the nanoparticle in the 3D space.
	@param lambda Wavelength of the source
	@param *ra    Atomic coordinate array
	@param Nfin   Number of atoms to compute for in this kernel call (less or equal to the total number of atoms, cause the kernel is called iteratively in the loop)
	@param *FF    X-ray atomic form-factor array (for one kernel call the computations are done only for the atoms of the same chemical element) (NULL if source is neutron)
	@param SL     Neutron scattering length of the current chemical element (for one kernel call the computations are done only for the atoms of the same chemical element) (0 if source is xray)
*/
template <unsigned int BlockSize2D, unsigned int SizeR> __global__ void calcInt2DKernel(const unsigned int source, float * const Ar, float * const Ai, const float * const q, const unsigned int Nq, const unsigned int Nfi, const float3 CS[], const float lambda, const float4 * const ra, const unsigned int Nfin, const float * const FF, const float SL);

/**
	Organazies the computations of the 2D scattering intensity in the polar coordinates (q,q_fi) of the reciprocal space with CUDA

	@param DeviceNUM  CUDA device number
	@param ***I2D     2D scattering intensity array (host). The memory is allocated inside the function.
	@param **I        1D (averaged over the polar angle) scattering intensity array (host). The memory is allocated inside the function.
	@param *cfg       Parameters of simulation
	@param *NatomEl	  Array containing the total number of atoms of each chemical element (host)
	@param *ra        Atomic coordinate array (device)
	@param *dFF       X-ray atomic form-factor array for all chemical elements (device)
	@param SL         Array of neutron scattering lengths for all chemical elements
	@param *dq        Scattering vector magnitude array (device)
*/
void calcInt2DCuda(const int DeviceNUM, double *** const I2D, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const float4 * const ra, const float * const dFF, const vector<double> SL, const float * const dq);

//the following functions are used to calculate the histogram of interatomic distances

/**
	Resets the histogram array (unsigned long long int)

	@param *rij_hist  Histogram of interatomic distances
	@param N          Size of the array
*/
__global__ void zeroHistKernel(unsigned long long int * const rij_hist, const unsigned int N);

/**
	Computes the total histogram (first Nhist elements) using the partial histograms (for the devices with the CUDA compute capability < 2.0)

	@param *rij_hist   Partial histograms of interatomic distances
	@param Nhistcopies Number of the partial histograms to sum
	@param Nfin        Number of bins to compute for one kernel call
*/
__global__ void sumHistKernel(unsigned long long int * const rij_hist, const unsigned int Nhistcopies, const unsigned int Nfin, const unsigned int Nhist);

/**
	Computes the histogram of interatomic distances

	@param *ri         Pointer to the coordinate of the 1st i-th atom in ra array
	@param *rj         Pointer to the coordinate of the 1st j-th atom in ra array
	@param iMax        Total number of i-th atoms for this kernel call
	@param jMax        Total number of j-th atoms for this kernel call
	@param *rij_hist   Histogram of interatomic distances
	@param bin         Width of the histogram bin
	@param Nhistcopies Number of partial histograms to compute (!=1 for the devices with the CUDA compute capability < 2.0 to reduce the number of atomicAdd() calls)
	@param Nhist       Size of the partial histogram of interatomic distances
	@param Rcut2       Square of the cut-off radius (if cfg->cutoff is true)
	@param add         Addendum to the histogram bin. Equals to 2 if cutoff is false, otherwise equals to 2 if j-th atoms belong to the inner sphere or to 1 if to the outer
	@param diag        True if the j-th atoms and the i-th atoms are the same (diagonal) for this kernel call
*/
template <unsigned int BlockSize2D, bool cutoff> __global__ void calcHistKernel(const float4 * const __restrict__ ri, const float4 *  const __restrict__ rj, const unsigned int iMax, const unsigned int jMax, unsigned long long int *const rij_hist, const float bin, const unsigned int Nhistcopies, const unsigned int Nhist, const float Rcut2, const unsigned long long int add, const bool diag);

/**
	Organazies the computations of the histogram of interatomic distances with CUDA 

	@param DeviceNUM   CUDA device number
	@param **rij_hist  Histogram of interatomic distances (device). The memory is allocated inside the function.
	@param *ra         Atomic coordinate array (device)
	@param *NatomEl    Array containing the total number of atoms of each chemical element (host)
	@param *NatomEl_outer  Array containing the number of atoms of each chemical element including the atoms in the outer sphere (only if cfg.cutoff is True) (host)
	@param *cfg      Parameters of simulation	
*/
void calcHistCuda(const int DeviceNUM, unsigned long long int ** const rij_hist, const float4 * const ra, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const config * const cfg);

//the following functions are used to calculate the powder diffraction pattern using the histogram of interatomic distances

/**
	Resets 1D float array of size N

	@param *A  Array
	@param N   Size of the array	
*/
__global__ void zero1DFloatArrayKernel(float * const A, const unsigned int N);

/**
	Computes the total scattering intensity (first Nq elements) from the partials sums computed by different thread blocks

	@param *I    Scattering intensity array
	@param Nq    Resolution of the total scattering intensity (powder diffraction pattern) 
	@param Nsum  Number of parts to sum (equalt to the total number of thread blocks in the grid)
*/
__global__ void sumIKernel(float * const I, const unsigned int Nq, const unsigned int Nsum);

/**
	Adds the diagonal elements (j==i) of the Debye double sum to the x-ray scattering intensity 

	@param *I    Scattering intensity array
	@param *FF   X-ray atomic form-factor array (for one kernel call the computations are done only for the atoms of the same chemical element)
	@param Nq    Resolution of the total scattering intensity (powder diffraction pattern)
	@param N     Total number of atoms of the chemical element for whcich the computations are done 
*/
__global__ void addIKernelXray(float * const I, const float * const FF, const unsigned int Nq, const unsigned int N);

/**
	Adds the diagonal elements (j==i) of the Debye double sum to the neutron scattering intensity 

	@param *I    Scattering intensity array
	@param Nq    Resolution of the total scattering intensity (powder diffraction pattern)
	@param Add   The value to add to the intensity (the result of multiplying the square of the scattering length 
                 to the total number of atoms of the chemical element for whcich the computations are done) 
*/
__global__ void addIKernelNeutron(float * const I, const unsigned int Nq, const float Add);

/**
	Computes the polarization factor and multiplies scattering intensity by this factor

	@param *I     Scattering intensity array
	@param Nq     Size of the scattering intensity array
	@param *q     Scattering vector magnitude array
	@param lambda Wavelength of the source
*/
__global__ void PolarFactor1DKernel(float * const I, const unsigned int Nq, const float * const q, const float lambda);

/**
	Computes the x-ray (source == xray) or neutron (source == neutron) scattering intensity (powder diffraction pattern) using the histogram of interatomic distances

	@param source          xray or neutron
	@param *I              Scattering intensity array
	@param *FFi            X-ray atomic form factor for the i-th atoms (all the i-th atoms are of the same chemical element for one kernel call) (NULL if source is neutron)
	@param *FFj            X-ray atomic form factor for the j-th atoms (all the j-th atoms are of the same chemical element for one kernel call)(NULL if source is neutron)
	@param SLij            Product of the scattering lenghts of i-th j-th atoms (0 if source is xray)
	@param *q              Scattering vector magnitude array
	@param Nq              Size of the scattering intensity array
	@param **rij_hist      Histogram of interatomic distances (device). The memory is allocated inside the function
	@param iBinSt          Starting index of the histogram bin for this kernel call (the kernel is called iteratively in a loop)
	@param Nhist           Size of the partial histogram of interatomic distances
	@param MaxBinsPerBlock Maximum number of histogram bins used by a single thread block
	@param bin             Width of the histogram bin
	@param Rcut            Cutoff radius in A (if cutoff is true)
	@param damping         cfg->damping
*/
template <unsigned int Size> __global__ void calcIntHistKernel(const unsigned int source, float * const I, const float * const FFi, const float * const FFj, const float SLij, const float *const q, const unsigned int Nq, const unsigned long long int *const rij_hist, const unsigned int iBinSt, const unsigned int Nhist, const unsigned int MaxBinsPerBlock, const float bin, const float Rcut, const bool damping);

/**
    Adds the average density correction to the xray scattering intensity when the cut-off is enabled (cfg.cutoff == true)

	@param *I        Scattering intensity array
	@param *q        Scattering vector magnitude array
	@param Nq        Size of the scattering intensity array
	@param *dFF      X-ray atomic form-factor array for all chemical elements
	@param *NatomEl  Array containing the total number of atoms of each chemical element
	@param Nel       Total number of different chemical elements in the nanoparticle
	@param Ntot      Total number of atoms in the nanoparticle
	@param Rcut      Cutoff radius in A (if cutoff is true)
	@param dens      Average atomic density of the nanoparticle
	@param damping   cfg->damping
*/
__global__ void AddCutoffKernelXray(float * const I, const float * const q, const float * const FF, const unsigned int * const NatomEl, const unsigned int Nel, const unsigned int Ntot, const unsigned int Nq, const float Rcut, const float dens, const bool damping);

/**
    Adds the average density correction to the neutron scattering intensity when the cut-off is enabled (cfg.cutoff == true)

	@param *I        Scattering intensity array
	@param *q        Scattering vector magnitude array
	@param Nq        Size of the scattering intensity array
	@param SLaver    Average neutron scattering length of the nanopaticle
	@param Rcut      Cutoff radius in A (if cutoff is true)
	@param dens      Average atomic density of the nanoparticle
	@param damping   cfg->damping
*/
__global__ void AddCutoffKernelNeutron(float * const I, const float * const q, const float SLaver, const unsigned int Nq, const float Rcut, const float dens, const bool damping);

/**
    Adds the average density correction to the scattering intensity when the cut-off is enabled (cfg.cutoff == true)

	@param GSadd     Grid size for AddCutoffKernel... kernels
	@param *dI       Scattering intensity array (device)
	@param *NatomEl  Array containing the total number of atoms of each chemical element (host)
	@param *cfg      Parameters of simulation
	@param *dFF      X-ray atomic form-factor array for all chemical elements (device)
	@param SL        Array of neutron scattering lengths for all chemical elements
	@param *dq       Scattering vector magnitude array (device)
	@param Ntot      Total number of atoms in the nanoparticle
*/
void AddCutoffCUDA(const unsigned int GSadd, float * const dI, const unsigned int *const NatomEl, const config * const cfg, const float * const dFF, const vector<double> SL, const float * const dq, const unsigned int Ntot);

/**
	Organazies the computations of the scattering intensity (powder diffraction pattern) using the histogram of interatomic distances with CUDA

	@param DeviceNUM CUDA device number
	@param **I       Scattering intensity array (host). The memory is allocated inside the function
	@param *rij_hist Histogram of interatomic distances (device).
	@param *NatomEl  Array containing the total number of atoms of each chemical element (host)
	@param *cfg      Parameters of simulation
	@param *dFF      X-ray atomic form-factor array for all chemical elements (device)
	@param SL        Array of neutron scattering lengths for all chemical elements
	@param *dq       Scattering vector magnitude array (device)
	@param Ntot      Total number of atoms in the nanoparticle
*/
void calcInt1DHistCuda(const int DeviceNUM, double ** const I, const unsigned long long int * const rij_hist, const unsigned int *const NatomEl, const config * const cfg, const float * const dFF, const vector<double> SL, const float * const dq, const unsigned int Ntot);

//the following functions are used to calculate the PDFs

/**
	Computes the partial radial distribution function (RDF)

	@param *dPDF     Partial PDF array
	@param *rij_hist Histogram of interatomic distances (device)
	@param Nhist     Size of the partial histogram of interatomic distances
	@param mult      1 / (Ntot * bin_width)
*/
__global__ void calcPartialRDFkernel(float * const dPDF, const unsigned long long int * const rij_hist, const unsigned int Nhist, const float mult);

/**
	Computes the partial pair distribution function (PDF)

	@param *dPDF     Prtial PDF array
	@param *rij_hist Histogram of interatomic distances (device)
	@param Nhist     Size of the partial histogram of interatomic distances
	@param mult      1 / (4 * PI * rho * Ntot * bin_width)
	@param bin       Width of the histogram bin
*/
__global__ void calcPartialPDFkernel(float * const dPDF, const unsigned long long int * const rij_hist, const unsigned int Nhist, const float mult, const float bin);

/**
	Computes the partial reduced pair distribution function (rPDF)

	@param *dPDF     Partial PDF array.
	@param *rij_hist Histogram of interatomic distances (device)
	@param Nhist     Size of the partial histogram of interatomic distances
	@param mult      1 / (Ntot * bin_width)
	@param submult   4 * PI * rho * NatomEl_i * NatomEl_j / SQR(Ntot)
	@param bin       Width of the histogram bin
*/
__global__ void calcPartialRPDFkernel(float * const dPDF, const unsigned long long int * const rij_hist, const unsigned int Nhist, const float mult, const float submult, const float bin);

/**
	Computes the total PDF using the partial PDFs

	@param *dPDF   Total (first Nhist elements) + partial PDF array. The memory is allocated inside the function.
	@param Nstart  Index of the first element of the partial PDF whcih will be added to the total PDF in this kernel call
	@param Nhist   Size of the partial histogram of interatomic distances
	@param multIJ  FF_i(q0) * FF_j(q0) / <FF> (for x-ray) and SL_i * SL_j / <SL> (for neutron)
*/
__global__ void calcPDFkernel(float * const dPDF, const unsigned int Nstart, const unsigned int Nhist, const float multIJ);

/**
	Depending on the computational scenario organazies the computations of the scattering intensity (powder diffraction pattern) or PDF using the histogram of interatomic distances with CUDA

	@param DeviceNUM       CUDA device number
	@param **I             Scattering intensity array (host). The memory is allocated inside the function.
	@param **PDF           PDF array (host). The memory is allocated inside the function.
	@param *cfg            Parameters of simulation
	@param *NatomEl        Array containing the total number of atoms of each chemical element (host)
	@param *NatomEl_outer  Array containing the number of atoms of each chemical element including the atoms in the outer sphere (only if cfg.cutoff is True) (host)
	@param *ra             Atomic coordinate array (device)
	@param *dFF            X-ray atomic form-factor array for all chemical elements (device)
	@param SL              Array of neutron scattering lengths for all chemical elements
	@param *dq             Scattering vector magnitude array (device)
*/
void calcPDFandDebyeCuda(const int DeviceNUM, double ** const I, double ** const PDF, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const float4 * const ra, const float * const dFF, const vector<double> SL, const float * const dq);

//the following functions are used to calculate the powder diffraction pattern using the original Debye equation (without the histogram approximation)

/**
	Computes xray (source == xray) or neutron (source == neutron) scattering intensity (powder diffraction pattern) using the histogram of interatomic distances

	@param source  xray or neutron
	@param *I      Scattering intensity array
	@param *FFi    X-ray atomic form factor for the i-th atoms (all the i-th atoms are of the same chemical element for one kernel call) (NULL if source is neutron)
	@param *FFj    X-ray atomic form factor for the j-th atoms (all the j-th atoms are of the same chemical element for one kernel call) (NULL if source is neutron)
	@param SLij    Product of the scattering lenghts of i-th j-th atoms (0 if source is xray)
	@param *q      Scattering vector magnitude array
	@param Nq      Size of the scattering intensity array
	@param *ri     Pointer to the coordinate of the 1st i-th atom in ra array
	@param *rj     Pointer to the coordinate of the 1st j-th atom in ra array
	@param iMax    Total number of i-th atoms for this kernel call
	@param jMax    Total number of j-th atoms for this kernel call
	@param diag    True if the j-th atoms and the i-th atoms are the same (diagonal) for this kernel call
	@param mult    Multiplier, equal to 2 if cutoff is false, otherwise equals to 2 if j-th atoms belong to the inner sphere or to 1 if to the outer
	@param Rcut2   Square fo the cutoff radius in A^2 (float(SQR(cfg->Rcutoff)), if cutoff is true)
	@param damping cfg->damping
*/
template <unsigned int BlockSize2D, bool cutoff> __global__ void calcIntDebyeKernel(const unsigned int source, float * const I, const float * const FFi, const float * const FFj, const float SLij, const float * const q, const unsigned int Nq, const float4 * const ri, const float4 * const rj, const unsigned int iMax, const unsigned int jMax, const bool diag, const float mult, const float Rcut2, const bool damping);

/**
	Organazies the computations of the scattering intensity (powder diffraction pattern) using the original Debye equation (without the histogram approximation) with CUDA

	@param DeviceNUM CUDA device number
	@param **I             Scattering intensity array (host). The memory is allocated inside the function.
	@param *cfg            Parameters of simulation
	@param *NatomEl        Array containing the total number of atoms of each chemical element (host)
	@param *NatomEl_outer  Array containing the number of atoms of each chemical element including the atoms in the outer sphere (only if cfg.cutoff is True) (host)
	@param *ra             Atomic coordinate array (device)
	@param *dFF            X-ray atomic form-factor array for all chemical elements (device)
	@param SL              Array of neutron scattering lengths for all chemical elements
	@param *dq             Scattering vector magnitude array (device)
*/
void calcIntDebyeCuda(const int DeviceNUM, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const float4 * const ra, const float * const dFF, const vector<double> SL, const float * const dq);

//the following functions are used to calculate the partial scattering intensities (for each pair of the structural blocks) using the original Debye equation (without the histogram approximation)

/**
	Computes the partial scattering intensity (*Ipart) from the partials sums (*I) computed by different thread blocks

	@param *I     Scattering intensity array (partials sums as computed by thread blocks)
	@param *Ipart Partial scattering intensity array
	@param Nq     Resolution of the total scattering intensity (powder diffraction pattern)
	@param Nsum   Number of parts to sum (equalt to the total number of thread blocks in the grid)
*/
__global__ void sumIpartialKernel(float * const I, float * const Ipart, const unsigned int Nq, const unsigned int Nsum);

/**
	Computes the total scattering intensity (powder diffraction pattern) using the partial scattering intensity

	@param *I     Partial + total (first Nq elements) scattering intensity array
	@param Nq     Resolution of the total scattering intensity (powder diffraction pattern)
	@param Npart  Number of the partial intensities to sum
*/
__global__ void integrateIpartialKernel(float * const I, const unsigned int Nq, const unsigned int Nparts);

/**
	Organazies the computations of the scattering intensity (powder diffraction pattern + partial intensities) using the original Debye equation (without the histogram approximation) with CUDA

	@param DeviceNUM CUDA device number
	@param **I       Partial + total scattering intensity array (host). The memory is allocated inside the function.
	@param *cfg      Parameters of simulation
	@param *NatomEl  Array containing the total number of atoms of each chemical element (host)
	@param *ra       Atomic coordinate array (device)
	@param *dFF      X-ray atomic form-factor array for all chemical elements (device)
	@param SL        Array of neutron scattering lengths for all chemical elements
	@param *dq       Scattering vector magnitude array (device)
	@param *Block    Array of the structural blocks 
*/
void calcIntPartialDebyeCuda(const int DeviceNUM, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const float4 * const ra, const float * const dFF, const vector <double> SL, const float * const dq, const block * const Block);

//the following functions are used to set the CUDA device, copy/delete the data to/from the device memory

/**
	Queries all CUDA devices. Checks and sets the CUDA device number
	Returns 0 if OK and -1 if no CUDA devices found

	@param *DeviceNUM CUDA device number
*/
int SetDeviceCuda(int * const DeviceNUM);

/**
	Copies the atomic coordinates (ra), scattering vector magnitude (q) and the x-ray atomic form-factors (FF) to the device memory	

	@param *q      Scattering vector magnitude (host)
	@param *cfg    Parameters of simulation
	@param *ra     Atomic coordinates (host)
	@param **dra   Atomic coordinates (device). The memory is allocated inside the function
	@param **dFF   X-ray atomic form-factors (device). The memory is allocated inside the function
	@param **dq    Scattering vector magnitude (device). The memory is allocated inside the function
	@param FF      X-ray atomic form-factors (host)
*/
void dataCopyCUDA(const double *const q, const config * const cfg, const vector < vect3d <double> > * const ra, float4 ** const dra, float ** const dFF, float ** const dq, const vector <double*> FF);

/**
	Deletes the atomic coordinates (ra), scattering vector magnitude (dq) and the x-ray atomic form-factors (dFF) from the device memory

	@param *ra    Atomic coordinates (device)
	@param *dFF   X-ray atomic form-factors (device)
	@param *dq    Scattering vector magnitude (device)
	@param Nel    Total number of different chemical elements in the nanoparticle
*/
void delDataFromDevice(float4 * const ra, float * const dFF, float * const dq, const unsigned int Nel);

/**
	Returns the theoretical peak performance of the CUDA device

	@param deviceProp  Device properties object
	@param show        If True, show the device information on screen
*/
unsigned int GetGFLOPS(const cudaDeviceProp deviceProp, const bool show);

//Returns the theoretical peak performance of the CUDA device
unsigned int GetGFLOPS(const cudaDeviceProp deviceProp, const bool show = false){
	const unsigned int cc = deviceProp.major * 10 + deviceProp.minor; //compute capability
	const unsigned int MP = deviceProp.multiProcessorCount; //number of multiprocessors
	const unsigned int clockRate = deviceProp.clockRate / 1000; //GPU clockrate
	unsigned int ALUlanes = 64;	
	switch (cc){
	case 10:
	case 11:
	case 12:
	case 13:
		ALUlanes = 8;
		break;
	case 20:
		ALUlanes = 32;
		break;
	case 21:
		ALUlanes = 48;
		break;
	case 30:
	case 35:
	case 37:
		ALUlanes = 192;
		break;
	case 50:
	case 52:
		ALUlanes = 128;
		break;
	case 60:
		ALUlanes = 64;
		break;
	case 61:
	case 62:
		ALUlanes = 128;
		break;
	case 70:
	case 72:
	case 75:
		ALUlanes = 64;
		break;
	}
	unsigned int GFLOPS = MP * ALUlanes * 2 * clockRate / 1000;
	if (show) {
		cout << "GPU name: " << deviceProp.name << "\n";
		cout << "CUDA compute capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
		cout << "Number of multiprocessors: " << MP << "\n";
		cout << "GPU clock rate: " << clockRate << " MHz" << "\n";
		cout << "Theoretical peak performance: " << GFLOPS << " GFLOPs\n" << endl;
	}
	return GFLOPS;
}

//Resets the 2D scattering intensity array
__global__ void zeroInt2DKernel(float * const I, const unsigned int Nq, const unsigned int Nfi){
	const unsigned int iq = blockDim.y * blockIdx.y + threadIdx.y, ifi = blockDim.x * blockIdx.x + threadIdx.x;
	if ((iq < Nq) && (ifi < Nfi))	I[iq*Nfi + ifi] = 0;
}

//Resets the 2D scattering amplitude arrays (real and imaginary parts)
__global__ void zeroAmp2DKernel(float * const Ar, float * const Ai, const unsigned int Nq, const unsigned int Nfi){
	const unsigned int iq = blockDim.y * blockIdx.y + threadIdx.y, ifi = blockDim.x * blockIdx.x + threadIdx.x;
	if ((iq < Nq) && (ifi < Nfi)){
		Ar[iq*Nfi + ifi] = 0;
		Ai[iq*Nfi + ifi] = 0;
	}
}

//Computes the 2D scattering intensity using the scattering amplitude
__global__ void Sum2DKernel(float * const I,const float * const Ar, const float * const Ai, const unsigned int Nq, const unsigned int Nfi){
	const unsigned int iq = blockDim.y * blockIdx.y + threadIdx.y, ifi = blockDim.x * blockIdx.x + threadIdx.x;
	if ((iq < Nq) && (ifi < Nfi))	I[iq * Nfi + ifi] += SQR(Ar[iq * Nfi + ifi]) + SQR(Ai[iq * Nfi + ifi]);
}

//Multiplies the 2D scattering intensity by a normalizing factor
__global__ void Norm2DKernel(float * const I, const unsigned int Nq, const unsigned int Nfi, const float norm){
	const unsigned int iq = blockDim.y * blockIdx.y + threadIdx.y, ifi = blockDim.x * blockIdx.x + threadIdx.x;
	if ((iq < Nq) && (ifi < Nfi))	I[iq * Nfi + ifi] *= norm;
}

//Computes the polarization factor and multiplies the 2D scattering intensity by this factor
template <unsigned int BlockSize2D> __global__ void PolarFactor2DKernel(float * const I, const unsigned int Nq, const unsigned int Nfi, const float * const q, const float lambda){
	const unsigned int iq = BlockSize2D * blockIdx.y + threadIdx.y, ifi = BlockSize2D * blockIdx.x + threadIdx.x;
	const unsigned int iqCopy = BlockSize2D * blockIdx.y + threadIdx.x;
	__shared__ float factor[BlockSize2D];
	if ((threadIdx.y == 0) && (iqCopy < Nq)) {
		//polarization factor is computed only by the threads of the first warp (half-warp for the devices with CC < 2.0) and stored in the shared memory
		const float sintheta = q[iqCopy] * (lambda * 0.25f / PIf);
		const float cos2theta = 1.f - 2.f * SQR(sintheta);
		factor[threadIdx.x] = 0.5f * (1.f + SQR(cos2theta));
	}
	__syncthreads();
	if ((iq < Nq) && (ifi < Nfi)) I[iq * Nfi + ifi] *= factor[threadIdx.y]; 
}

//Computes polarization factor and multiplies scattering intensity by this factor
__global__ void PolarFactor1DKernel(float * const I, const unsigned int Nq, const float * const q, const float lambda){
	const unsigned int iq = blockIdx.x * blockDim.x + threadIdx.x;
	if (iq < Nq)	{
		const float sintheta = q[iq] * (lambda * 0.25f / PIf);
		const float cos2theta = 1.f - 2.f * SQR(sintheta);
		const float factor = 0.5f * (1.f + SQR(cos2theta));
		I[blockIdx.y * Nq + iq] *= factor;
	}
}

//Computes the real and imaginary parts of the 2D x-ray (source == xray) or neutron (source == neutron) scattering amplitude in the polar coordinates(q, q_fi) of the reciprocal space
template <unsigned int BlockSize2D, unsigned int SizeR> __global__ void calcInt2DKernel(const unsigned int source, float * const Ar, float * const Ai, const float * const q, const unsigned int Nq, const unsigned int Nfi, const float3 CS[], const float lambda, const float4 * const ra, const unsigned int Nfin, const float * const FF, const float SL){
	//to avoid bank conflicts for shared memory operations BlockSize2D should be equal to the size of the warp (or half-warp for the devices with the CC < 2.0)
	//SizeR should be a multiple of BlockSize2D
	const unsigned int iq = BlockSize2D * blockIdx.y + threadIdx.y, ifi = BlockSize2D * blockIdx.x + threadIdx.x; //each thread computes only one element of the 2D amplitude matrix
	const unsigned int iqCopy = BlockSize2D * blockIdx.y + threadIdx.x;//copying of the scattering vector magnitude to the shared memory is performed by the threads of the same warp (half-warp)
	__shared__ float lFF[BlockSize2D]; //cache array for the x-ray  atomic from-factors
	__shared__ float qi[BlockSize2D]; //cache array for the scattering vector magnitude
	__shared__ float4 r[SizeR]; //cache array for the atomic coordinates
	if ((threadIdx.y == 0) && (iqCopy < Nq)) qi[threadIdx.x] = q[iqCopy]; //loading scattering vector magnitude to the shared memory (only threads from the third warp (first half of the second warp) are used)
	if ((source == xray) && (threadIdx.y == 2) && (iqCopy < Nq)) lFF[threadIdx.x] = FF[iqCopy]; //loading x-ray atomic form-factors to the shared memory (only threads from the first warp (half-warp) are used)
	__syncthreads(); //synchronizing after loading to the shared memory
	float cosfi = 0, sinfi = 0;
	float3 qv; //scattering vector	
	if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
		__sincosf(ifi * 2.f * PIf / Nfi, &sinfi, &cosfi); //computing sin(fi), cos(fi)
		const float sintheta = 0.25f * lambda * qi[threadIdx.y] / PIf; //q = 4pi/lambda*sin(theta)
		const float costheta = 1.f - SQR(sintheta); //theta in [0, pi/2];
		qv = make_float3(costheta * cosfi, costheta * sinfi, -sintheta) * qi[threadIdx.y]; //computing the scattering vector
		//instead of pre-multiplying the atomic coordinates by the rotational matrix we are pre-multiplying the scattering vector by the transposed rotational matrix (dot(qv,r) will be the same)
		qv = make_float3(dot(qv, CS[0]), dot(qv, CS[1]), dot(qv, CS[2]));
	}
	float lAr = 0, lAi = 0;
	const unsigned int Niter = Nfin / SizeR + BOOL(Nfin % SizeR);//we don't have enough shared memory to load the array of atomic coordinates as a whole, so we do it with iterations
	for (unsigned int iter = 0; iter < Niter; iter++){
		unsigned int NiterFin = MIN(Nfin - iter * SizeR, SizeR); //checking for the margins of the atomic coordinates array
		if (threadIdx.y < SizeR / BlockSize2D) {
			const unsigned int iAtom = threadIdx.y * BlockSize2D + threadIdx.x; 
			if (iAtom < NiterFin) r[iAtom] = ra[iter * SizeR + iAtom]; //loading the atomic coordinates to the shared memory
		}
		__syncthreads(); //synchronizing after loading to shared memory
		if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
			for (unsigned int iAtom = 0; iAtom < NiterFin; iAtom++){
				__sincosf(dot(qv, r[iAtom]), &sinfi, &cosfi); //cos(dot(qv*r)), sin(dot(qv,r))
				lAr += cosfi; //real part of the amplitute
				lAi += sinfi; //imaginary part of the amplitute
			}
		}
		__syncthreads(); //synchronizing before the next loading starts
	}
	if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
		if (source == xray) {
			Ar[iq * Nfi + ifi] += lFF[threadIdx.y] * lAr; //multiplying the real part of the amplitude by the form-factor and writing the results to the global memory
			Ai[iq * Nfi + ifi] += lFF[threadIdx.y] * lAi; //doing the same for the imaginary part of the amplitude
		}
		else {
			Ar[iq * Nfi + ifi] += SL * lAr;
			Ai[iq * Nfi + ifi] += SL * lAi;
		}
	}	
}

//Organazies the computations of the 2D scattering intensity in the polar coordinates(q, q_fi) of the reciprocal space with CUDA
void calcInt2DCuda(const int DeviceNUM, double *** const I2D, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const float4 * const ra, const float * const dFF, const vector<double> SL, const float * const dq){
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM); //getting device information
	const unsigned int GFLOPS = GetGFLOPS(deviceProp); //theoretical peak GPU performance
	const unsigned int BlockSize2D = BlockSize2Dsmall;
	unsigned int MaxAtomsPerLaunch = 0;
	if (deviceProp.kernelExecTimeoutEnabled){ //killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel execution time in seconds
		const double k = 4.e-8; // t = k * MaxAtomsPerLaunch * Nq * Nfi / GFLOPS
		MaxAtomsPerLaunch = (unsigned int)((tmax * GFLOPS) / (k * cfg->q.N * cfg->Nfi)); //maximum number of atoms per kernel launch
	}
	dim3 dimBlock(BlockSize2D, BlockSize2D); //2d thread block size
	dim3 dimGrid(cfg->Nfi / BlockSize2D + BOOL(cfg->Nfi % BlockSize2D), cfg->q.N / BlockSize2D + BOOL(cfg->q.N % BlockSize2D)); //grid size
	//2d scattering intensity should be calculated for the preset orientation of the sample (or averaged over multiple orientations specified by mesh)
	double dalpha = (cfg->Euler.max.x - cfg->Euler.min.x) / cfg->Euler.N.x, dbeta = (cfg->Euler.max.y - cfg->Euler.min.y) / cfg->Euler.N.y, dgamma = (cfg->Euler.max.z - cfg->Euler.min.z) / cfg->Euler.N.z;
	if (cfg->Euler.N.x < 2) dalpha = 0;
	if (cfg->Euler.N.y < 2) dbeta = 0;
	if (cfg->Euler.N.z < 2) dgamma = 0;
	float3 CS[3], *dCS; //three rows of the transposed rotational matrix for the host and the device
	cudaMalloc(&dCS, 3 * sizeof(float3)); //allocating the device memory for the transposed rotational matrix
	//allocating memory on the device for amplitude and intensity 2D arrays
	//GPU has linear memory, so we stretch 2D arrays into 1D arrays
	float *dI, *dAr, *dAi;
	const unsigned int Nm = cfg->q.N * cfg->Nfi; //dimension of 2D intensity array		
	cudaMalloc(&dAr, Nm * sizeof(float));
	cudaMalloc(&dAi, Nm * sizeof(float));
	cudaMalloc(&dI, Nm * sizeof(float));
	cudaThreadSynchronize(); //synchronizing before calculating the amplitude
	zeroInt2DKernel << <dimGrid, dimBlock >> >(dI, cfg->q.N, cfg->Nfi); //reseting the 2D intensity matrix
	for (unsigned int ia = 0; ia < cfg->Euler.N.x; ia++){
		const double alpha = cfg->Euler.min.x + (ia + 0.5)*dalpha;
		for (unsigned int ib = 0; ib < cfg->Euler.N.y; ib++){
			const double beta = cfg->Euler.min.y + (ib + 0.5)*dbeta;
			for (unsigned int ig = 0; ig < cfg->Euler.N.z; ig++){
				const double gamma = cfg->Euler.min.z + (ig + 0.5)*dgamma;
				const vect3d <double> euler(alpha, beta, gamma);
				vect3d <double> RM0, RM1, RM2; //three rows of the rotational matrix
				calcRotMatrix(&RM0, &RM1, &RM2, euler, cfg->EulerConvention); //calculating the rotational matrix
				CS[0] = make_float3(float(RM0.x), float(RM1.x), float(RM2.x)); //transposing the rotational matrix
				CS[1] = make_float3(float(RM0.y), float(RM1.y), float(RM2.y));
				CS[2] = make_float3(float(RM0.z), float(RM1.z), float(RM2.z));
				cudaMemcpy(dCS, CS, 3 * sizeof(float3), cudaMemcpyHostToDevice); //copying transposed rotational matrix from the host memory to the device memory 
				zeroAmp2DKernel << <dimGrid, dimBlock >> >(dAr, dAi, cfg->q.N, cfg->Nfi); //reseting 2D amplitude arrays
				cudaThreadSynchronize(); //synchronizing before calculation starts to ensure that amplitude arrays were successfully set to zero
				unsigned int inp = 0;
				for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){ //looping over chemical elements (or ions)
					if (MaxAtomsPerLaunch) { //killswitch is enabled so MaxAtomsPerLaunch is set
						for (unsigned int i = 0; i < NatomEl[iEl] / MaxAtomsPerLaunch + BOOL(NatomEl[iEl] % MaxAtomsPerLaunch); i++) { //looping over the iterations
							const unsigned int Nst = inp + i*MaxAtomsPerLaunch; //index for the first atom on the current iteration step
							const unsigned int Nfin = MIN(Nst + MaxAtomsPerLaunch, inp + NatomEl[iEl]) - Nst; //index for the last atom on the current iteration step
							//float time; //time control sequence
							//cudaEvent_t start, stop;
							//cudaEventCreate(&start);
							//cudaEventCreate(&stop);
							//cudaEventRecord(start, 0);
							if (cfg->source == xray) calcInt2DKernel <BlockSize2Dsmall, 8 * BlockSize2Dsmall> << <dimGrid, dimBlock >> >(xray, dAr, dAi, dq, cfg->q.N, cfg->Nfi, dCS, float(cfg->lambda), ra + Nst, Nfin, dFF + iEl * cfg->q.N, 0);
							else calcInt2DKernel <BlockSize2Dsmall, 8 * BlockSize2Dsmall> << <dimGrid, dimBlock >> >(neutron, dAr, dAi, dq, cfg->q.N, cfg->Nfi, dCS, float(cfg->lambda), ra + Nst, Nfin, NULL, float(SL[iEl]));
							cudaThreadSynchronize(); //synchronizing to ensure that additive operations does not overlap
							//cudaEventRecord(stop, 0);
							//cudaEventSynchronize(stop);
							//cudaEventElapsedTime(&time, start, stop);
							//cout << "calcInt2DKernel execution time is: " << time << " ms\n" << endl;
						}
					}
					else { //killswitch is disabled so we execute the kernels for the entire ensemble of atoms
						const unsigned int Nst = inp;
						const unsigned int Nfin = NatomEl[iEl];
						if (cfg->source == xray) calcInt2DKernel <BlockSize2Dsmall, 8 * BlockSize2Dsmall> << <dimGrid, dimBlock >> >(xray, dAr, dAi, dq, cfg->q.N, cfg->Nfi, dCS, float(cfg->lambda), ra + Nst, Nfin, dFF + iEl * cfg->q.N, 0);
						else calcInt2DKernel <BlockSize2Dsmall, 8 * BlockSize2Dsmall> << <dimGrid, dimBlock >> >(neutron, dAr, dAi, dq, cfg->q.N, cfg->Nfi, dCS, float(cfg->lambda), ra + Nst, Nfin, NULL, float(SL[iEl]));
						cudaThreadSynchronize(); //synchronizing to ensure that additive operations does not overlap
					}
					inp += NatomEl[iEl];
				}				
				Sum2DKernel << <dimGrid, dimBlock >> >(dI, dAr, dAi, cfg->q.N, cfg->Nfi); //calculating the 2d scattering intensity by the scattering amplitude
			}
		}
	}
	cudaFree(dCS);
	cudaFree(dAr);
	cudaFree(dAi);
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl]; //total number of atoms
	const float norm = 1.f / (Ntot*cfg->Euler.N.x*cfg->Euler.N.y*cfg->Euler.N.z); //normalizing factor
	Norm2DKernel << <dimGrid, dimBlock >> >(dI, cfg->q.N, cfg->Nfi, norm); //normalizing the 2d scattering intensity
	cudaThreadSynchronize(); //synchronizing to ensure that multiplying operations does not overlap
	if (cfg->PolarFactor) PolarFactor2DKernel <BlockSize2Dsmall> << <dimGrid, dimBlock >> >(dI, cfg->q.N, cfg->Nfi, dq, float(cfg->lambda));//multiplying the 2d intensity by polar factor
	float * const hI = new float[Nm]; //host array for 2D intensity
	cudaMemcpy(hI, dI, Nm*sizeof(float), cudaMemcpyDeviceToHost);  //copying the 2d intensity matrix from the device memory to the host memory 
	cudaFree(dI);
	*I = new double[cfg->q.N]; //array for 1d scattering intensity I[q] (I2D[q][fi] averaged over polar angle fi)
	*I2D = new double*[cfg->q.N]; //array for 2d scattering intensity 
	for (unsigned int iq = 0; iq < cfg->q.N; iq++){
		(*I)[iq] = 0;
		(*I2D)[iq] = new double[cfg->Nfi];
		for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++)	{
			(*I2D)[iq][ifi] = double(hI[iq * cfg->Nfi + ifi]);
			(*I)[iq] += (*I2D)[iq][ifi]; //calculating the 1d intensity (averaging I2D[q][fi] over the polar angle fi)
		}
		(*I)[iq] /= cfg->Nfi;
	}
	delete[] hI;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "2D pattern calculation time: " << time/1000 << " s" << endl;
}

//Resets 1D float array of size N
__global__ void zero1DFloatArrayKernel(float * const A, const unsigned int N){
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i<N) A[i]=0;
}

//Adds the diagonal elements(j == i) of the Debye double sum to the x - ray scattering intensity
__global__ void addIKernelXray(float * const I, const float * const FF, const unsigned int Nq, const unsigned int N) {
	const unsigned int iq = blockIdx.x * blockDim.x + threadIdx.x;
	if (iq < Nq)	{
		const float lFF = FF[iq];
		I[iq] += SQR(lFF) * N;
	}
}

//Adds the diagonal elements(j == i) of the Debye double sum to the neutron scattering intensity
__global__ void addIKernelNeutron(float * const I, const unsigned int Nq, const float Add) {
	const unsigned int iq = blockIdx.x * blockDim.x + threadIdx.x;
	if (iq < Nq)	I[iq] += Add;
}

//Computes the total scattering intensity (first Nq elements) from the partials sums computed by different thread blocks
__global__ void sumIKernel(float * const I, const unsigned int Nq, const unsigned int Nsum){
	const unsigned int iq = blockDim.x * blockIdx.x + threadIdx.x;
	if (iq<Nq) {
		for (unsigned int j = 1; j < Nsum; j++)	I[iq] += I[j * Nq + iq];
	}
}

//Resets the histogram array (unsigned long long int)
__global__ void zeroHistKernel(unsigned long long int * const rij_hist, const unsigned int N){
	const unsigned int i=blockDim.x * blockIdx.x + threadIdx.x;
	if (i<N) rij_hist[i]=0;
}	

//Computes the histogram of interatomic distances
template <unsigned int BlockSize2D, bool cutoff> __global__ void calcHistKernel(const float4 *  const __restrict__ ri, const float4 *  const __restrict__ rj, const unsigned int iMax, const unsigned int jMax, unsigned long long int *const rij_hist, const float bin, const unsigned int Nhistcopies, const unsigned int Nhist, const float Rcut2, const unsigned long long int add, const bool diag){
	if ((diag) && (blockIdx.x < blockIdx.y)) return; //we need to calculate inter-atomic distances only for j > i, so if we are in the diagonal grid, all the subdiagonal blocks (for which j < i for all threads) do nothing and return
	const unsigned int jt = threadIdx.x, it = threadIdx.y;
	const unsigned int j = blockIdx.x * BlockSize2D + jt;
	const unsigned int iCopy = blockIdx.y * BlockSize2D + jt; //jt!!! memory transaction are performed by the threads of the same warp to coalesce them
	const unsigned int i = blockIdx.y * BlockSize2D + it;
	unsigned int copyind = 0;
	if (Nhistcopies > 1) copyind = ((it * BlockSize2D + jt) % Nhistcopies) * Nhist; //some optimization for CC < 2.0. Making multiple copies of the histogram array reduces the number of atomicAdd() operations on the same elements.
	__shared__ float4 ris[BlockSize2D], rjs[BlockSize2D]; //cache arrays for atomic coordinates 
	if ((it == 0) && (j < jMax)) rjs[jt] = rj[j]; //copying atomic coordinates for j-th (column) atoms (only the threads of the first half-warp are used)
	if ((it == 2) && (iCopy < iMax)) ris[jt] = ri[iCopy]; //the same for i-th (row) atoms (only the threads of the first half-warp of the second warp for CC < 2.0 are used)
	__syncthreads(); //sync to ensure that copying is complete
	if ((j < jMax) && (i < iMax) && ((j > i) || (!diag))) {
		const float rij2 = SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z);//calculate square of distance	
		if (cutoff){	
			if (rij2 < Rcut2) {
				const unsigned int index = (unsigned int)(sqrtf(rij2) / bin); //get the index of histogram bin
				atomicAdd(&rij_hist[copyind + index], add); //add +2 or +1 to histogram bin
			}
		}
		else {
			const unsigned int index = (unsigned int)(sqrtf(rij2) / bin); //get the index of histogram bin
			atomicAdd(&rij_hist[copyind + index], add); //add +1 to histogram bin
		}
	}
}

//Computes the total histogram (first Nhist elements) using the partial histograms (for the devices with the CUDA compute capability < 2.0)
__global__ void sumHistKernel(unsigned long long int * const rij_hist, const unsigned int Nhistcopies, const unsigned int Nfin, const unsigned int Nhist){
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < Nfin){
		for (unsigned int iCopy = 1; iCopy < Nhistcopies; iCopy++)	rij_hist[i] += rij_hist[Nhist * iCopy + i];
	}
}

//Organazies the computations of the histogram of interatomic distances with CUDA 
void calcHistCuda(const int DeviceNUM, unsigned long long int ** const rij_hist, const float4 * const ra, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const config * const cfg){
	const unsigned int BlockSize = BlockSize1Dsmall, BlockSize2D = BlockSize2Dsmall; //size of the thread blocks (256, 16x16)
	const unsigned int NhistEl = (cfg->Nel * (cfg->Nel + 1)) / 2 * cfg->Nhist;//Number of partial (Element1<-->Element2) histograms
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM); //getting the device properties
	const int cc = deviceProp.major * 10 + deviceProp.minor; //device compute capability
	unsigned int Nhistcopies = 1;
	if (cc<20){//optimization for the devices with CC < 2.0
		//atomic operations work very slow for the devices with Tesla architecture as compared with the modern devices
		//we minimize the number of atomic operations on the same elements by making multiple copies of pair-distribution histograms
		size_t free, total;
		cuMemGetInfo(&free, &total); //checking the amount of the free GPU memory	
		Nhistcopies = MIN(BlockSize,(unsigned int)(0.25 * float(free) / (NhistEl * sizeof(unsigned long long int)))); //set optimal number for histogram copies 
		if (!Nhistcopies) Nhistcopies = 1;
	}
	unsigned int GridSizeExecMax = 2048;
	const unsigned int GFLOPS = GetGFLOPS(deviceProp); //theoretical peak GPU performance
	if (deviceProp.kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 1.e-6; // t = k * GridSizeExecMax^2 * BlockSize2D^2 / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / k) / BlockSize2D), GridSizeExecMax);
	}
	//total histogram size is equal to the product of: partial histogram size for one pair of elements (Nhist), number of partial histograms ((Nel*(Nel + 1)) / 2), number of histogram copies (Nhistcopies)
	const unsigned int NhistTotal = NhistEl * Nhistcopies;
	cudaError err = cudaMalloc(rij_hist, NhistTotal * sizeof(unsigned long long int));//trying to allocate large amount of memory, check for errors
	if (err != cudaSuccess) cout << "Error in calcHistCuda(), cudaMalloc(): " << cudaGetErrorString(err) << endl;
	const unsigned int GSzero = MIN(65535, NhistTotal / BlockSize + BOOL(NhistTotal % BlockSize));//Size of the grid for zeroHistKernel (it could not be large than 65535)
	//reseting pair-distribution histogram array
	for (unsigned int iter = 0; iter < NhistTotal / BlockSize + BOOL(NhistTotal % BlockSize); iter += GSzero)	zeroHistKernel << < GSzero, BlockSize >> >(*rij_hist + iter*BlockSize, NhistTotal - iter*BlockSize);
	cudaThreadSynchronize();//synchronizing before the calculation starts
	dim3 blockgrid(BlockSize2D, BlockSize2D);//2D thread block size
	const float4 * * const raEl = new const float4*[cfg->Nel];
	raEl[0] = ra;
	for (unsigned int iEl = 1; iEl < cfg->Nel; iEl++) {
		(cfg->cutoff) ? raEl[iEl] = raEl[iEl - 1] + NatomEl_outer[iEl - 1] : raEl[iEl] = raEl[iEl - 1] + NatomEl[iEl - 1];
	}
	const float bin = float(cfg->hist_bin);
	const float Rcut2 = float(SQR(cfg->Rcutoff));
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		unsigned int jElSt = iEl;
		if (cfg->cutoff) jElSt = 0;
		for (unsigned int jEl = jElSt; jEl < cfg->Nel; jEl++) {//each time we move to the next pair of elements (iEl,jEl) we also move to the respective part of histogram (Nstart += Nhist)
			unsigned int jAtomST = 0;
			if ((cfg->cutoff) && (jEl < iEl)) jAtomST = NatomEl[jEl];
			unsigned int Nstart = 0;
			(jEl > iEl) ? Nstart = cfg->Nhist * (cfg->Nel * iEl - (iEl * (iEl + 1)) / 2 + jEl) : Nstart = cfg->Nhist * (cfg->Nel * jEl - (jEl * (jEl + 1)) / 2 + iEl);
			for (unsigned int iAtom = 0; iAtom < NatomEl[iEl]; iAtom += BlockSize2D * GridSizeExecMax){
				const unsigned int GridSizeExecY = MIN((NatomEl[iEl] - iAtom) / BlockSize2D + BOOL((NatomEl[iEl] - iAtom) % BlockSize2D), GridSizeExecMax);//Y-size of the grid on the current step
				const unsigned int iMax = MIN(BlockSize2D * GridSizeExecY, NatomEl[iEl] - iAtom);//index of the last i-th (row) atom
				if (iEl == jEl) jAtomST = iAtom;//loop should exclude subdiagonal grids
				for (unsigned int jAtom = jAtomST; jAtom < NatomEl[jEl]; jAtom += BlockSize2D * GridSizeExecMax){
					const unsigned int GridSizeExecX = MIN((NatomEl[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of the grid on the current step
					const unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomEl[jEl] - jAtom);//index of the last j-th (column) atom
					dim3 grid(GridSizeExecX, GridSizeExecY);
					bool diag = false;
					if ((iEl == jEl) && (iAtom == jAtom)) diag = true;//checking if we are on the diagonal grid or not
					/*float time;
					cudaEvent_t start, stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start, 0);*/
					if (cfg->cutoff) calcHistKernel <BlockSize2Dsmall, true> << <grid, blockgrid >> >(raEl[iEl] + iAtom, raEl[jEl] + jAtom, iMax, jMax, *rij_hist + Nstart, bin, Nhistcopies, NhistEl, Rcut2, 2, diag);
					else calcHistKernel <BlockSize2Dsmall, false> << <grid, blockgrid >> >(raEl[iEl] + iAtom, raEl[jEl] + jAtom, iMax, jMax, *rij_hist + Nstart, bin, Nhistcopies, NhistEl, 0, 2, diag);
					if (deviceProp.kernelExecTimeoutEnabled) cudaThreadSynchronize();//the kernel above uses atomic operation, it's hard to predict the execution time of a single kernel, so sync to avoid the killswitch triggering 
					/*cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&time, start, stop);
					cout << "calcHistKernel execution time is: " << time << " ms\n" << endl;*/
				}
				if (cfg->cutoff) {
					for (unsigned int jAtom = NatomEl[jEl]; jAtom < NatomEl_outer[jEl]; jAtom += BlockSize2D * GridSizeExecMax){
						unsigned int GridSizeExecX = MIN((NatomEl_outer[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl_outer[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of the grid on the current step
						unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomEl_outer[jEl] - jAtom);//index of the last j-th (column) atom
						dim3 grid(GridSizeExecX, GridSizeExecY);
						calcHistKernel <BlockSize2Dsmall, true> << <grid, blockgrid >> >(raEl[iEl] + iAtom, raEl[jEl] + jAtom, iMax, jMax, *rij_hist + Nstart, bin, Nhistcopies, NhistEl, Rcut2, 1, false);
						if (deviceProp.kernelExecTimeoutEnabled) cudaThreadSynchronize();//the kernel above uses atomic operation, it's hard to predict the execution time of a single kernel, so sync to avoid the killswitch triggering 
					}
				}
			}
		}
	}
	cudaThreadSynchronize();//synchronizing to ensure that all calculations ended before histogram copies summation starts
	delete[] raEl;
	if (Nhistcopies>1) {//summing the histogram copies
		const unsigned int GSsum = MIN(65535, NhistEl / BlockSize + BOOL(NhistEl % BlockSize));
		for (unsigned int iter = 0; iter < NhistEl / BlockSize + BOOL(NhistEl % BlockSize); iter += GSsum)	sumHistKernel << <GSsum, BlockSize >> >(*rij_hist + iter * BlockSize, Nhistcopies, NhistEl - iter * BlockSize, NhistEl);
	}
	cudaThreadSynchronize();//synchronizing before the further usage of histogram in other functions
}

//Computes the x-ray (source == xray) or neutron (source == neutron) scattering intensity (powder diffraction pattern) using the histogram of interatomic distances
template <unsigned int Size> __global__ void calcIntHistKernel(const unsigned int source, float * const I, const float * const FFi, const float * const FFj, const float SLij, const float *const q, const unsigned int Nq, const unsigned long long int *const rij_hist, const unsigned int iBinSt, const unsigned int Nhist, const unsigned int MaxBinsPerBlock, const float bin, const float Rcut, const bool damping){
	__shared__ long long int Nrij[Size];//cache array for the histogram
	__shared__ float damp[Size];
	Nrij[threadIdx.x] = 0;
	damp[threadIdx.x] = 1.;
	__syncthreads();
	const unsigned int iBegin = iBinSt + blockIdx.x * MaxBinsPerBlock;//first index for histogram bin to process
	const unsigned int iEnd = MIN(Nhist, iBegin + MaxBinsPerBlock);//last index for histogram bin to process
	if (iEnd < iBegin) return;
	const unsigned int Niter = (iEnd - iBegin) / blockDim.x + BOOL((iEnd - iBegin) % blockDim.x);//number of iterations
	for (unsigned int iter = 0; iter < Niter; iter++){//we don't have enough shared memory to load the histogram array as a whole, so we do it with iterations
		const unsigned int NiterFin = MIN(iEnd - iBegin - iter * blockDim.x, blockDim.x);//maximum number of histogram bins on current iteration step
		if (threadIdx.x < NiterFin) {
			const unsigned int index = iBegin + iter * blockDim.x + threadIdx.x;
			Nrij[threadIdx.x] = rij_hist[index]; //loading the histogram array to shared memory
			if (damping) {
				const float rij = ((float)(index) + 0.5f) * bin;//distance that corresponds to the current histogram bin
				const float x = PIf * rij / Rcut;
				damp[threadIdx.x] = __sinf(x) / x;
			}
		}
		__syncthreads();//synchronizing after loading
		for (unsigned int iterq = 0; iterq < (Nq / blockDim.x) + BOOL(Nq % blockDim.x); iterq++) {//if Nq > blockDim.x there will be threads that compute more than one element of the intensity array
			const unsigned int iq = iterq * blockDim.x + threadIdx.x;//index of the intensity array element
			if (iq < Nq) {//checking for the array margin				
				const float lq = q[iq];//copying the scattering vector magnitude to the local memory
				float lI = 0;
				for (unsigned int i = 0; i < NiterFin; i++) {//looping over the histogram bins
					if (Nrij[i]){
						const float qrij = lq * ((float)(iBegin + iter * blockDim.x + i) + 0.5f) * bin + 0.000001f;//distance that corresponds to the current histogram bin
						lI += Nrij[i] * damp[i] * __sinf(qrij) / qrij;//scattering intensity without form factors
					}
				}
				if (source == xray) I[blockIdx.x * Nq + iq] += lI * FFi[iq] * FFj[iq];//multiplying intensity by form-factors and storing the results in global memory
				else I[blockIdx.x * Nq + iq] += lI * SLij;
			}
		}
		__syncthreads();//synchronizing threads before the next iteration step
	}
}

//Adds the average density correction to the xray scattering intensity when the cut-off is enabled (cfg.cutoff == true)
__global__ void AddCutoffKernelXray(float * const I, const float * const q, const float * const FF, const unsigned int * const NatomEl, const unsigned int Nel, const unsigned int Ntot, const unsigned int Nq, const float Rcut, const float dens, const bool damping){
	const unsigned int iq = blockIdx.x * blockDim.x + threadIdx.x;
	if (iq < Nq) {
		float FFaver = 0;
		for (unsigned int iEl = 0; iEl < Nel; iEl++) FFaver += FF[iEl * Nq + iq] * NatomEl[iEl];
		FFaver /= Ntot;
		const float lq = q[iq];
		if (lq > 0.000001f) {
			const float qrcut = lq * Rcut;
			if (damping) I[iq] += 4.f * PIf * Ntot * dens * SQR(FFaver) * SQR(Rcut) * __sinf(qrcut) / (lq * (SQR(qrcut) - SQR(PIf)));
			else I[iq] += 4.f * PIf * Ntot * dens * SQR(FFaver) * (Rcut * __cosf(qrcut) - __sinf(qrcut) / lq) / SQR(lq);
		}
	}
}

//Adds the average density correction to the neutron scattering intensity when the cut-off is enabled (cfg.cutoff == true)
__global__ void AddCutoffKernelNeutron(float * const I, const float * const q, const float SLaver, const unsigned int Ntot, const unsigned int Nq, const float Rcut, const float dens, const bool damping){
	const unsigned int iq = blockIdx.x * blockDim.x + threadIdx.x;
	if (iq < Nq) {
		const float lq = q[iq];
		if (lq > 0.000001f) {
			const float qrcut = lq * Rcut;
			if (damping) I[iq] += 4.f * PIf * Ntot * dens * SQR(SLaver) * SQR(Rcut) * __sinf(qrcut) / (lq * (SQR(qrcut) - SQR(PIf)));
			else I[iq] += 4.f * PIf * Ntot * dens * SQR(SLaver) * (Rcut * __cosf(qrcut) - __sinf(qrcut) / lq) / SQR(lq);
		}
	}
}

//Adds the average density correction to the scattering intensity when the cut-off is enabled (cfg.cutoff == true)
void AddCutoffCUDA(const unsigned int GSadd, float * const dI, const unsigned int *const NatomEl, const config * const cfg, const float * const dFF, const vector<double> SL, const float * const dq, const unsigned int Ntot) {
	if (cfg->source == xray) {
		unsigned int * dNatomEl = NULL;
		cudaMalloc(&dNatomEl, cfg->Nel * sizeof(unsigned int));
		cudaMemcpy(dNatomEl, NatomEl, cfg->Nel * sizeof(unsigned int), cudaMemcpyHostToDevice);
		AddCutoffKernelXray << <GSadd, BlockSize1Dsmall >> >(dI, dq, dFF, dNatomEl, cfg->Nel, Ntot, cfg->q.N, float(cfg->Rcutoff), float(cfg->p0), cfg->damping);
		cudaThreadSynchronize();
		cudaFree(dNatomEl);
	}
	else {
		float SLav = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) SLav += float(SL[iEl]) * NatomEl[iEl];
		SLav /= Ntot;
		AddCutoffKernelNeutron << <GSadd, BlockSize1Dsmall >> >(dI, dq, SLav, Ntot, cfg->q.N, float(cfg->Rcutoff), float(cfg->p0), cfg->damping);
		cudaThreadSynchronize();
	}
}

//Organazies the computations of the scattering intensity (powder diffraction pattern) using the histogram of interatomic distances with CUDA
void calcInt1DHistCuda(const int DeviceNUM, double ** const I, const unsigned long long int * const rij_hist, const unsigned int *const NatomEl, const config * const cfg, const float * const dFF, const vector<double> SL, const float * const dq, const unsigned int Ntot){
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM);//getting device properties
	const int cc = deviceProp.major * 10 + deviceProp.minor;//device compute capability
	unsigned int BlockSize = BlockSize1Dlarge;//setting the size of the thread blocks to 1024 (default)	
	if (cc < 30) BlockSize = BlockSize1Dmedium;//setting the size of the thread blocks to 512 for the devices with CC < 3.0
	const unsigned int GridSize = MIN(256, cfg->Nhist / BlockSize + BOOL(cfg->Nhist % BlockSize));
	const unsigned int GFLOPS = GetGFLOPS(deviceProp);//theoretical peak GPU performance
	unsigned int MaxBinsPerBlock = cfg->Nhist / GridSize + BOOL(cfg->Nhist % GridSize);
	if (deviceProp.kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 1.5e-5; // t = k * Nq * MaxBinsPerBlock / GFLOPS
		MaxBinsPerBlock = MIN((unsigned int)(tmax * GFLOPS / (k * cfg->q.N)), MaxBinsPerBlock);
	}
	float *dI = NULL;//device array for scattering intensity
	const unsigned int Isize = GridSize * cfg->q.N;//each block writes to it's own copy of scattering intensity array
	cudaMalloc(&dI, Isize * sizeof(float));//allocating the device memory for the scattering intensity array
	const unsigned int GSzero = MIN(65535, Isize / BlockSize + BOOL(Isize % BlockSize));//grid size for zero1DFloatArrayKernel
	for (unsigned int iter = 0; iter < Isize / BlockSize + BOOL(Isize % BlockSize); iter += GSzero) zero1DFloatArrayKernel << <GSzero, BlockSize >> >(dI + iter*BlockSize, Isize - iter*BlockSize);//reseting intensity array
	cudaThreadSynchronize();//synchronizing before calculation starts
	const unsigned int GSadd = cfg->q.N / BlockSize1Dsmall + BOOL(cfg->q.N % BlockSize1Dsmall);//grid size for addIKernelXray/addIKernelNeutron
	unsigned int Nstart = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		if (cfg->source == xray) addIKernelXray << <GSadd, BlockSize1Dsmall >> > (dI, dFF + iEl * cfg->q.N, cfg->q.N, NatomEl[iEl]);//add contribution form diagonal (i==j) elements in Debye sum
		else addIKernelNeutron << <GSadd, BlockSize1Dsmall >> > (dI, cfg->q.N, float(SQR(SL[iEl]) * NatomEl[iEl]));
		cudaThreadSynchronize();//synchronizing before main calculation starts
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){
			for (unsigned int iBin = 0; iBin < cfg->Nhist; iBin += GridSize * MaxBinsPerBlock) {//iterations to avoid killswitch triggering
				/*float time;
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);*/
				if (cfg->source == xray) {//Xray
					if (cc >= 30) calcIntHistKernel <BlockSize1Dlarge> << <GridSize, BlockSize >> > (xray, dI, dFF + iEl * cfg->q.N, dFF + jEl * cfg->q.N, 0, dq, cfg->q.N, rij_hist + Nstart, iBin, cfg->Nhist, MaxBinsPerBlock, float(cfg->hist_bin), float(cfg->Rcutoff), cfg->damping);
					else calcIntHistKernel <BlockSize1Dmedium> << <GridSize, BlockSize >> > (xray, dI, dFF + iEl * cfg->q.N, dFF + jEl * cfg->q.N, 0, dq, cfg->q.N, rij_hist + Nstart, iBin, cfg->Nhist, MaxBinsPerBlock, float(cfg->hist_bin), float(cfg->Rcutoff), cfg->damping);
				}
				else {//neutron
					if (cc >= 30) calcIntHistKernel <BlockSize1Dlarge> << <GridSize, BlockSize >> > (neutron, dI, NULL, NULL, float(SL[iEl] * SL[jEl]), dq, cfg->q.N, rij_hist + Nstart, iBin, cfg->Nhist, MaxBinsPerBlock, float(cfg->hist_bin), float(cfg->Rcutoff), cfg->damping);
					else calcIntHistKernel <BlockSize1Dmedium> << <GridSize, BlockSize >> > (neutron, dI, NULL, NULL, float(SL[iEl] * SL[jEl]), dq, cfg->q.N, rij_hist + Nstart, iBin, cfg->Nhist, MaxBinsPerBlock, float(cfg->hist_bin), float(cfg->Rcutoff), cfg->damping);
				}
				/*cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&time, start, stop);
				cout << "calcIntHistKernel execution time is: " << time << " ms\n" << endl;*/
				cudaThreadSynchronize();//synchronizing before the next iteration step
			}
		}
	}
	sumIKernel << <GSadd, BlockSize1Dsmall >> >(dI, cfg->q.N, GridSize);//summing intensity copies
	cudaThreadSynchronize();//synchronizing threads before multiplying the intensity by a polarization factor
	if (cfg->cutoff) AddCutoffCUDA(GSadd, dI, NatomEl, cfg, dFF, SL, dq, Ntot);
	if (cfg->PolarFactor) PolarFactor1DKernel << <GSadd, BlockSize1Dsmall >> >(dI, cfg->q.N, dq, float(cfg->lambda));
	float * const hI = new float[cfg->q.N];
	cudaMemcpy(hI, dI, cfg->q.N * sizeof(float), cudaMemcpyDeviceToHost);//copying intensity array from the device to the host
	cudaFree(dI);//deallocating memory for intensity array
	*I = new double[cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] = double(hI[iq]) / Ntot;//normalizing
	delete[] hI;
}

//Computes the partial radial distribution function (RDF)
__global__ void calcPartialRDFkernel(float * const dPDF, const unsigned long long int * const rij_hist, const unsigned int Nhist, const float mult) {
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < Nhist) dPDF[i] = rij_hist[i] * mult;
}

//Computes the partial pair distribution function (PDF)
__global__ void calcPartialPDFkernel(float * const dPDF, const unsigned long long int * const rij_hist, const unsigned int iStart, const unsigned int Nhist, const float mult, const float bin) {
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < Nhist) {
		const float r = (iStart + i + 0.5f) * bin;
		dPDF[i] = rij_hist[i] * (mult / SQR(r));
	}
}

//Computes the partial reduced pair distribution function(rPDF)
__global__ void calcPartialRPDFkernel(float * const dPDF, const unsigned long long int * const rij_hist, const unsigned int iStart, const unsigned int Nhist, const float mult, const float submult, const float bin) {
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < Nhist) {
		const float r = (iStart + i + 0.5f) * bin;
		dPDF[i] = rij_hist[i] * (mult / r) - submult * r;
	}
}

//Computes the total PDF using the partial PDFs
__global__ void calcPDFkernel(float * const dPDF, const unsigned int Nstart, const unsigned int Nhist, const float multIJ) {
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < Nhist) 	dPDF[i] += dPDF[Nstart + i] * multIJ;
}

//Depending on the computational scenario organazies the computations of the scattering intensity (powder diffraction pattern) or PDF using the histogram of interatomic distances with CUDA
void calcPDFandDebyeCuda(const int DeviceNUM, double ** const I, double ** const PDF, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const float4 * const ra, const float * const dFF, const vector<double> SL, const float * const dq) {
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	unsigned long long int *rij_hist = NULL;//array for pair-distribution histogram (device only)
	calcHistCuda(DeviceNUM, &rij_hist, ra, NatomEl, NatomEl_outer, cfg);//calculating the histogram
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "Histogram calculation time: " << time / 1000 << " s" << endl;
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl];//calculating the total number of atoms
	if ((cfg->scenario == PDFonly) || (cfg->scenario == DebyePDF)) {//calculating the PDFs
		cudaEventRecord(start, 0);
		const unsigned int BlockSize = BlockSize1Dmedium;
		const unsigned int NPDF = (1 + (cfg->Nel * (cfg->Nel + 1)) / 2) * cfg->Nhist;//total PDF array size (full (cfg->Nhist) + partial (cfg->Nhist*(cfg->Nel*(cfg->Nel + 1)) / 2) )
		float *dPDF = NULL;
		cudaMalloc(&dPDF, NPDF * sizeof(float));//allocating the device memory for PDF array
		//the size of the histogram array may exceed the maximum number of thread blocks in the grid (65535 for the devices with CC < 3.0) multiplied by the thread block size (512 for devices with CC < 2.0 or 1024 for others)
		//so any operations on histogram array should be performed iteratively
		const unsigned int GSzero = MIN(65535, NPDF / BlockSize + BOOL(NPDF % BlockSize));//grid size for zero1DFloatArrayKernel
		for (unsigned int iter = 0; iter < NPDF; iter += GSzero * BlockSize)	zero1DFloatArrayKernel << <NPDF / BlockSize + BOOL(NPDF % BlockSize), BlockSize >> >(dPDF + iter, NPDF - iter);//reseting the PDF array
		cudaThreadSynchronize();//synchronizing before calculation starts
		const unsigned int GridSizeMax = cfg->Nhist / BlockSize + BOOL(cfg->Nhist % BlockSize);
		const unsigned int GridSize = MIN(65535, GridSizeMax);//grid size for main kernels
		unsigned int Nstart = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){				
				switch (cfg->PDFtype){
					float mult, sub;
					case typeRDF://calculating partial RDFs
						mult = 1.f / (float(cfg->hist_bin) * Ntot);
						for (unsigned int iter = 0; iter < cfg->Nhist; iter += GridSize * BlockSize)	calcPartialRDFkernel << <GridSize, BlockSize >> > (dPDF + iter + cfg->Nhist + Nstart, rij_hist + iter + Nstart, cfg->Nhist - iter, mult);
						break;
					case typePDF://calculating partial PDFs
						mult = 0.25f / (PIf * float(cfg->hist_bin * cfg->p0) * Ntot);
						for (unsigned int iter = 0; iter < cfg->Nhist; iter += GridSize * BlockSize) calcPartialPDFkernel << <GridSize, BlockSize >> > (dPDF + iter + cfg->Nhist + Nstart, rij_hist + iter + Nstart, iter, cfg->Nhist - iter, mult, float(cfg->hist_bin));
						break;
					case typeRPDF://calculating partial rPDFs
						mult = 1.f / (float(cfg->hist_bin) * Ntot);
						(jEl > iEl) ? sub = 8.f * PIf * float(cfg->p0) * float(NatomEl[iEl]) * float(NatomEl[jEl]) / SQR(float(Ntot)) : sub=4.f * PIf * float(cfg->p0) * SQR(float(NatomEl[iEl])) / SQR(float(Ntot));
						for (unsigned int iter = 0; iter < cfg->Nhist; iter += GridSize * BlockSize) calcPartialRPDFkernel << <GridSize, BlockSize >> > (dPDF + iter + cfg->Nhist + Nstart, rij_hist + iter + Nstart, iter, cfg->Nhist - iter, mult, sub, float(cfg->hist_bin));
						break;
				}
			}
		}
		cudaThreadSynchronize();//synchronizing before calculating the full PDF
		Nstart = cfg->Nhist;
		float Faverage2 = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Faverage2 += float(SL[iEl] * NatomEl[iEl]); //calculating the average form-factor
		Faverage2 /= Ntot;
		Faverage2 *= Faverage2;//and squaring it
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {//calculating full PDF by summing partial PDFs
			for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){
				const float multIJ = float(SL[iEl] * SL[jEl]) / Faverage2;
				for (unsigned int iter = 0; iter < cfg->Nhist; iter += GridSize * BlockSize) calcPDFkernel << <GridSize, BlockSize >> > (dPDF + iter, Nstart, cfg->Nhist - iter, multIJ);
				cudaThreadSynchronize();//synchronizing before adding next partial PDF to the full PDF
			}
		}
		unsigned int NPDFh = NPDF;
		if (!cfg->PrintPartialPDF) NPDFh = cfg->Nhist;//if the partial PDFs are not needed, we are not copying them to the host
		float * const hPDF = new float[NPDFh];
		cudaMemcpy(hPDF, dPDF, NPDFh * sizeof(float), cudaMemcpyDeviceToHost);//copying the PDF from the device to the host
		*PDF = new double[NPDFh];//resulting array of doubles for PDF
		for (unsigned int i = 0; i < NPDFh; i++) (*PDF)[i] = double(hPDF[i]);//converting into double
		delete[] hPDF;
		if (dPDF != NULL) cudaFree(dPDF);
		cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&time, start, stop);
	    cout << "PDF calculation time: " << time/1000 << " s" << endl;
	}
	if ((cfg->scenario == Debye_hist) || (cfg->scenario == DebyePDF)) {
		cudaEventRecord(start, 0);
		calcInt1DHistCuda(DeviceNUM, I, rij_hist, NatomEl, cfg, dFF, SL, dq, Ntot);//calculating the scattering intensity using the pair-distribution histogram
		cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&time, start, stop);
	    cout << "1D pattern calculation time: " << time / 1000 << " s" << endl;
	}
	if (rij_hist != NULL) cudaFree(rij_hist);//deallocating memory for pair distribution histogram
}

//Computes xray (source == xray) or neutron (source == neutron) scattering intensity (powder diffraction pattern) using the histogram of interatomic distances
template <unsigned int BlockSize2D, bool cutoff> __global__ void calcIntDebyeKernel(const unsigned int source, float * const I, const float * const FFi, const float * const FFj, const float SLij, const float * const q, const unsigned int Nq, const float4 * const ri, const float4 * const rj, const unsigned int iMax, const unsigned int jMax, const bool diag, const float mult, const float Rcut, const bool damping){
	if ((diag) && (blockIdx.x < blockIdx.y)) return; //we need to calculate inter-atomic distances only for j > i, so if we are in the diagonal grid, all the subdiagonal blocks (for which j < i for all threads) do nothing and return
	const unsigned int jt = threadIdx.x, it = threadIdx.y;
	const unsigned int iCopy = blockIdx.y * BlockSize2D + jt; //jt!!! memory transaction are performed by the threads of the same warp to coalesce them
	unsigned int i = blockIdx.y * BlockSize2D + it;
	unsigned int j = blockIdx.x * BlockSize2D + jt;
	__shared__ float4 ris[BlockSize2D], rjs[BlockSize2D]; //cache arrays for the atomic coordinates
	__shared__ float rij[BlockSize2D][BlockSize2D]; //cache array for inter-atomic distances
	__shared__ float damp[BlockSize2D][BlockSize2D]; //cache array for damping coefficients
	rij[it][jt] = -1.f; //reseting inter-atomic distances array
	damp[it][jt] = 1.f;//if damping == false, all damp coefficients are equal to 1
	if (((diag) && (j <= i)) || ((j >= jMax) || (i >= iMax))) damp[it][jt] = 0;//damping coefficients are also used to zero the contribution of subdiagonal elements in the diagonal blocks
	if ((it == 0) && (j < jMax)) rjs[jt] = rj[j]; //copying the atomic coordinates for j-th (column) atoms (only the threads of the first warp (half-warp for CC < 2.0) are used)
	if ((it == 2) && (iCopy < iMax)) ris[jt] = ri[iCopy]; //the same for i-th (row) atoms (only the threads of the third warp (first half-warp of the second warp for CC < 2.0) are used)
	__syncthreads(); //synchronizing threads to ensure that the copying is complete
	//calculating distances
	const float Rcut2 = SQR(Rcut);
	if ((j < jMax) && (i < iMax) && ((j > i) || (!diag))) {
		const float rij2 = SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z);//calculate square of distance	
		if (cutoff){
			if (rij2 < Rcut2) {
				rij[it][jt] = sqrtf(rij2);
				if (damping) {
					const float x = PIf * rij[it][jt] / Rcut;
					damp[it][jt] = __sinf(x) / x;
				}
			}
		}
		else rij[it][jt] = sqrtf(rij2);
	}
	__syncthreads();//synchronizing threads to ensure that the calculation of the distances is complete
	const unsigned int iEnd = MIN(BlockSize2D, iMax - blockIdx.y * BlockSize2D); //last i-th (row) atom index for the current block
	for (unsigned int iterq = 0; iterq < Nq; iterq += SQR(BlockSize2D)) {//if Nq > SQR(BlockSize2D) there will be threads that compute more than one element of the intensity array
		const unsigned int iq = iterq + it * BlockSize2D + jt;
		if (iq < Nq) {//checking for array margin
			float lI = 0;
			const float lq = q[iq];//copying the scattering vector magnitude to the local memory
			for (i = 0; i < iEnd; i++) {
#pragma unroll 8
				for (j = 0; j < BlockSize2D; j++) {
					if (cutoff) {
						if (rij[i][j] > 0) {
							const float qrij = lq * rij[i][j] + 0.000001f;
							lI += damp[i][j] *__sinf(qrij) / qrij;
						}
					}
					else {
						const float qrij = lq * rij[i][j] + 0.000001f;
						lI += damp[i][j] * __sinf(qrij) / qrij;
					}
				}
			}
			if (source == xray) I[Nq * (gridDim.x * blockIdx.y + blockIdx.x) + iq] += mult * lI * FFi[iq] * FFj[iq]; //multiplying the intensity by form-factors and storing the results in the global memory
			else  I[Nq * (gridDim.x * blockIdx.y + blockIdx.x) + iq] += mult * lI * SLij;
		}
	}
}

//Organazies the computations of the scattering intensity(powder diffraction pattern) using the original Debye equation(without the histogram approximation) with CUDA
void calcIntDebyeCuda(const int DeviceNUM, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const float4 * const ra, const float * const dFF, const vector<double> SL, const float * const dq){
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM);//getting the device properties
	size_t free, total;
	cuMemGetInfo(&free, &total);//checking the amount of free GPU memory	
	const unsigned int BlockSize2D = BlockSize2Dsmall;//setting block size to 32x32 (default)
	const unsigned int BlockSize = SQR(BlockSize2D);//total number of threads per block
	const unsigned int GFLOPS = GetGFLOPS(deviceProp); //theoretical peak GPU performance
	unsigned int GridSizeExecMax = MIN(128, (unsigned int)(sqrtf(0.5f * free / (cfg->q.N * sizeof(float)))));//we use two-dimensional grid here, so checking the amount of free memory is really important 
	if (deviceProp.kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 5.e-8; // t = k * GridSizeExecMax^2 * BlockSize2D^2 * cfg->q.N / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / (k * cfg->q.N)) / BlockSize2D), GridSizeExecMax);
	}
	float *dI = NULL; //device array for scattering intensity
	const unsigned int Isize = SQR(GridSizeExecMax) * cfg->q.N;//total size of the intensity array	
	cudaError err=cudaMalloc(&dI, Isize * sizeof(float));//allocating memory for the intensity array and checking for errors
	if (err != cudaSuccess) cout << "Error in calcIntDebyeCuda(), cudaMalloc(dI): " << cudaGetErrorString(err) << endl;
	const unsigned int GSzero = MIN(65535, Isize / BlockSize + BOOL(Isize % BlockSize));//grid size for zero1DFloatArrayKernel
	for (unsigned int iter = 0; iter < Isize / BlockSize + BOOL(Isize % BlockSize); iter += GSzero) zero1DFloatArrayKernel << <GSzero, BlockSize >> >(dI + iter*BlockSize, Isize - iter*BlockSize);//reseting the intensity array
	cudaThreadSynchronize();//synchronizing before calculation starts
	dim3 blockgrid(BlockSize2D, BlockSize2D);
	const unsigned int GSadd = cfg->q.N / BlockSize1Dsmall + BOOL(cfg->q.N % BlockSize1Dsmall);//grid size for addIKernelXray/addIKernelNeutron
	const float4 * * const raEl = new const float4*[cfg->Nel];
	raEl[0] = ra;
	for (unsigned int iEl = 1; iEl < cfg->Nel; iEl++) {
		(cfg->cutoff) ? raEl[iEl] = raEl[iEl - 1] + NatomEl_outer[iEl - 1] : raEl[iEl] = raEl[iEl - 1] + NatomEl[iEl - 1];
	}
	const float Rcut = float(cfg->Rcutoff);
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		if (cfg->source == xray) addIKernelXray << <GSadd, BlockSize1Dsmall >> > (dI, dFF + iEl * cfg->q.N, cfg->q.N, NatomEl[iEl]);//adding contribution from diagonal (i==j) elements in Debye sum
		else addIKernelNeutron << <GSadd, BlockSize1Dsmall >> > (dI, cfg->q.N, float(SQR(SL[iEl]) * NatomEl[iEl]));
		cudaThreadSynchronize();//synchronizing before main calculation starts
		unsigned int jElSt = iEl;
		if (cfg->cutoff) jElSt = 0;
		for (unsigned int jEl = jElSt; jEl < cfg->Nel; jEl++) {
			unsigned int jAtomST = 0;
			if ((cfg->cutoff) && (jEl < iEl)) jAtomST = NatomEl[jEl];
			for (unsigned int iAtom = 0; iAtom < NatomEl[iEl]; iAtom += BlockSize2D * GridSizeExecMax){
				const unsigned int GridSizeExecY = MIN((NatomEl[iEl] - iAtom) / BlockSize2D + BOOL((NatomEl[iEl] - iAtom) % BlockSize2D), GridSizeExecMax);//Y-size of grid on current step
				const unsigned int iMax = MIN(BlockSize2D * GridSizeExecY, NatomEl[iEl] - iAtom);//last i-th (row) atom in current grid
				if (iEl == jEl) jAtomST = iAtom;
				for (unsigned int jAtom = jAtomST; jAtom < NatomEl[jEl]; jAtom += BlockSize2D * GridSizeExecMax){
					const unsigned int GridSizeExecX = MIN((NatomEl[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of grid on current step
					const unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomEl[jEl] - jAtom);//last j-th (column) atom in current grid
					dim3 grid(GridSizeExecX, GridSizeExecY);
					bool diag = false;
					if ((iEl == jEl) && (iAtom == jAtom)) diag = true;//checking if we are on the diagonal grid or not
					/*float time;
					cudaEvent_t start, stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start, 0);*/
					if (cfg->cutoff) {
						if (cfg->source == xray) calcIntDebyeKernel <BlockSize2Dsmall, true> << <grid, blockgrid >> > (xray, dI, dFF + iEl * cfg->q.N, dFF + jEl * cfg->q.N, 0, dq, cfg->q.N, raEl[iEl] + iAtom, raEl[jEl] + jAtom, iMax, jMax, diag, 2., Rcut, cfg->damping);
						else calcIntDebyeKernel <BlockSize2Dsmall, true> << <grid, blockgrid >> > (neutron, dI, NULL, NULL, float(SL[iEl] * SL[jEl]), dq, cfg->q.N, raEl[iEl] + iAtom, raEl[jEl] + jAtom, iMax, jMax, diag, 2., Rcut, cfg->damping);
					}
					else {
						if (cfg->source == xray) calcIntDebyeKernel <BlockSize2Dsmall, false> << <grid, blockgrid >> > (xray, dI, dFF + iEl * cfg->q.N, dFF + jEl * cfg->q.N, 0, dq, cfg->q.N, raEl[iEl] + iAtom, raEl[jEl] + jAtom, iMax, jMax, diag, 2., 0, false);
						else calcIntDebyeKernel <BlockSize2Dsmall, false> << <grid, blockgrid >> > (neutron, dI, NULL, NULL, float(SL[iEl] * SL[jEl]), dq, cfg->q.N, raEl[iEl] + iAtom, raEl[jEl] + jAtom, iMax, jMax, diag, 2., 0, false);
					}
					cudaThreadSynchronize();//synchronizing before launching next kernel (it will write the data to the same array)
					/*cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&time, start, stop);
					cout << "calcIntDebyeKernel execution time is: " << time << " ms\n" << endl;*/
				}
				if (cfg->cutoff) {
					for (unsigned int jAtom = NatomEl[jEl]; jAtom < NatomEl_outer[jEl]; jAtom += BlockSize2D * GridSizeExecMax){
						const unsigned int GridSizeExecX = MIN((NatomEl_outer[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl_outer[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of grid on current step
						const unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomEl_outer[jEl] - jAtom);//last j-th (column) atom in current grid
						dim3 grid(GridSizeExecX, GridSizeExecY);
						if (cfg->source == xray) calcIntDebyeKernel <BlockSize2Dsmall, true> << <grid, blockgrid >> > (xray, dI, dFF + iEl * cfg->q.N, dFF + jEl * cfg->q.N, 0, dq, cfg->q.N, raEl[iEl] + iAtom, raEl[jEl] + jAtom, iMax, jMax, false, 1., Rcut, cfg->damping);
						else calcIntDebyeKernel <BlockSize2Dsmall, true> << <grid, blockgrid >> > (neutron, dI, NULL, NULL, float(SL[iEl] * SL[jEl]), dq, cfg->q.N, raEl[iEl] + iAtom, raEl[jEl] + jAtom, iMax, jMax, false, 1., Rcut, cfg->damping);
					}
				}
			}
		}
	}
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl]; //calculating total number of atoms
	sumIKernel << <GSadd, BlockSize1Dsmall >> >(dI, cfg->q.N, SQR(GridSizeExecMax));//summing intensity copies
	cudaThreadSynchronize();//synchronizing before multiplying intensity by a polarization factor
	if (cfg->cutoff) AddCutoffCUDA(GSadd, dI, NatomEl, cfg, dFF, SL, dq, Ntot);
	if (cfg->PolarFactor) PolarFactor1DKernel << <GSadd, BlockSize1Dsmall >> >(dI, cfg->q.N, dq, float(cfg->lambda));
	float * const hI = new float[cfg->q.N];
	cudaMemcpy(hI, dI, cfg->q.N * sizeof(float), cudaMemcpyDeviceToHost);//copying the resulting scattering intensity from the device to the host
	cudaFree(dI);//deallocating device memory for intensity array
	*I = new double[cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] = double(hI[iq]) / Ntot;//normalizing	
	delete[] hI;
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "1D pattern calculation time: " << time / 1000 << " s" << endl;
}

//Computes the partial scattering intensity (*Ipart) from the partials sums (*I) computed by different thread blocks
__global__ void sumIpartialKernel(float * const I, float * const Ipart, const unsigned int Nq, const unsigned int Nsum){
	const unsigned int iq = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int ipart = blockIdx.y * Nsum * Nq;
	if (iq < Nq) {
		for (unsigned int j = 1; j < Nsum; j++)	I[ipart + iq] += I[ipart + j * Nq + iq];
		Ipart[(blockIdx.y + 1) * Nq + iq] = I[ipart + iq];
	}
}

//Computes the total scattering intensity (powder diffraction pattern) using the partial scattering intensity
__global__ void integrateIpartialKernel(float * const I, const unsigned int Nq, const unsigned int Nparts){
	const unsigned int iq = blockDim.x * blockIdx.x + threadIdx.x;
	if (iq<Nq) {
		I[iq] = 0;
		for (unsigned int ipart = 1; ipart < Nparts + 1; ipart++)	I[iq] += I[ipart * Nq + iq];
	}
}

//Organazies the computations of the scattering intensity (powder diffraction pattern) using the original Debye equation (without the histogram approximation) with CUDA
void calcIntPartialDebyeCuda(const int DeviceNUM, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const float4 * const ra, const float * const dFF, const vector <double> SL, const float * const dq, const block * const Block){
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);	
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, DeviceNUM);
	size_t free, total;
	cuMemGetInfo(&free, &total);
	const unsigned int BlockSize2D = BlockSize2Dsmall;
	const unsigned int BlockSize = SQR(BlockSize2D);
	const unsigned int GFLOPS = GetGFLOPS(deviceProp); //theoretical peak GPU performance	
	const unsigned int Nparts = (cfg->Nblocks * (cfg->Nblocks + 1)) / 2;
	unsigned int GridSizeExecMax = MIN(128, (unsigned int)(sqrtf(0.5f * free / (Nparts * cfg->q.N * sizeof(float)))));
	if (deviceProp.kernelExecTimeoutEnabled)	{
		//killswitch enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 5.e-8; // t = k * GridSizeExecMax^2 * BlockSize2D^2 / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / (k * cfg->q.N)) / BlockSize2D), GridSizeExecMax);
	}
	float *dI = NULL;
	const unsigned int IsizeBlock = SQR(GridSizeExecMax) * cfg->q.N;
	const unsigned int Isize = Nparts * IsizeBlock;//each block writes to it's own copy of scattering intensity
	cudaError err = cudaMalloc(&dI, Isize * sizeof(float));
	if (err != cudaSuccess) cout << "Error in calcIntPartialDebyeCuda(), cudaMalloc(dI): " << cudaGetErrorString(err) << endl;
	const unsigned int GSzero = MIN(65535, Isize / BlockSize + BOOL(Isize % BlockSize));
	for (unsigned int iter = 0; iter < Isize / BlockSize + BOOL(Isize % BlockSize); iter += GSzero) zero1DFloatArrayKernel << <GSzero, BlockSize >> >(dI+iter * BlockSize, Isize - iter * BlockSize);	
	dim3 blockgrid(BlockSize2D, BlockSize2D);
	unsigned int * const NatomElBlock = new unsigned int[cfg->Nel * cfg->Nblocks];
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			NatomElBlock[iEl*cfg->Nblocks + iB] = Block[iB].NatomEl[iEl];
		}
	}	
	const unsigned int GSadd = cfg->q.N / BlockSize1Dsmall + BOOL(cfg->q.N % BlockSize1Dsmall);
	cudaThreadSynchronize();
	unsigned int iAtomST = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iAtomST += NatomEl[iEl], iEl++) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			unsigned int Istart = IsizeBlock * (cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + iB);
			if (cfg->source == xray) addIKernelXray << <GSadd, BlockSize1Dsmall >> > (dI + Istart, dFF + iEl * cfg->q.N, cfg->q.N, NatomElBlock[iEl * cfg->Nblocks + iB]);
			else addIKernelNeutron << <GSadd, BlockSize1Dsmall >> > (dI + Istart, cfg->q.N, float(SQR(SL[iEl]) * NatomElBlock[iEl * cfg->Nblocks + iB]));
		}
		cudaThreadSynchronize();
		unsigned int jAtomST = iAtomST;
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jAtomST += NatomEl[jEl], jEl++) {
			unsigned int iAtomSB = 0;
			for (unsigned int iB = 0; iB < cfg->Nblocks; iAtomSB += NatomElBlock[iEl * cfg->Nblocks + iB], iB++) {
				for (unsigned int iAtom = 0; iAtom < NatomElBlock[iEl * cfg->Nblocks + iB]; iAtom += BlockSize2D*GridSizeExecMax){
					const unsigned int GridSizeExecY = MIN((NatomElBlock[iEl * cfg->Nblocks + iB] - iAtom) / BlockSize2D + BOOL((NatomElBlock[iEl * cfg->Nblocks + iB] - iAtom) % BlockSize2D), GridSizeExecMax);
					const unsigned int iMax = MIN(BlockSize2D * GridSizeExecY, NatomEl[iEl] - iAtom);
					const unsigned int i0 = iAtomST + iAtomSB + iAtom;
					unsigned int jAtomSB = 0;
					for (unsigned int jB = 0; jB < cfg->Nblocks; jAtomSB += NatomElBlock[jEl * cfg->Nblocks + jB], jB++) {
						unsigned int Istart = 0;
						(jB>iB) ? Istart = IsizeBlock * (cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + jB) : Istart = IsizeBlock * (cfg->Nblocks * jB - (jB * (jB + 1)) / 2 + iB);
						for (unsigned int jAtom = 0; jAtom < NatomElBlock[jEl * cfg->Nblocks + jB]; jAtom += BlockSize2D * GridSizeExecMax){
							const unsigned int j0 = jAtomST + jAtomSB + jAtom;
							if (j0 >= i0) {
								const unsigned int GridSizeExecX = MIN((NatomElBlock[jEl * cfg->Nblocks + jB] - jAtom) / BlockSize2D + BOOL((NatomElBlock[jEl * cfg->Nblocks + jB] - jAtom) % BlockSize2D), GridSizeExecMax);
								const unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomElBlock[jEl * cfg->Nblocks + jB] - jAtom);
								dim3 grid(GridSizeExecX, GridSizeExecY);
								bool diag = false;
								if (i0 == j0) diag = true;
								if (cfg->source == xray) calcIntDebyeKernel <BlockSize2Dsmall, false> << <grid, blockgrid >> > (xray, dI + Istart, dFF + iEl * cfg->q.N, dFF + jEl * cfg->q.N, 0, dq, cfg->q.N, ra + i0, ra + j0, iMax, jMax, diag, 2., 0, false);
								else calcIntDebyeKernel <BlockSize2Dsmall, false> << <grid, blockgrid >> > (neutron, dI + Istart, NULL, NULL, float(SL[iEl] * SL[jEl]), dq, cfg->q.N, ra + i0, ra + j0, iMax, jMax, diag, 2., 0, false);
								cudaThreadSynchronize();
							}
						}
					}					
				}
			}			
		}
	}
	delete[] NatomElBlock;
	const unsigned int IpartialSize = (Nparts + 1) * cfg->q.N;
	float *dIpart = NULL;
	cudaMalloc(&dIpart, IpartialSize*sizeof(float));
	dim3 gridAdd(GSadd, Nparts);
	sumIpartialKernel << <gridAdd, BlockSize1Dsmall >> >(dI, dIpart, cfg->q.N, SQR(GridSizeExecMax));
	cudaThreadSynchronize();
	cudaFree(dI);
	integrateIpartialKernel << <GSadd, BlockSize1Dsmall >> > (dIpart, cfg->q.N, Nparts);
	cudaThreadSynchronize();
	dim3 gridPolar(GSadd, Nparts + 1);
	if (cfg->PolarFactor) PolarFactor1DKernel << <gridPolar, BlockSize1Dsmall >> >(dIpart, cfg->q.N, dq, float(cfg->lambda));
	float * const hI = new float[IpartialSize];
	cudaMemcpy(hI, dIpart, IpartialSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dIpart);
	*I = new double[IpartialSize];
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl];
	for (unsigned int i = 0; i < IpartialSize; i++) (*I)[i] = double(hI[i]) / Ntot;	
	delete[] hI;
	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "1D pattern calculation time: " << time / 1000 << " s" << endl;
}

//Queries all CUDA devices. Checks and sets the CUDA device number
//Returns 0 if OK and - 1 if no CUDA devices found
int SetDeviceCuda(int * const DeviceNUM){
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	if (!nDevices) {
		cout << "Error: No CUDA devices found." << endl;
		return -1;
	}
	if (*DeviceNUM > -1){
		if (*DeviceNUM < nDevices){
			cudaSetDevice(*DeviceNUM);
			cudaSetDeviceFlags(cudaDeviceBlockingSync);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, *DeviceNUM);
			cout << "Selected CUDA device:" << endl;
			GetGFLOPS(deviceProp, true);
			return 0;
		}
		cout << "Error: Unable to set CUDA device " << *DeviceNUM << ". The total number of CUDA devices is " << nDevices << ".\n";
		cout << "Will use the fastest CUDA device." << endl;
	}
	cout << "The following CUDA devices are found.\n";
	cudaDeviceProp deviceProp;
	unsigned int GFOLPS=0, MaxGFOLPS=0;
	for (int i = 0; i < nDevices; i++) {
		cudaGetDeviceProperties(&deviceProp,i);
		cout << "Device " << i << ":" << endl;
		GFOLPS = GetGFLOPS(deviceProp, true);
		if (GFOLPS > MaxGFOLPS) {
			MaxGFOLPS = GFOLPS;
			*DeviceNUM = i;
		}
	}
	cout << "Will use CUDA device " << *DeviceNUM << "." << endl;
	cudaSetDevice(*DeviceNUM);
	cudaSetDeviceFlags(cudaDeviceBlockingSync);
	return 0;
}

//Copies the atomic coordinates (ra), scattering vector magnitude (q) and the x-ray atomic form-factors (FF) to the device memory	
void dataCopyCUDA(const double *const q, const config * const cfg, const vector < vect3d <double> > * const ra, float4 ** const dra, float ** const dFF, float ** const dq, const vector <double*> FF){
	//copying the main data to the device memory
	if (cfg->scenario != PDFonly) {
		float * const qfloat = new float[cfg->q.N]; // temporary float array for the scattering vector magnitude
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) qfloat[iq] = (float)q[iq];//converting scattering vector magnitude from double to float
		cudaMalloc(dq, cfg->q.N * sizeof(float));//allocating memory for the scattering vector magnitude array
		cudaMemcpy(*dq, qfloat, cfg->q.N * sizeof(float), cudaMemcpyHostToDevice);//copying scattering vector magnitude array from the host to the device
		delete[] qfloat;//deleting temporary array
		if (cfg->source == xray) {
			cudaMalloc(dFF, cfg->q.N * cfg->Nel * sizeof(float));//allocating device memory for the atomic form-factors
			float * const FFfloat = new float[cfg->q.N * cfg->Nel];
			for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) FFfloat[iEl * cfg->q.N + iq] = float(FF[iEl][iq]);//converting form-factors from double to float				
			}
			cudaMemcpy(*dFF, FFfloat, cfg->Nel * cfg->q.N * sizeof(float), cudaMemcpyHostToDevice);//copying form-factors from the host to the device
			delete[] FFfloat;//deleting temporary array
		}
	}
	unsigned int Nat = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Nat += (unsigned int)ra[iEl].size();
	cudaMalloc(dra, Nat * sizeof(float4));//allocating device memory for the atomic coordinates array
	float4 * const hra = new float4[Nat]; //temporary host array for atomic coordinates
	unsigned int iAtom = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = ra[iEl].begin(); ri != ra[iEl].end(); ri++, iAtom++){
			hra[iAtom] = make_float4((float)ri->x, (float)ri->y, (float)ri->z, 0);//converting atomic coordinates from vect3d <double> to float4
		}
	}	
	cudaMemcpy(*dra, hra, Nat * sizeof(float4), cudaMemcpyHostToDevice);//copying atomic coordinates from the host to the device
	delete[] hra;//deleting temporary array
}

//Deletes the atomic coordinates (ra), scattering vector magnitude (dq) and the x-ray atomic form-factors (dFF) from the device memory
void delDataFromDevice(float4 * const ra, float * const dFF,float * const dq, const unsigned int Nel){
	cudaFree(ra);//deallocating device memory for the atomic coordinates array
	if (dq != NULL) cudaFree(dq);//deallocating memory for the scattering vector magnitude array
	if (dFF != NULL) cudaFree(dFF);//deallocating device memory for the atomic form-factors
	cudaDeviceReset();//NVIDIA Profiler works improperly without this
}
#endif
