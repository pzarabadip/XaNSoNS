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

//Contains OpenCL kernels used to compute the powder diffraction patterns and PDFs with the hostogram approximation

//Some macros
#define neutron 0
#define xray 1
#define SQR(x) ((x)*(x))
#define BlockSize1Dsmall 256
#define BlockSize2D 16
#define PIf 3.14159265f
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define BOOL(x) ((x) ? 1 : 0)

//use built-in 64-bit atomic add if the GPU supports it, define custom 64-bit atomic add if not
#ifndef CustomInt64atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#else
void atomAdd64 (__global unsigned int * const counter, const unsigned int add) {
    const unsigned int old = atomic_add(&counter[0], add);
	if (old > 0xFFFFFFFF - add) atomic_inc(&counter[1]);
}
#endif

/**
	Computes polarization factor and multiplies scattering intensity by this factor

	@param *I     Scattering intensity array
	@param Nq     Size of the scattering intensity array
	@param *q     Scattering vector magnitude array
	@param lambda Wavelength of the source
*/
__kernel void PolarFactor1DKernel(__global float * const I, const unsigned int Nq, const __global float * const q, const float lambda){
	const unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq < Nq)	{
		const float sintheta = q[iq] * (lambda * 0.25f / PIf);
		const float cos2theta = 1.f - 2.f * SQR(sintheta);
		const float factor = 0.5f * (1.f + SQR(cos2theta));
		I[iq] *= factor;
	}
}

/**
	Resets 1D float array of size N

	@param *A  Array
	@param N   Size of the array
*/
__kernel void zero1DFloatArrayKernel(__global float * const A, const unsigned int N){
	const unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i<N) A[i] = 0;
}

/**
	Adds the diagonal elements (j==i) of the Debye double sum to the x-ray scattering intensity

	@param *I    Scattering intensity array
	@param Nq    Resolution of the total scattering intensity (powder diffraction pattern)
	@param *FF   X-ray atomic form-factors array
	@param iEl   Chemical element index (for one kernel call the computations are done only for the atoms of the same chemical element)	
	@param N     Total number of atoms of the chemical element for whcich the computations are done
*/
__kernel void addIKernelXray(__global float * const I, const unsigned int Nq, const __global float * const FF, const unsigned int iEl, const unsigned int N) {
	const unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq < Nq)	{
		const float lFF = FF[iEl * Nq + iq];
		I[iq] += SQR(lFF) * N;
	}
}

/**
	Adds the diagonal elements (j==i) of the Debye double sum to the neutron scattering intensity

	@param *I    Scattering intensity array
	@param Nq    Resolution of the total scattering intensity (powder diffraction pattern)
	@param Add   The value to add to the intensity (the result of multiplying the square of the scattering length
	to the total number of atoms of the chemical element for whcich the computations are done)
*/
__kernel void addIKernelNeutron(__global float * const I, const unsigned int Nq, const float Add) {
	const unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq < Nq)	I[iq] += Add;
}

/**
	Computes the total scattering intensity (first Nq elements) from the partials sums computed by different work-groups

	@param *I    Scattering intensity array
	@param Nq    Resolution of the total scattering intensity (powder diffraction pattern)
	@param Nsum  Number of parts to sum (equalt to the total number of work-groups)
*/
__kernel void sumIKernel(__global float * const I, const unsigned int Nq, const unsigned int Nsum){
	const unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq<Nq) {
		float lIsum = 0;
		for (unsigned int j = 1; j < Nsum; j++)	lIsum += I[j * Nq + iq];
		I[iq] += lIsum;
	}
}

/**
	Resets the histogram array (ulong)

	@param *rij_hist  Histogram of interatomic distances
	@param N          Size of the array
*/
__kernel void zeroHistKernel(__global ulong * const rij_hist, const unsigned int N){
	const unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i<N) rij_hist[i] = 0;
}

/**
	Computes the partial radial distribution function (RDF)

	@param *dPDF     Partial PDF array (contains all partial PDFs)
	@param *rij_hist Histogram of interatomic distances (device)
	@param iSt       Index of the first element of the current (for this kernel call) partial histogram/PDF
	@param Nhist     Size of the partial histogram of interatomic distances
	@param mult      1 / (Ntot * bin_width)
*/
__kernel void calcPartialRDFkernel(__global float * const dPDF, const __global ulong * const rij_hist, const unsigned int iSt, const unsigned int Nhist, const float mult) {
	const unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i < Nhist) dPDF[Nhist + iSt + i] = rij_hist[iSt + i] * mult;
}

/**
	Computes the partial pair distribution function (PDF)

	@param *dPDF     Prtial PDF array (contains all partial PDFs)
	@param *rij_hist Histogram of interatomic distances (device)
	@param iSt       Index of the first element of the current (for this kernel call) partial histogram/PDF
	@param Nhist     Size of the partial histogram of interatomic distances
	@param mult      1 / (4 * PI * rho * Ntot * bin_width)
	@param bin       Width of the histogram bin
*/
__kernel void calcPartialPDFkernel(__global float * const dPDF, const __global ulong * const rij_hist, const unsigned int iSt, const unsigned int Nhist, const float mult, const float bin) {
	const unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i < Nhist) {
		const float r = (i + 0.5f) * bin;
		dPDF[Nhist + iSt + i] = rij_hist[iSt + i] * (mult / SQR(r));
	}
}

/**
	Computes the partial reduced pair distribution function (rPDF)

	@param *dPDF     Partial PDF array (contains all partial PDFs)
	@param *rij_hist Histogram of interatomic distances (device)
	@param iSt       Index of the first element of the current (for this kernel call) partial histogram/PDF
	@param Nhist     Size of the partial histogram of interatomic distances
	@param mult      1 / (Ntot * bin_width)
	@param submult   4 * PI * rho * NatomEl_i * NatomEl_j / SQR(Ntot)
	@param bin       Width of the histogram bin
*/
__kernel void calcPartialRPDFkernel(__global float * const dPDF, const __global ulong * const rij_hist, const unsigned int iSt, const unsigned int Nhist, const float mult, const float submult, const float bin) {
	const unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i < Nhist) {
		const float r = (i + 0.5f) * bin;
		dPDF[Nhist + iSt + i] = rij_hist[iSt + i] * (mult / r) - submult * r;
	}
}

/**
	Computes the total PDF using the partial PDFs

	@param *dPDF   Total (first Nhist elements) + partial PDF array. The memory is allocated inside the function.
	@param iSt     Index of the first element of the partial PDF whcih will be added to the total PDF in this kernel call
	@param Nhist   Size of the partial histogram of interatomic distances
	@param multIJ  FF_i(q0) * FF_j(q0) / <FF> (for x-ray) and SL_i * SL_j / <SL> (for neutron)
*/
__kernel void calcPDFkernel(__global float * const dPDF, const unsigned int iSt, const unsigned int Nhist, const float multIJ) {
	const unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i < Nhist) 	dPDF[i] += dPDF[iSt + i] * multIJ;
}

/**
	Computes the histogram of interatomic distances

	@param *ra        Atomic coordinate array 
	@param i0         Index of the 1st i-th atom in ra array for this kernel call
	@param j0         Index of the 1st j-th atom in ra array for this kernel call
	@param iMax       Total number of i-th atoms for this kernel call
	@param jMax       Total number of j-th atoms for this kernel call
	@param *rij_hist  Histogram of interatomic distances
	@param Nstart     Index of the first element of the partial histogram corresponding to the i-th and j-th atoms for this kernel call
	@param bin        Width of the histogram bin
	@param diag       True if the j-th atoms and the i-th atoms are the same (diagonal) for this kernel call
*/
__kernel void calcHistKernel(const __global float4 * const restrict ra, const unsigned int i0, const unsigned int j0, const unsigned int iMax, const unsigned int jMax, __global ulong * const rij_hist, const unsigned int Nstart, \
			const float bin, const unsigned int diag, const unsigned int cutoff, const float Rcut2, const ulong add){
	//we need to calculate inter-atomic distances only for j > i, so if we are in the diagonal grid, all the subdiagonal work-groups (for which j < i for all work-items) do nothing and return
	if ((diag) && (get_group_id(0) < get_group_id(1))) return; 
	const unsigned int jt = get_local_id(0), it = get_local_id(1);
	const unsigned int j = get_group_id(0) * BlockSize2D + jt;
	const unsigned int iCopy = get_group_id(1) * BlockSize2D + jt; //jt!!! memory transaction are performed by the work-item of the same wavefront to coalesce them
	const unsigned int i = get_group_id(1) * BlockSize2D + it;
	__local float4 ris[BlockSize2D], rjs[BlockSize2D];
	if ((it == 0) && (j < jMax)) { //copying atomic coordinates for j-th (column) atoms
		rjs[jt] = ra[j0 + j];
	}
	if ((it == 4) && (iCopy < iMax)) { //the same for i-th (row) atoms
		ris[jt] = ra[i0 + iCopy];
	}
	barrier(CLK_LOCAL_MEM_FENCE);//sync to ensure that copying is complete
	if ((j < jMax) && (i < iMax) && ((j > i) || (!diag))) {
		const float rij2 = SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z); //calculate square of distance
		if (cutoff){
			if (rij2 < Rcut2){
				const unsigned int index = (unsigned int)(sqrt(rij2) / bin); //get the index of histogram bin
#ifndef CustomInt64atomics
				atom_add(&rij_hist[Nstart + index], add); //add +1 to histogram bin
#else
                atomAdd64(&rij_hist[Nstart + index], (uint)add);
#endif
			}
		}
		else {
			const unsigned int index = (unsigned int)(sqrt(rij2) / bin); //get the index of histogram bin
#ifndef CustomInt64atomics
			atom_add(&rij_hist[Nstart + index], add); //add +1 to histogram bin
#else
            atomAdd64(&rij_hist[Nstart + index], (uint)add);
#endif
		}
	}
}

/**
Adds the average density correction to the xray scattering intensity when the cut-off is enabled (cfg.cutoff == true)

@param *I        Scattering intensity array
@param *q        Scattering vector magnitude array
@param Nq        Size of the scattering intensity array
@param dFF       X-ray atomic form-factor array for all chemical elements
@param *NatomEl  Array containing the total number of atoms of each chemical element
@param Nel       Total number of different chemical elements in the nanoparticle
@param Ntot      Total number of atoms in the nanoparticle
@param Rcut      Cutoff radius in A (if cutoff is true)
@param dens      Average atomic density of the nanoparticle
@param damping   cfg->damping
*/
__kernel void AddCutoffKernelXray(__global float * const I, const __global float * const q, const __global float * const FF, const __global unsigned int * const NatomEl, const unsigned int Nel, const unsigned int Ntot, const unsigned int Nq, const float Rcut, \
	const float dens, const unsigned int damping){
	const unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq < Nq) {
		float FFaver = 0;
		for (unsigned int iEl = 0; iEl < Nel; iEl++) FFaver += FF[iEl * Nq + iq] * NatomEl[iEl];
		FFaver /= Ntot;
		const float lq = q[iq];
		if (lq > 0.000001f) {
			const float qrcut = lq * Rcut;
			if (damping) I[iq] += 4.f * PIf * Ntot * dens * SQR(FFaver) * SQR(Rcut) * native_sin(qrcut) / (lq * (SQR(qrcut) - SQR(PIf)));
			else I[iq] += 4.f * PIf * Ntot * dens * SQR(FFaver) * (Rcut * native_cos(qrcut) - native_sin(qrcut) / lq) / SQR(lq);
		}
	}
}

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
__kernel void AddCutoffKernelNeutron(__global float * const I, const __global float * const q, const float SLaver, const unsigned int Ntot, const unsigned int Nq, const float Rcut, const float dens, const unsigned int damping){
	const unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq < Nq) {
		const float lq = q[iq];
		if (lq > 0.000001f) {
			const float qrcut = lq * Rcut;
			if (damping) I[iq] += 4.f * PIf * Ntot * dens * SQR(SLaver) * SQR(Rcut) * native_sin(qrcut) / (lq * (SQR(qrcut) - SQR(PIf)));
			else I[iq] += 4.f * PIf * Ntot * dens * SQR(SLaver) * (Rcut * native_cos(qrcut) - native_sin(qrcut) / lq) / SQR(lq);
		}
	}
}

/**
	Computes the x-ray scattering intensity (powder diffraction pattern) using the histogram of interatomic distances

	@param source          xray or neutron
	@param *I              Scattering intensity array
	@param *FF             X-ray atomic form-factors array 
	@param iEl             Chemical element index of i-ths atoms (all the i-ths atoms are of the same chemical element for one kernel call)
	@param iEl             Chemical element index of j-ths atoms (all the j-ths atoms are of the same chemical element for one kernel call)
	@param SLij            Product of the scattering lenghts of i-th j-th atoms
	@param *q              Scattering vector magnitude array
	@param Nq              Size of the scattering intensity array
	@param **rij_hist      Histogram of interatomic distances (device). The memory is allocated inside the function
	@param Nstart             Index of the first element of the partial histogram corresponding to the i-th and j-th atoms for this kernel call
	@param iBinSt          Starting index of the partial histogram bin for this kernel call (the kernel is called iteratively in a loop)
	@param Nhist           Size of the partial histogram of interatomic distances
	@param MaxBinsPerBlock Maximum number of histogram bins used by a single work-group
	@param bin             Width of the histogram bin
	@param Rcut            Cutoff radius in A (if cutoff is true)
	@param damping         cfg->damping
*/
__kernel void calcIntHistKernel(const unsigned int source, __global float * const I, const __global float * const FF, const unsigned int iEl, const unsigned int jEl, const float SLij, const  __global float * const q, \
			const unsigned int Nq, const __global ulong * const rij_hist, const unsigned int Nstart, const unsigned int iBinSt, const unsigned int Nhist, const unsigned int MaxBinsPerBlock, const float bin, const float Rcut, const unsigned int damping){
	__local ulong Nrij[BlockSize1Dsmall];//cache array for the histogram
	__local float damp[BlockSize1Dsmall];
	unsigned int it = get_local_id(0);
	unsigned int ig = get_group_id(0);
	Nrij[it] = 0;
	damp[it] = 1.;
	barrier(CLK_LOCAL_MEM_FENCE);
	const unsigned int iBegin = iBinSt + ig * MaxBinsPerBlock;//first index for histogram bin to process
	const unsigned int iEnd = MIN(Nhist, iBegin + MaxBinsPerBlock);//last index for histogram bin to process
	if (iEnd < iBegin) return;
	const unsigned int Niter = (iEnd - iBegin) / BlockSize1Dsmall + BOOL((iEnd - iBegin) % BlockSize1Dsmall);//number of iterations
	for (unsigned int iter = 0; iter < Niter; iter++){//we don't have enough shared memory to load the histogram array as a whole, so we do it with iterations
		const unsigned int NiterFin = MIN(iEnd - iBegin - iter * BlockSize1Dsmall, BlockSize1Dsmall);//maximum number of histogram bins on current iteration step
		if (it < NiterFin) {
			const unsigned int index = iBegin + iter * BlockSize1Dsmall + it;
			Nrij[it] = rij_hist[Nstart + index]; //loading the histogram array to shared memory
			if (damping) {
				const float rij = ((float)(index)+0.5f) * bin;//distance that corresponds to the current histogram bin
				const float x = PIf * rij / Rcut;
				damp[it] = native_sin(x) / x;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);//synchronizing after loading
		for (unsigned int iterq = 0; iterq < (Nq / BlockSize1Dsmall) + BOOL(Nq % BlockSize1Dsmall); iterq++) {//if Nq > blockDim.x there will be threads that compute more than one element of the intensity array
			const unsigned int iq = iterq * BlockSize1Dsmall + it;//index of the intensity array element
			if (iq < Nq) {//checking for the array margin				
				const float lq = q[iq];//copying the scattering vector magnitude to the local memory
				float lI = 0;
				for (unsigned int i = 0; i < NiterFin; i++) {//looping over the histogram bins
					if (Nrij[i]){
						const float qrij = lq * ((float)(iBegin + iter * BlockSize1Dsmall + i) + 0.5f) * bin + 0.000001f;//distance that corresponds to the current histogram bin
						lI += Nrij[i] * damp[i] * native_sin(qrij) / qrij;//scattering intensity without form factors
					}
				}
				//if ((iterq==0)&&(ig == 93) && ((lI>0.0001f) || (lI<-0.0001f))) printf("%u %u %.2e\n", ig, it, lI);
				if (source == xray) I[ig * Nq + iq] += lI * FF[iEl * Nq + iq] * FF[jEl * Nq + iq];//multiplying intensity by form-factors and storing the results in global memory
				else I[ig * Nq + iq] += lI * SLij;
				//I[ig * Nq + iq] += lI;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);//synchronizing threads before the next iteration step
	}
}
