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

//Contains OpenCL kernels used to compute the powder diffraction patterns using the original Debye equation (without the histogram approximation)

//some macros
#define neutron 0
#define xray 1
#define SQR(x) ((x)*(x))
#define BlockSize2D 16
#define PIf 3.14159265f
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define BOOL(x) ((x) ? 1 : 0)

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
		I[get_group_id(1) * Nq + iq] *= factor;
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

	@param *I    Scattering intensity array
	@param *FFi  X-ray atomic form factor for the i-th atoms (all the i-th atoms are of the same chemical element for one kernel call)
	@param *FFj  X-ray atomic form factor for the j-th atoms (all the j-th atoms are of the same chemical element for one kernel call)
	@param *q    Scattering vector magnitude array
	@param Nq    Size of the scattering intensity array
	@param *ra   Atomic coordinate array
	@param i0    Index of the 1st i-th atom in ra array for this kernel call
	@param j0    Index of the 1st j-th atom in ra array for this kernel call
	@param iMax  Total number of i-th atoms for this kernel call
	@param jMax  Total number of j-th atoms for this kernel call
	@param diag  True if the j-th atoms and the i-th atoms are the same (diagonal) for this kernel call
*/
__kernel void calcIntDebyeKernel(const unsigned int source, __global float * const I, const __global float * const FF, const unsigned int iEl, const unsigned int jEl, const float SLij, const __global float * const q, const unsigned int Nq, \
	          const __global float4 * const ra, const unsigned int i0, const unsigned int j0, const unsigned int iMax, const unsigned int jMax, const unsigned int diag, const float mult, const unsigned int cutoff, const float Rcut, const unsigned int damping){
	if ((diag) && (get_group_id(0) < get_group_id(1))) return; //we need to calculate inter-atomic distances only for j > i, so if we are in the diagonal grid, all the subdiagonal work-groups (for which j < i for all work-items) do nothing and return
	const unsigned int jt = get_local_id(0), it = get_local_id(1);
	const unsigned int iCopy = get_group_id(1) * BlockSize2D + jt; //jt!!! memory transaction are performed by the work-items of the same wavefront/warp to coalesce them
	unsigned int j = get_group_id(0) * BlockSize2D + jt;	
	unsigned int i = get_group_id(1) * BlockSize2D + it;
	__local float4 ris[BlockSize2D], rjs[BlockSize2D];
	__local float rij[BlockSize2D][BlockSize2D]; //cache array for inter-atomic distances
	__local float damp[BlockSize2D][BlockSize2D]; //cache array for damping coefficients
	rij[it][jt] = -1.f;
	damp[it][jt] = 1.f; // if damping == false, all damp coefficients are equal to 1
	if (((diag) && (j <= i)) || ((j >= jMax) || (i >= iMax))) damp[it][jt] = 0;//damping coefficients are also used to zero the contribution of subdiagonal elements in the diagonal blocks
	if ((it == 0) && (j < jMax)) rjs[jt] = ra[j0 + j]; //copying atomic coordinates for j-th (column) atoms
	if ((it == 4) && (iCopy < iMax)) ris[jt] = ra[i0 + iCopy];//the same for i-th (row) atoms
	barrier(CLK_LOCAL_MEM_FENCE);//sync to ensure that copying is complete
	const float Rcut2 = SQR(Rcut);
	if ((j < jMax) && (i < iMax) && ((j > i) || (!diag))) {
		const float rij2 = SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z);//calculate square of distance	
		if (cutoff){
			if (rij2 < Rcut2) {
				rij[it][jt] = sqrt(rij2);
				if (damping) {
					const float x = PIf * rij[it][jt] / Rcut;
					damp[it][jt] = native_sin(x) / x;
				}
			}
		}
		else rij[it][jt] = sqrt(rij2);
	}
	barrier(CLK_LOCAL_MEM_FENCE);//synchronizing work-items to ensure that the calculation of the distances is complete
	const unsigned int iEnd = MIN(BlockSize2D, iMax - get_group_id(1) * BlockSize2D); //last i-th (row) atom index for the current work-group
	const unsigned int jEnd = MIN(BlockSize2D, jMax - get_group_id(0) * BlockSize2D); //last j-th (column) atom index for the current work-group
	for (unsigned int iterq = 0; iterq < Nq; iterq += SQR(BlockSize2D)) {//if Nq > SQR(BlockSize2D) there will be work-items that compute more than one element of the intensity array
		const unsigned int iq = iterq + it*BlockSize2D + jt;
		if (iq < Nq) {//checking for array margin
			float lI = 0;
			const float lq = q[iq];//copying the scattering vector magnitude to the private memory
			if (cutoff) {
				for (i = 0; i < iEnd; i++) {
					for (j = 0; j < jEnd; j += 8) {//unrolling to speed up
						if (rij[i][j] > 0) {
							const float qrij = lq * rij[i][j] + 0.000001f;
							lI += damp[i][j] *native_sin(qrij) / qrij;
						}
						if (rij[i][j + 1] > 0) {
							const float qrij = lq * rij[i][j + 1] + 0.000001f;
							lI += damp[i][j + 1] * native_sin(qrij) / qrij;
						}
						if (rij[i][j + 2] > 0) {
							const float qrij = lq * rij[i][j + 2] + 0.000001f;
							lI += damp[i][j + 2] * native_sin(qrij) / qrij;
						}
						if (rij[i][j + 3] > 0) {
							const float qrij = lq * rij[i][j + 3] + 0.000001f;
							lI += damp[i][j + 3] * native_sin(qrij) / qrij;
						}
						if (rij[i][j + 4] > 0) {
							const float qrij = lq * rij[i][j + 4] + 0.000001f;
							lI += damp[i][j + 4] * native_sin(qrij) / qrij;
						}
						if (rij[i][j + 5] > 0) {
							const float qrij = lq * rij[i][j + 5] + 0.000001f;
							lI += damp[i][j + 5] * native_sin(qrij) / qrij;
						}
						if (rij[i][j + 6] > 0) {
							const float qrij = lq * rij[i][j + 6] + 0.000001f;
							lI += damp[i][j + 6] * native_sin(qrij) / qrij;
						}
						if (rij[i][j + 7] > 0) {
							const float qrij = lq * rij[i][j + 7] + 0.000001f;
							lI += damp[i][j + 7] * native_sin(qrij) / qrij;
						}
					}
				}
			}
			else {
				for (i = 0; i < iEnd; i++) {
					for (j = 0; j < jEnd; j += 8) {//unrolling to speed up
						float qrij = lq * rij[i][j] + 0.000001f;
						lI += damp[i][j] * native_sin(qrij) / qrij;
						qrij = lq * rij[i][j + 1] + 0.000001f;
						lI += damp[i][j + 1] * native_sin(qrij) / qrij;
						 qrij = lq * rij[i][j + 2] + 0.000001f;
						lI += damp[i][j + 2] * native_sin(qrij) / qrij;
						qrij = lq * rij[i][j + 3] + 0.000001f;
						lI += damp[i][j + 3] * native_sin(qrij) / qrij;
						qrij = lq * rij[i][j + 4] + 0.000001f;
						lI += damp[i][j + 4] * native_sin(qrij) / qrij;
						qrij = lq * rij[i][j + 5] + 0.000001f;
						lI += damp[i][j + 5] * native_sin(qrij) / qrij;
						qrij = lq * rij[i][j + 6] + 0.000001f;
						lI += damp[i][j + 6] * native_sin(qrij) / qrij;
						qrij = lq * rij[i][j + 7] + 0.000001f;
						lI += damp[i][j + 7] * native_sin(qrij) / qrij;
					}
				}
			}			
			if (source == xray) I[Nq * (get_num_groups(0) * get_group_id(1) + get_group_id(0)) + iq] += mult * lI *FF[iEl * Nq + iq] * FF[jEl * Nq + iq]; //multiplying the intensity by form-factors and storing the results in the global memory
			else I[Nq * (get_num_groups(0) * get_group_id(1) + get_group_id(0)) + iq] += mult * lI * SLij;
		}
	}
}

/**
	Computes the partial scattering intensity (*Ipart) from the partials sums (*I) computed by different thread blocks

	@param *I     Scattering intensity array (partials sums as computed by thread blocks)
	@param *Ipart Array with partial scattering intensities
	@param ipart  Index of the current partial scattering intensity for this kernel call (the kernel is called iteratively in the loop)
	@param Nq     Resolution of the total scattering intensity (powder diffraction pattern)
	@param Nsum   Number of parts to sum (equalt to the total number of thread blocks in the grid)
*/
__kernel void sumIpartialKernel(const __global float * const I, __global float * const Ipart, const unsigned int ipart, const unsigned int Nq, const unsigned int Nsum){
    const unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
    if (iq<Nq) {
        float lIsum=0;
        for (unsigned int j = 0; j < Nsum; j++)	lIsum += I[j * Nq + iq];
        Ipart[(ipart + 1) * Nq + iq] = lIsum;
        Ipart[iq] = 0;
    }
}
