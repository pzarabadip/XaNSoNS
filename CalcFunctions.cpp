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

#include "config.h"
#include "block.h"
#include <random>
#include <chrono>
#ifdef UseOMP
#include <omp.h>
#endif
#ifdef UseMPI
#include "mpi.h"
#endif

extern int myid, numprocs;

int printAtoms(const vector < vect3d<double> > * const ra, const unsigned int * const NatomEl, const string name, const map <unsigned int, string> EN, const unsigned int Ntot);//IO.cpp

/**
Calculates rotational matrix using the Euler angles in the case of intristic rotation in the user-defined rotational convention.
See http://en.wikipedia.org/wiki/Euler_angles#Relationship_to_other_representations for details.

@param *RM0        1st row of the rotational matrix
@param *RM1        2nd row of the rotational matrix
@param *RM2        3rd row of the rotational matrix
@param euler       Euler angles
@param convention  Rotational convention
*/
void calcRotMatrix(vect3d <double> * const RM0, vect3d <double> * const RM1, vect3d <double> * const RM2, const vect3d <double> euler, const unsigned int convention);

/**
Returns the geometric center of the atomic ensemble (nanoparticle)

@param *r    Atomic coordinates
@param Nel   Total number of different chemical elements in the nanoparticle
@param Ntot  Total number of atoms in the nanoparticle
*/
vect3d<double> GetCenter(const vector < vect3d<double> > * const r, const unsigned int Nel, const unsigned int Ntot);

/**
Returns the largest possible interatomic distance in the atomic ensemble (nanoperticle)

@param *r      Atomic coordinates
@param rCenter Geometric mean of the atomic ensemble
@param Nel     Total number of different chemical elements in the nanoparticle
@param Ntot    Total number of atoms in the nanoparticle
*/
double GetEnsembleSize(const vector < vect3d<double> > * const r, const vect3d <double> rCenter, const unsigned int Nel);

/**
Returns the average atomic density of the atomic ensemble (nanoperticle)

@param *r      Atomic coordinates
@param rCenter Geometric mean of the atomic ensemble
@param Rcutoff Cut-off radius. The average atomic density is calculated for the sphere with this radius placed in the rCenter
@param Nel     Total number of different chemical elements in the nanoparticle
*/
double GetAtomicDensity(const vector < vect3d<double> > * const r, const vect3d <double> rCenter, const double Rcutoff, const unsigned int Nel);

/**
Returns the average atomic density of the atomic ensemble if cfg->BoxEdge parameters are defined
*/
double GetAtomicDensityBox(const unsigned int N, const vect3d <double> *BoxEdge);

/**
Returns the square of the distance between the atom 'rAtom' and its most distant neighboring atom from 'rAtomNeighb'.
Called only if the local rearrangement of atoms is enabled.

@param rAtom        Coordinates of atom
@param rAtomNeighb	Coordinates of rearranged atoms
*/
double Rmax2(const vect3d <double> rAtom, const vector< vect3d <double> > rAtomNeighb);

/**
Moves the atoms located inside the inner cut-off box in the beginning of r[iEl] vector.Removes from r[iEl] the atoms locted outside the outer cut-off box
@param *r              Atomic coordinates
@param *cfg            Parameters of simulation
@param **NatomEl       Array containing the total number of atoms of each chemical element (allocated in this fuction)
@param **NatomEl_outer Array containing the number of atoms of each chemical element including the atoms in the outer sphere (allocated in this fuction, only if cfg.cutoff is True)
*/
unsigned int cutoffBox(vector < vect3d <double> > * const r, const config * const cfg, unsigned int * const NatomEl, unsigned int * const NatomEl_outer);

/**
Moves the atoms located inside the inner cut-off sphere in the beginning of r[iEl] vector. Removes from r[iEl] the atoms locted outside the outer cut-off sphere

@param *r              Atomic coordinates
@param *cfg            Parameters of simulation
@param *NatomEl        Array containing the total number of atoms of each chemical element
@param *NatomEl_outer  Array containing the number of atoms of each chemical element including the atoms in the outer sphere (only if cfg.cutoff is True)
@param rCenter         Geometric mean of the atomic ensemble
*/
unsigned int cutoffSphere(vector < vect3d <double> > * const r, const config * const cfg, unsigned int * const NatomEl, unsigned int * const NatomEl_outer, const vect3d <double> rCenter);

/**
Calculates the total atomic ensemble and prints it in the .xyz file if cfg.PrintAtoms is true. 
Returns the total number of atoms in the atomic ensemble (nanoparticle).

@param *cfg            Parameters of simulation
@param *Block          Array of the structural blocks
@param **ra            Atomic coordinates (allocated in this fuction)
@param **NatomEl       Array containing the total number of atoms of each chemical element (allocated in this fuction)
@param **NatomEl_outer Array containing the number of atoms of each chemical element including the atoms in the outer sphere (allocated in this fuction, only if cfg.cutoff is True)
@param ID              Array of [chemical element name, index] pairs. The pairs are created in this function according to the file
*/
unsigned int CalcAndPrintAtoms(config * const cfg, block * const Block, vector < vect3d <double> > ** const ra, unsigned int ** const NatomEl, unsigned int ** const NatomEl_outer, const map <string, unsigned int> ID, const map <unsigned int, string> EN);


#if !defined(UseOCL) && !defined(UseCUDA)

/**
Computes the polarization factor and multiplies the 2D scattering intensity by this factor

@param **I2D   Intensity array
@param *q      Scattering vector magnitude array
@param *cfg    Parameters of simulation
*/
void PolarFactor2D(double * const * const I2D, const double * const q, const config * const cfg);

/**
Computes the polarization factor and multiplies scattering intensity by this factor

@param *I      Scattering intensity array
@param *q      Scattering vector magnitude array
@param *cfg    Parameters of simulation
*/
void PolarFactor1D(double * const I, const double * const q, const config * const cfg);

/**
Computes the scattering intensity (powder diffraction pattern) using the original Debye equation (without the histogram approximation)

@param **I            Scattering intensity array. The memory is allocated inside the function.
@param *cfg           Parameters of simulation
@param *NatomEl       Array containing the total number of atoms of each chemical element
@param *NatomEl_outer Array containing the number of atoms of each chemical element including the atoms in the outer sphere (only if cfg.cutoff is True)
@param *ra            Atomic coordinates
@param FF             X-ray atomic form-factor arrays for all chemical elements
@param SL             Array of neutron scattering lengths for all chemical elements
@param *q             Scattering vector magnitude array
@param Ntot           Total number of atoms in the nanoparticle
@param NumOMPthreads  Number of OpenMP threads
*/
void calcIntDebye(double ** const I, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const vector < vect3d <double> > * const ra, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads);

/**
Computes the contribution to the scattering intensity from the atoms of the chemical elements with indeces iEl and jEl using the Debye equation (without the histogram approximation)

@param *I             Scattering intensity array.
@param *cfg           Parameters of simulation
@param NatomEli       The total number of atoms of the chemical element iEl
@param NatomElj       The total number of atoms of the chemical element jEl
@param *ra            Atomic coordinates
@param iEl            i-th chemical element index
@param jEl            j-th chemical element index
@param *q             Scattering vector magnitude array
@param NumOMPthreads  Number of OpenMP threads
*/
void DebyeCore(double * const I, const config * cfg, const unsigned int NatomEli, const unsigned int NatomElj, const vector < vect3d <double> > * const ra, const unsigned int iEl, const unsigned int jEl, const double * const q, const int NumOMPthreads);

/**
Computes the contribution to the scattering intensity from the atoms of the chemical elements with indeces iEl and jEl using the Debye equation (without the histogram approximation) when the cut-off is enabled (cfg.cutoff == true)

@param *I             Scattering intensity array.
@param *cfg           Parameters of simulation
@param NatomEli       The total number of atoms of the chemical element iEl
@param NatomElj       The total number of atoms of the chemical element jEl
@param NatomElj_out   The total number of atoms of the chemical element jEl including the atoms in the outer sphere
@param *ra            Atomic coordinates
@param iEl            i-th chemical element index
@param jEl            j-th chemical element index
@param *q             Scattering vector magnitude array
@param NumOMPthreads  Number of OpenMP threads
*/
void DebyeCoreCutOff(double * const I, const config * const cfg, const unsigned int NatomEli, const unsigned int NatomElj, const unsigned int NatomElj_out, const vector < vect3d <double> > * const ra, const unsigned int iEl, const unsigned int jEl, const double * const q, const int NumOMPthreads);

/**
Adds the average density correction to the scattering intensity when the cut-off is enabled (cfg.cutoff == true)

@param *I             Scattering intensity array.
@param *cfg           Parameters of simulation
@param *NatomEl       Array containing the total number of atoms of each chemical element
@param FF             X-ray atomic form-factor arrays for all chemical elements
@param SL             Array of neutron scattering lengths for all chemical elements
@param *q             Scattering vector magnitude array
@param Ntot           Total number of atoms in the nanoparticle
*/
void AddCutoff(double * const I, const config * const cfg, const unsigned int * const NatomEl, const double * const q, const vector <double*> FF, const vector<double> SL, const unsigned int Ntot);

/**
Computes the scattering intensity (powder diffraction pattern + partial intensities) using the original Debye equation (without the histogram approximation)

@param **I            Scattering intensity array. The memory is allocated inside the function.
@param *cfg           Parameters of simulation
@param *NatomEl       Array containing the total number of atoms of each chemical element
@param *ra            Atomic coordinates
@param FF             X-ray atomic form-factor arrays for all chemical elements
@param SL             Array of neutron scattering lengths for all chemical elements
@param *q             Scattering vector magnitude array
@param *Block         Array of the structural blocks 
@param Ntot           Total number of atoms in the nanoparticle
@param NumOMPthreads  Number of OpenMP threads
*/
void calcIntPartialDebye(double ** const I, const config * const cfg, const unsigned int * const NatomEl, const vector < vect3d <double> > * const ra, const  vector <double*> FF, const vector<double> SL, const double * const q, const block * const Block, const unsigned int Ntot, const int NumOMPthreads);

/**
Computes the contribution to the histogram of interatomic distances from the atoms of the chemical elements with indeces iEl and jEl

@param *rij_hist      Histogram of interatomic distances.
@param *cfg           Parameters of simulation
@param *ra            Atomic coordinate array
@param NatomEli       The total number of atoms of the chemical element iEl
@param NatomElj       The total number of atoms of the chemical element jEl
@param iEl            i-th chemical element index
@param jEl            j-th chemical element index
@param id             Unique thread id (MPI, OpenMP, MPI + OpenMP)
@param NumOMPthreads  Number of OpenMP threads
*/
void HistCore(unsigned long long int * const __restrict rij_hist, const config * const __restrict cfg, const vector < vect3d <double> > * const __restrict ra, const unsigned int NatomEli, const unsigned int NatomElj, const unsigned int iEl, const unsigned int jEl, const unsigned int id, const int NumOMPthreads);

/**
Computes the contribution to the histogram of interatomic distances from the atoms of the chemical elements with indeces iEl and jEl (if cfg->cutoff == true)

@param *rij_hist      Histogram of interatomic distances.
@param *cfg           Parameters of simulation
@param *ra            Atomic coordinate array
@param NatomEli       The total number of atoms of the chemical element iEl
@param NatomElj       The total number of atoms of the chemical element jEl
@param NatomElj_out   The total number of atoms of the chemical element jEl including the atoms in the outer sphere
@param iEl            i-th chemical element index
@param jEl            j-th chemical element index
@param id             Unique thread id (MPI, OpenMP, MPI + OpenMP)
@param NumOMPthreads  Number of OpenMP threads
*/
void HistCoreCutOff(unsigned long long int * const __restrict rij_hist, const config * const __restrict cfg, const vector < vect3d <double> > * const __restrict ra, const unsigned int NatomEli, const unsigned int NatomElj, const unsigned int NatomElj_out, const unsigned int iEl, const unsigned int jEl, const unsigned int id, const int NumOMPthreads);

/**
Computes the histogram of interatomic distances 

@param **rij_hist     Histogram of interatomic distances. The memory is allocated inside the function
@param *cfg           Parameters of simulation
@param *ra            Atomic coordinate array
@param *NatomEl       Array containing the total number of atoms of each chemical element
@param *NatomEl_outer Array containing the number of atoms of each chemical element including the atoms in the outer sphere (only if cfg.cutoff is True)
@param NumOMPthreads  Number of OpenMP threads
*/
void calcHist(unsigned long long int ** const rij_hist, const config *const cfg, const vector < vect3d <double> > * const ra, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const int NumOMPthreads);

/**
Computes the partial scattering intensity for a pair of elements using the histogram of interatomic distances

@param *I             Scattering intensity array
@param *rij_hist      Part fo the histogram of interatomic distances corresponding to a pair of elements
@param *cfg           Parameters of simulation
@param Nhist          Length of the part of rij_hist processed by current MPI process (cfg->Nhist if UseMPI is not defined)
@param iStart         Index of the first histogram bin processed by current MPI process (0 if UseMPI is not defined)
@param NumOMPthreads  Number of OpenMP threads
*/
void Int1DHistCore(double * const __restrict I, const unsigned long long int * const __restrict rij_hist, const config * const __restrict cfg, const double * const __restrict q, const unsigned int Nhist, const unsigned int iStart, const int NumOMPthreads);

/**
Computes the scattering intensity (powder diffraction pattern) using the histogram of interatomic distances

@param **I            Scattering intensity array. The memory is allocated inside the function
@param *rij_hist      Histogram of interatomic distances
@param *cfg           Parameters of simulation
@param *NatomEl       Array containing the total number of atoms of each chemical element
@param FF             X-ray atomic form-factor arrays for all chemical elements
@param SL             Array of neutron scattering lengths for all chemical elements
@param *q             Scattering vector magnitude array
@param Ntot           Total number of atoms in the nanoparticle
@param NumOMPthreads  Number of OpenMP threads
*/
void calcInt1DHist(double ** const I, unsigned long long int *rij_hist, const config * const cfg, const unsigned int * const NatomEl, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads);

/**
Depending on the computational scenario computes the scattering intensity (powder diffraction pattern) or PDF using the histogram of interatomic distances

@param **I            Scattering intensity array. The memory is allocated inside the function.
@param **PDF          PDF array. The memory is allocated inside the function.
@param *cfg           Parameters of simulation
@param *NatomEl       Array containing the total number of atoms of each chemical element
@param *ra            Atomic coordinate array
@param FF             X-ray atomic form-factor arrays for all chemical elements
@param SL             Array of neutron scattering lengths for all chemical elements
@param *q             Scattering vector magnitude array
@param Ntot           Total number of atoms in the nanoparticle
@param NumOMPthreads  Number of OpenMP threads
*/
void calcPDFandDebye(double ** const I, double ** const PDF, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const vector < vect3d <double> > * const ra, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads);

/**
Computes the 2D scattering intensity in the polar coordinates (q,q_fi) of the reciprocal space

@param ***I2D         2D scattering intensity array. The memory is allocated inside the function.
@param **I            1D (averaged over the polar angle) scattering intensity array. The memory is allocated inside the function.
@param *cfg           Parameters of simulation
@param *NatomEl	      Array containing the total number of atoms of each chemical element
@param *ra            Atomic coordinate array
@param FF             X-ray atomic form-factor arrays for all chemical elements
@param SL             Array of neutron scattering lengths for all chemical elements
@param *q             Scattering vector magnitude array
@param Ntot           Total number of atoms in the nanoparticle
@param NumOMPthreads  Number of OpenMP threads
*/
void calcInt2D(double *** const I2D, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const vector < vect3d <double> > *ra, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads);

#endif

//Calculates rotational matrix using the Euler angles in the case of intristic rotation in the user - defined rotational convention.
//See http ://en.wikipedia.org/wiki/Euler_angles#Relationship_to_other_representations for details.
void calcRotMatrix(vect3d <double> * const RM0, vect3d <double> * const RM1, vect3d <double> * const RM2, const vect3d <double> euler, const unsigned int convention) {
	const vect3d <double> cosEul = cos(euler);
	const vect3d <double> sinEul = sin(euler);
	switch (convention){
		case EulerXZX: //XZX
			RM0->assign(cosEul.y,-cosEul.z*sinEul.y,sinEul.y*sinEul.z);
			RM1->assign(cosEul.x*sinEul.y,cosEul.x*cosEul.y*cosEul.z-sinEul.x*sinEul.z,-cosEul.z*sinEul.x-cosEul.x*cosEul.y*sinEul.z);
			RM2->assign(sinEul.x*sinEul.y,cosEul.x*sinEul.z+cosEul.y*cosEul.z*sinEul.x,cosEul.x*cosEul.z-cosEul.y*sinEul.x*sinEul.z);
			break;
		case EulerXYX: //XYX
			RM0->assign(cosEul.y, sinEul.y*sinEul.z, cosEul.z*sinEul.y);
			RM1->assign(sinEul.x*sinEul.y, cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z, -cosEul.x*sinEul.z - cosEul.y*cosEul.z*sinEul.x);
			RM2->assign(-cosEul.x*sinEul.y, cosEul.z*sinEul.x + cosEul.x*cosEul.y*sinEul.z, cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z);
			break;
		case EulerYXY: //YXY
			RM0->assign(cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z, sinEul.x*sinEul.y, cosEul.x*sinEul.z + cosEul.y*cosEul.z*sinEul.x);
			RM1->assign(sinEul.y*sinEul.z, cosEul.y, -cosEul.z*sinEul.y);
			RM2->assign(-cosEul.z*sinEul.x - cosEul.x*cosEul.y*sinEul.z, cosEul.x*sinEul.y, cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z);
			break;
		case EulerYZY: //YZY
			RM0->assign(cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z, -cosEul.x*sinEul.y, cosEul.z*sinEul.x + cosEul.x*cosEul.y*sinEul.z);
			RM1->assign(cosEul.z*sinEul.y, cosEul.y, sinEul.y*sinEul.z);
			RM2->assign(-cosEul.x*sinEul.z - cosEul.y*cosEul.z*sinEul.x, sinEul.x*sinEul.y, cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z);
			break;
		case EulerZYZ: //ZYZ
			RM0->assign(cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z, -cosEul.z*sinEul.x - cosEul.x*cosEul.y*sinEul.z, cosEul.x*sinEul.y);
			RM1->assign(cosEul.x*sinEul.z + cosEul.y*cosEul.z*sinEul.x, cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z, sinEul.x*sinEul.y);
			RM2->assign(-cosEul.z*sinEul.y, sinEul.y*sinEul.z, cosEul.y);
			break;
		case EulerZXZ: //ZXZ
			RM0->assign(cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z, -cosEul.x*sinEul.z - cosEul.y*cosEul.z*sinEul.x, sinEul.x*sinEul.y);
			RM1->assign(cosEul.z*sinEul.x + cosEul.x*cosEul.y*sinEul.z, cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z, -cosEul.x*sinEul.y);
			RM2->assign(sinEul.y*sinEul.z, cosEul.z*sinEul.y, cosEul.y);
			break;
		case EulerXZY: //XZY
			RM0->assign(cosEul.y*cosEul.z,-sinEul.y,cosEul.y*sinEul.z);
			RM1->assign(sinEul.x*sinEul.z+cosEul.x*cosEul.z*sinEul.y,cosEul.x*cosEul.y,cosEul.x*sinEul.y*sinEul.z-cosEul.z*sinEul.x);
			RM2->assign(cosEul.z*sinEul.x*sinEul.y-cosEul.x*sinEul.z,cosEul.y*sinEul.x,cosEul.x*cosEul.z+sinEul.x*sinEul.y*sinEul.z);
			break;
		case EulerXYZ: //XYZ
			RM0->assign(cosEul.y*cosEul.z, -cosEul.y*sinEul.z, sinEul.y);
			RM1->assign(cosEul.x*sinEul.z + cosEul.z*sinEul.x*sinEul.y, cosEul.x*cosEul.z - sinEul.x*sinEul.y*sinEul.z,-cosEul.y*sinEul.x);
			RM2->assign(sinEul.x*sinEul.z - cosEul.x*cosEul.z*sinEul.y,cosEul.z*sinEul.x + cosEul.x*sinEul.y*sinEul.z,cosEul.x*cosEul.y);
			break;
		case EulerYXZ: //YXZ
			RM0->assign(cosEul.x*cosEul.z + sinEul.x*sinEul.y*sinEul.z, cosEul.z*sinEul.x*sinEul.y - cosEul.x*sinEul.z,cosEul.y*sinEul.x);
			RM1->assign(cosEul.y*sinEul.z, cosEul.y*cosEul.z,-sinEul.y);
			RM2->assign(cosEul.x*sinEul.y*sinEul.z - cosEul.z*sinEul.x,cosEul.x*cosEul.z*sinEul.y+sinEul.x*sinEul.z,cosEul.x*cosEul.y);
			break;
		case EulerYZX: //YZX
			RM0->assign(cosEul.x*cosEul.y, sinEul.x*sinEul.z - cosEul.x*cosEul.z*sinEul.y, cosEul.z*sinEul.x+cosEul.x*sinEul.y*sinEul.z);
			RM1->assign(sinEul.y, cosEul.y*cosEul.z, -cosEul.y*sinEul.z);
			RM2->assign(-cosEul.y*sinEul.x, cosEul.x*sinEul.z + cosEul.z*sinEul.x*sinEul.y, cosEul.x*cosEul.z - sinEul.x*sinEul.y*sinEul.z);
			break;
		case EulerZYX: //ZYX
			RM0->assign(cosEul.x*cosEul.y, -cosEul.z*sinEul.x + cosEul.x*sinEul.z*sinEul.y, sinEul.x*sinEul.z + cosEul.x*cosEul.z*sinEul.y);
			RM1->assign(cosEul.y*sinEul.x, cosEul.x*cosEul.z + sinEul.x*sinEul.y*sinEul.z, -cosEul.x*sinEul.z + cosEul.z*sinEul.x*sinEul.y);
			RM2->assign(-sinEul.y, cosEul.y*sinEul.z, cosEul.y*cosEul.z);
			break;
		case EulerZXY: //ZXY
			RM0->assign(cosEul.x*cosEul.z - sinEul.x*sinEul.y*sinEul.z, -cosEul.y*sinEul.x, cosEul.x*sinEul.z+cosEul.z*sinEul.x*sinEul.y);
			RM1->assign(cosEul.z*sinEul.x + cosEul.x*sinEul.y*sinEul.z, cosEul.x*cosEul.y, sinEul.x*sinEul.z - cosEul.x*cosEul.z*sinEul.y);
			RM2->assign(-cosEul.y*sinEul.z, sinEul.y, cosEul.y*cosEul.z);
			break;
	}
}

//Returns the geometric center of the atomic ensemble (nanoparticle)
vect3d<double> GetCenter(const vector < vect3d<double> > * const r, const unsigned int Nel, const unsigned int Ntot){
	vect3d <double> rC(0,0,0);
	for (unsigned int iEl = 0; iEl < Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = r[iEl].begin(); ri != r[iEl].end(); ri++)	rC += *ri;
	}
	return rC /= Ntot;
}

//Returns the largest possible interatomic distance in the atomic ensemble (nanoperticle)
double GetEnsembleSize(const vector < vect3d<double> > * const r, const vect3d <double> rCenter, const unsigned int Nel) {
	double Rmax = 0;
	for (unsigned int iEl = 0; iEl < Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = r[iEl].begin(); ri != r[iEl].end(); ri++) Rmax = MAX(Rmax, (rCenter - *ri).sqr());
	}
	return 2. * sqrt(Rmax);
}

//Returns the average atomic density of the atomic ensemble (nanoperticle)
double GetAtomicDensity(const vector < vect3d<double> > * const r, const vect3d <double> rCenter, const double Rcutoff, const unsigned int Nel){
	const double Rmax2 = SQR(Rcutoff);
	unsigned int count = 0;
	for (unsigned int iEl = 0; iEl < Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = r[iEl].begin(); ri != r[iEl].end(); ri++)	{
			if ((rCenter - *ri).sqr() < Rmax2) count++;
		}
	}
	return count / (4. / 3. * PI * Rmax2 * sqrt(Rmax2));
}

//Returns the average atomic density of the atomic ensemble if cfg->BoxEdge parameters are defined
double GetAtomicDensityBox(const unsigned int N, const vect3d <double> *BoxEdge){
	double dens = N / ABS(BoxEdge[0].dot(BoxEdge[1] * BoxEdge[2]));
	return dens;
}

//Returns the square of the distance between the atom 'rAtom' and its most distant neighboring atom from 'rAtomNeighb'.
//Called only if the local rearrangement of atoms is enabled.
double Rmax2(const vect3d <double> rAtom, const vector< vect3d <double> > rAtomNeighb){
	double rmax2=0;
	for (unsigned int iNeib=0;iNeib<rAtomNeighb.size();iNeib++)	rmax2 = MAX(rmax2, (rAtom - rAtomNeighb[iNeib]).sqr());
	return rmax2;
}

//Moves the atoms located inside the inner cut-off box in the beginning of r[iEl] vector. Removes from r[iEl] the atoms locted outside the outer cut-off box
unsigned int cutoffBox(vector < vect3d <double> > * const r, const config * const cfg, unsigned int * const NatomEl, unsigned int * const NatomEl_outer){
	const double a = cfg->BoxEdge[0].mag(), b = cfg->BoxEdge[1].mag(), c = cfg->BoxEdge[2].mag();
	const vect3d <double> e0 = cfg->BoxEdge[0]/a, e1 = cfg->BoxEdge[1]/b, e2 = cfg->BoxEdge[2]/c;
	const double det = e0.x * (e1.y * e2.z - e1.z * e2.y) + e0.y * (e1.z * e2.x - e1.x * e2.z) + e0.z * (e1.x * e2.y - e1.y * e2.x);
	const vect3d <double> n2 = (e0 * e1).norm(), n0 = (e1 * e2).norm(), n1 = (e2 * e0).norm();
	const double aRcut = cfg->Rcutoff / e0.dot(n0), bRcut = cfg->Rcutoff / e1.dot(n1), cRcut = cfg->Rcutoff / e2.dot(n2);
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
		vector < vect3d <double> > r_inner, r_outer;
		for (vector<vect3d <double> >::const_iterator ri = r[iEl].begin(); ri != r[iEl].end(); ri++) {
			const vect3d <double> rt = *ri - cfg->BoxCorner;
			vect3d <double> rtb;			
			rtb.x = (rt.x * (e1.y * e2.z - e1.z * e2.y) + e0.y * (e1.z * rt.z - rt.y * e2.z) + e0.z * (rt.y * e2.y - e1.y * rt.z)) / det;
			rtb.y = (e0.x * (rt.y * e2.z - e1.z * rt.z) + rt.x * (e1.z * e2.x - e1.x * e2.z) + e0.z * (e1.x * rt.z - rt.y * e2.x)) / det;
			rtb.z = (e0.x * (e1.y * rt.z - rt.y * e2.y) + e0.y * (rt.y * e2.x - e1.x * rt.z) + rt.x * (e1.x * e2.y - e1.y * e2.x)) / det;
			if ((rtb.x >= 0) && (rtb.x < a) && (rtb.y >= 0) && (rtb.y < b) && (rtb.z >= 0) && (rtb.z < c)) r_inner.push_back(*ri);
			else if ((rtb.x >= -aRcut) && (rtb.x < a + aRcut) && (rtb.y >= -bRcut) && (rtb.y < b + bRcut) && (rtb.z >= -cRcut) && (rtb.z < c + cRcut)) r_outer.push_back(*ri);
		}
		NatomEl[iEl] = (unsigned int)r_inner.size();
		Ntot += NatomEl[iEl];
		NatomEl_outer[iEl] = NatomEl[iEl] + (unsigned int)r_outer.size();
		r[iEl].clear();
		r[iEl].insert(r[iEl].end(), r_inner.begin(), r_inner.end());
		r[iEl].insert(r[iEl].end(), r_outer.begin(), r_outer.end());
	}
	return Ntot;
}

//Moves the atoms located inside the inner cut-off sphere in the beginning of r[iEl] vector. Removes from r[iEl] the atoms locted outside the outer cut-off sphere
unsigned int cutoffSphere(vector < vect3d <double> > * const r, const config * const cfg, unsigned int * const NatomEl, unsigned int * const NatomEl_outer, const vect3d <double> rCenter){
	const double Ri2 = SQR(cfg->Rsphere);
	const double Ro2 = SQR(cfg->Rsphere + cfg->Rcutoff);
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
		vector < vect3d <double> > r_inner, r_outer;
		for (vector<vect3d <double> >::const_iterator ri = r[iEl].begin(); ri != r[iEl].end(); ri++) {
			const double dist2 = (rCenter - *ri).sqr();
			if (dist2 < Ri2) r_inner.push_back(*ri);
			else if (dist2 < Ro2) r_outer.push_back(*ri);
		}
		NatomEl[iEl] = (unsigned int) r_inner.size();
		Ntot += NatomEl[iEl];
		NatomEl_outer[iEl] = NatomEl[iEl] + (unsigned int) r_outer.size();
		r[iEl].clear();
		r[iEl].insert(r[iEl].end(), r_inner.begin(), r_inner.end());
		r[iEl].insert(r[iEl].end(), r_outer.begin(), r_outer.end());
	}
	return Ntot;
}

//Calculates the total atomic ensemble and prints it in the .xyz file if cfg.PrintAtoms is true. 
//Returns the total number of atoms in the atomic ensemble(nanoparticle).
unsigned int CalcAndPrintAtoms(config * const cfg, block * const Block, vector < vect3d <double> > ** const ra, unsigned int ** const NatomEl, unsigned int ** const NatomEl_outer, const map <string, unsigned int> ID, const map <unsigned int, string> EN){
	unsigned int Ntot = 0;
	*ra = new vector < vect3d <double> >[cfg->Nel];
	*NatomEl = new unsigned int[cfg->Nel];
	if (cfg->cutoff) *NatomEl_outer = new unsigned int[cfg->Nel];
	if (!myid) {
		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
		for (unsigned int iB = 0; iB<cfg->Nblocks; iB++) Block[iB].calcAtoms(*ra, ID, cfg->Nel);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			(*NatomEl)[iEl] = (unsigned int) (*ra)[iEl].size();
			Ntot += (*NatomEl)[iEl];
		}
		const vect3d <double> rCenter = GetCenter(*ra, cfg->Nel, Ntot);
		double D = GetEnsembleSize(*ra, rCenter, cfg->Nel);
		if (cfg->cutoff) {
			if (cfg->sphere) {
				if (D < 2. * (cfg->Rsphere + cfg->Rcutoff)) cout << "/nWarning: the sum of sphere radius and cutoff radius " << (cfg->Rsphere + cfg->Rcutoff) << " A is to big for the atomic ensemble with the linear size " << D << " A.\n" << endl;
				Ntot = cutoffSphere(*ra, cfg, *NatomEl, *NatomEl_outer, rCenter);				
			}
			else Ntot = cutoffBox(*ra, cfg, *NatomEl, *NatomEl_outer);
			D = cfg->Rcutoff;
		}		
		if (cfg->PrintAtoms)	printAtoms(*ra, *NatomEl, cfg->name, EN, Ntot);
		if ((cfg->scenario == Debye_hist) || (cfg->scenario == DebyePDF) || (cfg->scenario == PDFonly)) cfg->Nhist = (unsigned int)(D / cfg->hist_bin) + 1;
		if (((cfg->scenario == DebyePDF) || (cfg->scenario == PDFonly) || (cfg->cutoff)) && (!cfg->p0)) {
			if (cfg->cutoff) {
				if (cfg->sphere) cfg->p0 = GetAtomicDensity(*ra, rCenter, cfg->Rcutoff, cfg->Nel);
				else cfg->p0 = GetAtomicDensityBox(Ntot, cfg->BoxEdge);
			}
			else cfg->p0 = GetAtomicDensity(*ra, rCenter, 0.25 * D, cfg->Nel);
			cout << "Approximate atomic density of the sample is " << cfg->p0 << " A^(-3).\n" << endl;
		}
		const chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
		cout << "Atomic ensemble calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s\n" << endl;
	}
#ifdef UseMPI
	MPI_Bcast(&Ntot, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if ((cfg->scenario == DebyePDF) || (cfg->scenario == PDFonly) || (cfg->cutoff)) MPI_Bcast(&cfg->p0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(*NatomEl, cfg->Nel, MPI_INT, 0, MPI_COMM_WORLD);
	if (cfg->cutoff) MPI_Bcast(*NatomEl_outer, cfg->Nel, MPI_INT, 0, MPI_COMM_WORLD);
	if (cfg->scenario == s2D){
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			if (!myid) {
				for (unsigned int pid = 1; pid<(unsigned int) (numprocs); pid++){
					unsigned int ist = (*NatomEl)[iEl] / numprocs*pid;
					(pid<(*NatomEl)[iEl]%numprocs) ? ist += pid : ist += (*NatomEl)[iEl]%numprocs;
					unsigned int Nsend = (*NatomEl)[iEl] / numprocs;
					if (pid<(*NatomEl)[iEl]%numprocs) Nsend++;
					MPI_Send(&(*ra)[iEl][ist], 3*Nsend, MPI_DOUBLE, pid, pid, MPI_COMM_WORLD);
				}
			}
			unsigned int NatomElNew = (*NatomEl)[iEl] / numprocs;
			if ((unsigned int)(myid)<(*NatomEl)[iEl] % (unsigned int)(numprocs)) NatomElNew++;
			(*NatomEl)[iEl] = NatomElNew;
			(*ra)[iEl].resize(NatomElNew);
			MPI_Status status;
			if (myid) 	MPI_Recv(&(*ra)[iEl][0], 3*NatomElNew, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD, &status);
		}
	}
	else {
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			if (cfg->cutoff) {
				if (myid) (*ra)[iEl].resize((*NatomEl_outer)[iEl]);
				MPI_Bcast(&(*ra)[iEl][0], 3 * (*NatomEl_outer)[iEl], MPI_DOUBLE, 0, MPI_COMM_WORLD);
			}
			else {
				if (myid) (*ra)[iEl].resize((*NatomEl)[iEl]);
				MPI_Bcast(&(*ra)[iEl][0], 3 * (*NatomEl)[iEl], MPI_DOUBLE, 0, MPI_COMM_WORLD);
			}
		}
	}
	if (cfg->calcPartialIntensity) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++) {
			if (myid) Block[iB].NatomEl = new unsigned int[cfg->Nel];
			MPI_Bcast(Block[iB].NatomEl, cfg->Nel, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}
	if ((cfg->scenario == Debye_hist) || (cfg->scenario == DebyePDF) || (cfg->scenario == PDFonly)) MPI_Bcast(&cfg->Nhist, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
	return Ntot;
}

#if !defined(UseOCL) && !defined(UseCUDA)

//Computes the polarization factor and multiplies the 2D scattering intensity by this factor
void PolarFactor2D(double * const * const I2D, const double * const q, const config * const cfg){
	for (unsigned int iq = 0; iq<cfg->q.N; iq++){
		const double sintheta = q[iq] * (cfg->lambda * 0.25 / PI);
		const double cos2theta = 1. - 2. * SQR(sintheta);
		const double factor = 0.5 * (1. + SQR(cos2theta));
		for (unsigned int ifi = 0; ifi<cfg->Nfi; ifi++) I2D[iq][ifi] *= factor;
	}
}

//Computes the polarization factor and multiplies scattering intensity by this factor
void PolarFactor1D(double * const I, const double * const q, const config * const cfg){
	for (unsigned int iq = 0; iq<cfg->q.N; iq++){
		const double sintheta = q[iq] * (cfg->lambda * 0.25 / PI);
		const double cos2theta = 1. - 2. * SQR(sintheta);
		I[iq] *= 0.5 * (1. + SQR(cos2theta));
	}
}

//Computes the contribution to the scattering intensity from the atoms of the chemical elements with indeces iEl and jEl using the Debye equation (without the histogram approximation)
void DebyeCore(double * const I, const config * cfg, const unsigned int NatomEli, const unsigned int NatomElj, const vector < vect3d <double> > * const ra, const unsigned int iEl, const unsigned int jEl, const double * const q, const int NumOMPthreads) {
#ifdef UseOMP
#pragma omp parallel num_threads(NumOMPthreads) 
#endif
	{
		int tid = 0;
#ifdef UseOMP
		tid = omp_get_thread_num();
#endif	
		unsigned int id = myid*NumOMPthreads + tid, jAtomST = 0;
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) I[tid*cfg->q.N + iq] = 0;
		unsigned int step = 2 * id + 1, count = 0;
		for (unsigned int iAtom = id; iAtom < NatomEli; iAtom += step, count++) {
			(count % 2) ? step = 2 * id + 1 : step = 2 * (numprocs*NumOMPthreads - id) - 1;
			(jEl == iEl) ? jAtomST = iAtom + 1 : jAtomST = 0;
			for (unsigned int jAtom = jAtomST; jAtom < NatomElj; jAtom++) {
				const double rij = (ra[iEl][iAtom] - ra[jEl][jAtom]).mag();
				for (unsigned int iq = 0; iq < cfg->q.N; iq++)	{
					const double qrij = rij * q[iq] + 0.00000001;
					I[tid*cfg->q.N + iq] += 2. * sin(qrij) / qrij;
				}
			}
		}
	}
}

//Computes the contribution to the scattering intensity from the atoms of the chemical elements with indeces iEl and jEl using the Debye equation (without the histogram approximation) when the cut-off is enabled (cfg.cutoff == true)
void DebyeCoreCutOff(double * const I, const config * const cfg, const unsigned int NatomEli, const unsigned int NatomElj, const unsigned int NatomElj_out, const vector < vect3d <double> > * const ra, const unsigned int iEl, const unsigned int jEl, const double * const q, const int NumOMPthreads) {
	const double Rcutoff2 = SQR(cfg->Rcutoff);
#ifdef UseOMP
#pragma omp parallel num_threads(NumOMPthreads) 
#endif
	{
		int tid = 0;
#ifdef UseOMP
		tid = omp_get_thread_num();
#endif	
		unsigned int jAtomST = 0;
		if (jEl < iEl) jAtomST = NatomElj;
		const unsigned int id = myid*NumOMPthreads + tid;
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) I[tid*cfg->q.N + iq] = 0;
		unsigned int step = 2 * id + 1, count = 0;
		for (unsigned int iAtom = id; iAtom < NatomEli; iAtom += step, count++) {
			(count % 2) ? step = 2 * id + 1 : step = 2 * (numprocs*NumOMPthreads - id) - 1;
			if (jEl == iEl) jAtomST = iAtom + 1;
			for (unsigned int jAtom = jAtomST; jAtom < NatomElj_out; jAtom++) {
				const double rij2 = (ra[iEl][iAtom] - ra[jEl][jAtom]).sqr();
				if (rij2 > Rcutoff2) continue;
				double damp = 1., mult = 2.;
				if (jAtom > NatomElj) mult = 1.;
				const double rij = sqrt(rij2);
				if (cfg->damping) {
					const double x = PI * rij / cfg->Rcutoff;
					damp = sin(x) / x;
				}
				for (unsigned int iq = 0; iq < cfg->q.N; iq++)	{
					const double qrij = rij * q[iq] + 0.00000001;
					I[tid*cfg->q.N + iq] += mult * damp * sin(qrij) / qrij;
				}
			}
		}
	}
}

//Adds the average density correction to the scattering intensity when the cut-off is enabled (cfg.cutoff == true)
void AddCutoff(double * const I, const config * const cfg, const unsigned int * const NatomEl, const double * const q, const vector <double*> FF, const vector<double> SL, const unsigned int Ntot){
	double FFaver = 0;
	if (cfg->source == neutron) {
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++)	FFaver += SL[iEl] * NatomEl[iEl];
		FFaver /= Ntot;
	}	
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
		if (q[iq] < 1.e-7) continue;
		if (cfg->source == xray) {
			FFaver = 0;
			for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++)	FFaver += FF[iEl][iq] * NatomEl[iEl];
			FFaver /= Ntot;
		}
		const double qrcut = q[iq] * cfg->Rcutoff;
		if (cfg->damping) I[iq] += 4. * PI * cfg->p0 * SQR(FFaver) * SQR(cfg->Rcutoff) * sin(qrcut) / (q[iq] * (SQR(qrcut) - SQR(PI)));
		else I[iq] += 4. * PI * cfg->p0 * SQR(FFaver) * (cfg->Rcutoff * cos(qrcut) - sin(qrcut) / q[iq]) / SQR(q[iq]);
	}
}

//Computes the scattering intensity (powder diffraction pattern) using the original Debye equation (without the histogram approximation)
void calcIntDebye(double ** const I, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const vector < vect3d <double> > * const ra, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads) {
	const chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	if (!myid) {
		*I = new double[cfg->q.N];
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] = 0;
	}
	double * const Itemp = new double[cfg->q.N*NumOMPthreads];
#ifdef UseMPI
	double * const Iloc = new double[cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) Iloc[iq] = 0;
#endif
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
		unsigned int jElSt = iEl;
		if (cfg->cutoff) jElSt = 0;
		for (unsigned int jEl = jElSt; jEl < cfg->Nel; jEl++) {
			if (cfg->cutoff) DebyeCoreCutOff(Itemp, cfg, NatomEl[iEl], NatomEl[jEl], NatomEl_outer[jEl], ra, iEl, jEl, q, NumOMPthreads);
			else DebyeCore(Itemp, cfg, NatomEl[iEl], NatomEl[jEl], ra, iEl, jEl, q, NumOMPthreads);
#ifdef UseOMP
#pragma omp parallel for num_threads(NumOMPthreads) 
			for (int iq = 0; iq < (int) cfg->q.N; iq++) {
				for (int tid = 1; tid < NumOMPthreads; tid++) Itemp[iq] += Itemp[tid * cfg->q.N + iq];
			}
#endif
			if (cfg->source == xray) {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
					Iloc[iq] += Itemp[iq] * FF[iEl][iq] * FF[jEl][iq];
#else
					(*I)[iq] += Itemp[iq] * FF[iEl][iq] * FF[jEl][iq];
#endif
				}
			}
			else {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
					Iloc[iq] += Itemp[iq] * SL[iEl] * SL[jEl];
#else
					(*I)[iq] += Itemp[iq] * SL[iEl] * SL[jEl];
#endif
				}
			}
		}
	}
	delete[] Itemp;
#ifdef UseMPI
	MPI_Reduce(Iloc, *I, cfg->q.N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	delete[] Iloc;
#endif
	if (!myid) {
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
			if (cfg->source == xray) {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += NatomEl[iEl] * SQR(FF[iEl][iq]);
			}
			else {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += NatomEl[iEl] * SQR(SL[iEl]);
			}
		}
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] /= Ntot;
		if (cfg->cutoff) AddCutoff(*I, cfg,  NatomEl, q, FF, SL, Ntot);
		if (cfg->PolarFactor) PolarFactor1D(*I, q, cfg);
	}
	const chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	if (!myid) cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}

//Computes the scattering intensity (powder diffraction pattern + partial intensities) using the original Debye equation (without the histogram approximation)
void calcIntPartialDebye(double ** const I, const config * const cfg, const unsigned int * const NatomEl, const vector < vect3d <double> > * const ra, const  vector <double*> FF, const vector<double> SL, const double * const q, const block * const Block, const unsigned int Ntot, const int NumOMPthreads) {
	const chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	const unsigned int Nparts = (cfg->Nblocks * (cfg->Nblocks + 1)) / 2;
	const unsigned int Isize = Nparts * cfg->q.N;
	if (!myid) {
		*I = new double[Isize + cfg->q.N];
		for (unsigned int iq = 0; iq < Isize + cfg->q.N; iq++) (*I)[iq] = 0;
	}
	double * const Itemp = new double[Isize*NumOMPthreads];
#ifdef UseMPI
	double * const Iloc = new double[Isize];
	for (unsigned int iq = 0; iq < Isize; iq++) Iloc[iq] = 0;
#endif
	unsigned int * const NatomElBlock = new unsigned int[cfg->Nel*cfg->Nblocks];
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			NatomElBlock[iEl*cfg->Nblocks + iB] = Block[iB].NatomEl[iEl];			
		}
	}
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){		
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++){
#ifdef UseOMP
#pragma omp parallel num_threads(NumOMPthreads) 
#endif
			{
				int tid = 0;
#ifdef UseOMP
				tid = omp_get_thread_num();
#endif
				const unsigned int id = myid*NumOMPthreads + tid;
				for (unsigned int iq = 0; iq < Isize; iq++) Itemp[tid*Isize + iq] = 0;
				unsigned int iAtomSB = 0, jAtomSB = 0, jBlockST = 0, jAtomST = 0;
				for (unsigned int iB = 0; iB < cfg->Nblocks; iAtomSB += NatomElBlock[iEl*cfg->Nblocks + iB], iB++) {
					unsigned int step = 2 * id + 1, count = 0, Istart = 0;
					for (unsigned int iAtom = iAtomSB + id; iAtom < iAtomSB + NatomElBlock[iEl*cfg->Nblocks + iB]; iAtom += step, count++) {
						(count % 2) ? step = 2 * id + 1 : step = 2 * (numprocs*NumOMPthreads - id) - 1;
						if (jEl == iEl) {
							jBlockST = iB;
							jAtomSB = iAtomSB;
							jAtomST = iAtom + 1;
						}
						else jAtomSB = 0;
						for (unsigned int jB = jBlockST; jB < cfg->Nblocks; jAtomSB += NatomElBlock[jEl*cfg->Nblocks + jB], jB++) {
							(jB > iB) ? Istart = cfg->q.N * (cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + jB) : Istart = cfg->q.N * (cfg->Nblocks * jB - (jB * (jB + 1)) / 2 + iB);
							for (unsigned int jAtom = MAX(jAtomSB, jAtomST); jAtom < jAtomSB + NatomElBlock[jEl*cfg->Nblocks + jB]; jAtom++) {
								const double rij = (ra[iEl][iAtom] - ra[jEl][jAtom]).mag();
								for (unsigned int iq = 0; iq < cfg->q.N; iq++)	{
									const double qrij = rij * q[iq] + 0.00000001;
									Itemp[tid*Isize + Istart + iq] += 2. * sin(qrij) / qrij;
								}
							}
						}
					}
				}				
			}
#ifdef UseOMP
#pragma omp parallel for num_threads(NumOMPthreads) 
			for (int iq = 0; iq < (int)Isize; iq++) {
				for (int tid = 1; tid < NumOMPthreads; tid++) Itemp[iq] += Itemp[tid * Isize + iq];
			}
#endif
			if (cfg->source == xray) {
				for (unsigned iPart = 0; iPart < Nparts; iPart++) {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
						Iloc[iPart *cfg->q.N + iq] += Itemp[iPart *cfg->q.N + iq] * FF[iEl][iq] * FF[jEl][iq];
#else
						(*I)[(iPart + 1)*cfg->q.N + iq] += Itemp[iPart *cfg->q.N + iq] * FF[iEl][iq] * FF[jEl][iq];
#endif
					}
				}
			}
			else {
				for (unsigned iPart = 0; iPart < Nparts; iPart++) {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
						Iloc[iPart *cfg->q.N + iq] += Itemp[iPart *cfg->q.N + iq] * SL[iEl] * SL[jEl];
#else
						(*I)[(iPart +1)*cfg->q.N + iq] += Itemp[iPart *cfg->q.N + iq] * SL[iEl] * SL[jEl];
#endif
					}
				}
			}
		}
	}
	delete[] Itemp;
#ifdef UseMPI
	MPI_Reduce(Iloc, *I + cfg->q.N, Isize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	delete[] Iloc;
#endif
	if (!myid) {
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
			for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
				const unsigned int Istart = cfg->q.N * (1 + (cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + iB));
				if (cfg->source == xray) {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[Istart + iq] += NatomElBlock[iEl*cfg->Nblocks + iB] * SQR(FF[iEl][iq]);
				}
				else {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[Istart + iq] += NatomElBlock[iEl*cfg->Nblocks + iB] * SQR(SL[iEl]);
				}
			}
		}
		for (unsigned int iq = cfg->q.N; iq < Isize + cfg->q.N; iq++) (*I)[iq] /= Ntot;
		if (cfg->PolarFactor) {
			for (unsigned iPart = 1; iPart < Nparts + 1; iPart++) PolarFactor1D(*I + iPart * cfg->q.N, q, cfg);
		}
		for (unsigned iPart = 1; iPart < Nparts + 1; iPart++) {
			for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += (*I)[iPart*cfg->q.N + iq];
		}
	}
	delete[] NatomElBlock;	
	const chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	if (!myid) cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}

//Computes the contribution to the histogram of interatomic distances from the atoms of the chemical elements with indeces iEl and jEl
void HistCore(unsigned long long int * const __restrict rij_hist, const config * const __restrict cfg, const vector < vect3d <double> > * const __restrict ra, const unsigned int NatomEli, const unsigned int NatomElj, const unsigned int iEl, const unsigned int jEl, const unsigned int id, const int NumOMPthreads) {
	unsigned int step = 2 * id + 1;
	unsigned int count = 0;
	unsigned int jAtomST = 0;
	for (unsigned int iAtom = id; iAtom < NatomEli; iAtom += step, count++) {
		(count % 2) ? step = 2 * id + 1 : step = 2 * (numprocs*NumOMPthreads - id) - 1;
		if (jEl == iEl) jAtomST = iAtom + 1;
		for (unsigned int jAtom = jAtomST; jAtom < NatomElj; jAtom++) {
			const double rij = (ra[iEl][iAtom] - ra[jEl][jAtom]).mag();
			// if (rij < 1.0) cout << rij << endl;
			rij_hist[(unsigned int)(rij / cfg->hist_bin)] += 2;
		}
	}
}

//Computes the contribution to the histogram of interatomic distances from the atoms of the chemical elements with indeces iEl and jEl (if cfg->cutoff == true)
void HistCoreCutOff(unsigned long long int * const __restrict rij_hist, const config * const __restrict  cfg, const vector < vect3d <double> > * const __restrict ra, const unsigned int NatomEli, const unsigned int NatomElj, const unsigned int NatomElj_out, const unsigned int iEl, const unsigned int jEl, const unsigned int id, const int NumOMPthreads) {
	unsigned int step = 2 * id + 1;
	unsigned int count = 0;
	unsigned int jAtomST = 0;
	if (jEl < iEl) jAtomST = NatomElj;
	const double Rcutoff2 = SQR(cfg->Rcutoff);
	for (unsigned int iAtom = id; iAtom < NatomEli; iAtom += step, count++) {
		(count % 2) ? step = 2 * id + 1 : step = 2 * (numprocs*NumOMPthreads - id) - 1;
		if (jEl == iEl) jAtomST = iAtom + 1;
		for (unsigned int jAtom = jAtomST; jAtom < NatomElj; jAtom++) {
			const double rij2 = (ra[iEl][iAtom] - ra[jEl][jAtom]).sqr();
			if (rij2 > Rcutoff2) continue;
			rij_hist[(unsigned int)(sqrt(rij2) / cfg->hist_bin)] += 2;
		}
		for (unsigned int jAtom = NatomElj; jAtom < NatomElj_out; jAtom++) {
			const double rij2 = (ra[iEl][iAtom] - ra[jEl][jAtom]).sqr();
			if (rij2 > Rcutoff2) continue;
			rij_hist[(unsigned int)(sqrt(rij2) / cfg->hist_bin)] += 1;
		}
	}
}

//Computes the histogram of interatomic distances
void calcHist(unsigned long long int ** const rij_hist, const config *const cfg, const vector < vect3d <double> > * const ra, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const int NumOMPthreads) {
	const unsigned int NhistEl = (cfg->Nel * (cfg->Nel + 1)) / 2 * cfg->Nhist;
#ifdef UseMPI
	unsigned long long int * const rij_hist_loc = new unsigned long long int[NhistEl * NumOMPthreads];
	for (unsigned int i = 0; i < NhistEl* NumOMPthreads; i++) rij_hist_loc[i] = 0;
#endif
	if (!myid) {
		*rij_hist = new unsigned long long int[NhistEl * NumOMPthreads];
		for (unsigned int i = 0; i < NhistEl * NumOMPthreads; i++) (*rij_hist)[i] = 0;
	}
#ifdef UseOMP
#pragma omp parallel num_threads(NumOMPthreads) 
#endif
{
	int tid = 0;
#ifdef UseOMP
	tid = omp_get_thread_num();
#endif	
	const unsigned int id = myid * NumOMPthreads + tid; 
	const unsigned int NstartID = NhistEl * tid;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		unsigned int jElSt = iEl;
		if (cfg->cutoff) jElSt = 0;
		for (unsigned int jEl = jElSt; jEl < cfg->Nel; jEl++) {
			unsigned int Nstart = NstartID;
			(jEl > iEl) ? Nstart += cfg->Nhist * (cfg->Nel * iEl - (iEl * (iEl + 1)) / 2 + jEl) : Nstart += cfg->Nhist * (cfg->Nel * jEl - (jEl * (jEl + 1)) / 2 + iEl);
#ifdef UseMPI
			if (cfg->cutoff) HistCoreCutOff(rij_hist_loc + Nstart, cfg, ra, NatomEl[iEl], NatomEl[jEl], NatomEl_outer[jEl], iEl, jEl, id, NumOMPthreads);
			else HistCore(rij_hist_loc + Nstart, cfg, ra, NatomEl[iEl], NatomEl[jEl], iEl, jEl, id, NumOMPthreads);
#else
			if (cfg->cutoff) HistCoreCutOff(*rij_hist + Nstart, cfg, ra, NatomEl[iEl], NatomEl[jEl], NatomEl_outer[jEl], iEl, jEl, id, NumOMPthreads);
			else HistCore(*rij_hist + Nstart, cfg, ra, NatomEl[iEl], NatomEl[jEl], iEl, jEl, id, NumOMPthreads);
#endif
		}
	}
}
#ifdef UseOMP
#pragma omp parallel for num_threads(NumOMPthreads) 
for (int i = 0; i < (int) NhistEl; i++) {
#ifdef UseMPI
	for (int tid = 1; tid < NumOMPthreads; tid++) rij_hist_loc[i] += rij_hist_loc[tid * NhistEl + i];
#else
	for (int tid = 1; tid < NumOMPthreads; tid++) (*rij_hist)[i] += (*rij_hist)[tid * NhistEl + i];
#endif
}
#endif
#ifdef UseMPI
	MPI_Reduce(rij_hist_loc, *rij_hist, NhistEl, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	delete[] rij_hist_loc;
#endif	
}

void Int1DHistCore(double * const __restrict  I, const unsigned long long int * const __restrict rij_hist, const config * const __restrict cfg, const double * const __restrict q, const unsigned int Nhist, const unsigned int iStart, const int NumOMPthreads){
#ifdef UseOMP
#pragma omp parallel num_threads(NumOMPthreads) 
#endif
	{
		int tid = 0;
#ifdef UseOMP
		tid = omp_get_thread_num();
#endif
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) I[tid * cfg->q.N + iq] = 0;
		double damp = 1.;
		for (int i = tid; i < (int)Nhist; i += NumOMPthreads) {
			if (!rij_hist[i]) continue;			
			const double rij = ((double)(iStart + i) + 0.5) * cfg->hist_bin;
			if (cfg->damping) {
				const double x = PI * rij / cfg->Rcutoff;
				damp = sin(x) / x;
			}
			for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
				const double qrij = rij * q[iq] + 0.00000001;
				I[tid * cfg->q.N + iq] += damp * rij_hist[i] * sin(qrij) / qrij;
			}
		}
	}
}

//Computes the scattering intensity (powder diffraction pattern) using the histogram of interatomic distances
void calcInt1DHist(double ** const I, unsigned long long int *rij_hist, const config * const cfg, const unsigned int * const NatomEl, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads) {
	unsigned int Nhist = cfg->Nhist/numprocs;
	unsigned int iStart = 0;
#ifdef UseMPI	
	const int reminder = int(cfg->Nhist)%numprocs;
	if (myid<reminder) Nhist++;
	const unsigned int NhistEl = (cfg->Nel*(cfg->Nel + 1)) / 2;
	iStart = cfg->Nhist/numprocs*myid;
	(myid<reminder) ? iStart += myid : iStart += (unsigned int) (reminder);
	if (myid) rij_hist = new unsigned long long int[NhistEl * Nhist];
	for (unsigned int i = 0; i < NhistEl; i++) {
		MPI_Status status;
		if (!myid) {
			for (int pid = 1; pid<numprocs; pid++){
				unsigned int ist = cfg->Nhist/numprocs*pid;
				(pid<reminder) ? ist += pid : ist += (unsigned int)(reminder);
				unsigned int Nsend = cfg->Nhist / numprocs;
				if (pid<reminder) Nsend++;
				MPI_Send(&rij_hist[i*cfg->Nhist + ist], Nsend, MPI_LONG_LONG_INT, pid, pid, MPI_COMM_WORLD);
			}
		}		
		else MPI_Recv(&rij_hist[i*Nhist], Nhist, MPI_LONG_LONG_INT, 0, myid, MPI_COMM_WORLD, &status);
	}
	double * const Iloc = new double [cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) Iloc[iq] = 0;
#endif
	unsigned int Nhist0 = Nhist;
	unsigned int Nstart = 0;
	if (!myid) Nhist0 = cfg->Nhist;
	*I = new double[cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] = 0;
	double * const Itemp = new double[cfg->q.N * NumOMPthreads];
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += Nhist0) {
			Int1DHistCore(Itemp, rij_hist + Nstart, cfg, q, Nhist, iStart, NumOMPthreads);
#ifdef UseOMP
#pragma omp parallel for num_threads(NumOMPthreads) 
			for (int iq = 0; iq < (int) cfg->q.N; iq++) {
				for (int tid = 1; tid < NumOMPthreads; tid++) Itemp[iq] += Itemp[tid * cfg->q.N + iq];
			}
#endif
			if (cfg->source == xray) {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
					Iloc[iq] += Itemp[iq] * FF[iEl][iq] * FF[jEl][iq];
#else
					(*I)[iq] += Itemp[iq] * FF[iEl][iq] * FF[jEl][iq];
#endif
				}
			}
			else {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
					Iloc[iq] += Itemp[iq] * SL[iEl] * SL[jEl];
#else
					(*I)[iq] += Itemp[iq] * SL[iEl] * SL[jEl];
#endif
				}
			}
		}
	}
	delete[] Itemp;
#ifdef UseMPI
	MPI_Reduce(Iloc, *I, cfg->q.N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	delete[] Iloc;
#endif
	if (!myid) {
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			if (cfg->source == xray) {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += NatomEl[iEl] * SQR(FF[iEl][iq]);
			}
			else {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += NatomEl[iEl] * SQR(SL[iEl]);
			}
		}
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] /= Ntot;
		if (cfg->cutoff) AddCutoff(*I, cfg, NatomEl, q, FF, SL, Ntot);
		if (cfg->PolarFactor) PolarFactor1D(*I, q, cfg);
	}
}


//template <bool cutoff, bool damping> void calcPartialRDF(double ** const PDF, const unsigned long long int * const rij_hist, const config * const cfg, unsigned int Nstart, unsigned int Ntot) {
//	const double mult = 1. / (cfg->hist_bin * Ntot);
//	if (cfg->cutoff) {
//		for (unsigned int i = 0; i < cfg->Nhist; i++) {
//			const double r = (i + 0.5) * cfg->hist_bin;
//			const double damp = 1;
//			if (damping) {
//				x = PI * r / cfg->Rcutoff;
//				damp = sin(x) / x;
//			}
//			(*PDF)[Nstart + cfg->Nhist + i] = ((double)rij_hist[Nstart + i] * mult - 4. * np.pi * SQR(r) * cfg->p0) * damp;
//		}
//	}
//	else {		
//		for (unsigned int i = 0; i < cfg->Nhist; i++) (*PDF)[Nstart + cfg->Nhist + i] = rij_hist[Nstart + i] * mult;
//	}
//}


//Depending on the computational scenario computes the scattering intensity (powder diffraction pattern) or PDF using the histogram of interatomic distances
void calcPDFandDebye(double ** const I, double ** const PDF, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const vector < vect3d <double> > * const ra, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads) {
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	unsigned long long int *rij_hist = NULL;
	calcHist(&rij_hist, cfg, ra, NatomEl, NatomEl_outer, NumOMPthreads);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	if (!myid) cout << "Histogram calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	if (((cfg->scenario == PDFonly) || (cfg->scenario == DebyePDF)) && (!myid)) {
		t1 = chrono::steady_clock::now();
		const unsigned int NPDF = (1 + (cfg->Nel*(cfg->Nel + 1)) / 2) * cfg->Nhist;
		*PDF = new double[NPDF];
		for (unsigned int i = 0; i < NPDF; i++) (*PDF)[i] = 0;
		double Faverage2 = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) 	Faverage2 += SL[iEl] * NatomEl[iEl];
		Faverage2 /= Ntot;
		Faverage2 *= Faverage2;
		unsigned int Nstart = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){	
				switch (cfg->PDFtype){
					double mult, sub;
				case typeRDF:
					mult = 1. / (cfg->hist_bin*Ntot);	
					for (unsigned int i = Nstart; i < Nstart + cfg->Nhist; i++) (*PDF)[cfg->Nhist + i] = rij_hist[i] * mult;
					break;
				case typePDF:
					mult = 0.25 / (PI*cfg->hist_bin*cfg->p0*Ntot);
					for (unsigned int i = Nstart; i < Nstart + cfg->Nhist; i++) {
						const double r = (i - Nstart + 0.5) * cfg->hist_bin;
						(*PDF)[cfg->Nhist + i] = rij_hist[i] * mult / SQR(r);
					}					
					break;
				case typeRPDF:
					mult = 1. / (cfg->hist_bin*Ntot);
					(jEl > iEl) ? sub = 8.*PI*cfg->p0*double(NatomEl[iEl]) * double(NatomEl[jEl]) / SQR(double(Ntot)) : sub = 4.*PI*cfg->p0*SQR(double(NatomEl[iEl])) / SQR(double(Ntot));
					for (unsigned int i = Nstart; i < Nstart + cfg->Nhist; i++) {
						const double r = (i - Nstart + 0.5) * cfg->hist_bin;
						(*PDF)[cfg->Nhist + i] = rij_hist[i] * mult / r - sub * r;
					}
					break;
				}
				const double multIJ = SL[iEl] * SL[jEl] / Faverage2;
				for (unsigned int i = 0; i < cfg->Nhist; i++) (*PDF)[i] += (*PDF)[cfg->Nhist + Nstart + i] * multIJ;
			}
		}
		t2 = chrono::steady_clock::now();
		if (!myid) cout << "PDF calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	}
	if ((cfg->scenario == Debye_hist) || (cfg->scenario == DebyePDF)) {
		t1 = chrono::steady_clock::now();
		calcInt1DHist(I, rij_hist, cfg, NatomEl, FF, SL, q, Ntot, NumOMPthreads);
		t2 = chrono::steady_clock::now();
		if (!myid) cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	}
	delete [] rij_hist;
}

//Computes the 2D scattering intensity in the polar coordinates (q,q_fi) of the reciprocal space
void calcInt2D(double *** const I2D, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const vector < vect3d <double> > *ra, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads) {
	const chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	const unsigned int N2D = cfg->q.N*cfg->Nfi;
#ifdef UseMPI
	double *A_im_sum = NULL, *A_real_sum = NULL;
#endif	
	if (!myid) {		
		*I2D = new double*[cfg->q.N];
		for (unsigned int iq = 0; iq < cfg->q.N; iq++){
			(*I2D)[iq] = new double[cfg->Nfi];
			for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++) (*I2D)[iq][ifi] = 0;
		}
#ifdef UseMPI
		A_im_sum = new double[N2D];
		A_real_sum = new double[N2D];
#endif
	}
	double *sintheta = new double[cfg->q.N];
	double *costheta = new double[cfg->q.N];
	double *sinfi = new double[cfg->Nfi];
	double *cosfi = new double[cfg->Nfi];
	for (unsigned int iq = 0; iq<cfg->q.N; iq++){
		sintheta[iq] = q[iq] * (cfg->lambda * 0.25 / PI);
		costheta[iq] = 1. - SQR(sintheta[iq]);
	}
	const double deltafi = 2.*PI / cfg->Nfi;
	for (unsigned int ifi = 0; ifi<cfg->Nfi; ifi++){
		cosfi[ifi] = cos(ifi*deltafi);
		sinfi[ifi] = sin(ifi*deltafi);
	}
	double dalpha = (cfg->Euler.max.x - cfg->Euler.min.x) / cfg->Euler.N.x;
	double dbeta = (cfg->Euler.max.y - cfg->Euler.min.y) / cfg->Euler.N.y;
	double dgamma = (cfg->Euler.max.z - cfg->Euler.min.z) / cfg->Euler.N.z;
	if (cfg->Euler.N.x < 2) dalpha = 0;
	if (cfg->Euler.N.y < 2) dbeta = 0;
	if (cfg->Euler.N.z < 2) dgamma = 0;
	double *A_im = new double[N2D];
	double *A_real = new double[N2D];
	for (unsigned int ia = 0; ia < cfg->Euler.N.x; ia++){
		const double alpha = cfg->Euler.min.x + (ia + 0.5)*dalpha;
		for (unsigned int ib = 0; ib < cfg->Euler.N.y; ib++){
			const double beta = cfg->Euler.min.y + (ib + 0.5)*dbeta;
			for (unsigned int ig = 0; ig < cfg->Euler.N.z; ig++){
				const double gamma = cfg->Euler.min.z + (ig + 0.5)*dgamma;
				const vect3d <double> euler(alpha,beta,gamma);
				vect3d <double> RM0, RM1, RM2;
				calcRotMatrix(&RM0, &RM1, &RM2, euler, cfg->EulerConvention);
				const vect3d <double> CS0(RM0.x, RM1.x, RM2.x), CS1(RM0.y, RM1.y, RM2.y),  CS2(RM0.z, RM1.z, RM2.z);
#ifdef UseOMP
#pragma omp parallel for num_threads(NumOMPthreads)
#endif
				for (int iq = 0; iq < (int)cfg->q.N; iq++){
					for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++){
						const unsigned int index = cfg->Nfi*iq + ifi;
						A_real[index] = 0;
						A_im[index] = 0;
						vect3d <double> qv(costheta[iq] * cosfi[ifi], costheta[iq] * sinfi[ifi], -sintheta[iq]);
						qv = qv*q[iq];
						qv.assign(qv.dot(CS0), qv.dot(CS1), qv.dot(CS2));
						for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
							double ar = 0, ai = 0;
							for (vector<vect3d <double> >::const_iterator ri = ra[iEl].begin(); ri != ra[iEl].end(); ri++){
								const double qr = qv.dot(*ri);
								ar += cos(qr);
								ai += sin(qr);
							}
							double lFF;
							(cfg->source == xray) ? lFF = FF[iEl][iq] : lFF = SL[iEl];
							A_real[index] += lFF * ar;
							A_im[index] += lFF * ai;
						}
					}
				}
#ifdef UseMPI
				MPI_Reduce(A_real, A_real_sum, N2D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				MPI_Reduce(A_im, A_im_sum, N2D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
				if (!myid) {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++){
						for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++){
							const unsigned int index = cfg->Nfi*iq + ifi;
#ifdef UseMPI
							(*I2D)[iq][ifi] += (SQR(A_real_sum[index]) + SQR(A_im_sum[index]));
#else
							(*I2D)[iq][ifi] += (SQR(A_real[index]) + SQR(A_im[index]));
#endif
						}
					}
				}
			}
		}
	}
	delete[] sintheta;
	delete[] costheta;
	delete[] sinfi;
	delete[] cosfi;
	delete[] A_im;
	delete[] A_real;
	if (!myid) {
#ifdef UseMPI
		delete[] A_im_sum;
		delete[] A_real_sum;
#endif
		const double norm = 1. / (Ntot*cfg->Euler.N.x*cfg->Euler.N.y*cfg->Euler.N.z);
		for (unsigned int iq = 0; iq < cfg->q.N; iq++){
			for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++) (*I2D)[iq][ifi] *= norm;
		}
		if (cfg->PolarFactor) PolarFactor2D(*I2D, q, cfg);
		*I = new double[cfg->q.N];
		for (unsigned int iq = 0; iq < cfg->q.N; iq++){
			(*I)[iq] = 0;
			for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++)	(*I)[iq] += (*I2D)[iq][ifi];
			(*I)[iq] /= cfg->Nfi;
		}
	}
	const chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	if (!myid) cout << "2D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}
#endif
