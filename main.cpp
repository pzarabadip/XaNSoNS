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

//main file
#include "config.h"
#include "block.h"
#include <chrono>
#ifdef UseMPI
#include "mpi.h"
#endif
#ifdef UseOMP
#include <omp.h>
#endif
#ifdef UseOCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

struct float4;
int myid, numprocs;

int ReadConfig(config * const cfg, double ** const q, block ** const Block, const char * const FileName, map<string, unsigned int> * const ID, map<unsigned int, string> * const EN, vector<double> * const SL, vector<double *> * const FF);//ReadXML.cpp
unsigned int CalcAndPrintAtoms(config * const cfg, block * const Block, vector < vect3d <double> > ** const ra, unsigned int ** const NatomEl, unsigned int ** const NatomEl_outer, const map <string, unsigned int> ID, const map <unsigned int, string> EN);//CalcFunctions.cpp
int printPDF(const double * const PDF, const config * const cfg);//IO.cpp
int printPartialPDF(const double * const PDF, const config * const cfg, const map <unsigned int, string> EN);//IO.cpp
int printI(const double * const I, const double * const q, const config * const cfg);//IO.cpp
int printPartialI(const double * const I, const double * const q, const config * const cfg);//IO.cpp
int printI2(const double * const * const I, const config * const cfg);//IO.cpp
#ifdef UseCUDA
void calcIntDebyeCuda(const int DeviceNUM, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const float4 * const ra, const float * const dFF, const vector<double> SL, const float * const dq);//CalcFunctionsCUDA.cu
void calcIntPartialDebyeCuda(const int DeviceNUM, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const float4 * const ra, const float * const dFF, const vector <double> SL, const float * const dq, const block * const Block);//CalcFunctionsCUDA.cu
void calcPDFandDebyeCuda(const int DeviceNUM, double ** const I, double ** const PDF, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const float4 * const ra, const float * const dFF, const vector<double> SL, const float * const dq);//CalcFunctionsCUDA.cu
void calcInt2DCuda(const int DeviceNUM, double *** const I2D, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const float4 * const ra, const float * const dFF, const vector<double> SL, const float * const dq);//CalcFunctionsCUDA.cu
void delDataFromDevice(float4 * const ra, float * const dFF, float * const dq, const unsigned int Nel);//CalcFunctionsCUDA.cu
void dataCopyCUDA(const double *const q, const config * const cfg, const vector < vect3d <double> > * const ra, float4 ** const dra, float ** const dFF, float ** const dq, const vector <double*> FF);//CalcFunctionsCUDA.cu
int SetDeviceCuda(int * const DeviceNUM);//CalcFunctionsCUDA.cu
#elif UseOCL
int GetOpenCLinfoFromInitDataXML(string *GPUtype, int *DeviceNUM); //ReadXML.cpp (for BOINC)
int GetOpenCLPlatfromNum(const string GPUtype, const int DeviceNUM); //CalcFunctionsOCL.cpp (for BOINC)
int SetDeviceOCL(cl_device_id * const OCLdevice, int DeviceNUM, int PlatformNUM); //CalcFunctionsOCL.cpp
int createContextOCL(cl_context * const OCLcontext, cl_program * const OCLprogram, const cl_device_id OCLdevice, const char * const argv0, const unsigned int scenario); //CalcFunctionsOCL.cpp
void dataCopyOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const double * const q, const config *const cfg, const vector < vect3d <double> > *const ra, cl_mem * const dra, cl_mem * const dFF, cl_mem * const dq, const vector <double*> FF); //CalcFunctionsOCL.cpp
void delDataFromDeviceOCL(const cl_context OCLcontext, const cl_program OCLprogram, const cl_mem ra, const cl_mem dFF, const cl_mem dq, const unsigned int Nel); //CalcFunctionsOCL.cpp
void calcInt2D_OCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double *** const I2D, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq); //CalcFunctionsOCL.cpp
void calcPDFandDebyeOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, double ** const PDF, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq); //CalcFunctionsOCL.cpp
void calcIntDebyeOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq); //CalcFunctionsOCL.cpp
void calcIntPartialDebyeOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq, const block * const Block); //CalcFunctionsOCL.cpp
#else
void calcInt2D(double *** const I2D, double ** const I, const config * const cfg, const unsigned int * const NatomEl, const vector < vect3d <double> > *ra, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads);//CalcFunctions.cpp
void calcIntDebye(double ** const I, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const vector < vect3d <double> > * const ra, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads);//CalcFunctions.cpp
void calcIntPartialDebye(double ** const I, const config * const cfg, const unsigned int * const NatomEl, const vector < vect3d <double> > * const ra, const  vector <double*> FF, const vector<double> SL, const double * const q, const block * const Block, const unsigned int Ntot, const int NumOMPthreads);//CalcFunctions.cpp
void calcPDFandDebye(double ** const I, double ** const PDF, const config * const cfg, const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const vector < vect3d <double> > * const ra, const vector <double*> FF, const vector<double> SL, const double * const q, const unsigned int Ntot, const int NumOMPthreads);//CalcFunctions.cpp
#endif

int FinalizeIfError(const unsigned int source, double * const q, block * const Block, const vector<double *> * const FF) {
	delete[] q;
	delete[] Block;
	if (source == xray)	for (unsigned int i = 0; i<FF->size(); i++)	delete[] (*FF)[i];
	cout << "\nTerminating with error. See stdout for the details" << endl;
	return -1;
}

int main(int argc, char *argv[]){
#ifdef UseMPI
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#else
	myid = 0;
	numprocs = 1;
#endif
	if (argc < 2) {
		if (!myid)	cout << "Error: Start XML filename not specified." << endl;
#ifdef UseMPI
		MPI_Finalize();
#endif
		return -1;
	}
	int NumOMPthreads = 1;
#ifdef UseOMP
	if (argc > 2) NumOMPthreads = atoi(argv[2]);
	else NumOMPthreads = omp_get_max_threads();
	if (!myid)	cout << "Number of OpenMP threads is set to " << NumOMPthreads << endl;
#endif
	config cfg;
	block *Block = NULL;
	double *q = NULL;
	map<string, unsigned int> ID;
	map<unsigned int, string> EN;
	vector<double> SL;
	vector<double *> FF;
	int error = ReadConfig(&cfg, &q, &Block, argv[1], &ID, &EN, &SL, &FF);//reading the calculation parameters, from-factors, etc., creating structural blocks
	if (error) {
#ifdef UseMPI
		MPI_Finalize();
#endif
		return FinalizeIfError(cfg.source, q, Block, &FF);
	}
	chrono::steady_clock::time_point t1;
	if (!myid) 	t1 = chrono::steady_clock::now();
	unsigned int *NatomEl = NULL, *NatomEl_outer = NULL;
	vector < vect3d <double> > *ra = NULL;
	const unsigned int Ntot = CalcAndPrintAtoms(&cfg, Block, &ra, &NatomEl, &NatomEl_outer, ID, EN);//creating the atomic ensemble
#ifdef UseCUDA
	int DeviceNUM = -1;
	float *dq = NULL, *dFF = NULL;
	float4 *dra = NULL;//using float4 structure for three coordinates to assure for global memory access coalescing
	if (argc > 2) DeviceNUM = atoi(argv[2]);
	error = SetDeviceCuda(&DeviceNUM);//queries CUDA devices, changes DeviceNUM to proper device number if required
	if (error) return FinalizeIfError(cfg.source, q, Block, &FF);
	dataCopyCUDA(q, &cfg, ra, &dra, &dFF, &dq, FF);//copying all the necessary data to the device memory
#elif UseOCL
	int DeviceNUM = -1, PlatformNUM=-1;
	cl_mem dq = NULL, dFF = NULL, dra = NULL;
	cl_context OCLcontext;
	cl_device_id OCLdevice;
	cl_program OCLprogram = NULL;
	if (argc > 2) DeviceNUM = atoi(argv[2]);
	if (argc > 3) PlatformNUM = atoi(argv[3]);
	if ((DeviceNUM < 0) && (PlatformNUM < 0)) {//trying to get the OpenCL info from init_data.xml (for compatibility with BOINC client)
		//Note, if something goes wrong here, the programm may start the calculation on the GPU device it is not allowed to...
		string GPUtype;
		error=GetOpenCLinfoFromInitDataXML(&GPUtype,&DeviceNUM);
		if (!error) PlatformNUM = GetOpenCLPlatfromNum(GPUtype, DeviceNUM);
	}
	error = SetDeviceOCL(&OCLdevice, DeviceNUM, PlatformNUM);
	if (error) return FinalizeIfError(cfg.source, q, Block, &FF);
	error = createContextOCL(&OCLcontext, &OCLprogram, OCLdevice, argv[0], cfg.scenario);//copying all the necessary data to the device memory 
	if (error) return FinalizeIfError(cfg.source, q, Block, &FF);
	dataCopyOCL(OCLcontext, OCLdevice, q, &cfg, ra, &dra, &dFF, &dq, FF);
#endif
	double **I2D = NULL, *I = NULL, *PDF = NULL;
	switch (cfg.scenario) {
		case s2D://calculating the 2D scatteing intensity
#ifdef UseCUDA
			calcInt2DCuda(DeviceNUM, &I2D, &I, &cfg, NatomEl, dra, dFF, SL, dq);
#elif UseOCL
			calcInt2D_OCL(OCLcontext, OCLdevice, OCLprogram, &I2D, &I, &cfg, NatomEl, dra, dFF, SL, dq);
#else
			calcInt2D(&I2D, &I, &cfg, NatomEl, ra, FF, SL, q, Ntot,NumOMPthreads);
#endif
			if (!myid) {
				chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2-t1).count() << " s" << endl;
				printI2(I2D, &cfg);
				printI(I, q, &cfg);
			}
			break;
		case Debye://calculating the 1D scatteing intensity using the Debye formula
#ifdef UseCUDA
			if (cfg.calcPartialIntensity) calcIntPartialDebyeCuda(DeviceNUM, &I, &cfg, NatomEl, dra, dFF, SL, dq, Block);
			else calcIntDebyeCuda(DeviceNUM, &I, &cfg, NatomEl, NatomEl_outer, dra, dFF, SL, dq);
#elif UseOCL
			if (cfg.calcPartialIntensity) calcIntPartialDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &cfg, NatomEl, dra, dFF, SL, dq, Block);
			else calcIntDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &cfg, NatomEl, NatomEl_outer, dra, dFF, SL, dq);
#else
			if (cfg.calcPartialIntensity) calcIntPartialDebye(&I, &cfg, NatomEl, ra, FF, SL, q, Block, Ntot,NumOMPthreads);
			else calcIntDebye(&I, &cfg, NatomEl, NatomEl_outer, ra, FF, SL, q, Ntot, NumOMPthreads);
#endif
			if (!myid) {
				chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2-t1).count() << " s" << endl;
				if (cfg.calcPartialIntensity) printPartialI(I + cfg.q.N, q, &cfg);
				printI(I, q, &cfg);
			}
			break;
		case Debye_hist://calculating the 1D scatteing intensity using the histogram of interatomic distances
#ifdef UseCUDA
			calcPDFandDebyeCuda(DeviceNUM, &I, &PDF, &cfg, NatomEl, NatomEl_outer, dra, dFF, SL, dq);
#elif UseOCL
			calcPDFandDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &PDF, &cfg, NatomEl, NatomEl_outer, dra, dFF, SL, dq);
#else
			calcPDFandDebye(&I, &PDF, &cfg, NatomEl, NatomEl_outer, ra, FF, SL, q, Ntot, NumOMPthreads);
#endif
			if (!myid) {
				chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2-t1).count() << " s" << endl;
				printI(I, q, &cfg);
			}
			break;
		case PDFonly://calculating the PDF
#ifdef UseCUDA
			calcPDFandDebyeCuda(DeviceNUM, &I, &PDF, &cfg, NatomEl, NatomEl_outer, dra, dFF, SL, dq);
#elif UseOCL
			calcPDFandDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &PDF, &cfg, NatomEl, NatomEl_outer, dra, dFF, SL, dq);
#else
			calcPDFandDebye(&I, &PDF, &cfg, NatomEl, NatomEl_outer, ra, FF, SL, q, Ntot, NumOMPthreads);
#endif
			if (!myid) {
				chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2-t1).count() << " s" << endl;
				if (cfg.PrintPartialPDF) printPartialPDF(PDF + cfg.Nhist, &cfg, EN);
				printPDF(PDF, &cfg);
			}
			break;
		case DebyePDF://calculating the PDF and 1D scatteing intensity using the histogram of interatomic distances
#ifdef UseCUDA
			calcPDFandDebyeCuda(DeviceNUM, &I, &PDF, &cfg, NatomEl, NatomEl_outer, dra, dFF, SL, dq);
#elif UseOCL
			calcPDFandDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &PDF, &cfg, NatomEl, NatomEl_outer, dra, dFF, SL, dq);
#else
			calcPDFandDebye(&I, &PDF, &cfg, NatomEl, NatomEl_outer, ra, FF, SL, q, Ntot, NumOMPthreads);
#endif
			if (!myid) {
				chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
				printI(I, q, &cfg);
				if (cfg.PrintPartialPDF) printPartialPDF(PDF + cfg.Nhist, &cfg, EN);
				printPDF(PDF, &cfg);
			}
			break;
	}
#ifdef UseMPI
	MPI_Finalize();
#elif UseCUDA
	delDataFromDevice(dra, dFF, dq, cfg.Nel);//deleting the data from the device memory
#elif UseOCL
	delDataFromDeviceOCL(OCLcontext, OCLprogram, dra, dFF, dq, cfg.Nel);
#endif
	if (I2D != NULL) {
		for (unsigned int iq = 0; iq < cfg.q.N; iq++) delete[] I2D[iq];
		delete[] I2D;
	}
	delete[] I;
	delete[] q;
	delete[] PDF;
	if (cfg.source == xray)	for (unsigned int i=0;i<FF.size();i++)	delete[] FF[i];
	delete[] Block;
	delete[] ra;
	delete[] NatomEl;
	delete[] NatomEl_outer;
	return 0;
}
