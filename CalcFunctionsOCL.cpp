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

#include "typedefs.h"
#ifdef UseOCL
#include "config.h"
#include "block.h"
#include <chrono>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//Nvidia specific OpenCL extensions
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
/* cl_nv_device_attribute_query extension - no extension #define since it has no functions */
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV            0x4005
#endif

//AMD specific OpenCL extensions
#ifndef CL_DEVICE_WAVEFRONT_WIDTH_AMD 
/* cl_amd_device_attribute_query extension - no extension #define since it has no functions */
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD               0x4043
#endif

//Calculates rotational matrix (from CalcFunctions.cpp)
void calcRotMatrix(vect3d <double> * const RM0, vect3d <double> * const RM1, vect3d <double> * const RM2, const vect3d <double> euler, const unsigned int convention);

/**
	Organazies the computations of the 2D scattering intensity in the polar coordinates (q,q_fi) of the reciprocal space with OpenCL

	@param OCLcontext OpenCL context
	@param OCLdevice  OpenCL device
	@param OCLprogram OpenCL program object for OCLcontext
	@param ***I2D     2D scattering intensity array (host). The memory is allocated inside the function.
	@param **I        1D (averaged over the polar angle) scattering intensity array (host). The memory is allocated inside the function.
	@param *cfg       Parameters of simulation
	@param *NatomEl	  Array containing the total number of atoms of each chemical element (host)
	@param ra         Atomic coordinate array (device)
	@param dFF        X-ray atomic form-factors for all chemical elements (device)
	@param SL         Array of neutron scattering lengths for all chemical elements
	@param dq         Scattering vector magnitude array (device)
*/
void calcInt2D_OCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double *** const I2D, double ** const I, const config * const cfg, \
	const unsigned int * const NatomEl, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq);

/**
	Organazies the computations of the histogram of interatomic distances with OpenCL

	@param OCLcontext      OpenCL context
	@param OCLdevice       OpenCL device
	@param OCLprogram      OpenCL program object for OCLcontext
	@param *rij_hist       Histogram of interatomic distances (device). The memory is allocated inside the function.
	@param ra              Atomic coordinate array (device)
	@param *NatomEl        Array containing the total number of atoms of each chemical element (host)
	@param *NatomEl_outer  Array containing the number of atoms of each chemical element including the atoms in the outer sphere (only if cfg.cutoff is True)
	@param *cfg            Parameters of simulation
*/
void calcHistOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, cl_mem * const rij_hist, const cl_mem ra, \
	const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const config * const cfg);

/**
Adds the average density correction to the scattering intensity when the cut-off is enabled (cfg.cutoff == true)

	@param OCLcontext      OpenCL context
	@param OCLdevice       OpenCL device
	@param OCLprogram      OpenCL program object for OCLcontext
	@param dI              Scattering intensity array (device)
	@param *NatomEl        Array containing the total number of atoms of each chemical element (host)
	@param *cfg            Parameters of simulation
	@param dFF             X-ray atomic form-factor array for all chemical elements (device)
	@param SL              Array of neutron scattering lengths for all chemical elements
	@param dq              Scattering vector magnitude array (device)
	@param Ntot            Total number of atoms in the nanoparticle
*/
void AddCutoffOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, const cl_mem dI, const unsigned int * const NatomEl, const config * const cfg, \
	const cl_mem dFF, const vector<double> SL, const cl_mem dq, const unsigned int Ntot);

/**
	Organazies the computations of the scattering intensity (powder diffraction pattern) using the histogram of interatomic distances with OpenCL

	@param OCLcontext      OpenCL context
	@param OCLdevice       OpenCL device
	@param OCLprogram      OpenCL program object for OCLcontext
	@param **I             Scattering intensity array (host). The memory is allocated inside the function
	@param rij_hist        Histogram of interatomic distances (device).
	@param *NatomEl        Array containing the total number of atoms of each chemical element (host)
	@param *cfg            Parameters of simulation
	@param dFF             X-ray atomic form-factor arrays for all chemical elements (device)
	@param SL              Array of neutron scattering lengths for all chemical elements
	@param dq              Scattering vector magnitude array (device)
	@param Ntot            Total number of atoms in the nanoparticle
*/
void calcInt1DHistOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, const cl_mem rij_hist, \
	const unsigned int * const NatomEl, const config * const cfg, const cl_mem dFF, const vector<double> SL, const cl_mem dq, const unsigned int Ntot);

/**
	Depending on the computational scenario organazies the computations of the scattering intensity (powder diffraction pattern) or PDF using the histogram of interatomic distances with OpenCL

	@param OCLcontext      OpenCL context
	@param OCLdevice       OpenCL device
	@param OCLprogram      OpenCL program object for OCLcontext
	@param **I             Scattering intensity array (host). The memory is allocated inside the function.
	@param **PDF           PDF array (host). The memory is allocated inside the function.
	@param *cfg            Parameters of simulation
	@param *NatomEl        Array containing the total number of atoms of each chemical element (host)
	@param *NatomEl_outer  Array containing the number of atoms of each chemical element including the atoms in the outer sphere (only if cfg.cutoff is True)
	@param ra              Atomic coordinate array (device)
	@param dFF             X-ray atomic form-factors for all chemical elements (device)
	@param SL              Array of neutron scattering lengths for all chemical elements
	@param dq              Scattering vector magnitude array (device)
*/
void calcPDFandDebyeOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, double ** const PDF, const config * const cfg, \
	const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq);

/**
	Organazies the computations of the scattering intensity (powder diffraction pattern) using the original Debye equation (without the histogram approximation) with OpenCL

	@param OCLcontext      OpenCL context
	@param OCLdevice       OpenCL device
	@param OCLprogram      OpenCL program object for OCLcontext
	@param **I             Scattering intensity array (host). The memory is allocated inside the function.
	@param *cfg            Parameters of simulation
	@param *NatomEl        Array containing the total number of atoms of each chemical element (host)
	@param *NatomEl_outer  Array containing the number of atoms of each chemical element including the atoms in the outer sphere (only if cfg.cutoff is True)
	@param ra              Atomic coordinate array (device)
	@param dFF             X-ray atomic form-factors for all chemical elements (device)
	@param SL              Array of neutron scattering lengths for all chemical elements
	@param dq              Scattering vector magnitude array (device)
*/
void calcIntDebyeOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, const config * const cfg, \
	const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq);

/**
	Organazies the computations of the scattering intensity (powder diffraction pattern + partial intensities) using the original Debye equation (without the histogram approximation) with OpenCL

	@param OCLcontext  OpenCL context
	@param OCLdevice   OpenCL device
	@param OCLprogram  OpenCL program object for OCLcontext
	@param **I         Partial + total scattering intensity array (host). The memory is allocated inside the function.
	@param *cfg        Parameters of simulation
	@param *NatomEl    Array containing the total number of atoms of each chemical element (host)
	@param ra          Atomic coordinate array (device)
	@param dFF         X-ray atomic form-factors for all chemical elements (device)
	@param SL          Array of neutron scattering lengths for all chemical elements
	@param dq          Scattering vector magnitude array (device)
	@param *Block      Array of the structural blocks
*/
void calcIntPartialDebyeOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, const config * const cfg, const unsigned int * const NatomEl, \
	const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq, const block * const Block);

/**
	Returns BOINC-compatible device vendor name: NVIDIA, intel_gpu or ATI

	@param device_id OpenCL device ID
*/
const char* GetVendor(const cl_device_id device_id);

/**
	Returns OpenCL platfrom number for the specified vecndor name and device number. 
	Checks if the device with the given number is present in the OpenCL platfrom. Returns -1 if error

	@param GPUtype   OpenCL device vendor
	@param DeviceNUM OpenCL device number
*/
int GetOpenCLPlatfromNum(const string GPUtype, const int DeviceNUM);

/**
	Returns the theoretical peak performance of the OpenCL device. Return 0 if the vendor is unsupported.

	@param OCLdevice OpenCL device
	@param show      If True, show the device information on screen
*/
unsigned int GetGFLOPS(const cl_device_id OCLdevice, const bool show);

/**
	Queries all OpenCL devices in all OpenCL platforms. Checks and sets the OpenCL device
	Returns 0 if OK and -1 if no OpenCL devices found

	@param *OCLdevice  OpenCL device
	@param DeviceNUM   OpenCL device number (-1 if default)
	@param PlatformNUM OpenCL platform number (-1 if default)
*/
int SetDeviceOCL(cl_device_id * const OCLdevice, int DeviceNUM, int PlatformNUM);

/**
	Creates OpenCL context and builds OpenCL kernels

	@param *OCLcontext  OpenCL context
	@param *OCLprogram  OpenCL program object
	@param OCLdevice    OpenCL device
	@param *argv0       Absolute path to the executable (first argument in the argv[]). It is used to get the path to the .cl files
	@param scenario     Computational scenario
*/
int createContextOCL(cl_context * const OCLcontext, cl_program * const OCLprogram, const cl_device_id OCLdevice, const char * const argv0, const unsigned int scenario);

/**
	Copies the atomic coordinates (ra), scattering vector magnitude (q) and the x-ray atomic form-factors (FF) to the device memory

	@param OCLcontext  OpenCL context
	@param OCLdevice   OpenCL device
	@param *q          Scattering vector magnitude (host)
	@param *cfg        Parameters of simulation
	@param *ra         Atomic coordinates (host)
	@param *dra        Atomic coordinates (device). The memory is allocated inside the function
	@param *dFF        X-ray atomic form-factors (device). The memory is allocated inside the function
	@param *dq         Scattering vector magnitude (device). The memory is allocated inside the function
	@param FF          X-ray atomic form-factors (host)
	@param Ntot        Total number of atoms in the nanoparticle
*/
void dataCopyOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const double * const q, const config *const cfg, const vector < vect3d <double> > *const ra, \
	cl_mem * const dra, cl_mem * const dFF, cl_mem * const dq, const vector <double*> FF);

/**
	Deletes the atomic coordinates (ra), scattering vector magnitude (dq), the x-ray atomic form-factors (dFF) from the device memory and frees the OpenCL context and program object

	@param OCLcontext  OpenCL context
	@param OCLprogram  OpenCL program object
	@param ra          Atomic coordinates (device)
	@param dFF         X-ray atomic form-factors (device)
	@param dq          Scattering vector magnitude (device)
	@param Nel         Total number of different chemical elements in the nanoparticle
*/
void delDataFromDeviceOCL(const cl_context OCLcontext, const cl_program OCLprogram, const cl_mem ra, const cl_mem dFF, const cl_mem dq, const unsigned int Nel);

//Returns BOINC-compatible device vendor name: NVIDIA, intel_gpu or ATI
const char* GetVendor(const cl_device_id device_id) {
	size_t info_size;
	clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	char *vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("nvidia") != string::npos) return "NVIDIA";
	if (vendor_str.find("intel") != string::npos) return "intel_gpu";
	if ((vendor_str.find("amd") != string::npos) || (vendor_str.find("advanced") != string::npos)) return "ATI";
	return "";
}

//Returns OpenCL platfrom number for the specified vecndor name and device number.
//Checks if the device with the given number is present in the OpenCL platfrom.Returns - 1 if error
int GetOpenCLPlatfromNum(const string GPUtype, const int DeviceNUM) {
	if (DeviceNUM < 0) return -1;
	if (GPUtype.empty()) return -1;
	cl_uint ret_num_platforms;
	clGetPlatformIDs(0, NULL, &ret_num_platforms);
	if (!ret_num_platforms) {
		cout << "Error: No OpenCL platfroms found." << endl;
		return -1;
	}
	cl_platform_id *platform_id = new cl_platform_id[ret_num_platforms];
	clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
	for (cl_uint i = 0; i < ret_num_platforms; i++) {
		cl_uint ret_num_devices;
		clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &ret_num_devices);
		if (DeviceNUM >= (int)ret_num_devices) {
			continue;
		}
		cl_device_id *device_id = new cl_device_id[ret_num_devices];
		clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, ret_num_devices, device_id, NULL);
		const string gpu_type(GetVendor(device_id[DeviceNUM]));
		delete[] device_id;
		if (gpu_type == GPUtype) return i;
	}
	delete[] platform_id;
	return -1;
}

//Returns the theoretical peak performance of the OpenCL device. Return 0 if the vendor is unsupported.
unsigned int GetGFLOPS(const cl_device_id OCLdevice, const bool show = false){
	cl_uint CU, GPUclock;	
	clGetDeviceInfo(OCLdevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &CU, NULL);
	clGetDeviceInfo(OCLdevice, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &GPUclock, NULL);
	size_t info_size;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_NAME, 0, NULL, &info_size);
	char *name = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_NAME, info_size, name, NULL);
	if (show) {
		cout << "GPU: " << name << "\n";
		cout << "Number of compute units: " << CU << "\n";
		cout << "GPU clock rate: " << GPUclock << " MHz\n";
	}
	delete[] name;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	char *vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	unsigned int GFLOPS = 0;
	if (vendor_str.find("nvidia") != string::npos) {
		cl_uint CCmaj = 0;
		cl_uint CCmin = 0;
		clGetDeviceInfo(OCLdevice, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &CCmaj, NULL);
		clGetDeviceInfo(OCLdevice, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &CCmin, NULL);
		const unsigned int cc = CCmaj * 10 + CCmin; //compute capability
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
		GFLOPS = CU * ALUlanes * 2 * GPUclock / 1000;
	}
	else if (vendor_str.find("intel") != string::npos) {
		GFLOPS = CU * 16 * GPUclock / 1000;
	}
	else if ((vendor_str.find("amd") != string::npos) || (vendor_str.find("advanced") != string::npos)) {
		cl_uint WW = 0;
		const cl_int ret = clGetDeviceInfo(OCLdevice, CL_DEVICE_WAVEFRONT_WIDTH_AMD, sizeof(cl_uint), &WW, NULL);
		if ((ret != CL_SUCCESS) || (!WW)) WW = 64;//if CL_DEVICE_WAVEFRONT_WIDTH_AMD is not implemented
		GFLOPS = CU * WW * 2 * GPUclock / 1000;
	}
	else {
		cout << "Error. Unsupported device vendor." << endl;
		return 0;
	}
	if (show) 	cout << "Theoretical peak performance: " << GFLOPS << " GFLOPs\n" << endl;
	return GFLOPS;
}

//Queries all OpenCL devices in all OpenCL platforms. Checks and sets the OpenCL device
//Returns 0 if OK and - 1 if no OpenCL devices found
int SetDeviceOCL(cl_device_id * const OCLdevice, int DeviceNUM, int PlatformNUM){
	cl_uint ret_num_platforms;
	clGetPlatformIDs(0, NULL, &ret_num_platforms);
	if (!ret_num_platforms) {
		cout << "Error: No OpenCL platfroms found." << endl;
		return -1;
	}
	cl_platform_id *platform_id = new cl_platform_id[ret_num_platforms];
	cl_device_id **device_id = new cl_device_id*[ret_num_platforms];
	cl_uint *ret_num_devices = new cl_uint[ret_num_platforms];
	clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
	char **p_name = new char*[ret_num_platforms];
	for (cl_uint i = 0; i < ret_num_platforms; i++) {
		size_t info_size = 0;
		clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 0, NULL, &info_size);
		p_name[i] = new char[info_size / sizeof(char)];
		clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, info_size, p_name[i], NULL);
		clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &ret_num_devices[i]);
		if (!ret_num_devices[i]) {
			device_id[i] = NULL;
			continue;
		}
		device_id[i] = new cl_device_id[ret_num_devices[i]];
		clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, ret_num_devices[i], device_id[i], NULL);
	}
	if (PlatformNUM >= (int)ret_num_platforms) {
		cout << "Error: Unable to set the OpenCL platfrom " << PlatformNUM << ".\n";
		cout << "The total number of available platforms is " << ret_num_platforms << ". Will use the default platform.\n" << endl;
		PlatformNUM = -1;
	}
	if (PlatformNUM > -1) {
		cout << "Selected OpenCL platform:\n" << p_name[PlatformNUM] << endl;
		if (DeviceNUM >= (int)ret_num_devices[PlatformNUM]) {
			cout << "Error: Unable to set the OpenCL device " << DeviceNUM << ".\n";
			cout << "The total number of the devices for the selected platfrom is " << ret_num_devices[PlatformNUM] << ". Will use the fastest device available.\n" << endl;
			DeviceNUM = -1;
		}
		if (DeviceNUM > -1) {
			*OCLdevice = device_id[PlatformNUM][DeviceNUM];
			cout << "Selected OpenCL device:\n";
			GetGFLOPS(*OCLdevice, true);
			for (cl_uint i = 0; i < ret_num_platforms; i++) { delete[] p_name[i]; delete[] device_id[i]; }
			delete[] p_name; delete[] device_id; delete[] platform_id; delete[] ret_num_devices;
			return 0;
		}
		cout << "Platform contains the following OpenCL devices:\n";
		unsigned int MaxGFOLPS = 0;
		for (cl_uint i = 0; i< ret_num_devices[PlatformNUM]; i++) {
			cout << "Device " << i << ":\n";
			const unsigned int GFOLPS = GetGFLOPS(device_id[PlatformNUM][i], true);
			if (GFOLPS > MaxGFOLPS) {
				MaxGFOLPS = GFOLPS;
				DeviceNUM = (int)i;
			}
		}
		*OCLdevice = device_id[PlatformNUM][DeviceNUM];
		cout << "Will use device " << DeviceNUM << "." << endl;
		return 0;
	}	
	if (DeviceNUM > -1) {
		cout << "Device " << DeviceNUM << " is found in the following OpenCL platforms:\n";
		unsigned int MaxGFOLPS = 0;
		for (cl_uint i = 0; i < ret_num_platforms; i++) {
			if (DeviceNUM >= (int)ret_num_devices[i]) continue;
			cout << "Platfrom " << i << ": " << p_name[i] << "\n";
			const unsigned int GFOLPS = GetGFLOPS(device_id[i][DeviceNUM], false);
			if (GFOLPS > MaxGFOLPS) {
				MaxGFOLPS = GFOLPS;
				PlatformNUM = (int)i;
			}
		}
		if (MaxGFOLPS) {
			*OCLdevice = device_id[PlatformNUM][DeviceNUM];
			cout << "Will use OpenCL platform " << PlatformNUM << "." << endl;
			cout << "Selected OpenCL device:\n";
			GetGFLOPS(*OCLdevice, true);
			for (cl_uint i = 0; i < ret_num_platforms; i++) { delete[] p_name[i]; delete[] device_id[i]; }
			delete[] p_name; delete[] device_id; delete[] platform_id; delete[] ret_num_devices;
			return 0;
		}
		cout << "Error: No OpenCL platfroms found containing the device with number " << DeviceNUM << ". Will use the fastest device available.\n" << endl;
		DeviceNUM = -1;
	}
	cout << "The following OpenCL platforms are found:\n";
	unsigned int MaxGFOLPS = 0;
	for (cl_uint i = 0; i < ret_num_platforms; i++) {
		cout << "Platform " << i << ": " << p_name[i] << "\n";
		cout << "Platform contains the following OpenCL devices:\n";
		for (cl_uint j = 0; j < ret_num_devices[i]; j++) {
			cout << "Device " << j << ":\n";
			const unsigned int GFOLPS = GetGFLOPS(device_id[i][j], true);
			if (GFOLPS > MaxGFOLPS) {
				MaxGFOLPS = GFOLPS;
				DeviceNUM = (int)j;
				PlatformNUM = (int)i;
			}
		}
	}
	if (MaxGFOLPS) {
		*OCLdevice = device_id[PlatformNUM][DeviceNUM];
		cout << "Will use device " << DeviceNUM << " in the platform " << PlatformNUM << "." << endl;
		for (cl_uint i = 0; i < ret_num_platforms; i++) { delete[] p_name[i]; delete[] device_id[i]; }
		delete[] p_name; delete[] device_id; delete[] platform_id; delete[] ret_num_devices;
		return 0;
	}
	cout << "Error: No OpenCL devices found." << endl;
	for (cl_uint i = 0; i < ret_num_platforms; i++) { delete[] p_name[i]; delete[] device_id[i]; }
	delete[] p_name; delete[] device_id; delete[] platform_id; delete[] ret_num_devices;
	return -1;
}

//Creates OpenCL context and builds OpenCL kernels
int createContextOCL(cl_context * const OCLcontext, cl_program * const OCLprogram, const cl_device_id OCLdevice, const char * const argv0, const unsigned int scenario){		
	string path2kernels(argv0);
	size_t pos = path2kernels.rfind("/");
	if (pos == string::npos) pos = path2kernels.rfind("\\");
	switch (scenario){
	case s2D:
		path2kernels = path2kernels.replace(pos + 1, string::npos, "kernels2D.cl");
		break;
	case Debye:
		path2kernels = path2kernels.replace(pos + 1, string::npos, "kernelsDebye.cl");
		break;
	case Debye_hist:
	case PDFonly:
	case DebyePDF:
		path2kernels = path2kernels.replace(pos + 1, string::npos, "kernelsPDF.cl");
		break;
	}
	char *source_str = NULL;
	size_t source_size;
	ifstream is(path2kernels, ifstream::binary);
	if (is) {
		is.seekg(0, is.end);
		int length = (int)is.tellg();
		is.seekg(0, is.beg);
		source_str = new char[length];
		is.read(source_str, length);
		if (!is) cout << "error" << endl;
		source_size = sizeof(char)*length;
	}
	else {
		cout << "Failed to load OpenCL kernels from file.\n" << endl;
		exit(1);
	}
	size_t info_size = 0;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_EXTENSIONS, 0, NULL, &info_size);
	char *info = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_EXTENSIONS, info_size, info, NULL);
	const string extensions(info);
	delete[] info;
	*OCLcontext = clCreateContext(NULL, 1, &OCLdevice, NULL, NULL, NULL);
	cl_int err;
	*OCLprogram = clCreateProgramWithSource(*OCLcontext, 1, (const char **)&source_str, &source_size, &err);
	if ((extensions.find("cl_khr_int64_base_atomics") != string::npos) || (extensions.find("cl_nv_") != string::npos)) err = clBuildProgram(*OCLprogram, 1, &OCLdevice, "-cl-fast-relaxed-math", NULL, NULL);
	else err = clBuildProgram(*OCLprogram, 1, &OCLdevice, "-cl-fast-relaxed-math -DCustomInt64atomics", NULL, NULL);
	if (err) {
		size_t lengthErr;
		clGetProgramBuildInfo(*OCLprogram, OCLdevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &lengthErr);
		char *buffer = new char[lengthErr];
		clGetProgramBuildInfo(*OCLprogram, OCLdevice, CL_PROGRAM_BUILD_LOG, lengthErr, buffer, NULL);
		cout << "--- Build log ---\n " << buffer << endl;
		delete[] buffer;
	}
	delete[] source_str;
	return 0;
}

//Copies the atomic coordinates (ra), scattering vector magnitude (q) and the x-ray atomic form-factors (FF) to the device memory
void dataCopyOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const double * const q, const config *const cfg, const vector < vect3d <double> > *const ra, \
				cl_mem * const dra, cl_mem * const dFF, cl_mem * const dq, const vector <double*> FF){
	size_t info_size;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	char *vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	cl_bool UMflag = false;
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	//copying the main data to the device memory
	unsigned int Nat = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Nat += (unsigned int)ra[iEl].size();
	if (UMflag) {//zero copy approach
		cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
		if (cfg->scenario != PDFonly) {
			*dq = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, cfg->q.N * sizeof(cl_float), NULL, NULL);
			cl_float *qfloat = (cl_float *)clEnqueueMapBuffer(queue, *dq, true, CL_MAP_WRITE, 0, cfg->q.N * sizeof(cl_float), 0, NULL, NULL, NULL);
			for (unsigned int iq = 0; iq < cfg->q.N; iq++) qfloat[iq] = (cl_float)q[iq];
			clEnqueueUnmapMemObject(queue, *dq, (void *)qfloat, 0, NULL, NULL);
		}
		if (cfg->source == xray) {
			*dFF = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, cfg->Nel * cfg->q.N * sizeof(cl_float), NULL, NULL);//device array for atomic form-factors
			cl_float *FFfloat = (cl_float *)clEnqueueMapBuffer(queue, *dFF, true, CL_MAP_WRITE, 0, cfg->Nel * cfg->q.N * sizeof(cl_float), 0, NULL, NULL, NULL);
			for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) FFfloat[iEl * cfg->q.N + iq] = (cl_float)FF[iEl][iq];
			}
			clEnqueueUnmapMemObject(queue, *dFF, (void *)FFfloat, 0, NULL, NULL);
		}
		*dra = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, Nat * sizeof(cl_float4), NULL, NULL);
		cl_float4 *hra = (cl_float4 *)clEnqueueMapBuffer(queue, *dra, true, CL_MAP_WRITE, 0, Nat * sizeof(cl_float4), 0, NULL, NULL, NULL);
		unsigned int iAtom = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
			for (vector<vect3d <double> >::const_iterator ri = ra[iEl].begin(); ri != ra[iEl].end(); ri++, iAtom++){//converting atomic coordinates from vect3d <double> to float4
				hra[iAtom].s[0] = (cl_float)ri->x;
				hra[iAtom].s[1] = (cl_float)ri->y;
				hra[iAtom].s[2] = (cl_float)ri->z;
				hra[iAtom].s[3] = 0;
			}
		}
		clEnqueueUnmapMemObject(queue, *dra, (void *)hra, 0, NULL, NULL);
		clReleaseCommandQueue(queue);
		return;
	}
	//copying the data to device memory
	if(cfg->scenario != PDFonly) {
		cl_float *qfloat = new cl_float[cfg->q.N];
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) qfloat[iq] = (cl_float)q[iq];
		*dq = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cfg->q.N * sizeof(cl_float), qfloat, NULL);
		delete[] qfloat;
		if (cfg->source == xray) {
			cl_float *FFfloat = new cl_float[cfg->Nel * cfg->q.N];
			for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) FFfloat[iEl * cfg->q.N + iq] = (cl_float)FF[iEl][iq];
			}
			*dFF = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cfg->Nel * cfg->q.N * sizeof(cl_float), FFfloat, NULL);
			delete[] FFfloat;
		}
	}
	cl_float4 *hra = new cl_float4[Nat];
	unsigned int iAtom = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = ra[iEl].begin(); ri != ra[iEl].end(); ri++, iAtom++){
			hra[iAtom].s[0] = (cl_float)ri->x;
			hra[iAtom].s[1] = (cl_float)ri->y;
			hra[iAtom].s[2] = (cl_float)ri->z;
			hra[iAtom].s[3] = 0;
		}
	}
	*dra = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Nat * sizeof(cl_float4), hra, NULL);
	delete[] hra;
}

//Deletes the atomic coordinates (ra), scattering vector magnitude (dq), the x-ray atomic form-factors (dFF) from the device memory and frees the OpenCL context and program object
void delDataFromDeviceOCL(const cl_context OCLcontext, const cl_program OCLprogram, const cl_mem ra, const cl_mem dFF, const cl_mem dq, const unsigned int Nel){
	clReleaseProgram(OCLprogram);
	clReleaseContext(OCLcontext);
	clReleaseMemObject(ra);//deallocating device memory for the atomic coordinates array
	if (dq != NULL) clReleaseMemObject(dq);//deallocating memory for the scattering vector magnitude array
	if (dFF != NULL) clReleaseMemObject(dFF);//deallocating device memory for the atomic form-factors
}

//Organazies the computations of the 2D scattering intensity in the polar coordinates (q,q_fi) of the reciprocal space with OpenCL
void calcInt2D_OCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double *** const I2D, double ** const I, const config * const cfg, \
					const unsigned int * const NatomEl, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq){
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();	
	cl_bool kernelExecTimeoutEnabled = true;
    cl_device_type device_type;
    clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled = false;
	size_t info_size;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	char *vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("nvidia") != string::npos) {
		clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	}
	const unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	unsigned int MaxAtomsPerLaunch = 0;
	if (kernelExecTimeoutEnabled){ //killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel execution time in seconds
		const double k = 4.e-8; // t = k * MaxAtomsPerLaunch * Nq * Nfi / GFLOPS
		MaxAtomsPerLaunch = (unsigned int)((tmax * GFLOPS) / (k * cfg->q.N * cfg->Nfi)); //maximum number of atoms per kernel launch
	}
	const unsigned int Nm = cfg->q.N * cfg->Nfi; //dimension of 2D intensity array
	//allocating memory on the device for amplitude and intensity 2D arrays
	//GPU has linear memory, so we stretch 2D arrays into 1D arrays
	cl_bool UMflag = false;
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	cl_mem dI = NULL;
	(UMflag) ? dI = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, Nm * sizeof(cl_float), NULL, NULL) : dI = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Nm * sizeof(cl_float), NULL, NULL);
	cl_mem dAr = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Nm * sizeof(cl_float), NULL, NULL);
	cl_mem dAi = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Nm * sizeof(cl_float), NULL, NULL);
	cl_mem dCS = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY, 3 * sizeof(cl_float4), NULL, NULL); //allocating the device memory for the transposed rotational matrix
	cl_float4 CS[3];//three rows of the transposed rotational matrix for the host and the device
	//creating kernels
	cl_kernel zeroInt2DKernel = clCreateKernel(OCLprogram, "zeroInt2DKernel", NULL); //reseting the 2D intensity matrix
	clSetKernelArg(zeroInt2DKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(zeroInt2DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(zeroInt2DKernel, 2, sizeof(cl_uint), (void *)&cfg->Nfi);
	cl_kernel zeroAmp2DKernel = clCreateKernel(OCLprogram, "zeroAmp2DKernel", NULL); //reseting 2D amplitude arrays
	clSetKernelArg(zeroAmp2DKernel, 0, sizeof(cl_mem), (void *)&dAr);
	clSetKernelArg(zeroAmp2DKernel, 1, sizeof(cl_mem), (void *)&dAi);
	clSetKernelArg(zeroAmp2DKernel, 2, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(zeroAmp2DKernel, 3, sizeof(cl_uint), (void *)&cfg->Nfi);
	cl_float lambdaf = (cl_float)cfg->lambda;
	cl_kernel calcInt2DKernel = clCreateKernel(OCLprogram, "calcInt2DKernel", NULL);
	clSetKernelArg(calcInt2DKernel, 0, sizeof(cl_uint), (void *)&cfg->source);
	clSetKernelArg(calcInt2DKernel, 1, sizeof(cl_mem), (void *)&dAr);
	clSetKernelArg(calcInt2DKernel, 2, sizeof(cl_mem), (void *)&dAi);
	clSetKernelArg(calcInt2DKernel, 3, sizeof(cl_mem), (void *)&dq);
	clSetKernelArg(calcInt2DKernel, 4, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(calcInt2DKernel, 5, sizeof(cl_uint), (void *)&cfg->Nfi);
	clSetKernelArg(calcInt2DKernel, 6, sizeof(cl_mem), (void *)&dCS);
	clSetKernelArg(calcInt2DKernel, 7, sizeof(cl_float), (void *)&lambdaf);
	clSetKernelArg(calcInt2DKernel, 8, sizeof(cl_mem), (void *)&ra);
	if (cfg->source == xray) {
		clSetKernelArg(calcInt2DKernel, 11, sizeof(cl_mem), (void *)&dFF);
		const cl_float zero = 0;
		clSetKernelArg(calcInt2DKernel, 13, sizeof(cl_float), (void *)&zero);
	}
	else {
		clSetKernelArg(calcInt2DKernel, 11, sizeof(cl_mem), NULL);
		const cl_uint zero = 0;
		clSetKernelArg(calcInt2DKernel, 12, sizeof(cl_uint), (void *)&zero);
	}
	cl_kernel Sum2DKernel = clCreateKernel(OCLprogram, "Sum2DKernel", NULL); //calculating the 2d scattering intensity using the scattering amplitude
	clSetKernelArg(Sum2DKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(Sum2DKernel, 1, sizeof(cl_mem), (void *)&dAr);
	clSetKernelArg(Sum2DKernel, 2, sizeof(cl_mem), (void *)&dAi);
	clSetKernelArg(Sum2DKernel, 3, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(Sum2DKernel, 4, sizeof(cl_uint), (void *)&cfg->Nfi);
	//creating command queue
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
	const unsigned int BlockSize2D = BlockSize2Dsmall;
	const size_t local_work_size[2] = { BlockSize2D, BlockSize2D }; //2d work-group size
	const size_t global_work_size[2] = { (cfg->Nfi / BlockSize2D + BOOL(cfg->Nfi % BlockSize2D))*BlockSize2D, (cfg->q.N / BlockSize2D + BOOL(cfg->q.N % BlockSize2D))*BlockSize2D };//2d global work-items numbers
	clEnqueueNDRangeKernel(queue, zeroInt2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);//reseting the 2D intensity matrix
	//2d scattering intensity should be calculated for the preset orientation of the sample (or averaged over multiple orientations specified by mesh)
	double dalpha = (cfg->Euler.max.x - cfg->Euler.min.x) / cfg->Euler.N.x, dbeta = (cfg->Euler.max.y - cfg->Euler.min.y) / cfg->Euler.N.y, dgamma = (cfg->Euler.max.z - cfg->Euler.min.z) / cfg->Euler.N.z;
	if (cfg->Euler.N.x<2) dalpha = 0;
	if (cfg->Euler.N.y<2) dbeta = 0;
	if (cfg->Euler.N.z<2) dgamma = 0;
	for (unsigned int ia = 0; ia < cfg->Euler.N.x; ia++){
		const double alpha = cfg->Euler.min.x + (ia + 0.5) * dalpha;
		for (unsigned int ib = 0; ib < cfg->Euler.N.y; ib++){
			const double beta = cfg->Euler.min.y + (ib + 0.5) * dbeta;
			for (unsigned int ig = 0; ig < cfg->Euler.N.z; ig++){
				const double gamma = cfg->Euler.min.z + (ig + 0.5) * dgamma;
				const vect3d <double> euler(alpha, beta, gamma);
				vect3d <double> RM0, RM1, RM2; //three rows of the rotational matrix
				calcRotMatrix(&RM0, &RM1, &RM2, euler, cfg->EulerConvention); //calculating the rotational matrix
				CS[0].s[0] = (cl_float)RM0.x; CS[0].s[1] = (cl_float)RM1.x; CS[0].s[2] = (cl_float)RM2.x; CS[0].s[3] = 0; //transposing the rotational matrix
				CS[1].s[0] = (cl_float)RM0.y; CS[1].s[1] = (cl_float)RM1.y; CS[1].s[2] = (cl_float)RM2.y; CS[1].s[3] = 0;
				CS[2].s[0] = (cl_float)RM0.z; CS[2].s[1] = (cl_float)RM1.z; CS[2].s[2] = (cl_float)RM2.z; CS[2].s[3] = 0;
				clEnqueueWriteBuffer(queue,dCS,true,0,3*sizeof(cl_float4),CS,0,NULL,NULL);//copying transposed rotational matrix from the host memory to the device memory
				clEnqueueNDRangeKernel(queue, zeroAmp2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL); //reseting 2D amplitude arrays
				unsigned int inp = 0;
				for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){ //looping over chemical elements (or ions)
					if (cfg->source == xray) clSetKernelArg(calcInt2DKernel, 12, sizeof(cl_uint), (void *)&iEl);
					else {//neutron scattering
						const float SLf = float(SL[iEl]);
						clSetKernelArg(calcInt2DKernel, 13, sizeof(cl_float), (void *)&SLf);
					}
					if (MaxAtomsPerLaunch) { //killswitch is enabled so MaxAtomsPerLaunch is set
						for (unsigned int i = 0; i < NatomEl[iEl] / MaxAtomsPerLaunch + BOOL(NatomEl[iEl] % MaxAtomsPerLaunch); i++) { //looping over the iterations
							const cl_uint Nst = inp + i*MaxAtomsPerLaunch; //index for the first atom on the current iteration step
							const cl_uint Nfin = MIN(Nst + MaxAtomsPerLaunch, inp + NatomEl[iEl]) - Nst; //index for the last atom on the current iteration step
							clSetKernelArg(calcInt2DKernel, 9, sizeof(cl_uint), (void *)&Nst);
							clSetKernelArg(calcInt2DKernel, 10, sizeof(cl_uint), (void *)&Nfin);							
							clEnqueueNDRangeKernel(queue, calcInt2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
							clFlush(queue);
							clFinish(queue);
						}
					}
					else { //killswitch is disabled so we execute the kernels for the entire ensemble of atoms
						clSetKernelArg(calcInt2DKernel, 9, sizeof(cl_uint), (void *)&inp);
						clSetKernelArg(calcInt2DKernel, 10, sizeof(cl_uint), (void *)&NatomEl[iEl]);
						clEnqueueNDRangeKernel(queue, calcInt2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);						
					}
					inp += NatomEl[iEl];
				}
				clEnqueueNDRangeKernel(queue, Sum2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);//calculating the 2d scattering intensity by the scattering amplitude
			}
		}
	}
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl]; //total number of atoms
	const cl_float norm = 1.f / (Ntot*cfg->Euler.N.x*cfg->Euler.N.y*cfg->Euler.N.z); //normalizing factor
	cl_kernel Norm2DKernel = clCreateKernel(OCLprogram, "Norm2DKernel", NULL); //normalizing the 2d scattering intensity
	clSetKernelArg(Norm2DKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(Norm2DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(Norm2DKernel, 2, sizeof(cl_uint), (void *)&cfg->Nfi);
	clSetKernelArg(Norm2DKernel, 3, sizeof(cl_float), (void *)&norm);
	clEnqueueNDRangeKernel(queue, Norm2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL); //normalizing the 2d scattering intensity
	if (cfg->PolarFactor) { //multiplying the 2d intensity by polar factor
		cl_kernel PolarFactor2DKernel = clCreateKernel(OCLprogram, "PolarFactor2DKernel", NULL); //multiplying the 2d intensity by polarization factor
		clSetKernelArg(PolarFactor2DKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(PolarFactor2DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(PolarFactor2DKernel, 2, sizeof(cl_uint), (void *)&cfg->Nfi);
		clSetKernelArg(PolarFactor2DKernel, 3, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(PolarFactor2DKernel, 4, sizeof(cl_float), (void *)&lambdaf);
		clEnqueueNDRangeKernel(queue, PolarFactor2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		clFinish(queue);
		clReleaseKernel(PolarFactor2DKernel);
	}
	//copying the 2d intensity matrix from the device memory to the host memory 
	cl_float *hI = NULL;
	if (UMflag) hI = (cl_float *)clEnqueueMapBuffer(queue, dI, true, CL_MAP_READ, 0, Nm * sizeof(cl_float), 0, NULL, NULL, NULL); 
	else {
		hI = new cl_float[Nm];
		clEnqueueReadBuffer(queue,dI,true,0, Nm * sizeof(cl_float), (void *) hI, 0, NULL, NULL);
	}
	*I = new double[cfg->q.N]; //array for 1d scattering intensity I[q] (I2D[q][fi] averaged over polar angle fi)
	*I2D = new double*[cfg->q.N]; //array for 2d scattering intensity 
	for (unsigned int iq = 0; iq < cfg->q.N; iq++){
		(*I)[iq] = 0;
		(*I2D)[iq] = new double[cfg->Nfi];
		for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++)	{
			(*I2D)[iq][ifi] = double(hI[iq*cfg->Nfi + ifi]);
			(*I)[iq] += (*I2D)[iq][ifi]; //calculating the 1d intensity (averaging I2D[q][fi] over the polar angle fi)
		}
		(*I)[iq] /= cfg->Nfi;
	}
	if (UMflag) clEnqueueUnmapMemObject(queue,dI,(void *) hI,0,NULL,NULL);
	else delete[] hI;
	//deallocating the device memory
	clFinish(queue);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(dI);
	clReleaseMemObject(dCS);
	clReleaseMemObject(dAr);
	clReleaseMemObject(dAi);
	clReleaseKernel(zeroInt2DKernel);
	clReleaseKernel(zeroAmp2DKernel);
	clReleaseKernel(calcInt2DKernel);
	clReleaseKernel(Sum2DKernel);
	clReleaseKernel(Norm2DKernel);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	cout << "2D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}

//rganazies the computations of the histogram of interatomic distances with OpenCL
void calcHistOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, cl_mem * const rij_hist, const cl_mem ra, \
					const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const config * const cfg){
	const unsigned int BlockSize = BlockSize1Dsmall, BlockSize2D = BlockSize2Dsmall; //size of the work-groups (256 by default, 16x16)
	const unsigned int NhistTotal = (cfg->Nel * (cfg->Nel + 1)) / 2 * cfg->Nhist;//NhistEl - number of partial (Element1<-->Element2) histograms
	cl_bool kernelExecTimeoutEnabled = true;	
	cl_device_type device_type;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled = false;
	size_t info_size;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	char *vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("nvidia") != string::npos) {
		clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	}
	const unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	unsigned int GridSizeExecMax = 2048;
	if (kernelExecTimeoutEnabled){ //killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 1.e-6; // t = k * GridSizeExecMax^2 * BlockSize2D^2 / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / k) / BlockSize2D), GridSizeExecMax);
	}
	*rij_hist = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, NhistTotal * sizeof(cl_ulong), NULL, NULL);
	//creating kernels
	cl_kernel zeroHistKernel = clCreateKernel(OCLprogram, "zeroHistKernel", NULL); //reseting the histogram array
	clSetKernelArg(zeroHistKernel, 0, sizeof(cl_mem), (void *)rij_hist);
	clSetKernelArg(zeroHistKernel, 1, sizeof(cl_uint), (void *)&NhistTotal);
	cl_kernel calcHistKernel = clCreateKernel(OCLprogram, "calcHistKernel", NULL);
	clSetKernelArg(calcHistKernel, 0, sizeof(cl_mem), (void *)&ra);
	clSetKernelArg(calcHistKernel, 5, sizeof(cl_mem), (void *)rij_hist);
	const cl_float bin = (cl_float)cfg->hist_bin;
	clSetKernelArg(calcHistKernel, 7, sizeof(cl_float), (void *)&bin);
	const cl_uint cutoff = (cl_uint)cfg->cutoff;
	clSetKernelArg(calcHistKernel, 9, sizeof(cl_uint), (void *)&cutoff);
	const cl_float Rcut2 = (cl_float)SQR(cfg->Rcutoff);
	clSetKernelArg(calcHistKernel, 10, sizeof(cl_float), (void *)&Rcut2);
	const unsigned int GSzero = NhistTotal / BlockSize + BOOL(NhistTotal % BlockSize);//Size of the grid for zeroHistKernel (it must not be large than 65535)
	const size_t local_work_size_zero = BlockSize;
	const size_t global_work_size_zero = GSzero * local_work_size_zero;
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);//Replace 0 with CL_QUEUE_PROFILING_ENABLE for profiling
	clEnqueueNDRangeKernel(queue, zeroHistKernel, 1, NULL, &global_work_size_zero, &local_work_size_zero, 0, NULL, NULL);
	const size_t local_work_size[2] = { BlockSize2D, BlockSize2D};//2D thread block size
	unsigned int *indEl = new unsigned int[cfg->Nel];
	indEl[0] = 0;
	for (unsigned int iEl = 1; iEl < cfg->Nel; iEl++) {
		(cfg->cutoff) ? indEl[iEl] = indEl[iEl - 1] + NatomEl_outer[iEl - 1] : indEl[iEl] = indEl[iEl - 1] + NatomEl[iEl - 1];
	}
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		unsigned int jElSt = iEl;
		if (cfg->cutoff) jElSt = 0;
		for (unsigned int jEl = jElSt; jEl < cfg->Nel; jEl++) {
			unsigned int jAtomST = 0;
			if ((cfg->cutoff) && (jEl < iEl)) jAtomST = NatomEl[jEl];
			unsigned int Nstart = 0;
			(jEl > iEl) ? Nstart = cfg->Nhist * (cfg->Nel * iEl - (iEl * (iEl + 1)) / 2 + jEl) : Nstart = cfg->Nhist * (cfg->Nel * jEl - (jEl * (jEl + 1)) / 2 + iEl);
			clSetKernelArg(calcHistKernel, 6, sizeof(cl_uint), (void *)&Nstart);
			for (unsigned int iAtom = 0; iAtom < NatomEl[iEl]; iAtom += BlockSize2D * GridSizeExecMax){
				const unsigned int i0 = indEl[iEl] + iAtom;
				clSetKernelArg(calcHistKernel, 1, sizeof(cl_uint), (void *)&i0);
				const unsigned int GridSizeExecY = MIN((NatomEl[iEl] - iAtom) / BlockSize2D + BOOL((NatomEl[iEl] - iAtom) % BlockSize2D), GridSizeExecMax);//Y-size of the grid on the current step
				const unsigned int iMax = MIN(BlockSize2D * GridSizeExecY, NatomEl[iEl] - iAtom);//index of the last i-th (row) atom				
				clSetKernelArg(calcHistKernel, 3, sizeof(cl_uint), (void *)&iMax);
				if (iEl == jEl) jAtomST = iAtom;//loop should exclude subdiagonal grids
				cl_ulong add = 2;
				clSetKernelArg(calcHistKernel, 11, sizeof(cl_ulong), (void *)&add);
				for (unsigned int jAtom = jAtomST; jAtom < NatomEl[jEl]; jAtom += BlockSize2D * GridSizeExecMax){
					const unsigned int j0 = indEl[jEl] + jAtom;
					clSetKernelArg(calcHistKernel, 2, sizeof(cl_uint), (void *)&j0);
					const unsigned int GridSizeExecX = MIN((NatomEl[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of the grid on the current step
					const unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomEl[jEl] - jAtom);//index of the last j-th (column) atom
					clSetKernelArg(calcHistKernel, 4, sizeof(cl_uint), (void *)&jMax);
					const size_t global_work_size[2] = { BlockSize2D*GridSizeExecX, BlockSize2D*GridSizeExecY };
					cl_uint diag = 0;
					if ((iEl == jEl) && (iAtom == jAtom)) diag = 1;//checking if we are on the diagonal grid or not		
					clSetKernelArg(calcHistKernel, 8, sizeof(cl_uint), (void *)&diag);
					//cl_event event;
					clEnqueueNDRangeKernel(queue, calcHistKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
					if (kernelExecTimeoutEnabled) {						
						//clWaitForEvents(1, &event);
						clFlush(queue);
						clFinish(queue);
						//cl_ulong time_start, time_end;
						//double time;
						//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
						//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
						//time = time_end - time_start;
						//cout << "calcHistKernel execution time is: " << time / 1000000.0 << " ms\n" << endl;
					}
				}
				if (cfg->cutoff) {
					const cl_uint diag = 0;
					clSetKernelArg(calcHistKernel, 8, sizeof(cl_uint), (void *)&diag);
					add = 1;
					clSetKernelArg(calcHistKernel, 11, sizeof(cl_ulong), (void *)&add);					
					for (unsigned int jAtom = NatomEl[jEl]; jAtom < NatomEl_outer[jEl]; jAtom += BlockSize2D * GridSizeExecMax){
						const unsigned int j0 = indEl[jEl] + jAtom;
						clSetKernelArg(calcHistKernel, 2, sizeof(cl_uint), (void *)&j0);
						const unsigned int GridSizeExecX = MIN((NatomEl_outer[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl_outer[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of the grid on the current step
						const unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomEl_outer[jEl] - jAtom);//index of the last j-th (column) atom
						clSetKernelArg(calcHistKernel, 4, sizeof(cl_uint), (void *)&jMax);
						const size_t global_work_size[2] = { BlockSize2D * GridSizeExecX, BlockSize2D * GridSizeExecY };
						clEnqueueNDRangeKernel(queue, calcHistKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
						if (kernelExecTimeoutEnabled) {
							clFlush(queue);
							clFinish(queue);
						}
					}
				}

			}
		}
	}
	clFinish(queue);
	delete[] indEl;
	clReleaseCommandQueue(queue);
	clReleaseKernel(calcHistKernel);
	clReleaseKernel(zeroHistKernel);
}


void AddCutoffOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, const cl_mem dI, const unsigned int * const NatomEl, const config * const cfg, \
					const cl_mem dFF, const vector<double> SL, const cl_mem dq, const unsigned int Ntot) {
	const size_t local_work_size = BlockSize1Dsmall;
	const unsigned int GSadd = cfg->q.N / BlockSize1Dsmall + BOOL(cfg->q.N % BlockSize1Dsmall);
	const size_t global_work_size_add = GSadd * local_work_size;
	if (cfg->source == xray) {		
		cl_mem dNatomEl = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cfg->Nel * sizeof(cl_uint), (void *)NatomEl, NULL);
		cl_kernel AddCutoffKernel = clCreateKernel(OCLprogram, "AddCutoffKernelXray", NULL);
		clSetKernelArg(AddCutoffKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(AddCutoffKernel, 1, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(AddCutoffKernel, 2, sizeof(cl_mem), (void *)&dFF);
		clSetKernelArg(AddCutoffKernel, 3, sizeof(cl_mem), (void *)&dNatomEl);
		clSetKernelArg(AddCutoffKernel, 4, sizeof(cl_uint), (void *)&cfg->Nel);
		clSetKernelArg(AddCutoffKernel, 5, sizeof(cl_uint), (void *)&Ntot);
		clSetKernelArg(AddCutoffKernel, 6, sizeof(cl_uint), (void *)&cfg->q.N);
		cl_float Rcut = (cl_float)cfg->Rcutoff;
		clSetKernelArg(AddCutoffKernel, 7, sizeof(cl_float), (void *)&Rcut);
		cl_float p0 = (cl_float)cfg->p0;
		clSetKernelArg(AddCutoffKernel, 8, sizeof(cl_float), (void *)&p0);
		cl_uint damping = (cl_uint)cfg->damping;
		clSetKernelArg(AddCutoffKernel, 9, sizeof(cl_uint), (void *)&damping);
		cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
		clEnqueueNDRangeKernel(queue, AddCutoffKernel, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);
		clFinish(queue);
		clReleaseCommandQueue(queue);
		clReleaseKernel(AddCutoffKernel);
		clReleaseMemObject(dNatomEl);
	}
	else {
		cl_float SLav = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) SLav += (cl_float)(SL[iEl]) * NatomEl[iEl];
		SLav /= Ntot;
		cl_kernel AddCutoffKernel = clCreateKernel(OCLprogram, "AddCutoffKernelNeutron", NULL);
		clSetKernelArg(AddCutoffKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(AddCutoffKernel, 1, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(AddCutoffKernel, 2, sizeof(cl_float), (void *)&SLav);
		clSetKernelArg(AddCutoffKernel, 3, sizeof(cl_uint), (void *)&Ntot);
		clSetKernelArg(AddCutoffKernel, 4, sizeof(cl_uint), (void *)&cfg->q.N);
		cl_float Rcut = (cl_float)cfg->Rcutoff;
		clSetKernelArg(AddCutoffKernel, 5, sizeof(cl_float), (void *)&Rcut);
		cl_float p0 = (cl_float)cfg->p0;
		clSetKernelArg(AddCutoffKernel, 6, sizeof(cl_float), (void *)&p0);
		cl_uint damping = (cl_uint)cfg->damping;
		clSetKernelArg(AddCutoffKernel, 7, sizeof(cl_uint), (void *)&damping);
		cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
		clEnqueueNDRangeKernel(queue, AddCutoffKernel, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);
		clFinish(queue);
		clReleaseCommandQueue(queue);
		clReleaseKernel(AddCutoffKernel);
	}
}


//Organazies the computations of the scattering intensity (powder diffraction pattern) using the histogram of interatomic distances with OpenCL
void calcInt1DHistOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, const cl_mem rij_hist, \
						const unsigned int * const NatomEl, const config * const cfg, const cl_mem dFF, const vector<double> SL, const cl_mem dq, const unsigned int Ntot){
	size_t info_size;
	cl_device_type device_type;	
	clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	cl_bool kernelExecTimeoutEnabled = true;
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled = false;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	char *vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("nvidia") != string::npos) {
		clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	}
	const unsigned int BlockSize = BlockSize1Dsmall;//setting the size of the thread blocks to 256 (default)
	const unsigned int GridSize = MIN(256, cfg->Nhist / BlockSize + BOOL(cfg->Nhist % BlockSize));
	const unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	unsigned int MaxBinsPerBlock = cfg->Nhist / GridSize + BOOL(cfg->Nhist % GridSize);
	if (kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 1.5e-5; // t = k * Nq * MaxBinsPerBlock / GFLOPS
		MaxBinsPerBlock = MIN((unsigned int)(tmax * GFLOPS / (k * cfg->q.N)), MaxBinsPerBlock);
	}
	const unsigned int Isize = GridSize * cfg->q.N;//each wotk-group writes to it's own copy of scattering intensity array
	cl_bool UMflag = false;
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	cl_mem dI = NULL;
	(UMflag) ? dI = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, Isize * sizeof(cl_float), NULL, NULL) : dI = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Isize * sizeof(cl_float), NULL, NULL);
	//creating kernels
	cl_kernel zero1DFloatArrayKernel = clCreateKernel(OCLprogram, "zero1DFloatArrayKernel", NULL);
	clSetKernelArg(zero1DFloatArrayKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(zero1DFloatArrayKernel, 1, sizeof(cl_uint), (void *)&Isize);
	const unsigned int GSzero = Isize / BlockSize + BOOL(Isize % BlockSize);//grid size for zero1DFloatArrayKernel
	const size_t local_work_size = BlockSize;
	const size_t global_work_size_zero = GSzero * local_work_size;
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);//Replace 0 with CL_QUEUE_PROFILING_ENABLE for profiling
	clEnqueueNDRangeKernel(queue, zero1DFloatArrayKernel, 1, NULL, &global_work_size_zero, &local_work_size, 0, NULL, NULL);//reseting intensity array
	const unsigned int GSadd = cfg->q.N / BlockSize + BOOL(cfg->q.N % BlockSize);//grid size for addIKernelXray/addIKernelNeutron
	const size_t global_work_size_add = GSadd * local_work_size;
	if (cfg->source == xray) {
		cl_kernel addIKernel = clCreateKernel(OCLprogram, "addIKernelXray", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(addIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(addIKernel, 2, sizeof(cl_mem), (void *)&dFF);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			clSetKernelArg(addIKernel, 3, sizeof(cl_uint), (void *)&iEl);
			clSetKernelArg(addIKernel, 4, sizeof(cl_uint), (void *)&NatomEl[iEl]);
			clEnqueueNDRangeKernel(queue, addIKernel, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
		}
		clFinish(queue);
		clReleaseKernel(addIKernel);
	}
	else {
		cl_kernel addIKernel = clCreateKernel(OCLprogram, "addIKernelNeutron", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(addIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			cl_float mult = float(SQR(SL[iEl]) * NatomEl[iEl]);
			clSetKernelArg(addIKernel, 2, sizeof(cl_float), (void *)&mult);
			clEnqueueNDRangeKernel(queue, addIKernel, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
		}
		clFinish(queue);
		clReleaseKernel(addIKernel);
	}	
	cl_kernel calcIntHistKernel = clCreateKernel(OCLprogram, "calcIntHistKernel", NULL);
	clSetKernelArg(calcIntHistKernel, 0, sizeof(cl_uint), (void *)&cfg->source);
	clSetKernelArg(calcIntHistKernel, 1, sizeof(cl_mem), (void *)&dI);
	if (cfg->source == xray) {
		clSetKernelArg(calcIntHistKernel, 2, sizeof(cl_mem), (void *)&dFF);
		const cl_float zero = 0;
		clSetKernelArg(calcIntHistKernel, 5, sizeof(cl_float), (void *)&zero);
	}
	else {
		clSetKernelArg(calcIntHistKernel, 2, sizeof(cl_mem), NULL);
		const cl_uint zero = 0;
		clSetKernelArg(calcIntHistKernel, 3, sizeof(cl_uint), (void *)&zero);
		clSetKernelArg(calcIntHistKernel, 4, sizeof(cl_uint), (void *)&zero);
	}
	clSetKernelArg(calcIntHistKernel, 6, sizeof(cl_mem), (void *)&dq);
	clSetKernelArg(calcIntHistKernel, 7, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(calcIntHistKernel, 8, sizeof(cl_mem), (void *)&rij_hist);
	clSetKernelArg(calcIntHistKernel, 11, sizeof(cl_uint), (void *)&cfg->Nhist);
	clSetKernelArg(calcIntHistKernel, 12, sizeof(cl_uint), (void *)&MaxBinsPerBlock);
	const cl_float hbin = (cl_float)cfg->hist_bin;
	clSetKernelArg(calcIntHistKernel, 13, sizeof(cl_float), (void *)&hbin);
	const cl_float Rcut = (cl_float)cfg->Rcutoff;
	clSetKernelArg(calcIntHistKernel, 14, sizeof(cl_float), (void *)&Rcut);
	const cl_uint damping = (cl_uint)cfg->damping;
	clSetKernelArg(calcIntHistKernel, 15, sizeof(cl_uint), (void *)&damping);
	unsigned int Nstart = 0;
	size_t global_work_size = GridSize * local_work_size;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		if (cfg->source == xray) clSetKernelArg(calcIntHistKernel, 3, sizeof(cl_uint), (void *)&iEl);
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){
			clSetKernelArg(calcIntHistKernel, 9, sizeof(cl_uint), (void *)&Nstart);
			if (cfg->source == xray) clSetKernelArg(calcIntHistKernel, 4, sizeof(cl_uint), (void *)&jEl);
			else {
				const cl_float SLij = (cl_float) (SL[iEl] * SL[jEl]);
				clSetKernelArg(calcIntHistKernel, 5, sizeof(cl_float), (void *)&SLij);
			}
			for (unsigned int iBin = 0; iBin < cfg->Nhist; iBin += GridSize * MaxBinsPerBlock) {//iterations to avoid killswitch triggering
				//cl_event event;				
				clSetKernelArg(calcIntHistKernel, 10, sizeof(cl_uint), (void *)&iBin);
				clEnqueueNDRangeKernel(queue, calcIntHistKernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
				//clWaitForEvents(1, &event);
				clFlush(queue);
				clFinish(queue);
				//cl_ulong time_start, time_end;
				//double time;
				//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
				//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
				//time = time_end - time_start;
				//cout << "calcIntHistKernel execution time is: " << time / 1000000.0 << " ms\n" << endl;
			}
		}
	}
	cl_kernel sumIKernel = clCreateKernel(OCLprogram, "sumIKernel", NULL); //summing intensity copies
	clSetKernelArg(sumIKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(sumIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(sumIKernel, 2, sizeof(cl_uint), (void *)&GridSize);
	clEnqueueNDRangeKernel(queue, sumIKernel, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);//summing intensity copies
	if (cfg->cutoff) AddCutoffOCL(OCLcontext, OCLdevice, OCLprogram, dI, NatomEl, cfg, dFF, SL, dq, Ntot);
	if (cfg->PolarFactor) {
		cl_kernel PolarFactor1DKernel = clCreateKernel(OCLprogram, "PolarFactor1DKernel", NULL);
		const cl_float lambdaf = float(cfg->lambda);
		clSetKernelArg(PolarFactor1DKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(PolarFactor1DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(PolarFactor1DKernel, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(PolarFactor1DKernel, 3, sizeof(cl_float), (void *)&lambdaf);
		clEnqueueNDRangeKernel(queue, PolarFactor1DKernel, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);
		clFinish(queue);
		clReleaseKernel(PolarFactor1DKernel);
	}
	cl_float *hI = NULL; //host array for scattering intensity
	if (UMflag) hI = (cl_float *)clEnqueueMapBuffer(queue, dI, true, CL_MAP_READ, 0, cfg->q.N * sizeof(cl_float), 0, NULL, NULL, NULL);
	else {
		hI = new cl_float[cfg->q.N];
		clEnqueueReadBuffer(queue, dI, true, 0, cfg->q.N * sizeof(cl_float), (void *)hI, 0, NULL, NULL);
	}
	*I = new double[cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] = double(hI[iq]) / Ntot;//normalizing	
	if (UMflag) clEnqueueUnmapMemObject(queue, dI, (void *)hI, 0, NULL, NULL);
	else delete[] hI;
	clFinish(queue);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(dI);
	clReleaseKernel(zero1DFloatArrayKernel);
	clReleaseKernel(calcIntHistKernel);
	clReleaseKernel(sumIKernel);

}

//Depending on the computational scenario organazies the computations of the scattering intensity(powder diffraction pattern) or PDF using the histogram of interatomic distances with OpenCL
void calcPDFandDebyeOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, double ** const PDF, const config * const cfg, \
						const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq) {
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	cl_mem rij_hist = NULL;//array for pair-distribution histogram (device only)
	calcHistOCL(OCLcontext, OCLdevice, OCLprogram, &rij_hist, ra, NatomEl, NatomEl_outer, cfg);//calculating the histogram
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	cout << "Histogram calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl];//calculating the total number of atoms
	if ((cfg->scenario == PDFonly) || (cfg->scenario == DebyePDF)) {//calculating the PDFs
		t1 = chrono::steady_clock::now();
		const unsigned int NPDF = (1 + (cfg->Nel * (cfg->Nel + 1)) / 2) * cfg->Nhist;
		size_t info_size;
		clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
		char * vendor = new char[info_size / sizeof(char)];
		clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
		string vendor_str(vendor);
		delete[] vendor;
		transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
		cl_bool UMflag = false;
		if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
		cl_mem dPDF = NULL;
		//allocating the device memory for PDF array
		(UMflag) ? dPDF = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NPDF * sizeof(cl_float), NULL, NULL) : dPDF = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, NPDF * sizeof(cl_float), NULL, NULL);
		cl_kernel zero1DFloatArrayKernel = clCreateKernel(OCLprogram, "zero1DFloatArrayKernel", NULL); 
		clSetKernelArg(zero1DFloatArrayKernel, 0, sizeof(cl_mem), (void *)&dPDF);
		clSetKernelArg(zero1DFloatArrayKernel, 1, sizeof(cl_uint), (void *)&NPDF);
		cl_float hbin = (cl_float)cfg->hist_bin;
		cl_kernel calcPartialPDFkernel = NULL;
		switch (cfg->PDFtype){
		case typeRDF://calculating partial RDFs
			calcPartialPDFkernel = clCreateKernel(OCLprogram, "calcPartialRDFkernel", NULL);
			break;
		case typePDF://calculating partial PDFs
			calcPartialPDFkernel = clCreateKernel(OCLprogram, "calcPartialPDFkernel", NULL);
			clSetKernelArg(calcPartialPDFkernel, 5, sizeof(cl_float), (void *)&hbin);
			break;
		case typeRPDF://calculating partial rPDFs
			calcPartialPDFkernel = clCreateKernel(OCLprogram, "calcPartialRPDFkernel", NULL);
			clSetKernelArg(calcPartialPDFkernel, 6, sizeof(cl_float), (void *)&hbin);
			break;
		}
		clSetKernelArg(calcPartialPDFkernel, 0, sizeof(cl_mem), (void *)&dPDF);
		clSetKernelArg(calcPartialPDFkernel, 1, sizeof(cl_mem), (void *)&rij_hist);
		clSetKernelArg(calcPartialPDFkernel, 3, sizeof(cl_uint), (void *)&cfg->Nhist);
		const unsigned int BlockSize = BlockSize1Dsmall;
		unsigned int GSzero = NPDF / BlockSize + BOOL(NPDF % BlockSize);
		unsigned int GridSize = cfg->Nhist / BlockSize + BOOL(cfg->Nhist % BlockSize);//grid size for zero1DFloatArrayKernel and other kernels
		const size_t local_work_size = BlockSize;
		const size_t global_work_size_zero = GSzero * local_work_size;
		const size_t global_work_size = GridSize * local_work_size;
		cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
		clEnqueueNDRangeKernel(queue, zero1DFloatArrayKernel, 1, NULL, &global_work_size_zero, &local_work_size, 0, NULL, NULL);//reseting the PDF array
		cl_uint Nstart = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){
				cl_float mult, sub;
				switch (cfg->PDFtype){
				case typeRDF://calculating partial RDFs
					mult = 1.f / (float(cfg->hist_bin) * Ntot);					
					break;
				case typePDF://calculating partial PDFs
					mult = 0.25f / (PIf * float(cfg->hist_bin * cfg->p0) * Ntot);
					break;
				case typeRPDF://calculating partial rPDFs
					mult = 1.f / (float(cfg->hist_bin) * Ntot);
					(jEl > iEl) ? sub = 8.f * PIf * float(cfg->p0) * float(NatomEl[iEl]) * float(NatomEl[jEl]) / SQR(float(Ntot)) : sub = 4.f * PIf * float(cfg->p0) * SQR(float(NatomEl[iEl])) / SQR(float(Ntot));
					clSetKernelArg(calcPartialPDFkernel, 5, sizeof(cl_float), (void *)&sub);
					break;
				}
				clSetKernelArg(calcPartialPDFkernel, 2, sizeof(cl_uint), (void *)&Nstart);
				clSetKernelArg(calcPartialPDFkernel, 4, sizeof(cl_float), (void *)&mult);
				clEnqueueNDRangeKernel(queue, calcPartialPDFkernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			}
		}		
		Nstart = cfg->Nhist;
		float Faverage2 = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Faverage2 += float(SL[iEl] * NatomEl[iEl]); //calculating the average form-factor
		Faverage2 /= Ntot;
		Faverage2 *= Faverage2;//and squaring it
		cl_kernel calcPDFkernel = clCreateKernel(OCLprogram, "calcPDFkernel", NULL);
		clSetKernelArg(calcPDFkernel, 0, sizeof(cl_mem), (void *)&dPDF);
		clSetKernelArg(calcPDFkernel, 2, sizeof(cl_uint), (void *)&cfg->Nhist);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {//calculating full PDF by summing partial PDFs
			for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){
				const cl_float multIJ = (cl_float)(SL[iEl] * SL[jEl]) / Faverage2;
				clSetKernelArg(calcPDFkernel, 1, sizeof(cl_uint), (void *)&Nstart);
				clSetKernelArg(calcPDFkernel, 3, sizeof(cl_float), (void *)&multIJ);
				clEnqueueNDRangeKernel(queue, calcPDFkernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			}
		}		
		unsigned int NPDFh = NPDF;//total PDF array size (full (cfg->Nhist) + partial (cfg->Nhist*(cfg->Nel*(cfg->Nel + 1)) / 2) )
		if (!cfg->PrintPartialPDF) NPDFh = cfg->Nhist;//if the partial PDFs are not needed, we are not copying them to the host	
		cl_float *hPDF = NULL;
		if (UMflag) hPDF = (cl_float *)clEnqueueMapBuffer(queue, dPDF, true, CL_MAP_READ, 0, NPDFh * sizeof(cl_float), 0, NULL, NULL, NULL);
		else {//copying the PDF from the device to the host
			hPDF = new cl_float[NPDFh];
			clEnqueueReadBuffer(queue, dPDF, true, 0, NPDFh * sizeof(cl_float), (void *)hPDF, 0, NULL, NULL);
		}
		*PDF = new double[NPDFh];//resulting array of doubles for PDF	
		for (unsigned int i = 0; i < NPDFh; i++) (*PDF)[i] = double(hPDF[i]);//converting into double
		if (UMflag) clEnqueueUnmapMemObject(queue, dPDF, (void *)hPDF, 0, NULL, NULL);
		else delete[] hPDF;
		clFinish(queue);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(dPDF);
		clReleaseKernel(calcPDFkernel);
		clReleaseKernel(calcPartialPDFkernel);
		clReleaseKernel(zero1DFloatArrayKernel);
		t2 = chrono::steady_clock::now();
		cout << "PDF calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	}
	if ((cfg->scenario == Debye_hist) || (cfg->scenario == DebyePDF)) {
		t1 = chrono::steady_clock::now();
		calcInt1DHistOCL(OCLcontext, OCLdevice, OCLprogram, I, rij_hist, NatomEl, cfg, dFF, SL, dq, Ntot);//calculating the scattering intensity using the pair-distribution histogram
		t2 = chrono::steady_clock::now();
		cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	}
	clReleaseMemObject(rij_hist);//deallocating memory for pair distribution histogram
}

//Organazies the computations of the scattering intensity (powder diffraction pattern) using the original Debye equation (without the histogram approximation) with OpenCL
void calcIntDebyeOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, const config * const cfg, \
					const unsigned int * const NatomEl, const unsigned int * const NatomEl_outer, const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq){
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();		
	cl_device_type device_type;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	cl_bool kernelExecTimeoutEnabled = true;
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled = false;
	size_t info_size;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	char *vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);	
	if (vendor_str.find("nvidia") != string::npos)	clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	const unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	const unsigned int BlockSize2D = BlockSize2Dsmall;//setting block size to 16x16 (default)
	unsigned int GridSizeExecMax = 64;
	if (kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 5.e-8; // t = k * GridSizeExecMax^2 * BlockSize2D^2 * cfg->q.N / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / (k * cfg->q.N)) / BlockSize2D), GridSizeExecMax);
	}	
	cl_bool UMflag = false;
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	cl_mem dI = NULL; 
	const unsigned int Isize = SQR(GridSizeExecMax)*cfg->q.N;//total size of the intensity array
	//allocating the device memory for the scattering intensity array
	(UMflag) ? dI = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, Isize * sizeof(cl_float), NULL, NULL) : dI = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Isize * sizeof(cl_float), NULL, NULL);
	cl_kernel zero1DFloatArrayKernel = clCreateKernel(OCLprogram, "zero1DFloatArrayKernel", NULL);
	clSetKernelArg(zero1DFloatArrayKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(zero1DFloatArrayKernel, 1, sizeof(cl_uint), (void *)&Isize);	
	const unsigned int BlockSize = SQR(BlockSize2D);//total number of threads per block
	const unsigned int GSzero = Isize / BlockSize + BOOL(Isize % BlockSize);//grid size for zero1DFloatArrayKernel
	const size_t local_work_size_zero = BlockSize;
	const size_t global_work_size_zero = GSzero * local_work_size_zero;
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
	clEnqueueNDRangeKernel(queue, zero1DFloatArrayKernel, 1, NULL, &global_work_size_zero, &local_work_size_zero, 0, NULL, NULL);//reseting intensity array
	const unsigned int GSadd = cfg->q.N / BlockSize + BOOL(cfg->q.N % BlockSize);//grid size for addIKernelXray/addIKernelNeutron
	const size_t global_work_size_add = GSadd * local_work_size_zero;
	if (cfg->source == xray) {
		cl_kernel addIKernel = clCreateKernel(OCLprogram, "addIKernelXray", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(addIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(addIKernel, 2, sizeof(cl_mem), (void *)&dFF);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			clSetKernelArg(addIKernel, 3, sizeof(cl_uint), (void *)&iEl);
			clSetKernelArg(addIKernel, 4, sizeof(cl_uint), (void *)&NatomEl[iEl]);
			clEnqueueNDRangeKernel(queue, addIKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
		}
		clFinish(queue);
		clReleaseKernel(addIKernel);
	}
	else {
		cl_kernel addIKernel = clCreateKernel(OCLprogram, "addIKernelNeutron", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(addIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			cl_float mult = float(SQR(SL[iEl]) * NatomEl[iEl]);
			clSetKernelArg(addIKernel, 2, sizeof(cl_float), (void *)&mult);
			clEnqueueNDRangeKernel(queue, addIKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
		}
		clFinish(queue);
		clReleaseKernel(addIKernel);
	}
	cl_kernel calcIntDebyeKernel = clCreateKernel(OCLprogram, "calcIntDebyeKernel", NULL);
	clSetKernelArg(calcIntDebyeKernel, 0, sizeof(cl_uint), (void *)&cfg->source);
	clSetKernelArg(calcIntDebyeKernel, 1, sizeof(cl_mem), (void *)&dI);
	if (cfg->source == xray) {
		clSetKernelArg(calcIntDebyeKernel, 2, sizeof(cl_mem), (void *)&dFF);
		const cl_float zero = 0;
		clSetKernelArg(calcIntDebyeKernel, 5, sizeof(cl_float), (void *)&zero);
	}
	else {
		clSetKernelArg(calcIntDebyeKernel, 2, sizeof(cl_mem), NULL);
		const cl_uint zero = 0;
		clSetKernelArg(calcIntDebyeKernel, 3, sizeof(cl_uint), (void *)&zero);
		clSetKernelArg(calcIntDebyeKernel, 4, sizeof(cl_uint), (void *)&zero);
	}
	clSetKernelArg(calcIntDebyeKernel, 6, sizeof(cl_mem), (void *)&dq);
	clSetKernelArg(calcIntDebyeKernel, 7, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(calcIntDebyeKernel, 8, sizeof(cl_mem), (void *)&ra);
	const cl_uint cutoff = (cl_uint)cfg->cutoff;
	clSetKernelArg(calcIntDebyeKernel, 15, sizeof(cl_uint), (void *)&cutoff);
	const cl_float Rcut = (cl_float)cfg->Rcutoff;
	clSetKernelArg(calcIntDebyeKernel, 16, sizeof(cl_float), (void *)&Rcut);
	const cl_uint damping = (cl_uint)cfg->damping;
	clSetKernelArg(calcIntDebyeKernel, 17, sizeof(cl_uint), (void *)&damping);
	const size_t local_work_size[2] = { BlockSize2D, BlockSize2D };
	unsigned int *indEl = new unsigned int[cfg->Nel];
	indEl[0] = 0;
	for (unsigned int iEl = 1; iEl < cfg->Nel; iEl++) {
		(cfg->cutoff) ? indEl[iEl] = indEl[iEl - 1] + NatomEl_outer[iEl - 1] : indEl[iEl] = indEl[iEl - 1] + NatomEl[iEl - 1];
	}	
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		if (cfg->source == xray) clSetKernelArg(calcIntDebyeKernel, 3, sizeof(cl_uint), (void *)&iEl);
		unsigned int jElSt = iEl;
		if (cfg->cutoff) jElSt = 0;
		for (unsigned int jEl = jElSt; jEl < cfg->Nel; jEl++) {
			if (cfg->source == xray) clSetKernelArg(calcIntDebyeKernel, 4, sizeof(cl_uint), (void *)&jEl);
			else {
				const cl_float SLij = (cl_float)(SL[iEl] * SL[jEl]);
				clSetKernelArg(calcIntDebyeKernel, 5, sizeof(cl_float), (void *)&SLij);
			}
			unsigned int jAtomST = 0;
			if ((cfg->cutoff) && (jEl < iEl)) jAtomST = NatomEl[jEl];
			for (unsigned int iAtom = 0; iAtom < NatomEl[iEl]; iAtom += BlockSize2D * GridSizeExecMax){
				const unsigned int i0 = indEl[iEl] + iAtom;
				clSetKernelArg(calcIntDebyeKernel, 9, sizeof(cl_uint), (void *)&i0);
				const unsigned int GridSizeExecY = MIN((NatomEl[iEl] - iAtom) / BlockSize2D + BOOL((NatomEl[iEl] - iAtom) % BlockSize2D), GridSizeExecMax);//Y-size of the grid on the current step
				const unsigned int iMax = MIN(BlockSize2D * GridSizeExecY, NatomEl[iEl] - iAtom);//index of the last i-th (row) atom
				clSetKernelArg(calcIntDebyeKernel, 11, sizeof(cl_uint), (void *)&iMax);
				if (iEl == jEl) jAtomST = iAtom;//loop should exclude subdiagonal grids
				cl_float mult = 2.f;
				clSetKernelArg(calcIntDebyeKernel, 14, sizeof(cl_float), (void *)&mult);
				for (unsigned int jAtom = jAtomST; jAtom < NatomEl[jEl]; jAtom += BlockSize2D * GridSizeExecMax){
					const unsigned int j0 = indEl[jEl] + jAtom;
					clSetKernelArg(calcIntDebyeKernel, 10, sizeof(cl_uint), (void *)&j0);
					const unsigned int GridSizeExecX = MIN((NatomEl[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of the grid on the current step
					const unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomEl[jEl] - jAtom);//index of the last j-th (column) atom
					clSetKernelArg(calcIntDebyeKernel, 12, sizeof(cl_uint), (void *)&jMax);
					const size_t global_work_size[2] = { BlockSize2D * GridSizeExecX, BlockSize2D * GridSizeExecY };
					cl_uint diag = 0;
					if ((iEl == jEl) && (iAtom == jAtom)) diag = 1;//checking if we are on the diagonal grid or not		
					clSetKernelArg(calcIntDebyeKernel, 13, sizeof(cl_uint), (void *)&diag);
					clEnqueueNDRangeKernel(queue, calcIntDebyeKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
					if (kernelExecTimeoutEnabled) {
						clFlush(queue);
						clFinish(queue);
					}
				}
				if (cfg->cutoff) {
					const cl_uint diag = 0;
					clSetKernelArg(calcIntDebyeKernel, 13, sizeof(cl_uint), (void *)&diag);
					mult = 1.f;
					clSetKernelArg(calcIntDebyeKernel, 14, sizeof(cl_float), (void *)&mult);
					for (unsigned int jAtom = NatomEl[jEl]; jAtom < NatomEl_outer[jEl]; jAtom += BlockSize2D * GridSizeExecMax){
						const unsigned int j0 = indEl[jEl] + jAtom;
						clSetKernelArg(calcIntDebyeKernel, 10, sizeof(cl_uint), (void *)&j0);
						const unsigned int GridSizeExecX = MIN((NatomEl_outer[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl_outer[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of the grid on the current step
						const unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomEl_outer[jEl] - jAtom);//index of the last j-th (column) atom
						clSetKernelArg(calcIntDebyeKernel, 12, sizeof(cl_uint), (void *)&jMax);
						const size_t global_work_size[2] = { BlockSize2D * GridSizeExecX, BlockSize2D * GridSizeExecY };
						clEnqueueNDRangeKernel(queue, calcIntDebyeKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
						if (kernelExecTimeoutEnabled) {
							clFlush(queue);
							clFinish(queue);
						}
					}
				}
			}
		}
	}
	delete[] indEl;
	cl_kernel sumIKernel = clCreateKernel(OCLprogram, "sumIKernel", NULL); //summing intensity copies
	clSetKernelArg(sumIKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(sumIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	cl_uint Ncopies = SQR(GridSizeExecMax);
	clSetKernelArg(sumIKernel, 2, sizeof(cl_uint), (void *)&Ncopies);
	clEnqueueNDRangeKernel(queue, sumIKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//summing intensity copies
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl]; //calculating total number of atoms		
	if (cfg->cutoff) AddCutoffOCL(OCLcontext, OCLdevice, OCLprogram, dI, NatomEl, cfg, dFF, SL, dq, Ntot);
	if (cfg->PolarFactor) {
		cl_kernel PolarFactor1DKernel = clCreateKernel(OCLprogram, "PolarFactor1DKernel", NULL);
		const cl_float lambdaf = float(cfg->lambda);
		clSetKernelArg(PolarFactor1DKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(PolarFactor1DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(PolarFactor1DKernel, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(PolarFactor1DKernel, 3, sizeof(cl_float), (void *)&lambdaf);
		clEnqueueNDRangeKernel(queue, PolarFactor1DKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);
		clFinish(queue);
		clReleaseKernel(PolarFactor1DKernel);
	}
	float *hI = NULL; //host and device arrays for scattering intensity		
	if (UMflag) hI = (cl_float *)clEnqueueMapBuffer(queue, dI, true, CL_MAP_READ, 0, cfg->q.N * sizeof(cl_float), 0, NULL, NULL, NULL);
	else {
		hI = new cl_float[cfg->q.N];
		clEnqueueReadBuffer(queue, dI, true, 0, cfg->q.N * sizeof(cl_float), (void *)hI, 0, NULL, NULL);
	}
	*I = new double[cfg->q.N];
	for (unsigned int iq = 0; iq<cfg->q.N; iq++) (*I)[iq] = double(hI[iq]) / Ntot;//normalizing	
	if (UMflag) clEnqueueUnmapMemObject(queue, dI, (void *)hI, 0, NULL, NULL);
	else delete[] hI;
	clFinish(queue);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(dI);
	clReleaseKernel(zero1DFloatArrayKernel);
	clReleaseKernel(calcIntDebyeKernel);
	clReleaseKernel(sumIKernel);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}

//Organazies the computations of the scattering intensity(powder diffraction pattern) using the original Debye equation(without the histogram approximation) with OpenCL
void calcIntPartialDebyeOCL(const cl_context OCLcontext, const cl_device_id OCLdevice, const cl_program OCLprogram, double ** const I, const config * const cfg, const unsigned int * const NatomEl, \
					const cl_mem ra, const cl_mem dFF, const vector<double> SL, const cl_mem dq, const block * const Block){
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();	
	cl_device_type device_type;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	cl_bool kernelExecTimeoutEnabled = true;
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled = false;
	size_t info_size;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	char *vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("nvidia") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	const unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	const unsigned int BlockSize2D = BlockSize2Dsmall;
	unsigned int GridSizeExecMax = 64;
	if (kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 5.e-8; // t = k * GridSizeExecMax^2 * BlockSize2D^2 * cfg->q.N / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / (k * cfg->q.N)) / BlockSize2D), GridSizeExecMax);
	}
	
	const unsigned int Nparts = (cfg->Nblocks * (cfg->Nblocks + 1)) / 2;
	cl_mem *dI = new cl_mem[Nparts];
	const unsigned int IsizeBlock = SQR(GridSizeExecMax) * cfg->q.N;//size of the intensity array per block
	for (unsigned int ipart = 0; ipart < Nparts; ipart++) dI[ipart] = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, IsizeBlock * sizeof(cl_float), NULL, NULL);//allocating the device memory for the scattering intensity array
	cl_kernel zero1DFloatArrayKernel = clCreateKernel(OCLprogram, "zero1DFloatArrayKernel", NULL);
	clSetKernelArg(zero1DFloatArrayKernel, 1, sizeof(cl_uint), (void *)&IsizeBlock);
	const unsigned int BlockSize = SQR(BlockSize2Dsmall);//setting block size to 16x16 (default)
	const unsigned int GSzero = IsizeBlock / BlockSize + BOOL(IsizeBlock % BlockSize);//grid size for zero1DFloatArrayKernel
	const size_t local_work_size_zero = BlockSize;
	const size_t global_work_size_zero = GSzero*local_work_size_zero;
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
	for (unsigned int ipart=0; ipart<Nparts; ipart++) {
		clSetKernelArg(zero1DFloatArrayKernel, 0, sizeof(cl_mem), (void *)&dI[ipart]);
		clEnqueueNDRangeKernel(queue, zero1DFloatArrayKernel, 1, NULL, &global_work_size_zero, &local_work_size_zero, 0, NULL, NULL);//reseting intensity array
	}
	unsigned int *NatomElBlock = new unsigned int[cfg->Nel * cfg->Nblocks];
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			NatomElBlock[iEl*cfg->Nblocks + iB] = Block[iB].NatomEl[iEl];
		}
	}	
	const unsigned int GSadd = cfg->q.N / BlockSize + BOOL(cfg->q.N % BlockSize);//grid size for addIKernelXray/addIKernelNeutron
	const size_t global_work_size_add = GSadd * local_work_size_zero;
	if (cfg->source == xray) {
		cl_kernel addIKernel = clCreateKernel(OCLprogram, "addIKernelXray", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(addIKernel, 2, sizeof(cl_mem), (void *)&dFF);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			clSetKernelArg(addIKernel, 3, sizeof(cl_uint), (void *)&iEl);
			for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
				const unsigned int Ipart = cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + iB;
				clSetKernelArg(addIKernel, 0, sizeof(cl_mem), (void *)&dI[Ipart]);
				clSetKernelArg(addIKernel, 4, sizeof(cl_uint), (void *)&NatomElBlock[iEl * cfg->Nblocks + iB]);
				clEnqueueNDRangeKernel(queue, addIKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
			}
		}
		clFinish(queue);
		clReleaseKernel(addIKernel);
	}
	else {
		cl_kernel addIKernel = clCreateKernel(OCLprogram, "addIKernelNeutron", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
				const unsigned int Ipart = cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + iB;
				clSetKernelArg(addIKernel, 0, sizeof(cl_mem), (void *)&dI[Ipart]);
				cl_float mult = float(SQR(SL[iEl]) * NatomElBlock[iEl * cfg->Nblocks + iB]);
				clSetKernelArg(addIKernel, 2, sizeof(cl_uint), (void *)&mult);
				clEnqueueNDRangeKernel(queue, addIKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
			}
		}
		clFinish(queue);
		clReleaseKernel(addIKernel);
	}
	cl_kernel calcIntDebyeKernel = clCreateKernel(OCLprogram, "calcIntDebyeKernel", NULL);
	clSetKernelArg(calcIntDebyeKernel, 0, sizeof(cl_uint), (void *)&cfg->source);
	if (cfg->source == xray) {
		clSetKernelArg(calcIntDebyeKernel, 2, sizeof(cl_mem), (void *)&dFF);
		const cl_float zero = 0;
		clSetKernelArg(calcIntDebyeKernel, 5, sizeof(cl_float), (void *)&zero);
	}
	else {
		clSetKernelArg(calcIntDebyeKernel, 2, sizeof(cl_mem), NULL);
		const cl_uint zero = 0;
		clSetKernelArg(calcIntDebyeKernel, 3, sizeof(cl_uint), (void *)&zero);
		clSetKernelArg(calcIntDebyeKernel, 4, sizeof(cl_uint), (void *)&zero);
	}
	clSetKernelArg(calcIntDebyeKernel, 6, sizeof(cl_mem), (void *)&dq);
	clSetKernelArg(calcIntDebyeKernel, 7, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(calcIntDebyeKernel, 8, sizeof(cl_mem), (void *)&ra);
	cl_float mult = 2.f;
	clSetKernelArg(calcIntDebyeKernel, 14, sizeof(cl_float), (void *)&mult);
	const cl_uint cutoff = 0;
	clSetKernelArg(calcIntDebyeKernel, 15, sizeof(cl_uint), (void *)&cutoff);
	const cl_float Rcut = 0;
	clSetKernelArg(calcIntDebyeKernel, 16, sizeof(cl_float), (void *)&Rcut);
	const cl_uint damping = 0;
	clSetKernelArg(calcIntDebyeKernel, 17, sizeof(cl_uint), (void *)&damping);
	const size_t local_work_size[2] = { BlockSize2D, BlockSize2D };
	unsigned int iAtomST = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iAtomST += NatomEl[iEl], iEl++) {
		if (cfg->source == xray)	if (cfg->source == xray) clSetKernelArg(calcIntDebyeKernel, 3, sizeof(cl_uint), (void *)&iEl);		
		unsigned int jAtomST = iAtomST;
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jAtomST += NatomEl[jEl], jEl++) {
			if (cfg->source == xray) clSetKernelArg(calcIntDebyeKernel, 4, sizeof(cl_uint), (void *)&jEl);
			else {
				const cl_float SLij = (cl_float)(SL[iEl] * SL[jEl]);
				clSetKernelArg(calcIntDebyeKernel, 5, sizeof(cl_float), (void *)&SLij);
			}
			unsigned int iAtomSB = 0;
			for (unsigned int iB = 0; iB < cfg->Nblocks; iAtomSB += NatomElBlock[iEl * cfg->Nblocks + iB], iB++) {
				for (unsigned int iAtom = 0; iAtom < NatomElBlock[iEl * cfg->Nblocks + iB]; iAtom += BlockSize2D * GridSizeExecMax){
					const unsigned int i0 = iAtomST + iAtomSB + iAtom;
					clSetKernelArg(calcIntDebyeKernel, 9, sizeof(cl_uint), (void *)&i0);
					const unsigned int GridSizeExecY = MIN((NatomElBlock[iEl * cfg->Nblocks + iB] - iAtom) / BlockSize2D + BOOL((NatomElBlock[iEl * cfg->Nblocks + iB] - iAtom) % BlockSize2D), GridSizeExecMax);
					const unsigned int iMax = MIN(BlockSize2D * GridSizeExecY, NatomEl[iEl] - iAtom);
					clSetKernelArg(calcIntDebyeKernel, 11, sizeof(cl_uint), (void *)&iMax);
					unsigned int jAtomSB = 0;
					for (unsigned int jB = 0; jB < cfg->Nblocks; jAtomSB += NatomElBlock[jEl * cfg->Nblocks + jB], jB++) {
						unsigned int Ipart = 0;
						(jB>iB) ? Ipart = cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + jB : Ipart = cfg->Nblocks * jB - (jB * (jB + 1)) / 2 + iB;
						clSetKernelArg(calcIntDebyeKernel, 1, sizeof(cl_mem), (void *)&dI[Ipart]);
						for (unsigned int jAtom = 0; jAtom < NatomElBlock[jEl * cfg->Nblocks + jB]; jAtom += BlockSize2D * GridSizeExecMax){
							const unsigned int j0 = jAtomST + jAtomSB + jAtom;
							clSetKernelArg(calcIntDebyeKernel, 10, sizeof(cl_uint), (void *)&j0);
							if (j0 >= i0) {
								const unsigned int GridSizeExecX = MIN((NatomElBlock[jEl * cfg->Nblocks + jB] - jAtom) / BlockSize2D + BOOL((NatomElBlock[jEl * cfg->Nblocks + jB] - jAtom) % BlockSize2D), GridSizeExecMax);
								const unsigned int jMax = MIN(BlockSize2D * GridSizeExecX, NatomElBlock[jEl * cfg->Nblocks + jB] - jAtom);
								clSetKernelArg(calcIntDebyeKernel, 12, sizeof(cl_uint), (void *)&jMax);
								const size_t global_work_size[2] = { BlockSize2D * GridSizeExecX, BlockSize2D * GridSizeExecY };
								cl_uint diag = 0;
								if (i0 == j0) diag = 1;//checking if we are on the diagonal grid or not
								clSetKernelArg(calcIntDebyeKernel, 13, sizeof(cl_uint), (void *)&diag);
								clEnqueueNDRangeKernel(queue, calcIntDebyeKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
								if (kernelExecTimeoutEnabled) {
									clFlush(queue);
									clFinish(queue);
								}
							}
						}
					}
				}
			}
		}
	}
	delete[] NatomElBlock;
	cl_bool UMflag = false;
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	unsigned int IpartialSize = (Nparts + 1) * cfg->q.N;
	cl_mem dIpart = NULL;
	//allocating the device memory for the scattering intensity array
	(UMflag) ? dIpart = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, IpartialSize * sizeof(cl_float), NULL, NULL) : dIpart = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, IpartialSize * sizeof(cl_float), NULL, NULL);
	const cl_uint Ncopies = SQR(GridSizeExecMax);
	cl_kernel sumIpartialKernel = clCreateKernel(OCLprogram, "sumIpartialKernel", NULL); //summing intensity copies
	clSetKernelArg(sumIpartialKernel, 1, sizeof(cl_mem), (void *)&dIpart);
	clSetKernelArg(sumIpartialKernel, 3, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(sumIpartialKernel, 4, sizeof(cl_uint), (void *)&Ncopies);
	for (unsigned int ipart=0; ipart<Nparts; ipart++) {
		clSetKernelArg(sumIpartialKernel, 0, sizeof(cl_mem), (void *)&dI[ipart]);
		clSetKernelArg(sumIpartialKernel, 2, sizeof(cl_uint), (void *)&ipart);
		clEnqueueNDRangeKernel(queue, sumIpartialKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//summing intensity copies
	}
	for (unsigned int ipart = 0; ipart<Nparts; ipart++) clReleaseMemObject(dI[ipart]);
	delete[] dI;
	if (cfg->PolarFactor) {		
		const cl_kernel PolarFactor1DKernel = clCreateKernel(OCLprogram, "PolarFactor1DKernel", NULL);		
		clSetKernelArg(PolarFactor1DKernel, 0, sizeof(cl_mem), (void *)&dIpart);
		clSetKernelArg(PolarFactor1DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(PolarFactor1DKernel, 2, sizeof(cl_mem), (void *)&dq);
		const cl_float lambdaf = float(cfg->lambda);
		clSetKernelArg(PolarFactor1DKernel, 3, sizeof(cl_float), (void *)&lambdaf);
		const size_t global_work_size_polar[2] = { GSadd * local_work_size_zero, Nparts + 1 };
		const size_t local_work_size_polar[2] = { local_work_size_zero, 1 };
		clEnqueueNDRangeKernel(queue, PolarFactor1DKernel, 2, NULL, global_work_size_polar, local_work_size_polar, 0, NULL, NULL);
		clFinish(queue);
		clReleaseKernel(PolarFactor1DKernel);
	}
	cl_kernel sumIKernel = clCreateKernel(OCLprogram, "sumIKernel", NULL); //summing partial intensities
	clSetKernelArg(sumIKernel, 0, sizeof(cl_mem), (void *)&dIpart);
	clSetKernelArg(sumIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	const cl_uint Nsum = Nparts + 1;
	clSetKernelArg(sumIKernel, 2, sizeof(cl_uint), (void *)&Nsum);
	clEnqueueNDRangeKernel(queue, sumIKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//summing intensity copies
	float *hI = NULL; //host and device arrays for scattering intensity
	if (UMflag) hI = (cl_float *)clEnqueueMapBuffer(queue, dIpart, true, CL_MAP_READ, 0, IpartialSize * sizeof(cl_float), 0, NULL, NULL, NULL);
	else {
		hI = new cl_float[IpartialSize];
		clEnqueueReadBuffer(queue, dIpart, true, 0, IpartialSize * sizeof(cl_float), (void *)hI, 0, NULL, NULL);
	}
	*I = new double[IpartialSize];
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl]; //calculating total number of atoms
	for (unsigned int iq = 0; iq < IpartialSize; iq++) (*I)[iq] = double(hI[iq]) / Ntot;//normalizing
	if (UMflag) clEnqueueUnmapMemObject(queue, dIpart, (void *)hI, 0, NULL, NULL);
	else delete[] hI;
	clFinish(queue);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(dIpart);
	clReleaseKernel(zero1DFloatArrayKernel);
	clReleaseKernel(sumIKernel);
	clReleaseKernel(calcIntDebyeKernel);
	clReleaseKernel(sumIpartialKernel);
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}
#endif
