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

//functions, that read parameters from xml file are here
#include "config.h"
#include "block.h"
#include "ReadXML_utils.h"

#ifdef UseMPI
#include "mpi.h"
#endif

int getN(const char * const filename);//IO.cpp

int checkFile(const char * const filename);//IO.cpp

int readNeuData(const char * const filename, map<string, unsigned int> * const ID, map<unsigned int, string> * const EN, vector<double> * const SL, const set <string> * const names);//IO.cpp

int loadFF(const char *  const filename, map<string, unsigned int> * const ID, map<unsigned int, string> * const EN, vector <double *> * const FF, const double * const q, const unsigned int Nq, const set <string> * const names);//IO.cpp

void calcRotMatrix(vect3d <double> * const RM0, vect3d <double> * const RM1, vect3d <double> * const RM2, const vect3d <double> euler, const unsigned int convention);//CalcFunctions.cpp

#ifdef UseOCL
/**
Gets the GPU vendor name and GPU number form init_data.xml, if it's present in working directory.
Implements compatibility with BOINC clients (version 7.0.12+). Will be removed when the native BOINC app will be created.
Return 0, if OK and -1 if error.

@param *GPUtype   GPU vendor name obtained from init_data.xml
@param *DeviceNUM GPU number obtained from init_data.xml
*/
int GetOpenCLinfoFromInitDataXML(string *GPUtype, int *DeviceNUM);
#endif

/**
In the case of x-ray scattering when calculating the total PDF, SL array is used to store the values of the form-factors, FF(q0), where q0 is a single value of the scattering vector magnitude for which the total PDF is calculated.
This function set the values of SL array in this case.

@param FF   X-ray atomic form-factors
@param Nq   Number of points in the scattering vector magnitude mesh
@param *q   Scattering vector magnitude
@param *SL  Scattering lengths array
@param qPDF Value of the scattering vector magnitude for which the total PDF is calculated
*/
void setSLforPDF(const vector <double *> FF, const unsigned int Nq, const double * const q, vector<double> * const SL, const double qPDF);

/**
Returns the number of XML elements with the specified name

@param *xNode  first XML element with the name 'name'
@param *name   Name of the XML element
*/
unsigned int getNumberofElements(tinyxml2::XMLElement *xNode, const char * const name);

/**
Parses the XML file with parameters of the simulation and the model sample
Returns negative value if error and 0 if OK.

@param *cfg       Parameters of simulation
@param **q        Scattering vector magnitude
@param **Block    Array of the structural blocks
@param *FileName  Path to XML file
@param *ID        Array of [chemical element name, index] pairs. The pairs are created in this function according to the file
@param *SL        Array of neutron scattering lengths for all chemical elements
@param *FF        X-ray atomic form-factor arrays for all chemical elements
*/
int ReadConfig(config * const cfg, double ** const q, block ** const Block, const char * const FileName, map<string, unsigned int> * const ID, map<unsigned int, string> * const EN, vector<double> * const SL, vector<double *> * const FF);

#ifdef UseOCL
//Gets the GPU vendor name and GPU number form init_data.xml, if it's present in working directory.
//Implements compatibility with BOINC clients (version 7.0.12+). Will be removed when the native BOINC app will be created.
int GetOpenCLinfoFromInitDataXML(string *GPUtype, int *DeviceNUM){
	tinyxml2::XMLDocument doc;
	const char *FileName = "init_data.xml";
	tinyxml2::XMLError res = doc.LoadFile(FileName);
	if (res != tinyxml2::XML_SUCCESS)  return -1;
	tinyxml2::XMLElement *GPUtypeElement = doc.RootElement()->FirstChildElement("gpu_type");
	*GPUtype = string(GPUtypeElement->GetText());
	tinyxml2::XMLElement *GPUOpenCLindexElement = doc.RootElement()->FirstChildElement("gpu_opencl_dev_index");
	*DeviceNUM = atoi(GPUOpenCLindexElement->GetText());
	return 0;
}
#endif

//In the case of x-ray scattering when calculating the total PDF, SL array is used to store the values of the form-factors, FF(q0), where q0 is a single value of the scattering vector magnitude for which the total PDF is calculated.
//This function set the values of SL array in this case.
void setSLforPDF(const vector <double *> FF, const unsigned int Nq, const double * const q, vector<double> * const SL, const double qPDF) {
	for (unsigned int iq = 0; iq < Nq - 1; iq++) {
		if ((qPDF < q[iq]) || (qPDF > q[iq + 1])) continue;
		const double dq = q[iq] - q[iq + 1];
		for (vector<double *>::const_iterator ff = FF.begin(); ff != FF.end(); ff++) {
			const double a = ((*ff)[iq] - (*ff)[iq + 1]) / dq;
			const double b = ((*ff)[iq] * q[iq] - (*ff)[iq] * q[iq + 1]) / dq;
			SL->push_back(a*q[iq] + b);
		}
		break;
	}
}

//Returns the number of XML elements with the specified name
unsigned int getNumberofElements(tinyxml2::XMLElement *xNode, const char * const name){
	unsigned int count = 0;
	while (xNode){
		xNode = xNode->NextSiblingElement(name);
		count++;
	}
	return count;
}

//Parses the XML file with parameters of the simulation and the model sample
int ReadConfig(config * const cfg, double ** const q, block ** const Block, const char * const FileName, map<string, unsigned int> * const ID, map<unsigned int, string> * const EN, vector<double> * const SL, vector<double *> * const FF){
	tinyxml2::XMLDocument doc;
	const tinyxml2::XMLError res = doc.LoadFile(FileName);
	if (res != tinyxml2::XML_SUCCESS)  {
		if (!myid) cout << "Error: file " << FileName << " does not exist or contains errors." << endl;
		return -1;
	}
	tinyxml2::XMLElement *xMainNode = doc.RootElement();
	if (!myid) cout << "\nParsing calculation parameters..." << endl;
	tinyxml2::XMLElement *calcNode = xMainNode->FirstChildElement("Calculation");
	if (!calcNode) {
		if (!myid) cout << "Parsing error: 'Calculation' element is missing." << endl;
		return -1;
	}
	int error = cfg->ReadParameters(calcNode);
	if (cfg->scenario != PDFonly) {		
		*q = new double[cfg->q.N];
		const double deltaq = (cfg->q.max - cfg->q.min) / MAX(1,cfg->q.N-1);
		for (unsigned int iq = 0; iq<cfg->q.N; iq++) (*q)[iq] = cfg->q.min + iq*deltaq;
		if ((cfg->scenario == DebyePDF) && (cfg->source == xray)) {
			if (cfg->qPDF < cfg->q.min) {
				cfg->qPDF = (*q)[0];
				cout << "\nWarning: Calculation-->PDF-->q value is lower than Calculation-->q-->min value. Setting PDF-->q equal to q-->min." << endl;
			}
			else if (cfg->qPDF > cfg->q.max) {
				cfg->qPDF = (*q)[cfg->q.N - 1];
				cout << "\nWarning: Calculation-->PDF-->q value is higher than Calculation-->q-->max value. Setting PDF-->q equal to q-->max." << endl;
			}
		}
	}	
	cfg->Nblocks = getNumberofElements(xMainNode->FirstChildElement("Block"), "Block");
	if (!cfg->Nblocks) {
		if (!myid) cout << "Parsing error: structural blocks are not specified." << endl;
		return error - 1;
	}
	if (cfg->Nblocks < 2) cfg->calcPartialIntensity = false;
	*Block = new block[cfg->Nblocks];
	if (!myid) {
		tinyxml2::XMLElement *BlockNode = xMainNode->FirstChildElement("Block");
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			cout << "\nParsing Block " << iB << "..." << endl;
			error += (*Block)[iB].ReadBlockParameters(BlockNode);
			BlockNode = BlockNode->NextSiblingElement("Block");
		}
		cout << "\nAll blocks have been parsed.\n" << endl;
		set <string> names((*Block)[0].names);
		for (unsigned int iB = 1; iB < cfg->Nblocks; iB++) names.insert((*Block)[iB].names.begin(), (*Block)[iB].names.end());
		cfg->Nel = (unsigned int) names.size();
		if ((cfg->source == xray) && (cfg->scenario != PDFonly)) error += loadFF(cfg->FFfilename.c_str(), ID, EN, FF, *q, cfg->q.N, &names);
		else error += readNeuData(cfg->FFfilename.c_str(), ID, EN, SL, &names);
		if ((cfg->source == xray) && (cfg->scenario == DebyePDF)) setSLforPDF(*FF, cfg->q.N, *q, SL, cfg->qPDF);
	}	
#ifdef UseMPI
	MPI_Bcast(&cfg->Nel, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if ((cfg->source == xray) && (cfg->scenario != PDFonly)) {
		if (myid) {
			FF->resize(cfg->Nel);
			for (vector <double*>::iterator iFF = FF->begin(); iFF != FF->end(); iFF++) *iFF = new double [cfg->q.N];
		}
		for (vector <double*>::iterator iFF = FF->begin(); iFF != FF->end(); iFF++)	MPI_Bcast(*iFF, cfg->q.N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	if ((cfg->source == neutron) || (cfg->scenario > Debye_hist)) {
		if (myid) SL->resize(cfg->Nel);
		MPI_Bcast(&(*SL)[0], cfg->Nel, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
#endif
	return error;
}

