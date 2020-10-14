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

//Containts config class (parameters of calculation)

#ifndef _CONFIG_H_
#define _CONFIG_H_

#include "tinyxml2.h"
#include "vect3d.h"


//1D uniform mesh structure (part of config structure)
typedef struct {	
	double min, max; //mesh start and end points
	unsigned int N; //number of points in the mesh
} mesh;

//3D uniform  mesh structure (part of config structure)
typedef struct {
	vect3d <double> min, max;
	vect3d <unsigned int> N;
} mesh3d;

//Parameters of simulation
class config {
public:
	unsigned int scenario; // Computational scenario (see typedefs.h)
	unsigned int source; // Type of the source (xray or neutron, see typedefs.h)
	unsigned int Nblocks; // Total number of different structural blocks
	unsigned int Nel; // Total number of different chemical elements (ions, isotopes) in the sample
	unsigned int Nfi; // Number of points in polar angle for 2D diffraction pattern
	unsigned int Nhist; // Number of bins in the partial histogram of interatomic distances
	unsigned int PDFtype; // Type of the pair-distribution function (see typedefs.h)
	unsigned int EulerConvention; // Convention for Euler angles (order of rotations, see typedefs.h)
	double lambda; // Source wavelength
	double hist_bin; // Histogram bin width
	double qPDF; // Scattering vector magnitude for which the PDF is calculated in the case of source == xray
	double p0; // Average atomic density of the sample
	double Rcutoff; // Cut-off radius (to simulate the infinite systems)
	double Rsphere; // Radius of a sphere with i-th atoms (for infinite systems)
	vect3d <double> BoxEdge[3]; // Edges of parallelepiped with i-th atoms (for infinite systems)
	vect3d <double> BoxCorner; // Coordinates of the corner of parallelepiped with i-th atoms (for infinite systems)
	bool cutoff; //if true, the infinite system is simulated and the cutoff radius, Rcutoff, should be specified
	bool sphere; //if true, the i-th atoms are placed into the sphere, otherwise into the parallepiped (for infinite systems)
	bool damping; // if true, damping factor sin(pi*r/Rcutoff) / (pi*r/Rcutoff) is used to damp the ripples in the diffraction pattern
	bool PolarFactor; // if true, do polarization correction for unpolarized source (source == xray)
	bool PrintAtoms; // if true, print final atomic ensemble
	bool PrintPartialPDF; // if true, print partial PDFs
	bool calcPartialIntensity; // if true, calculate partial scattering intensities for each pair of different structural blocks
	mesh q; // Mesh for scattering vector magnitude
	mesh3d Euler; // Mesh for Euler angles if scenario == s2D (2D diffraction pattern)
	string name; // Prefix of the output file names
	string FFfilename; // Path to file with atomic form-factors (xray or neutron)

	config(){
		SetDefault();
	}

	//Set the parameters to their defaults
	void SetDefault();

	/**
	Reads properties of the parallelepiped box if cutoff is true.
	Returns -1 if error and 0 if OK.

	@param *boxNode XML element with parallelepiped parameters
	*/
	int ReadBoxProps(tinyxml2::XMLElement *boxNode);

	/**
	Reads the parameters required for "2D" scenario
	Returns -1 if error and 0 if OK.

	@param *calcNode XML element with calculation parameters
	*/
	int ReadParameters2d(tinyxml2::XMLElement *calcNode);

	/**
	Reads the parameters required for "Debye" scenario
	Returns -1 if error and 0 if OK.

	@param *calcNode XML element with calculation parameters
	*/
	int ReadParametersDebye(tinyxml2::XMLElement *calcNode);

	/**
	Reads the parameters required for "Debye_hist" scenario
	Returns -1 if error and 0 if OK.

	@param *calcNode XML element with calculation parameters
	*/
	int ReadParametersDebye_hist(tinyxml2::XMLElement *calcNode);

	/**
	Reads the parameters required for "PDFonly" scenario
	Returns -1 if error and 0 if OK.

	@param *calcNode XML element with calculation parameters
	*/
	int ReadParametersPDFonly(tinyxml2::XMLElement *calcNode);

	/**
	Reads the parameters required for "DebyePDF" scenario
	Returns -1 if error and 0 if OK.

	@param *calcNode XML element with calculation parameters
	*/
	int ReadParametersDebyePDF(tinyxml2::XMLElement *calcNode);

	/**
	Reads the parameters of calculation
	Returns -1 if error and 0 if OK.

	@param *calcNode XML element with calculation parameters
	*/
	int ReadParameters(tinyxml2::XMLElement *calcNode);

};
#endif