//Copyright (C) 2015, Vladislav Neverov, NRC "Kurchatov institute"
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

//some macros and constants are defined here
#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include <math.h>
#include <string>
#include <iostream>
#include <map>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>

using namespace std;

//#define UseOMP
//#define UseMPI
//#define UseCUDA

//sizes of thread blocks for CUDA and OpenCL
#define BlockSize1Dsmall 256
#define BlockSize1Dmedium 512
#define BlockSize1Dlarge 1024
#define BlockSize2Dsmall 16
#define BlockSize2Dlarge 32

//computational scenarios
#define s2D 0 //2D diffraction pattern
#define Debye 1 //Powder diffraction pattern using the Debye equation
#define Debye_hist 2 //Powder diffraction pattern using the histogram approximation
#define PDFonly 3 //PDF only
#define DebyePDF 4 //PDF and powder diffraction pattern using the histogram approximation

//source types
#define neutron 0
#define xray 1

//PDF types
#define typeRDF 0
#define typePDF 1
#define typeRPDF 2

//Euler conventions
#define EulerXZX 0
#define EulerXYX 1
#define EulerYXY 2
#define EulerYZY 3
#define EulerZYZ 4
#define EulerZXZ 5
#define EulerXZY 6
#define EulerXYZ 7
#define EulerYXZ 8
#define EulerYZX 9
#define EulerZYX 10
#define EulerZXY 11

//constants
#define PI 3.1415926535897932384626433832795028841971693993
#define PIf 3.14159265f
#define MINIADIST 0.5 //Minimum interatomic distance in A
#define MINIADIST2 0.25 //Minimum interatomic distance square

//some macros
#define BOOL(x) ((x) ? 1 : 0)
#define ABS(x) ((x)<0 ?-(x):(x))
#define SQR(x) ((x)*(x))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#endif
