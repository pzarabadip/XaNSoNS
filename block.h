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

//Contains class definition for structural block

#ifndef _BLOCK_H_
#define _BLOCK_H_

#include "tinyxml2.h"
#include "vect3d.h"


//structure with atom information used to calculate final atomic ensemble (see block::calcAtoms())
typedef struct {
	double occ;
	unsigned int num;
	unsigned int el;
	unsigned int symm;
} atom_info;


struct site_inhabitant {
	string name;
	double occ;
	double uncert;	
	site_inhabitant(string in_name, double in_occ, double in_uncert) {
		name = in_name;
		occ = in_occ;
		uncert = in_uncert;
	}
};


struct atomic_site {
	vector < vect3d <double> > r;
	vect3d <double> r0;
	vector <site_inhabitant> in;
	bool shared;
	atomic_site(vect3d <double> r_new, site_inhabitant in_new) {
		r0 = r_new;
		// r.push_back(r_new);
		in.push_back(in_new);
		(in_new.occ < 0.999999) ? shared = true : shared = false;
	}
};


//defines structural block and all related data
class block {
public:
	unsigned int *NatomEl; //Contains numbers of atoms for each chemical elements in the smallest structural element
	vector <atomic_site> AtomicSite;
	unsigned int Nsymm; //Number of symmetry equivalent positions in the unit cell
	unsigned int Ncopy; //Number of copies of the structural block in the model sample
	unsigned int EulerConvention; //Euler angle convention index (order of rotations), 1 of 12, see typedefs.h
	set <string> names;
	string CellFile; // path to file containing the data of the smallest structural element (.cif or .txt) if this data is not in XML
	string CopiesFile; //path to file containing coordinates and Euler angles for copies of structural block (.cif or .txt) if they are not present in XML
	double Rcut; // Radius of the sphere to cut from the atomic ensemble of a single copy of the structural block (cutoff == true)
	double RcutCopies; // Radius of the sphere to cut from the entire atomic ensemble which includes all copies of this structural block (cutoffcopies == true)
	double dev_mol; // Uncertainty of the position of the smallest structural element (for molecules), sqrt(mol_Uiso)
	bool centered; // If true, sets the geometric center of a single copy of this structural block to (0,0,0)
	bool centeredAtoms; // If true, sets the geometric center of the atoms of the smallest structural element to (0,0,0)
	bool fractional; // If true, the coordinates of atoms are set in fractions of unit cell lengths 
	bool cutoff; // If true, the sphere will be cut-off from the atomic ensemble of a single copy of the structural block
	bool ro_mol; // Enable/disable random rotations of the smallest structural element (for spherical molecules)
	bool cutoffcopies; // If true, the sphere will be cut-off from the entire atomic ensemble which includes all copies of this structural block
	vect3d <unsigned int> Ncell; //Number of unit cells in a,b,c directions
	vect3d <double> rMean; // Geometric mean of a single copy of this structural block
	vect3d<double> *euler; // Euler angles of the copies of this structural block
	vect3d<double> *RM[3]; // Rotational matrices (rows) of the copies of this structural block
	vect3d<double> *rCopy; // Coordinates of copies of the structural blocks
	vect3d<double> e[3]; // Lattice vectors
	vect3d<double> *rSymm; // Centers of symmetry equivalent positions
	vect3d<double> *RMsymm[3]; // Rotational matrices for symmetry equivalent positions (rows)

	block(){
		Nsymm = 0; Ncopy = 1; Ncell.assign(1, 1, 1);
		Rcut = 0; RcutCopies = 0; dev_mol = 0;
		EulerConvention = EulerZYX;
		e[0].assign(1, 0, 0);	e[1].assign(0, 1, 0);	e[2].assign(0, 0, 1);
		centered = false; centeredAtoms = false; fractional = false; cutoff = false; cutoffcopies = false; ro_mol = false;
		NatomEl = NULL;	euler = NULL; rCopy = NULL; rSymm = NULL;
		RM[0] = NULL; RM[1] = NULL; RM[2] = NULL; RMsymm[0] = NULL; RMsymm[1] = NULL; RMsymm[2] = NULL;
	};

	/**
	Allocates memory for all arrays except id and idNeighb.
	Call after Natom, Nsymm and Ncopy and rearrangement are specified
	*/
	void create();

	/**
	Calculates absolute values of atomic coordinates by multiplying the fractional coordinates and the cell vectors.
	Returns immediately if fractional == false
	*/
	void redefAtoms();

	/**
	Sets the geometric center of the atoms of the smallest structural element to (0,0,0)
	Returns immediately if centeredAtoms == false
	*/
	void centerAtoms();	


	void updateAtomicSites(const string name, const vect3d <double> r, const double occ, const double Uiso);

	bool is_pos_degenerate(const vect3d <double> rSymmOp);

	bool is_pos_degenerate_fine_cut(const vect3d <double> rTemp);

	void calcUniqueSites();

	/**
	Sets the geometric center of a single copy of this structural block to (0,0,0)
	Returns immediately if centered == false
	*/
	void calcMean();

	/**
	Calculates atomic coordinates related to this structural block (and all its copies) in the final atomic ensemble.

	@param *ra Coordinates of atoms in the final ensemble
	*/
	void calcAtoms(vector < vect3d <double> > * const ra, const map <string, unsigned int> ID, const unsigned int Nel);

	/**
	Reads the data about the smallest structural element (atomic coordinates, occupancy numbers, etc.) from the text file.
	Returns -1 if error and 0 if OK.

	@param elements[] List of chemical elements ordered by atomic numbers (as in the periodic table)
	*/
	int ReadAtomsFromTXT(const string elements[]);

	/**
	Reads coordinates and Euler angles of the copies of the structural block from text file.
	Also computes the rotational matrices.
	Returns -1 if error and 0 if OK.
	*/
	int ReadCopiesFromTXT();
	
	/**
	//Reads the data about the smallest structural element (atomic coordinates, occupancy numbers, etc.).
	Returns -1 if error and 0 if OK.

	@param *AtomsNode XML element with atomic data
	@param elements[] List of chemical elements ordered by atomic numbers (as in the periodic table)
	*/
	int ReadAtoms(tinyxml2::XMLElement *AtomsNode, const string elements[]);

	/**
	Reads the symmetry elements data.
	Returns -1 if error and 0 if OK.

	@param *SymmNode First XML element containing the symmetry elements data
	*/
	int ReadSymmElements(tinyxml2::XMLElement *SymmNode);

	/**
	Reads the coordinates and Euler angles of the copies of the structural block.
	Returns -1 if error and 0 if OK.

	@param *CopiesNode XML element containing the coordinates and Euler angles of the copies of the structural block 
	@param word (key, value) pairs
	*/
	int ReadCopies(tinyxml2::XMLElement *CopiesNode, const map<string, unsigned int> word);

	/**
	Reads the lattice vectors and crystalline size along each direction.
	Returns -1 if error and 0 if OK.

	@param *CellNode XML element with lattice parameters
	*/
	int ReadLattice(tinyxml2::XMLElement *CellNode);

	/**
	Reads the parameters of the sphere which will be cut-off from the atomic enseble of the single structural block (Rcut) and/or the complete set of its copies (RcutCopies).
	Note, do not mix Rcut parameter with the cut-off radius for the PBC.
	Returns -1 if error and 0 if OK.

	@param *CutOffNode XML element with the parameters of the sphere to cut-off
	*/
	int ReadCutOffSphere(tinyxml2::XMLElement *CutOffNode);

	/**
	Reads the parameters of the structural block
	Returns -1 if error and 0 if OK.

	@param XML element containing the parameters of the structural block
	*/
	int ReadBlockParameters(tinyxml2::XMLElement *BlockNode);
	
	~block(){
		delete[] NatomEl;
		delete[] rCopy;
		delete[] euler;
		delete[] RM[0];
		delete[] RM[1];
		delete[] RM[2];
		delete[] rSymm;
		delete[] RMsymm[0];
		delete[] RMsymm[1];
		delete[] RMsymm[2];
	};
};

#endif