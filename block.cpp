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

//Defines function of block class

#include "block.h"
#include "ReadXML_utils.h"
#include <random>
#include <chrono>


void calcRotMatrix(vect3d <double> * const RM0, vect3d <double> * const RM1, vect3d <double> * const RM2, const vect3d <double> euler, const unsigned int convention);//CalcFunctions.cpp

int getN(const char * const filename);//IO.cpp

unsigned int getNumberofElements(tinyxml2::XMLElement *xNode, const char * const name);//ReadXML.cpp

//Allocates memory for all arrays except id and idNeighb.
void block::create(){
	if (Ncopy) {
		rCopy = new vect3d<double>[Ncopy];
		euler = new vect3d<double>[Ncopy];
		for (unsigned int i = 0; i<3; i++) RM[i] = new vect3d<double>[Ncopy];
	}
	if (Nsymm) {
		rSymm = new vect3d<double>[Nsymm];
		for (unsigned int i = 0; i<3; i++) RMsymm[i] = new vect3d<double>[Nsymm];
	}
}

//Calculates absolute values of atomic coordinates by multiplying the fractional coordinates and the cell vectors.
void block::redefAtoms(){
	if (!fractional) return;
	for (vector <atomic_site>::iterator it = AtomicSite.begin(); it != AtomicSite.end(); it++) {
		for (vector < vect3d <double> >::iterator ir = it->r.begin(); ir != it->r.end(); ir++) {
			*ir = e[0] * ir->x + e[1] * ir->y + e[2] * ir->z;
		}
	}
}


//Sets the geometric center of the atoms of the smallest structural element to (0,0,0).
void block::centerAtoms(){
	if ((!centeredAtoms) || (fractional)) return;
	vect3d <double> rTemp;
	for (vector <atomic_site>::iterator it = AtomicSite.begin(); it != AtomicSite.end(); it++) rTemp += it->r0;
	rTemp = rTemp / (double)AtomicSite.size();
	for (vector <atomic_site>::iterator it = AtomicSite.begin(); it != AtomicSite.end(); it++) it->r0 += rTemp;
}

//Sets the geometric center of a single copy of this structural block to (0,0,0)
void block::calcMean(){
	if (!centered) return;
	unsigned int Nat = 0;
	for (vector <atomic_site>::iterator it = AtomicSite.begin(); it != AtomicSite.end(); it++) {
		Nat += (unsigned int)it->r.size();
		for (vector < vect3d <double> >::iterator ir = it->r.begin(); ir != it->r.end(); ir++) {
			for (unsigned int iCellX = 0; iCellX < Ncell.x; iCellX++){
				for (unsigned int iCellY = 0; iCellY < Ncell.y; iCellY++){
					for (unsigned int iCellZ = 0; iCellZ < Ncell.z; iCellZ++){
						rMean += *ir + e[0] * (double)iCellX + e[1] * (double)iCellY + e[2] * (double)iCellZ;
					}
				}
			}
		}
	}
	Nat *= Ncell.x * Ncell.y * Ncell.z;
	rMean = rMean / (double)Nat;
}


//bool block::is_pos_degenerate(const vect3d <double> rSymmOp) {
//	int cellStX = -2, cellFinX = 3, cellStY = -2, cellFinY = 3, cellStZ = -2, cellFinZ = 3;
//	for (int iCellX = cellStX; iCellX < cellFinX; iCellX++){
//		for (int iCellY = cellStY; iCellY < cellFinY; iCellY++){
//			for (int iCellZ = cellStZ; iCellZ < cellFinZ; iCellZ++){
//				vect3d <double> rShift(1.0 * iCellX, 1.0 * iCellY, 1.0 * iCellZ);
//				const vect3d <double> rTemp = rShift + rSymmOp;
//				for (vector <atomic_site>::iterator it = AtomicSite.begin(); it != AtomicSite.end(); it++){
//					for (vector < vect3d <double> >::iterator ir = it->r.begin(); ir != it->r.end(); ir++) {
//						if ((ABS(ir->x - rTemp.x) < 1.e-3) && (ABS(ir->y - rTemp.y) < 1.e-3) && (ABS(ir->z - rTemp.z) < 1.e-3)) {
//							return true;
//						}
//					}
//				}
//			}
//		}
//	}
//	return false;
//}

bool block::is_pos_degenerate(const vect3d <double> rTemp) {
	for (vector <atomic_site>::iterator jt = AtomicSite.begin(); jt != AtomicSite.end(); jt++){
		for (vector < vect3d <double> >::iterator ir = jt->r.begin(); ir != jt->r.end(); ir++) {
			if ((ABS(ir->x - rTemp.x) < 1.e-3) && (ABS(ir->y - rTemp.y) < 1.e-3) && (ABS(ir->z - rTemp.z) < 1.e-3)) {
				return true;
			}
		}
	}
	return false;
}


void block::calcUniqueSites(){
	for (unsigned int iSymm = 0; iSymm < Nsymm; iSymm++){
		for (vector <atomic_site>::iterator it = AtomicSite.begin(); it != AtomicSite.end(); it++){
			vect3d <double> r(RMsymm[0][iSymm].dot(it->r0), RMsymm[1][iSymm].dot(it->r0), RMsymm[2][iSymm].dot(it->r0));
			r += rSymm[iSymm];
			if (!fractional) it->r.push_back(r);
			else{
				int cellStX = -2, cellFinX = 3, cellStY = -2, cellFinY = 3, cellStZ = -2, cellFinZ = 3;
				for (int iCellX = cellStX; iCellX < cellFinX; iCellX++){
					for (int iCellY = cellStY; iCellY < cellFinY; iCellY++){
						for (int iCellZ = cellStZ; iCellZ < cellFinZ; iCellZ++){
							vect3d <double> rShift(1.0 * iCellX, 1.0 * iCellY, 1.0 * iCellZ);
							vect3d <double> rTemp = rShift + r;
							if ((rTemp.x < -0.001) || (rTemp.x >= 0.999) || (rTemp.y < -0.001) || (rTemp.y >= 0.999) || (rTemp.z < -0.001) || (rTemp.z >= 0.999)) continue;
							if (!is_pos_degenerate(rTemp)) it->r.push_back(rTemp);
						}
					}
				}
			}
			// if ((!fractional) || (!is_pos_degenerate(r))) it->r.push_back(r);
		}
	}
}

//Calculates atomic coordinates related to this structural block (and all its copies) in the final atomic ensemble
void block::calcAtoms(vector < vect3d <double> > * const ra, const map <string, unsigned int> ID, const unsigned int Nel){
	normal_distribution<double> distr_mol = normal_distribution<double>(0, dev_mol);
	normal_distribution<double> distr = normal_distribution<double>(0, 1.);
	unsigned int seed = (unsigned int)chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	uniform_real_distribution<double> uniform_real(0.0, 1.0);
	vect3d <double> rMeanC;
	if (cutoffcopies) {
		for (unsigned int iCopy = 0; iCopy < Ncopy; iCopy++) rMeanC += rCopy[iCopy];
		rMeanC = rMeanC / (double)Ncopy;
	}
	vect3d <double> *r_mol_dev = NULL;
	vect3d <double> *RM_mol[3] = { NULL, NULL, NULL };
	bool shift_mol = false;
	if (dev_mol >= 1.e-7) {
		shift_mol = true;
		r_mol_dev = new vect3d <double>[Nsymm];
	}
	if (ro_mol){
		for (unsigned int k = 0; k < 3; k++) RM_mol[k] = new vect3d <double>[Nsymm];
	}
	NatomEl = new unsigned int[Nel];
	for (unsigned int iEl = 0; iEl < Nel; iEl++) NatomEl[iEl] = 0;
	for (unsigned int iCopy = 0; iCopy < Ncopy; iCopy++){
		for (unsigned int iCellX = 0; iCellX < Ncell.x; iCellX++){
			for (unsigned int iCellY = 0; iCellY < Ncell.y; iCellY++){
				for (unsigned int iCellZ = 0; iCellZ < Ncell.z; iCellZ++){
					const vect3d <double> celltrans = e[0] * (double)iCellX + e[1] * (double)iCellY + e[2] * (double)iCellZ;
					if (shift_mol){
						for (unsigned int iSymm = 0; iSymm < Nsymm; iSymm++){
							const double fi = 2 * PI*uniform_real(generator);
							const double theta = PI*uniform_real(generator);
							double Radd = distr_mol(generator);
							Radd = ABS(Radd);
							r_mol_dev[iSymm].assign(Radd*sin(theta)*cos(fi), Radd*sin(theta)*sin(fi), Radd*cos(theta));
						}
					}
					if (ro_mol){
						for (unsigned int iMol = 0; iMol < Nsymm; iMol++){
							const vect3d <double> Euler(2.*PI*uniform_real(generator), PI*uniform_real(generator), 2.*PI*uniform_real(generator));
							const vect3d <double> cosEul = cos(Euler);
							const vect3d <double> sinEul = sin(Euler);
							RM_mol[0][iMol].assign(cosEul.x*cosEul.y - sinEul.x*sinEul.y*cosEul.z, -cosEul.x*sinEul.y - sinEul.x*cosEul.z*cosEul.y, sinEul.x*sinEul.z);
							RM_mol[1][iMol].assign(sinEul.x*cosEul.y + cosEul.x*sinEul.y*cosEul.z, -sinEul.x*sinEul.y + cosEul.x*cosEul.y*cosEul.z, -cosEul.x*sinEul.z);
							RM_mol[2][iMol].assign(sinEul.z*sinEul.y, sinEul.z*cosEul.y, cosEul.z);
						}
					}
					for (vector <atomic_site>::iterator it = AtomicSite.begin(); it != AtomicSite.end(); it++) {
						site_inhabitant *inhabitant = &(it->in.front());
						for (unsigned int ir = 0; ir < (unsigned int) it->r.size(); ir++) {
							if (it->shared) {
								inhabitant = NULL;
								const double randval = uniform_real(generator);
								//cout << randval << "\n" << endl;
								//for (vector <site_inhabitant>::iterator inh = it->in.begin(); inh != it->in.end(); inh++) {
								//	cout << inh->name << " : " << inh->occ << " ";
								//}
								//cout << "\n" << endl;
								for (vector <site_inhabitant>::iterator inh = it->in.begin(); inh != it->in.end(); inh++) {
									if (randval < inh->occ) {
										inhabitant = &(*inh);
										break;
									}
								}
							}
							if (!inhabitant) continue;
							const unsigned int iEl = ID.at(inhabitant->name);
							//if (it->shared) cout << "Selected: " << inhabitant->name << "\n" << endl;
							vect3d <double> ratom = it->r[ir];
							if (ro_mol) {
								const vect3d <double> rdiff = ratom - rSymm[ir];
								ratom.assign(rdiff.dot(RM_mol[0][ir]), rdiff.dot(RM_mol[1][ir]), rdiff.dot(RM_mol[2][ir]));
								ratom += rSymm[ir];
							}
							if (shift_mol) ratom += r_mol_dev[ir];
							ratom += celltrans;
							const vect3d <double> ratom_c = ratom - rMean;
							if ((!cutoff) || (ratom_c.sqr() <= SQR(Rcut))) {
								const vect3d <double> rTrans(ratom_c.dot(RM[0][iCopy]), ratom_c.dot(RM[1][iCopy]), ratom_c.dot(RM[2][iCopy]));
								ratom = rCopy[iCopy] + rTrans;
								if (inhabitant->uncert >= 1.e-7) {
									const double fi = 2 * PI*uniform_real(generator);
									const double theta = PI*uniform_real(generator);
									double Radd = inhabitant->uncert * distr(generator);
									Radd = ABS(Radd);
									const vect3d <double> radd(Radd*sin(theta)*cos(fi), Radd*sin(theta)*sin(fi), Radd*cos(theta));
									ratom += radd;
								}
								if ((!cutoffcopies) || ((ratom - rMeanC).sqr() <= SQR(RcutCopies)))	{
									ra[iEl].push_back(ratom);
									NatomEl[iEl]++;
								}
							}
						}
					}
				}
			}
		}
	}
	delete[] r_mol_dev;
	for (unsigned int k = 0; k<3; k++) delete[] RM_mol[k];
}


void block::updateAtomicSites(const string name, const vect3d <double> r, const double occ, const double Uiso) {
	bool is_new_site = true;
	if (occ < 0.999999){
		for (vector <atomic_site>::iterator it = AtomicSite.begin(); it != AtomicSite.end(); it++) {
			if (!it->shared) continue;
			if ((ABS(it->r0.x - r.x) < 1.e-4) && (ABS(it->r0.y - r.y) < 1.e-4) && (ABS(it->r0.z - r.z) < 1.e-4)) {
				is_new_site = false;
				site_inhabitant in(name, occ + it->in.back().occ, sqrt(Uiso));
				it->in.push_back(in);
				names.insert(name);
				break;
			}
		}
	}
	if (is_new_site) {
		site_inhabitant in(name, occ, sqrt(Uiso));
		atomic_site new_site(r, in);
		AtomicSite.push_back(new_site);
		names.insert(name);
	}
}


//Reads the data about the smallest structural element (atomic coordinates, occupancy numbers, etc.) from the text file
int block::ReadAtomsFromTXT(const string elements[]){
	cout << "Reading " << CellFile << endl;
	ifstream in(CellFile.c_str());
	int iAtom = 0;
	if (in.is_open()){
		string line;
		while (getline(in, line)){
			if ((line.size() < 2) || (line[0] == '#')) continue;
			//data order: name/Z r Uiso occ
			istringstream streamline(line);
			string name;
			int Ztmp = 0;
			vect3d <double> r;
			double Uiso = 0;
			double occ = 1.;
			streamline >> name;
			streamline >> r;
			if (streamline.fail()) continue;
			if (streamline.good()) streamline >> Uiso;
			if (streamline.good()) streamline >> occ;
			Ztmp = atoi(name.c_str());
			if (Ztmp) name = elements[Ztmp];
			updateAtomicSites(name, r, occ, Uiso);			
			iAtom++;
		}
		in.close();
		return 0;
	}
	cout << "Error: cannot open file " << CellFile << endl;
	return -1;
}

//Reads coordinates and Euler angles of the copies of the structural block from text file.
//Also computes the rotational matrices.
int block::ReadCopiesFromTXT(){
	cout << "Reading " << CopiesFile << endl;
	ifstream in(CopiesFile.c_str());
	if (in.is_open()){
		int iCopy = 0;
		vect3d <double> cosEul, sinEul;
		string line;
		while (getline(in, line)){
			if ((line.size() < 2) || (line[0] == '#')) continue;
			istringstream streamline(line);
			streamline >> rCopy[iCopy] >> euler[iCopy];
			euler[iCopy] = euler[iCopy] / 180.f*PI;
			calcRotMatrix(&RM[0][iCopy], &RM[1][iCopy], &RM[2][iCopy], euler[iCopy], EulerConvention);
			iCopy++;
		}
		in.close();
		return 0;
	}
	cout << "Error: cannot open file " << CopiesFile << endl;
	return -1;
};


//Reads the data about the smallest structural element (atomic coordinates, occupancy numbers, etc.)
int block::ReadAtoms(tinyxml2::XMLElement *AtomsNode, const string elements[]){
	if (!AtomsNode){
		cout << "XML parsing error: 'Atoms' node is missing." << endl;
		return -1;
	}
	int error = GetAttribute(AtomsNode, "Atoms", "filename", CellFile);
	unsigned int Natom = 0;
	CellFile.length() ? Natom = getN(CellFile.c_str()) : Natom = getNumberofElements(AtomsNode->FirstChildElement("Atom"), "Atom");//getting number of atoms to allocate memory
	if (!Natom){
		cout << "Error: cell info not specified." << endl;
		return -1;
	}
	if (CellFile.length()) return ReadAtomsFromTXT(elements); //text file is preferable
	tinyxml2::XMLElement *atomNode = AtomsNode->FirstChildElement("Atom"); //text file is not provided so getting the data directly from XML
	for (unsigned int iAtom = 0; iAtom < Natom; iAtom++) {
		unsigned int Z = 0;		
		string name;
		error += GetAttribute(atomNode, "Atom", "name", name, true, "table value according to it's Z");
		if (!name.length())	{
			error += GetAttribute(atomNode, "Atom", "Z", Z, false);
			if (error) {
				cout << "Parsing error: Neither Z or name of the atom with number" << iAtom << "is set." << endl;
				return error;
			}
			name = elements[Z];
		}
		vect3d <double> r;
		double occ = 1.;
		double Uiso = 0;
		error += GetAttribute(atomNode, "Atom", "r", r, false);
		error += GetAttribute(atomNode, "Atom", "occ", occ, true, "1.0");
		error += GetAttribute(atomNode, "Atom", "Uiso", Uiso, true, "0");
		updateAtomicSites(name, r, occ, Uiso);
		atomNode = atomNode->NextSiblingElement("Atom");
	}
	return error;
}


//Reads the symmetry elements data
int block::ReadSymmElements(tinyxml2::XMLElement *SymmNode){
	Nsymm = getNumberofElements(SymmNode, "SymmEqPos") + 1;
	rSymm = new vect3d<double>[Nsymm];
	for (unsigned int i = 0; i<3; i++) RMsymm[i] = new vect3d<double>[Nsymm];
	rSymm[0].assign(0, 0, 0);
	RMsymm[0][0].assign(1., 0, 0);
	RMsymm[1][0].assign(0, 1., 0);
	RMsymm[2][0].assign(0, 0, 1.);
	if (!SymmNode) return 0;
	int error = 0;
	for (unsigned int iSymm = 1; iSymm < Nsymm; iSymm++) {
		error += GetAttribute(SymmNode, "SymmEqPos", "r", rSymm[iSymm], false);
		error += GetAttribute(SymmNode, "SymmEqPos", "R1", RMsymm[0][iSymm], false);
		error += GetAttribute(SymmNode, "SymmEqPos", "R2", RMsymm[1][iSymm], false);
		error += GetAttribute(SymmNode, "SymmEqPos", "R3", RMsymm[2][iSymm], false);
		SymmNode = SymmNode->NextSiblingElement("SymmEqPos");
	}
	return error;
}

//Reads the coordinates and Euler angles of the copies of the structural block
int block::ReadCopies(tinyxml2::XMLElement *CopiesNode, const map<string, unsigned int> word){
	if (!CopiesNode){//Just a single copy (Ncopy == 1)
		rCopy = new vect3d<double>[Ncopy];
		euler = new vect3d<double>[Ncopy];
		for (unsigned int i = 0; i<3; i++) RM[i] = new vect3d<double>[Ncopy];
		euler[0].assign(0, 0, 0);
		rCopy[0].assign(0, 0, 0);
		RM[0][0].assign(1., 0, 0);
		RM[1][0].assign(0, 1., 0);
		RM[2][0].assign(0, 0, 1.);
		return 0;
	}
	int error = 0;
	error += GetAttribute(CopiesNode, "Copies", "filename", CopiesFile);
	CopiesFile.length() ? Ncopy = getN(CopiesFile.c_str()) : Ncopy = getNumberofElements(CopiesNode->FirstChildElement("Copy"), "Copy");//getting number of copies to allocate memory
	error += GetWord(CopiesNode, "Copies", "convention", word, EulerConvention, true, "ZYX");
	//allcating memory for arrays
	rCopy = new vect3d<double>[Ncopy];
	euler = new vect3d<double>[Ncopy];
	for (unsigned int i = 0; i<3; i++) RM[i] = new vect3d<double>[Ncopy];
	if (CopiesFile.length()) return ReadCopiesFromTXT(); //text file is preferable		
	tinyxml2::XMLElement *copyNode = CopiesNode->FirstChildElement("Copy");//text file is not provided so getting the data directly from XML
	for (unsigned int iCopy = 0; iCopy < Ncopy; iCopy++) {
		error += GetAttribute(copyNode, "Copy", "r", rCopy[iCopy], false);
		error += GetAttribute(copyNode, "Copy", "Euler", euler[iCopy], false);
		euler[iCopy] = euler[iCopy] / 180.*PI;
		calcRotMatrix(&RM[0][iCopy], &RM[1][iCopy], &RM[2][iCopy], euler[iCopy], EulerConvention);
		copyNode = copyNode->NextSiblingElement("Copy");
	}
	return error;
}

//Reads the lattice vectors and crystalline size along each direction
int block::ReadLattice(tinyxml2::XMLElement *CellNode) {
	if (!CellNode) return 0;//Non-crystalline structure
	int error = 0;
	error += GetAttribute(CellNode, "CellVectors", "a", e[0], true, "(1,0,0)");
	error += GetAttribute(CellNode, "CellVectors", "b", e[1], true, "(0,1,0)");
	error += GetAttribute(CellNode, "CellVectors", "c", e[2], true, "(0,0,1)");
	error += GetAttribute(CellNode, "CellVectors", "N", Ncell, true, "(1,1,1)");
	return error;
}

//Reads the parameters of the sphere which will be cut-off from the atomic enseble of the single structural block (Rcut) and/or the complete set of its copies (RcutCopies)
int block::ReadCutOffSphere(tinyxml2::XMLElement *CutOffNode) {
	if (!CutOffNode) return 0;
	int error = 0;
	if (CutOffNode->Attribute("Rcut")) {
		error += GetAttribute(CutOffNode, "CutOff", "Rcut", Rcut, false);
		if (Rcut>1.e-2) cutoff = true;
	}
	if (CutOffNode->Attribute("RcutCopies")) {
		error += GetAttribute(CutOffNode, "CutOff", "RcutCopies", RcutCopies, false);
		if (RcutCopies>1.e-2) cutoffcopies = true;
	}
	return error;
}

int block::ReadBlockParameters(tinyxml2::XMLElement *BlockNode){
	string elements[] = { "na", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
		"Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
		"Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
		"Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
		"Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np",
		"Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Uun" };
	map<string, unsigned int> word;
	word["xzx"] = EulerXZX; word["xyx"] = EulerXYX; word["yxy"] = EulerYXY; word["yzy"] = EulerYZY; word["zyz"] = EulerZYZ; word["zxz"] = EulerZXZ;
	word["xzy"] = EulerXZY; word["xyz"] = EulerXYZ; word["yxz"] = EulerYXZ; word["yzx"] = EulerYZX; word["zyx"] = EulerZYX; word["zxy"] = EulerZXY;
	map<string, bool> flag;
	flag["yes"] = true; flag["no"] = false; flag["true"] = true; flag["false"] = false; flag["1"] = true; flag["0"] = false;
	int error = GetWord(BlockNode, "Block", "fractional", flag, fractional, true, "No");
	error += GetWord(BlockNode, "Block", "centered", flag, centered, true, "No");
	if (!fractional) {
		error += GetWord(BlockNode, "Block", "centeredAtoms", flag, centeredAtoms, true, "No");
		error += GetWord(BlockNode, "Block", "mol_rotation", flag, ro_mol, true, "No");
		error += GetAttribute(BlockNode, "Block", "mol_Uiso", dev_mol, true, "0");
		dev_mol = sqrt(dev_mol);
	}
	error += ReadCutOffSphere(BlockNode->FirstChildElement("CutOff"));
	error += ReadLattice(BlockNode->FirstChildElement("CellVectors"));
	error += ReadAtoms(BlockNode->FirstChildElement("Atoms"), elements);
	error += ReadCopies(BlockNode->FirstChildElement("Copies"), word);
	error += ReadSymmElements(BlockNode->FirstChildElement("SymmEqPos"));
	if (error) return error;
	centerAtoms();
	calcUniqueSites();
	redefAtoms();
	calcMean();
	return 0;
}