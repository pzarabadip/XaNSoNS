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

//General input and output functions and utility file operations here.
#include "config.h"

/**
Checks whether the file is available for reading.
Returns 0, if OK and -1 if not.

@param *filename  Path to file
*/
int checkFile(const char * const filename);

/**
Returns the number of valuable lines in the files used to obtain the sizes of the arrays.

@param *filename  Path to file
*/
int getN(const char * const filename);

/**
Returns the position of the first space/tab in the string.
Returns 0 if space/tab is not found.

@param line  String been processed
*/
int findSpace(const string line);

/**
Reads neutron scattering lengths from file. 
Return 0, if OK and -1 if error.

@param *filename  Path to file with neutron scattering lengths
@param *ID        Array of [chemical element name, index] pairs. The pairs are created in this function according to the file
@param *SL        Array of neutron scattering lengths for all chemical elements
@param *names     List of the names of all chemical elements (ions, isotopes) in the sample
*/
int readNeuData(const char * const filename, map<string, unsigned int> * const ID, map<unsigned int, string> * const EN, vector<double> * const SL, const vector <string> * const names);

/**
Reads x-ray atomic form-factors from file.
Return 0, if OK and -1 if error.

@param *filename  Path to file with neutron scattering lengths
@param *ID        Array of [chemical element name, index] pairs. The pairs are created in this function according to the file
@param *FF        X-ray atomic form-factor arrays for all chemical elements
@param *q         Scattering vector magnitude array
@param Nq         Number of points in the scattering vector magnitude array
@param *names     List of the names of all chemical elements (ions, isotopes) in the sample
*/
int loadFF(const char * const filename, map<string, unsigned int> * const ID, map<unsigned int, string> * const EN, vector <double *> * const FF, const double * const q, const unsigned int Nq, const vector <string> * const names);

/**
Prints final atomic ensemble to file.
Return 0, if OK and -1 if error.

@param *ra        Atomic coordinate array sorted by chemical elements
@param *NatomEl   Array containing the total number of atoms of each chemical element
@param name       Prefix of the file name
@param *ID        Array of [chemical element name, index] pairs. The pairs are created in this function according to the file
@param Ntot       Total number of atoms in the ensemble
*/
int printAtoms(const vector < vect3d<double> > * const ra, const unsigned int * const NatomEl, const string name, const map <unsigned int, string> EN, const unsigned int Ntot);

/**
Prints total pair-distribution function to file.
Return 0, if OK and -1 if error.

@param *PDF       Pair-distribution function, PDF(r)
@param *cfg       Parameters of simulation
*/
int printPDF(const double * const PDF, const config * const cfg);

/**
Prints partial pair-distribution function to file.
Return 0, if OK and -1 if error.

@param *PDF       Partial pair-distribution function, PDF(r)
@param *cfg       Parameters of simulation
@param ID         Array of [chemical element name, index] pairs
*/
int printPartialPDF(const double * const PDF, const config * const cfg, const map <unsigned int, string> EN);

/**
Prints 1D scattering intensity (powder difraction pattern) to file.
Return 0, if OK and -1 if error.

@param *I         Scattering intensity array 
@param *q         Scattering vector magnitude array
@param *cfg       Parameters of simulation
*/
int printI(const double * const I, const double * const q, const config * const cfg);

/**
Prints partial 1D scattering intensity to file.
Return 0, if OK and -1 if error.

@param *I         Scattering intensity array
@param *q         Scattering vector magnitude array
@param *cfg       Parameters of simulation
*/
int printPartialI(const double * const I, const double * const q, const config * const cfg);

/**
Prints 2D scattering intensity (single-crystal diffraction pattern) to file.
Return 0, if OK and -1 if error.

@param *I         Scattering intensity array
@param *cfg       Parameters of simulation
*/
int printI2(const double * const * const I, const config * const cfg);

//Checks whether the file is available for reading.
int checkFile(const char * const filename) {
	ifstream in(filename);
	if (in.is_open()) {
		in.close();
		return 0;
	}
	cout << "Error: cannot open file " << filename << endl;
	return -1;
}

//Returns the number of valuable lines in the files used to obtain the sizes of the arrays.
int getN(const char * const filename){
	unsigned int N = 0;
	ifstream in(filename);
	if (in.is_open()){
		string line;
		while (getline(in, line)){
			if ((line.size() < 2) || (line[0] == '#')) continue;
			istringstream streamline(line);
			string tmp;
			vect3d <double> rtmp;
			streamline >> tmp;
			streamline >> rtmp;
			if (!streamline.fail()) N++;
		}
		in.close();
		return N;
	}
	cout << "Error: cannot open file " << filename << endl;
	return -1;
}

//Returns the position of the first space/tab in the string.
int findSpace(const string line){
	int posS=(int)line.find(' ');
	int posT=(int)line.find('\t');
	if ((posS>0)&&(posT>0))	return MIN(posS,posT);
	if (posT > 0) return posT;
	if (posS > 0) return posS;
	return 0;
}

//Reads neutron scattering lengths from file. 
int readNeuData(const char * const filename, map<string, unsigned int> * const ID, map<unsigned int, string> * const EN, vector<double> * const SL, const set <string> * const names){
	cout << "Reading " << filename << endl;
	ifstream in(filename);
	if (in.is_open()){
		string line, elstr, otherstr;
		int err = 0;
		unsigned int id = 0, linenum=0;
		double iSL;
		while (getline(in, line)){
			linenum++;
			if ((line.size() < 2) || (line[0] == '#')) continue;
			int pos = findSpace(line);
			if (!pos) {
				cout << "\tError: file with atomic form factors has wrong format. Check line " << linenum << ":\n" << line << endl;
				err = -1;
				break;
			}
			elstr = line.substr(0, pos);//get element name
			if (find(names->begin(), names->end(), elstr) == names->end()) continue;//read only required scattering lengths
			otherstr = line.substr(pos + 1);
			istringstream streamline(otherstr);
			(*ID)[elstr] = id;
			(*EN)[id] = elstr;
			streamline >> iSL;
			SL->push_back(iSL);
			id++;
		}
		in.close();
		if (err) return err;
		for (set<string>::const_iterator it = names->begin(); it != names->end(); it++) {//check if file contain all scattering lengths
			if (!ID->count(*it)) {
				cout << "\tError: missing atomic form factor for " << *it << endl;
				err = -1;
			}
		}
		return err;
	}
	cout << "Error: cannot open file " << filename << endl;
	return -1;
}

//Reads x-ray atomic form-factors from file.
int loadFF(const char *  const filename, map<string, unsigned int> * const ID, map<unsigned int, string> * const EN, vector <double *> * const FF, const double * const q, const unsigned int Nq, const set <string> * const names){
	cout << "Reading " << filename << endl;
	ifstream in(filename);
	if (in.is_open()){
		vector <vector <double> > ff;
		vector <double> qff;
		int err = 0;
		unsigned int id = 0, linenum = 0;
		string line;
		while (getline(in, line)){
			linenum++;
			if ((line.size() < 2) || (line[0] == '#')) continue;
			int pos = findSpace(line);
			if (!pos) {
				cout << "\tError: file with atomic form factors has wrong format. Check line " << linenum << ":\n" << line << endl;
				err = -1;
				break;
			}
			string elstr = line.substr(0, pos);
			string otherstr = line.substr(pos + 1);
			istringstream streamline(otherstr);
			double value;
			if (elstr == "q") {
				while (streamline >> value)	qff.push_back(value);
				continue;
			}
			else if (find(names->begin(), names->end(), elstr) == names->end()) continue;//read only required form factors
			vector <double> tempvec;
			double *temp;
			temp = new double[Nq];
			FF->push_back(temp);
			(*ID)[elstr] = id;
			(*EN)[id] = elstr;
			while (streamline >> value)	tempvec.push_back(value);
			ff.push_back(tempvec);
			id++;
		}
		in.close();
		if (err) return err;
		for (set <string>::const_iterator it = names->begin(); it != names->end(); it++) {//check if file contain all x-ray atomic form factors
			if (!ID->count(*it)) {
				cout << "\tError: missing atomic form factor for " << *it << endl;
				err = -1;
			}
		}
		if (err) return err;
		//Interpolating
		if ((q[0] < qff.front()) || (q[Nq - 1] > qff.back())) {
			cout << "\tError: scattering vector magnitude range mismatch. Check the line labeled 'q' or change the scattering vector magnitude range in XML" << endl;
			return -1;
		}
		unsigned int iFFstart = 0;
		for (unsigned int iq = 0; iq < Nq; iq++) {
			for (unsigned int iff = iFFstart; iff < qff.size() - 1; iff++) {
				if ((q[iq] >= qff[iff]) && (q[iq] <= qff[iff + 1])){
					double dq = qff[iff] - qff[iff + 1];
					for (unsigned int i = 0; i < ff.size(); i++){
						double a = (ff[i][iff] - ff[i][iff + 1]) / dq;
						double b = (ff[i][iff + 1] * qff[iff] - ff[i][iff] * qff[iff + 1]) / dq;
						(*FF)[i][iq] = (a*q[iq] + b);
					}
					iFFstart = iff;
					break;
				}
			}
		}
		return 0;
	}
	cout << "Error: cannot open file " << filename << endl;
	return -1;
}

//Prints final atomic ensemble to file.
int printAtoms(const vector < vect3d<double> > * const ra, const unsigned int * const NatomEl, const string name, const map <unsigned int, string> EN, const unsigned int Ntot) {
	ofstream out;
	ostringstream outname;
	outname << name << "_atoms.xyz";
	out.open(outname.str().c_str());
	if (out.is_open()){
		out << Ntot << "\n\n";
		out.setf(ios::scientific);
		for (unsigned int iEl = 0; iEl < EN.size(); iEl++){
			for (unsigned int i = 0; i < NatomEl[iEl]; i++){
				out << EN.at(iEl) << " " << ra[iEl][i] << "\n";
			}
		}
		out.close();
		return 0;
	}
	cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
	return -1;
}

//Prints total pair-distribution function to file.
int printPDF(const double * const PDF, const config * const cfg) {
	ostringstream outname;
	string PDFtype_str;
	if (cfg->PDFtype == typeRDF) PDFtype_str = "RDF";
	else if (cfg->PDFtype == typePDF) PDFtype_str = "PDF";
	else if (cfg->PDFtype == typeRPDF) PDFtype_str = "rPDF";
	if (cfg->source == xray) outname << cfg->name << "_xray_" << PDFtype_str << ".txt";
	else outname << cfg->name << "_neut_" << PDFtype_str << ".txt";
	ofstream out;
	out.open(outname.str().c_str());
	if (out.is_open()){
		unsigned int NhistLast;
		for (NhistLast = cfg->Nhist - 1; NhistLast > 0; NhistLast--) {
			if (ABS(PDF[NhistLast])>1.e-10) break;
		}
		out.setf(ios::scientific);
		for (unsigned int i = 0; i < NhistLast + 1; i++)	out << (i + 0.5)*cfg->hist_bin << "	" << PDF[i] << "\n";
		out.close();
		return 0;
	}
	cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
	return -1;
}

//Prints partial pair - distribution function to file.
int printPartialPDF(const double * const PDF, const config * const cfg, const map <unsigned int, string> EN) {
	unsigned int count = 0;
	int error = 0;
	string PDFtype_str;
	if (cfg->PDFtype == typeRDF) PDFtype_str = "RDF";
	else if (cfg->PDFtype == typePDF) PDFtype_str = "PDF";
	else if (cfg->PDFtype == typeRPDF) PDFtype_str = "rPDF";
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, count += cfg->Nhist) {
			ostringstream outname;
			if (cfg->source == xray) outname << cfg->name << "_xray_partial_" << PDFtype_str << "_" << EN.at(iEl) << "_" << EN.at(jEl) << ".txt";
			else outname << cfg->name << "_neut_partial_" << PDFtype_str << "_" << EN.at(iEl) << "_" << EN.at(jEl) << ".txt";
			ofstream out;
			out.open(outname.str().c_str());
			if (out.is_open()){
				unsigned int NhistLast;
				for (NhistLast = cfg->Nhist - 1; NhistLast > 0; NhistLast--) {
					if (PDF[count + NhistLast]>1.e-10) break;
				}
				out.setf(ios::scientific);
				for (unsigned int i = 0; i < NhistLast + 1; i++)	out << (i + 0.5)*cfg->hist_bin << "	" << PDF[count + i] << "\n";
				out.close();
			}
			else {
				error -= 1;
				cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
			}
		}
	}
	return error;
}

//Prints 1D scattering intensity (powder difraction pattern) to file.
int printI(const double * const I, const double * const q, const config * const cfg){
	ostringstream outname;
	if (cfg->source == xray) outname << cfg->name << "_xray_1D.txt";
	else outname << cfg->name << "_neut_1D.txt";
	ofstream out;
	out.open(outname.str().c_str());
	if (out.is_open()){
		out.setf(ios::scientific);
		for (unsigned int iq = 0; iq<cfg->q.N; iq++)	out << q[iq] << "	" << I[iq] << "\n";
		out.close();
		return 0;
	}
	cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
	return -1;
}

//Prints partial 1D scattering intensity to file.
int printPartialI(const double * const I, const double * const q, const config * const cfg){
	unsigned int count = 0;
	int error = 0;
	for (unsigned int iB = 0; iB < cfg->Nblocks; iB++) {
		for (unsigned int jB = iB; jB < cfg->Nblocks; jB++, count += cfg->q.N) {
			ostringstream outname;
			if (cfg->source == xray) outname << cfg->name << "_xray_" << iB << "-" << jB << "_1D.txt";
			else outname << cfg->name << "_neut_" << iB << "-" << jB << "_1D.txt";
			ofstream out;
			out.open(outname.str().c_str());
			if (out.is_open()){
				out.setf(ios::scientific);
				for (unsigned int iq = 0; iq < cfg->q.N; iq++)	out << q[iq] << "	" << I[count + iq] << "\n";
				out.close();
			}
			else {
				error -= 1;
				cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
			}
		}
	}
	return error;
}

//Prints 2D scattering intensity (single-crystal diffraction pattern) to file.
int printI2(const double * const * const I, const config * const cfg){
	//raws - q; columns - fi
	ostringstream outname;
	if (cfg->source == xray) outname << cfg->name << "_xray_2D.txt";
	else outname << cfg->name << "_neut_2D.txt";
	ofstream out;
	out.open(outname.str().c_str());
	if (out.is_open()){
		out.setf(ios::scientific);
		for (unsigned int iq = 0; iq<cfg->q.N; iq++){
			for (unsigned int ifi = 0; ifi<cfg->Nfi; ifi++)	out << I[iq][ifi] << "	";
			out << "\n";
		}
		out.close();
		return 0;
	}
	cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
	return -1;
}
