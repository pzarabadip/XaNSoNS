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

//Defines function of config class

#include "config.h"
#include "ReadXML_utils.h"

//Set the parameters to their defaults
void config::SetDefault(){
	source = xray;
	scenario = Debye;
	Nblocks = 0;
	q.N = 1024;
	Nfi = 1024;
	PDFtype = typePDF;
	PolarFactor = false;
	PrintAtoms = false;
	PrintPartialPDF = false;
	calcPartialIntensity = false;
	cutoff = false;
	sphere = false;
	damping = false;
	EulerConvention = EulerZXZ;
	lambda = -1.;
	Rcutoff = 0;
	Rsphere = 0;
	hist_bin = 0.001;
	qPDF = 0;
	p0 = 0.;
	Euler.min.assign(0, 0, 0);
	Euler.max.assign(0, 0, 0);
	Euler.N.assign(1, 1, 1);
	BoxCorner.assign(0, 0, 0);
	BoxEdge[0].assign(0, 0, 0);
	BoxEdge[1].assign(0, 0, 0);
	BoxEdge[2].assign(0, 0, 0);
}

//Reads properties of the parallelepiped box if cutoff is true.
int config::ReadBoxProps(tinyxml2::XMLElement *boxNode){
	if (!boxNode) {
		sphere = true;
		return 0;
	}
	sphere = false;
	int error = 0;
	error += GetAttribute(boxNode, "Box", "a", BoxEdge[0], false);
	error += GetAttribute(boxNode, "Box", "b", BoxEdge[1], false);
	error += GetAttribute(boxNode, "Box", "c", BoxEdge[2], false);
	error += GetAttribute(boxNode, "Box", "corner", BoxCorner, false);
	double Vol = ABS(BoxEdge[0].dot(BoxEdge[1] * BoxEdge[2]));
	if (Vol < 1.e-8) {
		cout << "Error. Box has zero volume. Check Calculation-->Sample-->Box-->a(b,c) values." << endl;
		error -= 1;
	}
	return error;
}

//Reads the parameters required for "2D" scenario
int config::ReadParameters2d(tinyxml2::XMLElement *calcNode){
	if (!calcNode) return -1;
	map<string, unsigned int> word;
	word["xzx"] = EulerXZX; word["xyx"] = EulerXYX; word["yxy"] = EulerYXY; word["yzy"] = EulerYZY; word["zyz"] = EulerZYZ; word["zxz"] = EulerZXZ;
	word["xzy"] = EulerXZY; word["xyz"] = EulerXYZ; word["yxz"] = EulerYXZ; word["yzx"] = EulerYZX; word["zyx"] = EulerZYX; word["zxy"] = EulerZXY;
	map<string, bool> flag;
	flag["yes"] = true; flag["no"] = false; flag["true"] = true; flag["false"] = false; flag["1"] = true; flag["0"] = false;
	int error = 0;
	if (source == xray) error += GetWord(calcNode, "Calculation", "PolarFactor", flag, PolarFactor, true, "No");
	if (PolarFactor) error += GetAttribute(calcNode, "Calculation", "wavelength", lambda, false);
	else error += GetAttribute(calcNode, "Calculation", "wavelength", lambda, true, "default");
	tinyxml2::XMLElement *tempNode = calcNode->FirstChildElement("q");
	error += GetAttribute(tempNode, "q", "min", q.min, false);
	error += GetAttribute(tempNode, "q", "max", q.max, false);
	error += GetAttribute(tempNode, "q", "N", q.N, true, "1024");
	error += GetAttribute(tempNode, "q", "Nfi", Nfi, true, "1024");
	if (lambda < 0) lambda = 4.*PI / q.max;
	tempNode = calcNode->FirstChildElement("Euler");
	error += GetWord(tempNode, "Euler", "convention", word, EulerConvention, true, "ZXZ");
	error += GetAttribute(tempNode, "Euler", "min", Euler.min, true, "0 0 0");
	error += GetAttribute(tempNode, "Euler", "max", Euler.max, true, "0 0 0");
	error += GetAttribute(tempNode, "Euler", "N", Euler.N, true, "1 1 1");
	Euler.max = Euler.max / 180.*PI;
	Euler.min = Euler.min / 180.*PI;
	return error;
}

//Reads the parameters required for "Debye" scenario
int config::ReadParametersDebye(tinyxml2::XMLElement *calcNode){
	if (!calcNode) return -1;
	map<string, bool> flag;
	flag["yes"] = true; flag["no"] = false; flag["true"] = true; flag["false"] = false; flag["1"] = true; flag["0"] = false;
	int error = 0;
	if (source == xray) error += GetWord(calcNode, "Calculation", "PolarFactor", flag, PolarFactor, true, "No");
	if (PolarFactor) error += GetAttribute(calcNode, "Calculation", "wavelength", lambda, false);
	else error += GetAttribute(calcNode, "Calculation", "wavelength", lambda, true, "default");
	error += GetWord(calcNode, "Calculation", "PartialIntensity", flag, calcPartialIntensity, true, "No");
	tinyxml2::XMLElement *tempNode = calcNode->FirstChildElement("q");
	error += GetAttribute(tempNode, "q", "min", q.min, false);
	error += GetAttribute(tempNode, "q", "max", q.max, false);
	error += GetAttribute(tempNode, "q", "N", q.N, true, "1024");
	if (lambda < 0) lambda = 4.*PI / q.max;
	tempNode = calcNode->FirstChildElement("Sample");
	error += GetAttribute(tempNode, "Sample", "Rcutoff", Rcutoff, true, "0");
	if (Rcutoff > 0.5) cutoff = true;
	if (cutoff) {
		Rsphere = Rcutoff;
		error += GetAttribute(tempNode, "Sample", "density", p0, true, "approximately calculated value");
		error += GetWord(tempNode, "Sample", "damping", flag, damping, true, "No");
		error += ReadBoxProps(tempNode->FirstChildElement("Box"));
		if (sphere) error += GetAttribute(tempNode, "Sample", "Rsphere", Rsphere, true, "cutoff radius");
		if (calcPartialIntensity) {
			cout << "\nWarning: PartialIntensity option will be ignored because it's not compatible with Sample-->Rcutoff option.\n" << endl;
			calcPartialIntensity = false;
		}
	}
	return error;
}

//Reads the parameters required for "Debye_hist" scenario
int config::ReadParametersDebye_hist(tinyxml2::XMLElement *calcNode){
	if (!calcNode) return -1;
	map<string, bool> flag;
	flag["yes"] = true; flag["no"] = false; flag["true"] = true; flag["false"] = false; flag["1"] = true; flag["0"] = false;
	int error = 0;
	if (source == xray) error += GetWord(calcNode, "Calculation", "PolarFactor", flag, PolarFactor, true, "No");
	if (PolarFactor) error += GetAttribute(calcNode, "Calculation", "wavelength", lambda, false);
	else error += GetAttribute(calcNode, "Calculation", "wavelength", lambda, true, "default");
	error += GetAttribute(calcNode, "Calculation", "hist_bin", hist_bin, true, "0.001");
	tinyxml2::XMLElement *tempNode = calcNode->FirstChildElement("q");
	error += GetAttribute(tempNode, "q", "min", q.min, false);
	error += GetAttribute(tempNode, "q", "max", q.max, false);
	error += GetAttribute(tempNode, "q", "N", q.N, true, "1024");
	if (lambda < 0) lambda = 4.*PI / q.max;
	tempNode = calcNode->FirstChildElement("Sample");
	error += GetAttribute(tempNode, "Sample", "Rcutoff", Rcutoff, true, "0");
	if (Rcutoff > 0.5) cutoff = true;
	if (cutoff) {
		Rsphere = Rcutoff;
		error += GetAttribute(tempNode, "Sample", "density", p0, true, "approximately calculated value");
		error += GetWord(tempNode, "Sample", "damping", flag, damping, true, "No");
		error += ReadBoxProps(tempNode->FirstChildElement("Box"));
		if (sphere) error += GetAttribute(tempNode, "Sample", "Rsphere", Rsphere, true, "cutoff radius");
	}
	return error;
}

//Reads the parameters required for "PDFonly" scenario
int config::ReadParametersPDFonly(tinyxml2::XMLElement *calcNode){
	if (!calcNode) return -1;
	map<string, unsigned int> word;
	word["rdf"] = typeRDF; word["pdf"] = typePDF; word["rpdf"] = typeRPDF;
	map<string, bool> flag;
	flag["yes"] = true; flag["no"] = false; flag["true"] = true; flag["false"] = false; flag["1"] = true; flag["0"] = false;
	int error = 0;
	error += GetAttribute(calcNode, "Calculation", "hist_bin", hist_bin, true, "0.001");
	tinyxml2::XMLElement *tempNode = calcNode->FirstChildElement("PDF");
	error += GetWord(tempNode, "PDF", "type", word, PDFtype, true, "PDF");
	if (source == xray) error += GetAttribute(tempNode, "PDF", "q", qPDF, true, "0");
	error += GetWord(tempNode, "PDF", "PrintPartial", flag, PrintPartialPDF, true, "No");
	tempNode = calcNode->FirstChildElement("Sample");
	error += GetAttribute(tempNode, "Sample", "density", p0, true, "approximately calculated value");
	error += GetAttribute(tempNode, "Sample", "Rcutoff", Rcutoff, true, "0");
	if (Rcutoff > 0.5) cutoff = true;
	if (cutoff) {
		Rsphere = Rcutoff;
		error += GetWord(tempNode, "Sample", "damping", flag, damping, true, "No");
		error += ReadBoxProps(tempNode->FirstChildElement("Box"));
		if (sphere) error += GetAttribute(tempNode, "Sample", "Rsphere", Rsphere, true, "cutoff radius");
	}
	return error;
}

//Reads the parameters required for "DebyePDF" scenario
int config::ReadParametersDebyePDF(tinyxml2::XMLElement *calcNode){
	if (!calcNode) return -1;
	map<string, unsigned int> word;
	word["rdf"] = typeRDF; word["pdf"] = typePDF; word["rpdf"] = typeRPDF;
	map<string, bool> flag;
	flag["yes"] = true; flag["no"] = false; flag["true"] = true; flag["false"] = false; flag["1"] = true; flag["0"] = false;
	int error = 0;
	if (source == xray) error += GetWord(calcNode, "Calculation", "PolarFactor", flag, PolarFactor, true, "No");
	if (PolarFactor) error += GetAttribute(calcNode, "Calculation", "wavelength", lambda, false);
	else error += GetAttribute(calcNode, "Calculation", "wavelength", lambda, true, "default");
	error += GetAttribute(calcNode, "Calculation", "hist_bin", hist_bin, true, "0.001");
	tinyxml2::XMLElement *tempNode = calcNode->FirstChildElement("q");
	error += GetAttribute(tempNode, "q", "min", q.min, false);
	error += GetAttribute(tempNode, "q", "max", q.max, false);
	error += GetAttribute(tempNode, "q", "N", q.N, true, "1024");
	if (lambda < 0) lambda = 4.*PI / q.max;
	tempNode = calcNode->FirstChildElement("PDF");
	error += GetWord(tempNode, "PDF", "type", word, PDFtype, true, "PDF");
	if (source == xray) error += GetAttribute(tempNode, "PDF", "q", qPDF, true, "0");
	error += GetWord(tempNode, "PDF", "PrintPartial", flag, PrintPartialPDF, true, "No");
	tempNode = calcNode->FirstChildElement("Sample");
	error += GetAttribute(tempNode, "Sample", "density", p0, true, "approximately calculated value");
	error += GetAttribute(tempNode, "Sample", "Rcutoff", Rcutoff, true, "0");
	if (Rcutoff > 0.5) cutoff = true;
	if (cutoff) {
		Rsphere = Rcutoff;
		error += GetWord(tempNode, "Sample", "damping", flag, damping, true, "No");
		error += ReadBoxProps(tempNode->FirstChildElement("Box"));
		if (sphere) error += GetAttribute(tempNode, "Sample", "Rsphere", Rsphere, true, "cutoff radius");
	}
	return error;
}

//Reads the parameters of calculation
int config::ReadParameters(tinyxml2::XMLElement *calcNode){
	if (!calcNode) return -1;
	map<string, unsigned int> word;
	word["xray"] = xray; word["neutron"] = neutron; word["debye"] = Debye; word["2d"] = s2D; word["debye_hist"] = Debye_hist;
	word["pdfonly"] = PDFonly; word["debyepdf"] = DebyePDF;
	map<string, bool> flag;
	flag["yes"] = true; flag["no"] = false; flag["true"] = true; flag["false"] = false; flag["1"] = true; flag["0"] = false;
	int error = 0;
	error += GetAttribute(calcNode, "Calculation", "name", name, false);
	error += GetWord(calcNode, "Calculation", "source", word, source, true, "Xray");
	error += GetWord(calcNode, "Calculation", "scenario", word, scenario, true, "Debye");
	error += GetWord(calcNode, "Calculation", "PrintAtoms", flag, PrintAtoms, true, "No");
	error += GetAttribute(calcNode, "Calculation", "FFfilename", FFfilename, false);
	switch (scenario){
	case s2D:
		error += ReadParameters2d(calcNode);
		break;
	case Debye:
		error += ReadParametersDebye(calcNode);
		break;
	case Debye_hist:
		error += ReadParametersDebye_hist(calcNode);
		break;
	case PDFonly:
		error += ReadParametersPDFonly(calcNode);
		break;
	case DebyePDF:
		error += ReadParametersDebyePDF(calcNode);
		break;
	}
	return error;
}