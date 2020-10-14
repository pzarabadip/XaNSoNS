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

//utilities to read the parameters from XML

#ifndef _XML_utils_H_
#define _XML_utils_H_

#include "typedefs.h"
#include "tinyxml2.h"


extern int myid;

/**
Reads XML attribute value
Returns -1 if error and 0 if OK.

@param *xNode      XML element containing the attribute to read
@param *NodeName   Name of the XML element (used for stdout only)
@param *Attribute  XML attribute name
@param &obj	       Object to store the value
@param deflt       If true, the attribute is supposed to have the default value, so no error will be raised if the attribute is missing or empty or the XML element does not exist
@param *defval     Default value of the attribute (if any) as a text (used for stdout only)
*/
template <class X> int GetAttribute(tinyxml2::XMLElement* xNode, const char * const NodeName, const char * const Attribute, X &obj, const bool deflt = true, const char * const defval = "default value"){
	stringstream outs;
	const char* value = NULL;
	if (!xNode) {
		if (!deflt){
			if (!myid) cout << "Parsing error: " << NodeName << " element is missing." << endl;
			return -1;
		}
		if (!myid) cout << NodeName << "-->" << Attribute << " is set to " << defval << "." << endl;
		return 0;
	}
	value = xNode->Attribute(Attribute);
	if (!value){
		if (!deflt){
			if (!myid) cout << "Parsing error:  Attribute " << Attribute << " of " << NodeName << " element is missing." << endl;
			return -1;
		}
		if (!myid) cout << NodeName << "-->" << Attribute << " is set to " << defval << "." << endl;
		return 0;
	}
	outs << value;
	string tempstr(outs.str());
	if (!tempstr.length()){
		if (!deflt){
			if (!myid) cout << "Parsing error:  Attribute " << Attribute << " of " << NodeName << " element is empty." << endl;
			return -1;
		}
		if (!myid) cout << NodeName << "-->" << Attribute << " is set to " << defval << "." << endl;
		return 0;
	}
	outs >> obj;
	return 0;
}

/**
Reads XML attribute value as a text word (key) and converts it to the corresponding non-text value (e.g. 'No' to false)
Returns -1 if error and 0 if OK.

@param *xNode      XML element containing the attribute to read
@param *NodeName   Name of the XML element (used for stdout only)
@param *Attribute  XML attribute name
@param word        (key, value) pairs
@param &obj	       Object to store the value
@param deflt       If true, the attribute is supposed to have the default value, so no error will be raised if the attribute is missing or empty or the XML element does not exist
@param *defval     Default value of the attribute (if any) as a text (used for stdout only)
*/
template <class X> int GetWord(tinyxml2::XMLElement* xNode, const char * const NodeName, const char * const Attribute, const map <string, X> word, X &obj, const bool deflt = true, const char * const defval = "default value"){
	int error = 0;
	string tempstr;
	error = GetAttribute(xNode, NodeName, Attribute, tempstr, deflt, defval);
	if (error) return error;
	transform(tempstr.begin(), tempstr.end(), tempstr.begin(), ::tolower);
	if (tempstr.length()){
		if (!word.count(tempstr)){
			if (!myid) cout << "Parsing error:  Attribute " << Attribute << " of " << NodeName << " element has an unexpected value." << endl;
			return -2;
		}
		obj = word.at(tempstr);
	}
	return 0;
}
#endif