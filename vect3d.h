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

//Simple 3d vector class is defined here

#ifndef _VECT3D_H_
#define _VECT3D_H_

#include "typedefs.h"

template <class T> class vect3d{
	//based on Will Perone 3D vector class, http://willperone.net/Code/vector3.php
public:
	T x, y, z;
	vect3d(){ x = 0; y = 0; z = 0; }
	vect3d(const T X, const T Y, const T Z) { x = X; y = Y; z = Z; }; //constructor
	void operator ()(const T X, const T Y, const T Z) { x = X; y = Y; z = Z; }
	void assign(const T X, const T Y, const T Z) { x = X; y = Y; z = Z; };
	bool operator==(const vect3d<T> &t){ return (x == t.x && y == t.y && z == t.z); };
	bool operator!=(const vect3d<T> &t){ return (x != t.x || y != t.y || z != t.z); };
	const vect3d <T> &operator=(const vect3d <T> &t){
		x = t.x; y = t.y; z = t.z;
		return *this;
	};
	const vect3d<T> operator -(void) const { return vect3d<T>(-x, -y, -z); }
	const vect3d <T> operator+(const vect3d <T> &t) const { return vect3d <T>(x + t.x, y + t.y, z + t.z); };
	const vect3d <T> operator-(const vect3d <T> &t) const { return vect3d <T>(x - t.x, y - t.y, z - t.z); };
	const vect3d <T> &operator+=(const vect3d <T> &t){
		x += t.x;	y += t.y;	z += t.z;
		return *this;
	};
	const vect3d <T> &operator-=(const vect3d <T> &t){
		x -= t.x;	y -= t.y;	z -= t.z;
		return *this;
	};
	const vect3d<T> &operator *=(const T &t){
		x *= t; y *= t; z *= t;
		return *this;
	}
	const vect3d<T> &operator /=(const T &t){
		x /= t; y /= t; z /= t;
		return *this;
	}
	const vect3d<T> operator *(const T &t) const {
		vect3d<T> temp;
		temp = *this;
		return temp *= t;
	}
	const vect3d<T> operator /(const T &t) const {
		vect3d<T> temp;
		temp = *this;
		return temp /= t;
	}

	const vect3d <T> operator*(const vect3d <T> &t) const {
		vect3d <T> temp(y*t.z - z*t.y, -x*t.z + z*t.x, x*t.y - y*t.x);
		return temp;
	};
	T dot(const vect3d <T> &t) const { return x*t.x + y*t.y + z*t.z; }
	T mag() const { return sqrt(SQR(x) + SQR(y) + SQR(z)); }
	T sqr() const { return SQR(x) + SQR(y) + SQR(z); }
	void normalize(){ *this /= mag(); }
	vect3d <T> norm() const { return *this / mag(); }
	void project(const vect3d <T> &t){ *this = t*(this->dot(t) / t.sqr()); }
	vect3d <T> proj(const vect3d <T> &t) const { return t*(this->dot(t) / t.sqr()); }
	const vect3d <T> operator % (const vect3d <T> &t) const {
		vect3d <T> temp(x*t.x, y*t.y, z*t.z);
		return temp;
	}
};

template <class T> ostream &operator<<(ostream &stream, vect3d <T> t){
	//vect3d output
	stream.setf(ios::scientific);
	stream << t.x << "  ";
	stream << t.y << "  ";
	stream << t.z;
	stream.unsetf(ios::scientific);	
	return stream;
};

template <class T> istream &operator>>(istream &stream, vect3d <T> &t){
	//vect3d input
	stream >> t.x >> t.y >> t.z;
	return stream;
};

//cosine for vect3d
template <class T> vect3d <T> cos(vect3d <T> t){
	vect3d <T> temp(cos(t.x), cos(t.y), cos(t.z));
	return temp;
}

// sine for vect3d
template <class T> vect3d <T> sin(vect3d <T> t){
	vect3d <T> temp(sin(t.x), sin(t.y), sin(t.z));
	return temp;
}

#endif
