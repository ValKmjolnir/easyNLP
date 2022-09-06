/*actvfunc.h header file made by ValK*/
/*2019/5/7                        version 1.0*/
#ifndef __ACTIVATEFUNCTION_H__
#define __ACTIVATEFUNCTION_H__

#include <cmath>

using namespace std;

double sigmoid(double x);
double diffsigmoid(double x);
double tanh(double x);
double difftanh(double x);
double relu(double x);
double diffrelu(double x);
double leakyrelu(double x);
double diffleakyrelu(double x);
double elu(double x);
double diffelu(double x);
double ClipGradient(double x);

#endif
