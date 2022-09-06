/*actvfunc.h header file made by ValK*/
/*2019/5/7                        version 1.0*/
#ifndef __ACTIVATEFUNCTION_H__
#define __ACTIVATEFUNCTION_H__

#include <cmath>

double sigmoid(double);
double diffsigmoid(double);
double tanh(double);
double difftanh(double);
double relu(double);
double diffrelu(double);
double leakyrelu(double);
double diffleakyrelu(double);
double elu(double);
double diffelu(double);
double clipgrad(double);

#endif
