/*activatefunction.h header file made by ValK*/
/*2019/3/15                       version 0.1*/
#ifndef __ACTIVATEFUNCTION_H__
#define __ACTIVATEFUNCTION_H__
#include<cmath>
using namespace std;


double sigmoid(double x)
{
	return 1.0/(1.0+exp(-x));
}
double diffsigmoid(double x)
{
	x=1.0/(1.0+exp(-x));
	return x*(1-x);
}
double tanh(double x)
{
	return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}
double difftanh(double x)
{
	x=tanh(x);
	return 1-x*x;
}
double relu(double x)
{
	return x>0? x:0;
}
double diffrelu(double x)
{
	return x>0? 1:0;
}
double leakyrelu(double x)
{
	return x>0? x:0.01*x;
}
double diffleakyrelu(double x)
{
	return x>0? 1:0.01;
}
double elu(double x)
{
	return x>0? (1+x):exp(x);
}
double diffelu(double x)
{
	return x>0? 1:exp(x);
}

#endif
