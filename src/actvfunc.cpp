
#include "actvfunc.h"
#include <cmath>

double sigmoid(double x){
    return 1.0/(1.0+std::exp(-x));
}
double diffsigmoid(double x){
    x=1.0/(1.0+std::exp(-x));
    return x*(1-x);
}
double tanh(double x){
    double t1=std::exp(x);
    double t2=1.0/t1;
    return (t1-t2)/(t1+t2);
}
double difftanh(double x){
    x=tanh(x);
    return 1-x*x;
}
double relu(double x){
    return x>0? x:0;
}
double diffrelu(double x){
    return x>0? 1:0;
}
double leakyrelu(double x){
    return x>0? x:0.01*x;
}
double diffleakyrelu(double x){
    return x>0? 1:0.01;
}
double elu(double x){
    return x>0? (1+x):std::exp(x);
}
double diffelu(double x){
    return x>0? 1:std::exp(x);
}
double clipgrad(double x){
    double upper_threshold=0.01;
    double lower_threshold=0.000001;
    double sign=x>0? 1:-1;
    x*=sign;
    if(x>upper_threshold){
        return sign*upper_threshold;
    }else if(x<lower_threshold){
        return lower_threshold*sign;
    }
    return x*sign;
}