/*bp.h header file made by ValK*/
/*2019/5/5          version 1.0*/
#ifndef __BP_H__
#define __BP_H__

#include <iostream>
#include <cmath>
#include <ctime>
#include <cstring>

struct neuron
{
    double in,out,bia,diff;
    double *w;
};

class NormalBP
{
private:
    int INUM;
    int HNUM;
    int ONUM;
    int batch_size;
    neuron *hide;
    neuron *output;
    double *input;
    double *expect;
    double lr;
    double error;
    std::string func_name;
    double ActivateFunction(double);
    double DiffFunction(double);
public:
    NormalBP(int,int,int);
    ~NormalBP();
    void Init();
    void Calc();
    void ErrorCalc();
    double GetError();
    void Training();
    void Datain(const std::string&);
    void Dataout(const std::string&);
    void SetFunction(const std::string&);
    void SetLearningrate(double);
    void TotalWork(const std::string&,const std::string&,const std::string&);
};
class DeepBP
{
private:
    int INUM;
    int HNUM;
    int ONUM;
    int DEPTH;
    int batch_size;
    neuron *hlink;
    neuron **hide;
    neuron *output;
    double *input;
    double *expect;
    double lr;
    double error;
    std::string func_name;
    double ActivateFunction(double);
    double DiffFunction(double);
public:
    DeepBP(int,int,int,int);
    ~DeepBP();
    void Init();
    void Calc();
    void ErrorCalc();
    double GetError();
    void Training();
    void Datain(const std::string&);
    void Dataout(const std::string&);
    void SetFunction(const std::string&);
    void SetLearningrate(double);
    void TotalWork(const std::string&,const std::string&,const std::string&);
};

#endif
