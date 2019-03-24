/*bp.h header file made by ValK*/
/*2019/3/24         version 0.2*/
#ifndef __BP_H__
#define __BP_H__

#include<iostream>
#include<cmath>
#include<ctime>
#include<cstring>

using namespace std;

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
		double learningrate;
		double error;
		string func_name;
		double ActivateFunction(double);
		double DiffFunction(double);
	public:
		NormalBP(int,int,int);
		~NormalBP();
		void INIT();
		void Calc();
		void ErrorCalc();
		double GetError();
		void Training();
		void Datain(const char*);
		void Dataout(const char*);
		void SetFunction(const char*);
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
		double learningrate;
		double error;
		string func_name;
		double ActivateFunction(double);
		double DiffFunction(double);
	public:
		DeepBP(int,int,int,int);
		~DeepBP();
		void INIT();
		void Calc();
		void ErrorCalc();
		double GetError();
		void Training();
		void Datain(const char*);
		void Dataout(const char*);
		void SetFunction(const char*);
};

#include "bpfunction.h"

#endif
