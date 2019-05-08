/*bpfunction.h header file made by ValK*/
/*2019/5/5                  version 1.0*/
#ifndef __BPFUNCTION_H__
#define __BPFUNCTION_H__


#include "bp.h"
#include "activatefunction.h"

#include<iostream>
#include<ctime>
#include<fstream>
#include<cstdlib>
using namespace std;
NormalBP::NormalBP(int inputlayer_num,int hiddenlayer_num,int outputlayer_num)
{
	error=1e8;
	learningrate=0;
	func_name="Unknown";
	INUM=inputlayer_num;
	HNUM=hiddenlayer_num;
	ONUM=outputlayer_num;
	batch_size=1;
	input=new double[INUM];
	expect=new double[ONUM];
	hide=new neuron[HNUM];
	output=new neuron[ONUM];
	for(int i=0;i<HNUM;i++)
		hide[i].w=new double[INUM];
	for(int i=0;i<ONUM;i++)
		output[i].w=new double[HNUM];
}

NormalBP::~NormalBP()
{
	for(int i=0;i<HNUM;i++)
		delete []hide[i].w;
	for(int i=0;i<ONUM;i++)
		delete []output[i].w;
	delete []hide;
	delete []output;
	delete []input;
	delete []expect;
}

double NormalBP::ActivateFunction(double x)
{
	if(func_name=="Unknown")
	{
		cout<<"easyNLP>>[Error]You haven't chose a correct funtion.";
		system("pause");
		exit(0);
	}
	else if(func_name=="sigmoid")
		return sigmoid(x);
	else if(func_name=="tanh")
		return tanh(x);
	else if(func_name=="relu")
		return relu(x);
	else if(func_name=="leakyrelu")
		return leakyrelu(x);
	else if(func_name=="elu")
		return elu(x);
	else
	{
		cout<<"easyNLP>>[Error]You haven't chose a correct funtion.";
		system("pause");
		exit(0);
	}
}

double NormalBP::DiffFunction(double x)
{
	if(func_name=="Unknown")
	{
		cout<<"easyNLP>>[Error]You haven't chose a correct funtion.";
		system("pause");
		exit(0);
	}
	else if(func_name=="sigmoid")
		return diffsigmoid(x);
	else if(func_name=="tanh")
		return difftanh(x);
	else if(func_name=="relu")
		return diffrelu(x);
	else if(func_name=="leakyrelu")
		return diffleakyrelu(x);
	else if(func_name=="elu")
		return diffelu(x);
	else
	{
		cout<<"easyNLP>>[Error]You haven't chose a correct funtion.";
		system("pause");
		exit(0);
	}
}

void NormalBP::INIT()
{
	srand(unsigned(time(NULL)));
	for(int i=0;i<HNUM;i++)
	{
		hide[i].bia=(1+rand()%10)/10.0;
		for(int j=0;j<INUM;j++)
			hide[i].w[j]=(1+rand()%10)/50.0;
	}
	for(int i=0;i<ONUM;i++)
	{
		output[i].bia=(1+rand()%10)/10.0;
		for(int j=0;j<HNUM;j++)
			output[i].w[j]=(1+rand()%10)/50.0;
	}
	return;
}

void NormalBP::Calc()
{
	for(int i=0;i<HNUM;i++)
	{
		hide[i].in=hide[i].bia;
		for(int j=0;j<INUM;j++)
			hide[i].in+=hide[i].w[j]*input[j];
		hide[i].out=ActivateFunction(hide[i].in);
	}
	for(int i=0;i<ONUM;i++)
	{
		output[i].in=output[i].bia;
		for(int j=0;j<INUM;j++)
			output[i].in+=output[i].w[j]*hide[j].out;
		output[i].out=ActivateFunction(output[i].in);
	}
	return;
}

void NormalBP::ErrorCalc()
{
	double trans;
	error=0;
	for(int i=0;i<ONUM;i++)
	{
		trans=expect[i]-output[i].out;
		error+=trans*trans;
	}
	error*=0.5;
	return;
}

double NormalBP::GetError()
{
	return error;
}

void NormalBP::SetLearningrate(double __lr)
{
	learningrate=__lr;
}

void NormalBP::Training()
{
	for(int i=0;i<ONUM;i++)
		output[i].diff=(expect[i]-output[i].out)*DiffFunction(output[i].in);
	for(int i=0;i<HNUM;i++)
	{
		hide[i].diff=0;
		for(int j=0;j<ONUM;j++)
			hide[i].diff+=output[j].w[i]*output[j].diff;
		hide[i].diff*=DiffFunction(hide[i].in);
	}
	
	for(int i=0;i<ONUM;i++)
	{
		output[i].bia+=2*learningrate*output[i].diff;
		for(int j=0;j<HNUM;j++)
			output[i].w[j]+=learningrate*output[i].diff*hide[j].out;
	}
	for(int i=0;i<HNUM;i++)
	{
		hide[i].bia+=2*learningrate*hide[i].diff;
		for(int j=0;j<INUM;j++)
			hide[i].w[j]+=learningrate*hide[i].diff*input[j];
	}
	return;
}

void NormalBP::Datain(const char* FILENAME)
{
	ifstream fin(FILENAME);
	if(fin.fail())
	{
		cout<<"easyNLP>>[Error]Cannot open file."<<endl;
		system("pause");
		exit(0);
	}
	for(int i=0;i<HNUM;i++)
	{
		fin>>hide[i].bia;
		for(int j=0;j<INUM;j++)
			fin>>hide[i].w[j];
	}
	for(int i=0;i<ONUM;i++)
	{
		fin>>output[i].bia;
		for(int j=0;j<HNUM;j++)
			fin>>output[i].w[j];
	}
	fin.close();
}

void NormalBP::Dataout(const char* FILENAME)
{
	ofstream fout(FILENAME);
	if(fout.fail())
	{
		cout<<"easyNLP>>[Error]Cannot open file."<<endl;
		system("pause");
		exit(0);
	}
	for(int i=0;i<HNUM;i++)
	{
		fout<<hide[i].bia<<endl;
		for(int j=0;j<INUM;j++)
			fout<<hide[i].w[j]<<endl;
	}
	for(int i=0;i<ONUM;i++)
	{
		fout<<output[i].bia<<endl;
		for(int j=0;j<HNUM;j++)
			fout<<output[i].w[j]<<endl;
	}
	fout.close();
	cout<<"easyNLP>>Output finished"<<endl;
}

void NormalBP::SetFunction(const char* function_name)
{
	func_name=function_name;
}

void NormalBP::TotalWork(const char* dataFilename,const char *QuestiondataName,const char *TrainingdataName)
{
	if(!fopen(dataFilename,"r"))
	{
		INIT();
		Dataout(dataFilename);
		cout<<"easyNLP>>[NormalBP] Initializing completed.\n";
	}
	else
		Datain(dataFilename);
	double maxerror=1e8;
	int epoch=0;
	while(maxerror>0.01)
	{
		epoch++;
		maxerror=0;
		ifstream finq(QuestiondataName);
		ifstream fint(TrainingdataName);
		if(finq.fail()||fint.fail())
		{
			cout<<"easyNLP>>[Error]Cannot open data file!"<<endl;
			cout<<"easyNLP>>[Lack] "<<QuestiondataName<<" and "<<TrainingdataName<<endl;
			system("pause");
			exit(0);
		}
		for(int b=0;b<batch_size;b++)
		{
			for(int i=0;i<INUM;i++)
				finq>>input[i];
			for(int i=0;i<ONUM;i++)
				fint>>expect[i];
			Calc();
			ErrorCalc();
			Training();
			maxerror+=error;
		}
		finq.close();
		fint.close();
		if(epoch%10==0)
		{
			cout<<"easyNLP>>Epoch "<<epoch<<": Error :"<<maxerror<<endl;
			if(epoch%50==0)
				Dataout(dataFilename);
		}
	}
	cout<<"easyNLP>>Final output in progress..."<<endl;
	Dataout(dataFilename);
	cout<<"easyNLP>>Training complete."<<endl;
	return;
}

DeepBP::DeepBP(int inputlayer_num,int hiddenlayer_num,int outputlayer_num,int depth)
{
	error=1e8;
	learningrate=0;
	func_name="Unknown";
	INUM=inputlayer_num;
	HNUM=hiddenlayer_num;
	ONUM=outputlayer_num;
	batch_size=1;
	DEPTH=depth-1;
	input=new double[INUM];
	expect=new double[ONUM];
	hlink=new neuron[HNUM];
	hide=new neuron*[HNUM];
	output=new neuron[ONUM];
	for(int i=0;i<HNUM;i++)
		hide[i]=new neuron[DEPTH];
	for(int d=0;d<DEPTH;d++)
		for(int i=0;i<HNUM;i++)
			hide[i][d].w=new double[HNUM];
	for(int i=0;i<HNUM;i++)
		hlink[i].w=new double[INUM];
	for(int i=0;i<ONUM;i++)
		output[i].w=new double[HNUM];
}

DeepBP::~DeepBP()
{
	delete []input;
	delete []expect;
	for(int d=0;d<DEPTH;d++)
		for(int i=0;i<HNUM;i++)
			delete []hide[i][d].w;
	for(int i=0;i<HNUM;i++)
		delete []hide[i];
	for(int i=0;i<HNUM;i++)
		delete []hlink[i].w;
	for(int i=0;i<ONUM;i++)
		delete []output[i].w;
	delete []hlink;
	delete []hide;
	delete []output;
}

double DeepBP::ActivateFunction(double x)
{
	if(func_name=="Unknown")
	{
		cout<<"easyNLP>>[Error]You haven't chose a correct funtion.";
		system("pause");
		exit(0);
	}
	else if(func_name=="sigmoid")
		return sigmoid(x);
	else if(func_name=="tanh")
		return tanh(x);
	else if(func_name=="relu")
		return relu(x);
	else if(func_name=="leakyrelu")
		return leakyrelu(x);
	else if(func_name=="elu")
		return elu(x);
	else
	{
		cout<<"easyNLP>>[Error]You haven't chose a correct funtion.";
		system("pause");
		exit(0);
	}
}

double DeepBP::DiffFunction(double x)
{
	if(func_name=="Unknown")
	{
		cout<<"easyNLP>>[Error]You haven't chose a correct funtion.";
		system("pause");
		exit(0);
	}
	else if(func_name=="sigmoid")
		return diffsigmoid(x);
	else if(func_name=="tanh")
		return difftanh(x);
	else if(func_name=="relu")
		return diffrelu(x);
	else if(func_name=="leakyrelu")
		return diffleakyrelu(x);
	else if(func_name=="elu")
		return diffelu(x);
	else
	{
		cout<<"easyNLP>>[Error]You haven't chose a correct funtion.";
		system("pause");
		exit(0);
	}
}

void DeepBP::INIT()
{
	srand(unsigned(time(NULL)));
	for(int i=0;i<HNUM;i++)
	{
		hlink[i].bia=(1+rand()%10)/10.0;
		for(int j=0;j<INUM;j++)
			hlink[i].w[j]=(1+rand()%10)/50.0;
	}
	for(int d=0;d<DEPTH;d++)
		for(int i=0;i<HNUM;i++)
		{
			hide[i][d].bia=(1+rand()%10)/10.0;
			for(int j=0;j<HNUM;j++)
				hide[i][d].w[j]=(1+rand()%10)/50.0;
		}
	for(int i=0;i<ONUM;i++)
	{
		output[i].bia=(1+rand()%10)/10.0;
		for(int j=0;j<HNUM;j++)
			output[i].w[j]=(1+rand()%10)/50.0;
	}
	return;
}

void DeepBP::Calc()
{
	for(int i=0;i<HNUM;i++)
	{
		hlink[i].in=hlink[i].bia;
		for(int j=0;j<INUM;j++)
			hlink[i].in+=hlink[i].w[j]*input[j];
		hlink[i].out=ActivateFunction(hlink[i].in);
	}
	for(int d=0;d<DEPTH;d++)
		for(int i=0;i<HNUM;i++)
		{
			hide[i][d].in=hide[i][d].bia;
			for(int j=0;j<HNUM;j++)
				hide[i][d].in+=hide[i][d].w[j]*(d==0? hlink[j].out:hide[j][d-1].out);
			hide[i][d].out=ActivateFunction(hide[i][d].in);
		}
	for(int i=0;i<ONUM;i++)
	{
		output[i].in=output[i].bia;
		for(int j=0;j<HNUM;j++)
			output[i].in+=output[i].w[j]*hide[j][DEPTH-1].out;
		output[i].out=ActivateFunction(output[i].in);
	}
	return;
}

void DeepBP::ErrorCalc()
{
	double trans;
	error=0;
	for(int i=0;i<ONUM;i++)
	{
		trans=expect[i]-output[i].out;
		error+=trans*trans;
	}
	error*=0.5;
	return;
}

double DeepBP::GetError()
{
	return error;
}

void DeepBP::SetLearningrate(double __lr)
{
	learningrate=__lr;
}

void DeepBP::Training()
{
	for(int i=0;i<ONUM;i++)
		output[i].diff=(expect[i]-output[i].out)*DiffFunction(output[i].in);
	for(int i=0;i<HNUM;i++)
	{
		hide[i][DEPTH-1].diff=0;
		for(int j=0;j<ONUM;j++)
			hide[i][DEPTH-1].diff+=output[j].w[i]*output[j].diff;
		hide[i][DEPTH-1].diff*=DiffFunction(hide[i][DEPTH-1].in);
	}
	for(int d=DEPTH-2;d>=0;d--)
		for(int i=0;i<HNUM;i++)
		{
			hide[i][d].diff=0;
			for(int j=0;j<HNUM;j++)
				hide[i][d].diff+=hide[j][d+1].w[i]*hide[j][d+1].diff;
			hide[i][d].diff*=DiffFunction(hide[i][d].in);
		}
	for(int i=0;i<HNUM;i++)
	{
		hlink[i].diff=0;
		for(int j=0;j<HNUM;j++)
			hlink[i].diff+=hide[j][0].w[i]*hide[j][0].diff;
		hlink[i].diff*=DiffFunction(hlink[i].in);
	}
	
	for(int i=0;i<HNUM;i++)
	{
		hlink[i].bia+=2*learningrate*hlink[i].diff;
		for(int j=0;j<INUM;j++)
			hlink[i].w[j]+=learningrate*hlink[i].diff*input[j];
	}
	for(int d=0;d<DEPTH;d++)
		for(int i=0;i<HNUM;i++)
		{
			hide[i][d].bia+=2*learningrate*hide[i][d].diff;
			for(int j=0;j<HNUM;j++)
				hide[i][d].w[j]+=learningrate*hide[i][d].diff*(d==0? hlink[j].out:hide[j][d-1].out);
		}
	for(int i=0;i<ONUM;i++)
	{
		output[i].bia+=2*learningrate*output[i].diff;
		for(int j=0;j<HNUM;j++)
			output[i].w[j]+=learningrate*output[i].diff*hide[j][DEPTH-1].out;
	}
	return;
}

void DeepBP::Datain(const char* FILENAME)
{
	ifstream fin(FILENAME);
	if(fin.fail())
	{
		cout<<"easyNLP>>[Error]Cannot open file."<<endl;
		system("pause");
		exit(0);
	}
	for(int i=0;i<HNUM;i++)
	{
		fin>>hlink[i].bia;
		for(int j=0;j<INUM;j++)
			fin>>hlink[i].w[j];
	}
	for(int d=0;d<DEPTH;d++)
		for(int i=0;i<HNUM;i++)
		{
			fin>>hide[i][d].bia;
			for(int j=0;j<HNUM;j++)
				fin>>hide[i][d].w[j];
		}
	for(int i=0;i<ONUM;i++)
	{
		fin>>output[i].bia;
		for(int j=0;j<HNUM;j++)
			fin>>output[i].w[j];
	}
	fin.close();
}

void DeepBP::Dataout(const char* FILENAME)
{
	ofstream fout(FILENAME);
	if(fout.fail())
	{
		cout<<"easyNLP>>[Error]Cannot open file."<<endl;
		system("pause");
		exit(0);
	}
	for(int i=0;i<HNUM;i++)
	{
		fout<<hlink[i].bia<<endl;
		for(int j=0;j<INUM;j++)
			fout<<hlink[i].w[j]<<endl;
	}
	for(int d=0;d<DEPTH;d++)
		for(int i=0;i<HNUM;i++)
		{
			fout<<hide[i][d].bia<<endl;
			for(int j=0;j<HNUM;j++)
				fout<<hide[i][d].w[j]<<endl;
		}
	for(int i=0;i<ONUM;i++)
	{
		fout<<output[i].bia<<endl;
		for(int j=0;j<HNUM;j++)
			fout<<output[i].w[j]<<endl;
	}
	fout.close();
	cout<<"easyNLP>>Output finished"<<endl;
}

void DeepBP::SetFunction(const char* function_name)
{
	func_name=function_name;
}

void DeepBP::TotalWork(const char* dataFilename,const char *QuestiondataName,const char *TrainingdataName)
{
	if(!fopen(dataFilename,"r"))
	{
		INIT();
		Dataout(dataFilename);
		cout<<"easyNLP>>[DeepBP] Initializing completed.\n";
	}
	else
		Datain(dataFilename);
	double maxerror=1e8;
	int epoch=0;
	while(maxerror>0.01)
	{
		epoch++;
		maxerror=0;
		ifstream finq(QuestiondataName);
		ifstream fint(TrainingdataName);
		if(finq.fail()||fint.fail())
		{
			cout<<"easyNLP>>[Error]Cannot open data file!"<<endl;
			cout<<"easyNLP>>[Lack] "<<QuestiondataName<<" and "<<TrainingdataName<<endl;
			system("pause");
			exit(0);
		}
		for(int b=0;b<batch_size;b++)
		{
			for(int i=0;i<INUM;i++)
				finq>>input[i];
			for(int i=0;i<ONUM;i++)
				fint>>expect[i];
			Calc();
			ErrorCalc();
			Training();
			maxerror+=error;
		}
		finq.close();
		fint.close();
		if(epoch%10==0)
		{
			cout<<"easyNLP>>Epoch "<<epoch<<": Error :"<<maxerror<<endl;
			if(epoch%50==0)
				Dataout(dataFilename);
		}
	}
	cout<<"easyNLP>>Final output in progress..."<<endl;
	Dataout(dataFilename);
	cout<<"easyNLP>>Training complete."<<endl;
	return;
}

#endif
