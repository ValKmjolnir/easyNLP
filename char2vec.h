/*char2vec.h header file made by valk*/
/*2019/5/2               version 1.1*/
#ifndef __CHAR2VEC_H__
#define __CHAR2VEC_H__
#include<iostream>
#include<fstream>
#include<cmath>
#include<cstdlib>
#include "bp.h"
#include "activatefunction.h"
using namespace std;

class Char2Vec
{
	private:
		int INUM;
		int HNUM;
		int ONUM;
		int **cnt;
		double *input;
		double *expect;
		double learningrate;
		neuron *hide;
		neuron *output;
	public:
		Char2Vec(const int __Hnum=256)
		{
			learningrate=0.1;
			INUM=95;
			ONUM=95;
			HNUM=__Hnum;
			cnt=new int*[95];
			for(int i=0;i<95;i++)
				cnt[i]=new int[95];
			expect= new double[95];
			input = new double[95];
			hide=new neuron[HNUM];
			output=new neuron[ONUM];
			for(int i=0;i<HNUM;i++)
				hide[i].w=new double[INUM];
			for(int i=0;i<ONUM;i++)
				output[i].w=new double[HNUM];

			srand(unsigned(time(NULL)));
			for(int i=0;i<HNUM;i++)
			{
				hide[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)*0.1;
				for(int j=0;j<INUM;j++)
					hide[i].w[j]=(rand()%2? 1:-1)*(1.0+rand()%10)*0.02;
			}
			for(int i=0;i<ONUM;i++)
			{
				output[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)*0.1;
				for(int j=0;j<HNUM;j++)
					output[i].w[j]=(rand()%2? 1:-1)*(1.0+rand()%10)*0.02;
			}
			return;
		}
		~Char2Vec()
		{
			for(int i=0;i<95;i++)
				delete []cnt[i];
			delete []cnt;
			delete []expect;
			delete []input;
			for(int i=0;i<HNUM;i++)
				delete []hide[i].w;
			delete []hide;
			for(int i=0;i<ONUM;i++)
				delete []output[i].w;
			delete []output;
		}
		void TotalWork(const char*,const char*);
		void Mainwork(const char*);
		void Calc();
		void Training();
		void Datain(const char*);
		void Dataout(const char*);
		void Print();
		void CountChar(const char*);
		void CharDataIllustration(const char*);
};

void Char2Vec::TotalWork(const char *dataFilename,const char *TrainingdataName)
{
	if(!fopen(dataFilename,"r"))
	{
		Dataout(dataFilename);
		cout<<"easyNLP>>[Char2Vec-95char] Initializing completed.\n";
	}
	else
		Datain(dataFilename);
	CountChar(TrainingdataName);
	Mainwork(dataFilename);
	Print();
}

void Char2Vec::Mainwork(const char *Filename)
{
	int epoch=0;
	double maxerror=1e8;
	double error=1e8;
	double softmax=0;
	while(maxerror>0.1)
	{
		epoch++;
		maxerror=0;
		for(int i=0;i<95;i++)
		{
			for(int j=0;j<INUM;j++)
				input[j]=0;
			input[i]=1;
			softmax=0;
			for(int j=0;j<95;j++)
				softmax+=exp(cnt[i][j]);
			for(int j=0;j<ONUM;j++)
				expect[j]=exp(cnt[i][j])/softmax;
			Calc();
			error=0;
			for(int j=0;j<ONUM;j++)
				error+=(expect[j]-output[j].out)*(expect[j]-output[j].out);
			error*=0.5;
			maxerror+=error;
			Training();
		}
		if(epoch%10==0)
		{
			cout<<"easyNLP>>Epoch "<<epoch<<": Error :"<<maxerror<<endl;
			if(epoch%50==0)
				Dataout(Filename);
		}
	}
	cout<<"easyNLP>>Final output in progress..."<<endl;
	Dataout(Filename);
	cout<<"easyNLP>>Training complete."<<endl;
	return;
}
void Char2Vec::Calc()
{
	double softmax=0;
	for(int i=0;i<HNUM;i++)
	{
		hide[i].in=hide[i].bia;
		for(int j=0;j<INUM;j++)
			hide[i].in+=hide[i].w[j]*input[j];
		hide[i].out=tanh(hide[i].in);
	}
	for(int i=0;i<ONUM;i++)
	{
		output[i].in=output[i].bia;
		for(int j=0;j<HNUM;j++)
			output[i].in+=output[i].w[j]*hide[j].out;
		softmax+=exp(output[i].in);
	}
	for(int i=0;i<ONUM;i++)
		output[i].out=exp(output[i].in)/softmax;
	return;
}
void Char2Vec::Training()
{
	for(int i=0;i<ONUM;i++)
		output[i].diff=(expect[i]-output[i].out)*output[i].out*(1-output[i].out);
	for(int i=0;i<HNUM;i++)
	{
		hide[i].diff=0;
		for(int j=0;j<ONUM;j++)
			hide[i].diff+=output[j].diff*output[j].w[i];
	}
	for(int i=0;i<ONUM;i++)
	{
		output[i].bia+=learningrate*2*output[i].diff;
		for(int j=0;j<HNUM;j++)
			output[i].w[j]+=learningrate*output[i].diff*hide[j].out;
	}
	for(int i=0;i<HNUM;i++)
	{
		hide[i].bia+=learningrate*2*hide[i].diff;
		for(int j=0;j<INUM;j++)
			hide[i].w[j]+=learningrate*hide[i].diff*input[j];
	}
	return;
}
void Char2Vec::Datain(const char *Filename)
{
	ifstream fin(Filename);
	if(fin.fail())
	{
		cout<<"easyNLP>>[Error]Cannot open data file!"<<endl;
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
	return;
}
void Char2Vec::Dataout(const char *Filename)
{
	ofstream fout(Filename);
	if(fout.fail())
	{
		cout<<"easyNLP>>[Error]Cannot open data file!"<<endl;
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
	cout<<"easyNLP>>Output Finished.\n";
	return;
}
void Char2Vec::Print()
{
	cout<<"easyNLP>>[Result-Char2Vec-95char]"<<endl;
	for(int i=0;i<95;i++)
	{
		for(int j=0;j<INUM;j++)
			input[j]=0;
		input[i]=1;
		Calc();
	cout<<"        |"<<(char)(i+32)<<":  ";
		for(int j=0;j<ONUM;j++)
			if(output[j].out>0.1)
				cout<<"|"<<(char)(j+32)<<':'<<100*output[j].out<<"% ";
		cout<<endl;
	}
	return;
}
void Char2Vec::CountChar(const char *Filename)
{
	for(int i=0;i<95;i++)
		for(int j=0;j<95;j++)
			cnt[i][j]=0;
	char temp[1024];
	ifstream fin(Filename);
	if(fin.fail())
	{
		cout<<"easyNLP>>[Error]Cannot open data file!"<<endl;
		cout<<"easyNLP>>[Lack] "<<Filename<<endl;
		system("pause");
		exit(0);
	}
	while(!fin.eof())
	{
		for(int i=0;i<1024;i++)
			temp[i]=0;
		fin.getline(temp,1024,'\n');
		if(fin.eof())
			break;
		for(int i=1;temp[i]!=0;i++)
			if(temp[i-1]>=32&&temp[i-1]<=126&&temp[i]>=32&&temp[i]<=126)
				cnt[temp[i-1]-32][temp[i]-32]++;
	}
	return;
}
void Char2Vec::CharDataIllustration(const char* Filename)
{
	fstream fout(Filename,ios::out);
	if(fout.fail())
	{
		cout<<"easyNLP>>[Error]Cannot open data file!"<<endl;
		system("pause");
		exit(0);
	}
	fout<<"# ";
	for(int i=0;i<95;i++)
	{
		if(i==0)
			fout<<"space ";
		else
			fout<<(char)(i+32)<<' ';
	}
	fout<<endl;
	for(int i=0;i<95;i++)
	{
		if(i==0)
			fout<<"space ";
		else
			fout<<(char)(i+32)<<' ';
		for(int j=0;j<95;j++)
			fout<<cnt[i][j]/100.0<<' ';
		fout<<endl;
	}
	fout.close();
}
#endif
