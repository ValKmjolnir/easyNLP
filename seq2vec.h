/*seq2vec.h header file by ValK*/
/*2019/3/29          version0.1*/
#ifndef __SEQ2VEC_H__
#define __SEQ2VEC_H__
#include<cstring>
#include<cstdlib>
#include<ctime>
#include "bp.h"
#include "rnn.h"
#include "lstm.h"
#include "gru.h"
#include "activatefunction.h"
using namespace std;

class NormalSeq2Vec
{
	private:
		int INUM;
		int HNUM;
		int ONUM;
		int MAXTIME;
		double learningrate;
		double **input;
		double *expect;
		double error;
		double maxerror;
		neuron *output;
		NormalRNN *rnnencoder;
		NormalLSTM *lstmencoder;
		NormalGRU *gruencoder;
	public:
		NormalSeq2Vec(const char*,int,int,int,int);
		~NormalSeq2Vec();
		void SetLearningRate(const double);
		void Calc(const char*,const int);
		void Training(const char*,const int);
		void ErrorCalc();
};

class DeepSeq2Vec
{
	private:
		int INUM;
		int HNUM;
		int ONUM;
		int DEPTH;
		int MAXTIME;
		double learningrate;
		double **input;
		double *expect;
		double error;
		double maxerror;
		neuron *output;
		DeepRNN *rnnencoder;
		DeepLSTM *lstmencoder;
		DeepGRU *gruencoder;
	public:
		DeepSeq2Vec(const char*,int,int,int,int,int);
		~DeepSeq2Vec();
		void SetLearningRate(const double);
		void Calc(const char*,const int);
		void Training(const char*,const int);
		void ErrorCalc();
};

NormalSeq2Vec::NormalSeq2Vec(const char* __Typename,int InputlayerNum,int HiddenlayerNum,int OutputlayerNum,int Maxtime)
{
	INUM=InputlayerNum;
	HNUM=HiddenlayerNum;
	ONUM=OutputlayerNum;
	MAXTIME=Maxtime;
	rnnencoder=NULL;
	lstmencoder=NULL;
	gruencoder=NULL;
	if(__Typename=="rnn")
	{
		rnnencoder=new NormalRNN(INUM,HNUM,MAXTIME);
		rnnencoder->INIT();
	}
	else if(__Typename=="lstm")
	{
		lstmencoder=new NormalLSTM(INUM,HNUM,MAXTIME);
		lstmencoder->INIT();
	}
	else if(__Typename=="gru")
	{
		gruencoder=new NormalGRU(INUM,HNUM,MAXTIME);
		gruencoder->INIT();
	}
	else
	{
		cout<<"Unexpected error occurred..."<<endl;
		cout<<"[Error]Unknown neural network name."<<endl;
		exit(0);
	}
	expect=new double[ONUM];
	input=new double*[INUM];
	for(int i=0;i<INUM;i++)
		input[i]=new double[MAXTIME];
	output=new neuron[ONUM];
	for(int i=0;i<ONUM;i++)
		output[i].w=new double[HNUM];
	for(int i=0;i<ONUM;i++)
	{
		output[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
		for(int j=0;j<HNUM;j++)
			output[i].w[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
	}
}

NormalSeq2Vec::~NormalSeq2Vec()
{
	delete rnnencoder;
	delete lstmencoder;
	delete gruencoder;
	delete []expect;
	for(int i=0;i<INUM;i++)
		delete []input[i];
	delete []input;
	for(int i=0;i<ONUM;i++)
		delete []output[i].w;
	delete []output;
}

void NormalSeq2Vec::SetLearningRate(const double __lr)
{
	learningrate=__lr;
}

void NormalSeq2Vec::Calc(const char* __Typename,const int T)
{
	if(__Typename=="rnn")
	{
		
	}
	else if(__Typename=="lstm")
	{
		
	}
	else if(__Typename=="gru")
	{
		
	}
	else
	{
		cout<<"Unexpected error occurred..."<<endl;
		cout<<"[Error]Unknown neural network name."<<endl;
		exit(0);
	}
}

void NormalSeq2Vec::Training(const char* __Typename,const int T)
{
	if(__Typename=="rnn")
	{
		
	}
	else if(__Typename=="lstm")
	{
		
	}
	else if(__Typename=="gru")
	{
		
	}
	else
	{
		cout<<"Unexpected error occurred..."<<endl;
		cout<<"[Error]Unknown neural network name."<<endl;
		exit(0);
	}
}

void NormalSeq2Vec::ErrorCalc()
{
	error=0;
	double trans;
	for(int i=0;i<ONUM;i++)
	{
		trans=expect[i]-output[i].out;
		error+=trans*trans;
	}
	error*=0.5;
	return;
}

DeepSeq2Vec::DeepSeq2Vec(const char* __Typename,int InputlayerNum,int HiddenlayerNum,int OutputlayerNum,int Depth,int Maxtime)
{
	INUM=InputlayerNum;
	HNUM=HiddenlayerNum;
	ONUM=OutputlayerNum;
	DEPTH=Depth-1;
	MAXTIME=Maxtime;
	rnnencoder=NULL;
	lstmencoder=NULL;
	gruencoder=NULL;
	if(__Typename=="rnn")
	{
		rnnencoder=new DeepRNN(INUM,HNUM,DEPTH,MAXTIME);
		rnnencoder->INIT();
	}
	else if(__Typename=="lstm")
	{
		lstmencoder=new DeepLSTM(INUM,HNUM,DEPTH,MAXTIME);
		lstmencoder->INIT();
	}
	else if(__Typename=="gru")
	{
		gruencoder=new DeepGRU(INUM,HNUM,DEPTH,MAXTIME);
		gruencoder->INIT();
	}
	else
	{
		cout<<"Unexpected error occurred..."<<endl;
		cout<<"[Error]Unknown neural network name."<<endl;
		exit(0);
	}
	expect=new double[ONUM];
	input=new double*[INUM];
	for(int i=0;i<INUM;i++)
		input[i]=new double[MAXTIME];
	output=new neuron[ONUM];
	for(int i=0;i<ONUM;i++)
		output[i].w=new double[HNUM];
	for(int i=0;i<ONUM;i++)
	{
		output[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
		for(int j=0;j<HNUM;j++)
			output[i].w[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
	}
}

DeepSeq2Vec::~DeepSeq2Vec()
{
	delete rnnencoder;
	delete lstmencoder;
	delete gruencoder;
	delete []expect;
	for(int i=0;i<INUM;i++)
		delete []input[i];
	delete []input;
	for(int i=0;i<ONUM;i++)
		delete []output[i].w;
	delete []output;
}

void DeepSeq2Vec::SetLearningRate(const double __lr)
{
	learningrate=__lr;
}

void DeepSeq2Vec::Calc(const char* __Typename,const int T)
{
	if(__Typename=="rnn")
	{
		
	}
	else if(__Typename=="lstm")
	{
		
	}
	else if(__Typename=="gru")
	{
		
	}
	else
	{
		cout<<"Unexpected error occurred..."<<endl;
		cout<<"[Error]Unknown neural network name."<<endl;
		exit(0);
	}
}

void DeepSeq2Vec::Training(const char* __Typename,const int T)
{
	if(__Typename=="rnn")
	{
		
	}
	else if(__Typename=="lstm")
	{
		
	}
	else if(__Typename=="gru")
	{
		
	}
	else
	{
		cout<<"Unexpected error occurred..."<<endl;
		cout<<"[Error]Unknown neural network name."<<endl;
		exit(0);
	}
}

void DeepSeq2Vec::ErrorCalc()
{
	error=0;
	double trans;
	for(int i=0;i<ONUM;i++)
	{
		trans=expect[i]-output[i].out;
		error+=trans*trans;
	}
	error*=0.5;
	return;
}
#endif
