/*seq2vec.h header file by ValK*/
/*2019/5/7           version1.4*/
#ifndef __SEQ2VEC_H__
#define __SEQ2VEC_H__
#include<cstring>
#include<cstdlib>
#include<ctime>
#include "bp.h"
#include "rnn.h"
#include "lstm.h"
#include "gru.h"
#include "actvfunc.h"

class Seq2Vec
{
protected:
	int INUM;
	int HNUM;
	int ONUM;
	int DEPTH;
	int MAXTIME;
	int batch_size;
	double lr;
	double **input;
	double *expect;
	double error;
	double maxerror;
	neuron *output;
	std::string func_name;
public:
	virtual void SetFunction(const char *FunctionName)=0;
	virtual void SetBatchSize(const int __b)=0;
	virtual void SetLearningRate(const double __lr)=0;
	virtual void Calc(const char* __Typename,const int T)=0;
	virtual void Training(const char* __Typename,const int T)=0;
	virtual void ErrorCalc()=0;
	virtual void Datain(const char *__Typename,const char *EncoderFile,const char* OutputFile)=0;
	virtual void Dataout(const char *__Typename,const char *EncoderFile,const char* OutputFile)=0;
	virtual void TotalWork(const char *__Typename,const char *EncoderFile,const char *OutputFile,const char *Sequencedata,const char *Trainingdata)=0;
};

class NormalSeq2Vec:public Seq2Vec
{
private:
	NormalRNN *rnnencoder;
	NormalLSTM *lstmencoder;
	NormalGRU *gruencoder;
public:
	NormalSeq2Vec(const char*,int,int,int,int);
	~NormalSeq2Vec();
	void SetFunction(const char*);
	void SetLearningRate(const double);
	void SetBatchSize(const int);
	void Calc(const char*,const int);
	void Training(const char*,const int);
	void ErrorCalc();
	void Datain(const char*,const char*,const char*);
	void Dataout(const char*,const char*,const char*);
	void TotalWork(const char*,const char*,const char*,const char*,const char*);
};

class DeepSeq2Vec:public Seq2Vec
{
private:
	DeepRNN *rnnencoder;
	DeepLSTM *lstmencoder;
	DeepGRU *gruencoder;
public:
	DeepSeq2Vec(const char*,int,int,int,int,int);
	~DeepSeq2Vec();
	void SetFunction(const char*);
	void SetLearningRate(const double);
	void SetBatchSize(const int);
	void Calc(const char*,const int);
	void Training(const char*,const int);
	void ErrorCalc();
	void Datain(const char*,const char*,const char*);
	void Dataout(const char*,const char*,const char*);
	void TotalWork(const char*,const char*,const char*,const char*,const char*);
};

void Seq2VecDataMaker(const char*,const char*,const char*,const int);

#endif
