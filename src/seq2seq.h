/*seq2seq.h header file by ValK*/
/*2019/5/7           version1.5*/
#ifndef __SEQ2SEQ_H__
#define __SEQ2SEQ_H__

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "rnn.h"
#include "lstm.h"
#include "gru.h"
#include "actvfunc.h"

using namespace std;

/*abstract class Seq2Seq*/
struct seq_neuron
{
	double *in,*out,*w,bia,*diff;
	double *transw,transbia;
};

class Seq2Seq
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
		double **expect;
		double error;
		double maxerror;
		seq_neuron *output;
		string func_name;
		//output function is set as softmax so this doesn't work
	public:
		virtual void SetBatchSize(const int __b)=0;
		//this decides how many batches of data are used in training process
		virtual void SetLearningRate(const double __lr)=0;
		//this decides the speed of updating model,but if you set a large rate,network may be corrupted
		virtual void Calc(const char* __Typename,const int ET,const int DT)=0;
		//used to calculate the forward propagation
		virtual void Training(const char*,const int,const int)=0;
		//used to calculate the back propagation through time
		virtual void ErrorCalc(const int DT)=0;
		//used to calculate the loss of training set
		virtual void SetFunction(const char* function_name)=0;
		//set output activate function but the function has been set as softmax so this doesn't work
		virtual void Datain(const char* __Typename,const char* EncoderFile,const char* DecoderFile,const char* OutputFile)=0;
		//input data before calculation
		virtual void Dataout(const char* __Typename,const char* EncoderFile,const char* DecoderFile,const char* OutputFile)=0;
		//output data to save data.But if network is too large,this may take a lot of time
		virtual void TotalWork(const char* __Typename,const char* EncoderFile,const char* DecoderFile,const char* OutputFile,const char* QuestiondataName,const char* TrainingdataName)=0;
};
/*NormalSeq2Seq with only one hidden layer*/
class NormalSeq2Seq:public Seq2Seq
{
private:
	NormalRNN *rnnencoder;
	NormalRNN *rnndecoder;
	NormalLSTM *lstmencoder;
	NormalLSTM *lstmdecoder;
	NormalGRU *gruencoder;
	NormalGRU *grudecoder;
public:
	NormalSeq2Seq(const char*,int,int,int,int);
	~NormalSeq2Seq();
	void SetBatchSize(const int);
	void SetLearningRate(const double);
	void Calc(const char*,const int,const int);
	void Training(const char*,const int,const int);
	void ErrorCalc(const int);
	void SetFunction(const char*);
	void Datain(const char*,const char*,const char*,const char*);
	void Dataout(const char*,const char*,const char*,const char*);
	void TotalWork(const char*,const char*,const char*,const char*,const char*,const char*);
};
/*DeepSeq2Seq with deep neural networks*/
class DeepSeq2Seq:public Seq2Seq
{
private:
	DeepRNN *rnnencoder;
	DeepRNN *rnndecoder;
	DeepLSTM *lstmencoder;
	DeepLSTM *lstmdecoder;
	DeepGRU *gruencoder;
	DeepGRU *grudecoder;
public:
	DeepSeq2Seq(const char*,int,int,int,int,int);
	~DeepSeq2Seq();
	void SetBatchSize(const int);
	void SetLearningRate(const double);
	void Calc(const char*,const int,const int);
	void Training(const char*,const int,const int);
	void ErrorCalc(const int);
	void SetFunction(const char*);
	void Datain(const char*,const char*,const char*,const char*);
	void Dataout(const char*,const char*,const char*,const char*);
	void TotalWork(const char*,const char*,const char*,const char*,const char*,const char*);
};

#endif
