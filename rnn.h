/*rnn.h header file made by ValK*/
/*2019/3/24          version 0.1*/
#ifndef __RNN_H__
#define __RNN_H__

struct rnn_neuron
{
	double *in,*out,bia,*wi,*wh,*diff;
	double transdiff,*transwi,*transwh;
};
class NormalRNN
{
	private:
		int INUM;
		int HNUM;
		int MAXTIME;
		rnn_neuron *hide;
	public:
		NormalRNN(int,int,int);
		~NormalRNN();
		void INIT();
		void Datain(const char*);
		void Dataout(const char*);
};
class DeepRNN
{
	private:
		int INUM;
		int HNUM;
		int DEPTH;
		int MAXTIME;
		rnn_neuron *hlink;
		rnn_neuron **hide;
	public:
		DeepRNN(int,int,int,int);
		~DeepRNN();
		void INIT();
		void Datain(const char*);
		void Dataout(const char*);
};

#include "rnnfunction.h"

#endif
