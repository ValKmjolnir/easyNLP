/*seq2seq.h header file by ValK*/
/*2019/4/16          version0.3*/
#ifndef __SEQ2SEQ_H__
#define __SEQ2SEQ_H__
#include<cstring>
#include<cstdlib>
#include<ctime>
#include "rnn.h"
#include "lstm.h"
#include "gru.h"
#include "activatefunction.h"
using namespace std;

class Seq2Seq
{
	protected:
		int INUM;
		int HNUM;
		int ONUM;
		int DEPTH;
		int MAXTIME;
		double learningrate;
		double **input;
		double **expect;
		double error;
		double maxerror;
		seq_neuron *output;
		string func_name;//output function
	public:
		virtual void SetLearningRate(const double __lr)=0;
		virtual void Calc(const char* __Typename,const int ET,const int DT)=0;
		virtual void Training(const char*,const int,const int)=0;
		virtual void ErrorCalc(const int DT)=0;
		virtual void SetFunction(const char* function_name)=0;
		virtual void Datain(const char* __Typename,const char* EncoderFile,const char* DecoderFile,const char* OutputFile)=0;
		virtual void Dataout(const char* __Typename,const char* EncoderFile,const char* DecoderFile,const char* OutputFile)=0;
};

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
		void SetLearningRate(const double);
		void Calc(const char*,const int,const int);
		void Training(const char*,const int,const int);
		void ErrorCalc(const int);
		void SetFunction(const char*);
		void Datain(const char*,const char*,const char*,const char*);
		void Dataout(const char*,const char*,const char*,const char*);
};

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
		void SetLearningRate(const double);
		void Calc(const char*,const int,const int);
		void Training(const char*,const int,const int);
		void ErrorCalc(const int);
		void SetFunction(const char*);
		void Datain(const char*,const char*,const char*,const char*);
		void Dataout(const char*,const char*,const char*,const char*);
};

NormalSeq2Seq::NormalSeq2Seq(const char* __Typename,int InputlayerNum,int HiddenlayerNum,int OutputlayerNum,int Maxtime)
{
	srand(unsigned(time(NULL)));
	INUM=InputlayerNum;
	HNUM=HiddenlayerNum;
	ONUM=OutputlayerNum;
	MAXTIME=Maxtime;
	rnnencoder=NULL;
	rnndecoder=NULL;
	lstmencoder=NULL;
	lstmdecoder=NULL;
	gruencoder=NULL;
	grudecoder=NULL;
	if(__Typename=="rnn")
	{
		rnnencoder=new NormalRNN(INUM,HNUM,MAXTIME);
		rnndecoder=new NormalRNN(ONUM,HNUM,MAXTIME);
		rnnencoder->INIT();
		rnndecoder->INIT();
	}
	else if(__Typename=="lstm")
	{
		lstmencoder=new NormalLSTM(INUM,HNUM,MAXTIME);
		lstmdecoder=new NormalLSTM(ONUM,HNUM,MAXTIME);
		lstmencoder->INIT();
		lstmdecoder->INIT();
	}
	else if(__Typename=="gru")
	{
		gruencoder=new NormalGRU(INUM,HNUM,MAXTIME);
		grudecoder=new NormalGRU(ONUM,HNUM,MAXTIME);
		gruencoder->INIT();
		grudecoder->INIT();
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		system("pause");
		exit(0);
	}
	input=new double*[INUM];
	for(int i=0;i<INUM;i++)
		input[i]=new double[MAXTIME];
	expect=new double*[ONUM];
	for(int i=0;i<ONUM;i++)
		expect[i]=new double[MAXTIME];
	output=new seq_neuron[ONUM];
	for(int i=0;i<ONUM;i++)
	{
		output[i].in=new double[MAXTIME];
		output[i].out=new double[MAXTIME];
		output[i].diff=new double[MAXTIME];
		output[i].w=new double[HNUM];
		output[i].transw=new double[HNUM];
	}
	for(int i=0;i<ONUM;i++)
	{
		output[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
		for(int j=0;j<HNUM;j++)
			output[i].w[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
	}
}

NormalSeq2Seq::~NormalSeq2Seq()
{
	delete rnnencoder;
	delete rnndecoder;
	delete lstmencoder;
	delete lstmdecoder;
	delete gruencoder;
	delete grudecoder;
	for(int i=0;i<INUM;i++)
		delete []input[i];
	delete []input;
	for(int i=0;i<ONUM;i++)
		delete []expect[i];
	delete []expect;
	for(int i=0;i<ONUM;i++)
	{
		delete []output[i].in;
		delete []output[i].out;
		delete []output[i].diff;
		delete []output[i].w;
		delete []output[i].transw;
	}
	delete []output;
}

void NormalSeq2Seq::SetLearningRate(const double __lr)
{
	learningrate=__lr;
}

void NormalSeq2Seq::Calc(const char* __Typename,const int ET,const int DT)
{
	if(__Typename=="rnn")
	{
		double softmax_max;
		for(int t=1;t<=ET;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				rnnencoder->hide[i].in[t]=rnnencoder->hide[i].bia;
				for(int j=0;j<INUM;j++)
					rnnencoder->hide[i].in[t]+=rnnencoder->hide[i].wi[j]*input[j][t];
				for(int j=0;j<HNUM;j++)
					rnnencoder->hide[i].in[t]+=rnnencoder->hide[i].wh[j]*rnnencoder->hide[j].out[t-1];
				rnnencoder->hide[i].out[t]=tanh(rnnencoder->hide[i].in[t]);
			}
		}
		for(int i=0;i<HNUM;i++)
			rnndecoder->hide[i].out[0]=rnnencoder->hide[i].out[ET];
		softmax_max=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in[0]=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in[0]+=output[i].w[j]*rnndecoder->hide[j].out[0];
			softmax_max+=exp(output[i].in[0]);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out[0]=exp(output[i].in[0])/softmax_max;
		for(int t=1;t<=DT;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				rnndecoder->hide[i].in[t]=rnndecoder->hide[i].bia;
				for(int j=0;j<ONUM;j++)
					rnndecoder->hide[i].in[t]+=rnndecoder->hide[i].wi[j]*expect[j][t-1];//output[j].out[t-1];
				for(int j=0;j<HNUM;j++)
					rnndecoder->hide[i].in[t]+=rnndecoder->hide[i].wh[j]*rnndecoder->hide[j].out[t-1];
				rnndecoder->hide[i].out[t]=tanh(rnndecoder->hide[i].in[t]);
			}
			softmax_max=0;
			for(int i=0;i<ONUM;i++)
			{
				output[i].in[t]=output[i].bia;
				for(int j=0;j<HNUM;j++)
					output[i].in[t]+=output[i].w[j]*rnndecoder->hide[j].out[t];
				softmax_max+=exp(output[i].in[t]);
			}
			for(int i=0;i<ONUM;i++)
				output[i].out[t]=exp(output[i].in[t])/softmax_max;
		}
		return;
	}
	else if(__Typename=="lstm")
	{
		double softmax_max;
		for(int t=1;t<=ET;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				lstmencoder->hide[i].fog_in[t]=lstmencoder->hide[i].fog_bia;
				lstmencoder->hide[i].sig_in[t]=lstmencoder->hide[i].sig_bia;
				lstmencoder->hide[i].tan_in[t]=lstmencoder->hide[i].tan_bia;
				lstmencoder->hide[i].out_in[t]=lstmencoder->hide[i].out_bia;
				for(int j=0;j<INUM;j++)
				{
					lstmencoder->hide[i].fog_in[t]+=lstmencoder->hide[i].fog_wi[j]*input[j][t];
					lstmencoder->hide[i].sig_in[t]+=lstmencoder->hide[i].sig_wi[j]*input[j][t];
					lstmencoder->hide[i].tan_in[t]+=lstmencoder->hide[i].tan_wi[j]*input[j][t];
					lstmencoder->hide[i].out_in[t]+=lstmencoder->hide[i].out_wi[j]*input[j][t];
				}
				for(int j=0;j<HNUM;j++)
				{
					lstmencoder->hide[i].fog_in[t]+=lstmencoder->hide[i].fog_wh[j]*lstmencoder->hide[j].out[t-1];
					lstmencoder->hide[i].sig_in[t]+=lstmencoder->hide[i].sig_wh[j]*lstmencoder->hide[j].out[t-1];
					lstmencoder->hide[i].tan_in[t]+=lstmencoder->hide[i].tan_wh[j]*lstmencoder->hide[j].out[t-1];
					lstmencoder->hide[i].out_in[t]+=lstmencoder->hide[i].out_wh[j]*lstmencoder->hide[j].out[t-1];
				}
				lstmencoder->hide[i].fog_out[t]=sigmoid(lstmencoder->hide[i].fog_in[t]);
				lstmencoder->hide[i].sig_out[t]=sigmoid(lstmencoder->hide[i].sig_in[t]);
				lstmencoder->hide[i].tan_out[t]=tanh(lstmencoder->hide[i].tan_in[t]);
				lstmencoder->hide[i].out_out[t]=sigmoid(lstmencoder->hide[i].out_in[t]);
				lstmencoder->hide[i].cell[t]=lstmencoder->hide[i].cell[t-1]*lstmencoder->hide[i].fog_out[t]+lstmencoder->hide[i].sig_out[t]*lstmencoder->hide[i].tan_out[t];
				lstmencoder->hide[i].out[t]=tanh(lstmencoder->hide[i].cell[t])*lstmencoder->hide[i].out_out[t];
			}
		}
		for(int i=0;i<HNUM;i++)
		{
			lstmdecoder->hide[i].out[0]=lstmencoder->hide[i].out[ET];
			lstmdecoder->hide[i].cell[0]=lstmencoder->hide[i].cell[ET];
		}
		softmax_max=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in[0]=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in[0]+=output[i].w[j]*lstmdecoder->hide[j].out[0];
			softmax_max+=exp(output[i].in[0]);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out[0]=exp(output[i].in[0])/softmax_max;
		for(int t=1;t<=DT;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				lstmdecoder->hide[i].fog_in[t]=lstmdecoder->hide[i].fog_bia;
				lstmdecoder->hide[i].sig_in[t]=lstmdecoder->hide[i].sig_bia;
				lstmdecoder->hide[i].tan_in[t]=lstmdecoder->hide[i].tan_bia;
				lstmdecoder->hide[i].out_in[t]=lstmdecoder->hide[i].out_bia;
				for(int j=0;j<ONUM;j++)
				{
					lstmdecoder->hide[i].fog_in[t]+=lstmdecoder->hide[i].fog_wi[j]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hide[i].sig_in[t]+=lstmdecoder->hide[i].sig_wi[j]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hide[i].tan_in[t]+=lstmdecoder->hide[i].tan_wi[j]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hide[i].out_in[t]+=lstmdecoder->hide[i].out_wi[j]*expect[j][t-1];//output[j].out[t-1];
				}
				for(int j=0;j<HNUM;j++)
				{
					lstmdecoder->hide[i].fog_in[t]+=lstmdecoder->hide[i].fog_wh[j]*lstmdecoder->hide[j].out[t-1];
					lstmdecoder->hide[i].sig_in[t]+=lstmdecoder->hide[i].sig_wh[j]*lstmdecoder->hide[j].out[t-1];
					lstmdecoder->hide[i].tan_in[t]+=lstmdecoder->hide[i].tan_wh[j]*lstmdecoder->hide[j].out[t-1];
					lstmdecoder->hide[i].out_in[t]+=lstmdecoder->hide[i].out_wh[j]*lstmdecoder->hide[j].out[t-1];
				}
				lstmdecoder->hide[i].fog_out[t]=sigmoid(lstmdecoder->hide[i].fog_in[t]);
				lstmdecoder->hide[i].sig_out[t]=sigmoid(lstmdecoder->hide[i].sig_in[t]);
				lstmdecoder->hide[i].tan_out[t]=tanh(lstmdecoder->hide[i].tan_in[t]);
				lstmdecoder->hide[i].out_out[t]=sigmoid(lstmdecoder->hide[i].out_in[t]);
				lstmdecoder->hide[i].cell[t]=lstmdecoder->hide[i].cell[t-1]*lstmdecoder->hide[i].fog_out[t]+lstmdecoder->hide[i].sig_out[t]*lstmdecoder->hide[i].tan_out[t];
				lstmdecoder->hide[i].out[t]=tanh(lstmdecoder->hide[i].cell[t])*lstmdecoder->hide[i].out_out[t];
			}
			softmax_max=0;
			for(int i=0;i<ONUM;i++)
			{
				output[i].in[t]=output[i].bia;
				for(int j=0;j<HNUM;j++)
					output[i].in[t]+=output[i].w[j]*lstmdecoder->hide[j].out[t];
				softmax_max+=exp(output[i].in[t]);
			}
			for(int i=0;i<ONUM;i++)
				output[i].out[t]=exp(output[i].in[t])/softmax_max;
		}
		return;
	}
	else if(__Typename=="gru")
	{
		double softmax_max;
		for(int t=1;t<=ET;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				gruencoder->hide[i].sig_update_in[t]=gruencoder->hide[i].sig_update_bia;
				gruencoder->hide[i].sig_replace_in[t]=gruencoder->hide[i].sig_replace_bia;
				gruencoder->hide[i].tan_replace_in[t]=gruencoder->hide[i].tan_replace_bia;
				for(int j=0;j<INUM;j++)
				{
					gruencoder->hide[i].sig_update_in[t]+=gruencoder->hide[i].sig_update_wi[j]*input[j][t];
					gruencoder->hide[i].sig_replace_in[t]+=gruencoder->hide[i].sig_replace_wi[j]*input[j][t];
					gruencoder->hide[i].tan_replace_in[t]+=gruencoder->hide[i].tan_replace_wi[j]*input[j][t];
				}
				for(int j=0;j<HNUM;j++)
				{
					gruencoder->hide[i].sig_update_in[t]+=gruencoder->hide[i].sig_update_wh[j]*gruencoder->hide[j].out[t-1];
					gruencoder->hide[i].sig_replace_in[t]+=gruencoder->hide[i].sig_replace_wh[j]*gruencoder->hide[j].out[t-1];
				}
				gruencoder->hide[i].sig_update_out[t]=sigmoid(gruencoder->hide[i].sig_update_in[t]);
				gruencoder->hide[i].sig_replace_out[t]=sigmoid(gruencoder->hide[i].sig_replace_in[t]);
			}
			for(int i=0;i<HNUM;i++)
			{
				for(int j=0;j<HNUM;j++)
					gruencoder->hide[i].tan_replace_in[t]+=gruencoder->hide[i].tan_replace_wh[j]*gruencoder->hide[j].sig_update_out[t]*gruencoder->hide[j].out[t-1];
				gruencoder->hide[i].tan_replace_out[t]=tanh(gruencoder->hide[i].tan_replace_in[t]);
				gruencoder->hide[i].out[t]=gruencoder->hide[i].out[t-1]*gruencoder->hide[i].sig_replace_out[t]+(1-gruencoder->hide[i].sig_replace_out[t])*gruencoder->hide[i].tan_replace_out[t];
			}
		}
		for(int i=0;i<HNUM;i++)
			grudecoder->hide[i].out[0]=gruencoder->hide[i].out[ET];
		softmax_max=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in[0]=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in[0]+=output[i].w[j]*grudecoder->hide[j].out[0];
			softmax_max+=exp(output[i].in[0]);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out[0]=exp(output[i].in[0])/softmax_max;
		for(int t=1;t<=DT;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				grudecoder->hide[i].sig_update_in[t]=grudecoder->hide[i].sig_update_bia;
				grudecoder->hide[i].sig_replace_in[t]=grudecoder->hide[i].sig_replace_bia;
				grudecoder->hide[i].tan_replace_in[t]=grudecoder->hide[i].tan_replace_bia;
				for(int j=0;j<ONUM;j++)
				{
					grudecoder->hide[i].sig_update_in[t]+=grudecoder->hide[i].sig_update_wi[j]*expect[j][t-1];//output[j].out[t-1];
					grudecoder->hide[i].sig_replace_in[t]+=grudecoder->hide[i].sig_replace_wi[j]*expect[j][t-1];//output[j].out[t-1];
					grudecoder->hide[i].tan_replace_in[t]+=grudecoder->hide[i].tan_replace_wi[j]*expect[j][t-1];//output[j].out[t-1];
				}
				for(int j=0;j<HNUM;j++)
				{
					grudecoder->hide[i].sig_update_in[t]+=grudecoder->hide[i].sig_update_wh[j]*grudecoder->hide[j].out[t-1];
					grudecoder->hide[i].sig_replace_in[t]+=grudecoder->hide[i].sig_replace_wh[j]*grudecoder->hide[j].out[t-1];
				}
				grudecoder->hide[i].sig_update_out[t]=sigmoid(grudecoder->hide[i].sig_update_in[t]);
				grudecoder->hide[i].sig_replace_out[t]=sigmoid(grudecoder->hide[i].sig_replace_in[t]);
			}
			for(int i=0;i<HNUM;i++)
			{
				for(int j=0;j<HNUM;j++)
					grudecoder->hide[i].tan_replace_in[t]+=grudecoder->hide[i].tan_replace_wh[j]*grudecoder->hide[j].sig_update_out[t]*grudecoder->hide[j].out[t-1];//*(grudecoder->hide[j].out[t-1]+hiddenstate[j][t]);
				grudecoder->hide[i].tan_replace_out[t]=tanh(grudecoder->hide[i].tan_replace_in[t]);
				grudecoder->hide[i].out[t]=grudecoder->hide[i].out[t-1]*grudecoder->hide[i].sig_replace_out[t]+(1-grudecoder->hide[i].sig_replace_out[t])*grudecoder->hide[i].tan_replace_out[t];
			}
			softmax_max=0;
			for(int i=0;i<ONUM;i++)
			{
				output[i].in[t]=output[i].bia;
				for(int j=0;j<HNUM;j++)
					output[i].in[t]+=output[i].w[j]*grudecoder->hide[j].out[t];
				softmax_max+=exp(output[i].in[t]);
			}
			for(int i=0;i<ONUM;i++)
				output[i].out[t]=exp(output[i].in[t])/softmax_max;
		}
		return;
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		system("pause");
		exit(0);
	}
}

void NormalSeq2Seq::Training(const char* __Typename,const int ET,const int DT)
{
	if(__Typename=="rnn")
	{
		double trans;
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
				output[i].diff[t]=expect[i][t]-output[i].out[t];
		for(int i=0;i<ONUM;i++)
			output[i].diff[DT]*=output[i].out[DT]*(1-output[i].out[DT]);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[DT]*output[j].w[i];
			rnndecoder->hide[i].diff[DT]=trans*difftanh(rnndecoder->hide[i].in[DT]);
		}
		for(int t=DT-1;t>=1;t--)
		{
			for(int i=0;i<ONUM;i++)
			{
				for(int j=0;j<HNUM;j++)
					output[i].diff[t]+=rnndecoder->hide[j].diff[t+1]*rnndecoder->hide[j].wi[i];
				output[i].diff[t]*=output[i].out[t]*(1-output[i].out[t]);
			}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<ONUM;j++)
					trans+=output[j].diff[t]*output[j].w[i];
				for(int j=0;j<HNUM;j++)
					trans+=rnndecoder->hide[j].diff[t+1]*rnndecoder->hide[j].wh[i];
				rnndecoder->hide[i].diff[t]=trans*difftanh(rnndecoder->hide[i].in[t]);
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			for(int j=0;j<HNUM;j++)
				output[i].diff[0]+=rnndecoder->hide[j].diff[1]*rnndecoder->hide[j].wi[i];
			output[i].diff[0]*=output[i].out[0]*(1-output[i].out[0]);
		}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[0]*output[j].w[i];
			for(int j=0;j<HNUM;j++)
				trans+=rnndecoder->hide[j].diff[1]*rnndecoder->hide[j].wh[i];
			rnnencoder->hide[i].diff[ET]=trans*difftanh(rnnencoder->hide[i].in[ET]);
		}
		for(int t=ET-1;t>=1;t--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=rnnencoder->hide[j].diff[t+1]*rnnencoder->hide[j].wh[i];
				rnnencoder->hide[i].diff[t]=difftanh(trans*rnnencoder->hide[i].in[t]);
			}
		for(int i=0;i<HNUM;i++)
		{
			rnnencoder->hide[i].transbia=0;
			for(int j=0;j<INUM;j++)
				rnnencoder->hide[i].transwi[j]=0;
			for(int j=0;j<HNUM;j++)
				rnnencoder->hide[i].transwh[j]=0;
		}
		for(int t=1;t<=ET;t++)
			for(int i=0;i<HNUM;i++)
			{
				rnnencoder->hide[i].transbia+=2*rnnencoder->hide[i].diff[t];
				for(int j=0;j<HNUM;j++)
					rnnencoder->hide[i].transwh[j]+=rnnencoder->hide[i].diff[t]*rnnencoder->hide[j].out[t-1];
				for(int j=0;j<INUM;j++)
					rnnencoder->hide[i].transwi[j]+=rnnencoder->hide[i].diff[t]*input[j][t];
			}
		for(int i=0;i<HNUM;i++)
		{
			rnnencoder->hide[i].bia+=ClipGradient(learningrate*rnnencoder->hide[i].transbia);
			for(int j=0;j<INUM;j++)
				rnnencoder->hide[i].wi[j]+=ClipGradient(learningrate*rnnencoder->hide[i].transwi[j]);
			for(int j=0;j<HNUM;j++)
				rnnencoder->hide[i].wh[j]+=ClipGradient(learningrate*rnnencoder->hide[i].transwh[j]);
		}
		for(int i=0;i<HNUM;i++)
		{
			rnndecoder->hide[i].transbia=0;
			for(int j=0;j<INUM;j++)
				rnndecoder->hide[i].transwi[j]=0;
			for(int j=0;j<HNUM;j++)
				rnndecoder->hide[i].transwh[j]=0;
		}
		for(int i=0;i<ONUM;i++)
		{
			output[i].transbia=0;
			for(int j=0;j<HNUM;j++)
				output[i].transw[j]=0;
		}
		for(int t=1;t<=DT;t++)
			for(int i=0;i<HNUM;i++)
			{
				rnndecoder->hide[i].transbia+=2*rnndecoder->hide[i].diff[t];
				for(int j=0;j<HNUM;j++)
					rnndecoder->hide[i].transwh[j]+=rnndecoder->hide[i].diff[t]*rnndecoder->hide[j].out[t-1];
				for(int j=0;j<INUM;j++)
					rnndecoder->hide[i].transwi[j]+=rnndecoder->hide[i].diff[t]*expect[j][t-1];//output[j].out[t-1];
			}
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
			{
				output[i].transbia+=2*output[i].diff[t];
				for(int j=0;j<HNUM;j++)
					output[i].transw[j]+=output[i].diff[t]*rnndecoder->hide[j].out[t];
			}
		for(int i=0;i<HNUM;i++)
		{
			rnndecoder->hide[i].bia+=ClipGradient(learningrate*rnndecoder->hide[i].transbia);
			for(int j=0;j<INUM;j++)
				rnndecoder->hide[i].wi[j]+=ClipGradient(learningrate*rnndecoder->hide[i].transwi[j]);
			for(int j=0;j<HNUM;j++)
				rnndecoder->hide[i].wh[j]+=ClipGradient(learningrate*rnndecoder->hide[i].transwh[j]);
		}
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].transbia);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].transw[j]);
		}
		return;
	}
	else if(__Typename=="lstm")
	{
		double trans;
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
				output[i].diff[t]=expect[i][t]-output[i].out[t];
		for(int i=0;i<ONUM;i++)
			output[i].diff[DT]*=output[i].out[DT]*(1-output[i].out[DT]);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[DT]*output[j].w[i];
			lstmdecoder->hide[i].fog_diff[DT]=trans*lstmdecoder->hide[i].out_out[DT]*difftanh(lstmdecoder->hide[i].cell[DT])*lstmdecoder->hide[i].cell[DT-1]*diffsigmoid(lstmdecoder->hide[i].fog_in[DT]);
			lstmdecoder->hide[i].sig_diff[DT]=trans*lstmdecoder->hide[i].out_out[DT]*difftanh(lstmdecoder->hide[i].cell[DT])*lstmdecoder->hide[i].tan_out[DT]*diffsigmoid(lstmdecoder->hide[i].sig_in[DT]);
			lstmdecoder->hide[i].tan_diff[DT]=trans*lstmdecoder->hide[i].out_out[DT]*difftanh(lstmdecoder->hide[i].cell[DT])*lstmdecoder->hide[i].sig_out[DT]*difftanh(lstmdecoder->hide[i].tan_in[DT]);
			lstmdecoder->hide[i].out_diff[DT]=trans*tanh(lstmdecoder->hide[i].cell[DT])*diffsigmoid(lstmdecoder->hide[i].out_in[DT]);
		}
		for(int t=DT-1;t>=1;t--)
		{
			for(int i=0;i<ONUM;i++)
			{
				for(int j=0;j<HNUM;j++)
					output[i].diff[t]+=lstmdecoder->hide[j].fog_diff[t+1]*lstmdecoder->hide[j].fog_wi[i]+lstmdecoder->hide[j].sig_diff[t+1]*lstmdecoder->hide[j].sig_wi[i]+lstmdecoder->hide[j].tan_diff[t+1]*lstmdecoder->hide[j].tan_wi[i]+lstmdecoder->hide[j].out_diff[t+1]*lstmdecoder->hide[j].out_wi[i];
				output[i].diff[t]*=output[i].out[t]*(1-output[i].out[t]);
			}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<ONUM;j++)
					trans+=output[j].diff[t]*output[j].w[i];
				for(int j=0;j<HNUM;j++)
					trans+=lstmdecoder->hide[j].fog_diff[t+1]*lstmdecoder->hide[j].fog_wh[i]+lstmdecoder->hide[j].sig_diff[t+1]*lstmdecoder->hide[j].sig_wh[i]+lstmdecoder->hide[j].tan_diff[t+1]*lstmdecoder->hide[j].tan_wh[i]+lstmdecoder->hide[j].out_diff[t+1]*lstmdecoder->hide[j].out_wh[i];
				lstmdecoder->hide[i].fog_diff[t]=trans*lstmdecoder->hide[i].out_out[t]*difftanh(lstmdecoder->hide[i].cell[t])*lstmdecoder->hide[i].cell[t-1]*diffsigmoid(lstmdecoder->hide[i].fog_in[t]);
				lstmdecoder->hide[i].sig_diff[t]=trans*lstmdecoder->hide[i].out_out[t]*difftanh(lstmdecoder->hide[i].cell[t])*lstmdecoder->hide[i].tan_out[t]*diffsigmoid(lstmdecoder->hide[i].sig_in[t]);
				lstmdecoder->hide[i].tan_diff[t]=trans*lstmdecoder->hide[i].out_out[t]*difftanh(lstmdecoder->hide[i].cell[t])*lstmdecoder->hide[i].sig_out[t]*difftanh(lstmdecoder->hide[i].tan_in[t]);
				lstmdecoder->hide[i].out_diff[t]=trans*tanh(lstmdecoder->hide[i].cell[t])*diffsigmoid(lstmdecoder->hide[i].out_in[t]);
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			for(int j=0;j<HNUM;j++)
				output[i].diff[0]+=lstmdecoder->hide[j].fog_diff[1]*lstmdecoder->hide[j].fog_wi[i]+lstmdecoder->hide[j].sig_diff[1]*lstmdecoder->hide[j].sig_wi[i]+lstmdecoder->hide[j].tan_diff[1]*lstmdecoder->hide[j].tan_wi[i]+lstmdecoder->hide[j].out_diff[1]*lstmdecoder->hide[j].out_wi[i];
			output[i].diff[0]*=output[i].out[0]*(1-output[i].out[0]);
		}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[0]*output[j].w[i];
			for(int j=0;j<HNUM;j++)
				trans+=lstmdecoder->hide[j].fog_diff[1]*lstmdecoder->hide[j].fog_wh[i]+lstmdecoder->hide[j].sig_diff[1]*lstmdecoder->hide[j].sig_wh[i]+lstmdecoder->hide[j].tan_diff[1]*lstmdecoder->hide[j].tan_wh[i]+lstmdecoder->hide[j].out_diff[1]*lstmdecoder->hide[j].out_wh[i];
			lstmencoder->hide[i].fog_diff[ET]=trans*lstmencoder->hide[i].out_out[ET]*difftanh(lstmencoder->hide[i].cell[ET])*lstmencoder->hide[i].cell[ET-1]*diffsigmoid(lstmencoder->hide[i].fog_in[ET]);
			lstmencoder->hide[i].sig_diff[ET]=trans*lstmencoder->hide[i].out_out[ET]*difftanh(lstmencoder->hide[i].cell[ET])*lstmencoder->hide[i].tan_out[ET]*diffsigmoid(lstmencoder->hide[i].sig_in[ET]);
			lstmencoder->hide[i].tan_diff[ET]=trans*lstmencoder->hide[i].out_out[ET]*difftanh(lstmencoder->hide[i].cell[ET])*lstmencoder->hide[i].sig_out[ET]*difftanh(lstmencoder->hide[i].tan_in[ET]);
			lstmencoder->hide[i].out_diff[ET]=trans*tanh(lstmencoder->hide[i].cell[ET])*diffsigmoid(lstmencoder->hide[i].out_in[ET]);
		}
		for(int t=ET-1;t>=1;t--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=lstmencoder->hide[j].fog_diff[t+1]*lstmencoder->hide[j].fog_wh[i]+lstmencoder->hide[j].sig_diff[t+1]*lstmencoder->hide[j].sig_wh[i]+lstmencoder->hide[j].tan_diff[t+1]*lstmencoder->hide[j].tan_wh[i]+lstmencoder->hide[j].out_diff[t+1]*lstmencoder->hide[j].out_wh[i];
				lstmencoder->hide[i].fog_diff[t]=trans*lstmencoder->hide[i].out_out[t]*difftanh(lstmencoder->hide[i].cell[t])*lstmencoder->hide[i].cell[t-1]*diffsigmoid(lstmencoder->hide[i].fog_in[t]);
				lstmencoder->hide[i].sig_diff[t]=trans*lstmencoder->hide[i].out_out[t]*difftanh(lstmencoder->hide[i].cell[t])*lstmencoder->hide[i].tan_out[t]*diffsigmoid(lstmencoder->hide[i].sig_in[t]);
				lstmencoder->hide[i].tan_diff[t]=trans*lstmencoder->hide[i].out_out[t]*difftanh(lstmencoder->hide[i].cell[t])*lstmencoder->hide[i].sig_out[t]*difftanh(lstmencoder->hide[i].tan_in[t]);
				lstmencoder->hide[i].out_diff[t]=trans*tanh(lstmencoder->hide[i].cell[t])*diffsigmoid(lstmencoder->hide[i].out_in[t]);
			}
		for(int i=0;i<HNUM;i++)
		{
			lstmencoder->hide[i].fog_transbia=0;
			lstmencoder->hide[i].sig_transbia=0;
			lstmencoder->hide[i].tan_transbia=0;
			lstmencoder->hide[i].out_transbia=0;
			for(int j=0;j<INUM;j++)
			{
				lstmencoder->hide[i].fog_transwi[j]=0;
				lstmencoder->hide[i].sig_transwi[j]=0;
				lstmencoder->hide[i].tan_transwi[j]=0;
				lstmencoder->hide[i].out_transwi[j]=0;
			}
			for(int j=0;j<HNUM;j++)
			{
				lstmencoder->hide[i].fog_transwh[j]=0;
				lstmencoder->hide[i].sig_transwh[j]=0;
				lstmencoder->hide[i].tan_transwh[j]=0;
				lstmencoder->hide[i].out_transwh[j]=0;
			}
		}
		for(int t=1;t<=ET;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				lstmencoder->hide[i].fog_transbia+=2*lstmencoder->hide[i].fog_diff[t];
				lstmencoder->hide[i].sig_transbia+=2*lstmencoder->hide[i].sig_diff[t];
				lstmencoder->hide[i].tan_transbia+=2*lstmencoder->hide[i].tan_diff[t];
				lstmencoder->hide[i].out_transbia+=2*lstmencoder->hide[i].out_diff[t];
				for(int j=0;j<HNUM;j++)
				{
					lstmencoder->hide[i].fog_transwh[j]+=lstmencoder->hide[i].fog_diff[t]*lstmencoder->hide[j].out[t-1];
					lstmencoder->hide[i].sig_transwh[j]+=lstmencoder->hide[i].sig_diff[t]*lstmencoder->hide[j].out[t-1];
					lstmencoder->hide[i].tan_transwh[j]+=lstmencoder->hide[i].tan_diff[t]*lstmencoder->hide[j].out[t-1];
					lstmencoder->hide[i].out_transwh[j]+=lstmencoder->hide[i].out_diff[t]*lstmencoder->hide[j].out[t-1];
				}
				for(int j=0;j<INUM;j++)
				{
					lstmencoder->hide[i].fog_transwi[j]+=lstmencoder->hide[i].fog_diff[t]*input[j][t];
					lstmencoder->hide[i].sig_transwi[j]+=lstmencoder->hide[i].sig_diff[t]*input[j][t];
					lstmencoder->hide[i].tan_transwi[j]+=lstmencoder->hide[i].tan_diff[t]*input[j][t];
					lstmencoder->hide[i].out_transwi[j]+=lstmencoder->hide[i].out_diff[t]*input[j][t];
				}
			}
		}
		for(int i=0;i<HNUM;i++)
		{
			lstmencoder->hide[i].fog_bia+=ClipGradient(learningrate*lstmencoder->hide[i].fog_transbia);
			lstmencoder->hide[i].sig_bia+=ClipGradient(learningrate*lstmencoder->hide[i].sig_transbia);
			lstmencoder->hide[i].tan_bia+=ClipGradient(learningrate*lstmencoder->hide[i].tan_transbia);
			lstmencoder->hide[i].out_bia+=ClipGradient(learningrate*lstmencoder->hide[i].out_transbia);
			for(int j=0;j<INUM;j++)
			{
				lstmencoder->hide[i].fog_wi[j]+=ClipGradient(learningrate*lstmencoder->hide[i].fog_transwi[j]);
				lstmencoder->hide[i].sig_wi[j]+=ClipGradient(learningrate*lstmencoder->hide[i].sig_transwi[j]);
				lstmencoder->hide[i].tan_wi[j]+=ClipGradient(learningrate*lstmencoder->hide[i].tan_transwi[j]);
				lstmencoder->hide[i].out_wi[j]+=ClipGradient(learningrate*lstmencoder->hide[i].out_transwi[j]);
			}
			for(int j=0;j<HNUM;j++)
			{
				lstmencoder->hide[i].fog_wh[j]+=ClipGradient(learningrate*lstmencoder->hide[i].fog_transwh[j]);
				lstmencoder->hide[i].sig_wh[j]+=ClipGradient(learningrate*lstmencoder->hide[i].sig_transwh[j]);
				lstmencoder->hide[i].tan_wh[j]+=ClipGradient(learningrate*lstmencoder->hide[i].tan_transwh[j]);
				lstmencoder->hide[i].out_wh[j]+=ClipGradient(learningrate*lstmencoder->hide[i].out_transwh[j]);
			}
		}
		for(int i=0;i<HNUM;i++)
		{
			lstmdecoder->hide[i].fog_transbia=0;
			lstmdecoder->hide[i].sig_transbia=0;
			lstmdecoder->hide[i].tan_transbia=0;
			lstmdecoder->hide[i].out_transbia=0;
			for(int j=0;j<INUM;j++)
			{
				lstmdecoder->hide[i].fog_transwi[j]=0;
				lstmdecoder->hide[i].sig_transwi[j]=0;
				lstmdecoder->hide[i].tan_transwi[j]=0;
				lstmdecoder->hide[i].out_transwi[j]=0;
			}
			for(int j=0;j<HNUM;j++)
			{
				lstmdecoder->hide[i].fog_transwh[j]=0;
				lstmdecoder->hide[i].sig_transwh[j]=0;
				lstmdecoder->hide[i].tan_transwh[j]=0;
				lstmdecoder->hide[i].out_transwh[j]=0;
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			output[i].transbia=0;
			for(int j=0;j<HNUM;j++)
				output[i].transw[j]=0;
		}
		for(int t=1;t<=DT;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				lstmdecoder->hide[i].fog_transbia+=2*lstmdecoder->hide[i].fog_diff[t];
				lstmdecoder->hide[i].sig_transbia+=2*lstmdecoder->hide[i].sig_diff[t];
				lstmdecoder->hide[i].tan_transbia+=2*lstmdecoder->hide[i].tan_diff[t];
				lstmdecoder->hide[i].out_transbia+=2*lstmdecoder->hide[i].out_diff[t];
				for(int j=0;j<HNUM;j++)
				{
					lstmdecoder->hide[i].fog_transwh[j]+=lstmdecoder->hide[i].fog_diff[t]*lstmdecoder->hide[j].out[t-1];
					lstmdecoder->hide[i].sig_transwh[j]+=lstmdecoder->hide[i].sig_diff[t]*lstmdecoder->hide[j].out[t-1];
					lstmdecoder->hide[i].tan_transwh[j]+=lstmdecoder->hide[i].tan_diff[t]*lstmdecoder->hide[j].out[t-1];
					lstmdecoder->hide[i].out_transwh[j]+=lstmdecoder->hide[i].out_diff[t]*lstmdecoder->hide[j].out[t-1];
				}
				for(int j=0;j<INUM;j++)
				{
					lstmdecoder->hide[i].fog_transwi[j]+=lstmdecoder->hide[i].fog_diff[t]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hide[i].sig_transwi[j]+=lstmdecoder->hide[i].sig_diff[t]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hide[i].tan_transwi[j]+=lstmdecoder->hide[i].tan_diff[t]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hide[i].out_transwi[j]+=lstmdecoder->hide[i].out_diff[t]*expect[j][t-1];//output[j].out[t-1];
				}
			}
		}
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
			{
				output[i].transbia+=2*output[i].diff[t];
				for(int j=0;j<HNUM;j++)
					output[i].transw[j]+=output[i].diff[t]*lstmdecoder->hide[j].out[t];
			}
		for(int i=0;i<HNUM;i++)
		{
			lstmdecoder->hide[i].fog_bia+=ClipGradient(learningrate*lstmdecoder->hide[i].fog_transbia);
			lstmdecoder->hide[i].sig_bia+=ClipGradient(learningrate*lstmdecoder->hide[i].sig_transbia);
			lstmdecoder->hide[i].tan_bia+=ClipGradient(learningrate*lstmdecoder->hide[i].tan_transbia);
			lstmdecoder->hide[i].out_bia+=ClipGradient(learningrate*lstmdecoder->hide[i].out_transbia);
			for(int j=0;j<INUM;j++)
			{
				lstmdecoder->hide[i].fog_wi[j]+=ClipGradient(learningrate*lstmdecoder->hide[i].fog_transwi[j]);
				lstmdecoder->hide[i].sig_wi[j]+=ClipGradient(learningrate*lstmdecoder->hide[i].sig_transwi[j]);
				lstmdecoder->hide[i].tan_wi[j]+=ClipGradient(learningrate*lstmdecoder->hide[i].tan_transwi[j]);
				lstmdecoder->hide[i].out_wi[j]+=ClipGradient(learningrate*lstmdecoder->hide[i].out_transwi[j]);
			}
			for(int j=0;j<HNUM;j++)
			{
				lstmdecoder->hide[i].fog_wh[j]+=ClipGradient(learningrate*lstmdecoder->hide[i].fog_transwh[j]);
				lstmdecoder->hide[i].sig_wh[j]+=ClipGradient(learningrate*lstmdecoder->hide[i].sig_transwh[j]);
				lstmdecoder->hide[i].tan_wh[j]+=ClipGradient(learningrate*lstmdecoder->hide[i].tan_transwh[j]);
				lstmdecoder->hide[i].out_wh[j]+=ClipGradient(learningrate*lstmdecoder->hide[i].out_transwh[j]);
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].transbia);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].transw[j]);
		}
		return;
	}
	else if(__Typename=="gru")
	{
		double trans;
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
				output[i].diff[t]=expect[i][t]-output[i].out[t];
		for(int i=0;i<ONUM;i++)
			output[i].diff[DT]*=output[i].out[DT]*(1-output[i].out[DT]);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[DT]*output[j].w[i];
			grudecoder->hide[i].sig_update_diff[DT]=trans*(1-grudecoder->hide[i].sig_replace_out[DT])*difftanh(grudecoder->hide[i].tan_replace_in[DT])*grudecoder->hide[i].tan_replace_wh[i]*grudecoder->hide[i].out[DT-1]*diffsigmoid(grudecoder->hide[i].sig_update_in[DT]);
			grudecoder->hide[i].sig_replace_diff[DT]=trans*(grudecoder->hide[i].out[DT-1]-grudecoder->hide[i].tan_replace_out[DT])*diffsigmoid(grudecoder->hide[i].sig_replace_in[DT]);
			grudecoder->hide[i].tan_replace_diff[DT]=trans*(1-grudecoder->hide[i].sig_replace_out[DT])*difftanh(grudecoder->hide[i].tan_replace_in[DT]);
		}
		for(int t=DT-1;t>=1;t--)
		{
			for(int i=0;i<ONUM;i++)
			{
				for(int j=0;j<HNUM;j++)
					output[i].diff[t]+=grudecoder->hide[j].sig_update_diff[t+1]*grudecoder->hide[j].sig_update_wi[i]+grudecoder->hide[j].sig_replace_diff[t+1]*grudecoder->hide[j].sig_replace_wi[i]+grudecoder->hide[j].tan_replace_diff[t+1]*grudecoder->hide[j].tan_replace_wi[i];
				output[i].diff[t]*=output[i].out[t]*(1-output[i].out[t]);
			}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<ONUM;j++)
					trans+=output[j].diff[t]*output[j].w[i];
				for(int j=0;j<HNUM;j++)
					trans+=grudecoder->hide[j].sig_update_diff[t+1]*grudecoder->hide[j].sig_update_wh[i]+grudecoder->hide[j].sig_replace_diff[t+1]*grudecoder->hide[j].sig_replace_wh[i]+grudecoder->hide[j].tan_replace_diff[t+1]*grudecoder->hide[j].tan_replace_wh[i]*grudecoder->hide[i].sig_replace_out[t+1];
	
				grudecoder->hide[i].sig_update_diff[t]=trans*(1-grudecoder->hide[i].sig_replace_out[t])*difftanh(grudecoder->hide[i].tan_replace_in[t])*grudecoder->hide[i].tan_replace_wh[i]*grudecoder->hide[i].out[t-1]*diffsigmoid(grudecoder->hide[i].sig_update_in[t]);
				grudecoder->hide[i].sig_replace_diff[t]=trans*(grudecoder->hide[i].out[t-1]-grudecoder->hide[i].tan_replace_out[t])*diffsigmoid(grudecoder->hide[i].sig_replace_in[t]);
				grudecoder->hide[i].tan_replace_diff[t]=trans*(1-grudecoder->hide[i].sig_replace_out[t])*difftanh(grudecoder->hide[i].tan_replace_in[t]);
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			for(int j=0;j<HNUM;j++)
				output[i].diff[0]+=grudecoder->hide[j].sig_update_diff[1]*grudecoder->hide[j].sig_update_wi[i]+grudecoder->hide[j].sig_replace_diff[1]*grudecoder->hide[j].sig_replace_wi[i]+grudecoder->hide[j].tan_replace_diff[1]*grudecoder->hide[j].tan_replace_wi[i];
			output[i].diff[0]*=output[i].out[0]*(1-output[i].out[0]);
		}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[0]*output[j].w[i];
			for(int j=0;j<HNUM;j++)
				trans+=grudecoder->hide[j].sig_update_diff[1]*grudecoder->hide[j].sig_update_wh[i]+grudecoder->hide[j].sig_replace_diff[1]*grudecoder->hide[j].sig_replace_wh[i]+grudecoder->hide[j].tan_replace_diff[1]*grudecoder->hide[j].tan_replace_wh[i]*grudecoder->hide[i].sig_replace_out[1];
	
			gruencoder->hide[i].sig_update_diff[ET]=trans*(1-gruencoder->hide[i].sig_replace_out[ET])*difftanh(gruencoder->hide[i].tan_replace_in[ET])*gruencoder->hide[i].tan_replace_wh[i]*gruencoder->hide[i].out[ET-1]*diffsigmoid(gruencoder->hide[i].sig_update_in[ET]);
			gruencoder->hide[i].sig_replace_diff[ET]=trans*(gruencoder->hide[i].out[ET-1]-gruencoder->hide[i].tan_replace_out[ET])*diffsigmoid(gruencoder->hide[i].sig_replace_in[ET]);
			gruencoder->hide[i].tan_replace_diff[ET]=trans*(1-gruencoder->hide[i].sig_replace_out[ET])*difftanh(gruencoder->hide[i].tan_replace_in[ET]);
		}
		for(int t=ET-1;t>=1;t--)
		{
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=gruencoder->hide[j].sig_update_diff[t+1]*gruencoder->hide[j].sig_update_wh[i]+gruencoder->hide[j].sig_replace_diff[t+1]*gruencoder->hide[j].sig_replace_wh[i]+gruencoder->hide[j].tan_replace_diff[t+1]*gruencoder->hide[j].tan_replace_wh[i]*gruencoder->hide[i].sig_replace_out[t+1];
	
				gruencoder->hide[i].sig_update_diff[t]=trans*(1-gruencoder->hide[i].sig_replace_out[t])*difftanh(gruencoder->hide[i].tan_replace_in[t])*gruencoder->hide[i].tan_replace_wh[i]*gruencoder->hide[i].out[t-1]*diffsigmoid(gruencoder->hide[i].sig_update_in[t]);
				gruencoder->hide[i].sig_replace_diff[t]=trans*(gruencoder->hide[i].out[t-1]-gruencoder->hide[i].tan_replace_out[t])*diffsigmoid(gruencoder->hide[i].sig_replace_in[t]);
				gruencoder->hide[i].tan_replace_diff[t]=trans*(1-gruencoder->hide[i].sig_replace_out[t])*difftanh(gruencoder->hide[i].tan_replace_in[t]);
			}
		}
		for(int i=0;i<HNUM;i++)
		{
			gruencoder->hide[i].sig_update_transbia=0;
			gruencoder->hide[i].sig_replace_transbia=0;
			gruencoder->hide[i].tan_replace_transbia=0;
			for(int j=0;j<INUM;j++)
			{
				gruencoder->hide[i].sig_update_transwi[j]=0;
				gruencoder->hide[i].sig_replace_transwi[j]=0;
				gruencoder->hide[i].tan_replace_transwi[j]=0;
			}
			for(int j=0;j<HNUM;j++)
			{
				gruencoder->hide[i].sig_update_transwh[j]=0;
				gruencoder->hide[i].sig_replace_transwh[j]=0;
				gruencoder->hide[i].tan_replace_transwh[j]=0;
			}
		}
		for(int t=1;t<=ET;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				gruencoder->hide[i].sig_update_transbia+=2*gruencoder->hide[i].sig_update_diff[t];
				gruencoder->hide[i].sig_replace_transbia+=2*gruencoder->hide[i].sig_replace_diff[t];
				gruencoder->hide[i].tan_replace_transbia+=2*gruencoder->hide[i].tan_replace_diff[t];
				for(int j=0;j<HNUM;j++)
				{
					gruencoder->hide[i].sig_update_transwh[j]+=gruencoder->hide[i].sig_update_diff[t]*gruencoder->hide[j].out[t-1];
					gruencoder->hide[i].sig_replace_transwh[j]+=gruencoder->hide[i].sig_replace_diff[t]*gruencoder->hide[j].sig_replace_out[t]*gruencoder->hide[j].out[t-1];
					gruencoder->hide[i].tan_replace_transwh[j]+=gruencoder->hide[i].tan_replace_diff[t]*gruencoder->hide[j].out[t-1];
				}
				for(int j=0;j<INUM;j++)
				{
					gruencoder->hide[i].sig_update_transwi[j]+=gruencoder->hide[i].sig_update_diff[t]*input[j][t];
					gruencoder->hide[i].sig_replace_transwi[j]+=gruencoder->hide[i].sig_replace_diff[t]*input[j][t];
					gruencoder->hide[i].tan_replace_transwi[j]+=gruencoder->hide[i].tan_replace_diff[t]*input[j][t];
				}
			}
		}
		for(int i=0;i<HNUM;i++)
		{
			gruencoder->hide[i].sig_update_bia+=ClipGradient(learningrate*gruencoder->hide[i].sig_update_transbia);
			gruencoder->hide[i].sig_replace_bia+=ClipGradient(learningrate*gruencoder->hide[i].sig_replace_transbia);
			gruencoder->hide[i].tan_replace_bia+=ClipGradient(learningrate*gruencoder->hide[i].tan_replace_transbia);
			for(int j=0;j<INUM;j++)
			{
				gruencoder->hide[i].sig_update_wi[j]+=ClipGradient(learningrate*gruencoder->hide[i].sig_update_transwi[j]);
				gruencoder->hide[i].sig_replace_wi[j]+=ClipGradient(learningrate*gruencoder->hide[i].sig_replace_transwi[j]);
				gruencoder->hide[i].tan_replace_wi[j]+=ClipGradient(learningrate*gruencoder->hide[i].tan_replace_transwi[j]);
			}
			for(int j=0;j<HNUM;j++)
			{
				gruencoder->hide[i].sig_update_wh[j]+=ClipGradient(learningrate*gruencoder->hide[i].sig_update_transwh[j]);
				gruencoder->hide[i].sig_replace_wh[j]+=ClipGradient(learningrate*gruencoder->hide[i].sig_replace_transwh[j]);
				gruencoder->hide[i].tan_replace_wh[j]+=ClipGradient(learningrate*gruencoder->hide[i].tan_replace_transwh[j]);
			}
		}
		for(int i=0;i<HNUM;i++)
		{
			grudecoder->hide[i].sig_update_transbia=0;
			grudecoder->hide[i].sig_replace_transbia=0;
			grudecoder->hide[i].tan_replace_transbia=0;
			for(int j=0;j<INUM;j++)
			{
				grudecoder->hide[i].sig_update_transwi[j]=0;
				grudecoder->hide[i].sig_replace_transwi[j]=0;
				grudecoder->hide[i].tan_replace_transwi[j]=0;
			}
			for(int j=0;j<HNUM;j++)
			{
				grudecoder->hide[i].sig_update_transwh[j]=0;
				grudecoder->hide[i].sig_replace_transwh[j]=0;
				grudecoder->hide[i].tan_replace_transwh[j]=0;
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			output[i].transbia=0;
			for(int j=0;j<HNUM;j++)
				output[i].transw[j]=0;
		}
		for(int t=1;t<=DT;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				grudecoder->hide[i].sig_update_transbia+=2*grudecoder->hide[i].sig_update_diff[t];
				grudecoder->hide[i].sig_replace_transbia+=2*grudecoder->hide[i].sig_replace_diff[t];
				grudecoder->hide[i].tan_replace_transbia+=2*grudecoder->hide[i].tan_replace_diff[t];
				for(int j=0;j<HNUM;j++)
				{
					grudecoder->hide[i].sig_update_transwh[j]+=grudecoder->hide[i].sig_update_diff[t]*grudecoder->hide[j].out[t-1];//(grudecoder->hide[j].out[t-1]+hiddenstate[j][t]);
					grudecoder->hide[i].sig_replace_transwh[j]+=grudecoder->hide[i].sig_replace_diff[t]*grudecoder->hide[j].sig_replace_out[t]*grudecoder->hide[j].out[t-1];//(grudecoder->hide[j].out[t-1]+hiddenstate[j][t]);
					grudecoder->hide[i].tan_replace_transwh[j]+=grudecoder->hide[i].tan_replace_diff[t]*grudecoder->hide[j].out[t-1];//(grudecoder->hide[j].out[t-1]+hiddenstate[j][t]);
				}
				for(int j=0;j<INUM;j++)
				{
					grudecoder->hide[i].sig_update_transwi[j]+=grudecoder->hide[i].sig_update_diff[t]*expect[j][t-1];//output[j].out[t-1];
					grudecoder->hide[i].sig_replace_transwi[j]+=grudecoder->hide[i].sig_replace_diff[t]*expect[j][t-1];//output[j].out[t-1];
					grudecoder->hide[i].tan_replace_transwi[j]+=grudecoder->hide[i].tan_replace_diff[t]*expect[j][t-1];//output[j].out[t-1];
				}
			}
		}
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
			{
				output[i].transbia+=2*output[i].diff[t];
				for(int j=0;j<HNUM;j++)
					output[i].transw[j]+=output[i].diff[t]*grudecoder->hide[j].out[t];
			}
		for(int i=0;i<HNUM;i++)
		{
			grudecoder->hide[i].sig_update_bia+=ClipGradient(learningrate*grudecoder->hide[i].sig_update_transbia);
			grudecoder->hide[i].sig_replace_bia+=ClipGradient(learningrate*grudecoder->hide[i].sig_replace_transbia);
			grudecoder->hide[i].tan_replace_bia+=ClipGradient(learningrate*grudecoder->hide[i].tan_replace_transbia);
			for(int j=0;j<INUM;j++)
			{
				grudecoder->hide[i].sig_update_wi[j]+=ClipGradient(learningrate*grudecoder->hide[i].sig_update_transwi[j]);
				grudecoder->hide[i].sig_replace_wi[j]+=ClipGradient(learningrate*grudecoder->hide[i].sig_replace_transwi[j]);
				grudecoder->hide[i].tan_replace_wi[j]+=ClipGradient(learningrate*grudecoder->hide[i].tan_replace_transwi[j]);
			}
			for(int j=0;j<HNUM;j++)
			{
				grudecoder->hide[i].sig_update_wh[j]+=ClipGradient(learningrate*grudecoder->hide[i].sig_update_transwh[j]);
				grudecoder->hide[i].sig_replace_wh[j]+=ClipGradient(learningrate*grudecoder->hide[i].sig_replace_transwh[j]);
				grudecoder->hide[i].tan_replace_wh[j]+=ClipGradient(learningrate*grudecoder->hide[i].tan_replace_transwh[j]);
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].transbia);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].transw[j]);
		}
		return;
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		system("pause");
		exit(0);
	}
}

void NormalSeq2Seq::ErrorCalc(const int DT)
{
	error=0;
	double trans;
	for(int t=1;t<=DT;t++)
		for(int i=0;i<ONUM;i++)
		{
			trans=expect[i][t]-output[i].out[t];
			error+=trans*trans;
		}
	error*=0.5;
	return;
}

void NormalSeq2Seq::SetFunction(const char* function_name)
{
	func_name=function_name;
}

void NormalSeq2Seq::Datain(const char *__Typename,const char *EncoderFile,const char *DecoderFile,const char *OutputFile)
{
	if(__Typename=="rnn")
	{
		rnnencoder->Datain(EncoderFile);
		rnndecoder->Datain(DecoderFile);
	}
	else if(__Typename=="lstm")
	{
		lstmencoder->Datain(EncoderFile);
		lstmdecoder->Datain(DecoderFile);
	}
	else if(__Typename=="gru")
	{
		gruencoder->Datain(EncoderFile);
		grudecoder->Datain(DecoderFile);
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		system("pause");
		exit(0);
	}
	ifstream fin(OutputFile);
	for(int i=0;i<ONUM;i++)
	{
		fin>>output[i].bia;
		for(int j=0;j<HNUM;j++)
			fin>>output[i].w[j];
	}
	fin.close();
}

void NormalSeq2Seq::Dataout(const char *__Typename,const char *EncoderFile,const char *DecoderFile,const char *OutputFile)
{
	if(__Typename=="rnn")
	{
		rnnencoder->Dataout(EncoderFile);
		rnndecoder->Dataout(DecoderFile);
	}
	else if(__Typename=="lstm")
	{
		lstmencoder->Dataout(EncoderFile);
		lstmdecoder->Dataout(DecoderFile);
	}
	else if(__Typename=="gru")
	{
		gruencoder->Dataout(EncoderFile);
		grudecoder->Dataout(DecoderFile);
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		system("pause");
		exit(0);
	}
	ofstream fout(OutputFile);
	for(int i=0;i<ONUM;i++)
	{
		fout<<output[i].bia<<endl;
		for(int j=0;j<HNUM;j++)
			fout<<output[i].w[j]<<endl;
	}
	fout.close();
}

DeepSeq2Seq::DeepSeq2Seq(const char* __Typename,int InputlayerNum,int HiddenlayerNum,int OutputlayerNum,int Depth,int Maxtime)
{
	srand(unsigned(time(NULL)));
	INUM=InputlayerNum;
	HNUM=HiddenlayerNum;
	ONUM=OutputlayerNum;
	MAXTIME=Maxtime;
	DEPTH=Depth-1;
	rnnencoder=NULL;
	rnndecoder=NULL;
	lstmencoder=NULL;
	lstmdecoder=NULL;
	gruencoder=NULL;
	grudecoder=NULL;
	if(__Typename=="rnn")
	{
		rnnencoder=new DeepRNN(INUM,HNUM,DEPTH,MAXTIME);
		rnndecoder=new DeepRNN(ONUM,HNUM,DEPTH,MAXTIME);
		rnnencoder->INIT();
		rnndecoder->INIT();
	}
	else if(__Typename=="lstm")
	{
		lstmencoder=new DeepLSTM(INUM,HNUM,DEPTH,MAXTIME);
		lstmdecoder=new DeepLSTM(ONUM,HNUM,DEPTH,MAXTIME);
		lstmencoder->INIT();
		lstmdecoder->INIT();
	}
	else if(__Typename=="gru")
	{
		gruencoder=new DeepGRU(INUM,HNUM,DEPTH,MAXTIME);
		grudecoder=new DeepGRU(ONUM,HNUM,DEPTH,MAXTIME);
		gruencoder->INIT();
		grudecoder->INIT();
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		system("pause");
		exit(0);
	}
	input=new double*[INUM];
	for(int i=0;i<INUM;i++)
		input[i]=new double[MAXTIME];
	expect=new double*[ONUM];
	for(int i=0;i<ONUM;i++)
		expect[i]=new double[MAXTIME];
	output=new seq_neuron[ONUM];
	for(int i=0;i<ONUM;i++)
	{
		output[i].in=new double[MAXTIME];
		output[i].out=new double[MAXTIME];
		output[i].diff=new double[MAXTIME];
		output[i].w=new double[HNUM];
		output[i].transw=new double[HNUM];
	}
	for(int i=0;i<ONUM;i++)
	{
		output[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
		for(int j=0;j<HNUM;j++)
			output[i].w[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
	}
}

DeepSeq2Seq::~DeepSeq2Seq()
{
	delete rnnencoder;
	delete rnndecoder;
	delete lstmencoder;
	delete lstmdecoder;
	delete gruencoder;
	delete grudecoder;
	for(int i=0;i<INUM;i++)
		delete []input[i];
	delete []input;
	for(int i=0;i<ONUM;i++)
		delete []expect[i];
	delete []expect;
	for(int i=0;i<ONUM;i++)
	{
		delete []output[i].in;
		delete []output[i].out;
		delete []output[i].diff;
		delete []output[i].w;
		delete []output[i].transw;
	}
	delete []output;
}

void DeepSeq2Seq::SetLearningRate(const double __lr)
{
	learningrate=__lr;
}

void DeepSeq2Seq::Calc(const char* __Typename,const int ET,const int DT)
{
	if(__Typename=="rnn")
	{
		double softmax_max;
		for(int t=1;t<=ET;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				rnnencoder->hlink[i].in[t]=rnnencoder->hlink[i].bia;
				for(int j=0;j<INUM;j++)
					rnnencoder->hlink[i].in[t]+=rnnencoder->hlink[i].wi[j]*input[j][t];
				for(int j=0;j<HNUM;j++)
					rnnencoder->hlink[i].in[t]+=rnnencoder->hlink[i].wh[j]*rnnencoder->hlink[j].out[t-1];
				rnnencoder->hlink[i].out[t]=sigmoid(rnnencoder->hlink[i].in[t]);
				rnnencoder->hlink[i].out[t]=tanh(rnnencoder->hlink[i].in[t]);
			}
			for(int d=0;d<DEPTH;d++)
				for(int i=0;i<HNUM;i++)
				{
					rnnencoder->hide[i][d].in[t]=rnnencoder->hide[i][d].bia;
					for(int j=0;j<HNUM;j++)
					{
						rnnencoder->hide[i][d].in[t]+=rnnencoder->hide[i][d].wi[j]*(d>0? rnnencoder->hide[j][d-1].out[t]:rnnencoder->hlink[j].out[t]);
						rnnencoder->hide[i][d].in[t]+=rnnencoder->hide[i][d].wh[j]*rnnencoder->hide[j][d].out[t-1];
					}
					rnnencoder->hide[i][d].out[t]=tanh(rnnencoder->hide[i][d].in[t]);
				}
		}
		
		for(int i=0;i<HNUM;i++)
		{
			rnndecoder->hlink[i].out[0]=rnnencoder->hlink[i].out[ET];
			for(int d=0;d<DEPTH;d++)
				rnndecoder->hide[i][d].out[0]=rnnencoder->hide[i][d].out[ET];
		}
		softmax_max=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in[0]=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in[0]+=output[i].w[j]*rnndecoder->hide[j][DEPTH-1].out[0];
			softmax_max+=exp(output[i].in[0]);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out[0]=exp(output[i].in[0])/softmax_max;
		
		for(int t=1;t<=DT;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				rnndecoder->hlink[i].in[t]=rnndecoder->hlink[i].bia;
				for(int j=0;j<ONUM;j++)
					rnndecoder->hlink[i].in[t]+=rnndecoder->hlink[i].wi[j]*expect[j][t-1];//output[j].out[t-1];
				for(int j=0;j<HNUM;j++)
					rnndecoder->hlink[i].in[t]+=rnndecoder->hlink[i].wh[j]*rnndecoder->hlink[j].out[t-1];
				rnndecoder->hlink[i].out[t]=tanh(rnndecoder->hlink[i].in[t]);
			}
			for(int d=0;d<DEPTH;d++)
				for(int i=0;i<HNUM;i++)
				{
					rnndecoder->hide[i][d].in[t]=rnndecoder->hide[i][d].bia;
					for(int j=0;j<HNUM;j++)
					{
						rnndecoder->hide[i][d].in[t]+=rnndecoder->hide[i][d].wi[j]*(d>0? rnndecoder->hide[j][d-1].out[t]:rnndecoder->hlink[j].out[t]);
						rnndecoder->hide[i][d].in[t]+=rnndecoder->hide[i][d].wh[j]*rnndecoder->hide[j][d].out[t-1];
					}
					rnndecoder->hide[i][d].out[t]=tanh(rnndecoder->hide[i][d].in[t]);
				}
			softmax_max=0;
			for(int i=0;i<ONUM;i++)
			{
				output[i].in[t]=output[i].bia;
				for(int j=0;j<HNUM;j++)
					output[i].in[t]+=output[i].w[j]*rnndecoder->hide[j][DEPTH-1].out[t];
				softmax_max+=exp(output[i].in[t]);
			}
			for(int i=0;i<ONUM;i++)
				output[i].out[t]=exp(output[i].in[t])/softmax_max;
		}
		return;
	}
	else if(__Typename=="lstm")
	{
		double softmax_max;
		for(int t=1;t<=ET;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				lstmencoder->hlink[i].fog_in[t]=lstmencoder->hlink[i].fog_bia;
				lstmencoder->hlink[i].sig_in[t]=lstmencoder->hlink[i].sig_bia;
				lstmencoder->hlink[i].tan_in[t]=lstmencoder->hlink[i].tan_bia;
				lstmencoder->hlink[i].out_in[t]=lstmencoder->hlink[i].out_bia;
				for(int j=0;j<INUM;j++)
				{
					lstmencoder->hlink[i].fog_in[t]+=lstmencoder->hlink[i].fog_wi[j]*input[j][t];
					lstmencoder->hlink[i].sig_in[t]+=lstmencoder->hlink[i].sig_wi[j]*input[j][t];
					lstmencoder->hlink[i].tan_in[t]+=lstmencoder->hlink[i].tan_wi[j]*input[j][t];
					lstmencoder->hlink[i].out_in[t]+=lstmencoder->hlink[i].out_wi[j]*input[j][t];
				}
				for(int j=0;j<HNUM;j++)
				{
					lstmencoder->hlink[i].fog_in[t]+=lstmencoder->hlink[i].fog_wh[j]*lstmencoder->hlink[j].out[t-1];
					lstmencoder->hlink[i].sig_in[t]+=lstmencoder->hlink[i].sig_wh[j]*lstmencoder->hlink[j].out[t-1];
					lstmencoder->hlink[i].tan_in[t]+=lstmencoder->hlink[i].tan_wh[j]*lstmencoder->hlink[j].out[t-1];
					lstmencoder->hlink[i].out_in[t]+=lstmencoder->hlink[i].out_wh[j]*lstmencoder->hlink[j].out[t-1];
				}
				lstmencoder->hlink[i].fog_out[t]=sigmoid(lstmencoder->hlink[i].fog_in[t]);
				lstmencoder->hlink[i].sig_out[t]=sigmoid(lstmencoder->hlink[i].sig_in[t]);
				lstmencoder->hlink[i].tan_out[t]=tanh(lstmencoder->hlink[i].tan_in[t]);
				lstmencoder->hlink[i].out_out[t]=sigmoid(lstmencoder->hlink[i].out_in[t]);
				lstmencoder->hlink[i].cell[t]=lstmencoder->hlink[i].cell[t-1]*lstmencoder->hlink[i].fog_out[t]+lstmencoder->hlink[i].sig_out[t]*lstmencoder->hlink[i].tan_out[t];
				lstmencoder->hlink[i].out[t]=tanh(lstmencoder->hlink[i].cell[t])*lstmencoder->hlink[i].out_out[t];
			}
			for(int d=0;d<DEPTH;d++)
				for(int i=0;i<HNUM;i++)
				{
					lstmencoder->hide[i][d].fog_in[t]=lstmencoder->hide[i][d].fog_bia;
					lstmencoder->hide[i][d].sig_in[t]=lstmencoder->hide[i][d].sig_bia;
					lstmencoder->hide[i][d].tan_in[t]=lstmencoder->hide[i][d].tan_bia;
					lstmencoder->hide[i][d].out_in[t]=lstmencoder->hide[i][d].out_bia;
					for(int j=0;j<HNUM;j++)
					{
						lstmencoder->hide[i][d].fog_in[t]+=lstmencoder->hide[i][d].fog_wi[j]*(d>0? lstmencoder->hide[j][d-1].out[t]:lstmencoder->hlink[j].out[t]);
						lstmencoder->hide[i][d].sig_in[t]+=lstmencoder->hide[i][d].sig_wi[j]*(d>0? lstmencoder->hide[j][d-1].out[t]:lstmencoder->hlink[j].out[t]);
						lstmencoder->hide[i][d].tan_in[t]+=lstmencoder->hide[i][d].tan_wi[j]*(d>0? lstmencoder->hide[j][d-1].out[t]:lstmencoder->hlink[j].out[t]);
						lstmencoder->hide[i][d].out_in[t]+=lstmencoder->hide[i][d].out_wi[j]*(d>0? lstmencoder->hide[j][d-1].out[t]:lstmencoder->hlink[j].out[t]);
	
						lstmencoder->hide[i][d].fog_in[t]+=lstmencoder->hide[i][d].fog_wh[j]*lstmencoder->hide[j][d].out[t-1];
						lstmencoder->hide[i][d].sig_in[t]+=lstmencoder->hide[i][d].sig_wh[j]*lstmencoder->hide[j][d].out[t-1];
						lstmencoder->hide[i][d].tan_in[t]+=lstmencoder->hide[i][d].tan_wh[j]*lstmencoder->hide[j][d].out[t-1];
						lstmencoder->hide[i][d].out_in[t]+=lstmencoder->hide[i][d].out_wh[j]*lstmencoder->hide[j][d].out[t-1];
					}
					lstmencoder->hide[i][d].fog_out[t]=sigmoid(lstmencoder->hide[i][d].fog_in[t]);
					lstmencoder->hide[i][d].sig_out[t]=sigmoid(lstmencoder->hide[i][d].sig_in[t]);
					lstmencoder->hide[i][d].tan_out[t]=tanh(lstmencoder->hide[i][d].tan_in[t]);
					lstmencoder->hide[i][d].out_out[t]=sigmoid(lstmencoder->hide[i][d].out_in[t]);
					lstmencoder->hide[i][d].cell[t]=lstmencoder->hide[i][d].cell[t-1]*lstmencoder->hide[i][d].fog_out[t]+lstmencoder->hide[i][d].sig_out[t]*lstmencoder->hide[i][d].tan_out[t];
					lstmencoder->hide[i][d].out[t]=tanh(lstmencoder->hide[i][d].cell[t])*lstmencoder->hide[i][d].out_out[t];
				}
		}
		
		for(int i=0;i<HNUM;i++)
		{
			lstmdecoder->hlink[i].out[0]=lstmencoder->hlink[i].out[ET];
			lstmdecoder->hlink[i].cell[0]=lstmencoder->hlink[i].cell[ET];
			for(int d=0;d<DEPTH;d++)
			{
				lstmdecoder->hide[i][d].out[0]=lstmencoder->hide[i][d].out[ET];
				lstmdecoder->hide[i][d].cell[0]=lstmencoder->hide[i][d].cell[ET];
			}
		}
		softmax_max=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in[0]=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in[0]+=output[i].w[j]*lstmdecoder->hide[j][DEPTH-1].out[0];
			softmax_max+=exp(output[i].in[0]);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out[0]=exp(output[i].in[0])/softmax_max;
		
		for(int t=1;t<=DT;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				lstmdecoder->hlink[i].fog_in[t]=lstmdecoder->hlink[i].fog_bia;
				lstmdecoder->hlink[i].sig_in[t]=lstmdecoder->hlink[i].sig_bia;
				lstmdecoder->hlink[i].tan_in[t]=lstmdecoder->hlink[i].tan_bia;
				lstmdecoder->hlink[i].out_in[t]=lstmdecoder->hlink[i].out_bia;
				for(int j=0;j<ONUM;j++)
				{
					lstmdecoder->hlink[i].fog_in[t]+=lstmdecoder->hlink[i].fog_wi[j]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hlink[i].sig_in[t]+=lstmdecoder->hlink[i].sig_wi[j]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hlink[i].tan_in[t]+=lstmdecoder->hlink[i].tan_wi[j]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hlink[i].out_in[t]+=lstmdecoder->hlink[i].out_wi[j]*expect[j][t-1];//output[j].out[t-1];
				}
				for(int j=0;j<HNUM;j++)
				{
					lstmdecoder->hlink[i].fog_in[t]+=lstmdecoder->hlink[i].fog_wh[j]*lstmdecoder->hlink[j].out[t-1];
					lstmdecoder->hlink[i].sig_in[t]+=lstmdecoder->hlink[i].sig_wh[j]*lstmdecoder->hlink[j].out[t-1];
					lstmdecoder->hlink[i].tan_in[t]+=lstmdecoder->hlink[i].tan_wh[j]*lstmdecoder->hlink[j].out[t-1];
					lstmdecoder->hlink[i].out_in[t]+=lstmdecoder->hlink[i].out_wh[j]*lstmdecoder->hlink[j].out[t-1];
				}
				lstmdecoder->hlink[i].fog_out[t]=sigmoid(lstmdecoder->hlink[i].fog_in[t]);
				lstmdecoder->hlink[i].sig_out[t]=sigmoid(lstmdecoder->hlink[i].sig_in[t]);
				lstmdecoder->hlink[i].tan_out[t]=tanh(lstmdecoder->hlink[i].tan_in[t]);
				lstmdecoder->hlink[i].out_out[t]=sigmoid(lstmdecoder->hlink[i].out_in[t]);
				lstmdecoder->hlink[i].cell[t]=lstmdecoder->hlink[i].cell[t-1]*lstmdecoder->hlink[i].fog_out[t]+lstmdecoder->hlink[i].sig_out[t]*lstmdecoder->hlink[i].tan_out[t];
				lstmdecoder->hlink[i].out[t]=tanh(lstmdecoder->hlink[i].cell[t])*lstmdecoder->hlink[i].out_out[t];
			}
			for(int d=0;d<DEPTH;d++)
				for(int i=0;i<HNUM;i++)
				{
					lstmdecoder->hide[i][d].fog_in[t]=lstmdecoder->hide[i][d].fog_bia;
					lstmdecoder->hide[i][d].sig_in[t]=lstmdecoder->hide[i][d].sig_bia;
					lstmdecoder->hide[i][d].tan_in[t]=lstmdecoder->hide[i][d].tan_bia;
					lstmdecoder->hide[i][d].out_in[t]=lstmdecoder->hide[i][d].out_bia;
					for(int j=0;j<HNUM;j++)
					{
						lstmdecoder->hide[i][d].fog_in[t]+=lstmdecoder->hide[i][d].fog_wi[j]*(d>0? lstmdecoder->hide[j][d-1].out[t]:lstmdecoder->hlink[j].out[t]);
						lstmdecoder->hide[i][d].sig_in[t]+=lstmdecoder->hide[i][d].sig_wi[j]*(d>0? lstmdecoder->hide[j][d-1].out[t]:lstmdecoder->hlink[j].out[t]);
						lstmdecoder->hide[i][d].tan_in[t]+=lstmdecoder->hide[i][d].tan_wi[j]*(d>0? lstmdecoder->hide[j][d-1].out[t]:lstmdecoder->hlink[j].out[t]);
						lstmdecoder->hide[i][d].out_in[t]+=lstmdecoder->hide[i][d].out_wi[j]*(d>0? lstmdecoder->hide[j][d-1].out[t]:lstmdecoder->hlink[j].out[t]);
	
						lstmdecoder->hide[i][d].fog_in[t]+=lstmdecoder->hide[i][d].fog_wh[j]*lstmdecoder->hide[j][d].out[t-1];
						lstmdecoder->hide[i][d].sig_in[t]+=lstmdecoder->hide[i][d].sig_wh[j]*lstmdecoder->hide[j][d].out[t-1];
						lstmdecoder->hide[i][d].tan_in[t]+=lstmdecoder->hide[i][d].tan_wh[j]*lstmdecoder->hide[j][d].out[t-1];
						lstmdecoder->hide[i][d].out_in[t]+=lstmdecoder->hide[i][d].out_wh[j]*lstmdecoder->hide[j][d].out[t-1];
					}
					lstmdecoder->hide[i][d].fog_out[t]=sigmoid(lstmdecoder->hide[i][d].fog_in[t]);
					lstmdecoder->hide[i][d].sig_out[t]=sigmoid(lstmdecoder->hide[i][d].sig_in[t]);
					lstmdecoder->hide[i][d].tan_out[t]=tanh(lstmdecoder->hide[i][d].tan_in[t]);
					lstmdecoder->hide[i][d].out_out[t]=sigmoid(lstmdecoder->hide[i][d].out_in[t]);
					lstmdecoder->hide[i][d].cell[t]=lstmdecoder->hide[i][d].cell[t-1]*lstmdecoder->hide[i][d].fog_out[t]+lstmdecoder->hide[i][d].sig_out[t]*lstmdecoder->hide[i][d].tan_out[t];
					lstmdecoder->hide[i][d].out[t]=tanh(lstmdecoder->hide[i][d].cell[t])*lstmdecoder->hide[i][d].out_out[t];
				}
			softmax_max=0;
			for(int i=0;i<ONUM;i++)
			{
				output[i].in[t]=output[i].bia;
				for(int j=0;j<HNUM;j++)
					output[i].in[t]+=output[i].w[j]*lstmdecoder->hide[j][DEPTH-1].out[t];
				softmax_max+=exp(output[i].in[t]);
			}
			for(int i=0;i<ONUM;i++)
				output[i].out[t]=exp(output[i].in[t])/softmax_max;
		}
		return;
	}
	else if(__Typename=="gru")
	{
		double softmax_max;
		for(int t=1;t<=ET;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				gruencoder->hlink[i].sig_update_in[t]=gruencoder->hlink[i].sig_update_bia;
				gruencoder->hlink[i].sig_replace_in[t]=gruencoder->hlink[i].sig_replace_bia;
				gruencoder->hlink[i].tan_replace_in[t]=gruencoder->hlink[i].tan_replace_bia;
				for(int j=0;j<INUM;j++)
				{
					gruencoder->hlink[i].sig_update_in[t]+=gruencoder->hlink[i].sig_update_wi[j]*input[j][t];
					gruencoder->hlink[i].sig_replace_in[t]+=gruencoder->hlink[i].sig_replace_wi[j]*input[j][t];
					gruencoder->hlink[i].tan_replace_in[t]+=gruencoder->hlink[i].tan_replace_wi[j]*input[j][t];
				}
				for(int j=0;j<HNUM;j++)
				{
					gruencoder->hlink[i].sig_update_in[t]+=gruencoder->hlink[i].sig_update_wh[j]*gruencoder->hlink[j].out[t-1];
					gruencoder->hlink[i].sig_replace_in[t]+=gruencoder->hlink[i].sig_replace_wh[j]*gruencoder->hlink[j].out[t-1];
				}
				gruencoder->hlink[i].sig_update_out[t]=sigmoid(gruencoder->hlink[i].sig_update_in[t]);
				gruencoder->hlink[i].sig_replace_out[t]=sigmoid(gruencoder->hlink[i].sig_replace_in[t]);
			}
			for(int i=0;i<HNUM;i++)
			{
				for(int j=0;j<HNUM;j++)
					gruencoder->hlink[i].tan_replace_in[t]+=gruencoder->hlink[i].tan_replace_wh[j]*gruencoder->hlink[j].sig_update_out[t]*gruencoder->hlink[j].out[t-1];
				gruencoder->hlink[i].tan_replace_out[t]=tanh(gruencoder->hlink[i].tan_replace_in[t]);
				gruencoder->hlink[i].out[t]=gruencoder->hlink[i].out[t-1]*gruencoder->hlink[i].sig_replace_out[t]+(1-gruencoder->hlink[i].sig_replace_out[t])*gruencoder->hlink[i].tan_replace_out[t];
			}
			for(int d=0;d<DEPTH;d++)
			{
				for(int i=0;i<HNUM;i++)
				{
					gruencoder->hide[i][d].sig_update_in[t]=gruencoder->hide[i][d].sig_update_bia;
					gruencoder->hide[i][d].sig_replace_in[t]=gruencoder->hide[i][d].sig_replace_bia;
					gruencoder->hide[i][d].tan_replace_in[t]=gruencoder->hide[i][d].tan_replace_bia;
					for(int j=0;j<INUM;j++)
					{
						gruencoder->hide[i][d].sig_update_in[t]+=gruencoder->hide[i][d].sig_update_wi[j]*(d==0? gruencoder->hlink[j].out[t-1]:gruencoder->hide[j][d-1].out[t-1]);
						gruencoder->hide[i][d].sig_replace_in[t]+=gruencoder->hide[i][d].sig_replace_wi[j]*(d==0? gruencoder->hlink[j].out[t-1]:gruencoder->hide[j][d-1].out[t-1]);
						gruencoder->hide[i][d].tan_replace_in[t]+=gruencoder->hide[i][d].tan_replace_wi[j]*(d==0? gruencoder->hlink[j].out[t-1]:gruencoder->hide[j][d-1].out[t-1]);
					}
					for(int j=0;j<HNUM;j++)
					{
						gruencoder->hide[i][d].sig_update_in[t]+=gruencoder->hide[i][d].sig_update_wh[j]*gruencoder->hide[j][d].out[t-1];
						gruencoder->hide[i][d].sig_replace_in[t]+=gruencoder->hide[i][d].sig_replace_wh[j]*gruencoder->hide[j][d].out[t-1];
					}
					gruencoder->hide[i][d].sig_update_out[t]=sigmoid(gruencoder->hide[i][d].sig_update_in[t]);
					gruencoder->hide[i][d].sig_replace_out[t]=sigmoid(gruencoder->hide[i][d].sig_replace_in[t]);
				}
				for(int i=0;i<HNUM;i++)
				{
					for(int j=0;j<HNUM;j++)
						gruencoder->hide[i][d].tan_replace_in[t]+=gruencoder->hide[i][d].tan_replace_wh[j]*gruencoder->hide[j][d].sig_update_out[t]*gruencoder->hide[j][d].out[t-1];
					gruencoder->hide[i][d].tan_replace_out[t]=tanh(gruencoder->hide[i][d].tan_replace_in[t]);
					gruencoder->hide[i][d].out[t]=gruencoder->hide[i][d].out[t-1]*gruencoder->hide[i][d].sig_replace_out[t]+(1-gruencoder->hide[i][d].sig_replace_out[t])*gruencoder->hide[i][d].tan_replace_out[t];
				}
			}
		}
		for(int i=0;i<HNUM;i++)
		{
			grudecoder->hlink[i].out[0]=gruencoder->hlink[i].out[ET];
			for(int d=0;d<DEPTH;d++)
				grudecoder->hide[i][d].out[0]=gruencoder->hide[i][d].out[ET];
		}
			
		softmax_max=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in[0]=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in[0]+=output[i].w[j]*grudecoder->hide[j][DEPTH-1].out[0];
			softmax_max+=exp(output[i].in[0]);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out[0]=exp(output[i].in[0])/softmax_max;
		for(int t=1;t<=DT;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				grudecoder->hlink[i].sig_update_in[t]=grudecoder->hlink[i].sig_update_bia;
				grudecoder->hlink[i].sig_replace_in[t]=grudecoder->hlink[i].sig_replace_bia;
				grudecoder->hlink[i].tan_replace_in[t]=grudecoder->hlink[i].tan_replace_bia;
				for(int j=0;j<INUM;j++)
				{
					grudecoder->hlink[i].sig_update_in[t]+=grudecoder->hlink[i].sig_update_wi[j]*input[j][t];
					grudecoder->hlink[i].sig_replace_in[t]+=grudecoder->hlink[i].sig_replace_wi[j]*input[j][t];
					grudecoder->hlink[i].tan_replace_in[t]+=grudecoder->hlink[i].tan_replace_wi[j]*input[j][t];
				}
				for(int j=0;j<HNUM;j++)
				{
					grudecoder->hlink[i].sig_update_in[t]+=grudecoder->hlink[i].sig_update_wh[j]*grudecoder->hlink[j].out[t-1];
					grudecoder->hlink[i].sig_replace_in[t]+=grudecoder->hlink[i].sig_replace_wh[j]*grudecoder->hlink[j].out[t-1];
				}
				grudecoder->hlink[i].sig_update_out[t]=sigmoid(grudecoder->hlink[i].sig_update_in[t]);
				grudecoder->hlink[i].sig_replace_out[t]=sigmoid(grudecoder->hlink[i].sig_replace_in[t]);
			}
			for(int i=0;i<HNUM;i++)
			{
				for(int j=0;j<HNUM;j++)
					grudecoder->hlink[i].tan_replace_in[t]+=grudecoder->hlink[i].tan_replace_wh[j]*grudecoder->hlink[j].sig_update_out[t]*grudecoder->hlink[j].out[t-1];
				grudecoder->hlink[i].tan_replace_out[t]=tanh(grudecoder->hlink[i].tan_replace_in[t]);
				grudecoder->hlink[i].out[t]=grudecoder->hlink[i].out[t-1]*grudecoder->hlink[i].sig_replace_out[t]+(1-grudecoder->hlink[i].sig_replace_out[t])*grudecoder->hlink[i].tan_replace_out[t];
			}
			for(int d=0;d<DEPTH;d++)
			{
				for(int i=0;i<HNUM;i++)
				{
					grudecoder->hide[i][d].sig_update_in[t]=grudecoder->hide[i][d].sig_update_bia;
					grudecoder->hide[i][d].sig_replace_in[t]=grudecoder->hide[i][d].sig_replace_bia;
					grudecoder->hide[i][d].tan_replace_in[t]=grudecoder->hide[i][d].tan_replace_bia;
					for(int j=0;j<INUM;j++)
					{
						grudecoder->hide[i][d].sig_update_in[t]+=grudecoder->hide[i][d].sig_update_wi[j]*(d==0? grudecoder->hlink[j].out[t-1]:grudecoder->hide[j][d-1].out[t-1]);
						grudecoder->hide[i][d].sig_replace_in[t]+=grudecoder->hide[i][d].sig_replace_wi[j]*(d==0? grudecoder->hlink[j].out[t-1]:grudecoder->hide[j][d-1].out[t-1]);
						grudecoder->hide[i][d].tan_replace_in[t]+=grudecoder->hide[i][d].tan_replace_wi[j]*(d==0? grudecoder->hlink[j].out[t-1]:grudecoder->hide[j][d-1].out[t-1]);
					}
					for(int j=0;j<HNUM;j++)
					{
						grudecoder->hide[i][d].sig_update_in[t]+=grudecoder->hide[i][d].sig_update_wh[j]*grudecoder->hide[j][d].out[t-1];
						grudecoder->hide[i][d].sig_replace_in[t]+=grudecoder->hide[i][d].sig_replace_wh[j]*grudecoder->hide[j][d].out[t-1];
					}
					grudecoder->hide[i][d].sig_update_out[t]=sigmoid(grudecoder->hide[i][d].sig_update_in[t]);
					grudecoder->hide[i][d].sig_replace_out[t]=sigmoid(grudecoder->hide[i][d].sig_replace_in[t]);
				}
				for(int i=0;i<HNUM;i++)
				{
					for(int j=0;j<HNUM;j++)
						grudecoder->hide[i][d].tan_replace_in[t]+=grudecoder->hide[i][d].tan_replace_wh[j]*grudecoder->hide[j][d].sig_update_out[t]*grudecoder->hide[j][d].out[t-1];
					grudecoder->hide[i][d].tan_replace_out[t]=tanh(grudecoder->hide[i][d].tan_replace_in[t]);
					grudecoder->hide[i][d].out[t]=grudecoder->hide[i][d].out[t-1]*grudecoder->hide[i][d].sig_replace_out[t]+(1-grudecoder->hide[i][d].sig_replace_out[t])*grudecoder->hide[i][d].tan_replace_out[t];
				}
			}
			softmax_max=0;
			for(int i=0;i<ONUM;i++)
			{
				output[i].in[t]=output[i].bia;
				for(int j=0;j<HNUM;j++)
					output[i].in[t]+=output[i].w[j]*grudecoder->hide[j][DEPTH-1].out[t];
				softmax_max+=exp(output[i].in[t]);
			}
			for(int i=0;i<ONUM;i++)
				output[i].out[t]=exp(output[i].in[t])/softmax_max;
		}
		return;
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		system("pause");
		exit(0);
	}
}

void DeepSeq2Seq::Training(const char* __Typename,const int ET,const int DT)
{
	if(__Typename=="rnn")
	{
		double trans;
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
				output[i].diff[t]=expect[i][t]-output[i].out[t];
		for(int i=0;i<ONUM;i++)
			output[i].diff[DT]*=output[i].out[DT]*(1-output[i].out[DT]);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[DT]*output[j].w[i];
			rnndecoder->hide[i][DEPTH-1].diff[DT]=trans*difftanh(rnndecoder->hide[i][DEPTH-1].in[DT]);
		}
		for(int d=DEPTH-2;d>=0;d--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=rnndecoder->hide[j][d+1].diff[DT]*rnndecoder->hide[j][d+1].wi[i];
				rnndecoder->hide[i][d].diff[DT]=trans*difftanh(rnndecoder->hide[i][d].in[DT]);
			}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<HNUM;j++)
				trans+=rnndecoder->hide[j][0].diff[DT]*rnndecoder->hide[j][0].wi[i];
			rnndecoder->hlink[i].diff[DT]=trans*difftanh(rnndecoder->hlink[i].in[DT]);
		}
		for(int t=DT-1;t>=1;t--)
		{
			for(int i=0;i<ONUM;i++)
			{
				for(int j=0;j<HNUM;j++)
					output[i].diff[t]+=rnndecoder->hlink[j].diff[t+1]*rnndecoder->hlink[j].wi[i];
				output[i].diff[t]*=output[i].out[t]*(1-output[i].out[t]);
			}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<ONUM;j++)
					trans+=output[j].diff[t]*output[j].w[i];
				for(int j=0;j<HNUM;j++)
					trans+=rnndecoder->hide[j][DEPTH-1].diff[t+1]*rnndecoder->hide[j][DEPTH-1].wh[i];
				rnndecoder->hide[i][DEPTH-1].diff[t]=trans*difftanh(rnndecoder->hide[i][DEPTH-1].in[t]);
			}
			for(int d=DEPTH-2;d>=0;d--)
				for(int i=0;i<HNUM;i++)
				{
					trans=0;
					for(int j=0;j<HNUM;j++)
					{
						trans+=rnndecoder->hide[j][d+1].diff[t]*rnndecoder->hide[j][d+1].wi[i];
						trans+=rnndecoder->hide[j][d].diff[t+1]*rnndecoder->hide[j][d].wh[i];
					}
					rnndecoder->hide[i][d].diff[t]=trans*difftanh(rnndecoder->hide[i][d].in[t]);
				}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
				{
					trans+=rnndecoder->hide[j][0].diff[t]*rnndecoder->hide[j][0].wi[i];
					trans+=rnndecoder->hlink[j].diff[t+1]*rnndecoder->hlink[j].wh[i];
				}
				rnndecoder->hlink[i].diff[t]=trans*difftanh(rnndecoder->hlink[i].in[t]);
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			for(int j=0;j<HNUM;j++)
				output[i].diff[0]+=rnndecoder->hlink[j].diff[1]*rnndecoder->hlink[j].wi[i];
			output[i].diff[0]*=output[i].out[0]*(1-output[i].out[0]);
		}
		
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[0]*output[j].w[i];
			for(int j=0;j<HNUM;j++)
				trans+=rnndecoder->hide[j][DEPTH-1].diff[1]*rnndecoder->hide[j][DEPTH-1].wh[i];
			rnnencoder->hide[i][DEPTH-1].diff[ET]=trans*difftanh(rnnencoder->hide[i][DEPTH-1].in[ET]);
		}
		for(int d=DEPTH-2;d>=0;d--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
				{
					trans+=rnnencoder->hide[j][d+1].diff[ET]*rnnencoder->hide[j][d+1].wi[i];
					trans+=rnndecoder->hide[j][d].diff[1]*rnndecoder->hide[j][d].wh[i];
				}
				rnnencoder->hide[i][d].diff[ET]=trans*difftanh(rnnencoder->hide[i][d].in[ET]);
			}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<HNUM;j++)
			{
				trans+=rnnencoder->hide[j][0].diff[ET]*rnnencoder->hide[j][0].wi[i];
				trans+=rnndecoder->hlink[j].diff[1]*rnndecoder->hlink[j].wh[i];
			}
			rnnencoder->hlink[i].diff[ET]=trans*difftanh(rnnencoder->hlink[i].in[ET]);
		}
		for(int t=ET-1;t>=1;t--)
		{
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=rnnencoder->hide[j][DEPTH-1].diff[t+1]*rnnencoder->hide[j][DEPTH-1].wh[i];
				rnnencoder->hide[i][DEPTH-1].diff[t]=trans*difftanh(rnnencoder->hide[i][DEPTH-1].in[t]);
			}
			for(int d=DEPTH-2;d>=0;d--)
				for(int i=0;i<HNUM;i++)
				{
					trans=0;
					for(int j=0;j<HNUM;j++)
					{
						trans+=rnnencoder->hide[j][d+1].diff[t]*rnnencoder->hide[j][d+1].wi[i];
						trans+=rnnencoder->hide[j][d].diff[t+1]*rnnencoder->hide[j][d].wh[i];
					}
					rnnencoder->hide[i][d].diff[t]=trans*difftanh(rnnencoder->hide[i][d].in[t]);
				}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
				{
					trans+=rnnencoder->hide[j][0].diff[t]*rnnencoder->hide[j][0].wi[i];
					trans+=rnnencoder->hlink[j].diff[t+1]*rnnencoder->hlink[j].wh[i];
				}
				rnnencoder->hlink[i].diff[t]=trans*difftanh(rnnencoder->hlink[i].in[t]);
			}
		}
		for(int i=0;i<HNUM;i++)
		{
			rnnencoder->hlink[i].transbia=0;
			for(int j=0;j<INUM;j++)
				rnnencoder->hlink[i].transwi[j]=0;
			for(int j=0;j<HNUM;j++)
				rnnencoder->hlink[i].transwh[j]=0;
		}
		for(int d=0;d<DEPTH;d++)
			for(int i=0;i<HNUM;i++)
			{
				rnnencoder->hide[i][d].transbia=0;
				for(int j=0;j<HNUM;j++)
				{
					rnnencoder->hide[i][d].transwi[j]=0;
					rnnencoder->hide[i][d].transwh[j]=0;
				}
			}
		for(int t=1;t<=ET;t++)
		{
			for(int d=DEPTH-1;d>=0;d--)
				for(int i=0;i<HNUM;i++)
				{
					rnnencoder->hide[i][d].transbia+=2*rnnencoder->hide[i][d].diff[t];
					for(int j=0;j<HNUM;j++)
					{
						rnnencoder->hide[i][d].transwh[j]+=rnnencoder->hide[i][d].diff[t]*rnnencoder->hide[j][d].out[t-1];
						rnnencoder->hide[i][d].transwi[j]+=rnnencoder->hide[i][d].diff[t]*(d>0? rnnencoder->hide[j][d-1].out[t]:rnnencoder->hlink[j].out[t]);
					}
				}
			for(int i=0;i<HNUM;i++)
			{
				rnnencoder->hlink[i].transbia+=2*rnnencoder->hlink[i].diff[t];
				for(int j=0;j<HNUM;j++)
					rnnencoder->hlink[i].transwh[j]+=rnnencoder->hlink[i].diff[t]*rnnencoder->hlink[j].out[t-1];
				for(int j=0;j<INUM;j++)
					rnnencoder->hlink[i].transwi[j]+=rnnencoder->hlink[i].diff[t]*input[j][t];
			}
		}
			
		for(int i=0;i<HNUM;i++)
		{
			rnnencoder->hlink[i].bia+=ClipGradient(learningrate*rnnencoder->hlink[i].transbia);
			for(int j=0;j<INUM;j++)
				rnnencoder->hlink[i].wi[j]+=ClipGradient(learningrate*rnnencoder->hlink[i].transwi[j]);
			for(int j=0;j<HNUM;j++)
				rnnencoder->hlink[i].wh[j]+=ClipGradient(learningrate*rnnencoder->hlink[i].transwh[j]);
		}
		for(int d=0;d<DEPTH;d++)
				for(int i=0;i<HNUM;i++)
				{
					rnnencoder->hide[i][d].bia+=ClipGradient(learningrate*rnnencoder->hide[i][d].transbia);
					for(int j=0;j<HNUM;j++)
					{
						rnnencoder->hide[i][d].wi[j]+=ClipGradient(learningrate*rnnencoder->hide[i][d].transwi[j]);
						rnnencoder->hide[i][d].wh[j]+=ClipGradient(learningrate*rnnencoder->hide[i][d].transwh[j]);
					}
				}
		for(int i=0;i<HNUM;i++)
		{
			rnndecoder->hlink[i].transbia=0;
			for(int j=0;j<INUM;j++)
				rnndecoder->hlink[i].transwi[j]=0;
			for(int j=0;j<HNUM;j++)
				rnndecoder->hlink[i].transwh[j]=0;
		}
		for(int d=0;d<DEPTH;d++)
			for(int i=0;i<HNUM;i++)
			{
				rnndecoder->hide[i][d].transbia=0;
				for(int j=0;j<HNUM;j++)
				{
					rnndecoder->hide[i][d].transwi[j]=0;
					rnndecoder->hide[i][d].transwh[j]=0;
				}
			}
		for(int i=0;i<ONUM;i++)
		{
			output[i].transbia=0;
			for(int j=0;j<HNUM;j++)
				output[i].transw[j]=0;
		}
		for(int t=1;t<=DT;t++)
		{
			for(int d=DEPTH-1;d>=0;d--)
				for(int i=0;i<HNUM;i++)
				{
					rnndecoder->hide[i][d].transbia+=2*rnndecoder->hide[i][d].diff[t];
					for(int j=0;j<HNUM;j++)
					{
						rnndecoder->hide[i][d].transwh[j]+=rnndecoder->hide[i][d].diff[t]*rnndecoder->hide[j][d].out[t-1];
						rnndecoder->hide[i][d].transwi[j]+=rnndecoder->hide[i][d].diff[t]*(d>0? rnndecoder->hide[j][d-1].out[t]:rnndecoder->hlink[j].out[t]);
					}
				}
			
			for(int i=0;i<HNUM;i++)
			{
				rnndecoder->hlink[i].transbia+=2*rnndecoder->hlink[i].diff[t];
				for(int j=0;j<HNUM;j++)
					rnndecoder->hlink[i].transwh[j]+=rnndecoder->hlink[i].diff[t]*rnndecoder->hlink[j].out[t-1];
				for(int j=0;j<INUM;j++)
					rnndecoder->hlink[i].transwi[j]+=rnndecoder->hlink[i].diff[t]*expect[j][t-1];//output[j].out[t-1];
			}
		}
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
			{
				output[i].transbia+=2*output[i].diff[t];
				for(int j=0;j<HNUM;j++)
					output[i].transw[j]+=output[i].diff[t]*rnndecoder->hide[j][DEPTH-1].out[t];
			}
	
		for(int i=0;i<HNUM;i++)
		{
			rnndecoder->hlink[i].bia+=ClipGradient(learningrate*rnndecoder->hlink[i].transbia);
			for(int j=0;j<INUM;j++)
				rnndecoder->hlink[i].wi[j]+=ClipGradient(learningrate*rnndecoder->hlink[i].transwi[j]);
			for(int j=0;j<HNUM;j++)
				rnndecoder->hlink[i].wh[j]+=ClipGradient(learningrate*rnndecoder->hlink[i].transwh[j]);
		}
		for(int d=0;d<DEPTH;d++)
				for(int i=0;i<HNUM;i++)
				{
					rnndecoder->hide[i][d].bia+=ClipGradient(learningrate*rnndecoder->hide[i][d].transbia);
					for(int j=0;j<HNUM;j++)
					{
						rnndecoder->hide[i][d].wi[j]+=ClipGradient(learningrate*rnndecoder->hide[i][d].transwi[j]);
						rnndecoder->hide[i][d].wh[j]+=ClipGradient(learningrate*rnndecoder->hide[i][d].transwh[j]);
					}
				}
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].transbia);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].transw[j]);
		}
		return;
	}
	else if(__Typename=="lstm")
	{
		double trans;
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
				output[i].diff[t]=expect[i][t]-output[i].out[t];
		for(int i=0;i<ONUM;i++)
			output[i].diff[DT]*=output[i].out[DT]*(1-output[i].out[DT]);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[DT]*output[j].w[i];
			lstmdecoder->hide[i][DEPTH-1].fog_diff[DT]=trans*lstmdecoder->hide[i][DEPTH-1].out_out[DT]*difftanh(lstmdecoder->hide[i][DEPTH-1].cell[DT])*lstmdecoder->hide[i][DEPTH-1].cell[DT-1]*diffsigmoid(lstmdecoder->hide[i][DEPTH-1].fog_in[DT]);
			lstmdecoder->hide[i][DEPTH-1].sig_diff[DT]=trans*lstmdecoder->hide[i][DEPTH-1].out_out[DT]*difftanh(lstmdecoder->hide[i][DEPTH-1].cell[DT])*lstmdecoder->hide[i][DEPTH-1].tan_out[DT]*diffsigmoid(lstmdecoder->hide[i][DEPTH-1].sig_in[DT]);
			lstmdecoder->hide[i][DEPTH-1].tan_diff[DT]=trans*lstmdecoder->hide[i][DEPTH-1].out_out[DT]*difftanh(lstmdecoder->hide[i][DEPTH-1].cell[DT])*lstmdecoder->hide[i][DEPTH-1].sig_out[DT]*difftanh(lstmdecoder->hide[i][DEPTH-1].tan_in[DT]);
			lstmdecoder->hide[i][DEPTH-1].out_diff[DT]=trans*tanh(lstmdecoder->hide[i][DEPTH-1].cell[DT])*diffsigmoid(lstmdecoder->hide[i][DEPTH-1].out_in[DT]);
		}
		for(int d=DEPTH-2;d>=0;d--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=lstmdecoder->hide[j][d+1].fog_diff[DT]*lstmdecoder->hide[j][d+1].fog_wi[i]+lstmdecoder->hide[j][d+1].sig_diff[DT]*lstmdecoder->hide[j][d+1].sig_wi[i]+lstmdecoder->hide[j][d+1].tan_diff[DT]*lstmdecoder->hide[j][d+1].tan_wi[i]+lstmdecoder->hide[j][d+1].out_diff[DT]*lstmdecoder->hide[j][d+1].out_wi[i];
				lstmdecoder->hide[i][d].fog_diff[DT]=trans*lstmdecoder->hide[i][d].out_out[DT]*difftanh(lstmdecoder->hide[i][d].cell[DT])*lstmdecoder->hide[i][d].cell[DT-1]*diffsigmoid(lstmdecoder->hide[i][d].fog_in[DT]);
				lstmdecoder->hide[i][d].sig_diff[DT]=trans*lstmdecoder->hide[i][d].out_out[DT]*difftanh(lstmdecoder->hide[i][d].cell[DT])*lstmdecoder->hide[i][d].tan_out[DT]*diffsigmoid(lstmdecoder->hide[i][d].sig_in[DT]);
				lstmdecoder->hide[i][d].tan_diff[DT]=trans*lstmdecoder->hide[i][d].out_out[DT]*difftanh(lstmdecoder->hide[i][d].cell[DT])*lstmdecoder->hide[i][d].sig_out[DT]*difftanh(lstmdecoder->hide[i][d].tan_in[DT]);
				lstmdecoder->hide[i][d].out_diff[DT]=trans*tanh(lstmdecoder->hide[i][d].cell[DT])*diffsigmoid(lstmdecoder->hide[i][d].out_in[DT]);
			}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<HNUM;j++)
				trans+=lstmdecoder->hide[j][0].fog_diff[DT]*lstmdecoder->hide[j][0].fog_wi[i]+lstmdecoder->hide[j][0].sig_diff[DT]*lstmdecoder->hide[j][0].sig_wi[i]+lstmdecoder->hide[j][0].tan_diff[DT]*lstmdecoder->hide[j][0].tan_wi[i]+lstmdecoder->hide[j][0].out_diff[DT]*lstmdecoder->hide[j][0].out_wi[i];
			lstmdecoder->hlink[i].fog_diff[DT]=trans*lstmdecoder->hlink[i].out_out[DT]*difftanh(lstmdecoder->hlink[i].cell[DT])*lstmdecoder->hlink[i].cell[DT-1]*diffsigmoid(lstmdecoder->hlink[i].fog_in[DT]);
			lstmdecoder->hlink[i].sig_diff[DT]=trans*lstmdecoder->hlink[i].out_out[DT]*difftanh(lstmdecoder->hlink[i].cell[DT])*lstmdecoder->hlink[i].tan_out[DT]*diffsigmoid(lstmdecoder->hlink[i].sig_in[DT]);
			lstmdecoder->hlink[i].tan_diff[DT]=trans*lstmdecoder->hlink[i].out_out[DT]*difftanh(lstmdecoder->hlink[i].cell[DT])*lstmdecoder->hlink[i].sig_out[DT]*difftanh(lstmdecoder->hlink[i].tan_in[DT]);
			lstmdecoder->hlink[i].out_diff[DT]=trans*tanh(lstmdecoder->hlink[i].cell[DT])*diffsigmoid(lstmdecoder->hlink[i].out_in[DT]);
		}
		for(int t=DT-1;t>=1;t--)
		{
			for(int i=0;i<ONUM;i++)
			{
				for(int j=0;j<HNUM;j++)
					output[i].diff[t]+=lstmdecoder->hlink[j].fog_diff[t+1]*lstmdecoder->hlink[j].fog_wi[i]+lstmdecoder->hlink[j].sig_diff[t+1]*lstmdecoder->hlink[j].sig_wi[i]+lstmdecoder->hlink[j].tan_diff[t+1]*lstmdecoder->hlink[j].tan_wi[i]+lstmdecoder->hlink[j].out_diff[t+1]*lstmdecoder->hlink[j].out_wi[i];
				output[i].diff[t]*=output[i].out[t]*(1-output[i].out[t]);
			}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<ONUM;j++)
					trans+=output[j].diff[t]*output[j].w[i];
				for(int j=0;j<HNUM;j++)
					trans+=lstmdecoder->hide[j][DEPTH-1].fog_diff[t+1]*lstmdecoder->hide[j][DEPTH-1].fog_wh[i]+lstmdecoder->hide[j][DEPTH-1].sig_diff[t+1]*lstmdecoder->hide[j][DEPTH-1].sig_wh[i]+lstmdecoder->hide[j][DEPTH-1].tan_diff[t+1]*lstmdecoder->hide[j][DEPTH-1].tan_wh[i]+lstmdecoder->hide[j][DEPTH-1].out_diff[t+1]*lstmdecoder->hide[j][DEPTH-1].out_wh[i];
				lstmdecoder->hide[i][DEPTH-1].fog_diff[t]=trans*lstmdecoder->hide[i][DEPTH-1].out_out[t]*difftanh(lstmdecoder->hide[i][DEPTH-1].cell[t])*lstmdecoder->hide[i][DEPTH-1].cell[t-1]*diffsigmoid(lstmdecoder->hide[i][DEPTH-1].fog_in[t]);
				lstmdecoder->hide[i][DEPTH-1].sig_diff[t]=trans*lstmdecoder->hide[i][DEPTH-1].out_out[t]*difftanh(lstmdecoder->hide[i][DEPTH-1].cell[t])*lstmdecoder->hide[i][DEPTH-1].tan_out[t]*diffsigmoid(lstmdecoder->hide[i][DEPTH-1].sig_in[t]);
				lstmdecoder->hide[i][DEPTH-1].tan_diff[t]=trans*lstmdecoder->hide[i][DEPTH-1].out_out[t]*difftanh(lstmdecoder->hide[i][DEPTH-1].cell[t])*lstmdecoder->hide[i][DEPTH-1].sig_out[t]*difftanh(lstmdecoder->hide[i][DEPTH-1].tan_in[t]);
				lstmdecoder->hide[i][DEPTH-1].out_diff[t]=trans*tanh(lstmdecoder->hide[i][DEPTH-1].cell[t])*diffsigmoid(lstmdecoder->hide[i][DEPTH-1].out_in[t]);
			}
			for(int d=DEPTH-2;d>=0;d--)
				for(int i=0;i<HNUM;i++)
				{
					trans=0;
					for(int j=0;j<HNUM;j++)
					{
						trans+=lstmdecoder->hide[j][d+1].fog_diff[t]*lstmdecoder->hide[j][d+1].fog_wi[i]+lstmdecoder->hide[j][d+1].sig_diff[t]*lstmdecoder->hide[j][d+1].sig_wi[i]+lstmdecoder->hide[j][d+1].tan_diff[t]*lstmdecoder->hide[j][d+1].tan_wi[i]+lstmdecoder->hide[j][d+1].out_diff[t]*lstmdecoder->hide[j][d+1].out_wi[i];
						trans+=lstmdecoder->hide[j][d].fog_diff[t+1]*lstmdecoder->hide[j][d].fog_wh[i]+lstmdecoder->hide[j][d].sig_diff[t+1]*lstmdecoder->hide[j][d].sig_wh[i]+lstmdecoder->hide[j][d].tan_diff[t+1]*lstmdecoder->hide[j][d].tan_wh[i]+lstmdecoder->hide[j][d].out_diff[t+1]*lstmdecoder->hide[j][d].out_wh[i];
					}
					lstmdecoder->hide[i][d].fog_diff[t]=trans*lstmdecoder->hide[i][d].out_out[t]*difftanh(lstmdecoder->hide[i][d].cell[t])*lstmdecoder->hide[i][d].cell[t-1]*diffsigmoid(lstmdecoder->hide[i][d].fog_in[t]);
					lstmdecoder->hide[i][d].sig_diff[t]=trans*lstmdecoder->hide[i][d].out_out[t]*difftanh(lstmdecoder->hide[i][d].cell[t])*lstmdecoder->hide[i][d].tan_out[t]*diffsigmoid(lstmdecoder->hide[i][d].sig_in[t]);
					lstmdecoder->hide[i][d].tan_diff[t]=trans*lstmdecoder->hide[i][d].out_out[t]*difftanh(lstmdecoder->hide[i][d].cell[t])*lstmdecoder->hide[i][d].sig_out[t]*difftanh(lstmdecoder->hide[i][d].tan_in[t]);
					lstmdecoder->hide[i][d].out_diff[t]=trans*tanh(lstmdecoder->hide[i][d].cell[t])*diffsigmoid(lstmdecoder->hide[i][d].out_in[t]);
				}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
				{
					trans+=lstmdecoder->hide[j][0].fog_diff[t]*lstmdecoder->hide[j][0].fog_wi[i]+lstmdecoder->hide[j][0].sig_diff[t]*lstmdecoder->hide[j][0].sig_wi[i]+lstmdecoder->hide[j][0].tan_diff[t]*lstmdecoder->hide[j][0].tan_wi[i]+lstmdecoder->hide[j][0].out_diff[t]*lstmdecoder->hide[j][0].out_wi[i];
					trans+=lstmdecoder->hlink[j].fog_diff[t+1]*lstmdecoder->hlink[j].fog_wh[i]+lstmdecoder->hlink[j].sig_diff[t+1]*lstmdecoder->hlink[j].sig_wh[i]+lstmdecoder->hlink[j].tan_diff[t+1]*lstmdecoder->hlink[j].tan_wh[i]+lstmdecoder->hlink[j].out_diff[t+1]*lstmdecoder->hlink[j].out_wh[i];
				}
				lstmdecoder->hlink[i].fog_diff[t]=trans*lstmdecoder->hlink[i].out_out[t]*difftanh(lstmdecoder->hlink[i].cell[t])*lstmdecoder->hlink[i].cell[t-1]*diffsigmoid(lstmdecoder->hlink[i].fog_in[t]);
				lstmdecoder->hlink[i].sig_diff[t]=trans*lstmdecoder->hlink[i].out_out[t]*difftanh(lstmdecoder->hlink[i].cell[t])*lstmdecoder->hlink[i].tan_out[t]*diffsigmoid(lstmdecoder->hlink[i].sig_in[t]);
				lstmdecoder->hlink[i].tan_diff[t]=trans*lstmdecoder->hlink[i].out_out[t]*difftanh(lstmdecoder->hlink[i].cell[t])*lstmdecoder->hlink[i].sig_out[t]*difftanh(lstmdecoder->hlink[i].tan_in[t]);
				lstmdecoder->hlink[i].out_diff[t]=trans*tanh(lstmdecoder->hlink[i].cell[t])*diffsigmoid(lstmdecoder->hlink[i].out_in[t]);
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			for(int j=0;j<HNUM;j++)
				output[i].diff[0]+=lstmdecoder->hlink[j].fog_diff[1]*lstmdecoder->hlink[j].fog_wi[i]+lstmdecoder->hlink[j].sig_diff[1]*lstmdecoder->hlink[j].sig_wi[i]+lstmdecoder->hlink[j].tan_diff[1]*lstmdecoder->hlink[j].tan_wi[i]+lstmdecoder->hlink[j].out_diff[1]*lstmdecoder->hlink[j].out_wi[i];
			output[i].diff[0]*=output[i].out[0]*(1-output[i].out[0]);
		}
		
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff[0]*output[j].w[i];
			for(int j=0;j<HNUM;j++)
				trans+=lstmdecoder->hide[j][DEPTH-1].fog_diff[1]*lstmdecoder->hide[j][DEPTH-1].fog_wh[i]+lstmdecoder->hide[j][DEPTH-1].sig_diff[1]*lstmdecoder->hide[j][DEPTH-1].sig_wh[i]+lstmdecoder->hide[j][DEPTH-1].tan_diff[1]*lstmdecoder->hide[j][DEPTH-1].tan_wh[i]+lstmdecoder->hide[j][DEPTH-1].out_diff[1]*lstmdecoder->hide[j][DEPTH-1].out_wh[i];
			lstmencoder->hide[i][DEPTH-1].fog_diff[ET]=trans*lstmencoder->hide[i][DEPTH-1].out_out[ET]*difftanh(lstmencoder->hide[i][DEPTH-1].cell[ET])*lstmencoder->hide[i][DEPTH-1].cell[ET-1]*diffsigmoid(lstmencoder->hide[i][DEPTH-1].fog_in[ET]);
			lstmencoder->hide[i][DEPTH-1].sig_diff[ET]=trans*lstmencoder->hide[i][DEPTH-1].out_out[ET]*difftanh(lstmencoder->hide[i][DEPTH-1].cell[ET])*lstmencoder->hide[i][DEPTH-1].tan_out[ET]*diffsigmoid(lstmencoder->hide[i][DEPTH-1].sig_in[ET]);
			lstmencoder->hide[i][DEPTH-1].tan_diff[ET]=trans*lstmencoder->hide[i][DEPTH-1].out_out[ET]*difftanh(lstmencoder->hide[i][DEPTH-1].cell[ET])*lstmencoder->hide[i][DEPTH-1].sig_out[ET]*difftanh(lstmencoder->hide[i][DEPTH-1].tan_in[ET]);
			lstmencoder->hide[i][DEPTH-1].out_diff[ET]=trans*tanh(lstmencoder->hide[i][DEPTH-1].cell[ET])*diffsigmoid(lstmencoder->hide[i][DEPTH-1].out_in[ET]);
		}
		for(int d=DEPTH-2;d>=0;d--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
				{
					trans+=lstmencoder->hide[j][d+1].fog_diff[ET]*lstmencoder->hide[j][d+1].fog_wi[i]+lstmencoder->hide[j][d+1].sig_diff[ET]*lstmencoder->hide[j][d+1].sig_wi[i]+lstmencoder->hide[j][d+1].tan_diff[ET]*lstmencoder->hide[j][d+1].tan_wi[i]+lstmencoder->hide[j][d+1].out_diff[ET]*lstmencoder->hide[j][d+1].out_wi[i];
					trans+=lstmdecoder->hide[j][d].fog_diff[1]*lstmdecoder->hide[j][d].fog_wh[i]+lstmdecoder->hide[j][d].sig_diff[1]*lstmdecoder->hide[j][d].sig_wh[i]+lstmdecoder->hide[j][d].tan_diff[1]*lstmdecoder->hide[j][d].tan_wh[i]+lstmdecoder->hide[j][d].out_diff[1]*lstmdecoder->hide[j][d].out_wh[i];
				}
				lstmencoder->hide[i][d].fog_diff[ET]=trans*lstmencoder->hide[i][d].out_out[ET]*difftanh(lstmencoder->hide[i][d].cell[ET])*lstmencoder->hide[i][d].cell[ET-1]*diffsigmoid(lstmencoder->hide[i][d].fog_in[ET]);
				lstmencoder->hide[i][d].sig_diff[ET]=trans*lstmencoder->hide[i][d].out_out[ET]*difftanh(lstmencoder->hide[i][d].cell[ET])*lstmencoder->hide[i][d].tan_out[ET]*diffsigmoid(lstmencoder->hide[i][d].sig_in[ET]);
				lstmencoder->hide[i][d].tan_diff[ET]=trans*lstmencoder->hide[i][d].out_out[ET]*difftanh(lstmencoder->hide[i][d].cell[ET])*lstmencoder->hide[i][d].sig_out[ET]*difftanh(lstmencoder->hide[i][d].tan_in[ET]);
				lstmencoder->hide[i][d].out_diff[ET]=trans*tanh(lstmencoder->hide[i][d].cell[ET])*diffsigmoid(lstmencoder->hide[i][d].out_in[ET]);
			}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<HNUM;j++)
			{
				trans+=lstmencoder->hide[j][0].fog_diff[ET]*lstmencoder->hide[j][0].fog_wi[i]+lstmencoder->hide[j][0].sig_diff[ET]*lstmencoder->hide[j][0].sig_wi[i]+lstmencoder->hide[j][0].tan_diff[ET]*lstmencoder->hide[j][0].tan_wi[i]+lstmencoder->hide[j][0].out_diff[ET]*lstmencoder->hide[j][0].out_wi[i];
				trans+=lstmdecoder->hlink[j].fog_diff[1]*lstmdecoder->hlink[j].fog_wh[i]+lstmdecoder->hlink[j].sig_diff[1]*lstmdecoder->hlink[j].sig_wh[i]+lstmdecoder->hlink[j].tan_diff[1]*lstmdecoder->hlink[j].tan_wh[i]+lstmdecoder->hlink[j].out_diff[1]*lstmdecoder->hlink[j].out_wh[i];
			}
			lstmencoder->hlink[i].fog_diff[ET]=trans*lstmencoder->hlink[i].out_out[ET]*difftanh(lstmencoder->hlink[i].cell[ET])*lstmencoder->hlink[i].cell[ET-1]*diffsigmoid(lstmencoder->hlink[i].fog_in[ET]);
			lstmencoder->hlink[i].sig_diff[ET]=trans*lstmencoder->hlink[i].out_out[ET]*difftanh(lstmencoder->hlink[i].cell[ET])*lstmencoder->hlink[i].tan_out[ET]*diffsigmoid(lstmencoder->hlink[i].sig_in[ET]);
			lstmencoder->hlink[i].tan_diff[ET]=trans*lstmencoder->hlink[i].out_out[ET]*difftanh(lstmencoder->hlink[i].cell[ET])*lstmencoder->hlink[i].sig_out[ET]*difftanh(lstmencoder->hlink[i].tan_in[ET]);
			lstmencoder->hlink[i].out_diff[ET]=trans*tanh(lstmencoder->hlink[i].cell[ET])*diffsigmoid(lstmencoder->hlink[i].out_in[ET]);
		}
		for(int t=ET-1;t>=1;t--)
		{
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=lstmencoder->hide[j][DEPTH-1].fog_diff[t+1]*lstmencoder->hide[j][DEPTH-1].fog_wh[i]+lstmencoder->hide[j][DEPTH-1].sig_diff[t+1]*lstmencoder->hide[j][DEPTH-1].sig_wh[i]+lstmencoder->hide[j][DEPTH-1].tan_diff[t+1]*lstmencoder->hide[j][DEPTH-1].tan_wh[i]+lstmencoder->hide[j][DEPTH-1].out_diff[t+1]*lstmencoder->hide[j][DEPTH-1].out_wh[i];
				lstmencoder->hide[i][DEPTH-1].fog_diff[t]=trans*lstmencoder->hide[i][DEPTH-1].out_out[t]*difftanh(lstmencoder->hide[i][DEPTH-1].cell[t])*lstmencoder->hide[i][DEPTH-1].cell[t-1]*diffsigmoid(lstmencoder->hide[i][DEPTH-1].fog_in[t]);
				lstmencoder->hide[i][DEPTH-1].sig_diff[t]=trans*lstmencoder->hide[i][DEPTH-1].out_out[t]*difftanh(lstmencoder->hide[i][DEPTH-1].cell[t])*lstmencoder->hide[i][DEPTH-1].tan_out[t]*diffsigmoid(lstmencoder->hide[i][DEPTH-1].sig_in[t]);
				lstmencoder->hide[i][DEPTH-1].tan_diff[t]=trans*lstmencoder->hide[i][DEPTH-1].out_out[t]*difftanh(lstmencoder->hide[i][DEPTH-1].cell[t])*lstmencoder->hide[i][DEPTH-1].sig_out[t]*difftanh(lstmencoder->hide[i][DEPTH-1].tan_in[t]);
				lstmencoder->hide[i][DEPTH-1].out_diff[t]=trans*tanh(lstmencoder->hide[i][DEPTH-1].cell[t])*diffsigmoid(lstmencoder->hide[i][DEPTH-1].out_in[t]);
			}
			for(int d=DEPTH-2;d>=0;d--)
				for(int i=0;i<HNUM;i++)
				{
					trans=0;
					for(int j=0;j<HNUM;j++)
					{
						trans+=lstmencoder->hide[j][d+1].fog_diff[t]*lstmencoder->hide[j][d+1].fog_wi[i]+lstmencoder->hide[j][d+1].sig_diff[t]*lstmencoder->hide[j][d+1].sig_wi[i]+lstmencoder->hide[j][d+1].tan_diff[t]*lstmencoder->hide[j][d+1].tan_wi[i]+lstmencoder->hide[j][d+1].out_diff[t]*lstmencoder->hide[j][d+1].out_wi[i];
						trans+=lstmencoder->hide[j][d].fog_diff[t+1]*lstmencoder->hide[j][d].fog_wh[i]+lstmencoder->hide[j][d].sig_diff[t+1]*lstmencoder->hide[j][d].sig_wh[i]+lstmencoder->hide[j][d].tan_diff[t+1]*lstmencoder->hide[j][d].tan_wh[i]+lstmencoder->hide[j][d].out_diff[t+1]*lstmencoder->hide[j][d].out_wh[i];
					}
					lstmencoder->hide[i][d].fog_diff[t]=trans*lstmencoder->hide[i][d].out_out[t]*difftanh(lstmencoder->hide[i][d].cell[t])*lstmencoder->hide[i][d].cell[t-1]*diffsigmoid(lstmencoder->hide[i][d].fog_in[t]);
					lstmencoder->hide[i][d].sig_diff[t]=trans*lstmencoder->hide[i][d].out_out[t]*difftanh(lstmencoder->hide[i][d].cell[t])*lstmencoder->hide[i][d].tan_out[t]*diffsigmoid(lstmencoder->hide[i][d].sig_in[t]);
					lstmencoder->hide[i][d].tan_diff[t]=trans*lstmencoder->hide[i][d].out_out[t]*difftanh(lstmencoder->hide[i][d].cell[t])*lstmencoder->hide[i][d].sig_out[t]*difftanh(lstmencoder->hide[i][d].tan_in[t]);
					lstmencoder->hide[i][d].out_diff[t]=trans*tanh(lstmencoder->hide[i][d].cell[t])*diffsigmoid(lstmencoder->hide[i][d].out_in[t]);
				}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
				{
					trans+=lstmencoder->hide[j][0].fog_diff[t]*lstmencoder->hide[j][0].fog_wi[i]+lstmencoder->hide[j][0].sig_diff[t]*lstmencoder->hide[j][0].sig_wi[i]+lstmencoder->hide[j][0].tan_diff[t]*lstmencoder->hide[j][0].tan_wi[i]+lstmencoder->hide[j][0].out_diff[t]*lstmencoder->hide[j][0].out_wi[i];
					trans+=lstmencoder->hlink[j].fog_diff[t+1]*lstmencoder->hlink[j].fog_wh[i]+lstmencoder->hlink[j].sig_diff[t+1]*lstmencoder->hlink[j].sig_wh[i]+lstmencoder->hlink[j].tan_diff[t+1]*lstmencoder->hlink[j].tan_wh[i]+lstmencoder->hlink[j].out_diff[t+1]*lstmencoder->hlink[j].out_wh[i];
				}
				lstmencoder->hlink[i].fog_diff[t]=trans*lstmencoder->hlink[i].out_out[t]*difftanh(lstmencoder->hlink[i].cell[t])*lstmencoder->hlink[i].cell[t-1]*diffsigmoid(lstmencoder->hlink[i].fog_in[t]);
				lstmencoder->hlink[i].sig_diff[t]=trans*lstmencoder->hlink[i].out_out[t]*difftanh(lstmencoder->hlink[i].cell[t])*lstmencoder->hlink[i].tan_out[t]*diffsigmoid(lstmencoder->hlink[i].sig_in[t]);
				lstmencoder->hlink[i].tan_diff[t]=trans*lstmencoder->hlink[i].out_out[t]*difftanh(lstmencoder->hlink[i].cell[t])*lstmencoder->hlink[i].sig_out[t]*difftanh(lstmencoder->hlink[i].tan_in[t]);
				lstmencoder->hlink[i].out_diff[t]=trans*tanh(lstmencoder->hlink[i].cell[t])*diffsigmoid(lstmencoder->hlink[i].out_in[t]);
			}
		}
		for(int i=0;i<HNUM;i++)
		{
			lstmencoder->hlink[i].fog_transbia=0;
			lstmencoder->hlink[i].sig_transbia=0;
			lstmencoder->hlink[i].tan_transbia=0;
			lstmencoder->hlink[i].out_transbia=0;
			for(int j=0;j<INUM;j++)
			{
				lstmencoder->hlink[i].fog_transwi[j]=0;
				lstmencoder->hlink[i].sig_transwi[j]=0;
				lstmencoder->hlink[i].tan_transwi[j]=0;
				lstmencoder->hlink[i].out_transwi[j]=0;
			}
			for(int j=0;j<HNUM;j++)
			{
				lstmencoder->hlink[i].fog_transwh[j]=0;
				lstmencoder->hlink[i].sig_transwh[j]=0;
				lstmencoder->hlink[i].tan_transwh[j]=0;
				lstmencoder->hlink[i].out_transwh[j]=0;
			}
		}
		for(int d=0;d<DEPTH;d++)
			for(int i=0;i<HNUM;i++)
			{
				lstmencoder->hide[i][d].fog_transbia=0;
				lstmencoder->hide[i][d].sig_transbia=0;
				lstmencoder->hide[i][d].tan_transbia=0;
				lstmencoder->hide[i][d].out_transbia=0;
				for(int j=0;j<HNUM;j++)
				{
					lstmencoder->hide[i][d].fog_transwi[j]=0;
					lstmencoder->hide[i][d].sig_transwi[j]=0;
					lstmencoder->hide[i][d].tan_transwi[j]=0;
					lstmencoder->hide[i][d].out_transwi[j]=0;
					lstmencoder->hide[i][d].fog_transwh[j]=0;
					lstmencoder->hide[i][d].sig_transwh[j]=0;
					lstmencoder->hide[i][d].tan_transwh[j]=0;
					lstmencoder->hide[i][d].out_transwh[j]=0;
				}
			}
		for(int t=1;t<=ET;t++)
		{
			for(int d=DEPTH-1;d>=0;d--)
			{
				for(int i=0;i<HNUM;i++)
				{
					lstmencoder->hide[i][d].fog_transbia+=2*lstmencoder->hide[i][d].fog_diff[t];
					lstmencoder->hide[i][d].sig_transbia+=2*lstmencoder->hide[i][d].sig_diff[t];
					lstmencoder->hide[i][d].tan_transbia+=2*lstmencoder->hide[i][d].tan_diff[t];
					lstmencoder->hide[i][d].out_transbia+=2*lstmencoder->hide[i][d].out_diff[t];
					for(int j=0;j<HNUM;j++)
					{
						lstmencoder->hide[i][d].fog_transwh[j]+=lstmencoder->hide[i][d].fog_diff[t]*lstmencoder->hide[j][d].out[t-1];
						lstmencoder->hide[i][d].sig_transwh[j]+=lstmencoder->hide[i][d].sig_diff[t]*lstmencoder->hide[j][d].out[t-1];
						lstmencoder->hide[i][d].tan_transwh[j]+=lstmencoder->hide[i][d].tan_diff[t]*lstmencoder->hide[j][d].out[t-1];
						lstmencoder->hide[i][d].out_transwh[j]+=lstmencoder->hide[i][d].out_diff[t]*lstmencoder->hide[j][d].out[t-1];
						lstmencoder->hide[i][d].fog_transwi[j]+=lstmencoder->hide[i][d].fog_diff[t]*(d>0? lstmencoder->hide[j][d-1].out[t]:lstmencoder->hlink[j].out[t]);
						lstmencoder->hide[i][d].sig_transwi[j]+=lstmencoder->hide[i][d].sig_diff[t]*(d>0? lstmencoder->hide[j][d-1].out[t]:lstmencoder->hlink[j].out[t]);
						lstmencoder->hide[i][d].tan_transwi[j]+=lstmencoder->hide[i][d].tan_diff[t]*(d>0? lstmencoder->hide[j][d-1].out[t]:lstmencoder->hlink[j].out[t]);
						lstmencoder->hide[i][d].out_transwi[j]+=lstmencoder->hide[i][d].out_diff[t]*(d>0? lstmencoder->hide[j][d-1].out[t]:lstmencoder->hlink[j].out[t]);
					}
				}
			}
			for(int i=0;i<HNUM;i++)
			{
				lstmencoder->hlink[i].fog_transbia+=2*lstmencoder->hlink[i].fog_diff[t];
				lstmencoder->hlink[i].sig_transbia+=2*lstmencoder->hlink[i].sig_diff[t];
				lstmencoder->hlink[i].tan_transbia+=2*lstmencoder->hlink[i].tan_diff[t];
				lstmencoder->hlink[i].out_transbia+=2*lstmencoder->hlink[i].out_diff[t];
				for(int j=0;j<HNUM;j++)
				{
					lstmencoder->hlink[i].fog_transwh[j]+=lstmencoder->hlink[i].fog_diff[t]*lstmencoder->hlink[j].out[t-1];
					lstmencoder->hlink[i].sig_transwh[j]+=lstmencoder->hlink[i].sig_diff[t]*lstmencoder->hlink[j].out[t-1];
					lstmencoder->hlink[i].tan_transwh[j]+=lstmencoder->hlink[i].tan_diff[t]*lstmencoder->hlink[j].out[t-1];
					lstmencoder->hlink[i].out_transwh[j]+=lstmencoder->hlink[i].out_diff[t]*lstmencoder->hlink[j].out[t-1];
				}
				for(int j=0;j<INUM;j++)
				{
					lstmencoder->hlink[i].fog_transwi[j]+=lstmencoder->hlink[i].fog_diff[t]*input[j][t];
					lstmencoder->hlink[i].sig_transwi[j]+=lstmencoder->hlink[i].sig_diff[t]*input[j][t];
					lstmencoder->hlink[i].tan_transwi[j]+=lstmencoder->hlink[i].tan_diff[t]*input[j][t];
					lstmencoder->hlink[i].out_transwi[j]+=lstmencoder->hlink[i].out_diff[t]*input[j][t];
				}
			}
		}
			
		for(int i=0;i<HNUM;i++)
		{
			lstmencoder->hlink[i].fog_bia+=ClipGradient(learningrate*lstmencoder->hlink[i].fog_transbia);
			lstmencoder->hlink[i].sig_bia+=ClipGradient(learningrate*lstmencoder->hlink[i].sig_transbia);
			lstmencoder->hlink[i].tan_bia+=ClipGradient(learningrate*lstmencoder->hlink[i].tan_transbia);
			lstmencoder->hlink[i].out_bia+=ClipGradient(learningrate*lstmencoder->hlink[i].out_transbia);
			for(int j=0;j<INUM;j++)
			{
				lstmencoder->hlink[i].fog_wi[j]+=ClipGradient(learningrate*lstmencoder->hlink[i].fog_transwi[j]);
				lstmencoder->hlink[i].sig_wi[j]+=ClipGradient(learningrate*lstmencoder->hlink[i].sig_transwi[j]);
				lstmencoder->hlink[i].tan_wi[j]+=ClipGradient(learningrate*lstmencoder->hlink[i].tan_transwi[j]);
				lstmencoder->hlink[i].out_wi[j]+=ClipGradient(learningrate*lstmencoder->hlink[i].out_transwi[j]);
			}
			for(int j=0;j<HNUM;j++)
			{
				lstmencoder->hlink[i].fog_wh[j]+=ClipGradient(learningrate*lstmencoder->hlink[i].fog_transwh[j]);
				lstmencoder->hlink[i].sig_wh[j]+=ClipGradient(learningrate*lstmencoder->hlink[i].sig_transwh[j]);
				lstmencoder->hlink[i].tan_wh[j]+=ClipGradient(learningrate*lstmencoder->hlink[i].tan_transwh[j]);
				lstmencoder->hlink[i].out_wh[j]+=ClipGradient(learningrate*lstmencoder->hlink[i].out_transwh[j]);
			}
		}
		for(int d=0;d<DEPTH;d++)
		{
				for(int i=0;i<HNUM;i++)
				{
					lstmencoder->hide[i][d].fog_bia+=ClipGradient(learningrate*lstmencoder->hide[i][d].fog_transbia);
					lstmencoder->hide[i][d].sig_bia+=ClipGradient(learningrate*lstmencoder->hide[i][d].sig_transbia);
					lstmencoder->hide[i][d].tan_bia+=ClipGradient(learningrate*lstmencoder->hide[i][d].tan_transbia);
					lstmencoder->hide[i][d].out_bia+=ClipGradient(learningrate*lstmencoder->hide[i][d].out_transbia);
					for(int j=0;j<HNUM;j++)
					{
						lstmencoder->hide[i][d].fog_wi[j]+=ClipGradient(learningrate*lstmencoder->hide[i][d].fog_transwi[j]);
						lstmencoder->hide[i][d].sig_wi[j]+=ClipGradient(learningrate*lstmencoder->hide[i][d].sig_transwi[j]);
						lstmencoder->hide[i][d].tan_wi[j]+=ClipGradient(learningrate*lstmencoder->hide[i][d].tan_transwi[j]);
						lstmencoder->hide[i][d].out_wi[j]+=ClipGradient(learningrate*lstmencoder->hide[i][d].out_transwi[j]);
						lstmencoder->hide[i][d].fog_wh[j]+=ClipGradient(learningrate*lstmencoder->hide[i][d].fog_transwh[j]);
						lstmencoder->hide[i][d].sig_wh[j]+=ClipGradient(learningrate*lstmencoder->hide[i][d].sig_transwh[j]);
						lstmencoder->hide[i][d].tan_wh[j]+=ClipGradient(learningrate*lstmencoder->hide[i][d].tan_transwh[j]);
						lstmencoder->hide[i][d].out_wh[j]+=ClipGradient(learningrate*lstmencoder->hide[i][d].out_transwh[j]);
					}
				}
		}
		for(int i=0;i<HNUM;i++)
		{
			lstmdecoder->hlink[i].fog_transbia=0;
			lstmdecoder->hlink[i].sig_transbia=0;
			lstmdecoder->hlink[i].tan_transbia=0;
			lstmdecoder->hlink[i].out_transbia=0;
			for(int j=0;j<INUM;j++)
			{
				lstmdecoder->hlink[i].fog_transwi[j]=0;
				lstmdecoder->hlink[i].sig_transwi[j]=0;
				lstmdecoder->hlink[i].tan_transwi[j]=0;
				lstmdecoder->hlink[i].out_transwi[j]=0;
			}
			for(int j=0;j<HNUM;j++)
			{
				lstmdecoder->hlink[i].fog_transwh[j]=0;
				lstmdecoder->hlink[i].sig_transwh[j]=0;
				lstmdecoder->hlink[i].tan_transwh[j]=0;
				lstmdecoder->hlink[i].out_transwh[j]=0;
			}
		}
		for(int d=0;d<DEPTH;d++)
			for(int i=0;i<HNUM;i++)
			{
				lstmdecoder->hide[i][d].fog_transbia=0;
				lstmdecoder->hide[i][d].sig_transbia=0;
				lstmdecoder->hide[i][d].tan_transbia=0;
				lstmdecoder->hide[i][d].out_transbia=0;
				for(int j=0;j<HNUM;j++)
				{
					lstmdecoder->hide[i][d].fog_transwi[j]=0;
					lstmdecoder->hide[i][d].sig_transwi[j]=0;
					lstmdecoder->hide[i][d].tan_transwi[j]=0;
					lstmdecoder->hide[i][d].out_transwi[j]=0;
					lstmdecoder->hide[i][d].fog_transwh[j]=0;
					lstmdecoder->hide[i][d].sig_transwh[j]=0;
					lstmdecoder->hide[i][d].tan_transwh[j]=0;
					lstmdecoder->hide[i][d].out_transwh[j]=0;
				}
			}
		for(int i=0;i<ONUM;i++)
		{
			output[i].transbia=0;
			for(int j=0;j<HNUM;j++)
				output[i].transw[j]=0;
		}
		for(int t=1;t<=DT;t++)
		{
			for(int d=DEPTH-1;d>=0;d--)
			{
				for(int i=0;i<HNUM;i++)
				{
					lstmdecoder->hide[i][d].fog_transbia+=2*lstmdecoder->hide[i][d].fog_diff[t];
					lstmdecoder->hide[i][d].sig_transbia+=2*lstmdecoder->hide[i][d].sig_diff[t];
					lstmdecoder->hide[i][d].tan_transbia+=2*lstmdecoder->hide[i][d].tan_diff[t];
					lstmdecoder->hide[i][d].out_transbia+=2*lstmdecoder->hide[i][d].out_diff[t];
					for(int j=0;j<HNUM;j++)
					{
						lstmdecoder->hide[i][d].fog_transwh[j]+=lstmdecoder->hide[i][d].fog_diff[t]*lstmdecoder->hide[j][d].out[t-1];
						lstmdecoder->hide[i][d].sig_transwh[j]+=lstmdecoder->hide[i][d].sig_diff[t]*lstmdecoder->hide[j][d].out[t-1];
						lstmdecoder->hide[i][d].tan_transwh[j]+=lstmdecoder->hide[i][d].tan_diff[t]*lstmdecoder->hide[j][d].out[t-1];
						lstmdecoder->hide[i][d].out_transwh[j]+=lstmdecoder->hide[i][d].out_diff[t]*lstmdecoder->hide[j][d].out[t-1];
						lstmdecoder->hide[i][d].fog_transwi[j]+=lstmdecoder->hide[i][d].fog_diff[t]*(d>0? lstmdecoder->hide[j][d-1].out[t]:lstmdecoder->hlink[j].out[t]);
						lstmdecoder->hide[i][d].sig_transwi[j]+=lstmdecoder->hide[i][d].sig_diff[t]*(d>0? lstmdecoder->hide[j][d-1].out[t]:lstmdecoder->hlink[j].out[t]);
						lstmdecoder->hide[i][d].tan_transwi[j]+=lstmdecoder->hide[i][d].tan_diff[t]*(d>0? lstmdecoder->hide[j][d-1].out[t]:lstmdecoder->hlink[j].out[t]);
						lstmdecoder->hide[i][d].out_transwi[j]+=lstmdecoder->hide[i][d].out_diff[t]*(d>0? lstmdecoder->hide[j][d-1].out[t]:lstmdecoder->hlink[j].out[t]);
					}
				}
			}
			for(int i=0;i<HNUM;i++)
			{
				lstmdecoder->hlink[i].fog_transbia+=2*lstmdecoder->hlink[i].fog_diff[t];
				lstmdecoder->hlink[i].sig_transbia+=2*lstmdecoder->hlink[i].sig_diff[t];
				lstmdecoder->hlink[i].tan_transbia+=2*lstmdecoder->hlink[i].tan_diff[t];
				lstmdecoder->hlink[i].out_transbia+=2*lstmdecoder->hlink[i].out_diff[t];
				for(int j=0;j<HNUM;j++)
				{
					lstmdecoder->hlink[i].fog_transwh[j]+=lstmdecoder->hlink[i].fog_diff[t]*lstmdecoder->hlink[j].out[t-1];
					lstmdecoder->hlink[i].sig_transwh[j]+=lstmdecoder->hlink[i].sig_diff[t]*lstmdecoder->hlink[j].out[t-1];
					lstmdecoder->hlink[i].tan_transwh[j]+=lstmdecoder->hlink[i].tan_diff[t]*lstmdecoder->hlink[j].out[t-1];
					lstmdecoder->hlink[i].out_transwh[j]+=lstmdecoder->hlink[i].out_diff[t]*lstmdecoder->hlink[j].out[t-1];
				}
				for(int j=0;j<INUM;j++)
				{
					lstmdecoder->hlink[i].fog_transwi[j]+=lstmdecoder->hlink[i].fog_diff[t]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hlink[i].sig_transwi[j]+=lstmdecoder->hlink[i].sig_diff[t]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hlink[i].tan_transwi[j]+=lstmdecoder->hlink[i].tan_diff[t]*expect[j][t-1];//output[j].out[t-1];
					lstmdecoder->hlink[i].out_transwi[j]+=lstmdecoder->hlink[i].out_diff[t]*expect[j][t-1];//output[j].out[t-1];
				}
			}
		}
		for(int t=0;t<=DT;t++)
			for(int i=0;i<ONUM;i++)
			{
				output[i].transbia+=2*output[i].diff[t];
				for(int j=0;j<HNUM;j++)
					output[i].transw[j]+=output[i].diff[t]*lstmdecoder->hide[j][DEPTH-1].out[t];
			}
	
		for(int i=0;i<HNUM;i++)
		{
			lstmdecoder->hlink[i].fog_bia+=ClipGradient(learningrate*lstmdecoder->hlink[i].fog_transbia);
			lstmdecoder->hlink[i].sig_bia+=ClipGradient(learningrate*lstmdecoder->hlink[i].sig_transbia);
			lstmdecoder->hlink[i].tan_bia+=ClipGradient(learningrate*lstmdecoder->hlink[i].tan_transbia);
			lstmdecoder->hlink[i].out_bia+=ClipGradient(learningrate*lstmdecoder->hlink[i].out_transbia);
			for(int j=0;j<INUM;j++)
			{
				lstmdecoder->hlink[i].fog_wi[j]+=ClipGradient(learningrate*lstmdecoder->hlink[i].fog_transwi[j]);
				lstmdecoder->hlink[i].sig_wi[j]+=ClipGradient(learningrate*lstmdecoder->hlink[i].sig_transwi[j]);
				lstmdecoder->hlink[i].tan_wi[j]+=ClipGradient(learningrate*lstmdecoder->hlink[i].tan_transwi[j]);
				lstmdecoder->hlink[i].out_wi[j]+=ClipGradient(learningrate*lstmdecoder->hlink[i].out_transwi[j]);
			}
			for(int j=0;j<HNUM;j++)
			{
				lstmdecoder->hlink[i].fog_wh[j]+=ClipGradient(learningrate*lstmdecoder->hlink[i].fog_transwh[j]);
				lstmdecoder->hlink[i].sig_wh[j]+=ClipGradient(learningrate*lstmdecoder->hlink[i].sig_transwh[j]);
				lstmdecoder->hlink[i].tan_wh[j]+=ClipGradient(learningrate*lstmdecoder->hlink[i].tan_transwh[j]);
				lstmdecoder->hlink[i].out_wh[j]+=ClipGradient(learningrate*lstmdecoder->hlink[i].out_transwh[j]);
			}
		}
		for(int d=0;d<DEPTH;d++)
		{
				for(int i=0;i<HNUM;i++)
				{
					lstmdecoder->hide[i][d].fog_bia+=ClipGradient(learningrate*lstmdecoder->hide[i][d].fog_transbia);
					lstmdecoder->hide[i][d].sig_bia+=ClipGradient(learningrate*lstmdecoder->hide[i][d].sig_transbia);
					lstmdecoder->hide[i][d].tan_bia+=ClipGradient(learningrate*lstmdecoder->hide[i][d].tan_transbia);
					lstmdecoder->hide[i][d].out_bia+=ClipGradient(learningrate*lstmdecoder->hide[i][d].out_transbia);
					for(int j=0;j<HNUM;j++)
					{
						lstmdecoder->hide[i][d].fog_wi[j]+=ClipGradient(learningrate*lstmdecoder->hide[i][d].fog_transwi[j]);
						lstmdecoder->hide[i][d].sig_wi[j]+=ClipGradient(learningrate*lstmdecoder->hide[i][d].sig_transwi[j]);
						lstmdecoder->hide[i][d].tan_wi[j]+=ClipGradient(learningrate*lstmdecoder->hide[i][d].tan_transwi[j]);
						lstmdecoder->hide[i][d].out_wi[j]+=ClipGradient(learningrate*lstmdecoder->hide[i][d].out_transwi[j]);
						lstmdecoder->hide[i][d].fog_wh[j]+=ClipGradient(learningrate*lstmdecoder->hide[i][d].fog_transwh[j]);
						lstmdecoder->hide[i][d].sig_wh[j]+=ClipGradient(learningrate*lstmdecoder->hide[i][d].sig_transwh[j]);
						lstmdecoder->hide[i][d].tan_wh[j]+=ClipGradient(learningrate*lstmdecoder->hide[i][d].tan_transwh[j]);
						lstmdecoder->hide[i][d].out_wh[j]+=ClipGradient(learningrate*lstmdecoder->hide[i][d].out_transwh[j]);
					}
				}
		}
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].transbia);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].transw[j]);
		}
		return;
	}
	else if(__Typename=="gru")
	{
		
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		exit(0);
	}
}

void DeepSeq2Seq::ErrorCalc(const int DT)
{
	error=0;
	double trans;
	for(int t=1;t<=DT;t++)
		for(int i=0;i<ONUM;i++)
		{
			trans=expect[i][t]-output[i].out[t];
			error+=trans*trans;
		}
	error*=0.5;
	return;
}

void DeepSeq2Seq::SetFunction(const char* function_name)
{
	func_name=function_name;
}

void DeepSeq2Seq::Datain(const char *__Typename,const char *EncoderFile,const char *DecoderFile,const char *OutputFile)
{
	if(__Typename=="rnn")
	{
		rnnencoder->Datain(EncoderFile);
		rnndecoder->Datain(DecoderFile);
	}
	else if(__Typename=="lstm")
	{
		lstmencoder->Datain(EncoderFile);
		lstmdecoder->Datain(DecoderFile);
	}
	else if(__Typename=="gru")
	{
		gruencoder->Datain(EncoderFile);
		grudecoder->Datain(DecoderFile);
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		system("pause");
		exit(0);
	}
	ifstream fin(OutputFile);
	for(int i=0;i<ONUM;i++)
	{
		fin>>output[i].bia;
		for(int j=0;j<HNUM;j++)
			fin>>output[i].w[j];
	}
	fin.close();
}

void DeepSeq2Seq::Dataout(const char *__Typename,const char *EncoderFile,const char *DecoderFile,const char *OutputFile)
{
	if(__Typename=="rnn")
	{
		rnnencoder->Dataout(EncoderFile);
		rnndecoder->Dataout(DecoderFile);
	}
	else if(__Typename=="lstm")
	{
		lstmencoder->Dataout(EncoderFile);
		lstmdecoder->Dataout(DecoderFile);
	}
	else if(__Typename=="gru")
	{
		gruencoder->Dataout(EncoderFile);
		grudecoder->Dataout(DecoderFile);
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		system("pause");
		exit(0);
	}
	ofstream fout(OutputFile);
	for(int i=0;i<ONUM;i++)
	{
		fout<<output[i].bia<<endl;
		for(int j=0;j<HNUM;j++)
			fout<<output[i].w[j]<<endl;
	}
	fout.close();
}

#endif
