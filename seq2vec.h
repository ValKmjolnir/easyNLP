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
#include "activatefunction.h"
using namespace std;

class Seq2Vec
{
	protected:
		int INUM;
		int HNUM;
		int ONUM;
		int DEPTH;
		int MAXTIME;
		int batch_size;
		double learningrate;
		double **input;
		double *expect;
		double error;
		double maxerror;
		neuron *output;
		string func_name;
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

//All models of NormalSeq2Vec is available
void NormalSeq2Vec::TotalWork(const char *__Typename,const char *EncoderFile,const char *OutputFile,const char *Sequencedata,const char *Trainingdata)
{
	if(!fopen(EncoderFile,"r")||!fopen(OutputFile,"r"))
	{
		Dataout(__Typename,EncoderFile,OutputFile);
		cout<<"easyNLP>>[NormalSeq2Vec] Initializing completed.\n";
	}
	else
		Datain(__Typename,EncoderFile,OutputFile);
	maxerror=1e8;
	string ques;
	char answ;
	int epoch=0;
	while(maxerror>0.1)
	{
		epoch++;
		maxerror=0;
		ifstream fin_seq(Sequencedata);
		ifstream fin_t(Trainingdata);
		if(fin_seq.fail()||fin_t.fail())
		{
			cout<<"easyNLP>>[Error]Cannot open data file!"<<endl;
			cout<<"easyNLP>>[Lack] "<<Sequencedata<<" and "<<Trainingdata<<endl;
			system("pause");
			exit(0);
		}
		for(int b=0;b<batch_size;b++)
		{
			getline(fin_seq,ques);
			fin_t>>answ;
			for(int t=0;t<MAXTIME;t++)
				for(int i=0;i<INUM;i++)
					input[i][t]=0;
			for(int i=0;i<ONUM;i++)
				expect[i]=0;
			for(int t=1;t<=ques.length();t++)
			{
				if(ques[t-1]<='z'&&ques[t-1]>='a')
					input[ques[t-1]-'a'+1][t]=1;
				else
					input[0][t]=1;
			}
			if(answ<='z'&&answ>='a')
				expect[answ-'a'+1]=1;
			else
				expect[0]=1;
			Calc(__Typename,ques.length());
			ErrorCalc();
			Training(__Typename,ques.length());
			maxerror+=error;
		}
		if(epoch%5==0)
		{
			cout<<"easyNLP>>Epoch "<<epoch<<": Error :"<<maxerror<<endl;
			if(epoch%20==0)
				Dataout(__Typename,EncoderFile,OutputFile);
		}
		fin_seq.close();
		fin_t.close();
	}
	cout<<"easyNLP>>Final output in progress..."<<endl;
	Dataout(__Typename,EncoderFile,OutputFile);
	cout<<"easyNLP>>Training complete."<<endl;
	return;
}

NormalSeq2Vec::NormalSeq2Vec(const char* __Typename,int InputlayerNum,int HiddenlayerNum,int OutputlayerNum,int Maxtime)
{
	INUM=InputlayerNum;
	HNUM=HiddenlayerNum;
	ONUM=OutputlayerNum;
	MAXTIME=Maxtime+1;
	rnnencoder=NULL;
	lstmencoder=NULL;
	gruencoder=NULL;
	if(strcmp(__Typename,"rnn")==0)
	{
		rnnencoder=new NormalRNN(INUM,HNUM,MAXTIME);
		rnnencoder->INIT();
	}
	else if(strcmp(__Typename,"lstm")==0)
	{
		lstmencoder=new NormalLSTM(INUM,HNUM,MAXTIME);
		lstmencoder->INIT();
	}
	else if(strcmp(__Typename,"gru")==0)
	{
		gruencoder=new NormalGRU(INUM,HNUM,MAXTIME);
		gruencoder->INIT();
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
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

void NormalSeq2Vec::SetBatchSize(const int __b)
{
	batch_size=__b;
}

void NormalSeq2Vec::SetFunction(const char *FunctionName)
{
	func_name=FunctionName;
}

void NormalSeq2Vec::SetLearningRate(const double __lr)
{
	learningrate=__lr;
}

void NormalSeq2Vec::Calc(const char* __Typename,const int T)
{
	if(strcmp(__Typename,"rnn")==0)
	{
		for(int t=1;t<T;t++)
			for(int i=0;i<HNUM;i++)
			{
				rnnencoder->hide[i].in[t]=rnnencoder->hide[i].bia;
				for(int j=0;j<INUM;j++)
					rnnencoder->hide[i].in[t]+=rnnencoder->hide[i].wi[j]*input[j][t];
				for(int j=0;j<HNUM;j++)
					rnnencoder->hide[i].in[t]+=rnnencoder->hide[i].wh[j]*rnnencoder->hide[j].out[t-1];
				rnnencoder->hide[i].out[t]=tanh(rnnencoder->hide[i].in[t]);
			}
		double softmax=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in+=output[i].w[j]*rnnencoder->hide[j].out[T];
			softmax+=exp(output[i].in);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out=exp(output[i].in)/softmax;
		return;
	}
	else if(strcmp(__Typename,"lstm")==0)
	{
		for(int t=1;t<T;t++)
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
		double softmax=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in+=output[i].w[j]*lstmencoder->hide[j].out[T];
			softmax+=exp(output[i].in);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out=exp(output[i].in)/softmax;
		return;
	}
	else if(strcmp(__Typename,"gru")==0)
	{
		double softmax_max;
		for(int t=1;t<=T;t++)
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
		softmax_max=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in+=output[i].w[j]*gruencoder->hide[j].out[T];
			softmax_max+=exp(output[i].in);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out=exp(output[i].in)/softmax_max;
		return;
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		exit(0);
	}
}

void NormalSeq2Vec::Training(const char* __Typename,const int T)
{
	if(strcmp(__Typename,"rnn")==0)
	{
		double trans;
		for(int i=0;i<ONUM;i++)
			output[i].diff=(expect[i]-output[i].out)*output[i].out*(1-output[i].out);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff*output[j].w[i];
			rnnencoder->hide[i].diff[T]=trans*difftanh(rnnencoder->hide[i].in[T]);
		}
		for(int t=T-1;t>=1;t--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=rnnencoder->hide[j].diff[t+1]*rnnencoder->hide[j].wh[i];
				rnnencoder->hide[i].diff[t]=trans*tanh(rnnencoder->hide[i].in[t]);
			}
		for(int i=0;i<HNUM;i++)
		{
			rnnencoder->hide[i].transbia=0;
			for(int j=0;j<INUM;j++)
				rnnencoder->hide[i].transwi[j]=0;
			for(int j=0;j<HNUM;j++)
				rnnencoder->hide[i].transwh[j]=0;
		}
		for(int t=1;t<=T;t++)
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
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].diff);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].diff*rnnencoder->hide[j].out[T]);
		}
		return;
	}
	else if(strcmp(__Typename,"lstm")==0)
	{
		double trans;
		for(int i=0;i<ONUM;i++)
			output[i].diff=(expect[i]-output[i].out)*output[i].out*(1-output[i].out);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff*output[j].w[i];
			lstmencoder->hide[i].fog_diff[T]=trans*lstmencoder->hide[i].out_out[T]*difftanh(lstmencoder->hide[i].cell[T])*lstmencoder->hide[i].cell[T-1]*diffsigmoid(lstmencoder->hide[i].fog_in[T]);
			lstmencoder->hide[i].sig_diff[T]=trans*lstmencoder->hide[i].out_out[T]*difftanh(lstmencoder->hide[i].cell[T])*lstmencoder->hide[i].tan_out[T]*diffsigmoid(lstmencoder->hide[i].sig_in[T]);
			lstmencoder->hide[i].tan_diff[T]=trans*lstmencoder->hide[i].out_out[T]*difftanh(lstmencoder->hide[i].cell[T])*lstmencoder->hide[i].sig_out[T]*difftanh(lstmencoder->hide[i].tan_in[T]);
			lstmencoder->hide[i].out_diff[T]=trans*tanh(lstmencoder->hide[i].cell[T])*diffsigmoid(lstmencoder->hide[i].out_in[T]);
		}
		for(int t=T-1;t>=1;t--)
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
		for(int t=1;t<=T;t++)
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
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].diff);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].diff*lstmencoder->hide[j].out[T]);
		}
		return;
	}
	else if(strcmp(__Typename,"gru")==0)
	{
		double trans;
		for(int t=0;t<=T;t++)
			for(int i=0;i<ONUM;i++)
				output[i].diff=expect[i]-output[i].out;
		for(int i=0;i<ONUM;i++)
			output[i].diff*=output[i].out*(1-output[i].out);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
					trans+=output[j].diff*output[j].w[i];
			gruencoder->hide[i].sig_update_diff[T]=trans*(1-gruencoder->hide[i].sig_replace_out[T])*difftanh(gruencoder->hide[i].tan_replace_in[T])*gruencoder->hide[i].tan_replace_wh[i]*gruencoder->hide[i].out[T-1]*diffsigmoid(gruencoder->hide[i].sig_update_in[T]);
			gruencoder->hide[i].sig_replace_diff[T]=trans*(gruencoder->hide[i].out[T-1]-gruencoder->hide[i].tan_replace_out[T])*diffsigmoid(gruencoder->hide[i].sig_replace_in[T]);
			gruencoder->hide[i].tan_replace_diff[T]=trans*(1-gruencoder->hide[i].sig_replace_out[T])*difftanh(gruencoder->hide[i].tan_replace_in[T]);
		}
		for(int t=T-1;t>=1;t--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=gruencoder->hide[j].sig_update_diff[t+1]*gruencoder->hide[j].sig_update_wh[i]+gruencoder->hide[j].sig_replace_diff[t+1]*gruencoder->hide[j].sig_replace_wh[i]+gruencoder->hide[j].tan_replace_diff[t+1]*gruencoder->hide[j].tan_replace_wh[i]*gruencoder->hide[i].sig_replace_out[t+1];
				gruencoder->hide[i].sig_update_diff[t]=trans*(1-gruencoder->hide[i].sig_replace_out[t])*difftanh(gruencoder->hide[i].tan_replace_in[t])*gruencoder->hide[i].tan_replace_wh[i]*gruencoder->hide[i].out[t-1]*diffsigmoid(gruencoder->hide[i].sig_update_in[t]);
				gruencoder->hide[i].sig_replace_diff[t]=trans*(gruencoder->hide[i].out[t-1]-gruencoder->hide[i].tan_replace_out[t])*diffsigmoid(gruencoder->hide[i].sig_replace_in[t]);
				gruencoder->hide[i].tan_replace_diff[t]=trans*(1-gruencoder->hide[i].sig_replace_out[t])*difftanh(gruencoder->hide[i].tan_replace_in[t]);
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
		for(int t=1;t<=T;t++)
			for(int i=0;i<HNUM;i++)
			{
				gruencoder->hide[i].sig_update_transbia+=2*gruencoder->hide[i].sig_update_diff[t];
				gruencoder->hide[i].sig_replace_transbia+=2*gruencoder->hide[i].sig_replace_diff[t];
				gruencoder->hide[i].tan_replace_transbia+=2*gruencoder->hide[i].tan_replace_diff[t];
				for(int j=0;j<HNUM;j++)
				{
					gruencoder->hide[i].sig_update_transwh[j]+=gruencoder->hide[i].sig_update_diff[t]*gruencoder->hide[j].out[t-1];//(gruencoder->hide[j].out[t-1]+hiddenstate[j][t]);
					gruencoder->hide[i].sig_replace_transwh[j]+=gruencoder->hide[i].sig_replace_diff[t]*gruencoder->hide[j].sig_replace_out[t]*gruencoder->hide[j].out[t-1];//(gruencoder->hide[j].out[t-1]+hiddenstate[j][t]);
					gruencoder->hide[i].tan_replace_transwh[j]+=gruencoder->hide[i].tan_replace_diff[t]*gruencoder->hide[j].out[t-1];//(gruencoder->hide[j].out[t-1]+hiddenstate[j][t]);
				}
				for(int j=0;j<INUM;j++)
				{
					gruencoder->hide[i].sig_update_transwi[j]+=gruencoder->hide[i].sig_update_diff[t]*input[j][t];
					gruencoder->hide[i].sig_replace_transwi[j]+=gruencoder->hide[i].sig_replace_diff[t]*input[j][t];
					gruencoder->hide[i].tan_replace_transwi[j]+=gruencoder->hide[i].tan_replace_diff[t]*input[j][t];
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
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].diff);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].diff*gruencoder->hide[j].out[T]);
		}
		return;
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
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

void NormalSeq2Vec::Datain(const char *__Typename,const char *EncoderFile,const char *OutputFile)
{
	if(strcmp(__Typename,"rnn")==0)
		rnnencoder->Datain(EncoderFile);
	else if(strcmp(__Typename,"lstm")==0)
		lstmencoder->Datain(EncoderFile);
	else if(strcmp(__Typename,"gru")==0)
		gruencoder->Datain(EncoderFile);
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

void NormalSeq2Vec::Dataout(const char *__Typename,const char *EncoderFile,const char *OutputFile)
{
	if(strcmp(__Typename,"rnn")==0)
		rnnencoder->Dataout(EncoderFile);
	else if(strcmp(__Typename,"lstm")==0)
		lstmencoder->Dataout(EncoderFile);
	else if(strcmp(__Typename,"gru")==0)
		gruencoder->Dataout(EncoderFile);
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

//Rnn is available
void DeepSeq2Vec::TotalWork(const char *__Typename,const char *EncoderFile,const char *OutputFile,const char *Sequencedata,const char *Trainingdata)
{
	if(!fopen(EncoderFile,"r")||!fopen(OutputFile,"r"))
	{
		Dataout(__Typename,EncoderFile,OutputFile);
		cout<<"easyNLP>>[DeepSeq2Vec] Initializing completed.\n";
	}
	else
		Datain(__Typename,EncoderFile,OutputFile);
	maxerror=1e8;
	string ques;
	char answ;
	int epoch=0;
	while(maxerror>0.1)
	{
		epoch++;
		maxerror=0;
		ifstream fin_seq(Sequencedata);
		ifstream fin_t(Trainingdata);
		if(fin_seq.fail()||fin_t.fail())
		{
			cout<<"easyNLP>>[Error]Cannot open data file!"<<endl;
			cout<<"easyNLP>>[Lack] "<<Sequencedata<<" and "<<Trainingdata<<endl;
			system("pause");
			exit(0);
		}
		for(int b=0;b<batch_size;b++)
		{
			getline(fin_seq,ques);
			fin_t>>answ;
			for(int t=0;t<MAXTIME;t++)
				for(int i=0;i<INUM;i++)
					input[i][t]=0;
			for(int i=0;i<ONUM;i++)
				expect[i]=0;
			for(int t=1;t<=ques.length();t++)
			{
				if(ques[t-1]<='z'&&ques[t-1]>='a')
					input[ques[t-1]-'a'+1][t]=1;
				else
					input[0][t]=1;
			}
			if(answ<='z'&&answ>='a')
				expect[answ-'a'+1]=1;
			else
				expect[0]=1;
			Calc(__Typename,ques.length());
			ErrorCalc();
			Training(__Typename,ques.length());
			maxerror+=error;
		}
		if(epoch%5==0)
		{
			cout<<"easyNLP>>Epoch "<<epoch<<": Error :"<<maxerror<<endl;
			if(epoch%20==0)
				Dataout(__Typename,EncoderFile,OutputFile);
		}
		fin_seq.close();
		fin_t.close();
	}
	cout<<"easyNLP>>Final output in progress..."<<endl;
	Dataout(__Typename,EncoderFile,OutputFile);
	cout<<"easyNLP>>Training complete."<<endl;
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
	if(strcmp(__Typename,"rnn")==0)
	{
		rnnencoder=new DeepRNN(INUM,HNUM,DEPTH+1,MAXTIME);
		rnnencoder->INIT();
	}
	else if(strcmp(__Typename,"lstm")==0)
	{
		lstmencoder=new DeepLSTM(INUM,HNUM,DEPTH+1,MAXTIME);
		lstmencoder->INIT();
	}
	else if(strcmp(__Typename,"gru")==0)
	{
		gruencoder=new DeepGRU(INUM,HNUM,DEPTH+1,MAXTIME);
		gruencoder->INIT();
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
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

void DeepSeq2Vec::SetBatchSize(const int __b)
{
	batch_size=__b;
}

void DeepSeq2Vec::SetFunction(const char *FunctionName)
{
	func_name=FunctionName;
}

void DeepSeq2Vec::SetLearningRate(const double __lr)
{
	learningrate=__lr;
}

void DeepSeq2Vec::Calc(const char* __Typename,const int T)
{
	if(strcmp(__Typename,"rnn")==0)
	{
		for(int t=1;t<T;t++)
		{
			for(int i=0;i<HNUM;i++)
			{
				rnnencoder->hlink[i].in[t]=rnnencoder->hlink[i].bia;
				for(int j=0;j<INUM;j++)
					rnnencoder->hlink[i].in[t]+=rnnencoder->hlink[i].wi[j]*input[j][t];
				for(int j=0;j<HNUM;j++)
					rnnencoder->hlink[i].in[t]+=rnnencoder->hlink[i].wh[j]*rnnencoder->hlink[j].out[t-1];
				rnnencoder->hlink[i].out[t]=tanh(rnnencoder->hlink[i].in[t]);
			}
			for(int d=0;d<DEPTH;d++)
				for(int i=0;i<HNUM;i++)
				{
					rnnencoder->hide[i][d].in[t]=rnnencoder->hide[i][d].bia;
					for(int j=0;j<INUM;j++)
						rnnencoder->hide[i][d].in[t]+=rnnencoder->hide[i][d].wi[j]*(d==0? rnnencoder->hlink[j].out[t]:rnnencoder->hide[j][d-1].out[t]);
					for(int j=0;j<HNUM;j++)
						rnnencoder->hide[i][d].in[t]+=rnnencoder->hide[i][d].wh[j]*rnnencoder->hide[j][d].out[t-1];
					rnnencoder->hide[i][d].out[t]=tanh(rnnencoder->hide[i][d].in[t]);
				}
		}
		double softmax=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in+=output[i].w[j]*rnnencoder->hide[j][DEPTH-1].out[T];
			softmax+=exp(output[i].in);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out=exp(output[i].in)/softmax;
		return;
	}
	else if(strcmp(__Typename,"lstm")==0)
	{
		for(int t=1;t<T;t++)
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
					for(int j=0;j<INUM;j++)
					{
						lstmencoder->hide[i][d].fog_in[t]+=lstmencoder->hide[i][d].fog_wi[j]*(d==0? lstmencoder->hlink[j].out[t]:lstmencoder->hide[j][d-1].out[t]);
						lstmencoder->hide[i][d].sig_in[t]+=lstmencoder->hide[i][d].sig_wi[j]*(d==0? lstmencoder->hlink[j].out[t]:lstmencoder->hide[j][d-1].out[t]);
						lstmencoder->hide[i][d].tan_in[t]+=lstmencoder->hide[i][d].tan_wi[j]*(d==0? lstmencoder->hlink[j].out[t]:lstmencoder->hide[j][d-1].out[t]);
						lstmencoder->hide[i][d].out_in[t]+=lstmencoder->hide[i][d].out_wi[j]*(d==0? lstmencoder->hlink[j].out[t]:lstmencoder->hide[j][d-1].out[t]);
					}
					for(int j=0;j<HNUM;j++)
					{
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
		double softmax=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in+=output[i].w[j]*lstmencoder->hide[j][DEPTH-1].out[T];
			softmax+=exp(output[i].in);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out=exp(output[i].in)/softmax;
		return;
	}
	else if(strcmp(__Typename,"gru")==0)
	{
		double softmax;
		for(int t=1;t<=T;t++)
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
						gruencoder->hide[i][d].sig_update_in[t]+=gruencoder->hide[i][d].sig_update_wi[j]*(d==0? gruencoder->hlink[j].out[t]:gruencoder->hide[j][d-1].out[t]);
						gruencoder->hide[i][d].sig_replace_in[t]+=gruencoder->hide[i][d].sig_replace_wi[j]*(d==0? gruencoder->hlink[j].out[t]:gruencoder->hide[j][d-1].out[t]);
						gruencoder->hide[i][d].tan_replace_in[t]+=gruencoder->hide[i][d].tan_replace_wi[j]*(d==0? gruencoder->hlink[j].out[t]:gruencoder->hide[j][d-1].out[t]);
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
		softmax=0;
		for(int i=0;i<ONUM;i++)
		{
			output[i].in=output[i].bia;
			for(int j=0;j<HNUM;j++)
				output[i].in+=output[i].w[j]*gruencoder->hide[j][DEPTH-1].out[T];
			softmax+=exp(output[i].in);
		}
		for(int i=0;i<ONUM;i++)
			output[i].out=exp(output[i].in)/softmax;
		return;
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
		exit(0);
	}
}

void DeepSeq2Vec::Training(const char* __Typename,const int T)
{
	if(strcmp(__Typename,"rnn")==0)
	{
		double trans;
		for(int t=0;t<=T;t++)
			for(int i=0;i<ONUM;i++)
				output[i].diff=expect[i]-output[i].out;
		for(int i=0;i<ONUM;i++)
			output[i].diff*=output[i].out*(1-output[i].out);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff*output[j].w[i];
			rnnencoder->hide[i][DEPTH-1].diff[T]=trans*difftanh(rnnencoder->hide[i][DEPTH-1].in[T]);
		}
		for(int d=DEPTH-2;d>=0;d--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=rnnencoder->hide[j][d+1].diff[T]*rnnencoder->hide[j][d+1].wi[i];
				rnnencoder->hide[i][d].diff[T]=trans*difftanh(rnnencoder->hide[i][d].in[T]);
			}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<HNUM;j++)
				trans+=rnnencoder->hide[j][0].diff[T]*rnnencoder->hide[j][0].wi[i];
			rnnencoder->hlink[i].diff[T]=trans*difftanh(rnnencoder->hlink[i].in[T]);
		}
		for(int t=T-1;t>=1;t--)
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
		for(int t=1;t<=T;t++)
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
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].diff);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].diff*rnnencoder->hide[j][DEPTH-1].out[T]);
		}
		return;
	}
	else if(strcmp(__Typename,"lstm")==0)
	{
		double trans;
		for(int t=0;t<=T;t++)
			for(int i=0;i<ONUM;i++)
				output[i].diff=expect[i]-output[i].out;
		for(int i=0;i<ONUM;i++)
			output[i].diff*=output[i].out*(1-output[i].out);
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<ONUM;j++)
				trans+=output[j].diff*output[j].w[i];
			lstmencoder->hide[i][DEPTH-1].fog_diff[T]=trans*lstmencoder->hide[i][DEPTH-1].out_out[T]*difftanh(lstmencoder->hide[i][DEPTH-1].cell[T])*lstmencoder->hide[i][DEPTH-1].cell[T-1]*diffsigmoid(lstmencoder->hide[i][DEPTH-1].fog_in[T]);
			lstmencoder->hide[i][DEPTH-1].sig_diff[T]=trans*lstmencoder->hide[i][DEPTH-1].out_out[T]*difftanh(lstmencoder->hide[i][DEPTH-1].cell[T])*lstmencoder->hide[i][DEPTH-1].tan_out[T]*diffsigmoid(lstmencoder->hide[i][DEPTH-1].sig_in[T]);
			lstmencoder->hide[i][DEPTH-1].tan_diff[T]=trans*lstmencoder->hide[i][DEPTH-1].out_out[T]*difftanh(lstmencoder->hide[i][DEPTH-1].cell[T])*lstmencoder->hide[i][DEPTH-1].sig_out[T]*difftanh(lstmencoder->hide[i][DEPTH-1].tan_in[T]);
			lstmencoder->hide[i][DEPTH-1].out_diff[T]=trans*tanh(lstmencoder->hide[i][DEPTH-1].cell[T])*diffsigmoid(lstmencoder->hide[i][DEPTH-1].out_in[T]);
		}
		for(int d=DEPTH-2;d>=0;d--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=lstmencoder->hide[j][d+1].fog_diff[T]*lstmencoder->hide[j][d+1].fog_wi[i]+lstmencoder->hide[j][d+1].sig_diff[T]*lstmencoder->hide[j][d+1].sig_wi[i]+lstmencoder->hide[j][d+1].tan_diff[T]*lstmencoder->hide[j][d+1].tan_wi[i]+lstmencoder->hide[j][d+1].out_diff[T]*lstmencoder->hide[j][d+1].out_wi[i];
				lstmencoder->hide[i][d].fog_diff[T]=trans*lstmencoder->hide[i][d].out_out[T]*difftanh(lstmencoder->hide[i][d].cell[T])*lstmencoder->hide[i][d].cell[T-1]*diffsigmoid(lstmencoder->hide[i][d].fog_in[T]);
				lstmencoder->hide[i][d].sig_diff[T]=trans*lstmencoder->hide[i][d].out_out[T]*difftanh(lstmencoder->hide[i][d].cell[T])*lstmencoder->hide[i][d].tan_out[T]*diffsigmoid(lstmencoder->hide[i][d].sig_in[T]);
				lstmencoder->hide[i][d].tan_diff[T]=trans*lstmencoder->hide[i][d].out_out[T]*difftanh(lstmencoder->hide[i][d].cell[T])*lstmencoder->hide[i][d].sig_out[T]*difftanh(lstmencoder->hide[i][d].tan_in[T]);
				lstmencoder->hide[i][d].out_diff[T]=trans*tanh(lstmencoder->hide[i][d].cell[T])*diffsigmoid(lstmencoder->hide[i][d].out_in[T]);
			}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<HNUM;j++)
				trans+=lstmencoder->hide[j][0].fog_diff[T]*lstmencoder->hide[j][0].fog_wi[i]+lstmencoder->hide[j][0].sig_diff[T]*lstmencoder->hide[j][0].sig_wi[i]+lstmencoder->hide[j][0].tan_diff[T]*lstmencoder->hide[j][0].tan_wi[i]+lstmencoder->hide[j][0].out_diff[T]*lstmencoder->hide[j][0].out_wi[i];
			lstmencoder->hlink[i].fog_diff[T]=trans*lstmencoder->hlink[i].out_out[T]*difftanh(lstmencoder->hlink[i].cell[T])*lstmencoder->hlink[i].cell[T-1]*diffsigmoid(lstmencoder->hlink[i].fog_in[T]);
			lstmencoder->hlink[i].sig_diff[T]=trans*lstmencoder->hlink[i].out_out[T]*difftanh(lstmencoder->hlink[i].cell[T])*lstmencoder->hlink[i].tan_out[T]*diffsigmoid(lstmencoder->hlink[i].sig_in[T]);
			lstmencoder->hlink[i].tan_diff[T]=trans*lstmencoder->hlink[i].out_out[T]*difftanh(lstmencoder->hlink[i].cell[T])*lstmencoder->hlink[i].sig_out[T]*difftanh(lstmencoder->hlink[i].tan_in[T]);
			lstmencoder->hlink[i].out_diff[T]=trans*tanh(lstmencoder->hlink[i].cell[T])*diffsigmoid(lstmencoder->hlink[i].out_in[T]);
		}
		for(int t=T-1;t>=1;t--)
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
		
		for(int t=1;t<=T;t++)
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
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].diff);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].diff*lstmencoder->hide[j][DEPTH-1].out[T]);
		}
		return;
	}
	else if(strcmp(__Typename,"gru")==0)
	{
		double trans;
		for(int t=0;t<=T;t++)
			for(int i=0;i<ONUM;i++)
				output[i].diff=expect[i]-output[i].out;
		for(int i=0;i<ONUM;i++)
			output[i].diff*=output[i].out*(1-output[i].out);
		for(int d=DEPTH-1;d>=0;d--)
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				if(d==DEPTH-1)
					for(int j=0;j<ONUM;j++)
						trans+=output[j].diff*output[j].w[i];
				else
					for(int j=0;j<HNUM;j++)
						trans+=gruencoder->hide[j][d+1].sig_update_diff[T]*gruencoder->hide[j][d+1].sig_update_wi[i]+gruencoder->hide[j][d+1].sig_replace_diff[T]*gruencoder->hide[j][d+1].sig_replace_wi[i]+gruencoder->hide[j][d+1].tan_replace_diff[T]*gruencoder->hide[j][d+1].tan_replace_wi[i];
				gruencoder->hide[i][d].sig_update_diff[T]=trans*(1-gruencoder->hide[i][d].sig_replace_out[T])*difftanh(gruencoder->hide[i][d].tan_replace_in[T])*gruencoder->hide[i][d].tan_replace_wh[i]*gruencoder->hide[i][d].out[T-1]*diffsigmoid(gruencoder->hide[i][d].sig_update_in[T]);
				gruencoder->hide[i][d].sig_replace_diff[T]=trans*(gruencoder->hide[i][d].out[T-1]-gruencoder->hide[i][d].tan_replace_out[T])*diffsigmoid(gruencoder->hide[i][d].sig_replace_in[T]);
				gruencoder->hide[i][d].tan_replace_diff[T]=trans*(1-gruencoder->hide[i][d].sig_replace_out[T])*difftanh(gruencoder->hide[i][d].tan_replace_in[T]);
			}
		for(int i=0;i<HNUM;i++)
		{
			trans=0;
			for(int j=0;j<HNUM;j++)
				trans+=gruencoder->hide[j][0].sig_update_diff[T]*gruencoder->hide[j][0].sig_update_wi[i]+gruencoder->hide[j][0].sig_replace_diff[T]*gruencoder->hide[j][0].sig_replace_wi[i]+gruencoder->hide[j][0].tan_replace_diff[T]*gruencoder->hide[j][0].tan_replace_wi[i];
			gruencoder->hlink[i].sig_update_diff[T]=trans*(1-gruencoder->hlink[i].sig_replace_out[T])*difftanh(gruencoder->hlink[i].tan_replace_in[T])*gruencoder->hlink[i].tan_replace_wh[i]*gruencoder->hlink[i].out[T-1]*diffsigmoid(gruencoder->hlink[i].sig_update_in[T]);
			gruencoder->hlink[i].sig_replace_diff[T]=trans*(gruencoder->hlink[i].out[T-1]-gruencoder->hlink[i].tan_replace_out[T])*diffsigmoid(gruencoder->hlink[i].sig_replace_in[T]);
			gruencoder->hlink[i].tan_replace_diff[T]=trans*(1-gruencoder->hlink[i].sig_replace_out[T])*difftanh(gruencoder->hlink[i].tan_replace_in[T]);
		}
		for(int t=T-1;t>=1;t--)
		{
			for(int d=DEPTH-1;d>=0;d--)
				for(int i=0;i<HNUM;i++)
				{
					trans=0;
					if(d==DEPTH-1)
						;
					else
						for(int j=0;j<HNUM;j++)
							trans+=gruencoder->hide[j][d+1].sig_update_diff[t]*gruencoder->hide[j][d+1].sig_update_wi[i]+gruencoder->hide[j][d+1].sig_replace_diff[t]*gruencoder->hide[j][d+1].sig_replace_wi[i]+gruencoder->hide[j][d+1].tan_replace_diff[t]*gruencoder->hide[j][d+1].tan_replace_wi[i];
					gruencoder->hide[i][d].sig_update_diff[t]=trans*(1-gruencoder->hide[i][d].sig_replace_out[t])*difftanh(gruencoder->hide[i][d].tan_replace_in[t])*gruencoder->hide[i][d].tan_replace_wh[i]*gruencoder->hide[i][d].out[t-1]*diffsigmoid(gruencoder->hide[i][d].sig_update_in[t]);
					gruencoder->hide[i][d].sig_replace_diff[t]=trans*(gruencoder->hide[i][d].out[t-1]-gruencoder->hide[i][d].tan_replace_out[t])*diffsigmoid(gruencoder->hide[i][d].sig_replace_in[t]);
					gruencoder->hide[i][d].tan_replace_diff[t]=trans*(1-gruencoder->hide[i][d].sig_replace_out[t])*difftanh(gruencoder->hide[i][d].tan_replace_in[t]);
				}
			for(int i=0;i<HNUM;i++)
			{
				trans=0;
				for(int j=0;j<HNUM;j++)
					trans+=gruencoder->hide[j][0].sig_update_diff[t]*gruencoder->hide[j][0].sig_update_wi[i]+gruencoder->hide[j][0].sig_replace_diff[t]*gruencoder->hide[j][0].sig_replace_wi[i]+gruencoder->hide[j][0].tan_replace_diff[t]*gruencoder->hide[j][0].tan_replace_wi[i];
				for(int j=0;j<HNUM;j++)
					trans+=gruencoder->hlink[j].sig_update_diff[t+1]*gruencoder->hlink[j].sig_update_wh[i]+gruencoder->hlink[j].sig_replace_diff[t+1]*gruencoder->hlink[j].sig_replace_wh[i]+gruencoder->hlink[j].tan_replace_diff[t+1]*gruencoder->hlink[j].tan_replace_wh[i]*gruencoder->hlink[i].sig_replace_out[t+1];
	
				gruencoder->hlink[i].sig_update_diff[t]=trans*(1-gruencoder->hlink[i].sig_replace_out[t])*difftanh(gruencoder->hlink[i].tan_replace_in[t])*gruencoder->hlink[i].tan_replace_wh[i]*gruencoder->hlink[i].out[t-1]*diffsigmoid(gruencoder->hlink[i].sig_update_in[t]);
				gruencoder->hlink[i].sig_replace_diff[t]=trans*(gruencoder->hlink[i].out[t-1]-gruencoder->hlink[i].tan_replace_out[t])*diffsigmoid(gruencoder->hlink[i].sig_replace_in[t]);
				gruencoder->hlink[i].tan_replace_diff[t]=trans*(1-gruencoder->hlink[i].sig_replace_out[t])*difftanh(gruencoder->hlink[i].tan_replace_in[t]);
			}
		}
		
		for(int d=0;d<DEPTH;d++)
			for(int i=0;i<HNUM;i++)
			{
				gruencoder->hide[i][d].sig_update_transbia=0;
				gruencoder->hide[i][d].sig_replace_transbia=0;
				gruencoder->hide[i][d].tan_replace_transbia=0;
				for(int j=0;j<INUM;j++)
				{
					gruencoder->hide[i][d].sig_update_transwi[j]=0;
					gruencoder->hide[i][d].sig_replace_transwi[j]=0;
					gruencoder->hide[i][d].tan_replace_transwi[j]=0;
				}
				for(int j=0;j<HNUM;j++)
				{
					gruencoder->hide[i][d].sig_update_transwh[j]=0;
					gruencoder->hide[i][d].sig_replace_transwh[j]=0;
					gruencoder->hide[i][d].tan_replace_transwh[j]=0;
				}
			}
		for(int i=0;i<HNUM;i++)
		{
			gruencoder->hlink[i].sig_update_transbia=0;
			gruencoder->hlink[i].sig_replace_transbia=0;
			gruencoder->hlink[i].tan_replace_transbia=0;
			for(int j=0;j<INUM;j++)
			{
				gruencoder->hlink[i].sig_update_transwi[j]=0;
				gruencoder->hlink[i].sig_replace_transwi[j]=0;
				gruencoder->hlink[i].tan_replace_transwi[j]=0;
			}
			for(int j=0;j<HNUM;j++)
			{
				gruencoder->hlink[i].sig_update_transwh[j]=0;
				gruencoder->hlink[i].sig_replace_transwh[j]=0;
				gruencoder->hlink[i].tan_replace_transwh[j]=0;
			}
		}
		for(int t=1;t<=T;t++)
		{
			for(int d=0;d<DEPTH;d++)
				for(int i=0;i<HNUM;i++)
				{
					gruencoder->hide[i][d].sig_update_transbia+=2*gruencoder->hide[i][d].sig_update_diff[t];
					gruencoder->hide[i][d].sig_replace_transbia+=2*gruencoder->hide[i][d].sig_replace_diff[t];
					gruencoder->hide[i][d].tan_replace_transbia+=2*gruencoder->hide[i][d].tan_replace_diff[t];
					for(int j=0;j<HNUM;j++)
					{
						gruencoder->hide[i][d].sig_update_transwh[j]+=gruencoder->hide[i][d].sig_update_diff[t]*gruencoder->hide[j][d].out[t-1];
						gruencoder->hide[i][d].sig_replace_transwh[j]+=gruencoder->hide[i][d].sig_replace_diff[t]*gruencoder->hide[j][d].sig_replace_out[t]*gruencoder->hide[j][d].out[t-1];
						gruencoder->hide[i][d].tan_replace_transwh[j]+=gruencoder->hide[i][d].tan_replace_diff[t]*gruencoder->hide[j][d].out[t-1];
					}
					for(int j=0;j<INUM;j++)
					{
						gruencoder->hide[i][d].sig_update_transwi[j]+=gruencoder->hide[i][d].sig_update_diff[t]*(d==0? gruencoder->hlink[j].out[t]:gruencoder->hide[j][d-1].out[t]);
						gruencoder->hide[i][d].sig_replace_transwi[j]+=gruencoder->hide[i][d].sig_replace_diff[t]*(d==0? gruencoder->hlink[j].out[t]:gruencoder->hide[j][d-1].out[t]);
						gruencoder->hide[i][d].tan_replace_transwi[j]+=gruencoder->hide[i][d].tan_replace_diff[t]*(d==0? gruencoder->hlink[j].out[t]:gruencoder->hide[j][d-1].out[t]);
					}
				}
			for(int i=0;i<HNUM;i++)
			{
				gruencoder->hlink[i].sig_update_transbia+=2*gruencoder->hlink[i].sig_update_diff[t];
				gruencoder->hlink[i].sig_replace_transbia+=2*gruencoder->hlink[i].sig_replace_diff[t];
				gruencoder->hlink[i].tan_replace_transbia+=2*gruencoder->hlink[i].tan_replace_diff[t];
				for(int j=0;j<HNUM;j++)
				{
					gruencoder->hlink[i].sig_update_transwh[j]+=gruencoder->hlink[i].sig_update_diff[t]*gruencoder->hlink[j].out[t-1];//(gruencoder->hlink[j].out[t-1]+hiddenstate[j][t]);
					gruencoder->hlink[i].sig_replace_transwh[j]+=gruencoder->hlink[i].sig_replace_diff[t]*gruencoder->hlink[j].sig_replace_out[t]*gruencoder->hlink[j].out[t-1];//(gruencoder->hlink[j].out[t-1]+hiddenstate[j][t]);
					gruencoder->hlink[i].tan_replace_transwh[j]+=gruencoder->hlink[i].tan_replace_diff[t]*gruencoder->hlink[j].out[t-1];//(gruencoder->hlink[j].out[t-1]+hiddenstate[j][t]);
				}
				for(int j=0;j<INUM;j++)
				{
					gruencoder->hlink[i].sig_update_transwi[j]+=gruencoder->hlink[i].sig_update_diff[t]*input[j][t];
					gruencoder->hlink[i].sig_replace_transwi[j]+=gruencoder->hlink[i].sig_replace_diff[t]*input[j][t];
					gruencoder->hlink[i].tan_replace_transwi[j]+=gruencoder->hlink[i].tan_replace_diff[t]*input[j][t];
				}
			}
		}
		for(int d=0;d<DEPTH-1;d++)
			for(int i=0;i<HNUM;i++)
			{
				gruencoder->hide[i][d].sig_update_bia+=ClipGradient(learningrate*gruencoder->hide[i][d].sig_update_transbia);
				gruencoder->hide[i][d].sig_replace_bia+=ClipGradient(learningrate*gruencoder->hide[i][d].sig_replace_transbia);
				gruencoder->hide[i][d].tan_replace_bia+=ClipGradient(learningrate*gruencoder->hide[i][d].tan_replace_transbia);
				for(int j=0;j<HNUM;j++)
				{
					gruencoder->hide[i][d].sig_update_wi[j]+=ClipGradient(learningrate*gruencoder->hide[i][d].sig_update_transwi[j]);
					gruencoder->hide[i][d].sig_replace_wi[j]+=ClipGradient(learningrate*gruencoder->hide[i][d].sig_replace_transwi[j]);
					gruencoder->hide[i][d].tan_replace_wi[j]+=ClipGradient(learningrate*gruencoder->hide[i][d].tan_replace_transwi[j]);
	
					gruencoder->hide[i][d].sig_update_wh[j]+=ClipGradient(learningrate*gruencoder->hide[i][d].sig_update_transwh[j]);
					gruencoder->hide[i][d].sig_replace_wh[j]+=ClipGradient(learningrate*gruencoder->hide[i][d].sig_replace_transwh[j]);
					gruencoder->hide[i][d].tan_replace_wh[j]+=ClipGradient(learningrate*gruencoder->hide[i][d].tan_replace_transwh[j]);
				}
			}
		for(int i=0;i<HNUM;i++)
		{
			gruencoder->hlink[i].sig_update_bia+=ClipGradient(learningrate*gruencoder->hlink[i].sig_update_transbia);
			gruencoder->hlink[i].sig_replace_bia+=ClipGradient(learningrate*gruencoder->hlink[i].sig_replace_transbia);
			gruencoder->hlink[i].tan_replace_bia+=ClipGradient(learningrate*gruencoder->hlink[i].tan_replace_transbia);
			for(int j=0;j<INUM;j++)
			{
				gruencoder->hlink[i].sig_update_wi[j]+=ClipGradient(learningrate*gruencoder->hlink[i].sig_update_transwi[j]);
				gruencoder->hlink[i].sig_replace_wi[j]+=ClipGradient(learningrate*gruencoder->hlink[i].sig_replace_transwi[j]);
				gruencoder->hlink[i].tan_replace_wi[j]+=ClipGradient(learningrate*gruencoder->hlink[i].tan_replace_transwi[j]);
			}
			for(int j=0;j<HNUM;j++)
			{
				gruencoder->hlink[i].sig_update_wh[j]+=ClipGradient(learningrate*gruencoder->hlink[i].sig_update_transwh[j]);
				gruencoder->hlink[i].sig_replace_wh[j]+=ClipGradient(learningrate*gruencoder->hlink[i].sig_replace_transwh[j]);
				gruencoder->hlink[i].tan_replace_wh[j]+=ClipGradient(learningrate*gruencoder->hlink[i].tan_replace_transwh[j]);
			}
		}
		for(int i=0;i<ONUM;i++)
		{
			output[i].bia+=ClipGradient(learningrate*output[i].diff);
			for(int j=0;j<HNUM;j++)
				output[i].w[j]+=ClipGradient(learningrate*output[i].diff*gruencoder->hide[j][DEPTH-1].out[T]);
		}
		return;
	}
	else
	{
		cout<<"easyNLP>>[Error]Unknown neural network name."<<endl;
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

void DeepSeq2Vec::Datain(const char *__Typename,const char *EncoderFile,const char *OutputFile)
{
	if(strcmp(__Typename,"rnn")==0)
		rnnencoder->Datain(EncoderFile);
	else if(strcmp(__Typename,"lstm")==0)
		lstmencoder->Datain(EncoderFile);
	else if(strcmp(__Typename,"gru")==0)
		gruencoder->Datain(EncoderFile);
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

void DeepSeq2Vec::Dataout(const char *__Typename,const char *EncoderFile,const char *OutputFile)
{
	if(strcmp(__Typename,"rnn")==0)
		rnnencoder->Dataout(EncoderFile);
	else if(strcmp(__Typename,"lstm")==0)
		lstmencoder->Dataout(EncoderFile);
	else if(strcmp(__Typename,"gru")==0)
		gruencoder->Dataout(EncoderFile);
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


void Seq2VecDataMaker(const char *Filename,const char *Sequencedata,const char *Trainingdata,const int MAXTIME)
{
	string txt;
	ifstream fin(Filename);
	ofstream fout_seq(Sequencedata);
	ofstream fout_t(Trainingdata);
	if(fin.fail()||fout_seq.fail()||fout_t.fail())
	{
		cout<<"easyNLP>>[Error]Cannot open file."<<endl;
		system("pause");
		exit(0);
	}
	getline(fin,txt);
	int k=0;
	while(txt[k+MAXTIME]!='\0')
	{
		for(int i=0;i<MAXTIME;i++)
		{
			if(txt[i+k]>='A'&&txt[i+k]<='Z')
				txt[i+k]+='a'-'A';
			fout_seq<<txt[i+k];
			if(i==MAXTIME-1)
			{
				if(txt[i+k+1]>='A'&&txt[i+k+1]<='Z')
					txt[i+k+1]+='a'-'A';
				if(txt[i+k+1]==' ')
					txt[i+k+1]='#';
				fout_t<<txt[i+k+1]<<endl;
			}
		}
		fout_seq<<endl;
		k++;
	}
	fin.close();
	fout_seq.close();
	fout_t.close();
	cout<<"easyNLP>>Data making complete."<<endl;
}
#endif
