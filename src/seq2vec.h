/*seq2vec.h header file by ValK*/
/*2019/5/7           version1.4*/
#ifndef __SEQ2VEC_H__
#define __SEQ2VEC_H__

#include <cstring>
#include <cstdlib>
#include <ctime>
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
    virtual void SetFunction(const std::string&)=0;
    virtual void SetBatchSize(const int)=0;
    virtual void SetLearningRate(const double)=0;
    virtual void Calc(const std::string&,const int)=0;
    virtual void Training(const std::string&,const int)=0;
    virtual void ErrorCalc()=0;
    virtual void Datain(const std::string&,const std::string&,const std::string&)=0;
    virtual void Dataout(const std::string&,const std::string&,const std::string&)=0;
    virtual void TotalWork(const std::string&,const std::string&,const std::string&,const std::string&,const std::string&)=0;
};

class NormalSeq2Vec:public Seq2Vec
{
private:
    NormalRNN *rnnencoder;
    NormalLSTM *lstmencoder;
    NormalGRU *gruencoder;
public:
    NormalSeq2Vec(const std::string&,int,int,int,int);
    ~NormalSeq2Vec();
    void SetFunction(const std::string&);
    void SetLearningRate(const double);
    void SetBatchSize(const int);
    void Calc(const std::string&,const int);
    void Training(const std::string&,const int);
    void ErrorCalc();
    void Datain(const std::string&,const std::string&,const std::string&);
    void Dataout(const std::string&,const std::string&,const std::string&);
    void TotalWork(const std::string&,const std::string&,const std::string&,const std::string&,const std::string&);
};

class DeepSeq2Vec:public Seq2Vec
{
private:
    DeepRNN *rnnencoder;
    DeepLSTM *lstmencoder;
    DeepGRU *gruencoder;
public:
    DeepSeq2Vec(const std::string&,int,int,int,int,int);
    ~DeepSeq2Vec();
    void SetFunction(const std::string&);
    void SetLearningRate(const double);
    void SetBatchSize(const int);
    void Calc(const std::string&,const int);
    void Training(const std::string&,const int);
    void ErrorCalc();
    void Datain(const std::string&,const std::string&,const std::string&);
    void Dataout(const std::string&,const std::string&,const std::string&);
    void TotalWork(const std::string&,const std::string&,const std::string&,const std::string&,const std::string&);
};

void Seq2VecDataMaker(const std::string&,const std::string&,const std::string&,const int);

#endif
