/*rnn.h header file made by ValK*/
/*2019/3/24          version 0.1*/
#ifndef __RNN_H__
#define __RNN_H__

#include <iostream>
#include <cstring>

struct rnn_neuron
{
    double *in,*out,bia,*wi,*wh,*diff;
    double transdiff,*transwi,*transwh,transbia;
};

class NormalRNN
{
protected:
    int INUM;
    int HNUM;
    int MAXTIME;
public:
    rnn_neuron *hide;
    NormalRNN(int,int,int);
    ~NormalRNN();
    void Init();
    void Datain(const std::string&);
    void Dataout(const std::string&);
};
class DeepRNN
{
protected:
    int INUM;
    int HNUM;
    int DEPTH;
    int MAXTIME;
public:
    rnn_neuron *hlink;
    rnn_neuron **hide;
    DeepRNN(int,int,int,int);
    ~DeepRNN();
    void Init();
    void Datain(const std::string&);
    void Dataout(const std::string&);
};

#endif
