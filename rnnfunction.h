/*rnnfunction.h header file made by ValK*/
/*2019/3/24                  version 0.1*/

#ifndef __RNNFUNCTION_H__
#define __RNNFUNCTION_H__

#include "rnn.h"

#include<iostream>
#include<ctime>
#include<fstream>
#include<cstdlib>
using namespace std;

NormalRNN::NormalRNN(int InputlayerNum,int HiddenlayerNum,int Maxtime)
{
    MAXTIME=Maxtime;
    INUM=InputlayerNum;
    HNUM=HiddenlayerNum;

    hide=new rnn_neuron[HNUM];
    for(int i=0;i<HNUM;i++)
    {
        hide[i].in=new double[MAXTIME];
        hide[i].out=new double[MAXTIME];
        hide[i].diff=new double[MAXTIME];
        hide[i].wi=new double[INUM];
        hide[i].transwi=new double[INUM];
        hide[i].wh=new double[HNUM];
        hide[i].transwh=new double[HNUM];
    }
}

NormalRNN::~NormalRNN()
{
    for(int i=0;i<HNUM;i++)
    {
        delete []hide[i].in;
        delete []hide[i].out;
        delete []hide[i].diff;
        delete []hide[i].wi;
        delete []hide[i].transwi;
        delete []hide[i].wh;
        delete []hide[i].transwh;
    }
    delete []hide;
}

DeepRNN::DeepRNN(int InputlayerNum,int HiddenlayerNum,int Depth,int Maxtime)
{
    MAXTIME=Maxtime;
    INUM=InputlayerNum;
    HNUM=HiddenlayerNum;
    DEPTH=Depth-1;

    hlink=new rnn_neuron[HNUM];
    hide=new rnn_neuron*[HNUM];
    for(int i=0;i<HNUM;i++)
        hide[i]=new rnn_neuron[DEPTH];
    for(int i=0;i<HNUM;i++)
    {
        hlink[i].in=new double[MAXTIME];
        hlink[i].out=new double[MAXTIME];
        hlink[i].diff=new double[MAXTIME];
        hlink[i].wi=new double[INUM];
        hlink[i].transwi=new double[INUM];
        hlink[i].wh=new double[HNUM];
        hlink[i].transwh=new double[HNUM];
    }
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            hide[i][d].in=new double[MAXTIME];
            hide[i][d].out=new double[MAXTIME];
            hide[i][d].diff=new double[MAXTIME];
            hide[i][d].wi=new double[HNUM];
            hide[i][d].transwi=new double[HNUM];
            hide[i][d].wh=new double[HNUM];
            hide[i][d].transwh=new double[HNUM];
        }
}

DeepRNN::~DeepRNN()
{
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            delete []hide[i][d].in;
            delete []hide[i][d].out;
            delete []hide[i][d].diff;
            delete []hide[i][d].wi;
            delete []hide[i][d].transwi;
            delete []hide[i][d].wh;
            delete []hide[i][d].transwh;
        }
    for(int i=0;i<HNUM;i++)
    {
        delete []hlink[i].in;
        delete []hlink[i].out;
        delete []hlink[i].diff;
        delete []hlink[i].wi;
        delete []hlink[i].transwi;
        delete []hlink[i].wh;
        delete []hlink[i].transwh;
    }
    for(int i=0;i<HNUM;i++)
        delete []hide[i];
    delete []hlink;
    delete []hide;
}

#endif
