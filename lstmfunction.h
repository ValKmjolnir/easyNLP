/*lstmfunction.h header file made by ValK*/
/*2019/3/24                   version 0.1*/
#ifndef __LSTMFUNCTION_H__
#define __LSTMFUNCTION_H__

#include "lstm.h"

#include<iostream>
#include<ctime>
#include<fstream>
#include<cstdlib>
using namespace std;

NormalLSTM::NormalLSTM(int InputlayerNum,int HiddenlayerNum,int Maxtime)
{
	MAXTIME=Maxtime;
	INUM=InputlayerNum;
	HNUM=HiddenlayerNum;
	hide=new lstm_neuron[HNUM];
	for(int i=0;i<HNUM;i++)
	{
		hide[i].cell=new double[MAXTIME];
		hide[i].out=new double[MAXTIME];

		hide[i].fog_in=new double[MAXTIME];
		hide[i].sig_in=new double[MAXTIME];
		hide[i].tan_in=new double[MAXTIME];
		hide[i].out_in=new double[MAXTIME];

		hide[i].fog_out=new double[MAXTIME];
		hide[i].sig_out=new double[MAXTIME];
		hide[i].tan_out=new double[MAXTIME];
		hide[i].out_out=new double[MAXTIME];

		hide[i].fog_diff=new double[MAXTIME];
		hide[i].sig_diff=new double[MAXTIME];
		hide[i].tan_diff=new double[MAXTIME];
		hide[i].out_diff=new double[MAXTIME];

		hide[i].fog_wi=new double[INUM];
		hide[i].sig_wi=new double[INUM];
		hide[i].tan_wi=new double[INUM];
		hide[i].out_wi=new double[INUM];

		hide[i].fog_transwi=new double[INUM];
		hide[i].sig_transwi=new double[INUM];
		hide[i].tan_transwi=new double[INUM];
		hide[i].out_transwi=new double[INUM];

		hide[i].fog_wh=new double[HNUM];
		hide[i].sig_wh=new double[HNUM];
		hide[i].tan_wh=new double[HNUM];
		hide[i].out_wh=new double[HNUM];

		hide[i].fog_transwh=new double[HNUM];
		hide[i].sig_transwh=new double[HNUM];
		hide[i].tan_transwh=new double[HNUM];
		hide[i].out_transwh=new double[HNUM];
	}
}

NormalLSTM::~NormalLSTM()
{
	for(int i=0;i<HNUM;i++)
	{
		delete []hide[i].cell;
		delete []hide[i].out;

		delete []hide[i].fog_in;
		delete []hide[i].sig_in;
		delete []hide[i].tan_in;
		delete []hide[i].out_in;

		delete []hide[i].fog_out;
		delete []hide[i].sig_out;
		delete []hide[i].tan_out;
		delete []hide[i].out_out;

		delete []hide[i].fog_diff;
		delete []hide[i].sig_diff;
		delete []hide[i].tan_diff;
		delete []hide[i].out_diff;

		delete []hide[i].fog_wi;
		delete []hide[i].sig_wi;
		delete []hide[i].tan_wi;
		delete []hide[i].out_wi;

		delete []hide[i].fog_transwi;
		delete []hide[i].sig_transwi;
		delete []hide[i].tan_transwi;
		delete []hide[i].out_transwi;

		delete []hide[i].fog_wh;
		delete []hide[i].sig_wh;
		delete []hide[i].tan_wh;
		delete []hide[i].out_wh;

		delete []hide[i].fog_transwh;
		delete []hide[i].sig_transwh;
		delete []hide[i].tan_transwh;
		delete []hide[i].out_transwh;
	}
    delete []hide;
}

DeepLSTM::DeepLSTM(int InputlayerNum,int HiddenlayerNum,int Depth,int Maxtime)
{
    MAXTIME=Maxtime;
    INUM=InputlayerNum;
    HNUM=HiddenlayerNum;
    DEPTH=Depth-1;

    hlink=new lstm_neuron[HNUM];
    hide=new lstm_neuron*[HNUM];
    for(int i=0;i<HNUM;i++)
        hide[i]=new lstm_neuron[DEPTH];
    for(int i=0;i<HNUM;i++)
	{
		hlink[i].cell=new double[MAXTIME];
		hlink[i].out=new double[MAXTIME];

		hlink[i].fog_in=new double[MAXTIME];
		hlink[i].sig_in=new double[MAXTIME];
		hlink[i].tan_in=new double[MAXTIME];
		hlink[i].out_in=new double[MAXTIME];

		hlink[i].fog_out=new double[MAXTIME];
		hlink[i].sig_out=new double[MAXTIME];
		hlink[i].tan_out=new double[MAXTIME];
		hlink[i].out_out=new double[MAXTIME];

		hlink[i].fog_diff=new double[MAXTIME];
		hlink[i].sig_diff=new double[MAXTIME];
		hlink[i].tan_diff=new double[MAXTIME];
		hlink[i].out_diff=new double[MAXTIME];

		hlink[i].fog_wi=new double[INUM];
		hlink[i].sig_wi=new double[INUM];
		hlink[i].tan_wi=new double[INUM];
		hlink[i].out_wi=new double[INUM];

		hlink[i].fog_transwi=new double[INUM];
		hlink[i].sig_transwi=new double[INUM];
		hlink[i].tan_transwi=new double[INUM];
		hlink[i].out_transwi=new double[INUM];

		hlink[i].fog_wh=new double[HNUM];
		hlink[i].sig_wh=new double[HNUM];
		hlink[i].tan_wh=new double[HNUM];
		hlink[i].out_wh=new double[HNUM];

		hlink[i].fog_transwh=new double[HNUM];
		hlink[i].sig_transwh=new double[HNUM];
		hlink[i].tan_transwh=new double[HNUM];
		hlink[i].out_transwh=new double[HNUM];
	}
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            hide[i][d].cell=new double[MAXTIME];
            hide[i][d].out=new double[MAXTIME];

            hide[i][d].fog_in=new double[MAXTIME];
            hide[i][d].sig_in=new double[MAXTIME];
            hide[i][d].tan_in=new double[MAXTIME];
            hide[i][d].out_in=new double[MAXTIME];

            hide[i][d].fog_out=new double[MAXTIME];
            hide[i][d].sig_out=new double[MAXTIME];
            hide[i][d].tan_out=new double[MAXTIME];
            hide[i][d].out_out=new double[MAXTIME];

            hide[i][d].fog_diff=new double[MAXTIME];
            hide[i][d].sig_diff=new double[MAXTIME];
            hide[i][d].tan_diff=new double[MAXTIME];
            hide[i][d].out_diff=new double[MAXTIME];

            hide[i][d].fog_wi=new double[INUM];
            hide[i][d].sig_wi=new double[INUM];
            hide[i][d].tan_wi=new double[INUM];
            hide[i][d].out_wi=new double[INUM];

            hide[i][d].fog_transwi=new double[INUM];
            hide[i][d].sig_transwi=new double[INUM];
            hide[i][d].tan_transwi=new double[INUM];
            hide[i][d].out_transwi=new double[INUM];

            hide[i][d].fog_wh=new double[HNUM];
            hide[i][d].sig_wh=new double[HNUM];
            hide[i][d].tan_wh=new double[HNUM];
            hide[i][d].out_wh=new double[HNUM];

            hide[i][d].fog_transwh=new double[HNUM];
            hide[i][d].sig_transwh=new double[HNUM];
            hide[i][d].tan_transwh=new double[HNUM];
            hide[i][d].out_transwh=new double[HNUM];
        }
}

DeepLSTM::~DeepLSTM()
{
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            delete []hide[i][d].cell;
            delete []hide[i][d].out;

            delete []hide[i][d].fog_in;
            delete []hide[i][d].sig_in;
            delete []hide[i][d].tan_in;
            delete []hide[i][d].out_in;

            delete []hide[i][d].fog_out;
            delete []hide[i][d].sig_out;
            delete []hide[i][d].tan_out;
            delete []hide[i][d].out_out;

            delete []hide[i][d].fog_diff;
            delete []hide[i][d].sig_diff;
            delete []hide[i][d].tan_diff;
            delete []hide[i][d].out_diff;

            delete []hide[i][d].fog_wi;
            delete []hide[i][d].sig_wi;
            delete []hide[i][d].tan_wi;
            delete []hide[i][d].out_wi;

            delete []hide[i][d].fog_transwi;
            delete []hide[i][d].sig_transwi;
            delete []hide[i][d].tan_transwi;
            delete []hide[i][d].out_transwi;

            delete []hide[i][d].fog_wh;
            delete []hide[i][d].sig_wh;
            delete []hide[i][d].tan_wh;
            delete []hide[i][d].out_wh;

            delete []hide[i][d].fog_transwh;
            delete []hide[i][d].sig_transwh;
            delete []hide[i][d].tan_transwh;
            delete []hide[i][d].out_transwh;
        }
    for(int i=0;i<HNUM;i++)
    {
        delete []hlink[i].cell;
		delete []hlink[i].out;

		delete []hlink[i].fog_in;
		delete []hlink[i].sig_in;
		delete []hlink[i].tan_in;
		delete []hlink[i].out_in;

		delete []hlink[i].fog_out;
		delete []hlink[i].sig_out;
		delete []hlink[i].tan_out;
		delete []hlink[i].out_out;

		delete []hlink[i].fog_diff;
		delete []hlink[i].sig_diff;
		delete []hlink[i].tan_diff;
		delete []hlink[i].out_diff;

		delete []hlink[i].fog_wi;
		delete []hlink[i].sig_wi;
		delete []hlink[i].tan_wi;
		delete []hlink[i].out_wi;

		delete []hlink[i].fog_transwi;
		delete []hlink[i].sig_transwi;
		delete []hlink[i].tan_transwi;
		delete []hlink[i].out_transwi;

		delete []hlink[i].fog_wh;
		delete []hlink[i].sig_wh;
		delete []hlink[i].tan_wh;
		delete []hlink[i].out_wh;

		delete []hlink[i].fog_transwh;
		delete []hlink[i].sig_transwh;
		delete []hlink[i].tan_transwh;
		delete []hlink[i].out_transwh;
    }
    for(int i=0;i<HNUM;i++)
        delete []hide[i];
    delete []hlink;
    delete []hide;
}
#endif