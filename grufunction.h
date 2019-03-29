/*grufunction.h header file made by ValK*/
/*2019/3/29                  version 0.1*/
#ifndef __GRUFUNCTION_H__
#define __GRUFUNCTION_H__
#include "gru.h"
#include<iostream>
#include<fstream>
#include<cmath>
#include<cstdlib>
using namespace std;

NormalGRU::NormalGRU(int InputlayerNum,int HiddenlayerNum,int Maxtime)
{
	INUM=InputlayerNum;
	HNUM=HiddenlayerNum;
	MAXTIME=Maxtime;
	hide=new gru_neuron[HNUM];
	for(int i=0;i<HNUM;i++)
	{
		hide[i].out=new double[Maxtime];
		hide[i].sig_update_in=new double[Maxtime];  hide[i].sig_replace_in=new double[Maxtime];  hide[i].tan_replace_in=new double[Maxtime];
		hide[i].sig_update_out=new double[Maxtime]; hide[i].sig_replace_out=new double[Maxtime]; hide[i].tan_replace_out=new double[Maxtime];
		hide[i].sig_update_diff=new double[Maxtime];hide[i].sig_replace_diff=new double[Maxtime];hide[i].tan_replace_diff=new double[Maxtime];
		hide[i].sig_update_wi=new double[INUM];     hide[i].sig_replace_wi=new double[INUM];     hide[i].tan_replace_wi=new double[INUM];
		hide[i].sig_update_wh=new double[HNUM];     hide[i].sig_replace_wh=new double[HNUM];     hide[i].tan_replace_wh=new double[HNUM];
		hide[i].sig_update_transwi=new double[INUM];hide[i].sig_replace_transwi=new double[INUM];hide[i].tan_replace_transwi=new double[INUM];
		hide[i].sig_update_transwh=new double[HNUM];hide[i].sig_replace_transwh=new double[HNUM];hide[i].tan_replace_transwh=new double[HNUM];
	}
	return;
}
NormalGRU::~NormalGRU()
{
	for(int i=0;i<HNUM;i++)
	{
		delete []hide[i].out;
		delete []hide[i].sig_update_in;     delete []hide[i].sig_replace_in;     delete []hide[i].tan_replace_in;
		delete []hide[i].sig_update_out;    delete []hide[i].sig_replace_out;    delete []hide[i].tan_replace_out;
		delete []hide[i].sig_update_diff;   delete []hide[i].sig_replace_diff;   delete []hide[i].tan_replace_diff;
		delete []hide[i].sig_update_wi;     delete []hide[i].sig_replace_wi;     delete []hide[i].tan_replace_wi;
		delete []hide[i].sig_update_wh;     delete []hide[i].sig_replace_wh;     delete []hide[i].tan_replace_wh;
		delete []hide[i].sig_update_transwi;delete []hide[i].sig_replace_transwi;delete []hide[i].tan_replace_transwi;
		delete []hide[i].sig_update_transwh;delete []hide[i].sig_replace_transwh;delete []hide[i].tan_replace_transwh;
	}
	delete []hide;
	return;
}

DeepGRU::DeepGRU(int InputlayerNum,int HiddenlayerNum,int Depth,int Maxtime)
{
	INUM=InputlayerNum;
	HNUM=HiddenlayerNum;
	DEPTH=Depth-1;
	MAXTIME=Maxtime;
	hlink=new gru_neuron[HNUM];
    hide=new gru_neuron*[HNUM];
    for(int i=0;i<HNUM;i++)
        hide[i]=new gru_neuron[DEPTH];
    for(int i=0;i<HNUM;i++)
	{
		hlink[i].out=new double[Maxtime];
		hlink[i].sig_update_in=new double[Maxtime];  hlink[i].sig_replace_in=new double[Maxtime];  hlink[i].tan_replace_in=new double[Maxtime];
		hlink[i].sig_update_out=new double[Maxtime]; hlink[i].sig_replace_out=new double[Maxtime]; hlink[i].tan_replace_out=new double[Maxtime];
		hlink[i].sig_update_diff=new double[Maxtime];hlink[i].sig_replace_diff=new double[Maxtime];hlink[i].tan_replace_diff=new double[Maxtime];
		hlink[i].sig_update_wi=new double[INUM];     hlink[i].sig_replace_wi=new double[INUM];     hlink[i].tan_replace_wi=new double[INUM];
		hlink[i].sig_update_wh=new double[HNUM];     hlink[i].sig_replace_wh=new double[HNUM];     hlink[i].tan_replace_wh=new double[HNUM];
		hlink[i].sig_update_transwi=new double[INUM];hlink[i].sig_replace_transwi=new double[INUM];hlink[i].tan_replace_transwi=new double[INUM];
		hlink[i].sig_update_transwh=new double[HNUM];hlink[i].sig_replace_transwh=new double[HNUM];hlink[i].tan_replace_transwh=new double[HNUM];
	}
	for(int d=0;d<DEPTH;d++)
		for(int i=0;i<HNUM;i++)
		{
			hide[i][d].out=new double[Maxtime];
			hide[i][d].sig_update_in=new double[Maxtime];  hide[i][d].sig_replace_in=new double[Maxtime];  hide[i][d].tan_replace_in=new double[Maxtime];
			hide[i][d].sig_update_out=new double[Maxtime]; hide[i][d].sig_replace_out=new double[Maxtime]; hide[i][d].tan_replace_out=new double[Maxtime];
			hide[i][d].sig_update_diff=new double[Maxtime];hide[i][d].sig_replace_diff=new double[Maxtime];hide[i][d].tan_replace_diff=new double[Maxtime];
			hide[i][d].sig_update_wi=new double[INUM];     hide[i][d].sig_replace_wi=new double[INUM];     hide[i][d].tan_replace_wi=new double[INUM];
			hide[i][d].sig_update_wh=new double[HNUM];     hide[i][d].sig_replace_wh=new double[HNUM];     hide[i][d].tan_replace_wh=new double[HNUM];
			hide[i][d].sig_update_transwi=new double[INUM];hide[i][d].sig_replace_transwi=new double[INUM];hide[i][d].tan_replace_transwi=new double[INUM];
			hide[i][d].sig_update_transwh=new double[HNUM];hide[i][d].sig_replace_transwh=new double[HNUM];hide[i][d].tan_replace_transwh=new double[HNUM];
		}
}
DeepGRU::~DeepGRU()
{
	for(int d=0;d<DEPTH;d++)
		for(int i=0;i<HNUM;i++)
		{
			delete []hide[i][d].out;
			delete []hide[i][d].sig_update_in;     delete []hide[i][d].sig_replace_in;     delete []hide[i][d].tan_replace_in;
			delete []hide[i][d].sig_update_out;    delete []hide[i][d].sig_replace_out;    delete []hide[i][d].tan_replace_out;
			delete []hide[i][d].sig_update_diff;   delete []hide[i][d].sig_replace_diff;   delete []hide[i][d].tan_replace_diff;
			delete []hide[i][d].sig_update_wi;     delete []hide[i][d].sig_replace_wi;     delete []hide[i][d].tan_replace_wi;
			delete []hide[i][d].sig_update_wh;     delete []hide[i][d].sig_replace_wh;     delete []hide[i][d].tan_replace_wh;
			delete []hide[i][d].sig_update_transwi;delete []hide[i][d].sig_replace_transwi;delete []hide[i][d].tan_replace_transwi;
			delete []hide[i][d].sig_update_transwh;delete []hide[i][d].sig_replace_transwh;delete []hide[i][d].tan_replace_transwh;
		}
	for(int i=0;i<HNUM;i++)
	{
		delete []hlink[i].out;
		delete []hlink[i].sig_update_in;     delete []hlink[i].sig_replace_in;     delete []hlink[i].tan_replace_in;
		delete []hlink[i].sig_update_out;    delete []hlink[i].sig_replace_out;    delete []hlink[i].tan_replace_out;
		delete []hlink[i].sig_update_diff;   delete []hlink[i].sig_replace_diff;   delete []hlink[i].tan_replace_diff;
		delete []hlink[i].sig_update_wi;     delete []hlink[i].sig_replace_wi;     delete []hlink[i].tan_replace_wi;
		delete []hlink[i].sig_update_wh;     delete []hlink[i].sig_replace_wh;     delete []hlink[i].tan_replace_wh;
		delete []hlink[i].sig_update_transwi;delete []hlink[i].sig_replace_transwi;delete []hlink[i].tan_replace_transwi;
		delete []hlink[i].sig_update_transwh;delete []hlink[i].sig_replace_transwh;delete []hlink[i].tan_replace_transwh;
	}
	for(int i=0;i<HNUM;i++)
        delete []hide[i];
    delete []hlink;
    delete []hide;
}
#endif
