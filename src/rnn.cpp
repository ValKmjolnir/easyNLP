/*rnnfunction.h header file made by ValK*/
/*2019/3/29                  version 0.2*/
#include "rnn.h"

#include<iostream>
#include<ctime>
#include<fstream>
#include<cstdlib>

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

void NormalRNN::Init()
{
    srand(unsigned(time(NULL)));
    for(int i=0;i<HNUM;i++)
    {
        hide[i].out[0]=0;
        hide[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
        for(int j=0;j<INUM;j++)
            hide[i].wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
        for(int j=0;j<HNUM;j++)
            hide[i].wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
    }
    return;
}

void NormalRNN::Datain(const std::string& filename)
{
    std::ifstream fin(filename);
    if(fin.fail())
    {
        std::cout<<">> [Error] Cannot open file."<<std::endl;
        exit(-1);
    }
    for(int i=0;i<HNUM;i++)
    {
        fin>>hide[i].out[0];
        fin>>hide[i].bia;
        for(int j=0;j<INUM;j++)
            fin>>hide[i].wi[j];
        for(int j=0;j<HNUM;j++)
            fin>>hide[i].wh[j];
    }
    fin.close();
    return;
}

void NormalRNN::Dataout(const std::string& filename)
{
    std::ofstream fout(filename);
    if(fout.fail())
    {
        std::cout<<">> [Error] Cannot open file."<<std::endl;
        exit(-1);
    }
    for(int i=0;i<HNUM;i++)
    {
        fout<<hide[i].out[0]<<std::endl;
        fout<<hide[i].bia<<std::endl;
        for(int j=0;j<INUM;j++)
            fout<<hide[i].wi[j]<<std::endl;
        for(int j=0;j<HNUM;j++)
            fout<<hide[i].wh[j]<<std::endl;
    }
    fout.close();
    return;
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

void DeepRNN::Init()
{
    srand(unsigned(time(NULL)));
    for(int i=0;i<HNUM;i++)
    {
        hlink[i].out[0]=0;
        hlink[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
        for(int j=0;j<INUM;j++)
            hlink[i].wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
        for(int j=0;j<HNUM;j++)
            hlink[i].wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
    }
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            hide[i][d].out[0]=0;
            hide[i][d].bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
            for(int j=0;j<HNUM;j++)
            {
                hide[i][d].wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
                hide[i][d].wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            }
        }
    return;
}

void DeepRNN::Datain(const std::string& filename)
{
    std::ifstream fin(filename);
    if(fin.fail())
    {
        std::cout<<">> [Error] Cannot open file."<<std::endl;
        exit(-1);
    }
    for(int i=0;i<HNUM;i++)
    {
        fin>>hlink[i].out[0];
        fin>>hlink[i].bia;
        for(int j=0;j<INUM;j++)
            fin>>hlink[i].wi[j];
        for(int j=0;j<HNUM;j++)
            fin>>hlink[i].wh[j];
    }
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            fin>>hide[i][d].out[0];
            fin>>hide[i][d].bia;
            for(int j=0;j<HNUM;j++)
            {
                fin>>hide[i][d].wi[j];
                fin>>hide[i][d].wh[j];
            }
        }
    fin.close();
    return;
}

void DeepRNN::Dataout(const std::string& filename)
{
    std::ofstream fout(filename);
    if(fout.fail())
    {
        std::cout<<">> [Error] Cannot open file."<<std::endl;
        exit(-1);
    }
    for(int i=0;i<HNUM;i++)
    {
        fout<<hlink[i].out[0]<<std::endl;
        fout<<hlink[i].bia<<std::endl;
        for(int j=0;j<INUM;j++)
            fout<<hlink[i].wi[j]<<std::endl;
        for(int j=0;j<HNUM;j++)
            fout<<hlink[i].wh[j]<<std::endl;
    }
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            fout<<hide[i][d].out[0]<<std::endl;
            fout<<hide[i][d].bia<<std::endl;
            for(int j=0;j<HNUM;j++)
            {
                fout<<hide[i][d].wi[j]<<std::endl;
                fout<<hide[i][d].wh[j]<<std::endl;
            }
        }
    fout.close();
    return;
}
