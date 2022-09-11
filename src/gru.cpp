/*grufunction.h header file made by ValK*/
/*2019/5/9                   version 1.2*/
#include "gru.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <ctime>

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

void NormalGRU::Init()
{
    srand(unsigned(time(NULL)));
    for(int i=0;i<HNUM;i++)
    {
        hide[i].out[0]=0;
        hide[i].sig_update_bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
        hide[i].sig_replace_bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
        hide[i].tan_replace_bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
        for(int j=0;j<INUM;j++)
        {
            hide[i].sig_update_wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            hide[i].sig_replace_wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            hide[i].tan_replace_wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
        }
        for(int j=0;j<HNUM;j++)
        {
            hide[i].sig_update_wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            hide[i].sig_replace_wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            hide[i].tan_replace_wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
        }
    }
    return;
}

void NormalGRU::Datain(const std::string& filename)
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
        fin>>hide[i].sig_update_bia;
        fin>>hide[i].sig_replace_bia;
        fin>>hide[i].tan_replace_bia;
        for(int j=0;j<INUM;j++)
        {
            fin>>hide[i].sig_update_wi[j];
            fin>>hide[i].sig_replace_wi[j];
            fin>>hide[i].tan_replace_wi[j];
        }
        for(int j=0;j<HNUM;j++)
        {
            fin>>hide[i].sig_update_wh[j];
            fin>>hide[i].sig_replace_wh[j];
            fin>>hide[i].tan_replace_wh[j];
        }
    }
    fin.close();
    return;
}

void NormalGRU::Dataout(const std::string& filename)
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
        fout<<hide[i].sig_update_bia<<std::endl;
        fout<<hide[i].sig_replace_bia<<std::endl;
        fout<<hide[i].tan_replace_bia<<std::endl;
        for(int j=0;j<INUM;j++)
        {
            fout<<hide[i].sig_update_wi[j]<<std::endl;
            fout<<hide[i].sig_replace_wi[j]<<std::endl;
            fout<<hide[i].tan_replace_wi[j]<<std::endl;
        }
        for(int j=0;j<HNUM;j++)
        {
            fout<<hide[i].sig_update_wh[j]<<std::endl;
            fout<<hide[i].sig_replace_wh[j]<<std::endl;
            fout<<hide[i].tan_replace_wh[j]<<std::endl;
        }
    }
    fout.close();
    return;
}

DeepGRU::DeepGRU(int InputlayerNum,int HiddenlayerNum,int Depth,int Maxtime)
{
    INUM=InputlayerNum;
    HNUM=HiddenlayerNum;
    DEPTH=Depth-1;
    MAXTIME=Maxtime;
    ConstructorAssist();
}

DeepGRU::~DeepGRU()
{
    DestructorAssist();
}

void DeepGRU::ConstructorAssist()
{
    int Maxtime=MAXTIME;
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
            hide[i][d].sig_update_wi=new double[HNUM];     hide[i][d].sig_replace_wi=new double[HNUM];     hide[i][d].tan_replace_wi=new double[HNUM];//must be HNUM
            hide[i][d].sig_update_wh=new double[HNUM];     hide[i][d].sig_replace_wh=new double[HNUM];     hide[i][d].tan_replace_wh=new double[HNUM];
            hide[i][d].sig_update_transwi=new double[HNUM];hide[i][d].sig_replace_transwi=new double[HNUM];hide[i][d].tan_replace_transwi=new double[HNUM];
            hide[i][d].sig_update_transwh=new double[HNUM];hide[i][d].sig_replace_transwh=new double[HNUM];hide[i][d].tan_replace_transwh=new double[HNUM];
        }
}

void DeepGRU::DestructorAssist()
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

void DeepGRU::Init()
{
    srand(unsigned(time(NULL)));
    for(int i=0;i<HNUM;i++)
    {
        hlink[i].out[0]=0;
        hlink[i].sig_update_bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
        hlink[i].sig_replace_bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
        hlink[i].tan_replace_bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
        for(int j=0;j<INUM;j++)
        {
            hlink[i].sig_update_wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            hlink[i].sig_replace_wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            hlink[i].tan_replace_wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
        }
        for(int j=0;j<HNUM;j++)
        {
            hlink[i].sig_update_wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            hlink[i].sig_replace_wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            hlink[i].tan_replace_wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
        }
    }
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            hide[i][d].out[0]=0;
            hide[i][d].sig_update_bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
            hide[i][d].sig_replace_bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
            hide[i][d].tan_replace_bia=(rand()%2? 1:-1)*(1.0+rand()%10)/10.0;
            for(int j=0;j<HNUM;j++)
            {
                hide[i][d].sig_update_wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
                hide[i][d].sig_replace_wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
                hide[i][d].tan_replace_wi[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
                hide[i][d].sig_update_wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
                hide[i][d].sig_replace_wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
                hide[i][d].tan_replace_wh[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/50.0;
            }
        }
    return;
}

void DeepGRU::Datain(const std::string& filename)
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
        fin>>hlink[i].sig_update_bia;
        fin>>hlink[i].sig_replace_bia;
        fin>>hlink[i].tan_replace_bia;
        for(int j=0;j<INUM;j++)
        {
            fin>>hlink[i].sig_update_wi[j];
            fin>>hlink[i].sig_replace_wi[j];
            fin>>hlink[i].tan_replace_wi[j];
        }
        for(int j=0;j<HNUM;j++)
        {
            fin>>hlink[i].sig_update_wh[j];
            fin>>hlink[i].sig_replace_wh[j];
            fin>>hlink[i].tan_replace_wh[j];
        }
    }
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            fin>>hide[i][d].out[0];
            fin>>hide[i][d].sig_update_bia;
            fin>>hide[i][d].sig_replace_bia;
            fin>>hide[i][d].tan_replace_bia;
            for(int j=0;j<HNUM;j++)
            {
                fin>>hide[i][d].sig_update_wi[j];
                fin>>hide[i][d].sig_replace_wi[j];
                fin>>hide[i][d].tan_replace_wi[j];
                fin>>hide[i][d].sig_update_wh[j];
                fin>>hide[i][d].sig_replace_wh[j];
                fin>>hide[i][d].tan_replace_wh[j];
            }
        }
    fin.close();
    return;
}

void DeepGRU::Dataout(const std::string& filename)
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
        fout<<hlink[i].sig_update_bia<<std::endl;
        fout<<hlink[i].sig_replace_bia<<std::endl;
        fout<<hlink[i].tan_replace_bia<<std::endl;
        for(int j=0;j<INUM;j++)
        {
            fout<<hlink[i].sig_update_wi[j]<<std::endl;
            fout<<hlink[i].sig_replace_wi[j]<<std::endl;
            fout<<hlink[i].tan_replace_wi[j]<<std::endl;
        }
        for(int j=0;j<HNUM;j++)
        {
            fout<<hlink[i].sig_update_wh[j]<<std::endl;
            fout<<hlink[i].sig_replace_wh[j]<<std::endl;
            fout<<hlink[i].tan_replace_wh[j]<<std::endl;
        }
    }
    for(int d=0;d<DEPTH;d++)
        for(int i=0;i<HNUM;i++)
        {
            fout<<hide[i][d].out[0];
            fout<<hide[i][d].sig_update_bia<<std::endl;
            fout<<hide[i][d].sig_replace_bia<<std::endl;
            fout<<hide[i][d].tan_replace_bia<<std::endl;
            for(int j=0;j<HNUM;j++)
            {
                fout<<hide[i][d].sig_update_wi[j]<<std::endl;
                fout<<hide[i][d].sig_replace_wi[j]<<std::endl;
                fout<<hide[i][d].tan_replace_wi[j]<<std::endl;
                fout<<hide[i][d].sig_update_wh[j]<<std::endl;
                fout<<hide[i][d].sig_replace_wh[j]<<std::endl;
                fout<<hide[i][d].tan_replace_wh[j]<<std::endl;
            }
        }
    fout.close();
    return;
}
