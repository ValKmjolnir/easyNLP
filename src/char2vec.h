/*char2vec.h header file made by valk*/
/*2019/5/2               version 1.1*/
#ifndef __CHAR2VEC_H__
#define __CHAR2VEC_H__
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "bp.h"
#include "actvfunc.h"

class Char2Vec
{
private:
    int INUM;
    int HNUM;
    int ONUM;
    std::vector<std::vector<double>> cnt;
    std::vector<double> input;
    std::vector<double> expect;
    double lr;
    std::vector<neuron> hide;
    std::vector<neuron> output;
public:
    Char2Vec(const int hnum=256)
    {
        lr=0.1;
        INUM=95;
        ONUM=95;
        HNUM=hnum;
        cnt.resize(95);
        for(int i=0;i<95;i++)
            cnt[i].resize(95,0);
        input.resize(95);
        expect.resize(95);
        hide.resize(HNUM);
        output.resize(ONUM);
        for(int i=0;i<HNUM;i++)
            hide[i].w=new double[INUM];
        for(int i=0;i<ONUM;i++)
            output[i].w=new double[HNUM];

        srand(unsigned(time(NULL)));
        for(int i=0;i<HNUM;i++)
        {
            hide[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)/(1.0*INUM);
            for(int j=0;j<INUM;j++)
                hide[i].w[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/(10.0*INUM);
        }
        for(int i=0;i<ONUM;i++)
        {
            output[i].bia=(rand()%2? 1:-1)*(1.0+rand()%10)/(1.0*HNUM);
            for(int j=0;j<HNUM;j++)
                output[i].w[j]=(rand()%2? 1:-1)*(1.0+rand()%10)/(10.0*HNUM);
        }
        return;
    }
    ~Char2Vec()
    {
        for(int i=0;i<HNUM;i++)
            delete []hide[i].w;
        for(int i=0;i<ONUM;i++)
            delete []output[i].w;
    }
    void TotalWork(const std::string&,const std::string&);
    void Mainwork(const std::string&);
    void Calc();
    void Training();
    void Datain(const std::string&);
    void Dataout(const std::string&);
    void Print();
    void CountChar(const std::string&);
    void CharDataIllustration(const std::string&);
};

void Char2Vec::TotalWork(const std::string& dataFilename,const std::string& TrainingdataName)
{
    if(!fopen(dataFilename.c_str(),"r"))
    {
        Dataout(dataFilename);
        std::cout<<">> [Char2Vec-95char] Initializing completed.\n";
    }
    else
        Datain(dataFilename);
    CountChar(TrainingdataName);
    Mainwork(dataFilename);
    Print();
}

void Char2Vec::Mainwork(const std::string& filename)
{
    int epoch=0;
    double maxerror=1e8;
    double error=1e8;
    double softmax=0;

    double max_cnt=-1;
    for(auto& i:cnt)
        for(auto j:i)
            if(j>max_cnt)
                max_cnt=j;
    // limit to 0~95
    if(max_cnt!=0)
        for(auto& i:cnt)
            for(auto& j:i)
                j=j/max_cnt*95;
    while(maxerror>0.1)
    {
        epoch++;
        maxerror=0;
        for(int i=0;i<95;i++)
        {
            for(int j=0;j<INUM;j++)
                input[j]=0;
            input[i]=1;
            softmax=0;
            for(int j=0;j<95;j++)
                softmax+=std::exp(cnt[i][j]);
            for(int j=0;j<ONUM;j++)
                expect[j]=std::exp(cnt[i][j])/softmax;
            Calc();
            error=0;
            for(int j=0;j<ONUM;j++)
                error+=(expect[j]-output[j].out)*(expect[j]-output[j].out);
            error*=0.5;
            maxerror+=error;
            Training();
        }
        if(epoch%100==0)
        {
            std::cout<<">> Epoch "<<epoch<<": Error :"<<maxerror<<std::endl;
            if(epoch%500==0)
                Dataout(filename);
        }
    }
    std::cout<<">> Finish training by "<<epoch<<" epoch, error="<<maxerror<<std::endl;
    std::cout<<">> Final output in progress..."<<std::endl;
    Dataout(filename);
    std::cout<<">> Training complete."<<std::endl;
    return;
}
void Char2Vec::Calc()
{
    double softmax=0;
    for(int i=0;i<HNUM;i++)
    {
        hide[i].in=hide[i].bia;
        for(int j=0;j<INUM;j++)
            hide[i].in+=hide[i].w[j]*input[j];
        hide[i].out=tanh(hide[i].in);
    }
    for(int i=0;i<ONUM;i++)
    {
        output[i].in=output[i].bia;
        for(int j=0;j<HNUM;j++)
            output[i].in+=output[i].w[j]*hide[j].out;
        softmax+=std::exp(output[i].in);
    }
    for(int i=0;i<ONUM;i++)
        output[i].out=std::exp(output[i].in)/softmax;
    return;
}
void Char2Vec::Training()
{
    for(int i=0;i<ONUM;i++)
        output[i].diff=(expect[i]-output[i].out)*output[i].out*(1-output[i].out);
    for(int i=0;i<HNUM;i++)
    {
        hide[i].diff=0;
        for(int j=0;j<ONUM;j++)
            hide[i].diff+=output[j].diff*output[j].w[i];
    }
    for(int i=0;i<ONUM;i++)
    {
        output[i].bia+=lr*2*output[i].diff;
        for(int j=0;j<HNUM;j++)
            output[i].w[j]+=lr*output[i].diff*hide[j].out;
    }
    for(int i=0;i<HNUM;i++)
    {
        hide[i].bia+=lr*2*hide[i].diff;
        for(int j=0;j<INUM;j++)
            hide[i].w[j]+=lr*hide[i].diff*input[j];
    }
    return;
}
void Char2Vec::Datain(const std::string& filename)
{
    std::ifstream fin(filename);
    if(fin.fail())
    {
        std::cout<<">> [Error] Cannot open data file!"<<std::endl;
        exit(-1);
    }
    for(int i=0;i<HNUM;i++)
    {
        fin>>hide[i].bia;
        for(int j=0;j<INUM;j++)
            fin>>hide[i].w[j];
    }
    for(int i=0;i<ONUM;i++)
    {
        fin>>output[i].bia;
        for(int j=0;j<HNUM;j++)
            fin>>output[i].w[j];
    }
    fin.close();
    return;
}
void Char2Vec::Dataout(const std::string& filename)
{
    std::ofstream fout(filename);
    if(fout.fail())
    {
        std::cout<<">> [Error] Cannot open data file!"<<std::endl;
        exit(-1);
    }
    for(int i=0;i<HNUM;i++)
    {
        fout<<hide[i].bia<<std::endl;
        for(int j=0;j<INUM;j++)
            fout<<hide[i].w[j]<<std::endl;
    }
    for(int i=0;i<ONUM;i++)
    {
        fout<<output[i].bia<<std::endl;
        for(int j=0;j<HNUM;j++)
            fout<<output[i].w[j]<<std::endl;
    }
    fout.close();
    return;
}
void Char2Vec::Print()
{
    std::cout<<">> [Result-Char2Vec-95char]"<<std::endl;
    for(int i=0;i<95;i++)
    {
        for(int j=0;j<INUM;j++)
            input[j]=0;
        input[i]=1;
        Calc();
        bool has_related=false;
        for(int j=0;j<ONUM;j++)
            if(output[j].out>0.1)
            {
                has_related=true;
                break;
            }
        if(!has_related)
            continue;
        std::cout<<"   |"<<(char)(i+32)<<":  ";
        for(int j=0;j<ONUM;j++)
            if(output[j].out>0.05)
                std::cout<<"|"<<(char)(j+32)<<':'<<int(100*output[j].out)<<"% ";
        std::cout<<std::endl;
    }
    return;
}
void Char2Vec::CountChar(const std::string& filename)
{
    for(int i=0;i<95;i++)
        for(int j=0;j<95;j++)
            cnt[i][j]=0;
    char temp[1024];
    std::ifstream fin(filename);
    if(fin.fail())
    {
        std::cout<<">> [Error] Cannot open data file!"<<std::endl;
        exit(-1);
    }
    while(!fin.eof())
    {
        for(int i=0;i<1024;i++)
            temp[i]=0;
        fin.getline(temp,1024,'\n');
        if(fin.eof())
            break;
        for(int i=1;temp[i]!=0;i++)
            if(temp[i-1]>=32&&temp[i-1]<=126&&temp[i]>=32&&temp[i]<=126)
                cnt[temp[i-1]-32][temp[i]-32]++;
    }
    std::cout<<">> [Info] character counting complete."<<std::endl;
    return;
}
void Char2Vec::CharDataIllustration(const std::string& filename)
{
    std::ofstream fout(filename);
    if(fout.fail())
    {
        std::cout<<">> [Error] Cannot open data file!"<<std::endl;
        exit(-1);
    }
    fout<<"# ";
    for(int i=0;i<95;i++)
    {
        if(i==0)
            fout<<"space ";
        else
            fout<<(char)(i+32)<<' ';
    }
    fout<<std::endl;
    for(int i=0;i<95;i++)
    {
        if(i==0)
            fout<<"space ";
        else
            fout<<(char)(i+32)<<' ';
        for(int j=0;j<95;j++)
            fout<<cnt[i][j]/100.0<<' ';
        fout<<std::endl;
    }
    fout.close();
}
#endif
