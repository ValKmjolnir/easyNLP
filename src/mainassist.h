/*mainassist.h header file by ValK*/
/*2019/5/9             version 1.1*/
#ifndef MAINASSIST_H
#define MAINASSIST_H
#include "NLPann.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

struct ObjElement
{
    char ObjName[40];
    char FileName_1[40];
    char FileName_2[40];
    char FileName_3[40];
    char FileName_4[40];
    char FileName_5[40];
    char FileName_6[40];
    char FileName_7[40];
    char FileName_8[40];
    char Function[10];
    int INUM;
    int HNUM;
    int ONUM;
    int DEPTH;
    int MAXTIME;
    int NetworkType;
    int BatchSize;
    double LearningRate;
};

class UserObject
{
    private:
        ObjElement Obj;
    public:
        UserObject *p;
        UserObject(
        const char *objname="NULL",
        const char *filename1="NULL",
        const char *filename2="NULL",
        const char *filename3="NULL",
        const char *filename4="NULL",
        const char *filename5="NULL",
        const char *filename6="NULL",
        const char *filename7="NULL",
        const char *filename8="NULL",
        const char *function="NULL",
        int inum=0,
        int hnum=0,
        int onum=0,
        int depth=0,
        int maxtime=0,
        int networktype=0,
        int batchsize=0,
        double lr=0.1
        )
        {
            strcpy(Obj.ObjName,objname);
            strcpy(Obj.FileName_1,filename1);
            strcpy(Obj.FileName_2,filename2);
            strcpy(Obj.FileName_3,filename3);
            strcpy(Obj.FileName_4,filename4);
            strcpy(Obj.FileName_5,filename5);
            strcpy(Obj.FileName_6,filename6);
            strcpy(Obj.FileName_7,filename7);
            strcpy(Obj.FileName_8,filename8);
            strcpy(Obj.Function,function);
            Obj.INUM=inum;
            Obj.HNUM=hnum;
            Obj.ONUM=onum;
            Obj.DEPTH=depth;
            Obj.MAXTIME=maxtime;
            Obj.NetworkType=networktype;
            Obj.BatchSize=batchsize;
            Obj.LearningRate=lr;
            p=NULL;
        }
        void PrintObj()
        {
            std::cout<<"   ------------------------------------------------------"<<std::endl;
            std::cout<<"   |Name         |"<<Obj.ObjName<<std::endl;
            if(strcmp(Obj.FileName_1,"NULL"))
                std::cout<<"   |File 1       |"<<Obj.FileName_1<<std::endl;
            if(strcmp(Obj.FileName_2,"NULL"))
                std::cout<<"   |File 2       |"<<Obj.FileName_2<<std::endl;
            if(strcmp(Obj.FileName_3,"NULL"))
                std::cout<<"   |File 3       |"<<Obj.FileName_3<<std::endl;
            if(strcmp(Obj.FileName_4,"NULL"))
                std::cout<<"   |File 4       |"<<Obj.FileName_4<<std::endl;
            if(strcmp(Obj.FileName_5,"NULL"))
                std::cout<<"   |File 5       |"<<Obj.FileName_5<<std::endl;
            if(strcmp(Obj.FileName_6,"NULL"))
                std::cout<<"   |File 6       |"<<Obj.FileName_6<<std::endl;
            if(strcmp(Obj.FileName_7,"NULL"))
                std::cout<<"   |File 7       |"<<Obj.FileName_7<<std::endl;
            if(strcmp(Obj.FileName_8,"NULL"))
                std::cout<<"   |File 8       |"<<Obj.FileName_8<<std::endl;
            std::cout<<"   |Function     |"<<Obj.Function<<std::endl;
            std::cout<<"   |INUM         |"<<Obj.INUM<<std::endl;
            std::cout<<"   |HNUM         |"<<Obj.HNUM<<std::endl;
            std::cout<<"   |ONUM         |"<<Obj.ONUM<<std::endl;
            std::cout<<"   |DEPTH        |"<<Obj.DEPTH<<std::endl;
            std::cout<<"   |MAXTIME      |"<<Obj.MAXTIME<<std::endl;
            std::cout<<"   |LearningRate |"<<Obj.LearningRate<<std::endl;
            std::cout<<"   |Batch Size   |"<<Obj.BatchSize<<std::endl;
            std::cout<<"   |Network Type |";
            switch(Obj.NetworkType)
            {
                case 1:
                    std::cout<<"BP(Normal neural network)"<<std::endl;
                    break;
                case 2:
                    std::cout<<"BP(Deep neural network)"<<std::endl;
                    break;
                case 3:
                    std::cout<<"RNN seq2seq(Normal neural network)"<<std::endl;
                    break;
                case 4:
                    std::cout<<"RNN seq2seq(Deep neural network)"<<std::endl;
                    break;
                case 5:
                    std::cout<<"LSTM seq2seq(Normal neural network)"<<std::endl;
                    break;
                case 6:
                    std::cout<<"LSTM seq2seq(Deep neural network)"<<std::endl;
                    break;
                case 7:
                    std::cout<<"GRU seq2seq(Normal neural network)"<<std::endl;
                    break;
                case 8:
                    std::cout<<"GRU seq2seq(Deep neural network)"<<std::endl;
                    break;
                case 9:
                    std::cout<<"RNN seq2vec(Normal neural network)"<<std::endl;
                    break;
                case 10:
                    std::cout<<"RNN seq2vec(Deep neural network)"<<std::endl;
                    break;
                case 11:
                    std::cout<<"LSTM seq2vec(Normal neural network)"<<std::endl;
                    break;
                case 12:
                    std::cout<<"LSTM seq2vec(Deep neural network)"<<std::endl;
                    break;
                case 13:
                    std::cout<<"GRU seq2vec(Normal neural network)"<<std::endl;
                    break;
                case 14:
                    std::cout<<"GRU seq2vec(Deep neural network)"<<std::endl;
                    break;
                case 15:
                    std::cout<<"BP char2vec(Normal neural network)"<<std::endl;
                    break;
                default:
                    std::cout<<"Unknown Type"<<std::endl;
                    break;
            }
            std::cout<<"   ------------------------------------------------------"<<std::endl;
        }
        ObjElement* getObjPointer()
        {
            return &Obj;
        }
        bool CheckObjName(std::string &TempName)
        {
            if(TempName==Obj.ObjName)
                return true;
            else
                return false;
        }
        bool CheckObjName(const char *t)
        {
            if(!strcmp(t,Obj.ObjName))
                return true;
            else
                return false;
        }
        void ObjChange(ObjElement &temp)
        {
            Obj=temp;
            return;
        }
};


class ObjManager
{
    private:
        UserObject *Head;
        ObjElement Datatemp;
    public:
        ObjManager();
        ~ObjManager();
        void ObjDataIn();
        void ObjDataOut();
        void MakeData();
        bool ObjChoose();
        void RunModule();
        void PrintAllObj();
        void FindObj();
        void EditObj();
        void DeleteObj();
        void ChangeLearningRate();
        void ChangeBatchSize();
        void FindSpecialObj(const char*);
};

ObjManager::ObjManager()
{
    Head=new UserObject;
    Head->p=NULL;
}

ObjManager::~ObjManager()
{
    UserObject *Node=Head;
    UserObject *Temp;
    while(Node->p!=NULL)
    {
        Temp=Node;
        Node=Node->p;
        delete Temp;
    }
    delete Node;
}

void ObjManager::ObjDataIn()
{
    UserObject *Node=Head;
    if(!fopen("ObjData.dat","r"))
    {
        FILE* fd=fopen("ObjData.dat","w");
        fclose(fd);
        std::cout<<">> [init] Initializing completed."<<std::endl;
    }
    std::ifstream fin("ObjData.dat",std::ios::binary);
    if(fin.fail())
    {
        std::cout<<">> [Error] Cannot open important data \"ObjData.dat\" or this data maybe lost!"<<std::endl;
        exit(-1);
    }
    while(!fin.eof())
    {
        fin.read((char*)&Datatemp,sizeof(ObjElement));
        if(fin.eof())
            break;
        Node->p=new UserObject(
        Datatemp.ObjName,
        Datatemp.FileName_1,
        Datatemp.FileName_2,
        Datatemp.FileName_3,
        Datatemp.FileName_4,
        Datatemp.FileName_5,
        Datatemp.FileName_6,
        Datatemp.FileName_7,
        Datatemp.FileName_8,
        Datatemp.Function,
        Datatemp.INUM,
        Datatemp.HNUM,
        Datatemp.ONUM,
        Datatemp.DEPTH,
        Datatemp.MAXTIME,
        Datatemp.NetworkType,
        Datatemp.BatchSize,
        Datatemp.LearningRate
        );
        Node=Node->p;
        Node->p=NULL;
    }
    fin.close();
    return;
}

void ObjManager::ObjDataOut()
{
    UserObject *Node=Head;
    std::ofstream fout("ObjData.dat",std::ios::binary);
    if(fout.fail())
    {
        std::cout<<">> [Error] Cannot open important data \"Objdata.dat\" or this data maybe lost!"<<std::endl;
        exit(-1);
    }
    while(Node->p!=NULL)
    {
        Node=Node->p;
        fout.write((char*)Node->getObjPointer(),sizeof(ObjElement));
    }
    fout.close(); 
    return;
}

void ObjManager::MakeData()
{
    UserObject *Node=Head;
    std::cout<<">> Name of your project: ";
    std::cin>>Datatemp.ObjName;
    while(Node->p!=NULL)
    {
        Node=Node->p;
        if(Node->CheckObjName(Datatemp.ObjName))
        {
            std::cout<<">> [Error] You have already created this project!"<<std::endl<<std::endl;
            std::cout<<"   |The project is:"<<std::endl;
            Node->PrintObj();
            return;
        }
    }
    if(!ObjChoose())
    {
        std::cout<<">> [Quiting]"<<std::endl;
        return;
    }
    Node->p=new UserObject(
    Datatemp.ObjName,
    Datatemp.FileName_1,
    Datatemp.FileName_2,
    Datatemp.FileName_3,
    Datatemp.FileName_4,
    Datatemp.FileName_5,
    Datatemp.FileName_6,
    Datatemp.FileName_7,
    Datatemp.FileName_8,
    Datatemp.Function,
    Datatemp.INUM,
    Datatemp.HNUM,
    Datatemp.ONUM,
    Datatemp.DEPTH,
    Datatemp.MAXTIME,
    Datatemp.NetworkType,
    Datatemp.BatchSize,
    Datatemp.LearningRate
    );
    Node=Node->p;
    Node->p=NULL;
    std::cout<<">> New project is established successfully."<<std::endl;
}

bool ObjManager::ObjChoose()
{
    std::string Command;
    std::cout<<">>"<<std::endl;
    std::cout<<"   |1. |BP project(Normal)            |"<<std::endl;
    std::cout<<"   |2. |BP project(Deep)              |"<<std::endl;
    std::cout<<"   |3. |seq2seq project(RNN:Normal)   |"<<std::endl;
    std::cout<<"   |4. |seq2seq project(RNN:Deep)     |"<<std::endl;
    std::cout<<"   |5. |seq2seq project(LSTM:Normal)  |"<<std::endl;
    std::cout<<"   |6. |seq2seq project(LSTM:Deep)    |"<<std::endl;
    std::cout<<"   |7. |seq2seq project(GRU:Normal)   |"<<std::endl;
    std::cout<<"   |8. |seq2seq project(GRU:Deep)     |"<<std::endl;
    std::cout<<"   |9. |seq2vec project(RNN:Normal)   |"<<std::endl;
    std::cout<<"   |10.|seq2vec project(RNN:Deep)     |"<<std::endl;
    std::cout<<"   |11.|seq2vec project(LSTM:Normal)  |"<<std::endl;
    std::cout<<"   |12.|seq2vec project(LSTM:Deep)    |"<<std::endl;
    std::cout<<"   |13.|seq2vec project(GRU:Normal)   |"<<std::endl;
    std::cout<<"   |14.|seq2vec project(GRU:Deep)     |"<<std::endl;
    std::cout<<"   |15.|char2vec project(BP:Normal)   |"<<std::endl;
    std::cout<<"   |16.|I don't want to choose.(quit) |"<<std::endl;
    while(1)
    {
        std::cout<<">> [Choice] Input your choice: ";
        std::cin>>Command;
        if(Command=="1")
        {
            std::cout<<">>"<<std::endl;
            std::cout<<"   |BP project(Normal) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for all layers)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the output file: ";
            std::cin>>Datatemp.FileName_1;
            strcpy(Datatemp.FileName_2,"NULL");
            strcpy(Datatemp.FileName_3,"NULL");
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function: ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.DEPTH=0;
            Datatemp.MAXTIME=0;
            Datatemp.NetworkType=1;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="2")
        {
            std::cout<<">>"<<std::endl;
            std::cout<<"   |BP project(Deep) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |DEPTH: Number of layers"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for all layers)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the output file: ";
            std::cin>>Datatemp.FileName_1;
            strcpy(Datatemp.FileName_2,"NULL");
            strcpy(Datatemp.FileName_3,"NULL");
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function: ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            std::cout<<">> Number of layers: ";
            std::cin>>Datatemp.DEPTH;
            Datatemp.MAXTIME=0;
            Datatemp.NetworkType=2;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="3")
        {
            std::cout<<">> [RNN]"<<std::endl;
            std::cout<<"   |seq2seq project(RNN/LSTM/GRU:Normal) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the decoder module file: ";
            std::cin>>Datatemp.FileName_2;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_3;
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            Datatemp.DEPTH=0;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=3;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="4")
        {
            std::cout<<">> [RNN]"<<std::endl;
            std::cout<<"   |seq2seq project(RNN/LSTM/GRU:Deep) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |DEPTH: Number of layers"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the decoder module file: ";
            std::cin>>Datatemp.FileName_2;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_3;
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            std::cout<<">> Number of layers: ";
            std::cin>>Datatemp.DEPTH;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=4;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="5")
        {
            std::cout<<">> [LSTM]"<<std::endl;
            std::cout<<"   |seq2seq project(RNN/LSTM/GRU:Normal) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the decoder module file: ";
            std::cin>>Datatemp.FileName_2;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_3;
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            Datatemp.DEPTH=0;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=5;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="6")
        {
            std::cout<<">> [LSTM]"<<std::endl;
            std::cout<<"   |seq2seq project(RNN/LSTM/GRU:Deep) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |DEPTH: Number of layers"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the decoder module file: ";
            std::cin>>Datatemp.FileName_2;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_3;
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            std::cout<<">> Number of layers: ";
            std::cin>>Datatemp.DEPTH;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            Datatemp.NetworkType=6;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="7")
        {
            std::cout<<">> [GRU]"<<std::endl;
            std::cout<<"   |seq2seq project(RNN/LSTM/GRU:Normal) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the decoder module file: ";
            std::cin>>Datatemp.FileName_2;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_3;
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            Datatemp.DEPTH=0;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=7;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="8")
        {
            std::cout<<">> [GRU]"<<std::endl;
            std::cout<<"   |seq2seq project(RNN/LSTM/GRU:Deep) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |DEPTH: Number of layers"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the decoder module file: ";
            std::cin>>Datatemp.FileName_2;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_3;
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            std::cout<<">> Number of layers: ";
            std::cin>>Datatemp.DEPTH;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=8;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="9")
        {
            std::cout<<">> [RNN]"<<std::endl;
            std::cout<<"   |seq2vec project(RNN/LSTM/GRU:Normal) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_2;
            strcpy(Datatemp.FileName_3,"NULL");
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            Datatemp.DEPTH=0;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=9;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="10")
        {
            std::cout<<">> [RNN]"<<std::endl;
            std::cout<<"   |seq2vec project(RNN/LSTM/GRU:Deep) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |DEPTH: Number of layers"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_2;
            strcpy(Datatemp.FileName_3,"NULL");
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            std::cout<<">> Number of layers: ";
            std::cin>>Datatemp.DEPTH;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=10;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="11")
        {
            std::cout<<">> [LSTM]"<<std::endl;
            std::cout<<"   |seq2vec project(RNN/LSTM/GRU:Normal) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_2;
            strcpy(Datatemp.FileName_3,"NULL");
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            Datatemp.DEPTH=0;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=11;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="12")
        {
            std::cout<<">> [LSTM]"<<std::endl;
            std::cout<<"   |seq2vec project(RNN/LSTM/GRU:Deep) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |DEPTH: Number of layers"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_2;
            strcpy(Datatemp.FileName_3,"NULL");
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            std::cout<<">> Number of layers: ";
            std::cin>>Datatemp.DEPTH;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=12;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="13")
        {
            std::cout<<">> [GRU]"<<std::endl;
            std::cout<<"   |seq2vec project(RNN/LSTM/GRU:Normal) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_2;
            strcpy(Datatemp.FileName_3,"NULL");
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            Datatemp.DEPTH=0;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=13;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="14")
        {
            std::cout<<">> [GRU]"<<std::endl;
            std::cout<<"   |seq2vec project(RNN/LSTM/GRU:Deep) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons"<<std::endl;
            std::cout<<"   |DEPTH: Number of layers"<<std::endl;
            std::cout<<"   |MAXTIME: Max length of sequence"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of input set file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions(for output layer)"<<std::endl;
            std::cout<<"   |Batch size: Size of input/training set batch"<<std::endl;
            std::cout<<"   |Learning rate:This decides how fast your model runs"<<std::endl;
            std::cout<<">> Name of the encoder module file: ";
            std::cin>>Datatemp.FileName_1;
            std::cout<<">> Name of the output layer file: ";
            std::cin>>Datatemp.FileName_2;
            strcpy(Datatemp.FileName_3,"NULL");
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            std::cout<<">> Name of input set file: ";
            std::cin>>Datatemp.FileName_7;
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            std::cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
            std::cin>>Datatemp.Function;
            std::cout<<">> Number of input layer neurons: ";
            std::cin>>Datatemp.INUM;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            std::cout<<">> Number of output layer neurons: ";
            std::cin>>Datatemp.ONUM;
            Datatemp.INUM=27;
            Datatemp.ONUM=27;
            std::cout<<">> Number of layers: ";
            std::cin>>Datatemp.DEPTH;
            std::cout<<">> Max length of sequence: ";
            std::cin>>Datatemp.MAXTIME;
            Datatemp.NetworkType=14;
            std::cout<<">> Learning rate(must more than 0 but less than 1): ";
            std::cin>>Datatemp.LearningRate;
            std::cout<<">> Size of input/training set batch: ";
            std::cin>>Datatemp.BatchSize;
            break;
        }
        else if(Command=="15")
        {
            std::cout<<">>"<<std::endl;
            std::cout<<"   |char2vec project(BP:Normal) needs:"<<std::endl;
            std::cout<<"   |INUM: Number of input layer neurons is set to 95"<<std::endl;
            std::cout<<"   |HNUM: Number of hidden layer neurons"<<std::endl;
            std::cout<<"   |ONUM: Number of output layer neurons is set to 95"<<std::endl;
            std::cout<<"   |File: Name of data file"<<std::endl;
            std::cout<<"   |File: Name of training set file"<<std::endl;
            std::cout<<"   |Function: one of activate functions is set as softmax"<<std::endl;
            std::cout<<"   |Learning rate:0.1"<<std::endl;
            std::cout<<">> Name of the output data file: ";
            std::cin>>Datatemp.FileName_1;
            strcpy(Datatemp.FileName_2,"NULL");
            strcpy(Datatemp.FileName_3,"NULL");
            strcpy(Datatemp.FileName_4,"NULL");
            strcpy(Datatemp.FileName_5,"NULL");
            strcpy(Datatemp.FileName_6,"NULL");
            strcpy(Datatemp.FileName_7,"NULL");
            std::cout<<">> Name of training set file: ";
            std::cin>>Datatemp.FileName_8;
            strcpy(Datatemp.Function,"softmax");
            Datatemp.INUM=95;
            std::cout<<">> Number of hidden layer neurons: ";
            std::cin>>Datatemp.HNUM;
            Datatemp.ONUM=95;
            Datatemp.DEPTH=0;
            Datatemp.MAXTIME=0;
            Datatemp.NetworkType=15;
            Datatemp.LearningRate=0.1;
            Datatemp.BatchSize=0;
            break;
        }
        else if(Command=="16")
        {
            return false;
        }
        else
            std::cout<<">> [Error] Undefined choice."<<std::endl;
    }
    return true;
}

void ObjManager::RunModule()
{
    bool FoundObj=false;
    UserObject *Node=Head;
    std::string temp_obj_name;
    std::cout<<">> Name of the project: ";
    std::cin>>temp_obj_name;
    while(Node->p!=NULL)
    {
        Node=Node->p;
        if(Node->CheckObjName(temp_obj_name))
        {
            Node->PrintObj();
            FoundObj=true;
            break;
        }
    }
    if(FoundObj)
    {
        if(Node->getObjPointer()->NetworkType==1)
        {
            std::cout<<">> [Running] BP(Normal neural network)"<<std::endl;
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            NormalBP MainBP(Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM);
            MainBP.SetFunction(Node->getObjPointer()->Function);
            MainBP.SetLearningrate(Node->getObjPointer()->LearningRate);
            MainBP.TotalWork(Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==2)
        {
            std::cout<<">> [Running] BP(Deep neural network)"<<std::endl;
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            DeepBP MainBP(Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->DEPTH);
            MainBP.SetFunction(Node->getObjPointer()->Function);
            MainBP.SetLearningrate(Node->getObjPointer()->LearningRate);
            MainBP.TotalWork(Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==3)
        {
            std::cout<<">> [Running] RNN seq2seq(Normal neural network)"<<std::endl;
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            NormalSeq2Seq MainSeq("rnn",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->MAXTIME);
            MainSeq.SetFunction(Node->getObjPointer()->Function);
            MainSeq.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainSeq.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainSeq.TotalWork("rnn",Node->getObjPointer()->FileName_1,
                                    Node->getObjPointer()->FileName_2,
                                    Node->getObjPointer()->FileName_3,
                                    Node->getObjPointer()->FileName_7,
                                    Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==4)
        {
            std::cout<<">> [Running] RNN seq2seq(Deep neural network)"<<std::endl;
            if(Node->getObjPointer()->DEPTH>2)
            {
                char Confirm;
                std::cout<<">> [Warning] Seq2Seq with two more layers may not work well,do you still want to run this model?(y/n)"<<std::endl;
                std::cin>>Confirm;
                if(Confirm!='y')
                {
                    std::cout<<">> [Error] Running process cancelled"<<std::endl;
                    return;
                }
            }
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            DeepSeq2Seq MainSeq("rnn",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->DEPTH,Node->getObjPointer()->MAXTIME);
            MainSeq.SetFunction(Node->getObjPointer()->Function);
            MainSeq.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainSeq.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainSeq.TotalWork("rnn",Node->getObjPointer()->FileName_1,
                                    Node->getObjPointer()->FileName_2,
                                    Node->getObjPointer()->FileName_3,
                                    Node->getObjPointer()->FileName_7,
                                    Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==5)
        {
            std::cout<<">> [Running] LSTM seq2seq(Normal neural network)"<<std::endl;
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            NormalSeq2Seq MainSeq("lstm",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->MAXTIME);
            MainSeq.SetFunction(Node->getObjPointer()->Function);
            MainSeq.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainSeq.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainSeq.TotalWork("lstm",Node->getObjPointer()->FileName_1,
                                    Node->getObjPointer()->FileName_2,
                                    Node->getObjPointer()->FileName_3,
                                    Node->getObjPointer()->FileName_7,
                                    Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==6)
        {
            std::cout<<">> [Running] LSTM seq2seq(Deep neural network)"<<std::endl;
            if(Node->getObjPointer()->DEPTH>2)
            {
                char Confirm;
                std::cout<<">> [Warning] Seq2Seq with two more layers may not work well,do you still want to run this model?(y/n)"<<std::endl;
                std::cin>>Confirm;
                if(Confirm!='y')
                {
                    std::cout<<">> [Error] Running process cancelled"<<std::endl;
                    return;
                }
            }
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            DeepSeq2Seq MainSeq("lstm",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->DEPTH,Node->getObjPointer()->MAXTIME);
            MainSeq.SetFunction(Node->getObjPointer()->Function);
            MainSeq.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainSeq.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainSeq.TotalWork("lstm",Node->getObjPointer()->FileName_1,
                                    Node->getObjPointer()->FileName_2,
                                    Node->getObjPointer()->FileName_3,
                                    Node->getObjPointer()->FileName_7,
                                    Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==7)
        {
            std::cout<<">> [Running] GRU seq2seq(Normal neural network)"<<std::endl;
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            NormalSeq2Seq MainSeq("gru",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->MAXTIME);
            MainSeq.SetFunction(Node->getObjPointer()->Function);
            MainSeq.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainSeq.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainSeq.TotalWork("gru",Node->getObjPointer()->FileName_1,
                                    Node->getObjPointer()->FileName_2,
                                    Node->getObjPointer()->FileName_3,
                                    Node->getObjPointer()->FileName_7,
                                    Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==8)
        {
            std::cout<<">> [Running] GRU seq2seq(Deep neural network)"<<std::endl;
            if(Node->getObjPointer()->DEPTH>2)
            {
                char Confirm;
                std::cout<<">> [Warning] Seq2Seq with two more layers may not work well,do you still want to run this model?(y/n)"<<std::endl;
                std::cin>>Confirm;
                if(Confirm!='y')
                {
                    std::cout<<">> [Error] Running process cancelled"<<std::endl;
                    return;
                }
            }
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            DeepSeq2Seq MainSeq("gru",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->DEPTH,Node->getObjPointer()->MAXTIME);
            MainSeq.SetFunction(Node->getObjPointer()->Function);
            MainSeq.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainSeq.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainSeq.TotalWork("gru",Node->getObjPointer()->FileName_1,
                                    Node->getObjPointer()->FileName_2,
                                    Node->getObjPointer()->FileName_3,
                                    Node->getObjPointer()->FileName_7,
                                    Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==9)
        {
            std::cout<<">> [Running] RNN seq2vec(Normal neural network)"<<std::endl;
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            NormalSeq2Vec MainVec("rnn",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->MAXTIME);
            MainVec.SetFunction(Node->getObjPointer()->Function);
            MainVec.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainVec.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainVec.TotalWork("rnn",Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_2,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==10)
        {
            std::cout<<">> [Running] RNN seq2vec(Deep neural network)"<<std::endl;
            if(Node->getObjPointer()->DEPTH>2)
            {
                char Confirm;
                std::cout<<">> [Warning] Seq2Vec with two more layers may not work well,do you still want to run this model?(y/n)"<<std::endl;
                std::cin>>Confirm;
                if(Confirm!='y')
                {
                    std::cout<<">> [Error] Running process cancelled"<<std::endl;
                    return;
                }
            }
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            DeepSeq2Vec MainVec("rnn",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->DEPTH,Node->getObjPointer()->MAXTIME);
            MainVec.SetFunction(Node->getObjPointer()->Function);
            MainVec.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainVec.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainVec.TotalWork("rnn",Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_2,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==11)
        {
            std::cout<<">> [Running] LSTM seq2vec(Normal neural network)"<<std::endl;
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            NormalSeq2Vec MainVec("lstm",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->MAXTIME);
            MainVec.SetFunction(Node->getObjPointer()->Function);
            MainVec.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainVec.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainVec.TotalWork("lstm",Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_2,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==12)
        {
            std::cout<<">> [Running] LSTM seq2vec(Deep neural network)"<<std::endl;
            if(Node->getObjPointer()->DEPTH>2)
            {
                char Confirm;
                std::cout<<">> [Warning] Seq2Vec with two more layers may not work well,do you still want to run this model?(y/n)"<<std::endl;
                std::cin>>Confirm;
                if(Confirm!='y')
                {
                    std::cout<<">> [Error] Running process cancelled"<<std::endl;
                    return;
                }
            }
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            DeepSeq2Vec MainVec("lstm",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->DEPTH,Node->getObjPointer()->MAXTIME);
            MainVec.SetFunction(Node->getObjPointer()->Function);
            MainVec.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainVec.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainVec.TotalWork("lstm",Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_2,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==13)
        {
            std::cout<<">> [Running] GRU seq2vec(Normal neural network)"<<std::endl;
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            NormalSeq2Vec MainVec("gru",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->MAXTIME);
            MainVec.SetFunction(Node->getObjPointer()->Function);
            MainVec.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainVec.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainVec.TotalWork("gru",Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_2,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==14)
        {
            std::cout<<">> [Running] GRU seq2vec(Deep neural network)"<<std::endl;
            if(Node->getObjPointer()->DEPTH>2)
            {
                char Confirm;
                std::cout<<">> [Warning] Seq2Vec with two more layers may not work well,do you still want to run this model?(y/n)"<<std::endl;
                std::cin>>Confirm;
                if(Confirm!='y')
                {
                    std::cout<<">> [Error] Running process cancelled"<<std::endl;
                    return;
                }
            }
            if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            DeepSeq2Vec MainVec("gru",Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->DEPTH,Node->getObjPointer()->MAXTIME);
            MainVec.SetFunction(Node->getObjPointer()->Function);
            MainVec.SetLearningRate(Node->getObjPointer()->LearningRate);
            MainVec.SetBatchSize(Node->getObjPointer()->BatchSize);
            MainVec.TotalWork("gru",Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_2,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
        }
        else if(Node->getObjPointer()->NetworkType==15)
        {
            std::cout<<">> [Running] BP char2vec(Normal neural network)"<<std::endl;
            if(!fopen(Node->getObjPointer()->FileName_8,"r"))
            {
                std::cout<<">> [Error] Cannot open file."<<std::endl;
                std::cout<<">> [Lack] "<<Node->getObjPointer()->FileName_8<<std::endl;
                return;
            }
            Char2Vec MainVec(Node->getObjPointer()->HNUM);
            MainVec.TotalWork(Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_8);
        }
        else
            std::cout<<">> [Error] Unknown Type"<<std::endl;
        return;
    }
    std::cout<<">> [Error] This project does not exist."<<std::endl;
    return;
}

void ObjManager::PrintAllObj()
{
    UserObject *Node=Head;
    if(Head->p==NULL)
    {
        std::cout<<">> [Error] Empty list.(0 project inside)"<<std::endl;
        return;
    }
    while(Node->p!=NULL)
    {
        Node=Node->p;
        Node->PrintObj();
    }
    return;
}

void ObjManager::FindObj()
{
    UserObject *Node=Head;
    std::string temp_obj_name;
    std::cout<<">> Name of the project: ";
    std::cin>>temp_obj_name;
    while(Node->p!=NULL)
    {
        Node=Node->p;
        if(Node->CheckObjName(temp_obj_name))
        {
            Node->PrintObj();
            return;
        }
    }
    std::cout<<">> [Error] This project does not exist."<<std::endl;
}

void ObjManager::EditObj()
{
    UserObject *Node=Head;
    ObjElement Temp;
    std::string temp_obj_name;
    std::cout<<">> Name of the project: ";
    std::cin>>temp_obj_name;
    while(Node->p!=NULL)
    {
        Node=Node->p;
        if(Node->CheckObjName(temp_obj_name))
        {
            Node->PrintObj();
            std::cout<<">> [Editing]"<<std::endl;
            std::cout<<"   ------------------------------------------------------"<<std::endl;
            std::cout<<"   |Name         |";std::cin>>Temp.ObjName;
            std::cout<<"   |File 1       |";std::cin>>Temp.FileName_1;
            std::cout<<"   |File 2       |";std::cin>>Temp.FileName_2;
            std::cout<<"   |File 3       |";std::cin>>Temp.FileName_3;
            std::cout<<"   |File 4(NULL) |";std::cin>>Temp.FileName_4;
            std::cout<<"   |File 5(NULL) |";std::cin>>Temp.FileName_5;
            std::cout<<"   |File 6(NULL) |";std::cin>>Temp.FileName_6;
            std::cout<<"   |Input File   |";std::cin>>Temp.FileName_7;
            std::cout<<"   |Train File   |";std::cin>>Temp.FileName_8;
            std::cout<<"   |Function     |";std::cin>>Temp.Function;
            std::cout<<"   |INUM         |";std::cin>>Temp.INUM;
            std::cout<<"   |HNUM         |";std::cin>>Temp.HNUM;
            std::cout<<"   |ONUM         |";std::cin>>Temp.ONUM;
            std::cout<<"   |DEPTH        |";std::cin>>Temp.DEPTH;
            std::cout<<"   |MAXTIME      |";std::cin>>Temp.MAXTIME;
            std::cout<<"   |LearningRate |";std::cin>>Temp.LearningRate;
            std::cout<<"   |Batch Size   |";std::cin>>Temp.BatchSize;
            std::cout<<"   |Network Type |You Cannot Edit This."<<std::endl;Temp.NetworkType=Node->getObjPointer()->NetworkType;
            std::cout<<"   ------------------------------------------------------"<<std::endl;
            Node->ObjChange(Temp);
            ObjDataOut();
            return;
        }
    }
    std::cout<<">> [Error] This project does not exist."<<std::endl;
}

void ObjManager::DeleteObj()
{
    std::string DelObjName;
    std::cout<<">> Input the name of the project you want to delete: ";
    std::cin>>DelObjName;
    UserObject *Node=Head;
    UserObject *Temp;
    while(Node->p!=NULL)
    {
        Temp=Node;
        Node=Node->p;
        if(Node->CheckObjName(DelObjName))
        {
            Temp->p=Node->p;
            delete Node;
            ObjDataOut();
            std::cout<<">> Finished.(But the data must be deleted by yourself!)"<<std::endl;
            return;
        }
    }
    std::cout<<">> [Error] Cannot find this project."<<std::endl;
    return;
}

void ObjManager::ChangeLearningRate()
{
    UserObject *Node=Head;
    double Temp_Learning_Rate;
    std::string temp_obj_name;
    std::cout<<">> Name of the project: ";
    std::cin>>temp_obj_name;
    while(Node->p!=NULL)
    {
        Node=Node->p;
        if(Node->CheckObjName(temp_obj_name))
        {
            Node->PrintObj();
            std::cout<<"   |LearningRate |";std::cin>>Temp_Learning_Rate;
            Node->getObjPointer()->LearningRate=Temp_Learning_Rate;
            ObjDataOut();
            return;
        }
    }
    std::cout<<">> [Error] This project does not exist."<<std::endl;
}

void ObjManager::ChangeBatchSize()
{
    UserObject *Node=Head;
    int Temp_Batch_size;
    std::string temp_obj_name;
    std::cout<<">> Name of the project: ";
    std::cin>>temp_obj_name;
    while(Node->p!=NULL)
    {
        Node=Node->p;
        if(Node->CheckObjName(temp_obj_name))
        {
            Node->PrintObj();
            std::cout<<"   |Batch Size   |";std::cin>>Temp_Batch_size;
            Node->getObjPointer()->BatchSize=Temp_Batch_size;
            ObjDataOut();
            return;
        }
    }
    std::cout<<">> [Error] This project does not exist."<<std::endl;
}

void ObjManager::FindSpecialObj(const char *Typename)
{
    if(strcmp(Typename,"bp")&&strcmp(Typename,"rnn")&&strcmp(Typename,"lstm")&&strcmp(Typename,"gru"))
    {
        std::cout<<">> [Error] Undefined type."<<std::endl;
        return;
    }
    UserObject *Node=Head;
    while(Node->p!=NULL)
    {
        Node=Node->p;
        if(strcmp(Typename,"bp")==0)
        {
            if(Node->getObjPointer()->NetworkType==1||Node->getObjPointer()->NetworkType==2||Node->getObjPointer()->NetworkType==15)
                Node->PrintObj();
        }
        else if(strcmp(Typename,"rnn")==0)
        {
            if(Node->getObjPointer()->NetworkType==3||Node->getObjPointer()->NetworkType==4||Node->getObjPointer()->NetworkType==9||Node->getObjPointer()->NetworkType==10)
                Node->PrintObj();
        }
        else if(strcmp(Typename,"lstm")==0)
        {
            if(Node->getObjPointer()->NetworkType==5||Node->getObjPointer()->NetworkType==6||Node->getObjPointer()->NetworkType==11||Node->getObjPointer()->NetworkType==12)
                Node->PrintObj();
        }
        else if(strcmp(Typename,"gru")==0)
        {
            if(Node->getObjPointer()->NetworkType==7||Node->getObjPointer()->NetworkType==8||Node->getObjPointer()->NetworkType==13||Node->getObjPointer()->NetworkType==14)
                Node->PrintObj();
        }
    }
    std::cout<<">> [End] End of the list."<<std::endl;
    return;
}

#endif
