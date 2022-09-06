/*mainassist.h header file by ValK*/
/*2019/5/9             version 1.1*/
#ifndef MAINASSIST_H
#define MAINASSIST_H
#include "NLPann.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
using namespace std;

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
			cout<<"   ------------------------------------------------------"<<endl;
			cout<<"   |Name         |"<<Obj.ObjName<<endl;
			if(strcmp(Obj.FileName_1,"NULL"))
				cout<<"   |File 1       |"<<Obj.FileName_1<<endl;
			if(strcmp(Obj.FileName_2,"NULL"))
				cout<<"   |File 2       |"<<Obj.FileName_2<<endl;
			if(strcmp(Obj.FileName_3,"NULL"))
				cout<<"   |File 3       |"<<Obj.FileName_3<<endl;
			if(strcmp(Obj.FileName_4,"NULL"))
				cout<<"   |File 4       |"<<Obj.FileName_4<<endl;
			if(strcmp(Obj.FileName_5,"NULL"))
				cout<<"   |File 5       |"<<Obj.FileName_5<<endl;
			if(strcmp(Obj.FileName_6,"NULL"))
				cout<<"   |File 6       |"<<Obj.FileName_6<<endl;
			if(strcmp(Obj.FileName_7,"NULL"))
				cout<<"   |File 7       |"<<Obj.FileName_7<<endl;
			if(strcmp(Obj.FileName_8,"NULL"))
				cout<<"   |File 8       |"<<Obj.FileName_8<<endl;
			cout<<"   |Function     |"<<Obj.Function<<endl;
			cout<<"   |INUM         |"<<Obj.INUM<<endl;
			cout<<"   |HNUM         |"<<Obj.HNUM<<endl;
			cout<<"   |ONUM         |"<<Obj.ONUM<<endl;
			cout<<"   |DEPTH        |"<<Obj.DEPTH<<endl;
			cout<<"   |MAXTIME      |"<<Obj.MAXTIME<<endl;
			cout<<"   |LearningRate |"<<Obj.LearningRate<<endl;
			cout<<"   |Batch Size   |"<<Obj.BatchSize<<endl;
			cout<<"   |Network Type |";
			switch(Obj.NetworkType)
			{
				case 1:
					cout<<"BP(Normal neural network)"<<endl;
					break;
				case 2:
					cout<<"BP(Deep neural network)"<<endl;
					break;
				case 3:
					cout<<"RNN seq2seq(Normal neural network)"<<endl;
					break;
				case 4:
					cout<<"RNN seq2seq(Deep neural network)"<<endl;
					break;
				case 5:
					cout<<"LSTM seq2seq(Normal neural network)"<<endl;
					break;
				case 6:
					cout<<"LSTM seq2seq(Deep neural network)"<<endl;
					break;
				case 7:
					cout<<"GRU seq2seq(Normal neural network)"<<endl;
					break;
				case 8:
					cout<<"GRU seq2seq(Deep neural network)"<<endl;
					break;
				case 9:
					cout<<"RNN seq2vec(Normal neural network)"<<endl;
					break;
				case 10:
					cout<<"RNN seq2vec(Deep neural network)"<<endl;
					break;
				case 11:
					cout<<"LSTM seq2vec(Normal neural network)"<<endl;
					break;
				case 12:
					cout<<"LSTM seq2vec(Deep neural network)"<<endl;
					break;
				case 13:
					cout<<"GRU seq2vec(Normal neural network)"<<endl;
					break;
				case 14:
					cout<<"GRU seq2vec(Deep neural network)"<<endl;
					break;
				case 15:
					cout<<"BP char2vec(Normal neural network)"<<endl;
					break;
				default:
					cout<<"Unknown Type"<<endl;
					break;
			}
			cout<<"   ------------------------------------------------------"<<endl;
		}
		ObjElement* getObjPointer()
		{
			return &Obj;
		}
		bool CheckObjName(string &TempName)
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
		cout<<">> [init] Initializing completed."<<endl;
	}
	ifstream fin("ObjData.dat",std::ios::binary);
	if(fin.fail())
	{
		cout<<">> [Error] Cannot open important data \"ObjData.dat\" or this data maybe lost!"<<endl;
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
	ofstream fout("ObjData.dat",std::ios::binary);
	if(fout.fail())
	{
		cout<<">> [Error] Cannot open important data \"Objdata.dat\" or this data maybe lost!"<<endl;
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
	cout<<">> Name of your project: ";
	cin>>Datatemp.ObjName;
	while(Node->p!=NULL)
	{
		Node=Node->p;
		if(Node->CheckObjName(Datatemp.ObjName))
		{
			cout<<">> [Error] You have already created this project!"<<endl<<endl;
			cout<<"   |The project is:"<<endl;
			Node->PrintObj();
			return;
		}
	}
	if(!ObjChoose())
	{
		cout<<">> [Quiting]"<<endl;
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
	cout<<">> New project is established successfully."<<endl;
}

bool ObjManager::ObjChoose()
{
	string Command;
	cout<<">>"<<endl;
	cout<<"   |1. |BP project(Normal)            |"<<endl;
	cout<<"   |2. |BP project(Deep)              |"<<endl;
	cout<<"   |3. |seq2seq project(RNN:Normal)   |"<<endl;
	cout<<"   |4. |seq2seq project(RNN:Deep)     |"<<endl;
	cout<<"   |5. |seq2seq project(LSTM:Normal)  |"<<endl;
	cout<<"   |6. |seq2seq project(LSTM:Deep)    |"<<endl;
	cout<<"   |7. |seq2seq project(GRU:Normal)   |"<<endl;
	cout<<"   |8. |seq2seq project(GRU:Deep)     |"<<endl;
	cout<<"   |9. |seq2vec project(RNN:Normal)   |"<<endl;
	cout<<"   |10.|seq2vec project(RNN:Deep)     |"<<endl;
	cout<<"   |11.|seq2vec project(LSTM:Normal)  |"<<endl;
	cout<<"   |12.|seq2vec project(LSTM:Deep)    |"<<endl;
	cout<<"   |13.|seq2vec project(GRU:Normal)   |"<<endl;
	cout<<"   |14.|seq2vec project(GRU:Deep)     |"<<endl;
	cout<<"   |15.|char2vec project(BP:Normal)   |"<<endl;
	cout<<"   |16.|I don't want to choose.(quit) |"<<endl;
	while(1)
	{
		cout<<">> [Choice] Input your choice: ";
		cin>>Command;
		if(Command=="1")
		{
			cout<<">>"<<endl;
			cout<<"   |BP project(Normal) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for all layers)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the output file: ";
			cin>>Datatemp.FileName_1;
			strcpy(Datatemp.FileName_2,"NULL");
			strcpy(Datatemp.FileName_3,"NULL");
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function: ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.DEPTH=0;
			Datatemp.MAXTIME=0;
			Datatemp.NetworkType=1;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="2")
		{
			cout<<">>"<<endl;
			cout<<"   |BP project(Deep) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |DEPTH: Number of layers"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for all layers)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the output file: ";
			cin>>Datatemp.FileName_1;
			strcpy(Datatemp.FileName_2,"NULL");
			strcpy(Datatemp.FileName_3,"NULL");
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function: ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			cout<<">> Number of layers: ";
			cin>>Datatemp.DEPTH;
			Datatemp.MAXTIME=0;
			Datatemp.NetworkType=2;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="3")
		{
			cout<<">> [RNN]"<<endl;
			cout<<"   |seq2seq project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the decoder module file: ";
			cin>>Datatemp.FileName_2;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_3;
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			Datatemp.DEPTH=0;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=3;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="4")
		{
			cout<<">> [RNN]"<<endl;
			cout<<"   |seq2seq project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |DEPTH: Number of layers"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the decoder module file: ";
			cin>>Datatemp.FileName_2;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_3;
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			cout<<">> Number of layers: ";
			cin>>Datatemp.DEPTH;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=4;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="5")
		{
			cout<<">> [LSTM]"<<endl;
			cout<<"   |seq2seq project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the decoder module file: ";
			cin>>Datatemp.FileName_2;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_3;
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			Datatemp.DEPTH=0;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=5;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="6")
		{
			cout<<">> [LSTM]"<<endl;
			cout<<"   |seq2seq project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |DEPTH: Number of layers"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the decoder module file: ";
			cin>>Datatemp.FileName_2;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_3;
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			cout<<">> Number of layers: ";
			cin>>Datatemp.DEPTH;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			Datatemp.NetworkType=6;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="7")
		{
			cout<<">> [GRU]"<<endl;
			cout<<"   |seq2seq project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the decoder module file: ";
			cin>>Datatemp.FileName_2;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_3;
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			Datatemp.DEPTH=0;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=7;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="8")
		{
			cout<<">> [GRU]"<<endl;
			cout<<"   |seq2seq project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |DEPTH: Number of layers"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the decoder module file: ";
			cin>>Datatemp.FileName_2;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_3;
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			cout<<">> Number of layers: ";
			cin>>Datatemp.DEPTH;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=8;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="9")
		{
			cout<<">> [RNN]"<<endl;
			cout<<"   |seq2vec project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_2;
			strcpy(Datatemp.FileName_3,"NULL");
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			Datatemp.DEPTH=0;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=9;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="10")
		{
			cout<<">> [RNN]"<<endl;
			cout<<"   |seq2vec project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |DEPTH: Number of layers"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_2;
			strcpy(Datatemp.FileName_3,"NULL");
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			cout<<">> Number of layers: ";
			cin>>Datatemp.DEPTH;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=10;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="11")
		{
			cout<<">> [LSTM]"<<endl;
			cout<<"   |seq2vec project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_2;
			strcpy(Datatemp.FileName_3,"NULL");
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			Datatemp.DEPTH=0;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=11;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="12")
		{
			cout<<">> [LSTM]"<<endl;
			cout<<"   |seq2vec project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |DEPTH: Number of layers"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_2;
			strcpy(Datatemp.FileName_3,"NULL");
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			cout<<">> Number of layers: ";
			cin>>Datatemp.DEPTH;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=12;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="13")
		{
			cout<<">> [GRU]"<<endl;
			cout<<"   |seq2vec project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_2;
			strcpy(Datatemp.FileName_3,"NULL");
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			Datatemp.DEPTH=0;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=13;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="14")
		{
			cout<<">> [GRU]"<<endl;
			cout<<"   |seq2vec project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons"<<endl;
			cout<<"   |DEPTH: Number of layers"<<endl;
			cout<<"   |MAXTIME: Max length of sequence"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of input set file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions(for output layer)"<<endl;
			cout<<"   |Batch size: Size of input/training set batch"<<endl;
			cout<<"   |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">> Name of the encoder module file: ";
			cin>>Datatemp.FileName_1;
			cout<<">> Name of the output layer file: ";
			cin>>Datatemp.FileName_2;
			strcpy(Datatemp.FileName_3,"NULL");
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			cout<<">> Name of input set file: ";
			cin>>Datatemp.FileName_7;
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			cout<<">> Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>Datatemp.Function;
			cout<<">> Number of input layer neurons: ";
			cin>>Datatemp.INUM;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
			cout<<">> Number of output layer neurons: ";
			cin>>Datatemp.ONUM;
			Datatemp.INUM=27;
			Datatemp.ONUM=27;
			cout<<">> Number of layers: ";
			cin>>Datatemp.DEPTH;
			cout<<">> Max length of sequence: ";
			cin>>Datatemp.MAXTIME;
			Datatemp.NetworkType=14;
			cout<<">> Learning rate(must more than 0 but less than 1): ";
			cin>>Datatemp.LearningRate;
			cout<<">> Size of input/training set batch: ";
			cin>>Datatemp.BatchSize;
			break;
		}
		else if(Command=="15")
		{
			cout<<">>"<<endl;
			cout<<"   |char2vec project(BP:Normal) needs:"<<endl;
			cout<<"   |INUM: Number of input layer neurons is set to 95"<<endl;
			cout<<"   |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"   |ONUM: Number of output layer neurons is set to 95"<<endl;
			cout<<"   |File: Name of data file"<<endl;
			cout<<"   |File: Name of training set file"<<endl;
			cout<<"   |Function: one of activate functions is set as softmax"<<endl;
			cout<<"   |Learning rate:0.1"<<endl;
			cout<<">> Name of the output data file: ";
			cin>>Datatemp.FileName_1;
			strcpy(Datatemp.FileName_2,"NULL");
			strcpy(Datatemp.FileName_3,"NULL");
			strcpy(Datatemp.FileName_4,"NULL");
			strcpy(Datatemp.FileName_5,"NULL");
			strcpy(Datatemp.FileName_6,"NULL");
			strcpy(Datatemp.FileName_7,"NULL");
			cout<<">> Name of training set file: ";
			cin>>Datatemp.FileName_8;
			strcpy(Datatemp.Function,"softmax");
			Datatemp.INUM=95;
			cout<<">> Number of hidden layer neurons: ";
			cin>>Datatemp.HNUM;
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
			cout<<">> [Error] Undefined choice."<<endl;
	}
	return true;
}

void ObjManager::RunModule()
{
	bool FoundObj=false;
	UserObject *Node=Head;
	string temp_obj_name;
	cout<<">> Name of the project: ";
	cin>>temp_obj_name;
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
			cout<<">> [Running] BP(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
				return;
			}
			NormalBP MainBP(Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM);
			MainBP.SetFunction(Node->getObjPointer()->Function);
			MainBP.SetLearningrate(Node->getObjPointer()->LearningRate);
			MainBP.TotalWork(Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
		}
		else if(Node->getObjPointer()->NetworkType==2)
		{
			cout<<">> [Running] BP(Deep neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
				return;
			}
			DeepBP MainBP(Node->getObjPointer()->INUM,Node->getObjPointer()->HNUM,Node->getObjPointer()->ONUM,Node->getObjPointer()->DEPTH);
			MainBP.SetFunction(Node->getObjPointer()->Function);
			MainBP.SetLearningrate(Node->getObjPointer()->LearningRate);
			MainBP.TotalWork(Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_7,Node->getObjPointer()->FileName_8);
		}
		else if(Node->getObjPointer()->NetworkType==3)
		{
			cout<<">> [Running] RNN seq2seq(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] RNN seq2seq(Deep neural network)"<<endl;
			if(Node->getObjPointer()->DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning] Seq2Seq with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error] Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] LSTM seq2seq(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] LSTM seq2seq(Deep neural network)"<<endl;
			if(Node->getObjPointer()->DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning] Seq2Seq with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error] Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] GRU seq2seq(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] GRU seq2seq(Deep neural network)"<<endl;
			if(Node->getObjPointer()->DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning] Seq2Seq with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error] Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] RNN seq2vec(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] RNN seq2vec(Deep neural network)"<<endl;
			if(Node->getObjPointer()->DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning] Seq2Vec with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error] Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] LSTM seq2vec(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] LSTM seq2vec(Deep neural network)"<<endl;
			if(Node->getObjPointer()->DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning] Seq2Vec with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error] Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] GRU seq2vec(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] GRU seq2vec(Deep neural network)"<<endl;
			if(Node->getObjPointer()->DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning] Seq2Vec with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error] Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->FileName_7,"r")||!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_7<<" and "<<Node->getObjPointer()->FileName_8<<endl;
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
			cout<<">> [Running] BP char2vec(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->FileName_8,"r"))
			{
				cout<<">> [Error] Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->FileName_8<<endl;
				return;
			}
			Char2Vec MainVec(Node->getObjPointer()->HNUM);
			MainVec.TotalWork(Node->getObjPointer()->FileName_1,Node->getObjPointer()->FileName_8);
		}
		else
			cout<<">> [Error] Unknown Type"<<endl;
		return;
	}
	cout<<">> [Error] This project does not exist."<<endl;
	return;
}

void ObjManager::PrintAllObj()
{
	UserObject *Node=Head;
	if(Head->p==NULL)
	{
		cout<<">> [Error] Empty list.(0 project inside)"<<endl;
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
	string temp_obj_name;
	cout<<">> Name of the project: ";
	cin>>temp_obj_name;
	while(Node->p!=NULL)
	{
		Node=Node->p;
		if(Node->CheckObjName(temp_obj_name))
		{
			Node->PrintObj();
			return;
		}
	}
	cout<<">> [Error] This project does not exist."<<endl;
}

void ObjManager::EditObj()
{
	UserObject *Node=Head;
	ObjElement Temp;
	string temp_obj_name;
	cout<<">> Name of the project: ";
	cin>>temp_obj_name;
	while(Node->p!=NULL)
	{
		Node=Node->p;
		if(Node->CheckObjName(temp_obj_name))
		{
			Node->PrintObj();
			cout<<">> [Editing]"<<endl;
			cout<<"   ------------------------------------------------------"<<endl;
			cout<<"   |Name         |";cin>>Temp.ObjName;
			cout<<"   |File 1       |";cin>>Temp.FileName_1;
			cout<<"   |File 2       |";cin>>Temp.FileName_2;
			cout<<"   |File 3       |";cin>>Temp.FileName_3;
			cout<<"   |File 4(NULL) |";cin>>Temp.FileName_4;
			cout<<"   |File 5(NULL) |";cin>>Temp.FileName_5;
			cout<<"   |File 6(NULL) |";cin>>Temp.FileName_6;
			cout<<"   |Input File   |";cin>>Temp.FileName_7;
			cout<<"   |Train File   |";cin>>Temp.FileName_8;
			cout<<"   |Function     |";cin>>Temp.Function;
			cout<<"   |INUM         |";cin>>Temp.INUM;
			cout<<"   |HNUM         |";cin>>Temp.HNUM;
			cout<<"   |ONUM         |";cin>>Temp.ONUM;
			cout<<"   |DEPTH        |";cin>>Temp.DEPTH;
			cout<<"   |MAXTIME      |";cin>>Temp.MAXTIME;
			cout<<"   |LearningRate |";cin>>Temp.LearningRate;
			cout<<"   |Batch Size   |";cin>>Temp.BatchSize;
			cout<<"   |Network Type |You Cannot Edit This."<<endl;Temp.NetworkType=Node->getObjPointer()->NetworkType;
			cout<<"   ------------------------------------------------------"<<endl;
			Node->ObjChange(Temp);
			ObjDataOut();
			return;
		}
	}
	cout<<">> [Error] This project does not exist."<<endl;
}

void ObjManager::DeleteObj()
{
	string DelObjName;
	cout<<">> Input the name of the project you want to delete: ";
	cin>>DelObjName;
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
			cout<<">> Finished.(But the data must be deleted by yourself!)"<<endl;
			return;
		}
	}
	cout<<">> [Error] Cannot find this project."<<endl;
	return;
}

void ObjManager::ChangeLearningRate()
{
	UserObject *Node=Head;
	double Temp_Learning_Rate;
	string temp_obj_name;
	cout<<">> Name of the project: ";
	cin>>temp_obj_name;
	while(Node->p!=NULL)
	{
		Node=Node->p;
		if(Node->CheckObjName(temp_obj_name))
		{
			Node->PrintObj();
			cout<<"   |LearningRate |";cin>>Temp_Learning_Rate;
			Node->getObjPointer()->LearningRate=Temp_Learning_Rate;
			ObjDataOut();
			return;
		}
	}
	cout<<">> [Error] This project does not exist."<<endl;
}

void ObjManager::ChangeBatchSize()
{
	UserObject *Node=Head;
	int Temp_Batch_size;
	string temp_obj_name;
	cout<<">> Name of the project: ";
	cin>>temp_obj_name;
	while(Node->p!=NULL)
	{
		Node=Node->p;
		if(Node->CheckObjName(temp_obj_name))
		{
			Node->PrintObj();
			cout<<"   |Batch Size   |";cin>>Temp_Batch_size;
			Node->getObjPointer()->BatchSize=Temp_Batch_size;
			ObjDataOut();
			return;
		}
	}
	cout<<">> [Error] This project does not exist."<<endl;
}

void ObjManager::FindSpecialObj(const char *Typename)
{
	if(strcmp(Typename,"bp")&&strcmp(Typename,"rnn")&&strcmp(Typename,"lstm")&&strcmp(Typename,"gru"))
	{
		cout<<">> [Error] Undefined type."<<endl;
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
	cout<<">> [End] End of the list."<<endl;
	return;
}

#endif
