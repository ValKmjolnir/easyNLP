/*mainassist.h header file by ValK*/
/*2019/5/9             version 1.1*/
#ifndef __MAINASSIST_H__
#define __MAINASSIST_H__
#include "NLPann.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
using namespace std;

struct ObjElement
{
	char __ObjName[40];
	char __FileName_1[40];
	char __FileName_2[40];
	char __FileName_3[40];
	char __FileName_4[40];
	char __FileName_5[40];
	char __FileName_6[40];
	char __FileName_7[40];
	char __FileName_8[40];
	char __Function[10];
	int __INUM;
	int __HNUM;
	int __ONUM;
	int __DEPTH;
	int __MAXTIME;
	int __NetworkType;
	int __BatchSize;
	double __LearningRate;
};

class UserObject
{
	private:
		ObjElement __Obj;
	public:
		UserObject *__p;
		UserObject(
		const char *__objname="NULL",
		const char *__filename1="NULL",
		const char *__filename2="NULL",
		const char *__filename3="NULL",
		const char *__filename4="NULL",
		const char *__filename5="NULL",
		const char *__filename6="NULL",
		const char *__filename7="NULL",
		const char *__filename8="NULL",
		const char *__function="NULL",
		int __inum=0,
		int __hnum=0,
		int __onum=0,
		int __depth=0,
		int __maxtime=0,
		int __networktype=0,
		int __batchsize=0,
		double __learningrate=0.1
		)
		{
			strcpy(__Obj.__ObjName,__objname);
			strcpy(__Obj.__FileName_1,__filename1);
			strcpy(__Obj.__FileName_2,__filename2);
			strcpy(__Obj.__FileName_3,__filename3);
			strcpy(__Obj.__FileName_4,__filename4);
			strcpy(__Obj.__FileName_5,__filename5);
			strcpy(__Obj.__FileName_6,__filename6);
			strcpy(__Obj.__FileName_7,__filename7);
			strcpy(__Obj.__FileName_8,__filename8);
			strcpy(__Obj.__Function,__function);
			__Obj.__INUM=__inum;
			__Obj.__HNUM=__hnum;
			__Obj.__ONUM=__onum;
			__Obj.__DEPTH=__depth;
			__Obj.__MAXTIME=__maxtime;
			__Obj.__NetworkType=__networktype;
			__Obj.__BatchSize=__batchsize;
			__Obj.__LearningRate=__learningrate;
			__p=NULL;
		}
		void PrintObj()
		{
			cout<<"         ------------------------------------------------------"<<endl;
			cout<<"         |Name         |"<<__Obj.__ObjName<<endl;
			if(strcmp(__Obj.__FileName_1,"NULL"))
				cout<<"         |File 1       |"<<__Obj.__FileName_1<<endl;
			if(strcmp(__Obj.__FileName_2,"NULL"))
				cout<<"         |File 2       |"<<__Obj.__FileName_2<<endl;
			if(strcmp(__Obj.__FileName_3,"NULL"))
				cout<<"         |File 3       |"<<__Obj.__FileName_3<<endl;
			if(strcmp(__Obj.__FileName_4,"NULL"))
				cout<<"         |File 4       |"<<__Obj.__FileName_4<<endl;
			if(strcmp(__Obj.__FileName_5,"NULL"))
				cout<<"         |File 5       |"<<__Obj.__FileName_5<<endl;
			if(strcmp(__Obj.__FileName_6,"NULL"))
				cout<<"         |File 6       |"<<__Obj.__FileName_6<<endl;
			if(strcmp(__Obj.__FileName_7,"NULL"))
				cout<<"         |File 7       |"<<__Obj.__FileName_7<<endl;
			if(strcmp(__Obj.__FileName_8,"NULL"))
				cout<<"         |File 8       |"<<__Obj.__FileName_8<<endl;
			cout<<"         |Function     |"<<__Obj.__Function<<endl;
			cout<<"         |INUM         |"<<__Obj.__INUM<<endl;
			cout<<"         |HNUM         |"<<__Obj.__HNUM<<endl;
			cout<<"         |ONUM         |"<<__Obj.__ONUM<<endl;
			cout<<"         |DEPTH        |"<<__Obj.__DEPTH<<endl;
			cout<<"         |MAXTIME      |"<<__Obj.__MAXTIME<<endl;
			cout<<"         |LearningRate |"<<__Obj.__LearningRate<<endl;
			cout<<"         |Batch Size   |"<<__Obj.__BatchSize<<endl;
			cout<<"         |Network Type |";
			switch(__Obj.__NetworkType)
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
			cout<<"         ------------------------------------------------------"<<endl;
		}
		ObjElement* getObjPointer()
		{
			return &__Obj;
		}
		bool CheckObjName(string &TempName)
		{
			if(TempName==__Obj.__ObjName)
				return true;
			else
				return false;
		}
		bool CheckObjName(const char *__t)
		{
			if(!strcmp(__t,__Obj.__ObjName))
				return true;
			else
				return false;
		}
		void ObjChange(ObjElement &__temp)
		{
			__Obj=__temp;
			return;
		}
};


class ObjManager
{
	private:
		UserObject *__Head;
		ObjElement __Datatemp;
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
	__Head=new UserObject;
	__Head->__p=NULL;
}

ObjManager::~ObjManager()
{
	UserObject *Node=__Head;
	UserObject *Temp;
	while(Node->__p!=NULL)
	{
		Temp=Node;
		Node=Node->__p;
		delete Temp;
	}
	delete Node;
}

void ObjManager::ObjDataIn()
{
	UserObject *Node=__Head;
	if(!fopen("ObjData.dat","r"))
	{
		FILE* fd=fopen("ObjData.dat","w");
		fclose(fd);
		cout<<">> [init] Initializing completed."<<endl;
	}
	fstream fin("ObjData.dat",ios::in|ios::binary);
	if(fin.fail())
	{
		cout<<">> [Error]Cannot open important data \"ObjData.dat\" or this data maybe lost!"<<endl;
		exit(-1);
	}
	while(!fin.eof())
	{
		fin.read((char*)&__Datatemp,sizeof(ObjElement));
		if(fin.eof())
			break;
		Node->__p=new UserObject(
		__Datatemp.__ObjName,
		__Datatemp.__FileName_1,
		__Datatemp.__FileName_2,
		__Datatemp.__FileName_3,
		__Datatemp.__FileName_4,
		__Datatemp.__FileName_5,
		__Datatemp.__FileName_6,
		__Datatemp.__FileName_7,
		__Datatemp.__FileName_8,
		__Datatemp.__Function,
		__Datatemp.__INUM,
		__Datatemp.__HNUM,
		__Datatemp.__ONUM,
		__Datatemp.__DEPTH,
		__Datatemp.__MAXTIME,
		__Datatemp.__NetworkType,
		__Datatemp.__BatchSize,
		__Datatemp.__LearningRate
		);
		Node=Node->__p;
		Node->__p=NULL;
	}
	fin.close();
	return;
}

void ObjManager::ObjDataOut()
{
	UserObject *Node=__Head;
	fstream fout("ObjData.dat",ios::out|ios::binary);
	if(fout.fail())
	{
		cout<<">> [Error]Cannot open important data \"Objdata.dat\" or this data maybe lost!"<<endl;
		exit(-1);
	}
	while(Node->__p!=NULL)
	{
		Node=Node->__p;
		fout.write((char*)Node->getObjPointer(),sizeof(ObjElement));
	}
	fout.close(); 
	return;
}

void ObjManager::MakeData()
{
	UserObject *Node=__Head;
	cout<<">>Name of your project: ";
	cin>>__Datatemp.__ObjName;
	while(Node->__p!=NULL)
	{
		Node=Node->__p;
		if(Node->CheckObjName(__Datatemp.__ObjName))
		{
			cout<<">> [Error]You have already created this project!"<<endl<<endl;
			cout<<"         |The project is:"<<endl;
			Node->PrintObj();
			return;
		}
	}
	if(!ObjChoose())
	{
		cout<<">> [Quiting]"<<endl;
		return;
	}
	Node->__p=new UserObject(
	__Datatemp.__ObjName,
	__Datatemp.__FileName_1,
	__Datatemp.__FileName_2,
	__Datatemp.__FileName_3,
	__Datatemp.__FileName_4,
	__Datatemp.__FileName_5,
	__Datatemp.__FileName_6,
	__Datatemp.__FileName_7,
	__Datatemp.__FileName_8,
	__Datatemp.__Function,
	__Datatemp.__INUM,
	__Datatemp.__HNUM,
	__Datatemp.__ONUM,
	__Datatemp.__DEPTH,
	__Datatemp.__MAXTIME,
	__Datatemp.__NetworkType,
	__Datatemp.__BatchSize,
	__Datatemp.__LearningRate
	);
	Node=Node->__p;
	Node->__p=NULL;
	cout<<">>New project is established successfully."<<endl;
}

bool ObjManager::ObjChoose()
{
	string Command;
	cout<<">>"<<endl;
	cout<<"         |1. |BP project(Normal)            |"<<endl;
	cout<<"         |2. |BP project(Deep)              |"<<endl;
	cout<<"         |3. |seq2seq project(RNN:Normal)   |"<<endl;
	cout<<"         |4. |seq2seq project(RNN:Deep)     |"<<endl;
	cout<<"         |5. |seq2seq project(LSTM:Normal)  |"<<endl;
	cout<<"         |6. |seq2seq project(LSTM:Deep)    |"<<endl;
	cout<<"         |7. |seq2seq project(GRU:Normal)   |"<<endl;
	cout<<"         |8. |seq2seq project(GRU:Deep)     |"<<endl;
	cout<<"         |9. |seq2vec project(RNN:Normal)   |"<<endl;
	cout<<"         |10.|seq2vec project(RNN:Deep)     |"<<endl;
	cout<<"         |11.|seq2vec project(LSTM:Normal)  |"<<endl;
	cout<<"         |12.|seq2vec project(LSTM:Deep)    |"<<endl;
	cout<<"         |13.|seq2vec project(GRU:Normal)   |"<<endl;
	cout<<"         |14.|seq2vec project(GRU:Deep)     |"<<endl;
	cout<<"         |15.|char2vec project(BP:Normal)   |"<<endl;
	cout<<"         |16.|I don't want to choose.(quit) |"<<endl;
	while(1)
	{
		cout<<">> [Choice]Input your choice: ";
		cin>>Command;
		if(Command=="1")
		{
			cout<<">>"<<endl;
			cout<<"         |BP project(Normal) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for all layers)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the output file: ";
			cin>>__Datatemp.__FileName_1;
			strcpy(__Datatemp.__FileName_2,"NULL");
			strcpy(__Datatemp.__FileName_3,"NULL");
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function: ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__DEPTH=0;
			__Datatemp.__MAXTIME=0;
			__Datatemp.__NetworkType=1;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="2")
		{
			cout<<">>"<<endl;
			cout<<"         |BP project(Deep) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |DEPTH: Number of layers"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for all layers)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the output file: ";
			cin>>__Datatemp.__FileName_1;
			strcpy(__Datatemp.__FileName_2,"NULL");
			strcpy(__Datatemp.__FileName_3,"NULL");
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function: ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			cout<<">>Number of layers: ";
			cin>>__Datatemp.__DEPTH;
			__Datatemp.__MAXTIME=0;
			__Datatemp.__NetworkType=2;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="3")
		{
			cout<<">> [RNN]"<<endl;
			cout<<"         |seq2seq project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the decoder module file: ";
			cin>>__Datatemp.__FileName_2;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_3;
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			__Datatemp.__DEPTH=0;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=3;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="4")
		{
			cout<<">> [RNN]"<<endl;
			cout<<"         |seq2seq project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |DEPTH: Number of layers"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the decoder module file: ";
			cin>>__Datatemp.__FileName_2;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_3;
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			cout<<">>Number of layers: ";
			cin>>__Datatemp.__DEPTH;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=4;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="5")
		{
			cout<<">> [LSTM]"<<endl;
			cout<<"         |seq2seq project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the decoder module file: ";
			cin>>__Datatemp.__FileName_2;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_3;
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			__Datatemp.__DEPTH=0;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=5;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="6")
		{
			cout<<">> [LSTM]"<<endl;
			cout<<"         |seq2seq project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |DEPTH: Number of layers"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the decoder module file: ";
			cin>>__Datatemp.__FileName_2;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_3;
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			cout<<">>Number of layers: ";
			cin>>__Datatemp.__DEPTH;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			__Datatemp.__NetworkType=6;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="7")
		{
			cout<<">> [GRU]"<<endl;
			cout<<"         |seq2seq project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the decoder module file: ";
			cin>>__Datatemp.__FileName_2;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_3;
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			__Datatemp.__DEPTH=0;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=7;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="8")
		{
			cout<<">> [GRU]"<<endl;
			cout<<"         |seq2seq project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |DEPTH: Number of layers"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the decoder module file: ";
			cin>>__Datatemp.__FileName_2;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_3;
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2seq uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			cout<<">>Number of layers: ";
			cin>>__Datatemp.__DEPTH;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=8;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="9")
		{
			cout<<">> [RNN]"<<endl;
			cout<<"         |seq2vec project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_2;
			strcpy(__Datatemp.__FileName_3,"NULL");
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			__Datatemp.__DEPTH=0;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=9;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="10")
		{
			cout<<">> [RNN]"<<endl;
			cout<<"         |seq2vec project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |DEPTH: Number of layers"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_2;
			strcpy(__Datatemp.__FileName_3,"NULL");
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			cout<<">>Number of layers: ";
			cin>>__Datatemp.__DEPTH;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=10;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="11")
		{
			cout<<">> [LSTM]"<<endl;
			cout<<"         |seq2vec project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_2;
			strcpy(__Datatemp.__FileName_3,"NULL");
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			__Datatemp.__DEPTH=0;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=11;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="12")
		{
			cout<<">> [LSTM]"<<endl;
			cout<<"         |seq2vec project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |DEPTH: Number of layers"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_2;
			strcpy(__Datatemp.__FileName_3,"NULL");
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			cout<<">>Number of layers: ";
			cin>>__Datatemp.__DEPTH;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=12;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="13")
		{
			cout<<">> [GRU]"<<endl;
			cout<<"         |seq2vec project(RNN/LSTM/GRU:Normal) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_2;
			strcpy(__Datatemp.__FileName_3,"NULL");
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			__Datatemp.__DEPTH=0;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=13;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="14")
		{
			cout<<">> [GRU]"<<endl;
			cout<<"         |seq2vec project(RNN/LSTM/GRU:Deep) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons"<<endl;
			cout<<"         |DEPTH: Number of layers"<<endl;
			cout<<"         |MAXTIME: Max length of sequence"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of input set file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions(for output layer)"<<endl;
			cout<<"         |Batch size: Size of input/training set batch"<<endl;
			cout<<"         |Learning rate:This decides how fast your model runs"<<endl;
			cout<<">>Name of the encoder module file: ";
			cin>>__Datatemp.__FileName_1;
			cout<<">>Name of the output layer file: ";
			cin>>__Datatemp.__FileName_2;
			strcpy(__Datatemp.__FileName_3,"NULL");
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			cout<<">>Name of input set file: ";
			cin>>__Datatemp.__FileName_7;
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			cout<<">>Name of the activate function(seq2vec uses softmax as output and your function may not work): ";
			cin>>__Datatemp.__Function;
			cout<<">>Number of input layer neurons: ";
			cin>>__Datatemp.__INUM;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			cout<<">>Number of output layer neurons: ";
			cin>>__Datatemp.__ONUM;
			__Datatemp.__INUM=27;
			__Datatemp.__ONUM=27;
			cout<<">>Number of layers: ";
			cin>>__Datatemp.__DEPTH;
			cout<<">>Max length of sequence: ";
			cin>>__Datatemp.__MAXTIME;
			__Datatemp.__NetworkType=14;
			cout<<">>Learning rate(must more than 0 but less than 1): ";
			cin>>__Datatemp.__LearningRate;
			cout<<">>Size of input/training set batch: ";
			cin>>__Datatemp.__BatchSize;
			break;
		}
		else if(Command=="15")
		{
			cout<<">>"<<endl;
			cout<<"         |char2vec project(BP:Normal) needs:"<<endl;
			cout<<"         |INUM: Number of input layer neurons is set to 95"<<endl;
			cout<<"         |HNUM: Number of hidden layer neurons"<<endl;
			cout<<"         |ONUM: Number of output layer neurons is set to 95"<<endl;
			cout<<"         |File: Name of data file"<<endl;
			cout<<"         |File: Name of training set file"<<endl;
			cout<<"         |Function: one of activate functions is set as softmax"<<endl;
			cout<<"         |Learning rate:0.1"<<endl;
			cout<<">>Name of the output data file: ";
			cin>>__Datatemp.__FileName_1;
			strcpy(__Datatemp.__FileName_2,"NULL");
			strcpy(__Datatemp.__FileName_3,"NULL");
			strcpy(__Datatemp.__FileName_4,"NULL");
			strcpy(__Datatemp.__FileName_5,"NULL");
			strcpy(__Datatemp.__FileName_6,"NULL");
			strcpy(__Datatemp.__FileName_7,"NULL");
			cout<<">>Name of training set file: ";
			cin>>__Datatemp.__FileName_8;
			strcpy(__Datatemp.__Function,"softmax");
			__Datatemp.__INUM=95;
			cout<<">>Number of hidden layer neurons: ";
			cin>>__Datatemp.__HNUM;
			__Datatemp.__ONUM=95;
			__Datatemp.__DEPTH=0;
			__Datatemp.__MAXTIME=0;
			__Datatemp.__NetworkType=15;
			__Datatemp.__LearningRate=0.1;
			__Datatemp.__BatchSize=0;
			break;
		}
		else if(Command=="16")
		{
			return false;
		}
		else
			cout<<">> [Error]Undefined choice."<<endl;
	}
	return true;
}

void ObjManager::RunModule()
{
	bool __FoundObj=false;
	UserObject *Node=__Head;
	string temp_obj_name;
	cout<<">>Name of the project: ";
	cin>>temp_obj_name;
	while(Node->__p!=NULL)
	{
		Node=Node->__p;
		if(Node->CheckObjName(temp_obj_name))
		{
			Node->PrintObj();
			__FoundObj=true;
			break;
		}
	}
	if(__FoundObj)
	{
		if(Node->getObjPointer()->__NetworkType==1)
		{
			cout<<">> [Running]BP(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			NormalBP __MainBP(Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM);
			__MainBP.SetFunction(Node->getObjPointer()->__Function);
			__MainBP.SetLearningrate(Node->getObjPointer()->__LearningRate);
			__MainBP.TotalWork(Node->getObjPointer()->__FileName_1,Node->getObjPointer()->__FileName_7,Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==2)
		{
			cout<<">> [Running]BP(Deep neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			DeepBP __MainBP(Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__DEPTH);
			__MainBP.SetFunction(Node->getObjPointer()->__Function);
			__MainBP.SetLearningrate(Node->getObjPointer()->__LearningRate);
			__MainBP.TotalWork(Node->getObjPointer()->__FileName_1,Node->getObjPointer()->__FileName_7,Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==3)
		{
			cout<<">> [Running]RNN seq2seq(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			NormalSeq2Seq __MainSeq("rnn",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__MAXTIME);
			__MainSeq.SetFunction(Node->getObjPointer()->__Function);
			__MainSeq.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainSeq.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainSeq.TotalWork("rnn",Node->getObjPointer()->__FileName_1,
									Node->getObjPointer()->__FileName_2,
									Node->getObjPointer()->__FileName_3,
									Node->getObjPointer()->__FileName_7,
									Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==4)
		{
			cout<<">> [Running]RNN seq2seq(Deep neural network)"<<endl;
			if(Node->getObjPointer()->__DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning]Seq2Seq with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error]Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			DeepSeq2Seq __MainSeq("rnn",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__DEPTH,Node->getObjPointer()->__MAXTIME);
			__MainSeq.SetFunction(Node->getObjPointer()->__Function);
			__MainSeq.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainSeq.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainSeq.TotalWork("rnn",Node->getObjPointer()->__FileName_1,
									Node->getObjPointer()->__FileName_2,
									Node->getObjPointer()->__FileName_3,
									Node->getObjPointer()->__FileName_7,
									Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==5)
		{
			cout<<">> [Running]LSTM seq2seq(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			NormalSeq2Seq __MainSeq("lstm",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__MAXTIME);
			__MainSeq.SetFunction(Node->getObjPointer()->__Function);
			__MainSeq.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainSeq.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainSeq.TotalWork("lstm",Node->getObjPointer()->__FileName_1,
									Node->getObjPointer()->__FileName_2,
									Node->getObjPointer()->__FileName_3,
									Node->getObjPointer()->__FileName_7,
									Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==6)
		{
			cout<<">> [Running]LSTM seq2seq(Deep neural network)"<<endl;
			if(Node->getObjPointer()->__DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning]Seq2Seq with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error]Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			DeepSeq2Seq __MainSeq("lstm",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__DEPTH,Node->getObjPointer()->__MAXTIME);
			__MainSeq.SetFunction(Node->getObjPointer()->__Function);
			__MainSeq.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainSeq.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainSeq.TotalWork("lstm",Node->getObjPointer()->__FileName_1,
									Node->getObjPointer()->__FileName_2,
									Node->getObjPointer()->__FileName_3,
									Node->getObjPointer()->__FileName_7,
									Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==7)
		{
			cout<<">> [Running]GRU seq2seq(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			NormalSeq2Seq __MainSeq("gru",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__MAXTIME);
			__MainSeq.SetFunction(Node->getObjPointer()->__Function);
			__MainSeq.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainSeq.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainSeq.TotalWork("gru",Node->getObjPointer()->__FileName_1,
									Node->getObjPointer()->__FileName_2,
									Node->getObjPointer()->__FileName_3,
									Node->getObjPointer()->__FileName_7,
									Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==8)
		{
			cout<<">> [Running]GRU seq2seq(Deep neural network)"<<endl;
			if(Node->getObjPointer()->__DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning]Seq2Seq with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error]Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			DeepSeq2Seq __MainSeq("gru",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__DEPTH,Node->getObjPointer()->__MAXTIME);
			__MainSeq.SetFunction(Node->getObjPointer()->__Function);
			__MainSeq.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainSeq.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainSeq.TotalWork("gru",Node->getObjPointer()->__FileName_1,
									Node->getObjPointer()->__FileName_2,
									Node->getObjPointer()->__FileName_3,
									Node->getObjPointer()->__FileName_7,
									Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==9)
		{
			cout<<">> [Running]RNN seq2vec(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			NormalSeq2Vec __MainVec("rnn",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__MAXTIME);
			__MainVec.SetFunction(Node->getObjPointer()->__Function);
			__MainVec.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainVec.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainVec.TotalWork("rnn",Node->getObjPointer()->__FileName_1,Node->getObjPointer()->__FileName_2,Node->getObjPointer()->__FileName_7,Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==10)
		{
			cout<<">> [Running]RNN seq2vec(Deep neural network)"<<endl;
			if(Node->getObjPointer()->__DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning]Seq2Vec with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error]Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			DeepSeq2Vec __MainVec("rnn",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__DEPTH,Node->getObjPointer()->__MAXTIME);
			__MainVec.SetFunction(Node->getObjPointer()->__Function);
			__MainVec.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainVec.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainVec.TotalWork("rnn",Node->getObjPointer()->__FileName_1,Node->getObjPointer()->__FileName_2,Node->getObjPointer()->__FileName_7,Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==11)
		{
			cout<<">> [Running]LSTM seq2vec(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			NormalSeq2Vec __MainVec("lstm",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__MAXTIME);
			__MainVec.SetFunction(Node->getObjPointer()->__Function);
			__MainVec.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainVec.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainVec.TotalWork("lstm",Node->getObjPointer()->__FileName_1,Node->getObjPointer()->__FileName_2,Node->getObjPointer()->__FileName_7,Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==12)
		{
			cout<<">> [Running]LSTM seq2vec(Deep neural network)"<<endl;
			if(Node->getObjPointer()->__DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning]Seq2Vec with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error]Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			DeepSeq2Vec __MainVec("lstm",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__DEPTH,Node->getObjPointer()->__MAXTIME);
			__MainVec.SetFunction(Node->getObjPointer()->__Function);
			__MainVec.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainVec.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainVec.TotalWork("lstm",Node->getObjPointer()->__FileName_1,Node->getObjPointer()->__FileName_2,Node->getObjPointer()->__FileName_7,Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==13)
		{
			cout<<">> [Running]GRU seq2vec(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			NormalSeq2Vec __MainVec("gru",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__MAXTIME);
			__MainVec.SetFunction(Node->getObjPointer()->__Function);
			__MainVec.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainVec.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainVec.TotalWork("gru",Node->getObjPointer()->__FileName_1,Node->getObjPointer()->__FileName_2,Node->getObjPointer()->__FileName_7,Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==14)
		{
			cout<<">> [Running]GRU seq2vec(Deep neural network)"<<endl;
			if(Node->getObjPointer()->__DEPTH>2)
			{
				char Confirm;
				cout<<">> [Warning]Seq2Vec with two more layers may not work well,do you still want to run this model?(y/n)"<<endl;
				cin>>Confirm;
				if(Confirm!='y')
				{
					cout<<">> [Error]Running process cancelled"<<endl;
					return;
				}
			}
			if(!fopen(Node->getObjPointer()->__FileName_7,"r")||!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_7<<" and "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			DeepSeq2Vec __MainVec("gru",Node->getObjPointer()->__INUM,Node->getObjPointer()->__HNUM,Node->getObjPointer()->__ONUM,Node->getObjPointer()->__DEPTH,Node->getObjPointer()->__MAXTIME);
			__MainVec.SetFunction(Node->getObjPointer()->__Function);
			__MainVec.SetLearningRate(Node->getObjPointer()->__LearningRate);
			__MainVec.SetBatchSize(Node->getObjPointer()->__BatchSize);
			__MainVec.TotalWork("gru",Node->getObjPointer()->__FileName_1,Node->getObjPointer()->__FileName_2,Node->getObjPointer()->__FileName_7,Node->getObjPointer()->__FileName_8);
		}
		else if(Node->getObjPointer()->__NetworkType==15)
		{
			cout<<">> [Running]BP char2vec(Normal neural network)"<<endl;
			if(!fopen(Node->getObjPointer()->__FileName_8,"r"))
			{
				cout<<">> [Error]Cannot open file."<<endl;
				cout<<">> [Lack] "<<Node->getObjPointer()->__FileName_8<<endl;
				return;
			}
			Char2Vec __MainVec(Node->getObjPointer()->__HNUM);
			__MainVec.TotalWork(Node->getObjPointer()->__FileName_1,Node->getObjPointer()->__FileName_8);
		}
		else
			cout<<">> [Error]Unknown Type"<<endl;
		return;
	}
	cout<<">> [Error]This project does not exist."<<endl;
	return;
}

void ObjManager::PrintAllObj()
{
	UserObject *Node=__Head;
	if(__Head->__p==NULL)
	{
		cout<<">> [Error]Empty list.(0 project inside)"<<endl;
		return;
	}
	while(Node->__p!=NULL)
	{
		Node=Node->__p;
		Node->PrintObj();
	}
	return;
}

void ObjManager::FindObj()
{
	UserObject *Node=__Head;
	string temp_obj_name;
	cout<<">>Name of the project: ";
	cin>>temp_obj_name;
	while(Node->__p!=NULL)
	{
		Node=Node->__p;
		if(Node->CheckObjName(temp_obj_name))
		{
			Node->PrintObj();
			return;
		}
	}
	cout<<">> [Error]This project does not exist."<<endl;
}

void ObjManager::EditObj()
{
	UserObject *Node=__Head;
	ObjElement __Temp;
	string temp_obj_name;
	cout<<">>Name of the project: ";
	cin>>temp_obj_name;
	while(Node->__p!=NULL)
	{
		Node=Node->__p;
		if(Node->CheckObjName(temp_obj_name))
		{
			Node->PrintObj();
			cout<<">> [Editing]"<<endl;
			cout<<"         ------------------------------------------------------"<<endl;
			cout<<"         |Name         |";cin>>__Temp.__ObjName;
			cout<<"         |File 1       |";cin>>__Temp.__FileName_1;
			cout<<"         |File 2       |";cin>>__Temp.__FileName_2;
			cout<<"         |File 3       |";cin>>__Temp.__FileName_3;
			cout<<"         |File 4(NULL) |";cin>>__Temp.__FileName_4;
			cout<<"         |File 5(NULL) |";cin>>__Temp.__FileName_5;
			cout<<"         |File 6(NULL) |";cin>>__Temp.__FileName_6;
			cout<<"         |Input File   |";cin>>__Temp.__FileName_7;
			cout<<"         |Train File   |";cin>>__Temp.__FileName_8;
			cout<<"         |Function     |";cin>>__Temp.__Function;
			cout<<"         |INUM         |";cin>>__Temp.__INUM;
			cout<<"         |HNUM         |";cin>>__Temp.__HNUM;
			cout<<"         |ONUM         |";cin>>__Temp.__ONUM;
			cout<<"         |DEPTH        |";cin>>__Temp.__DEPTH;
			cout<<"         |MAXTIME      |";cin>>__Temp.__MAXTIME;
			cout<<"         |LearningRate |";cin>>__Temp.__LearningRate;
			cout<<"         |Batch Size   |";cin>>__Temp.__BatchSize;
			cout<<"         |Network Type |You Cannot Edit This."<<endl;__Temp.__NetworkType=Node->getObjPointer()->__NetworkType;
			cout<<"         ------------------------------------------------------"<<endl;
			Node->ObjChange(__Temp);
			ObjDataOut();
			return;
		}
	}
	cout<<">> [Error]This project does not exist."<<endl;
}

void ObjManager::DeleteObj()
{
	string DelObjName;
	cout<<">>Input the name of the project you want to delete: ";
	cin>>DelObjName;
	UserObject *Node=__Head;
	UserObject *Temp;
	while(Node->__p!=NULL)
	{
		Temp=Node;
		Node=Node->__p;
		if(Node->CheckObjName(DelObjName))
		{
			Temp->__p=Node->__p;
			delete Node;
			ObjDataOut();
			cout<<">>Finished.(But the data must be deleted by yourself!)"<<endl;
			return;
		}
	}
	cout<<">> [Error]Cannot find this project."<<endl;
	return;
}

void ObjManager::ChangeLearningRate()
{
	UserObject *Node=__Head;
	double __Temp_Learning_Rate;
	string temp_obj_name;
	cout<<">>Name of the project: ";
	cin>>temp_obj_name;
	while(Node->__p!=NULL)
	{
		Node=Node->__p;
		if(Node->CheckObjName(temp_obj_name))
		{
			Node->PrintObj();
			cout<<"         |LearningRate |";cin>>__Temp_Learning_Rate;
			Node->getObjPointer()->__LearningRate=__Temp_Learning_Rate;
			ObjDataOut();
			return;
		}
	}
	cout<<">> [Error]This project does not exist."<<endl;
}

void ObjManager::ChangeBatchSize()
{
	UserObject *Node=__Head;
	int __Temp_Batch_size;
	string temp_obj_name;
	cout<<">>Name of the project: ";
	cin>>temp_obj_name;
	while(Node->__p!=NULL)
	{
		Node=Node->__p;
		if(Node->CheckObjName(temp_obj_name))
		{
			Node->PrintObj();
			cout<<"         |Batch Size   |";cin>>__Temp_Batch_size;
			Node->getObjPointer()->__BatchSize=__Temp_Batch_size;
			ObjDataOut();
			return;
		}
	}
	cout<<">> [Error]This project does not exist."<<endl;
}

void ObjManager::FindSpecialObj(const char *Typename)
{
	if(strcmp(Typename,"bp")&&strcmp(Typename,"rnn")&&strcmp(Typename,"lstm")&&strcmp(Typename,"gru"))
	{
		cout<<">> [Error]Undefined type."<<endl;
		return;
	}
	UserObject *Node=__Head;
	while(Node->__p!=NULL)
	{
		Node=Node->__p;
		if(strcmp(Typename,"bp")==0)
		{
			if(Node->getObjPointer()->__NetworkType==1||Node->getObjPointer()->__NetworkType==2||Node->getObjPointer()->__NetworkType==15)
				Node->PrintObj();
		}
		else if(strcmp(Typename,"rnn")==0)
		{
			if(Node->getObjPointer()->__NetworkType==3||Node->getObjPointer()->__NetworkType==4||Node->getObjPointer()->__NetworkType==9||Node->getObjPointer()->__NetworkType==10)
				Node->PrintObj();
		}
		else if(strcmp(Typename,"lstm")==0)
		{
			if(Node->getObjPointer()->__NetworkType==5||Node->getObjPointer()->__NetworkType==6||Node->getObjPointer()->__NetworkType==11||Node->getObjPointer()->__NetworkType==12)
				Node->PrintObj();
		}
		else if(strcmp(Typename,"gru")==0)
		{
			if(Node->getObjPointer()->__NetworkType==7||Node->getObjPointer()->__NetworkType==8||Node->getObjPointer()->__NetworkType==13||Node->getObjPointer()->__NetworkType==14)
				Node->PrintObj();
		}
	}
	cout<<">> [End]End of the list."<<endl;
	return;
}

#endif
