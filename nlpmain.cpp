#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <ctime>
#include "NLPann.h"
#include "mainassist.h"
using namespace std;

//PrintHelp() can be called if needed
void PrintHelp()
{
	cout<<"easyNLP>> [Help]"<<endl;
	cout<<"          | 1.|Delete a project      |"<<endl;
	cout<<"          | 2.|Create a new project  |"<<endl;
	cout<<"          | 3.|List all projects     |"<<endl;
	cout<<"          | 4.|Run a project         |"<<endl;
	cout<<"          | 5.|Clear the screen      |"<<endl;
	cout<<"          | 6.|Find a project        |"<<endl;
	cout<<"          | 7.|Edit a project        |"<<endl;
	cout<<"          | 8.|Change learningrate   |"<<endl;
	cout<<"          | 9.|Change batch size     |"<<endl;
	cout<<"          |10.|Make data for Seq2Vec |"<<endl;
	cout<<"          |11.|Find projects by type |"<<endl;
	cout<<"          |12.|Quit                  |"<<endl;
	cout<<"easyNLP>> You can find this help with command:\"h\" or \"help\""<<endl;
	return;
}
//PrintWarning() Updated by version
//This function is used to remind users that some functions are not available until next version
void PrintWarning()
{
	cout<<"easyNLP>> [Tips] [easyNLP-2019 version 1.4 by ValK]"<<endl;
	cout<<"          | BP neural networks can deal with all kinds of bp works"<<endl;
	cout<<"          | Char2Vec model is used to calculate the probibility of next character"<<endl;
	cout<<"          | Char2Vec now can deal with character between ASCII 32:' ' and 126:'~'"<<endl;
	cout<<"          | Seq2Seq is used to deal with sequence-like data"<<endl;
	cout<<"          | Seq2Seq now only works on sequences made with characters between ASCII 97:'a' and 122:'z' and ASCII 32:' '"<<endl;
	cout<<"          | Seq2Seq model uses two space "  " as the end of answer sequence so please be careful of your data!"<<endl;
	cout<<"          | Seq2Vec is used to predict a character with a sequence of input"<<endl;
	cout<<"          | Seq2Vec can make new texts beginning with an appropriate sequence you given before"<<endl;
	cout<<"          | Seq2Vec also only works on sequences made up with characters between ASCII 97:'a' and 122:'z' and 32:' '"<<endl;
	cout<<"          | The MAXTIME of Seq2Vec decides the length of input sequence"<<endl;
	cout<<"          | [Warning] Deep Seq2Seq or Seq2Vec may not have more than two layers or models will not work well!"<<endl;
	cout<<"          | [Warning] GRU doesn't work well on Deep Seq2Seq and Seq2Vec"<<endl;
	cout<<"easyNLP>> You can find tips with command:\"t\" or \"tips\""<<endl;
	return;
}

//main()
int main()
{
	ObjManagement __main;
	string Command;
	__main.ObjDataIn();
	PrintWarning();
	PrintHelp();
	while(1)
	{
		cout<<"easyNLP>>";
		cin>>Command;
		if(Command=="h"||Command=="help")
		{
			PrintHelp();
		}
		else if(Command=="t"||Command=="tips")
		{
			PrintWarning();
		}
		else if(Command=="1")
			__main.DeleteObj();
		else if(Command=="2")
		{
			__main.MakeData();
			__main.ObjDataOut();
		}
		else if(Command=="3")
		{
			__main.PrintAllObj();
		}
		else if(Command=="4")
		{
			__main.RunModule();
		}
		else if(Command=="5")
		{
			system("cls");
			PrintHelp();
		}
		else if(Command=="6")
		{
			__main.FindObj();
		}
		else if(Command=="7")
		{
			__main.EditObj();
		}
		else if(Command=="8")
		{
			__main.ChangeLearningRate();
		}
		else if(Command=="9")
		{
			__main.ChangeBatchSize();
		}
		else if(Command=="10")
		{
			int maxtime;
			char Filename[100];
			char Sequencedata[100];
			char Trainingdata[100];
			cout<<"easyNLP>>Please input the name of text data:";
			cin>>Filename;
			cout<<"easyNLP>>Please input the name of sequence data(input data):";
			cin>>Sequencedata;
			cout<<"easyNLP>>Please input the name of training data:";
			cin>>Trainingdata;
			if(!fopen(Filename,"r")||!fopen(Sequencedata,"w")||!fopen(Trainingdata,"w"))
			{
				cout<<"easyNLP>> [Error]Cannot open file."<<endl;
			}
			else
			{
				cout<<"easyNLP>>Please input the length of every input sequence:";
				cin>>maxtime;
				Seq2VecDataMaker(Filename,Sequencedata,Trainingdata,maxtime);
			}
		}
		else if(Command=="11")
		{
			char Typename[100];
			cout<<"easyNLP>>Which type of networks would you like to find?\neasyNLP>>";
			cin>>Typename;
			__main.FindSpecialObj(Typename);
		}
		else if(Command=="12")
		{
			cout<<"easyNLP>> [Quiting]Please wait."<<endl;
			__main.ObjDataOut();
			break;
		}
		else
		{
			cout<<"easyNLP>> [Error]Undefined command."<<endl;
			PrintHelp();
		}
	}
	return 0;
}
