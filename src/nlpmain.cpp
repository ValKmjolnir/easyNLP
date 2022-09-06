#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <ctime>

#include "NLPann.h"
#include "mainassist.h"

//PrintHelp() can be called if needed
void help(){
	std::cout
	<<">> [Help]"<<std::endl
	<<"   | 1.| Delete a project      |"<<std::endl
	<<"   | 2.| Create a new project  |"<<std::endl
	<<"   | 3.| List all projects     |"<<std::endl
	<<"   | 4.| Run a project         |"<<std::endl
	<<"   | 5.| Find a project        |"<<std::endl
	<<"   | 6.| Edit a project        |"<<std::endl
	<<"   | 7.| Change learningrate   |"<<std::endl
	<<"   | 8.| Change batch size     |"<<std::endl
	<<"   | 9.| Make data for Seq2Vec |"<<std::endl
	<<"   |10.| Find projects by type |"<<std::endl
	<<"   |11.| Quit                  |"<<std::endl
	<<">> You can find this help with cmd:\"h\" or \"help\""<<std::endl;
	return;
}

//PrintWarning() Updated by version
//This function is used to remind users that some functions are not available until next version
void warn(){
	std::cout
	<<">> [Tips] [easyNLP-2022 version 1.5 by ValK]"<<std::endl
	<<"   | BP neural networks can deal with all kinds of bp works"<<std::endl
	<<"   | Char2Vec model is used to calculate the probibility of next character"<<std::endl
	<<"   | Char2Vec now can deal with character between ASCII 32:' ' and 126:'~'"<<std::endl
	<<"   | Seq2Seq is used to deal with sequence-like data"<<std::endl
	<<"   | Seq2Seq now only works on sequences made with characters between ASCII 97:'a' and 122:'z' and ASCII 32:' '"<<std::endl
	<<"   | Seq2Seq model uses two space "  " as the end of answer sequence so please be careful of your data!"<<std::endl
	<<"   | Seq2Vec is used to predict a character with a sequence of input"<<std::endl
	<<"   | Seq2Vec can make new texts beginning with an appropriate sequence you given before"<<std::endl
	<<"   | Seq2Vec also only works on sequences made up with characters between ASCII 97:'a' and 122:'z' and 32:' '"<<std::endl
	<<"   | The MAXTIME of Seq2Vec decides the length of input sequence"<<std::endl
	<<"   | [Warning] Deep Seq2Seq or Seq2Vec may not have more than two layers or models will not work well!"<<std::endl
	<<"   | [Warning] GRU doesn't work well on Deep Seq2Seq and Seq2Vec"<<std::endl
	<<">> You can find tips with cmd:\"t\" or \"tips\""<<std::endl;
	return;
}

//main()
int main(){
	ObjManager manager;
	string cmd;
	manager.ObjDataIn();
	warn();
	help();
	while(1){
		std::cout<<">> ";
		cin>>cmd;
		if(cmd=="h"||cmd=="help"){
			help();
		}
		else if(cmd=="t"||cmd=="tips"){
			warn();
		}
		else if(cmd=="1"){
			manager.DeleteObj();
		}else if(cmd=="2"){
			manager.MakeData();
			manager.ObjDataOut();
		}else if(cmd=="3"){
			manager.PrintAllObj();
		}else if(cmd=="4"){
			manager.RunModule();
		}else if(cmd=="5"){
			manager.FindObj();
		}else if(cmd=="6"){
			manager.EditObj();
		}else if(cmd=="7"){
			manager.ChangeLearningRate();
		}else if(cmd=="8"){
			manager.ChangeBatchSize();
		}else if(cmd=="9"){
			int maxtime;
			char Filename[100];
			char Sequencedata[100];
			char Trainingdata[100];
			std::cout<<">> Please input the name of text data:";
			cin>>Filename;
			std::cout<<">> Please input the name of sequence data(input data):";
			cin>>Sequencedata;
			std::cout<<">> Please input the name of training data:";
			cin>>Trainingdata;
			if(!fopen(Filename,"r")||!fopen(Sequencedata,"w")||!fopen(Trainingdata,"w")){
				std::cout<<">> [Error]Cannot open file."<<std::endl;
			}else{
				std::cout<<">> Please input the length of every input sequence:";
				cin>>maxtime;
				Seq2VecDataMaker(Filename,Sequencedata,Trainingdata,maxtime);
			}
		}else if(cmd=="10"){
			char Typename[100];
			std::cout<<">> Which type of networks would you like to find?\neasyNLP>>";
			cin>>Typename;
			manager.FindSpecialObj(Typename);
		}else if(cmd=="11"){
			std::cout<<">> [Quiting] Please wait."<<std::endl;
			manager.ObjDataOut();
			break;
		}else{
			std::cout<<">> [Error] Undefined command."<<std::endl;
			help();
		}
	}
	return 0;
}
