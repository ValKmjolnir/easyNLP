#include<iostream>
#include<cstdlib>
#include<ctime>
#include<conio.h>
#include "bp.h"
using namespace std;

int main()
{
	NormalBP a(20,20,20);
	DeepBP b(10,10,10,3);

	
	a.INIT();
	a.Dataout("a.txt");
	a.Datain("a.txt");
	a.SetFunction("sigmoid");
	a.Calc();
	a.ErrorCalc();
	cout<<a.GetError()<<endl;
	a.Training();
	b.INIT();
	b.Dataout("b.txt");
	b.Datain("b.txt");
	b.SetFunction("tanh");
	b.Calc();
	b.ErrorCalc();
	cout<<b.GetError()<<endl;
	b.Training();


	cout<<"Press any key to continue...";
	getch();
	return 0;
}
