.PHONY=clean

SRC=actvfunc.h\
	bp.h\
	char2vec.h\
	gru.h\
	lstm.h\
	NLPann.h\
	rnn.h\
	seq2seq.h\
	seq2vec.h

OBJECT=actvfunc.o bp.o rnn.o gru.o lstm.o seq2seq.o seq2vec.o

actvfunc.o: actvfunc.h actvfunc.cpp
	$(CXX) -O2 -std=c++11 -c actvfunc.cpp

bp.o: bp.h bp.cpp
	$(CXX) -O2 -std=c++11 -c bp.cpp

rnn.o: rnn.h rnn.cpp
	$(CXX) -O2 -std=c++11 -c rnn.cpp

gru.o: gru.h gru.cpp
	$(CXX) -O2 -std=c++11 -c gru.cpp

lstm.o: lstm.h lstm.cpp
	$(CXX) -O2 -std=c++11 -c lstm.cpp
seq2seq.o: seq2seq.h seq2seq.cpp
	$(CXX) -O2 -std=c++11 -c seq2seq.cpp

seq2vec.o: seq2vec.h seq2vec.cpp
	$(CXX) -O2 -std=c++11 -c seq2vec.cpp

easynlp: nlpmain.cpp mainassist.h $(SRC) $(OBJECT)
	$(CXX) -O2 -std=c++11 -c nlpmain.cpp
	$(CXX) nlpmain.o $(OBJECT) -o easynlp
	rm $(OBJECT) nlpmain.o

clean:
	-@ rm easynlp