.PHONY=clean

SRC=./src/actvfunc.h\
	./src/bp.h\
	./src/char2vec.h\
	./src/gru.h\
	./src/lstm.h\
	./src/NLPann.h\
	./src/rnn.h\
	./src/seq2seq.h\
	./src/seq2vec.h

OBJECT=actvfunc.o bp.o rnn.o gru.o lstm.o seq2seq.o seq2vec.o

OPT=-O2

actvfunc.o: ./src/actvfunc.h ./src/actvfunc.cpp
	@ echo "[build] actvfunc"
	@ $(CXX) $(OPT) -std=c++11 -c ./src/actvfunc.cpp

bp.o: ./src/bp.h ./src/bp.cpp
	@ echo "[build] bp"
	@ $(CXX) $(OPT) -std=c++11 -c ./src/bp.cpp

rnn.o: ./src/rnn.h ./src/rnn.cpp
	@ echo "[build] rnn"
	@ $(CXX) $(OPT) -std=c++11 -c ./src/rnn.cpp

gru.o: ./src/gru.h ./src/gru.cpp
	@ echo "[build] gru"
	@ $(CXX) $(OPT) -std=c++11 -c ./src/gru.cpp

lstm.o: ./src/lstm.h ./src/lstm.cpp
	@ echo "[build] lstm"
	@ $(CXX) $(OPT) -std=c++11 -c ./src/lstm.cpp

seq2seq.o: ./src/seq2seq.h ./src/seq2seq.cpp
	@ echo "[build] seq2seq"
	@ $(CXX) $(OPT) -std=c++11 -c ./src/seq2seq.cpp

seq2vec.o: ./src/seq2vec.h ./src/seq2vec.cpp
	@ echo "[build] seq2vec"
	@ $(CXX) $(OPT) -std=c++11 -c ./src/seq2vec.cpp

easynlp: ./src/nlpmain.cpp ./src/mainassist.h $(SRC) $(OBJECT)
	@ $(CXX) $(OPT) -std=c++11 -c ./src/nlpmain.cpp
	@ echo "[link ] linking..."
	@ $(CXX) nlpmain.o $(OBJECT) -o easynlp
	@ rm nlpmain.o $(OBJECT)
	@ echo "[done ] build done"

clean:
	-@ rm easynlp