all: genTest.out
genTest.out: genTest.cpp
	g++ genTest.cpp -std=c++11 -o genTest.out
