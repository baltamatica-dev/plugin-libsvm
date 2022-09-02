# SPDX-License-Identifier: BSD-3-Clause

CXX ?= g++
CXXFLAGS = -Wall -std=c++17 -shared -O3 -fPIC \
	-I/s/bex-dev/bex/baltamatica/bex/include/ -I.. \
	-L/s/bex-dev/bex/baltamatica/bex/lib -lbex

BEX_EXT = dll

all: libsvm.$(BEX_EXT)

libsvm.$(BEX_EXT): ../svm.h ../svm.cpp libsvm.cpp svm_model_matlab.c svmpredict.c svmtrain.c libsvmread.c libsvmwrite.c
	$(CXX) $(CXXFLAGS) -o libsvm/main.$(BEX_EXT) ../svm.cpp libsvm.cpp svm_model_matlab.c svmpredict.c svmtrain.c libsvmread.c libsvmwrite.c

clean:
	rm -f *~ *.o *.dll *.obj
