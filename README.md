# easyNLP

Do NLP without coding! (in fact it is the library of common models)

In this program i will use dynamic arrays to make neural networks which can do NLP tasks. (refactoring...)

## Build

This project does not rely on any other third-party libraries.

Make sure you have `make`and `gcc` or `clang` in your environment. Then use this command:

```bash
make easynlp
```

Also can use `-j` to use multi-task of make.

## Future Work

This repo is too old and i'm trying to make it running correctly. (and only running this program takes lots of time...)

`nlpmain.cpp` and `mainassist.h` are useless. If you really want to use this project, only do this:

```C++
#include "src/NLPann.h"
```

Add the `NLPann.h` as the header, then you could use all models in it.

The documention will be added later, but before this i'll make this project more like a `C++` project, not `C` project...
