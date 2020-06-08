#ifndef __Mnist_
#define __Mnist_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

typedef struct MnistImg{
	int c;           // image width
	int r;           // image height
	float** ImgData; // image data
}MnistImg;

typedef struct MnistImgArr{
	int ImgNum;        // number of images
	MnistImg* ImgPtr;  // pointer to images
}MnistImgArrS, *ImgArr;

typedef struct MnistLabel{
	int l;            // len of labels
	float* LabelData; // labels
}MnistLabel;

typedef struct MnistLabelArr{
	int LabelNum;
	MnistLabel* LabelPtr;
} MnistLabelArrS, *LabelArr;

LabelArr read_Lable(const char* filename); 

ImgArr read_Img(const char* filename);

LabelArr read_Lable_ar();

ImgArr read_Img_ar();

void save_Img(ImgArr imgarr,char* filedir);

#endif