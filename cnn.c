#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "cnn.h"

extern unsigned char rawNNData[15592];

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize)
{
	cnn->layerNum=5;

	nSize inSize;
	int mapSize=5;
	inSize.c=inputSize.c;
	inSize.r=inputSize.r;
	cnn->C1=initCovLayer(inSize.c,inSize.r,5,1,6);
	inSize.c=inSize.c-mapSize+1;
	inSize.r=inSize.r-mapSize+1;
	cnn->S2=initPoolLayer(inSize.c,inSize.r,2,6,6,AvePool);
	inSize.c=inSize.c/2;
	inSize.r=inSize.r/2;
	cnn->C3=initCovLayer(inSize.c,inSize.r,5,6,12);
	inSize.c=inSize.c-mapSize+1;
	inSize.r=inSize.r-mapSize+1;
	cnn->S4=initPoolLayer(inSize.c,inSize.r,2,12,12,AvePool);
	inSize.c=inSize.c/2;
	inSize.r=inSize.r/2;
	cnn->O5=initOutLayer(inSize.c*inSize.r*12,outputSize);

	cnn->e=(float*)calloc(1, cnn->O5->outputNum*sizeof(float));
}

CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels)
{
	CovLayer* covL=(CovLayer*)malloc(sizeof(CovLayer));

	covL->inputHeight=inputHeight;
	covL->inputWidth=inputWidth;
	covL->mapSize=mapSize;

	covL->inChannels=inChannels;
	covL->outChannels=outChannels;

	covL->isFullConnect=1; // 默认为全连接

	// 权重空间的初始化，先行再列调用，[r][c]
	int i,j,c,r;
	srand((unsigned)0);
	covL->mapData=(float****)malloc(inChannels*sizeof(float***));
	for(i=0;i<inChannels;i++){
		covL->mapData[i]=(float***)malloc(outChannels*sizeof(float**));
		for(j=0;j<outChannels;j++){
			covL->mapData[i][j]=(float**)malloc(mapSize*sizeof(float*));
			for(r=0;r<mapSize;r++){
				covL->mapData[i][j][r]=(float*)malloc(mapSize*sizeof(float));
				for(c=0;c<mapSize;c++){
					float randnum=(((float)rand()/(float)2147483647.0f)-0.5f)*2.0f; //RAND_MAX
					covL->mapData[i][j][r][c]=randnum*sqrtf((float)6.0f/(float)(mapSize*mapSize*(inChannels+outChannels))); //sqrtf
				}
			}
		}
	}

	// 权重梯度变化
	covL->dmapData=(float****)malloc(inChannels*sizeof(float***));
	for(i=0;i<inChannels;i++){
		covL->dmapData[i]=(float***)malloc(outChannels*sizeof(float**));
		for(j=0;j<outChannels;j++){
			covL->dmapData[i][j]=(float**)malloc(mapSize*sizeof(float*));
			for(r=0;r<mapSize;r++){
				covL->dmapData[i][j][r]=(float*)calloc(1, mapSize*sizeof(float));
			}
		}
	}

	covL->basicData=(float*)calloc(1, outChannels*sizeof(float));

	int outW=inputWidth-mapSize+1;
	int outH=inputHeight-mapSize+1;


	covL->d=(float***)malloc(outChannels*sizeof(float**));
	covL->v=(float***)malloc(outChannels*sizeof(float**));
	covL->y=(float***)malloc(outChannels*sizeof(float**));
	for(j=0;j<outChannels;j++){
		covL->d[j]=(float**)malloc(outH*sizeof(float*));
		covL->v[j]=(float**)malloc(outH*sizeof(float*));
		covL->y[j]=(float**)malloc(outH*sizeof(float*));
		for(r=0;r<outH;r++){
			covL->d[j][r]=(float*)calloc(1, outW*sizeof(float));
			covL->v[j][r]=(float*)calloc(1, outW*sizeof(float));
			covL->y[j][r]=(float*)calloc(1, outW*sizeof(float));
		}
	}

	return covL;
}

PoolLayer* initPoolLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels,int poolType)
{
	PoolLayer* poolL=(PoolLayer*)malloc(sizeof(PoolLayer));

	poolL->inputHeight=inputHeight;
	poolL->inputWidth=inputWidth;
	poolL->mapSize=mapSize;
	poolL->inChannels=inChannels;
	poolL->outChannels=outChannels;
	poolL->poolType=poolType; 

	poolL->basicData=(float*)calloc(1, outChannels*sizeof(float));

	int outW=inputWidth/mapSize;
	int outH=inputHeight/mapSize;

	int j,r;
	poolL->d=(float***)malloc(outChannels*sizeof(float**));
	poolL->y=(float***)malloc(outChannels*sizeof(float**));
	for(j=0;j<outChannels;j++){
		poolL->d[j]=(float**)malloc(outH*sizeof(float*));
		poolL->y[j]=(float**)malloc(outH*sizeof(float*));
		for(r=0;r<outH;r++){
			poolL->d[j][r]=(float*)calloc(1, outW*sizeof(float));
			poolL->y[j][r]=(float*)calloc(1, outW*sizeof(float));
		}
	}

	return poolL;
}

OutLayer* initOutLayer(int inputNum,int outputNum)
{
	OutLayer* outL=(OutLayer*)malloc(sizeof(OutLayer));

	outL->inputNum=inputNum;
	outL->outputNum=outputNum;


	outL->basicData=(float*)calloc(1, outputNum*sizeof(float));

	outL->d=(float*)calloc(1, outputNum*sizeof(float));
	outL->v=(float*)calloc(1, outputNum*sizeof(float));
	outL->y=(float*)calloc(1, outputNum*sizeof(float));

	// 权重的初始化
	outL->wData=(float**)malloc(outputNum*sizeof(float*)); // 输入行，输出列
	int i,j;
	srand((unsigned)0);
	for(i=0;i<outputNum;i++){
		outL->wData[i]=(float*)malloc(inputNum*sizeof(float));
		for(j=0;j<inputNum;j++){
			float randnum=(((float)rand()/(float)2147483647.0f)-0.5f)*2.0f; // 产生一个-1到1的随机数 RAND_MAX
			outL->wData[i][j]=randnum*sqrtf((float)6.0f/(float)(inputNum+outputNum)); //sqrtf
		}
	}

	outL->isFullConnect=1;

	return outL;
}

int vecmaxIndex(float* vec, int veclength)// 返回向量最大数的序号
{
	int i;
	float maxnum=-1.0f;
	int maxIndex=0;
	for(i=0;i<veclength;i++){
		if(maxnum<vec[i]){
			maxnum=vec[i];
			maxIndex=i;
		}
	}
	return maxIndex;
}

// 测试cnn函数
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum)
{
	int n=0;
	int incorrectnum=0;  //错误预测的数目
	for(n=0;n<testNum;n++){
		cnnff(cnn,inputData->ImgPtr[n].ImgData);

		for (int i = 0; i < 10; i++)
		{
			printf("%f ", cnn->O5->y[i]);
		}
		printf("\n");

		printf("id => %d, should be %d\n", 
		vecmaxIndex(cnn->O5->y,cnn->O5->outputNum), 
		vecmaxIndex(outputData->LabelPtr[n].LabelData,cnn->O5->outputNum) );
		if(vecmaxIndex(cnn->O5->y,cnn->O5->outputNum)!=vecmaxIndex(outputData->LabelPtr[n].LabelData,cnn->O5->outputNum))
			incorrectnum++;
		cnnclear(cnn);
	}
	printf("inccorrect num %d, total num %d\n", incorrectnum, testNum);
	return (float)incorrectnum/(float)testNum;
}

// 导入cnn的数据
void importcnn(CNN* cnn)
{

	float* ap = (float*)rawNNData;

	int i,j,c,r;
	// C1的数据
	for(i=0;i<cnn->C1->inChannels;i++)
	{
		for(j=0;j<cnn->C1->outChannels;j++)
		{
			for(r=0;r<cnn->C1->mapSize;r++)
			{
				for(c=0;c<cnn->C1->mapSize;c++){
					cnn->C1->mapData[i][j][r][c] = *ap++;
					// printf("%f ", cnn->C1->mapData[i][j][r][c]);
				}
				// printf("\n");
			}
			// printf("\n");
		}
		// printf("\n");
	}

	for(i=0;i<cnn->C1->outChannels;i++)
	{
		cnn->C1->basicData[i] = *ap++;
		// printf("%f ", cnn->C1->basicData[i]);
	}

	// printf("\n");

	// C3网络
	for (i = 0; i < cnn->C3->inChannels; i++)
		for (j = 0; j < cnn->C3->outChannels; j++)
			for (r = 0; r < cnn->C3->mapSize; r++)
				for (c = 0; c < cnn->C3->mapSize; c++)
					cnn->C3->mapData[i][j][r][c] = *ap++;

	for (i = 0; i < cnn->C3->outChannels; i++)
		cnn->C3->basicData[i] = *ap++;

	// O5输出层
	for (i = 0; i < cnn->O5->outputNum; i++)
		for (j = 0; j < cnn->O5->inputNum; j++)
			cnn->O5->wData[i][j] = *ap++;

	for (i = 0; i < cnn->O5->outputNum; i++)
		cnn->O5->basicData[i] = *ap++;

}

// 这里InputData是图像数据，inputData[r][c],r行c列，这里根各权重模板是一致的
void cnnff(CNN* cnn,float** inputData)
{
	int outSizeW=cnn->S2->inputWidth;
	int outSizeH=cnn->S2->inputHeight;
	// 第一层的传播
	int i,j,r,c;
	// 第一层输出数据
	nSize mapSize={cnn->C1->mapSize,cnn->C1->mapSize};
	nSize inSize={cnn->C1->inputWidth,cnn->C1->inputHeight};
	nSize outSize={cnn->S2->inputWidth,cnn->S2->inputHeight};
	for(i=0;i<(cnn->C1->outChannels);i++){
		for(j=0;j<(cnn->C1->inChannels);j++){
			float** mapout=cov(cnn->C1->mapData[j][i],mapSize,inputData,inSize,valid);
			addmat(cnn->C1->v[i],cnn->C1->v[i],outSize,mapout,outSize);
			for(r=0;r<outSize.r;r++)
				free(mapout[r]);
			free(mapout);
		}
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				cnn->C1->y[i][r][c]=activation_Sigma(cnn->C1->v[i][r][c],cnn->C1->basicData[i]);

	}
	



	for(r=0;r<outSize.r;r++)
	{
		for(c=0;c<outSize.c;c++)
		{
			for(i=0;i<(cnn->C1->outChannels);i++){
				printf("%f ", cnn->C1->y[i][r][c]);
			}
			printf("\n");
		}
		printf("\n");
	}

	
	printf("\n");

	// 第二层的输出传播S2，采样层
	outSize.c=cnn->C3->inputWidth;
	outSize.r=cnn->C3->inputHeight;
	inSize.c=cnn->S2->inputWidth;
	inSize.r=cnn->S2->inputHeight;
	for(i=0;i<(cnn->S2->outChannels);i++){
		if(cnn->S2->poolType==AvePool)
			avgPooling(cnn->S2->y[i],outSize,cnn->C1->y[i],inSize,cnn->S2->mapSize);
	}

	// 第三层输出传播,这里是全连接
	outSize.c=cnn->S4->inputWidth;
	outSize.r=cnn->S4->inputHeight;
	inSize.c=cnn->C3->inputWidth;
	inSize.r=cnn->C3->inputHeight;
	mapSize.c=cnn->C3->mapSize;
	mapSize.r=cnn->C3->mapSize;
	for(i=0;i<(cnn->C3->outChannels);i++){
		for(j=0;j<(cnn->C3->inChannels);j++){
			float** mapout=cov(cnn->C3->mapData[j][i],mapSize,cnn->S2->y[j],inSize,valid);
			addmat(cnn->C3->v[i],cnn->C3->v[i],outSize,mapout,outSize);
			for(r=0;r<outSize.r;r++)
				free(mapout[r]);
			free(mapout);
		}
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				cnn->C3->y[i][r][c]=activation_Sigma(cnn->C3->v[i][r][c],cnn->C3->basicData[i]);
	}

	// 第四层的输出传播
	inSize.c=cnn->S4->inputWidth;
	inSize.r=cnn->S4->inputHeight;
	outSize.c=inSize.c/cnn->S4->mapSize;
	outSize.r=inSize.r/cnn->S4->mapSize;
	for(i=0;i<(cnn->S4->outChannels);i++){
		if(cnn->S4->poolType==AvePool)
			avgPooling(cnn->S4->y[i],outSize,cnn->C3->y[i],inSize,cnn->S4->mapSize);
	}

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float* O5inData=(float*)malloc((cnn->O5->inputNum)*sizeof(float)); 
	for(i=0;i<(cnn->S4->outChannels);i++)
		for(r=0;r<outSize.r;r++)
			for(c=0;c<outSize.c;c++)
				O5inData[i*outSize.r*outSize.c+r*outSize.c+c]=cnn->S4->y[i][r][c];

	nSize nnSize={cnn->O5->inputNum,cnn->O5->outputNum};
	nnff(cnn->O5->v,O5inData,cnn->O5->wData,cnn->O5->basicData,nnSize);
	for(i=0;i<cnn->O5->outputNum;i++)
		cnn->O5->y[i]=activation_Sigma(cnn->O5->v[i],cnn->O5->basicData[i]);
	free(O5inData);
}

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input,float bas) // sigma激活函数
{
	float temp=input+bas;
	return (float)1.0f/((float)(1.0f+expf(-temp)));	//expf
}

void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize) // 求平均值
{
	int outputW=inputSize.c/mapSize;
	int outputH=inputSize.r/mapSize;
	if(outputSize.c!=outputW||outputSize.r!=outputH)
		printf("ERROR: output size is wrong!!");

	int i,j,m,n;
	for(i=0;i<outputH;i++)
		for(j=0;j<outputW;j++)
		{
			float sum=0.0f;
			for(m=i*mapSize;m<i*mapSize+mapSize;m++)
				for(n=j*mapSize;n<j*mapSize+mapSize;n++)
					sum=sum+input[m][n];

			output[i][j]=sum/(float)(mapSize*mapSize);
		}
}

// 单层全连接神经网络的前向传播
float vecMulti(float* vec1,float* vec2,int vecL)// 两向量相乘
{
	int i;
	float m=0.0f;
	for(i=0;i<vecL;i++)
		m=m+vec1[i]*vec2[i];
	return m;
}

void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize)
{
	int w=nnSize.c;
	int h=nnSize.r;
	
	int i;
	for(i=0;i<h;i++)
		output[i]=vecMulti(input,wdata[i],w)+bas[i];
}

float sigma_derivation(float y){ // Logic激活函数的自变量微分
	return y*(1.0f-y); // 这里y是指经过激活函数的输出值，而不是自变量
}

void cnnclear(CNN* cnn)
{
	// 将神经元的部分数据清除
	int j,c,r;
	// C1网络
	for(j=0;j<cnn->C1->outChannels;j++){
		for(r=0;r<cnn->S2->inputHeight;r++){
			for(c=0;c<cnn->S2->inputWidth;c++){
				cnn->C1->d[j][r][c]=(float)0.0f;
				cnn->C1->v[j][r][c]=(float)0.0f;
				cnn->C1->y[j][r][c]=(float)0.0f;
			}
		}
	}
	// S2网络
	for(j=0;j<cnn->S2->outChannels;j++){
		for(r=0;r<cnn->C3->inputHeight;r++){
			for(c=0;c<cnn->C3->inputWidth;c++){
				cnn->S2->d[j][r][c]=(float)0.0f;
				cnn->S2->y[j][r][c]=(float)0.0f;
			}
		}
	}
	// C3网络
	for(j=0;j<cnn->C3->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight;r++){
			for(c=0;c<cnn->S4->inputWidth;c++){
				cnn->C3->d[j][r][c]=(float)0.0f;
				cnn->C3->v[j][r][c]=(float)0.0f;
				cnn->C3->y[j][r][c]=(float)0.0f;
			}
		}
	}
	// S4网络
	for(j=0;j<cnn->S4->outChannels;j++){
		for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
			for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
				cnn->S4->d[j][r][c]=(float)0.0f;
				cnn->S4->y[j][r][c]=(float)0.0f;
			}
		}
	}
	// O5输出
	for(j=0;j<cnn->O5->outputNum;j++){
		cnn->O5->d[j]=(float)0.0f;
		cnn->O5->v[j]=(float)0.0f;
		cnn->O5->y[j]=(float)0.0f;
	}
}
