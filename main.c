// Copyright 2017 ETH Zurich and University of Bologna.
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the “License”); you may not use this file except in
// compliance with the License.  You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

/* Scheduler include files. */
#include "cnn.h"
#include "mnist.h"

int main( void )
{

	printf("Starting MNIST\n");
	printf("Reading test data\n");
	LabelArr testLabel = read_Lable_ar();
	ImgArr testImg = read_Img_ar();
	nSize inputSize={testImg->ImgPtr[0].c,testImg->ImgPtr[0].r};
	int outSize=testLabel->LabelPtr[0].l;
	printf("Test data readed\n");

	// CNN结构的初始化
	CNN s_cnn;
	CNN* cnn = &s_cnn;
	printf("Init CNN\n");
	cnnsetup(cnn,inputSize,outSize);
	printf("CNN init done\n");

	// CNN测试
	printf("Import CNN\n");
	importcnn(cnn);
	printf("CNN imported\n");
	int testNum=1;
	float incorrectRatio=0.0f;
	incorrectRatio=cnntest(cnn,testImg,testLabel,testNum);
	printf("test finished!!\n");
	printf("test incorrectRatio %d / %d \n", (int)incorrectRatio*1000, 1000);
	return 0;
}
