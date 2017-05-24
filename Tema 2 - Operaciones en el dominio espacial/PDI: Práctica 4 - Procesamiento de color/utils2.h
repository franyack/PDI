#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace cv;
using namespace pdi;
using namespace std;


///GUIA 2

Mat transformacionLogaritmica(Mat img){
	Mat lut(1,256,CV_8U);
	float c = 255.0/log(256);
	float aux;
	for(int i=0;i<255;i++){
		aux = c*log(1+i);
		if(aux>255) aux = 255;
		if(aux<0) aux = 0;
		lut.at<unsigned char>(i)= aux;
	}
	Mat output;
	LUT(img,lut,output);
	return output;
}

Mat transformacionPotencia(Mat img, float gamma=1){
	Mat lut(1,256,CV_8U);
	float c = 255.0/log(256);
	float aux;
	for(int i=0;i<255;i++){
		
		aux = pow(i/255.0,gamma)*255.0;
		if(aux>255) aux = 255;
		if(aux<0) aux = 0;
		lut.at<unsigned char>(i)= aux;
	}
	Mat output;
	LUT(img,lut,output);
	return output;
}


Mat ruidoGaussiano(Mat img, float media, float desvio){
	img.convertTo(img,CV_32F,1/255.0);
	Mat ruido(img.size(),img.type());
	randn(ruido,media,desvio);
	Mat aux=(ruido+img);
	return aux;
}

//En la ubicacion 0 el bit menos significativo, en la 7 el mas significativo
std::vector <Mat> planoBits(Mat img){
	std::vector<Mat>BitPlane;
	for (int i=0; i < 8; i++) {
		Mat outPut;
		//Aplico mascara AND para acceder a cada bit plane (2^i)
		bitwise_and(img, pow(2,i), outPut);
		//Multiplico por 255 para hacerlo blanco y negro
		outPut *= 255;
		BitPlane.push_back(outPut);
	}
	return BitPlane;
}
