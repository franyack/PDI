#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "pdi_functions.h"
#include "utils.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <bitset>

using namespace cv;
using namespace pdi;
using namespace std;


void TP3_Ejercicio1_1(){
	Mat img=imread("futbol.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat canvas(img.rows,img.cols,CV_32F);
	Mat histo = histogram(img,255);
	normalize(histo,histo,0,1,CV_MINMAX);
	draw_graph(canvas,histo);
	imshow("Histograma sin equalizar",canvas);
	imshow("Imagen original",img);
	
	
	Mat ecualizado;
	equalizeHist(img,ecualizado);
	imshow("Imagen ecualizada",ecualizado);
	Mat canvas2(ecualizado.rows,ecualizado.cols,CV_32F);
	Mat histo2 = histogram(ecualizado,255);
	normalize(histo2,histo2,0,1,CV_MINMAX);
	draw_graph(canvas2,histo2);
	imshow("Histograma equalizado", canvas2);

}

Mat filtro_promediador(int tam){
	Mat kernel(tam,tam,CV_32F);
	float aux=1/pow(tam,2);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=aux;
		}
	}
	return kernel;
}

void TP3_Ejercicio2_1(){
	Mat img=imread("huang2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat k3,k5,k11,k15;
	k3=filtro_promediador(3);
	k5=filtro_promediador(5);
	k11=filtro_promediador(11);
	k15=filtro_promediador(15);
	Mat f3,f5,f11,f15;
	f3=convolve(img,k3);
	f5=convolve(img,k5);
	f11=convolve(img,k11);
	f15=convolve(img,k15);
	vector<Mat> images;
	images.push_back(img);
	images.push_back(f3);
	images.push_back(f5);
	images.push_back(f11);
	images.push_back(f15);
	Mat m=mosaic(images,1);
	putText(m,"Original",cvPoint(0,240),FONT_HERSHEY_PLAIN,2,cvScalar(255,255,255),4);
	putText(m,"Kernel 3",cvPoint(257,240),FONT_HERSHEY_PLAIN,2,cvScalar(255,255,255),4);
	putText(m,"Kernel 5",cvPoint(513,240),FONT_HERSHEY_PLAIN,2,cvScalar(255,255,255),4);
	putText(m,"Kernel 11",cvPoint(767,240),FONT_HERSHEY_PLAIN,2,cvScalar(255,255,255),4);
	putText(m,"Kernel 15",cvPoint(1025,240),FONT_HERSHEY_PLAIN,2,cvScalar(255,255,255),4);
	//	putText(output,"Hello World :)",cvPoint(15,70),	FONT_HERSHEY_PLAIN,	3,cvScalar(0,255,0),4);
	imshow("Filtro Promediador",m);
}

Mat filtro_gaussiano(int tam,double sigma){
	int tamreal=tam;
	tam=tam/2;
	Mat kernel(tamreal,tamreal,CV_32F);
	// set standard deviation to 1.0
	double r, s = 2.0 * sigma * sigma;
	// sum is for normalization
	double sum = 0.0;
	// generate kernel
	for (int x = -tam; x <= tam; x++)
	{
		for(int y = -tam; y <= tam; y++)
		{
			r = sqrt(x*x + y*y);
			kernel.at<float>(x + tam,y + tam) = (exp(-(r*r)/s))/(M_PI * s);
			sum += kernel.at<float>(x + tam,y + tam);
		}
	}
	
	// normalize the Kernel
	for(int i = 0; i < tamreal; ++i)
		for(int j = 0; j < tamreal; ++j)
		kernel.at<float>(i,j) /= sum;
	
	return kernel;
}

void TP3_Ejercicio2_2(){
	Mat img=imread("huang2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//	Mat k3=filtro_gaussiano(3,1);
//	Mat k5=filtro_gaussiano(5,1);
//	Mat k33=filtro_gaussiano(3,10);
//	Mat k55=filtro_gaussiano(5,10);
	
	//De esta manera se hace lo mismo, solo que utilizando el filtro Gaussiano implementado por OpenCV
	Mat f3,f5,f33,f55;
	GaussianBlur(img,f3,Size(3,3),1);
	GaussianBlur(img,f5,Size(5,5),1);
	GaussianBlur(img,f33,Size(3,3),10);
	GaussianBlur(img,f55,Size(5,5),10);
	
//	Mat f3=convolve(img,k3);
//	Mat f5=convolve(img,k5);
//	Mat f33=convolve(img,k33);
//	Mat f55=convolve(img,k55);
	vector<Mat> images;
	images.push_back(img);
	images.push_back(f3);
	images.push_back(f5);
	images.push_back(f33);
	images.push_back(f55);
	Mat m=mosaic(images,1);
	putText(m,"Original",cvPoint(0,240),FONT_HERSHEY_PLAIN,1.4,cvScalar(255,255,255),2);
	putText(m,"Tam=3,Sigma=1",cvPoint(257,240),FONT_HERSHEY_PLAIN,1.4,cvScalar(255,255,255),2);
	putText(m,"Tam=5,Sigma=1",cvPoint(514,240),FONT_HERSHEY_PLAIN,1.4,cvScalar(255,255,255),2);
	putText(m,"Tam=3,Sigma=10",cvPoint(769,240),FONT_HERSHEY_PLAIN,1.4,cvScalar(255,255,255),2);
	putText(m,"Tam=5,Sigma=10",cvPoint(1026,240),FONT_HERSHEY_PLAIN,1.4,cvScalar(255,255,255),2);
	imshow("Filtro Gaussiano",m);
	
}


void TP3_Ejercicio2_3(){
	Mat img=imread("hubble.tif",CV_LOAD_IMAGE_GRAYSCALE);
	Mat kernel=filtro_promediador(9);
	Mat filtro=convolve(img,kernel);
	Mat umbral(filtro.size(),CV_8UC(1));
	for (int i=0;i<filtro.rows;i++){
		for (int j=0;j<filtro.cols;j++){
			if ((int)filtro.at<uchar>(i,j)>150){umbral.at<uchar>(i,j)=255;}
			else{umbral.at<uchar>(i,j)=0;}
		}
	}
	imshow("Original",img);
	imshow("Filtrado",filtro);
	imshow("Umbral Binario",umbral);

}

Mat filtro_pasa_alto_suma1(int tam){
	Mat kernel(tam,tam,CV_32F);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=-1;
		}
	}
	kernel.at<float>(tam/2,tam/2)=tam*tam;
	return kernel;
}

void TP3_Ejercicio3_1(){
	Mat img=imread("camaleon.tif",CV_LOAD_IMAGE_GRAYSCALE);
	Mat kernel=filtro_pasa_alto_suma1(3);
	Mat k5=filtro_pasa_alto_suma1(5);
	Mat k7=filtro_pasa_alto_suma1(7);
	Mat filtro=convolve(img,kernel);
	Mat filtro5=convolve(img,k5);
	Mat filtro7=convolve(img,k7);
	vector<Mat> images;
	images.push_back(img);
	images.push_back(filtro);
	images.push_back(filtro5);
	images.push_back(filtro7);
	Mat m=mosaic(images,1);
	imshow("Filtro Pasa-Altos Suma 1",m);
	
}

Mat filtro_pasa_alto_suma0(int tam){
	Mat kernel(tam,tam,CV_32F);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=-1;
		}
	}
	kernel.at<float>(tam/2,tam/2)=tam*tam-1;
	return kernel;
}
void TP3_Ejercicio3_2(){
	Mat img=imread("camaleon.tif",CV_LOAD_IMAGE_GRAYSCALE);
	Mat kernel=filtro_pasa_alto_suma0(3);
	Mat k5=filtro_pasa_alto_suma0(5);
	Mat k7=filtro_pasa_alto_suma0(7);
	Mat filtro=convolve(img,kernel);
	Mat filtro5=convolve(img,k5);
	Mat filtro7=convolve(img,k7);
	vector<Mat> images;
	images.push_back(img);
	images.push_back(filtro);
	images.push_back(filtro5);
	images.push_back(filtro7);
	Mat m=mosaic(images,1);
	imshow("Filtro Pasa-Altos Suma 0",m);
}

void TP3_Ejercicio4_1(){
	Mat img=imread("camaleon.tif",CV_LOAD_IMAGE_GRAYSCALE);
	Mat kernel = filtro_promediador(9);
	Mat filtrada = convolve(img,kernel);
	Mat difusa = (img - filtrada + 255)/2;
	imshow("Original",img);
	imshow("Filtro Pasa Bajos",filtrada);
	imshow("Mascara Difusa",difusa);
}

void TP3_Ejercicio4_2(float A){
	Mat img=imread("camaleon.tif",CV_LOAD_IMAGE_GRAYSCALE);
	Mat kernel = filtro_promediador(9);
	Mat filtrada = convolve(img,kernel);
	Mat difusa = (A*img - filtrada + 255)/2;
	imshow("Original",img);
	imshow("Filtro Pasa Bajos",filtrada);
	imshow("Filtrado Alta Potencia",difusa);
}



int main(int argc, char** argv) {
	
//	TP3_Ejercicio1_1();
//	TP3_Ejercicio2_1();
//	TP3_Ejercicio2_2();
//	TP3_Ejercicio2_3();
//	TP3_Ejercicio3_1();
//	TP3_Ejercicio3_2();
//	TP3_Ejercicio4_1();
	TP3_Ejercicio4_2(2);
	
	
	waitKey(0);
	return 0;
} 
