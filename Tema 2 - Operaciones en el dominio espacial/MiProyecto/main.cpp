#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "pdi_functions.h"
#include "utils.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <ctime>

using namespace cv;
using namespace pdi;
using namespace std;

float generarRandom(float a, float b){
	std::mt19937_64 rng;
	// initialize the random number generator with time-dependent seed
	uint64_t timeSeed = rand();//std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
	rng.seed(ss);
	// initialize a uniform distribution between 0 and 1
	std::uniform_real_distribution<double> unif(a, b);
	double currentRandomNumber = unif(rng);
	return currentRandomNumber;
}


void TP2_Ejercicio1(float a,float c){
	//create a gui window:
	namedWindow("Output",1);
	//initialize a 120X350 matrix of black pixels:
	Mat img = imread("huang1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Foto original",img);
	
	Mat lut(1,256,CV_8U);
	
	float aux;
//	for(int i=0;i<10;i++){
//		aux = 0;
//		lut.at<unsigned char>(i)= aux;
//	}
	for(int i=0;i<255;i++){
		
		aux = a * i + c;
		if(aux>255) aux = 255;
		if(aux<0) aux = 0;
		lut.at<unsigned char>(i)= aux;
	}
//	for(int i=60;i<255;i++){
//		aux = 0;
//		lut.at<unsigned char>(i)= aux;
//	}
	
	Mat output;
	
	Mat grafico(255,255,CV_8U);
	grafico.setTo(Scalar(0,0,0));
	draw_graph(grafico,lut);
	imshow("Grafico LUT",grafico);
	
	
	
	LUT(img,lut,output);
	
	normalize(output,output,0,255,CV_MINMAX);
	
	//display the image:
	imshow("Output", output);
	
//	Mat *k = new Mat[2];
//	k[0] = img;
//	//k[1] = grafico;
//	k[1] = output;
//	
//	ShowMultipleImages(k,2,2,256,256,20);
	//wait for the user to press any key:
	waitKey(0);
}



void TP2_Ejercicio2(int inciso,float gamma=1){
	//create a gui window:
	namedWindow("Output",1);
	//initialize a 120X350 matrix of black pixels:
	Mat img = imread("huang1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Foto original",img);
	
	Mat lut(1,256,CV_8U);
	float c = 255.0/log(256);
	float aux;
	if (inciso == 1){
		for(int i=0;i<255;i++){
			aux = c*log(1+i);
			if(aux>255) aux = 255;
			if(aux<0) aux = 0;
			lut.at<unsigned char>(i)= aux;
		}
	}else{
		for(int i=0;i<255;i++){
			
			aux = pow(i/255.0,gamma)*255.0;
			if(aux>255) aux = 255;
			if(aux<0) aux = 0;
			lut.at<unsigned char>(i)= aux;
		}
	}
	

	
	Mat output;
	
	Mat grafico(256,256,CV_8U);
	grafico.setTo(Scalar(0,0,0));
	draw_graph(grafico,lut);
	imshow("Grafico LUT",grafico);
	
	
	
	LUT(img,lut,output);
	
	//normalize(output,output,0,255,CV_MINMAX);
	
	//display the image:
	imshow("Output", output);
	
	
	

	
}


void TP2_Ejercicio3(){
	///SUMA DE DOS IMAGENES
	///LAS IMAGENES DEBEN SER DEL MISMO TAMAÑO, POR LO TANTO HAY QUE DEFINIR UN ROI DE IGUAL DE MAGNITUD DE LAS MISMAS. ADEMAS, SE LAS PUEDE PONDERAR
	///COMO SE MUESTRA EN EL EJEMPLO DE ABAJO
	
	
	Mat img1 = imread("huang1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("huang2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Rect region_of_interest = Rect(1, 1, 256, 256);
	img1 = img1(region_of_interest);
	Mat result = (img1 + img2)/2;
	imshow("Suma", result);
	
	///RESTA Y REESCALADOS TIPICOS PARA EVITAR DESBORDES DE RANGO
	
	result = (img1 - img2 + 255)/2;
	imshow("Resta", result);
	
	///MULTIPLICACION
	
	Mat mask= Mat::zeros(img1.size(),img1.type());
	
	mask(Rect(img1.rows/2-50,img1.cols/2-50,100,100)) = 1;
	
	Mat resultado;
	
	img1.copyTo(resultado,mask);
	
	imshow("Resultado",resultado);
	//Mat mask = Mat::zeros(img1.size(),img1.type());
	
	//circle(mask,cv::Point(mask.rows/2,mask.cols/2),50,Scalar(255,0,0),-1,8,0);
	
//	Rect mask = Rect(1,1,256,256);
//	Mat roi = img1(mask);
//	
//	
//	imshow("Mascara", roi);
//	
	
	
}


void TP2_Ejercicio3_2(){
	Mat img = imread("huang1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imagen Original",img);
	Mat aux(300,300,CV_8U);
	
	float aux2;
	for (int i=0;i<aux.rows;i++){
		for(int j=0;j<aux.cols;j++){
			aux2 = aux.at<uchar>(i,j)+generarRandom(0,0.05);
			if(aux2>255) aux2 = 255;
			img.at<uchar>(i,j)=aux2;
			cout<<(float)aux.at<uchar>(i,j)<<endl;
		}
	}
	//imshow("Imagen contaminada",img);
	
}



int main(int argc, char** argv) {
	
	//TP2_Ejercicio1(100,100);
	
//	TP2_Ejercicio2(2,2);
	
	TP2_Ejercicio3();
	
//	TP2_Ejercicio3_2();
	
	

	
	
	waitKey(0);
	return 0;
} 
