#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "pdi_functions.h"
#include "utils.h"

using namespace cv;
using namespace pdi;


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
	
	float aux;
	if (inciso == 1){
		for(int i=0;i<255;i++){
			
			aux = log(1+i);///255.0)*255;
			if(aux>255) aux = 255;
			if(aux<0) aux = 0;
			lut.at<unsigned char>(i)= aux;
		}
	}else{
		for(int i=0;i<255;i++){
			
			aux = pow(i,gamma);
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
	
	normalize(output,output,0,255,CV_MINMAX);
	
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
	
	//Mat mask = Mat::zeros(img1.size(),img1.type());
	
	//circle(mask,cv::Point(mask.rows/2,mask.cols/2),50,Scalar(255,0,0),-1,8,0);
	
//	Rect mask = Rect(1,1,256,256);
//	Mat roi = img1(mask);
//	
//	
//	imshow("Mascara", roi);
//	
	
	
}


int main(int argc, char** argv) {
	
	//TP2_Ejercicio1(100,100);
	
	//TP2_Ejercicio2(2,0.5);
	
	TP2_Ejercicio3();
	
	waitKey(0);
	return 0;
} 
