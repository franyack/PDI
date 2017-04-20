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
//	Mat img = imread("huang1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//	imshow("Imagen Original",img);
//	vector<Mat>imagenes;
//	
//	for(int i=0;i<50;i++){
//		Mat ruido = Mat::zeros(img.rows,img.cols,CV_8U);
//		randn(ruido,0,0.05);
//		Mat resultado(img.size(),img.type());
//		Mat resultado = (img+ruido);
//		imagenes.push_back(resultado);
//	}
//	Mat promedio = imagenes[0];
//
//	for(int i=1;i<imagenes.size();i++){
//		promedio+=imagenes[i];
//	}
//	
//	promedio/=imagenes.size();
//	
//	imshow("Promedio",promedio);
//	
//	
////	float aux2;
////	for (int i=0;i<aux.rows;i++){
////		for(int j=0;j<aux.cols;j++){
////			aux2 = aux.at<uchar>(i,j)+generarRandom(0,0.05);
////			if(aux2>255) aux2 = 255;
////			img.at<uchar>(i,j)=aux2;
////			cout<<(float)aux.at<uchar>(i,j)<<endl;
////		}
////	}
//	//imshow("Imagen contaminada",img);
	
	Mat img=imread("huang1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//	Mat ruido(img.size(),img.type());
	//	randn(ruido,0,0.5*255);
	Mat aux;
	img.convertTo(aux,CV_32F);
	vector<Mat> imagenes;
	for (int i=0;i<50;i++){
		Mat ruido(img.size(),img.type());
		randn(ruido,0,0.5*255);
		Mat resultado=(aux+ruido)/2;
		imagenes.push_back(resultado);
	}
	Mat mezcla=imagenes[0];
	for (int i=1;i<imagenes.size();i++){
		mezcla+=imagenes[i];
	}
	mezcla/=imagenes.size();
	imshow("Original",img);
	imshow("Ruido",imagenes[0]);
	imshow("Resultado",mezcla);
	
}

void TP2_Ejercicio4(){
	
	Mat input_img(imread("huang1.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	imshow("Original",input_img);
	//Clono la imagen..
	Mat work_img(input_img.clone());
	//Creo vector de Mat
	vector<Mat>BitPlane;
	//Creo un string con el nombre de archivo
	string file_name("huang2_bit0.png");
	for (int i=0; i < 8; i++) {
		Mat outPut;
		//Aplico mascara AND para acceder a cada bit plane (2^i)
		bitwise_and(work_img, pow(2,i), outPut);
		//Multiplico por 255 para hacerlo blanco y negro
		outPut *= 255;
		BitPlane.push_back(outPut);
		//Escribo la imagen en el disco
		//		cv::imwrite(file_name, outPut);
		//		//Modifico el nombre del archivo
		//		file_name[10] += 1;
	}
	//Muestro todas las imagenes en una sola pantalla
	//mosaic(Vector,cantidadFilas);
	Mat Graficar=mosaic(BitPlane,2);
	imshow("Bit Planes 0 - 7",Graficar);
	///2da Parte Construya una imagen solo con la informacion del plano del bit mas
	///significativo, luego construya otra imagen con la informacion de los dos planos 
	///mas significativos, y asi sucesivamente
	vector<Mat>SumaBitPlanes;
	int cont=0;
	for(int i=7;i>=0;i--) {
		Mat Aux(input_img.rows,input_img.cols,CV_8UC(1),Scalar(0));
		if(i<7){
			//Sumatoria de BitPlanes anteriores con actual
			//A cada bitplane lo tengo que multiplicar con pow(2,i) para pasar de binario
			//a un numero entero
			//La primera vez que entra cont=0 
			//Entonces Suma BitPlane[6] con SumaBitPlanes[0] (que tiene el valor del Bit 7)
			//La segunda vez que entra cont=1
			//Entonces Suma BitPlane[5] con SumaBitPlanes[1] (que tiene la suma de Bit 6 y 7)
			//Asi sucesivamente..
			Aux=(BitPlane[i]/255)*pow(2,i)+(SumaBitPlanes[cont]);
			SumaBitPlanes.push_back(Aux);
			cont++;
		}else{
			//La primera vez entra aca Bit7
			SumaBitPlanes.push_back((BitPlane[i]/255)*pow(2,i));
		}
	}
	
	Mat Graficar2=mosaic(SumaBitPlanes,2);
	imshow("Sumatoria BitPlane",Graficar2);
}


void TP2_Ejercicio5_2(Mat entrada){
	Mat img1=imread("a7v600-SE.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2=imread("a7v600-X.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	
	
	imshow("SE",img1);
	imshow("X",img2);
	
	Mat mask= Mat::zeros(img2.size(),img2.type());
	mask(Rect(img2.rows/2-10,img2.cols/2-120,100,100)) = 1;
	Mat resultadox;
	img2.copyTo(resultadox,mask);
	
	Mat maskx= Mat::zeros(img1.size(),img1.type());
	maskx(Rect(img1.rows/2-10,img1.cols/2-120,100,100)) = 1;
	Mat resultadose;
	img1.copyTo(resultadose,maskx);
	
	imshow("ss",resultadose);
	float x,se = 0;
	
	for(int i=150;i<250;++i){
		x += (float)resultadox.at<uchar>(116,i);
		se += (float)resultadose.at<uchar>(116,i);
	}
	x /= (250-150);
	se /= (250-150);
	
	cout<<"Promedio de X: "<<x<<endl<<"Promedio de SE: "<<se<<endl;
	
	Mat mask_input= Mat::zeros(entrada.size(),entrada.type());
	mask_input(Rect(entrada.rows/2-10,entrada.cols/2-120,100,100)) = 1;
	Mat resultado_entrada;
	entrada.copyTo(resultado_entrada,mask_input);
	
	float in = 0;
	
	for(int i=150;i<250;++i){
		in += (float)resultado_entrada.at<uchar>(116,i);
	}
	in /= (250-150);
	cout<<"Promedio de la entrada: "<<in<<endl<<endl;
	if(abs(in - x) < abs(in - se)){
		cout<<"La placa es A7V600-x"<<endl;
	}else{
		cout<<"La placa es A7V600-SE"<<endl;
	}
}



int main(int argc, char** argv) {
	
	//TP2_Ejercicio1(100,100);
	
//	TP2_Ejercicio2(2,2);
	
//	TP2_Ejercicio3();
	
//	TP2_Ejercicio3_2();
	
//	TP2_Ejercicio4();
	
	Mat entrada = imread("a7v600-SE-ruido.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	TP2_Ejercicio5_2(entrada);
	
	waitKey(0);
	return 0;
} 
