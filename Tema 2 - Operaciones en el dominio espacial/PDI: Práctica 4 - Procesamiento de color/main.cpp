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


void TP4_Ejercicio1_1(){
	Mat img_rgb = imread("patron.tif");
	
	vector<Mat>bgr;
	//Split multi-channel image into single-channel arrays
	split(img_rgb,bgr);
//	imshow("Original RGB",img_rgb);
//	imshow("Blue Plane",bgr[0]);
//	imshow("Green Plane",bgr[1]);
//	imshow("Red Plane",bgr[2]);
	
	Mat img_hsv;
	//cvtColor converts the color space to another color space or gray space
	cvtColor(img_rgb,img_hsv,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img_hsv,hsv);
//	cv::print(hsv[0]);  //---->>>>> Sirve para ver los distintos niveles de saturacion que tiene la imagen, va de 0(rojo) a 120(azul)
//	imshow("Original HSV",img_hsv);
//	imshow("H",hsv[0]);
//	imshow("S",hsv[1]);
//	imshow("V",hsv[2]);
	// H es el tono, S la saturacion y V el valor o brillo
	
	//si quiero cambiar el patron, donde habia azul poner rojo y viceversa
	//simplemente debo cambiar la componente del tono (H) haciendo 120-H.
	
	hsv[0]=120-hsv[0]; //Invierte la imagen, ya que los valores cercanos a 0 son rojos y los cercanos a 120 son azules
	Mat img_hsv2;
	Mat	img_rgb2;
	merge(hsv,img_hsv2);
	cvtColor(img_hsv2, img_rgb2, CV_HSV2BGR);
	imshow("RGB invertido",img_rgb2);
	
	cout<<"Nivel de staruacion: "<<(int)hsv[1].at<uchar>(1,1);
	//Como el nivel de saturacion ya es maximo, solo debo modificar el brillo
	hsv[2] = 255;
	merge(hsv,img_hsv2);
	cvtColor(img_hsv2,img_rgb2,CV_HSV2BGR);
	imshow("Saturacion y Brillo al maximo",img_rgb2);
	
}

void TP4_Ejercicio1_2(){
	Mat img = imread("rosas.jpg");
	imshow("Original", img);
	Mat img_hsv;
	cvtColor(img,img_hsv,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img_hsv,hsv);
//	cv::print(hsv[0]);
	hsv[0]=hsv[0]+90; //Complemento del tono, como los colores van de 0 a 180 en c++ y no hasta 360, solo debo sumar 90
//	hsv[1]=255-hsv[1]; //complemento de la saturacion 
	hsv[2]=255-hsv[2]; //complemento del brillo
	//Para sacar el complemento solo se modifican los canales de hue y value, saturation no se toca
	merge(hsv,img_hsv);
	cvtColor(img_hsv,img,CV_HSV2BGR);
	
	imshow("Complementario",img);
}

void TP4_Ejercicio2(){
	Mat img = imread("rio.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Original",img);
	//Realizo el histograma
	Mat canvas(200,400,CV_32F);
	Mat histo=histogram(img,255);
	normalize(histo,histo,0,1,CV_MINMAX);
	draw_graph(canvas,histo);
	imshow("Histograma",canvas);
	//Los niveles de grises de interes rondan entre 0 y 20 aproximadamente
	Mat img_color;
	cvtColor(img,img_color,CV_GRAY2BGR);
	for (int i = 0; i< img_color.rows;i++){
		for (int j=0;j<img_color.cols;j++){
			if(img.at<uchar>(i,j)<25){
				img_color.at<Vec3b>(i,j)[0]=0;  //Blue
				img_color.at<Vec3b>(i,j)[1]=255;  //Green
				img_color.at<Vec3b>(i,j)[2]=255;  //Red
				
			}
		}
	}
	imshow("Modificada",img_color);
}

void TP4_Ejercicio3_1(){
	Mat img = imread("flowers_oscura.tif");
	Mat img1 = imread("flowers_oscura.tif");
	imshow("Original",img);
	vector<Mat>bgr;
	split(img,bgr);
	equalizeHist(bgr[0],bgr[0]);
	equalizeHist(bgr[1],bgr[1]);
	equalizeHist(bgr[2],bgr[2]);
	merge(bgr,img);
	imshow("Imagen equalizada RGB",img);
	
	//Trabajare en HSV
	Mat img_hsv;
	cvtColor(img1, img_hsv,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img_hsv,hsv);
	//Solo se ecualiza el canal V
	equalizeHist(hsv[2],hsv[2]);
	merge(hsv,img_hsv);
	//Vuelvo a RGB para poder mostrarla
	cvtColor(img_hsv,img_hsv,CV_HSV2BGR);
	imshow("Imagen equalizada HSV",img_hsv);
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


void TP4_Ejercicio3_2(){
	Mat img = imread("camino.tif");
	Mat img1 = imread("camino.tif");
	imshow("Original",img);
	Mat kernel = filtro_pasa_alto_suma1(3);
	vector<Mat> bgr;
	split(img,bgr);
	bgr[0] = convolve(bgr[0],kernel);
	bgr[1] = convolve(bgr[1],kernel);
	bgr[2] = convolve(bgr[2],kernel);
	merge(bgr,img);
	imshow("Imagen RGB Filtrada",img);
	
	Mat img_hsv;
	cvtColor(img1,img_hsv,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img_hsv,hsv);
	//Nuevamente, el kernel se aplica solo sobre el plano de brillo o valor
	hsv[2]=convolve(hsv[2],kernel);
	merge(hsv,img_hsv);
	cvtColor(img_hsv,img1,CV_HSV2BGR);
	imshow("Imagen HSV filtrada",img1);
	
}







int main(int argc, char** argv) {
	
//	TP4_Ejercicio1_1();
//	TP4_Ejercicio1_2();
//	TP4_Ejercicio2();
	TP4_Ejercicio3_1();
//	TP4_Ejercicio3_2();

	
	
	waitKey(0);
	return 0;
} 
