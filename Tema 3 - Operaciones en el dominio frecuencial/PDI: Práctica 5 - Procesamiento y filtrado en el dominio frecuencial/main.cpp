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



Mat TP5_Ejercicio1_1(int tipo){
	Mat figura(512,512,CV_32F);
	switch (tipo){
	case 1:{ //linea horizontal
		//		line(figura,Point(0,figura.rows/2-100),Point(figura.cols,figura.rows/2-100),Scalar(1));
		//Este for si quiero hacer una linea con mayor grosor
		//		for(int i=(figura.rows/2)-2;i<(figura.rows/2)+2;i++) { 
		//			for(int j=0;j<figura.cols;j++) { 
		//				figura.at<float>(i,j)=1;
		//			}
		//		}
		line(figura,Point(0,figura.rows/2),Point(figura.cols,figura.rows/2),Scalar(1));
		//		line(figura,Point(0,figura.rows/2+100),Point(figura.cols,figura.rows/2+100),Scalar(1));
		break;}
	case 2:{ //linea vertical
			line(figura,Point(figura.cols/2,0),Point(figura.cols/2,figura.rows),Scalar(1));
			break;}
	case 3:{ //cuadrado centrado
			rectangle(figura,Point((figura.cols/2)-30,(figura.rows/2)-30),Point((figura.cols/2)+30,(figura.rows/2)+30),Scalar(1));
//			for para hacer cuadrado relleno
//			for(int i=figura.rows/2-30;i<figura.rows/2+30;i++) { 
//				for(int j=figura.cols/2-30;j<figura.cols/2+30;j++) { 
//					figura.at<float>(i,j)=1;
//				}
//			}
				break;}
	case 4:{ //rectangulo centrado
				rectangle(figura,Point((figura.cols/2)-40,(figura.rows/2)-20),Point((figura.cols/2)+40,(figura.rows/2)+20),Scalar(1));	
				break;}
	case 5:{ //circulo centrado
				circle(figura,Point(figura.cols/4,figura.rows/4),100,Scalar(1));
				break;}
	default:{
				cout<<"Figura Invalida"<<endl;
				break;}
	}
	return figura;
}

void TP5_Ejercicio1_2(Mat img){
	Mat transformada(img.size(),img.type());
	transformada=spectrum(img);
	namedWindow("Espectro de Intensidad",CV_WINDOW_KEEPRATIO);
	imshow("Espectro de Intensidad",transformada);
	imshow("Imagen espacio",img);
}

void rotate(cv::Mat& src, double angle, cv::Mat& dst){ //Funcion para rotar una imagen.
	cv::Point2f ptCp(src.cols*0.5, src.rows*0.5);
	cv::Mat M = cv::getRotationMatrix2D(ptCp, angle, 1.0);
	cv::warpAffine(src, dst, M, src.size(), cv::INTER_CUBIC); //Nearest is too rough, 
}

void TP5_Ejercicio1_3(){
	Mat img(512,512,CV_32F);
	line(img,Point(img.cols/2,0),Point(img.cols/2,img.rows),Scalar(1));
	Mat roi1=img(Rect(img.cols/2-128,img.rows/2-128,256,256));
	imshow("Imagen Original",roi1);
	imshow("Espectro Img Original",spectrum(roi1));
	rotate(img,20,img);
	Mat roi2=img(Rect(img.cols/2-128,img.rows/2-128,256,256));
	imshow("Imagen Rotada",roi2);
	imshow("Espectro Img Rotada",spectrum(roi2));
}

void TP5_Ejercicio1_4(string nombre){
	Mat img=imread(nombre,CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img,CV_32F,1./255);
	imshow("Original",img);
	imshow("Magnitud",spectrum(img));
}

void TP5_Ejercicio2(string nombre,string pasa,int tipo,double D0,int orden){
	Mat img=imread(nombre,CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img,CV_32F,1./255);
	int filas=img.rows;
	int columnas=img.cols;
	img=optimum_size(img); //Copia la imagen en una cuya dimensiones hacen eficiente la fft
	Mat filtro;
	switch (tipo){
	case 1:{
		filtro=filter_ideal(img.rows,img.cols,D0); //filtro ideal
		break;}
	case 2:{
		filtro=filter_butterworth(img.rows,img.cols,D0,orden); //filtro de Butterworth
		break;}
	case 3:{
		filtro=filter_gaussian(img.rows,img.cols,D0); //Filtro Gaussiano
		break;}
	}
	if (pasa=="Pasa Alto"){
		filtro=1-filtro; //Si es pasa alto se invierte todo, ya que ahora lo que no se filtra es el circulo
		//central que es donde estan las frecuencias bajas y se dejan pasar el resto que son las altas
	}
	Mat filtrada=filter(img,filtro);
	filtrada=filtrada(Range(0,filas),Range(0,columnas));
	img=img(Range(0,filas),Range(0,columnas));
	imshow("Original",img);
	imshow("Espectro Original",spectrum(img));
	namedWindow("Imagen Filtrada",CV_WINDOW_KEEPRATIO);
	imshow("Imagen Filtrada",filtrada);
	imshow("Espectro filtro",spectrum(filtrada));
}

void TP5_Ejercicio3(float A, float a, float b){
	Mat img=imread("camaleon.tif",CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img,CV_32F,1./255);
	int filas=img.rows;
	int columnas=img.cols;
	img=optimum_size(img);
	//Filtrado de alta potencia (high boost)
	Mat high_boost=filter_butterworth(img.rows,img.cols,50/255.0,2);
	high_boost=1-high_boost;
	high_boost=(A-1)+high_boost;
	Mat filtrado_AltaPotencia=filter(img,high_boost);
	filtrado_AltaPotencia=filtrado_AltaPotencia(Range(0,filas),Range(0,columnas));
	img=img(Range(0,filas),Range(0,columnas));
	
	//Filtrado de Alta Frecuencia
	img=optimum_size(img);
	//Filtrado de alta potencia (high boost)
	Mat enfasis=filter_butterworth(img.rows,img.cols,50/255.0,2);
	enfasis=1-enfasis;
	enfasis=a+b*enfasis;
	Mat filtrado_AltaFrecuencia=filter(img,enfasis);
	filtrado_AltaFrecuencia=filtrado_AltaFrecuencia(Range(0,filas),Range(0,columnas));
	img=img(Range(0,filas),Range(0,columnas));
	
	imshow("Original",img);
	imshow("Espectro Original",spectrum(img));
	namedWindow("Filtrado de Alta Potencia",CV_WINDOW_KEEPRATIO);
	imshow("Filtrado de Alta Potencia",filtrado_AltaPotencia);
	imshow("Espectro Filtro Alta Potencia",spectrum(filtrado_AltaPotencia));
	namedWindow("Filtrado de Alta Frecuencia",CV_WINDOW_KEEPRATIO);
	imshow("Filtrado de Alta Frecuencia",filtrado_AltaFrecuencia);
	imshow("Espectro Filtro Alta Frecuencia",spectrum(filtrado_AltaFrecuencia));
}

void TP5_Ejercicio4(float yl, float yh, double D0){
	Mat img=imread("casilla.tif",CV_LOAD_IMAGE_GRAYSCALE);
	Mat ecualizada;
	equalizeHist(img,ecualizada);
	imshow("Original",img);
	imshow("Original Ecualizada",ecualizada);
	
	img.convertTo(img,CV_32F,1./255);
	imshow("Spec Original",spectrum(img));
	img=img+0.00001;
	log(img,img);
	int filas=img.rows;
	int columnas=img.cols;
	img=optimum_size(img);
	Mat filtro;
	filtro=(yh-yl)*(1-filter_gaussian(img.rows,img.cols,D0))+yl; 
	//	Mat espectro=spectrum(filtro);
	Mat filtrada=filter(img,filtro);
	filtrada=filtrada(Range(0,filas),Range(0,columnas));
	img=img(Range(0,filas),Range(0,columnas));
	exp(filtrada,filtrada);
	imshow("Filtro Homomorfico",filtrada);
	//	stats(filtrada);
	imshow("Spec Filtrada",spectrum(filtrada));
	normalize(filtrada,filtrada,0,255,CV_MINMAX);
	filtrada.convertTo(filtrada,ecualizada.type());
	equalizeHist(filtrada,filtrada);
	//	stats(filtrada);
	
	imshow("Homomorfico Ecualizado",filtrada);
	//	imshow("espectro",filtro);
	waitKey(0);
}



int main(int argc, char** argv) {
//	Mat img;
//	img = TP5_Ejercicio1_1(5);
//	TP5_Ejercicio1_2(img);	
//	TP5_Ejercicio1_3();
//	TP5_Ejercicio1_4("huang3.jpg");
//	TP5_Ejercicio2("huang3.jpg","Pasa Bajo",3,25/255.0,3);
//	TP5_Ejercicio3(1,2,6);
	TP5_Ejercicio4(0.7,1.3,0.1);
	waitKey(0);
	return 0;
} 
