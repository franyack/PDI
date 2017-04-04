#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "pdi_functions.h"
#include "utils.h"
#include <iostream>

using namespace cv;
using namespace pdi;
using namespace std;

void TP1_Ejercicio1(){
	
	
	///CARGA y VISUALIZACION
	Mat img = imread("huang1.jpg"/*,CV_LOAD_IMAGE_GRAYSCALE*/);
	imshow("Foto original",img);
	
	///INFORMACION SOBRE LA IMAGEN
	cout<< "Info Imagen:"<<endl<<endl;
	info(img);
	cout<<endl<<endl;
	
	///LEER UN PIXEL
	cout<<"Informacion del pixel"<<endl;
	Vec3b px = img.at<Vec3b>(120,45);
	int b = px.val[0];
	int g = px.val[1];
	int r = px.val[2];
	
	cout << "R: " << r << endl;
	cout << "G: " << g << endl;
	cout << "B: " << b << endl;
	
	
	cout<<img.at<Vec3b>(120,45)/*[2] -----> Preguntar bien para que sirve esto */;  // Esta funcion me indica cuanto vale un pixel en rgb para una imagen de 3 canales
	
	//cout<<img.at<unsigned char>(120,45); ///Esta funcion no entiendo porque me devuelve caracteres 	

	///ESCRIBIR UN PIXEL
	Vec3b color;
	color[0] = 0;
	color[1] = 0;
	color[2] = 255;
	img.at<Vec3b>(Point(120,45)) = color; 
	
	imshow("Foto modificada en un pixel",img);
	
	///DEFINIR Y RECORTAR UNA SUBIMAGEN DE UNA IMAGEN --> ROI (Region Of Interest)
	
	Rect region_of_interest = Rect(1, 1, 121, 46);
	Mat image_roi = img(region_of_interest);
	imshow("ROI",image_roi);
	cout<<endl<<endl;
	cout<< "Info Imagen ROI:"<<endl<<endl;
	info(img);
	
	
	///MOSTAR VARIAS IMAGENES EN UNA SOLA VENTANA -> FUNCION MOSAIC, SOLO CON IMAGENES DEL MISMO TAMAÑO
	
	Mat img2 = imread("futbol.jpg");
	
	Rect trescientos = Rect(1,1,300,300);
	img2 = img2(trescientos);
	
	Mat m = mosaic(img,img2,false);
	
	imshow("Mosaico",m);
	
	/// ------------>>>>>>>>>><<<<<>>>>>>>>>>>><< PREGUNTAR PORQUE LO DE ABAJO NO ANDA
	
//	vector<cv::Mat> m;
//	m.push_back(img);
//	m.push_back(img2);
//	m.push_back(img);
//	m.push_back(img2);
//	
//	Mat muestra = mosaic(m);	
//	
//	imshow("Mosaico",m);
	
	
	
	
	waitKey(0);
	
}


void TP1_Ejercicio2(){
	
	
	///OBTENGO LA INTENSIDAD DEL PIXEL DESEADO, SI ESTÁ EN ESCALA DE GRISES ES DE 0 A 255, SI ESTA EN ESCALA DE COLORES OBTENGO
	///EL VALOR DE LOS 3 COLORES, EN ORDEN [B,G,R,0]. CUIDADO CON EL VALOR QUE ESTA ENTRE <> EN AT
	
	Mat img = imread("futbol.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Foto original",img);
	Scalar intensity = img.at<uchar>(Point(120,120));
	info(img);
	
	cout<<intensity;
	
	cout<<endl<<"Columnas: "<<img.cols<<"       "<<"Filas: "<<img.rows;
	
	///OBTENER LOS VALORES DE CIERTA FILA O COLUMNA Y GRAFICAR
	
	Mat lut(1,img.cols,CV_8U);
	
	
	for(int i=0;i<img.cols;i++){
		lut.at<uchar>(i) = img.at<uchar>(381,i);  /// Si no agrego Point(), se lee (fila,columnas), caso contrario: (Point(columnas,filas))
		
	}
	
	Mat grafico(460,460,CV_8U);
	grafico.setTo(Scalar(0,0,0));
	draw_graph(grafico,lut);
	imshow("Grafico LUT",grafico);
	
	
	///GRAFICAR EL PERFIL DE INTENSIDAD PARA UN SEGMENTO DE INTERES CUALQUIERA
	
	
	
	
	waitKey(0);
}

void perfilIntensidad(Mat img, int x1, int y1,int x2, int y2){
	int koko,deltaY,deltaX,m,b,y;
	if(y1 == y2){
		koko = abs(x2-x1);
		
	}else{
		if(x1 == x2){
			koko = abs(y2-y1);
		}else{
			deltaY = abs(y2-y1);
			deltaX = abs(x2-x1);
			m = deltaY/deltaX;
			b= y1 - m*x1;
			koko = sqrt(deltaY*deltaY + deltaX+deltaX);
		}
	}
	
	
	Mat aux(1,koko,CV_8U);
	
	if(y1==y2){
		for(int i=0; i<koko ;i++){
			aux.at<uchar>(i) = img.at<uchar>(x1,i);
		}
	}
	if(x1==x2){
		for(int i=0; i<koko ;i++){
			aux.at<uchar>(i) = img.at<uchar>(i,y1);
		}
	}else{
		for(int i=1; i<deltaX ;i++){
			y = m*x1 + b;
			aux.at<uchar>(i) = img.at<uchar>(y,x1);
		}
	}
	
	
	
	



	Mat grafico(koko+1,koko+1,CV_8U);
	grafico.setTo(Scalar(0,0,0));
	draw_graph(grafico,aux);
	imshow("Grafico",grafico);
	waitKey(0);
}


int main(int argc, char** argv) {
	
//	TP1_Ejercicio1();
//	TP1_Ejercicio2();
	Mat img = imread("futbol.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	perfilIntensidad(img, 1, 1,228, 228);
	
	return 0;
} 
