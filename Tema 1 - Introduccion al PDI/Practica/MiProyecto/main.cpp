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
//	cout<< img.type;
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
	
	
	///OBTENER LOS VALORES DE CIERTA FILA O COLUMNA Y GRAFICAR
	
	Mat lut(1,382,CV_8U);
	
	
	for(int i=0;i<382;i++){
		lut.at<uchar>(i) = img.at<uchar>(i,200);
		
	}
	
	Mat grafico(400,400,CV_8U);
	grafico.setTo(Scalar(0,0,0));
	draw_graph(grafico,lut);
	imshow("Grafico LUT",grafico);
	
	
	///GRAFICAR EL PERFIL DE INTENSIDAD PARA UN SEGMENTO DE INTERES CUALQUIERA
	
	
	
	waitKey(0);
}


int main(int argc, char** argv) {
	
//	TP1_Ejercicio1();
	TP1_Ejercicio2();
	
	
	return 0;
} 
