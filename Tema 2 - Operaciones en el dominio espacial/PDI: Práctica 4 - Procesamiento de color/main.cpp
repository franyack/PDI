#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "pdi_functions.h"
#include "utils.h"
#include "utils2.h"
#include "utils3.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <bitset>

using namespace cv;
using namespace pdi;
using namespace std;



//usados para recortar una imagen a partir de otra
Rect cropRect(0, 0, 0, 0);
Point P1(0, 0);
Point P2(0, 0);

void Mouse(int event, int x, int y, int flags, void* userdata) { 
	int x_ini = 0; int y_ini = 0; int x_fin = 0; int y_fin = 0;
	
	switch (event) {
	case CV_EVENT_LBUTTONDOWN:
		P1.x = x;
		P1.y = y;
		P2.x = x;
		P2.y = y;
		cout << "Boton izquierdo presionado en las coord (x, y): " << x << " , " << y << endl;
		break;
	case CV_EVENT_LBUTTONUP:
		P2.x = x;
		P2.y = y;
		cout << "Boton izquierdo liberado en las coord (x,y): " << x << " , " << y << endl;
		break;
	default: break;
	}
}
void crop_mouse(string image) { //Funcion para armar un roi a traves de seleccion de ventana en imagen
	//	Utilizo la funcion "Mouse" definida arriba. Solo recortar desde arriba-izq hacia abajo-derecha
	Mat img = imread(image, 1);
	namedWindow("ImageDisplay", 1);
	setMouseCallback("ImageDisplay", Mouse, NULL);
	imshow("ImageDisplay", img);
	waitKey(10000);
	
	int ancho = P2.x - P1.x;
	int alto = P2.y - P1.y;
	Rect r_crop(P1.x, P1.y, ancho, alto);
	cout << P1.x << ", " << P2.x << ", " << P1.y << ", " << P2.y << endl;
	Mat img_crop = img(r_crop);
	namedWindow("cropped", 1);
	imshow("cropped", img_crop);
	waitKey(0);
	system("PAUSE");
	
}




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
//	vector<Mat>bgr;
//	split(img,bgr);
//	equalizeHist(bgr[0],bgr[0]);
//	equalizeHist(bgr[1],bgr[1]);
//	equalizeHist(bgr[2],bgr[2]);
//	merge(bgr,img);
	imshow("Imagen equalizada RGB",equalizarRGB(img));
	
	//Trabajare en HSV
//	Mat img_hsv;
//	cvtColor(img1, img_hsv,CV_BGR2HSV);
//	vector<Mat>hsv;
//	split(img_hsv,hsv);
//	//Solo se ecualiza el canal V
//	equalizeHist(hsv[2],hsv[2]);
//	merge(hsv,img_hsv);
//	//Vuelvo a RGB para poder mostrarla
//	cvtColor(img_hsv,img_hsv,CV_HSV2BGR);
	imshow("Imagen equalizada HSV",equalizarHSV(img1));
}



void TP4_Ejercicio3_2(){
	Mat img = imread("camino.tif");
	Mat img1 = imread("camino.tif");
	imshow("Original",img);
//	Mat kernel = filtro_pasa_alto_suma1(3);
//	vector<Mat> bgr;
//	split(img,bgr);
//	bgr[0] = convolve(bgr[0],kernel);
//	bgr[1] = convolve(bgr[1],kernel);
//	bgr[2] = convolve(bgr[2],kernel);
//	merge(bgr,img);
	
	imshow("Imagen RGB Filtrada",realceMedianteAcentuadoRGB(img));
	
//	Mat img_hsv;
//	cvtColor(img1,img_hsv,CV_BGR2HSV);
//	vector<Mat>hsv;
//	split(img_hsv,hsv);
//	//Nuevamente, el kernel se aplica solo sobre el plano de brillo o valor
//	hsv[2]=convolve(hsv[2],kernel);
//	merge(hsv,img_hsv);
//	cvtColor(img_hsv,img1,CV_HSV2BGR);
	imshow("Imagen HSV filtrada",realceMedianteAcentuadoHSV(img1));
	
}


void TP4_Ejercicio4_1(){
	Mat img = imread("futbol.jpg");
	imshow("Original",img);
	Mat roi=img(Rect(135,155,52,80));
	imshow("ROI",roi);
	vector<Mat>bgr;
	split(roi,bgr);

	//Hago los histogramas para saber que region utilizar luego.
	Mat histoB=histogram(bgr[0],255);
	Mat histoG=histogram(bgr[1],255);
	Mat histoR=histogram(bgr[2],255);
	
	normalize(histoB,histoB,0,1,CV_MINMAX);
	normalize(histoG,histoG,0,1,CV_MINMAX);
	normalize(histoR,histoR,0,1,CV_MINMAX);
	
//	for(int i=0;i<histoB.cols;i++) { 
//		for(int j=0;j<histoB.rows;j++) { 
//			cout<<(float)histoB.at<uchar>(i,j)<<endl;
//		}
//	}
	
	Mat canvasB(200,400,CV_32F);
	Mat canvasG(200,400,CV_32F);
	Mat canvasR(200,400,CV_32F);
	
	draw_graph(canvasB,histoB);
	draw_graph(canvasG,histoG);
	draw_graph(canvasR,histoR);
	
	imshow("B",canvasB);
	imshow("G",canvasG);
	imshow("R",canvasR);
	
	Mat mascara= Mat::zeros(img.size(), img.type());
	Mat segmentacionRGB= Mat::zeros(img.size(), img.type());    
	inRange(img,Scalar(100,17,0),Scalar(212,113,60),mascara); //regiones aproximadas sacadas de los histogramas
	img.copyTo(segmentacionRGB, mascara);
	imshow("Segmentacion RGB",segmentacionRGB);
	
}

void TP4_Ejercicio4_2(){
	Mat img = imread("rostro0.png");
	imshow("Original",img);
	//Comienza la busqueda del Tono y Saturacion para realizar la segmentacion, el brillo no es necesario
	//Defino el roi
//	Mat roi=img(Rect(150,60,60,100)); //cara de Marcelo  Rect(x,y,cuantosx,cuantosy)
	Mat roi=img(Rect(138,114,42,66)); //cara de Cesar
	imshow("ROI",roi);
	Mat roi_hsv;
	//Convierto el ROI de BGR a HSV
	cvtColor(roi,roi_hsv,CV_BGR2HSV);
	vector<Mat>hsv;
	//Lo divido en los 3 planos
	split(roi_hsv,hsv);
	
	//Realizo los 2 histogramas para ver los rangos de interes
	Mat histH = histogram(hsv[0],255);
	Mat histS = histogram(hsv[1],255);
	
	normalize(histH,histH,0,1,CV_MINMAX);
	normalize(histS,histS,0,1,CV_MINMAX);
	
	Mat canvasH(200,400,CV_32F);
	Mat canvasS(200,400,CV_32F);
	
	draw_graph(canvasH,histH);
	draw_graph(canvasS,histS);
	
	imshow("Histo H", canvasH);
	imshow("Histo S", canvasS);
	
	//Ahora paso a realizar el procesamiento sobre la imagen original
	//Convierto la imagen original de BGR a HSV
	cvtColor(img,img,CV_BGR2HSV);
	//Creo la mascara y la matriz de segmentacion
	Mat mascara= Mat::zeros(img.size(), img.type());
	Mat segmentacion= Mat::zeros(img.size(), img.type());
    //Busco con esta funcion, en img, los valores que se definen en Scalar y los pego en la mascara
//	inRange(img,Scalar(6,0,0),Scalar(30,100,255),mascara); //regiones aproximadas sacadas de los histogramas  --->>> Rostro de Marcelo
	inRange(img,Scalar(1,1,0),Scalar(35,70,255),mascara); //regiones aproximadas sacadas de los histogramas  ---->Rostro de Cesar
	//con copyTo hago una copia en segmentacion de la multiplicacion entre img y mascara, descartando
	//aquellos valores que no son iguales a los encontrados en inRange
	img.copyTo(segmentacion, mascara);
	//Convierto segmentacion de HSV a BGR para poder visualizarlo
	cvtColor(segmentacion,segmentacion,CV_HSV2BGR);
	imshow("Segmentacion HSV",segmentacion);
	
}




int main(int argc, char** argv) {
	
//	TP4_Ejercicio1_1();
//	TP4_Ejercicio1_2();
//	TP4_Ejercicio2();
//	TP4_Ejercicio3_1();
//	TP4_Ejercicio3_2();
	TP4_Ejercicio4_1();
//	TP4_Ejercicio4_2();
//	Averiguar();
//	crop_mouse("huang2.jpg");
	
	waitKey(0);
	return 0;
} 
