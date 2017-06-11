#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "pdi_functions.h"
#include "utils.h"

using namespace cv;
using namespace std;
using namespace pdi;



vector<Point> xy;
int cantclick = 2; 
bool reset = false;

void Click(int event, int x, int y, int flags, void* userdata){
	if(reset){
		xy.clear();
		reset = false;
	}
	if(event == EVENT_LBUTTONDOWN){
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		xy.push_back(Point(x,y));
		if(xy.size()==cantclick){
			reset = true;
			destroyWindow("My Window");
		}
	}
}

//Devuelve un Roi haciendo 2 clicks
Mat makeRoi(Mat imgInput){
	cantclick=2; 
	namedWindow("My Window",CV_WINDOW_KEEPRATIO);
	imshow("My Window",imgInput);
	setMouseCallback("My Window", Click, NULL);
	waitKey(0);
	if(xy[0].x>xy[1].x) swap(xy[0].x,xy[1].x); //Para que no explote si hace al revez los clicks
	if(xy[0].y>xy[1].y) swap(xy[0].y,xy[1].y);
	//ROI de la imagen
	Mat roi=imgInput(Rect(xy[0].x,xy[0].y,xy[1].x-xy[0].x,xy[1].y-xy[0].y));
	return roi;
}

/*void onMouse( int event, int x, int y, int, void* );*/
/*string nombreImagen="Examen2017/1.png";*/
//Mat imagen = imread(nombreImagen);
void onMouse( int event, int x, int y, int, void* )
{
	if( event != CV_EVENT_LBUTTONDOWN )
		return;
	
	Point pt = Point(x,y);
	cout<<"x="<<pt.x<<"\t y="<<pt.y;
}
//Con esta funcion averiguo el x y de un pixel que hago click
void Averiguar(string nombreVentana,Mat imagen){
	namedWindow(nombreVentana);
	setMouseCallback(nombreVentana, onMouse, 0 );
	imshow(nombreVentana,imagen);
	waitKey(0);
}

void histogramaHSV(vector<Mat>hsv){
	//Histograma H - S - V
	Mat canvas_HSV_H(256,256,CV_32F);
	Mat canvas_HSV_S(256,256,CV_32F);
	Mat canvas_HSV_V(256,256,CV_32F);
	Mat histo_HSV_H = histogram(hsv[0],256,Mat());
	Mat histo_HSV_S = histogram(hsv[1],256,Mat());
	Mat histo_HSV_V = histogram(hsv[2],256,Mat());
	normalize(histo_HSV_H,histo_HSV_H,0,1,CV_MINMAX);
	normalize(histo_HSV_S,histo_HSV_S,0,1,CV_MINMAX);
	normalize(histo_HSV_V,histo_HSV_V,0,1,CV_MINMAX);
	draw_graph(canvas_HSV_H,histo_HSV_H);
	draw_graph(canvas_HSV_S,histo_HSV_S);
	draw_graph(canvas_HSV_V,histo_HSV_V);
	Averiguar("Histo H",canvas_HSV_H);
	Averiguar("Histo S",canvas_HSV_S);
	Averiguar("Histo V",canvas_HSV_V);
	//	imshow("Histograma HSV_H",canvas_BGR_B);
	//	imshow("Histograma HSV_S",canvas_HSV_S);
	//	imshow("Histograma HSV_V",canvas_HSV_V);
}
void parcialBrujula(Mat imgInput){
	///Trabajo en una copia de la imagen
	Mat imgCopy;
	imgInput.copyTo(imgCopy);
	
	///Hago roi para ver los componentes HSV del color de la brujula
	//Tener en cuenta de agarrar un buen ROI, porque el color rojo de la brujula no es uniforme
	Mat roi = makeRoi(imgCopy);
	namedWindow("Roi",CV_WINDOW_KEEPRATIO);
	imshow("Roi",roi);
	
	///Calculo componentes HSV
	vector<Mat>HSV;
	split(roi,HSV);
	
	///Hago histogramas para ver minimos y maximo de HSV
	//Haciendo click en la zona de interes averiguo las posiciones
//	histogramaHSV(HSV); //Funcion definida mas arriba
	//Los valores de abajo los obtuve con la funcion histogramaHSV haciendo click en cada imagen
	//min,max de H,S,V
	MostrarHistogramas(roi);
	
	float min_H=0;  
	float min_S=0;
	float min_V=129;
	float max_H=87;
	float max_S=88;
	float max_V=256;
	
	//Creo mascara para segmentar
	Mat mascaraHSV;
	inRange(imgCopy,Scalar(min_H,min_S,min_V),Scalar(max_H,max_S,max_V),mascaraHSV);
	//	imshow("Mascara para segmentar",mascaraHSV);
	
	///Segmento
	//La imagen original la copio a "segmentada" teniendo en cuenta la mascaraHSV
	//Es decir, va a copiar solo las posiciones coincidentes con la mascaraHSV
	Mat segmentada;
	imgCopy.copyTo(segmentada,mascaraHSV);
	imshow("Imagen Segmentada",segmentada);
	
	//Con Hough obtengo las lineas y el grado
	//Necesito pasar a escala de GRISES para detectar los bordes con Canny
	Mat segmentadaGRIS;
	cvtColor(segmentada,segmentadaGRIS,CV_BGR2GRAY);
	Mat Gradiente;
	Canny(segmentadaGRIS,Gradiente,50,200,3); //Detecto los bordes de la imagen
	//	imshow("Bordes",Gradiente);
	
	///Esto lo agregue despues de que anduvo para los primeros dos casos
	//Busco donde esta el Norte(Esta en color rojo por lo que al segmentar 
	//puedo encontrar la ubicacion), si no esta en el medio roto la imagen
	//Necesito el vertice que esta arriba de la "N", por lo que voy a hacer un circulo de 
	//radio=200-35 para usar como mascara binaria
	int radio=200-35;
	Mat mascaraRadio(imgInput.cols,imgInput.rows,CV_8UC1,Scalar(0)); //matriz de ceros
	circle(mascaraRadio,Point(200,200),radio,Scalar(255));
	imshow("Circulo",mascaraRadio);
	Mat posicionPunto;
	Gradiente.copyTo(posicionPunto,mascaraRadio);
	imshow("Posicion Punto",posicionPunto);
	//Buscoa el pixel pintado para obtener la posicion x,y y con eso sacar el angulo
	//Que necesito rotar
	int fila,columna;
	for(int i=0;i<posicionPunto.rows;i++) { 
		for(int j=0;j<posicionPunto.cols;j++) { 
			if(posicionPunto.at<uchar>(i,j)!=0){
				fila=i; //Seria mi y en el eje de coordenadas (x,y)
				columna=j; //Seria mi x en el eje de coordenadas (x,y)
			}
		}
	}
	
	bool izquierda=false;
	bool derecha=false;
	if(columna<180) izquierda=true;
	if(columna>210) derecha=true;
	
	if(izquierda) Gradiente=rotate(Gradiente,270);
	if(derecha) Gradiente=rotate(Gradiente,45);	
	
	//Calculo hough
	//lines contiene en la primer posicion del vector el angulo en radianes donde hay
	//mayor coincidencia de puntos
	
	vector<Vec2f> lines;
	HoughLines(Gradiente,lines,1,CV_PI/180,50,0,0);
	float grados=lines[0][1]*180/CV_PI; //Convierto de radianes a grados
	
	///El hough va de 0 a 180, y no sabe el cuadrante que nosotros queremos analizar
	///Si la brujula esta en el 1 o 4 cuadrante ( zona derecha), el angulo que nos devuelve esta bien
	///Si la brujula esta en el 2 o 3 cuadrante ( zona izquierda), debemos sumarle 180 grados
	
	//Para hacer esto, elijo una columna fija que contenga a la brujula
	//Recorro todas las filas, y si encuentra un pixel distinto de 0 quiere decir
	//que encontro a la brujula, es decir esta en la zona izquierda por lo que sumamos 180 grados
	
	bool bandera = false;
	for(int i=0;i<Gradiente.cols;i++) { 
		if(Gradiente.at<uchar>(i,170)!=0) bandera=true;
	}
	if(bandera) grados=grados+180;
	
	cout<<"Grados: "<<grados<<endl;
	
	waitKey(0);
}


int main(int argc, char** argv) {
	//	EjecucionBilletes();
	Mat imgInput1=imread("1.png");
	Mat imgInput2=imread("2.png");
	Mat imgInput3=imread("3.png");
	Mat imgInput4=imread("4.png");
	
	parcialBrujula(imgInput1);
//	parcialBrujula(imgInput2);
//	parcialBrujula(imgInput3);
//	parcialBrujula(imgInput4);
	waitKey(0);
	return 0;
} 
