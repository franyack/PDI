#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include "pdi_functions.h"
#include <vector>
#include <opencv2/core/cvdef.h>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iterator>
#include <opencv2/core/types.hpp>

using namespace cv;
using namespace pdi;
using namespace std;

void onMouse( int event, int x, int y, int, void* );
//
Mat imagen = imread("HeadCT_degradada_spectrum.jpg");
//imagen.converTo(imagen,CV_32F,1./255);
//Mat imagen=spectrum(imagen1);
//Mat imagen=aux(Rect(0,740,132,44));
void onMouse( int event, int x, int y, int, void* )
{
	if( event != CV_EVENT_LBUTTONDOWN )
		return;
	
	Point pt = Point(x,y);
	cout<<"x="<<pt.x<<"\t y="<<pt.y;
	Vec3b valores=imagen.at<cv::Vec3b>(y,x);
	cout<<"\t B="<<(int)valores.val[0]<<" G="<<(int)valores.val[1]<<" R="<<(int)valores.val[2]<<endl;
	
}
void Averiguar(){
	//EJERCICIO 2.1
	namedWindow("Averiguar Pixeles",CV_WINDOW_KEEPRATIO);
	setMouseCallback( "Averiguar Pixeles", onMouse, 0 );
	imshow("Averiguar Pixeles",imagen);
	
}

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



Mat ruido_sal_pimienta(Mat img, float pa, float pb ){   
	//pa y pb cuantos de sal y cuantos de pimienta agrego.
	RNG rng; // rand number generate
	
	int cantidad1=img.rows*img.cols*pa; //determino cuantos pixeles debo cambiar en total,
	//entonces cuanto menor sea la cantidad de pa y pb menos pixeles va a cambiar
	int cantidad2=img.rows*img.cols*pb;
	for(int i=0; i<cantidad1; i++){
		img.at<uchar>(rng.uniform(0,img.rows), rng.uniform(0, img.cols)) =0; //elige valores al azar de la imagen
		//y los pone en negro. Esto lo hace cantidad1 de veces que es la cantidad de pixeles que quiero poner de pimienta.
	}
	for (int i=0; i<cantidad2; i++){
		img.at<uchar>(rng.uniform(0,img.rows), rng.uniform(0,img.cols)) = 255;
	}
	return img;
}

Mat ruido_gaussiano(Mat img,double mean,double sigma){
	Mat ruido = img.clone();
	//	img.convertTo(img,CV_32F,1./255);
	//	img.convertTo(ruido,CV_32F,1./255);
	RNG rng;
	rng.fill(ruido, RNG::NORMAL, mean,sigma); 
	add(img, ruido, img);
	return img;
}

void Ejercicio1(){
	Mat img=imread("huang2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat canvas_original(img.rows,img.cols,CV_32F);
	Mat histo_original=histogram(img,255);
	normalize(histo_original,histo_original,0,1,CV_MINMAX);
	draw_graph(canvas_original,histo_original);
	
	//Ruido sal y pimienta
	Mat syp=img.clone();
	syp=ruido_sal_pimienta(syp,30/255.0,30/255.0);
	Mat canvas_syp(syp.rows,syp.cols,CV_32F);
	Mat histo_syp=histogram(syp,255);
	normalize(histo_syp,histo_syp,0,1,CV_MINMAX);
	draw_graph(canvas_syp,histo_syp);
	
	//Ruido Gaussiano
	Mat rg=img.clone();
	rg=ruido_gaussiano(rg,0,0.30*255);
	Mat canvas_rg(rg.rows,rg.cols,CV_32F);
	Mat histo_rg=histogram(rg,255);
	normalize(histo_rg,histo_rg,0,1,CV_MINMAX);
	draw_graph(canvas_rg,histo_rg);
	
	imshow("Original",img);
	imshow("Histograma Original",canvas_original);
	imshow("Ruido Sal y Pimienta",syp);
	imshow("Histograma Sal y Pimienta",canvas_syp);
	imshow("Ruido Gaussiano",rg);
	imshow("Histograma Ruido Gaussiano",canvas_rg);
	waitKey(0);
}

Mat MediaGeometrica(Mat img,int tam){ //BUENO PARA RUIDO GAUSSIANO, MALO PARA IMPULSIVO
	img.convertTo(img,CV_32F,1./255);
	Mat img2=img.clone();
	int m=tam/2;
	for(int i=m;i<img.rows-m;i++) { 
		for(int j=m;j<img.cols-m;j++) { 
			for(int tt=0;tt<3;tt++){
				float aux=1;
				for(int k=-m;k<=m;k++) { 
					for(int l=-m;l<=m;l++) { 
						aux*=img.at<float>(i+k,j+l);
					}
				}
				img2.at<float>(i,j)=pow(aux,1./(tam*tam));
			}
		}
	}
	img=img2;
	return img;
}

//MEDIA ARMONICA BUENO PARA SAL, MALO PARA PIMIENTA, BUENO CON GAUSSIANO.
Mat MediaContraArmonica(Mat img,int tam,float Q){ //BUENO PARA RUIDO SAL Y PIMIENTA. Q>0 ELIMINA PIMIENTA, Q<0 ELIMINA SAL, 
	//Q=0 MEDIA ARITMETICA, Q=-1 MEDIA ARMONICA
	img.convertTo(img,CV_32F,1./255);
	Mat img2=img.clone();
	int m=tam/2;
	for(int i=m;i<img.rows-m;i++) { 
		for(int j=m;j<img.cols-m;j++) { 
			for(int tt=0;tt<3;tt++){
				float aux1=0;
				float aux2=0;
				for(int k=-m;k<=m;k++) { 
					for(int l=-m;l<=m;l++) { 
						aux1+=pow(img.at<float>(i+k,j+l),Q+1);
						aux2+=pow(img.at<float>(i+k,j+l),Q);
					}
				}
				img2.at<float>(i,j)=aux1/aux2;
			}
		}
	}
	img=img2;
	return img;
}
float ECM(Mat img1,Mat img2){
	img1.convertTo(img1,CV_32F,1./255);
	img2.convertTo(img2,CV_32F,1./255);
	float error=0;
	if(img1.rows==img2.rows && img1.cols==img2.cols){
		for(int i=0;i<img1.rows;i++) { 
			for(int j=0;j<img1.cols;j++) { 
				error+=pow(img1.at<float>(i,j)-img2.at<float>(i,j),2);
			}
		}
		error=sqrt(error/(img1.rows*img1.cols));
	}else{
		cout<<endl<<"TIENEN QUE TENER EL MISMO TAMANO"<<endl;
	}
	return error;
}

//POSIBLE PREGUNTA DE PARCIAL, SACADA DEL FACU C.
//Si tengo ruido impulsivo y gaussiano.. que sacamos primero? el impulsivo, pq si saco el gaussiano primero, 
//me 'borronea' la sal y pimienta


void Ejercicio2(){
	Mat img=imread("sangre.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Original",img);
	
	Mat ruido=img.clone();
//	ruido=ruido_sal_pimienta(ruido,0,10.0/255); 
	ruido=ruido_gaussiano(ruido,0,0.10*255);
	imshow("Con ruido",ruido);
	
	Mat fmg=ruido.clone(); 
	fmg=MediaGeometrica(fmg,3); //bueno para ruido gausiano, malo para ruido impulsivo. De todas formas si
								//tengo solo ruido solo sal, tambien lo soluciona
	imshow("Filtro Media Geometrica",fmg);
	
	Mat fca=ruido.clone();
	fca=MediaContraArmonica(fca,3,-3); //-3 por tener ruido sal, si fuera pimienta tengo que poner positivo
	imshow("Filtro Media Contra Armonica",fca);
	
	cout<<"El Error Cuadratico Medio (ECM) entre la imagen original y el ruido es: "<<ECM(img,ruido)<<endl;
	cout<<"El Error Cuadratico Medio (ECM) entre la imagen original y filtrada por Media Geometrica es: "<<ECM(img,fmg)<<endl;
	cout<<"El Error Cuadratico Medio (ECM) entre la imagen original y filtrada por Media Contra Armonica es: "<<ECM(img,fca)<<endl;
	waitKey(0);
}

Mat OrdenMediana(Mat img,int tam){ //BUENO PARA RUIDOS IMPULSIVOS SIN DESENFOQUE
	img.convertTo(img,CV_32F,1./255);
	Mat img2=img.clone();
	int m=tam/2;
	for(int i=m;i<img.rows-m;i++) { 
		for(int j=m;j<img.cols-m;j++) { 
			for(int tt=0;tt<3;tt++){
				vector<float> aux;
				for(int k=-m;k<=m;k++) { 
					for(int l=-m;l<=m;l++) { 
						aux.push_back(img.at<float>(i+k,j+l));
					}
				}
				sort(aux.begin(),aux.end());
				img2.at<float>(i,j)=aux[((tam*tam)/2)+1];
			}
		}
	}
	img=img2;
	return img;
}

Mat OrdenPuntoMedio(Mat img,int tam){ //UTIL PARA RUIDO GAUSSIANO O UNIFORME
	img.convertTo(img,CV_32F,1./255);
	Mat img2=img.clone();
	int m=tam/2;
	for(int i=m;i<img.rows-m;i++) { 
		for(int j=m;j<img.cols-m;j++) { 
			for(int tt=0;tt<3;tt++){
				float aux1=0;
				float aux2=1;
				for(int k=-m;k<=m;k++) { 
					for(int l=-m;l<=m;l++) { 
						if(aux1<img.at<float>(i+k,j+l))
							aux1=img.at<float>(i+k,j+l);
						if(aux2>img.at<float>(i+k,j+l))
							aux2=img.at<float>(i+k,j+l);
					}
				}
				img2.at<float>(i,j)=(aux1+aux2)/2;
			}
		}
	}
	img=img2;
	return img;
}

Mat OrdenMinimo(Mat img,int tam){ //UTIL PARA RUIDO TIPO SAL
	img.convertTo(img,CV_32F,1./255);	
	Mat img2=img.clone();
	int m=tam/2;
	for(int i=m;i<img.rows-m;i++) { 
		for(int j=m;j<img.cols-m;j++) { 
			for(int tt=0;tt<3;tt++){
				float aux=1;
				for(int k=-m;k<=m;k++) { 
					for(int l=-m;l<=m;l++) { 
						if(aux>img.at<float>(i+k,j+l))
							aux=img.at<float>(i+k,j+l);
					}
				}
				img2.at<float>(i,j)=aux;
			}
		}
	}
	img=img2;
	return img;
}

Mat OrdenMaximo(Mat img,int tam){ //UTIL PARA RUIDO TIPO PIMIENTA
	img.convertTo(img,CV_32F,1./255);
	Mat img2=img.clone();
	int m=tam/2;
	for(int i=m;i<img.rows-m;i++) { 
		for(int j=m;j<img.cols-m;j++) { 
			for(int tt=0;tt<3;tt++){
				float aux=0;
				for(int k=-m;k<=m;k++) { 
					for(int l=-m;l<=m;l++) { 
						if(aux<img.at<float>(i+k,j+l))
							aux=img.at<float>(i+k,j+l);
					}
				}
				img2.at<float>(i,j)=aux;
			}
		}
	}
	img=img2;
	return img;
}
Mat OrdenAlfaRecortado(Mat img,int tam,int d){ //UTIL PARA COMBINACIONES DE GAUSSIANO Y SAL & PIMIENTA.
	img.convertTo(img,CV_32F,1./255);
	Mat img2=img.clone();
	int m=tam/2;
	for(int i=m;i<img.rows-m;i++) { 
		for(int j=m;j<img.cols-m;j++) { 
			for(int tt=0;tt<3;tt++){
				vector<float> aux;
				for(int k=-m;k<=m;k++) { 
					for(int l=-m;l<=m;l++) { 
						aux.push_back(img.at<float>(i+k,j+l));
					}
				}
				sort(aux.begin(),aux.end());
				aux.resize(aux.size()-d/2);
				float aux2=0;
				for(int k=d/2;k<aux.size();k++){ 
					aux2+=aux[k];
				}
				img2.at<float>(i,j)=aux2/(tam*tam-d);
			}
		}
	}
	img=img2;
	return img;
}

void Ejercicio3(){
	Mat img=imread("huang1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Original",img);
	
	Mat ruido=img.clone();
	ruido=ruido_sal_pimienta(ruido,10.0/255,10.0/255);
	ruido=ruido_gaussiano(ruido,0,0.1*255);
	imshow("Ruido",ruido);
	
	Mat filtro_mediana=ruido.clone();
	filtro_mediana=OrdenMediana(filtro_mediana,3);
	imshow("Filtro Mediana",filtro_mediana);
	
	Mat filtro_ptomedio=ruido.clone();
	filtro_ptomedio=OrdenPuntoMedio(filtro_ptomedio,3);
	imshow("Filtro Punto Medio",filtro_ptomedio);
	
	Mat filtro_ordenmin=ruido.clone();
	filtro_ordenmin=OrdenMinimo(filtro_ordenmin,3); //para ruido sal
	imshow("Filtro Orden Minimo",filtro_ordenmin);
	
	Mat filtro_ordenmax=ruido.clone();
	filtro_ordenmax=OrdenMaximo(filtro_ordenmax,3); //para ruido pimienta
	imshow("Filtro Orden Maximo",filtro_ordenmax);
	
	Mat filtro_alfarecortado=ruido.clone();
	filtro_alfarecortado=OrdenAlfaRecortado(filtro_alfarecortado,3,5); //para ruido pimienta
	imshow("Filtro Orden Alfa Recortado",filtro_alfarecortado);
	
	waitKey(0);
}

Mat filtro_notch_ideal(int rows,int cols,int _x,int _y,double corte){
	Mat magnitud = Mat::zeros(rows, cols, CV_32F);
	circle(
		   magnitud,
		   Point(cols/2 + _y,rows/2 + _x), //punto central
		   rows*corte, //radio
		   cv::Scalar::all(1),
		   -1 //círculo relleno
		   );
	circle(
		   magnitud,
		   cv::Point(cols/2 - _y,rows/2 - _x), //punto central
		   rows*corte, //radio
		   cv::Scalar::all(1),
		   -1 //círculo relleno
		   );
	magnitud = 1 - magnitud;
	return magnitud;
}

Mat filtro_notch_butterworth(int rows,int cols,int _x,int _y,double corte,int order){
	Mat	magnitud = Mat::zeros(rows,cols,CV_32F);
	corte *= rows;
	for(size_t K=0; K<rows; ++K){
		for(size_t L=0; L<cols; ++L){
			double d2 = distance2(K+.5, L+.5, rows/2. + _x, cols/2. + _y);
			magnitud.at<float>(K,L) = 1.0/(1 + std::pow(((corte*corte)*(corte*corte))/(d2*d2), order) );
			d2 = distance2(K+.5, L+.5, rows/2. - _x, cols/2. - _y);
			magnitud.at<float>(K,L) += 1.0/(1 + std::pow(((corte*corte)*(corte*corte))/(d2*d2), order) ) - 1;
		}
	}
	return magnitud;
}

Mat filtro_notch_gaussiano(int rows,int cols,int _x,int _y,double corte){
	Mat	magnitud = Mat::zeros(rows, cols, CV_32F);
	corte *= rows;
	for(size_t K=0; K<rows; ++K)
		for(size_t L=0; L<cols; ++L){
		double distance = distance2(K+.5, L+.5, rows/2. + _x, cols/2. + _y);
		magnitud.at<float>(K,L) = 1 - exp(-(distance*distance)/(2*(corte*corte)*(corte*corte)));
		distance = distance2(K+.5, L+.5, rows/2. - _x, cols/2. - _y);
		magnitud.at<float>(K,L) += 1 - exp(-(distance*distance)/(2*(corte*corte)*(corte*corte))) - 1;
	}
		return magnitud;
}

vector <int> Obtener_Frecuencias(Mat img){
	vector <int> frecuencias;
	Mat transformada=spectrum(img);
	//Esto lo hago para seber donde estan los puntos blancos en el espectro
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			if(transformada.at<float>(i,j) > 200.0/255.0){
				transformada.at<float>(i,j) = 0;
				cout<<i<<"-"<<j<<endl;
				frecuencias.push_back(i);
				frecuencias.push_back(j);
				
				//38-58
				//218-198
			}
		}
	}
	return frecuencias;
}

void Ejercicio4(){
	Mat img = imread("img_degradada.tif",CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img,CV_32F,1./255);
	imshow("Imagen Con Ruido",img);
	
	Mat img_sr = imread("img.tif",CV_LOAD_IMAGE_GRAYSCALE);
	img_sr.convertTo(img_sr,CV_32F,1./255);
	imshow("Imagen Sin Ruido",img_sr);
	
	Mat transformada(img.size(),img.type());
	transformada = spectrum(img);
	imshow("Espectro Sin Ruido",spectrum(img_sr));
	imshow("Espectro Con Ruido",transformada);
	//	imshow("Espectro Fourier",transformada);
	
//	Esto lo hago para seber donde estan los puntos blancos en el espectro
//		for(int i=0;i<img.rows;i++){
//			for(int j=0;j<img.cols;j++){
//				if(transformada.at<float>(i,j) > 200.0/255.0){
//					transformada.at<float>(i,j) = 0;
//					cout<<i<<"-"<<j<<endl;
//					//38-58
//					//218-198
//				}
//			}
//		}
//		imshow("Imagen Original 2",transformada);

	//Imagen Filtrada con un filtro notch ideal
	Mat notch = filtro_notch_ideal(img.rows,img.cols,38,58,0.08);
	
	//	imshow("Filtro Notch Ideal",notch);
	Mat filtrada_ideal = filter(img,notch);
	Mat ruido=1-notch;
	Mat filtradaruido=filter(img,ruido);
	imshow("Ruido ideal",filtradaruido);
	imshow("Espectro ruido ideal",spectrum(filtradaruido));
	//	filtrada_ideal = filtrada_ideal(Range(0,img.rows),Range(0,img.cols));
	imshow("Notch Ideal",filtrada_ideal);
	imshow("Espectro Ideal",spectrum(filtrada_ideal));
	
	//Imagen Filtrada con un filtro notch butterworth
	Mat butter = filtro_notch_butterworth(img.rows,img.cols,38,58,0.1,5);
	Mat ruidob=1-butter;
	Mat filtradaruidob=filter(img,ruidob);
	imshow("Ruido butt",filtradaruidob);
	imshow("Espectro ruido butt",spectrum(filtradaruido));
	//	imshow("Filtro Notch Butterworth",butter);
	Mat filtrada_butt = filter(img,butter);
	//	filtrada_butt = filtrada_butt(Range(0,img.rows),Range(0,img.cols));
	imshow("Notch Butt",filtrada_butt);
	imshow("Espectro Butt",spectrum(filtrada_butt));
	
	//Imagen Filtrada con un filtro notch gaussiano
	Mat gaussiano = filtro_notch_gaussiano(img.rows,img.cols,38,58,0.1);
	Mat ruidog=1-gaussiano;
	Mat filtradaruidog=filter(img,ruidog);
	imshow("Ruido Gauss",filtradaruidog);
//	imshow("Espectro ruido Gauss",spectrum(filtradaruidog));
	//	imshow("Filtro Notch Butterworth",gaussiano);
	Mat filtrada_gauss = filter(img,gaussiano);
	//	filtrada_gauss = filtrada_gauss(Range(0,img.rows),Range(0,img.cols));
	imshow("Notch Gauss",filtrada_gauss);
	imshow("Espectro Gauss",spectrum(filtrada_gauss));
	
	
	
	cout<<"El ECM entre la imagen original y el ruido es                      : "<<ECM(img_sr,img)<<endl;
	cout<<"El ECM entre la imagen original y filtrada por Notch Ideal es      : "<<ECM(img_sr,filtrada_ideal)<<endl;
	cout<<"El ECM entre la imagen original y filtrada por Notch Butterworth es: "<<ECM(img_sr,filtrada_butt)<<endl;
	cout<<"El ECM entre la imagen original y filtrada por Notch Gaussiano es  : "<<ECM(img_sr,filtrada_gauss)<<endl;
	waitKey(0);
};

void Ejercicio4_7(){
	Mat img1=imread("noisy_moon.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2=imread("HeadCT_degradada.tif",CV_LOAD_IMAGE_GRAYSCALE);
	img1.convertTo(img1,CV_32F,1./255);
	img2.convertTo(img2,CV_32F,1./255);
	imshow("Luna Original",img1);
	imshow("Head Original",img2);
	//	namedWindow("Espectro Luna",CV_WINDOW_KEEPRATIO);
	//	namedWindow("Espectro Head",CV_WINDOW_KEEPRATIO);
	imshow("Espectro Luna",spectrum(img1));
	imshow("Espectro Head",spectrum(img2));
	Mat f=filtro_notch_ideal(img1.rows,img1.cols,84,167,0.1);
	f=filter(img1,f);
	Mat f1=filtro_notch_ideal(f.rows,f.cols,167,84,0.1);
	f1=filter(f,f1);
	imshow("Filtro Luna",f1);
	namedWindow("Espectro Luna",CV_WINDOW_KEEPRATIO);
	imshow("Espectro Luna",spectrum(f1));
	
	
	Mat filtro=filtro_notch_ideal(img2.rows,img2.cols,216,216,0.01);
	filtro=filter(img2,filtro);
	Mat filtro2=filtro_notch_ideal(filtro.rows,filtro.cols,236,256,0.01);
	filtro2=filter(filtro,filtro2);
	Mat filtro3=filtro_notch_ideal(filtro2.rows,filtro2.cols,256,246,0.01);
	filtro3=filter(filtro2,filtro3);
	imshow("Filtro",filtro3);
	namedWindow("Espectro",CV_WINDOW_KEEPRATIO);
	imshow("Espectro",spectrum(filtro3));
	waitKey(0);
}
Mat filter_imaginario(cv::Mat image, cv::Mat filtro_magnitud){
	//se asume imágenes de 32F y un canal, con tamaño óptimo
	cv::Mat transformada;
	
	//como la fase es 0 la conversión de polar a cartesiano es directa (magnitud->x, fase->y)
	//	cv::Mat x[2];
	//	x[0] = filtro_magnitud.clone();
	//	x[1] = cv::Mat::zeros(filtro_magnitud.size(), CV_32F);
	//	
	//	cv::Mat filtro;
	//	cv::merge(x, 2, filtro);
	
	cv::dft(image, transformada, cv::DFT_COMPLEX_OUTPUT);
	cv::mulSpectrums(transformada, filtro_magnitud, transformada, cv::DFT_ROWS);
	
	cv::Mat result;
	cv::idft(transformada, result, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
	return result;
}

void Ejercicio5(){
	Mat img=imread("huang3_movida.tif",CV_LOAD_IMAGE_GRAYSCALE);
	img.convertTo(img,CV_32F,1./255);
	imshow("Original",img);
	imshow("Espectro",spectrum(img));
	
	Mat filtro=motion_blur(img.size(),-1,0); //Falta encontrar el valor correcto de a o b
	filtro=filter_imaginario(img,filtro);
	
	imshow("Filtrada",filtro);
	imshow("Espectro Filtrada",spectrum(filtro));
	waitKey(0);
}

int main(int argc, char** argv) {
//		Ejercicio1();
	
		Ejercicio2();
	
//		Ejercicio3();
	
//	Ejercicio4();
	
	//	Ejercicio4_7();
	
//	Ejercicio5();
	
	//	Averiguar();
	waitKey(0);
	return 0;
} 
