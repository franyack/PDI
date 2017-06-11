#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace pdi;
using namespace std;


///GUIA 2 - Operaciones Puntuales


//Hace mas claras las zonas oscuras
Mat transformacionLogaritmica(Mat img){
	Mat lut(1,256,CV_8U);
	float c = 255.0/log(256);
	float aux;
	for(int i=0;i<255;i++){
		aux = c*log(1+i);
		if(aux>255) aux = 255;
		if(aux<0) aux = 0;
		lut.at<unsigned char>(i)= aux;
	}
	Mat output;
	LUT(img,lut,output);
	return output;
}


//Oscurece las imagenes
Mat transformacionPotencia(Mat img, float gamma=1){
	Mat lut(1,256,CV_8U);
	float c = 255.0/log(256);
	float aux;
	for(int i=0;i<255;i++){
		
		aux = pow(i/255.0,gamma)*255.0;
		if(aux>255) aux = 255;
		if(aux<0) aux = 0;
		lut.at<unsigned char>(i)= aux;
	}
	Mat output;
	LUT(img,lut,output);
	return output;
}

//Aplica ruido gaussiano a la imagen
Mat ruidoGaussiano(Mat img, float mean, float sigma){
	Mat ruido = img.clone();
	RNG rng;
	rng.fill(ruido, RNG::NORMAL, mean,sigma); 
	add(img, ruido, img);
	return img;
}

//En la ubicacion 0 el bit menos significativo, en la 7 el mas significativo
std::vector <Mat> planoBits(Mat img){
	std::vector<Mat>BitPlane;
	for (int i=0; i < 8; i++) {
		Mat outPut;
		//Aplico mascara AND para acceder a cada bit plane (2^i)
		bitwise_and(img, pow(2,i), outPut);
		//Multiplico por 255 para hacerlo blanco y negro
		outPut *= 255;
		BitPlane.push_back(outPut);
	}
	return BitPlane;
}

///GUIA 3 - Manejo de Histograma y Filtrado Espacial

//devuelve el histograma de una imagen
Mat histograma(Mat img){
	Mat canvas(img.rows,img.cols,CV_32F);
	Mat histo = histogram(img,255);
	normalize(histo,histo,0,1,CV_MINMAX);
	draw_graph(canvas,histo);
	return canvas;
}

//Devuelve una imagen ecualizada
Mat ecualizarImagen(Mat img){
	Mat ecualizado;
	equalizeHist(img,ecualizado);
	return ecualizado;
}

//devuelve el histograma de una imagen ecualizada
Mat histogramaEcualizado(Mat img){
	Mat ecualizado;
	equalizeHist(img,ecualizado);
	Mat canvas(ecualizado.rows,ecualizado.cols,CV_32F);
	Mat histo = histogram(ecualizado,255);
	normalize(histo,histo,0,1,CV_MINMAX);
	draw_graph(canvas,histo);
	return canvas;
}

Mat aplicar_filtro_promediador(Mat img, int tam=3){
	Mat kernel(tam,tam,CV_32F);
	float aux=1/pow(tam,2);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=aux;
		}
	}
	Mat f = convolve(img,kernel);
	return f;
}

Mat aplicar_filtro_gaussiano(Mat img, int tam=3, float sigma = 1){
	Mat f;
	GaussianBlur(img,f,Size(tam,tam),sigma);
	return f;
}

Mat aplicar_filtro_suma1(Mat img, int tam=3){
	Mat kernel(tam,tam,CV_32F);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=-1;
		}
	}
	kernel.at<float>(tam/2,tam/2)=tam*tam;
	
	Mat result = convolve(img,kernel);
	return result;
}

Mat aplicar_filtro_suma0(Mat img, int tam = 3){
	Mat kernel(tam,tam,CV_32F);
	for (int i=0;i<kernel.rows;i++){
		for (int j=0;j<kernel.cols;j++){
			kernel.at<float>(i,j)=-1;
		}
	}
	kernel.at<float>(tam/2,tam/2)=tam*tam-1;
	
	Mat result = convolve(img,kernel);
	return result;
}

Mat aplicar_filtro_altaPotencia(Mat img, float A = 1, int tam=3){
	// Si A = 1 -> Mascara Difusa, sino, Alta Potencia
	Mat pb = aplicar_filtro_promediador(img,tam);
	
	Mat result = (A * img - pb + 255)/2;
	
	return result;
	
}

///GUIA 4 - Procesamiento de color

//Devuelve los planos RGB de una imagen. Recordar que el [0] corresponde al azul, [1] al verde y [2] al rojo
vector<Mat> planosRGB(Mat img){
	vector<Mat>bgr;
	split(img,bgr);
	return bgr;
}

//Devuelve los planos HSV.
vector<Mat> planosHSV(Mat img){
	cvtColor(img,img,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img,hsv);
	return hsv;   
}

//convierte un vector de HSV a una imagen RGB
Mat planosHSVtoimagenRGB(vector<Mat> hsv){
	Mat img_hsv,img_rgb;
	merge(hsv,img_hsv);
	cvtColor(img_hsv, img_rgb, CV_HSV2BGR);
	return img_rgb;
}

//Ecualiza los planos RGB de una imagen
Mat equalizarRGB(Mat img){
	vector<Mat>bgr;
	split(img,bgr);
	equalizeHist(bgr[0],bgr[0]);
	equalizeHist(bgr[1],bgr[1]);
	equalizeHist(bgr[2],bgr[2]);
	merge(bgr,img);
	return img;
}

//Ecualiza los planos HSV de una imagen
Mat equalizarHSV(Mat img){
	cvtColor(img, img,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img,hsv);
	//Solo se ecualiza el canal V
	equalizeHist(hsv[2],hsv[2]);
	merge(hsv,img);
	//Vuelvo a RGB
	cvtColor(img,img,CV_HSV2BGR);
	return img;
	
}
	
//Filtra cada plano RGB con un pasa alto suma 1, realzando la imagen
Mat realceMedianteAcentuadoRGB(Mat img, int n = 3){
	vector<Mat> bgr;
	split(img,bgr);
	bgr[0] = aplicar_filtro_suma1(bgr[0], n);
	bgr[1] = aplicar_filtro_suma1(bgr[1], n);
	bgr[2] = aplicar_filtro_suma1(bgr[2], n);
	merge(bgr,img);
	return img;	
}

//Filtra el plano de Intensidad con un pasa alto suma 1, realzando la imagen
Mat realceMedianteAcentuadoHSV(Mat img, int n =3 ){
	cvtColor(img,img,CV_BGR2HSV);
	vector<Mat>hsv;
	split(img,hsv);
	//El kernel se aplica solo sobre el plano de brillo o valor
	hsv[2]=aplicar_filtro_suma1(hsv[2], n);
	merge(hsv,img);
	cvtColor(img,img,CV_HSV2BGR);
	return img;
}

///GUIA 5 - Procesamiento y filtrado en el dominio frecuencial

//Devuelve la TDF de una imagen. La imagen debe ser cargada en escala de grises
Mat espectroFrecuencia(Mat img){
	img.convertTo(img,CV_32F,1./255);
	return spectrum(img);
}


//La imagen debe estar en escala de grises. Filtros pasa alto y pasa bajo en frecuencia. altobajo = 0 -> Pasa bajos, altobajo = 1 -> Pasa altos.
//tipo = 1 -> Ideal, tipo = 2 -> Butterworth, tipo = 3 -> Gaussiano. 
//D0 : indica cuan grande o chica es la circunsferencia de filtrado.
//orden es para el filtro de butteworth, cuanto mas grande es el valor, mas parecido al ideal
vector<Mat> aplicar_filtro_frecuencia(Mat img, int altobajo, int tipo, double D0=25/255.0,int orden=3){
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
	if (altobajo == 1){
		filtro=1-filtro; //Si es pasa alto se invierte todo, ya que ahora lo que no se filtra es el circulo
		//central que es donde estan las frecuencias bajas y se dejan pasar el resto que son las altas
	}
	Mat filtrada=filter(img,filtro);
	filtrada=filtrada(Range(0,filas),Range(0,columnas));
	
	//En la posicion 0 del vector, la imagen filtrada, en la posicion 1 el espectro en frecuencias
	vector<Mat> devolver;
	devolver.push_back(filtrada);
	devolver.push_back(spectrum(filtrada));
	return devolver;
}

//La imagen debe estar en escala de grises. AltaPotencia = (A - 1) + AltaPotencia
Mat aplicar_filtrado_altapotencia(Mat img,float A = 1){
	img.convertTo(img,CV_32F,1./255);
	int filas=img.rows;
	int columnas=img.cols;
	img=optimum_size(img);
	Mat high_boost=filter_butterworth(img.rows,img.cols,50/255.0,2);
	high_boost=1-high_boost;
	high_boost=(A-1)+high_boost;
	Mat filtrado_AltaPotencia=filter(img,high_boost);
	filtrado_AltaPotencia=filtrado_AltaPotencia(Range(0,filas),Range(0,columnas));
	return filtrado_AltaPotencia;
}

//La imagen debe estar en escala de grises. AltaFrecuencia = a + b*filtroEnfasis
Mat aplicar_filtrado_altafrecuencia(Mat img, float a, float b){
	img.convertTo(img,CV_32F,1./255);
	int filas=img.rows;
	int columnas=img.cols;
	img=optimum_size(img);
	Mat enfasis=filter_butterworth(img.rows,img.cols,50/255.0,2);
	enfasis=1-enfasis;
	enfasis=a+b*enfasis;
	Mat filtrado_AltaFrecuencia=filter(img,enfasis);
	filtrado_AltaFrecuencia=filtrado_AltaFrecuencia(Range(0,filas),Range(0,columnas));
	return filtrado_AltaFrecuencia;
}


//La imagen debe estar en escala de grises. No tengo idea los parametros, a prueba y error dice en la guia
//Evaluar la necesidad de ecualizar luego de aplicar el filtro homomorfico.
Mat aplicar_filtro_homomorfico(Mat img,float yl, float yh, double D0){
	img.convertTo(img,CV_32F,1./255);
	img=img+0.00001;
	log(img,img);
	int filas=img.rows;
	int columnas=img.cols;
	img=optimum_size(img);
	Mat filtro;
	filtro=(yh-yl)*(1-filter_gaussian(img.rows,img.cols,D0))+yl; 
	Mat filtrada=filter(img,filtro);
	filtrada=filtrada(Range(0,filas),Range(0,columnas));
	exp(filtrada,filtrada);
	return filtrada;
}

///GUIA 6 - Restauracion de imagenes

//Agregar ruido sal y pimienta a una imagen, pa es cuanta pimienta y pb cuanta sal.
Mat ruido_sal_pimienta(Mat img, float pa = 30/255.0, float pb = 30/255.0 ){   
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


//La imagen debe estar en tono de grises. Bueno para ruido gaussiano, malo para impulsivo
Mat aplicar_filtro_mediageometrica(Mat img, int tam=3){
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

//BUENO PARA RUIDO SAL Y PIMIENTA. Q>0 ELIMINA PIMIENTA, Q<0 ELIMINA SAL, 
//Q=0 MEDIA ARITMETICA, Q=-1 MEDIA ARMONICA
Mat aplicar_filtro_mediacontraArmonica(Mat img,int tam = 3,float Q = 0){
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

//Calcualr el error cuadratico medio entre dos imagenes
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

//BUENO PARA RUIDOS IMPULSIVOS SIN DESENFOQUE
Mat aplicar_filtro_ordenmediana(Mat img,int tam=3){
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

//UTIL PARA RUIDO GAUSSIANO O UNIFORME
Mat aplicar_filtro_ordenpuntomedio(Mat img,int tam=3){
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


//UTIL PARA COMBINACIONES DE GAUSSIANO Y SAL & PIMIENTA.
Mat aplicar_filtro_ordenalfarecortado(Mat img,int tam=3,int d=3){ 
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


//Funcion que necesita la funcion de abajo: el filtro pseudoinverso
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

//Filtro pseudoinverso - Sirve para restautar desenfoques por movimiento
//Falta encontrar el valor correcto de a o b
Mat aplicar_filtro_pseudoinverso(Mat img, float a = -1 , float b = 0){	
	img.convertTo(img,CV_32F,1./255);
	Mat filtro=motion_blur(img.size(),a,b); //Falta encontrar el valor correcto de a o b
	filtro=filter_imaginario(img,filtro);
	return filtro;
}

///GUIA 7 - Nociones de Segmentacion 


//Depende del tipo de detector que se elija, se modifican las matrices que conforman la mascara
Mat deteccionBordes(Mat img, int tipo=0){
	img.convertTo(img,CV_32F,1./255);
	Mat Gx,Gy;
	Mat magnitud;
	
	switch(tipo){
	case 0:		//Deteccion de bordes por Roberts
		Gx = (Mat_<double>(3,3) << 0, 0, 0, 0,-1, 0, 0, 0, 1); // Gx 0 0 0  Gy 0 0 0
		//    0-1 0     0 0-1
		Gy = (Mat_<double>(3,3) << 0, 0, 0, 0, 0,-1, 0, 1, 0); //    0 0 1     0 1 0
		break;
	case 1:		//Deteccion de bordes por Prewitt
		//Respuesta nula en zona de gris constante
		Gx = (Mat_<double>(3,3) << -1,-1,-1, 0, 0, 0, 1, 1, 1);
		Gy = (Mat_<double>(3,3) << -1, 0, 1,-1, 0, 1,-1, 0, 1);
		break;
	case 2:		//Deteccion de bordes por Sobel
		//Respuesta nula en zona de gris constante.
		//Enfatiza los pixeles mas cercanos al centro consiguiendo una mejor rta en presencia de ruido gaussiano.
		Gx = (Mat_<double>(3,3) << -1,-2,-1, 0, 0, 0, 1, 2, 1);
		Gy = (Mat_<double>(3,3) << -1, 0, 1,-2, 0, 2,-1, 0, 1);
		break;
	case 3:		//Deteccion de bordes por Laplaciano
		//Muy sensible al ruido, produce bordes dobles, no detecta direccion de bordes
		//Utilidad para clasificar ptos que pertenecen a zona clara y oscura a cada lado del borde.
		Gx = (Mat_<double>(3,3) << -1,-1,-1,-1, 8,-1,-1,-1,-1);
		break;
	case 4: 	//Deteccion de bordes por LoG (Laplaciano del Gaussiano).
		img.convertTo(img,CV_8U,255);
		Canny(img, magnitud, 50, 200, 5);
		break;
	}
	if(tipo!=4){
		if(tipo<3){
			//APLICO LAS MASCARAS A LA IMAGEN
			Mat x[2];
			x[0] = convolve(img,Gx);
			x[1] = convolve(img,Gy);
			//CALCULO LA MAGNITUD
			Mat derivada;
			merge(x,2,derivada);
			magnitud = magnitude(derivada);
		}else{
			magnitud = convolve(img,Gx);
		}
		//UMBRALIZO
		//entrada,salida,umbral,valor maximo q agarra,tipo de umbral.
		threshold(magnitud,magnitud,0.5,1,0);
	}
	
	return magnitud;
}


//Transformada de Hough de una imagen
vector<Mat> TransformadaHough(Mat img){
	//Paso 1 del algoritmo: Calcular el gradiente de la imagen y umbralizar el resultado
	Mat Gradiente;
	Canny(img, Gradiente, 50, 200, 5);
	
	//Paso 2 del algoritmo: Generar el espacio de parametros mediante valores equiespaciados de ro y sigma,
	//inicializandolo en cero al acumulador.
	//Para cada punto de borde de la imagen, calcular la curva correspondiente en el espacio de parametros
	//para cada sigma, despejando el valor de ro. Incrementar el acumulador en la posicion (ro,sigma).
	Mat acumulador = Mat::zeros(400,400,CV_8U);
	vector<Vec2i> coordenadas;
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			if(Gradiente.at<uchar>(i,j) == 255){ //Quiere decir que corresponde a un borde
				float angulo = atan(i/j*1.0);
				float ro = i*cos(angulo) + j*sin(angulo); //Curva en el espacio de parametro
				acumulador.at<uchar>(ro,angulo*180.0) += 1; //Incremento el acumulador en la posicion de ro-sigma
				Vec2i aux;
				aux[0] = i;
				aux[1] = j;
				coordenadas.push_back(aux);
			}
		}
	}
	
	//Paso 3: La interseccion de curvas en el espacio ro-sigma (altos valores en el acumulador) identifica 
	//la colinearidad de los puntos a los cuales corresponden tales curvas.
	//Conectar los puntos colineales cercanos (adyacentes o cuya distancia entre si sea menor a un umbral).
	Mat resultado = Mat::zeros(img.rows,img.cols,img.type());
	for(int i=0;i<coordenadas.size();i++){ //Recorro por coordenadas para conectar los puntos
		Vec2i aux1 = coordenadas[i];
		Vec2i aux2 = coordenadas[i+1];
		float angulo1 = atan2(aux1[1],aux1[0]);
		float angulo2 = atan2(aux2[1],aux2[0]);
		float ro1 = aux1[0]*cos(angulo1) + aux1[1]*sin(angulo1);
		float ro2 = aux2[0]*cos(angulo2) + aux2[1]*sin(angulo2);
		if( sqrt(((angulo2-angulo1)*(angulo2-angulo1))+((ro2-ro1)*(ro2-ro1))) < 3){
			line(resultado,Point(aux1[1],aux1[0]),Point(aux2[1],aux2[0]),Scalar(255));
		}
	}
	vector <Mat> Solucion;
	Solucion.push_back(Gradiente);
	Solucion.push_back(acumulador*255);
	Solucion.push_back(resultado);
	return Solucion;
}


////////////////////////////////////////////////
//Funcion super util para saber las coordenadas de determinado click, en cantclik debemos setear
//cuantas coordenadas se van a guardar en xy.
//Luego a la funcion debemos llamarla as�: 
//1� : namedWindow("My Window",WINDOW_NORMAL);
//2� : imshow("My Window",img);
//3�: setMouseCallback("My Window", Click, NULL);
vector<Point> xy;
int cantclick = 10;
bool reset = false;

void  Click(int event, int x, int y, int flags, void* userdata){
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


//Ejemplo de uso: Mat resultado = CrecimientoRegiones(img,xy[0].x,xy[0].y);
Mat CrecimientoRegiones(Mat img,int x,int y){ //Recibe la imagen a segmentar y un (x,y) donde se planta la semilla.
	//RELLENA ELEMENTOS CONECTADOS CON UN VALOR DADO
	//Parámetros de floodFill:
	//1º Imagen a segmentar.
	//2º Punto x,y donde se planta la semilla.
	//3º Nuevo valor del dominio de los pixeles pintados (valores que toman regiones cercanas a la semilla).
	//4º Opcional --> NULL
	//5º Minima diferencia de brillo / color entre el pixel observado y uno de sus vecinos (tolerancia inferior)
	//6º Maxima diferencia de brillo / color entre el pixel observado y uno de sus vecinos (tolerancia superior)
	
	floodFill(img,Point(x,y),1.01,NULL,0.05,0.05);	
	//UMBRALIZO PARA QUE PASEN UNICAMENTE LOS ELEMENTOS CON INTENSIDAD 1
	//La funcion threshold aplica un umbral de nivel fijo a cada elemento del arreglo
	//Parametros: Origen, Destino, Valor del umbral, Maximo valor a usar con THRESH_BINARY y el INV, y por ultimo el tipo de umbral.
	threshold(img,img,1,1,THRESH_BINARY);			
	return img;
}

//Obtener la media de una imagen
float obtenerMedia(Mat img){
	float media=0;
	for(int i=0;i<img.rows;i++) { 
		for(int j=0;j<img.cols;j++) { 
			media+=img.at<uchar>(i,j);
		}
	}
	media/=(img.rows*img.cols);
	return media;
}

//Obtener el Desvio de una imagen
float obtenerDesvio(Mat img){
	float Media = obtenerMedia(img);
	float desvio=0;
	for(int i=0;i<img.rows;i++) { 
		for(int j=0;j<img.cols;j++) { 
			desvio+=pow(img.at<uchar>(i,j)-Media,2);
		}
	}
	desvio/=(img.rows*img.cols);
	desvio=sqrt(desvio);
	return desvio;
}

void Hough(cv::Mat im,cv::Mat &cdst,cv::Mat &dst,double threshold,int &cx,int &cy,bool vertical=0,bool horizontal=0){
	//Inputs:
	// YOU CAN MOVE the Canny threshold also
	// Im: image
	// threshold: for HoughLines
	// vertical and horizontal: for detect vertical or horizontal lines
	//Outputs:
	// cdst: image and HoughLines
	// dst: contour of image
	//Notes: you can modify the degrees at line 30 and 31, also you can modify the large in 38 to 41
	
	Canny(im, dst, 50, 200, 3); //MOVE THIS PARAMETERS IF YOU NEED
	cvtColor(dst, cdst, CV_GRAY2BGR); 
	cx=0;cy=0;
	vector<Vec2f> lines;
	// detect lines
	HoughLines(dst, lines, 1, CV_PI/180, threshold, 0, 0 );
	int ymin=0;
	
	// draw lines
	for( size_t i = 0; i < lines.size(); i++ ){
		float rho = lines[i][0], theta = lines[i][1];
		// tener en cuenta sistema x-y normal y 0<theta<180
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		if(( theta>(CV_PI/180)*170 || theta<(CV_PI/180)*10) and vertical){ line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);cy=x0;} //vertical
		if(( theta>CV_PI/180*80 && theta<CV_PI/180*100) and horizontal){ 
			if(y0>ymin){
				line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
				ymin=y0;
			}
			cx=ymin;
		} //horizontal
		
	}
	
	return;
}

void MostrarHistogramas(Mat img){
	Mat img_bgr;
	img_bgr = img;
	imshow("Orignial BGR",img_bgr);
	Mat img_hsv;
	cvtColor(img,img_hsv,CV_BGR2HSV);
	imshow("Orignial HSV",img_hsv);
	
	vector<Mat> bgr; 	
	vector<Mat> hsv; 	
	split(img_bgr,bgr);
	split(img_hsv,hsv);
	
	//HISTOGRAMA DE LAS IMAGENES EN LOS 3 CANALES PARA BGR Y HSV
	Mat canvas_BGR(200,400,CV_32F);
	Mat canvas_HSV(200,400,CV_32F);
	Mat histo_BGR = histogram(img_bgr,256,Mat());
	Mat histo_HSV = histogram(img_hsv,256,Mat());
	normalize(histo_BGR,histo_BGR,0,1,CV_MINMAX);
	normalize(histo_HSV,histo_HSV,0,1,CV_MINMAX);
	draw_graph(canvas_BGR,histo_BGR);
	draw_graph(canvas_HSV,histo_HSV);
	imshow("Histograma BGR",canvas_BGR);
	imshow("Histograma HSV",canvas_HSV);
	
	//Histograma B - G - R
	Mat canvas_BGR_B(200,400,CV_32F);
	Mat canvas_BGR_G(200,400,CV_32F);
	Mat canvas_BGR_R(200,400,CV_32F);
	Mat histo_BGR_B = histogram(bgr[0],256,Mat());
	Mat histo_BGR_G = histogram(bgr[1],256,Mat());
	Mat histo_BGR_R = histogram(bgr[2],256,Mat());
	normalize(histo_BGR_B,histo_BGR_B,0,1,CV_MINMAX);
	normalize(histo_BGR_G,histo_BGR_G,0,1,CV_MINMAX);
	normalize(histo_BGR_R,histo_BGR_R,0,1,CV_MINMAX);
	draw_graph(canvas_BGR_B,histo_BGR_B);
	draw_graph(canvas_BGR_G,histo_BGR_G);
	draw_graph(canvas_BGR_R,histo_BGR_R);
	imshow("Histograma BGR_B",canvas_BGR_B);
	imshow("Histograma BGR_G",canvas_BGR_G);
	imshow("Histograma BGR_R",canvas_BGR_R);
	
	//Histograma H - S - V
	Mat canvas_HSV_H(200,400,CV_32F);
	Mat canvas_HSV_S(200,400,CV_32F);
	Mat canvas_HSV_V(200,400,CV_32F);
	Mat histo_HSV_H = histogram(hsv[0],256,Mat());
	Mat histo_HSV_S = histogram(hsv[1],256,Mat());
	Mat histo_HSV_V = histogram(hsv[2],256,Mat());
	normalize(histo_HSV_H,histo_HSV_H,0,1,CV_MINMAX);
	normalize(histo_HSV_S,histo_HSV_S,0,1,CV_MINMAX);
	normalize(histo_HSV_V,histo_HSV_V,0,1,CV_MINMAX);
	draw_graph(canvas_HSV_H,histo_HSV_H);
	draw_graph(canvas_HSV_S,histo_HSV_S);
	draw_graph(canvas_HSV_V,histo_HSV_V);
	imshow("Histograma HSV_H",canvas_BGR_B);
	imshow("Histograma HSV_S",canvas_HSV_S);
	imshow("Histograma HSV_V",canvas_HSV_V);
	
	waitKey(0);
}




//Mat img,gradiente,transformada;
//
//img=imread("Tenis/1.jpg");
//info(img);
//vector <vector <Point> > pt;
//HoughComun(img,gradiente,transformada,800,pt);

void HoughComun(Mat img, Mat &Gradiente, Mat &transformada, int tamaniolineas, vector <vector <Point> > &pt ){
	cvtColor(img,img,CV_BGR2GRAY);
	Canny(img,Gradiente,50,200,3); //Detecto los bordes de la imagen
	//			imshow("Original",img);
	//			imshow("Bordes",Gradiente);
	vector<Vec2f> lines;
	cvtColor(Gradiente, transformada, CV_GRAY2BGR);
	//HoughLines parámetros:
	//1º Salida del detector del borde.
	//2º Vector que almacenara los parametros ro, theta de las lineas detectadas.
	//3º La resolucion de ro en pixeles, usamos 1 pixel
	//4º La resolucion del parametro theta en radianes utilizamos un grado (CV_PI/180)
	//5º Umbral numero minimo de intersecciones para detectar una linea, a mayor umbral lineas mas largas.
	//6º y 7º parametros por defecto en 0
	//	HoughLines(Gradiente, lines, 1, CV_PI/180, 50, 0, 0 ); //Parametros para "letras1.tif"
	HoughLines(Gradiente, lines, 1, CV_PI/180, tamaniolineas, 0, 0 ); //Parametros para "snowman.png"
	//			cout<<lines.size()<<endl;
	
	for( size_t i = 0; i < lines.size(); i++ )
	{
		float ro = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		vector <Point> ptt;
		double a = cos(theta), b = sin(theta);
		double x0 = a*ro, y0 = b*ro;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		ptt.push_back(pt1);
		ptt.push_back(pt2);
		line( transformada, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
		pt.push_back(ptt);
	}
	//			for(int i=0;i<lines.size();i++) { 
	//				cout<<lines[i]<<endl;
	//			}
}

void RecorrerVectorVectoresPoint(vector <vector <Point> > pt){
	for(int i=0;i<pt.size();i++) { 
		vector <Point> aux;
		aux=pt[i];
		cout<<"LINEA "<<i;
		Point p1=aux[0];
		Point p2=aux[1];
		cout<<" PUNTO 1: X: "<<p1.x<<" Y: "<<p1.y<<" - PUNTO 2: X: "<<p2.x<<" Y: "<<p2.y<<endl;
	}
}


void HoughComunAngulos(Mat img, Mat &Gradiente, Mat &transformada,float angulo, int tamaniolineas, vector <vector <Point> > &pt ){
	//Le paso el angulo de las que quiero encontrar, recordar que:
	//Verticales: 0º
	//Horizontales: 90º
	
	cvtColor(img,img,CV_BGR2GRAY);
	Canny(img,Gradiente,50,200,3); //Detecto los bordes de la imagen
	vector<Vec2f> lines;
	cvtColor(Gradiente, transformada, CV_GRAY2BGR);
	HoughLines(Gradiente, lines, 1, (CV_PI/180), tamaniolineas, 0, 0 ); //Parametros para "snowman.png"
	
	for( size_t i = 0; i < lines.size(); i++ )
	{	
		cout<<(180*(lines[i][1])/M_PI)<<endl;
		if (angulo==(float)(180*(lines[i][1])/M_PI)){
			float ro = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			vector <Point> ptt;
			double a = cos(theta), b = sin(theta);
			double x0 = a*ro, y0 = b*ro;
			pt1.x = cvRound(x0 + 1000*(-b));
			pt1.y = cvRound(y0 + 1000*(a));
			pt2.x = cvRound(x0 - 1000*(-b));
			pt2.y = cvRound(y0 - 1000*(a));
			ptt.push_back(pt1);
			ptt.push_back(pt2);
			line( transformada, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
			pt.push_back(ptt);
		}
		
	}
}


void TransformadaHoughCircular(Mat src){ //Imagen en color, dps convierto en gris aca
	Mat src_gray;
	/// Convert it to gray
	cvtColor( src, src_gray, CV_BGR2GRAY );
	
	/// Reduce the noise so we avoid false circle detection
	GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );
	
	vector<Vec3f> circles;
	
	/// Apply the Hough Transform to find the circles
	//Parametros del HoughCircles:
	//1º imagen de entrada en grises
	//2º vector de salida de los circulos encontrados, con x,y del centro y r del radio.
	//3º CV_HOUGH_GRADIENT
	//4º 1
	//5º Distancia entre los centros de los circulos encontrados, si es muy pequeño varios circulos pueden ser falsamente detectados
	//si es muy grande algunos se pueden perder.
	//6º 
	//7º 
	//8º Minimo radio
	//9º Maximo radio
	
	HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
	
	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
		circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
	}
	
	/// Show your results
	namedWindow( "Hough Circle Transform Demo", CV_WINDOW_KEEPRATIO );
	imshow( "Hough Circle Transform Demo", src );
	
	waitKey(0);
}
