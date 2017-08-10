// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
This example program shows how to find frontal human faces in an image and
estimate their pose.  The pose takes the form of 68 landmarks.  These are
points on the face such as the corners of the mouth, along the eyebrows, on
the eyes, and so forth.

This example is essentially just a version of the face_landmark_detection_ex.cpp
example modified to use OpenCV's VideoCapture object to read from a camera instead
of files.
Es una version modificada del face_landmark_detection_ex para utilizar el VideoCapture de OpenCV para leer la camara en lugar
de archivos (imagenes).
Finally, note that the face detector is fastest when compiled with at least
SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
chip then you should enable at least SSE2 instructions.  If you are using
cmake to compile this program you can enable them by using one of the
following commands when you create the build project:
cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
This will set the appropriate compiler options for GCC, clang, Visual
Studio, or the Intel compiler.  If you are using another compiler then you
need to consult your compiler's manual to determine how to enable these
instructions.  Note that AVX is the fastest but requires a CPU from at least
2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

int main()
{
  try
  {
    cv::VideoCapture cap(0); //funcion de OpenCv para abrir la camara de la compu. El contenido esta en cap, por defecto va 0.
    if (!cap.isOpened()) //retorna true si se inicio la captura de video, sino tira error que la camara esta desconectada.
    {
      cerr << "Unable to connect to camera" << endl;
      return 1;
    }

    image_window win; //creo ventana para mostrar imagen.

    // Load face detection and pose estimation models.
    // Cargo la deteccion de rostros y poses al igual que en face_landmark_detection.
    //La funcion get_frontal_face_detector devuelve un object_detector (del tipo frontal_face_detector), o sea los rostros frontales.
    frontal_face_detector detector = get_frontal_face_detector(); //detecta las caras frontales en el video.

    //shape_predictor toma una region de img que contiene algun objeto y emite un conjunto de puntos. El ejemplo es el de tomar
    //una cara humana y se identifican los landmarks en la boca, ojos, nariz, etc.
    shape_predictor pose_model; //antes era shape_predictor sp, esto contiene los puntos de interes de la cara.

    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    // Grab and process frames until the main window is closed by the user.
    // Grabo y proceso los frames hasta que se cierre la ventana que muestra la filmacion.
    // A diferencia del anterior que iba recorriendo las imagenes y esperaba un enter para pasar a la siguiente o terminar.
    //int contador=0;
    int prueba=0;
    full_object_detection auxiliar;
    while(!win.is_closed()) //entonces pregunto que mientras este abierta...
    {
      //if (contador==10){
      //	contador=0;
      //}
      //Este contador lo hago para obtener la cara neutra, es decir si pasaron 20 frames que se actualice
      if (prueba==20){
        prueba=0;
      }
      // Grab a frame
      //Grabo el frame en una imagen temporal tipo Mat de OpenCV.
      cv::Mat temp; //Creo la imagen temporal "temp"
      cap >> temp;  //A la imagen creada le copio el frame del video que esta grabando "cap".

      // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
      // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
      // long as temp is valid.  Also don't do anything to temp that would cause it
      // to reallocate the memory which stores the image as that will make cimg
      // contain dangling pointers.  This basically means you shouldn't modify temp
      // while using cimg.

      //Como estoy trabajando en una variable Mat que es de OpenCV debo convertirla a algo en lo que pueda trabajar el dlib.
      //Tenga en cuenta que esto solo envuelve el objeto Mat sin copiar nada. Asi que "cimg" solo es valido siempre y cuando
      //"temp" sea valido. Tambien no se debe realizar nada en "temp" que haga que se reasigne la memoria que almacena
      //la imagen, ya que hara que "cimg" quede con los puntos colgando. Resumidamente significa que no hay que modificar "temp"
      //mientras se este usando el "cimg".

      cv_image<bgr_pixel> cimg(temp); //convierte img de opencv a dlib.

      // Detect faces
      //Detectar las caras.
      //Al igual que antes creo un rectangulo (bounding box) que me detecta la cara del FRAME del video (cimg).
      std::vector<rectangle> faces = detector(cimg); //antes se llamaba dets (caras detectadas)

      // Find the pose of each face.
      //Encuentro la pose de cada cara, para obtener los puntos de interes.
      std::vector<full_object_detection> shapes;
      //el tipo full_object_detection es un objeto que representa la ubicacion
      //de un objeto en una imagen junto con las posiciones de cada una de sus partes constitutivas.

      for (unsigned long i = 0; i < faces.size(); ++i){ //Recorro por cada cara detectada "faces" lo que antes era "dets"
        //Ahora directamente voy almacenando los puntos de interes en shapes, no accedo a cada uno de ellos
        // para mostrarlos
        full_object_detection shape = pose_model(cimg, faces[i]);
        //Pregunto si prueba es 0 que actualice la cara neutra que se va a alojar en neutro.
        if (prueba==0){
          auxiliar=pose_model(cimg,faces[i]);
        }
        //if (contador==0){
        //En Macho (1 y 2) guardo la seña del frame actual, utilizando los dos puntos de cada ojo, entre el 20 y 38 & 25 y 45 .
        //En NeutroMacho (1 y 2) calculo la distancia euclidea en la cara almacenada en neutro.
        double Macho1=sqrt(pow((shape.part(37)(1) - shape.part(19)(1)),2) + pow((shape.part(37)(0) - shape.part(19)(0)),2));
        double Macho2=sqrt(pow((shape.part(44)(1) - shape.part(24)(1)),2) + pow((shape.part(44)(0) - shape.part(24)(0)),2));
        double NeutroMacho1=sqrt(pow((auxiliar.part(37)(1) - auxiliar.part(19)(1)),2) + pow((auxiliar.part(37)(0) - auxiliar.part(19)(0)),2));
        double NeutroMacho2=sqrt(pow((auxiliar.part(44)(1) - auxiliar.part(24)(1)),2) + pow((auxiliar.part(44)(0) - auxiliar.part(24)(0)),2));

        double siete = sqrt(pow((shape.part(48)(1) - shape.part(54)(1)),2) + pow((shape.part(48)(0) - shape.part(54)(0)),2));
        double sieteneutro = sqrt(pow((auxiliar.part(48)(1) - auxiliar.part(54)(1)),2) + pow((auxiliar.part(48)(0) - auxiliar.part(54)(0)),2));

        double nada1 = sqrt(pow((shape.part(37)(1) - shape.part(41)(1)),2) + pow((shape.part(37)(0) - shape.part(41)(0)),2));
        double neutronada1 = sqrt(pow((auxiliar.part(37)(1) - auxiliar.part(41)(1)),2) + pow((auxiliar.part(37)(0) - auxiliar.part(41)(0)),2));
        double nada2 = sqrt(pow((shape.part(44)(1) - shape.part(46)(1)),2) + pow((shape.part(44)(0) - shape.part(46)(0)),2));
        double neutronada2 = sqrt(pow((auxiliar.part(44)(1) - auxiliar.part(46)(1)),2) + pow((auxiliar.part(44)(0) - auxiliar.part(46)(0)),2));

        double ancho1 = sqrt(pow((shape.part(51)(1) - shape.part(57)(1)),2) + pow((shape.part(51)(0) - shape.part(57)(0)),2));
        double neutroancho1 = sqrt(pow((auxiliar.part(51)(1) - auxiliar.part(57)(1)),2) + pow((auxiliar.part(51)(0) - auxiliar.part(57)(0)),2));
        double ancho2 = sqrt(pow((shape.part(65)(1) - shape.part(63)(1)),2) + pow((shape.part(65)(0) - shape.part(63)(0)),2));
        double neutroancho2 = sqrt(pow((auxiliar.part(63)(1) - auxiliar.part(65)(1)),2) + pow((auxiliar.part(63)(0) - auxiliar.part(65)(0)),2));

        double dos1 = sqrt(pow((shape.part(50)(1) - shape.part(62)(1)),2) + pow((shape.part(50)(0) - shape.part(62)(0)),2));
        double neutrodos1 = sqrt(pow((auxiliar.part(50)(1) - auxiliar.part(62)(1)),2) + pow((auxiliar.part(50)(0) - auxiliar.part(62)(0)),2));
        double dos2 = sqrt(pow((shape.part(51)(1) - shape.part(33)(1)),2) + pow((shape.part(51)(0) - shape.part(33)(0)),2));
        double neutrodos2 = sqrt(pow((auxiliar.part(51)(1) - auxiliar.part(33)(1)),2) + pow((auxiliar.part(51)(0) - auxiliar.part(33)(0)),2));

        //cout<<"dos 1 "<<dos1 * 100 / neutrodos1<<endl<<endl;
        //cout<<"dos 2 "<<dos2 * 100 / neutrodos2<<endl<<endl;

        //cout<<"resta oro "<<shape.part(48)(1) - shape.part(54)(1)<<endl<<endl;
        //cout<<"resta espada"<<shape.part(54)(1) - shape.part(48)(1)<<endl<<endl;

        if(nada1 * 100 / neutronada1 < 50 && nada2 * 100 / neutronada2 < 50){
          cout<<"NO TIENE NADA!!!"<<endl;
        }else{
          if (((Macho1*100/NeutroMacho1)>130) && ((Macho2*100/NeutroMacho2)>130)){
            cout<< "TIENE EL MACHO "<< endl;
          }else {
            if(shape.part(48)(1) - shape.part(54)(1)>15){
              cout<<"TIENE EL SIETE DE ORO"<<endl;
            }else{
              if(shape.part(54)(1) - shape.part(48)(1)>8){
                cout<<"TIENE EL SIETE DE ESPADA"<<endl;
              }else{
                if(dos1 * 100 / neutrodos1 > 115 && dos2 * 100 / neutrodos2 < 85){
                  cout<<"TIENE EL DOS"<<endl;
                }else{
                  if(ancho1 * 100 / neutroancho1 > 200){
                      cout<<"TIENE EL ANCHO FALSO "<<endl;
                  }else{
                      cout<<"NEUTRO"<<endl;
                  }
                }
              }
            }
          }
        }

    shapes.push_back(shape);


    }
    //contador++;
    prueba++;
    // Display it all on the screen
    //Muestra todo en pantalla.
    win.clear_overlay();
    win.set_image(cimg);
    win.add_overlay(render_face_detections(shapes));
  }
}
catch(serialization_error& e)
{
  cout << "You need dlib's default face landmarking model file to run this example." << endl;
  cout << "You can get it from the following URL: " << endl;
  cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
  cout << endl << e.what() << endl;
}
catch(exception& e)
{
  cout << e.what() << endl;
}
}
