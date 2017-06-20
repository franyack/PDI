// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

This example program shows how to find frontal human faces in an image and
estimate their pose.  The pose takes the form of 68 landmarks.  These are
points on the face such as the corners of the mouth, along the eyebrows, on
the eyes, and so forth.



This face detector is made using the classic Histogram of Oriented
Gradients (HOG) feature combined with a linear classifier, an image pyramid,
and sliding window detection scheme.  The pose estimator was created by
using dlib's implementation of the paper:
One Millisecond Face Alignment with an Ensemble of Regression Trees by
Vahid Kazemi and Josephine Sullivan, CVPR 2014
and was trained on the iBUG 300-W face landmark dataset.

Also, note that you can train your own models using dlib's machine learning
tools.  See train_shape_predictor_ex.cpp to see an example.


//Averiguar por funciones que van o vienen de dlib a opencv

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


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <vector>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
  try
  {
    // This example takes in a shape model file and then a list of images to
    // process.  We will take these filenames in as command line arguments.
    // Dlib comes with example images in the examples/faces folder so give
    // those as arguments to this program.
    if (argc == 1)
    {
      cout << "Call this program like this:" << endl;
      cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
      cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
      cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
      return 0;
    }

    // We need a face detector.  We will use this to get bounding boxes for
    // each face in an image.
    frontal_face_detector detector = get_frontal_face_detector();
    // And we also need a shape_predictor.  This is the tool that will predict face
    // landmark positions given an image and face bounding box.  Here we are just
    // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
    // as a command line argument.
    shape_predictor sp;
    deserialize(argv[1]) >> sp;

    std::vector<float> gonzaneutro;
    //Neutro ojos
    gonzaneutro.push_back(95);
    gonzaneutro.push_back(81.25);
    gonzaneutro.push_back(80.06);
    gonzaneutro.push_back(94.04);
    //Neutro boca
    gonzaneutro.push_back(244.074);

    std::vector<float> facuneutro;
    //Neutro ojos
    facuneutro.push_back(93.53);
    facuneutro.push_back(84.59);
    facuneutro.push_back(82.22);
    facuneutro.push_back(87.05);
    //Neutro boca
    facuneutro.push_back(278.007);
    facuneutro.push_back(781);

    std::vector<float> franneutro;
    //Neutro ojos
    franneutro.push_back(94.76);
    franneutro.push_back(83.38);
    franneutro.push_back(89.10);
    franneutro.push_back(82.93);
    //Neutro boca
    franneutro.push_back(244.002);
    franneutro.push_back(915);

    image_window win, win_faces;
    // Loop over all the images provided on the command line.
    for (int i = 2; i < argc; ++i)
    {
      cout << "processing image " << argv[i] << endl;
      array2d<rgb_pixel> img;
      load_image(img, argv[i]);
      // Make the image larger so we can detect small faces.
      pyramid_up(img);

      // Now tell the face detector to give us a list of bounding boxes
      // around all the faces in the image.
      std::vector<rectangle> dets = detector(img);
      cout << "Number of faces detected: " << dets.size() << endl;

      // Now we will go ask the shape_predictor to tell us the pose of
      // each face we detected.
      std::vector<full_object_detection> shapes;
      for (unsigned long j = 0; j < dets.size(); ++j)
      {
        full_object_detection shape = sp(img, dets[j]);
        cout << "number of parts: "<< shape.num_parts() << endl;

        //Distancias para el macho
        double dist1 = sqrt(pow((shape.part(37)(1) - shape.part(19)(1)),2) + pow((shape.part(37)(0) - shape.part(19)(0)),2));
        double dist2 = sqrt(pow((shape.part(38)(1) - shape.part(20)(1)),2) + pow((shape.part(38)(0) - shape.part(20)(0)),2));
        double dist3 = sqrt(pow((shape.part(43)(1) - shape.part(23)(1)),2) + pow((shape.part(43)(0) - shape.part(23)(0)),2));
        double dist4 = sqrt(pow((shape.part(44)(1) - shape.part(24)(1)),2) + pow((shape.part(44)(0) - shape.part(24)(0)),2));

        cout<<"El 48: x: "<<shape.part(48)(0)<<" y: "<<shape.part(48)(1)<<endl;
        cout<<"El 54: x: "<<shape.part(54)(0)<<" y: "<<shape.part(54)(1)<<endl;

        //cout<<"El 33: x: "<<shape.part(33)(0)<<" y: "<<shape.part(33)(1)<<endl;
        //cout<<"El 51: x: "<<shape.part(51)(0)<<" y: "<<shape.part(51)(1)<<endl;

        //cout<<"medio "<<shape.part(66)(0)<<endl;
        //cout<<"Euclidea entre 45 y 54 "<<sqrt(pow((shape.part(54)(1) - shape.part(45)(1)),2) + pow((shape.part(54)(0) - shape.part(45)(0)),2))<<endl;
        //cout<<"Euclidea entre 36 y 48 "<<sqrt(pow((shape.part(36)(1) - shape.part(48)(1)),2) + pow((shape.part(36)(0) - shape.part(48)(0)),2))<<endl;

        //Distancia siete de oro
        double siete = sqrt(pow((shape.part(48)(1) - shape.part(54)(1)),2) + pow((shape.part(48)(0) - shape.part(54)(0)),2));
        double bocanasoderecho = sqrt(pow((shape.part(48)(1) - shape.part(31)(1)),2) + pow((shape.part(48)(0) - shape.part(31)(0)),2));
        double bocanasoizquierdo = sqrt(pow((shape.part(54)(1) - shape.part(35)(1)),2) + pow((shape.part(54)(0) - shape.part(35)(0)),2));


        //siete = siete *100 / gonzaneutro[4];
        //siete = siete *100 / facuneutro[4];
        siete = siete *100 / franneutro[4];
        cout<<"siete "<<siete<<endl;

        cout<<"dist1 "<<dist1<<endl<<"dist2 "<<dist2<<endl<<"dist3 "<<dist3<<endl<<"dist 4 "<<dist4<<endl;

        double ojo1 = (dist1*100/gonzaneutro[0] + dist2*100 /gonzaneutro[1])/2;
        double ojo2 = (dist3*100/ gonzaneutro[2] + dist4*100 / gonzaneutro[3])/2;
        cout<<"ojo 1 "<<ojo1<<endl<<"ojo 2 "<<ojo2<<endl;

        if(ojo1 >130 && ojo2 >130){
          cout<<"MACHO PAPÁ"<<endl;
        }else{
          cout<<"HACÉ EL MACHO CAGON"<<endl;
        }

        if(siete > 110 && bocanasoizquierdo < bocanasoderecho){
          cout<<"Siete de oro"<<endl;
        }else{
          if (siete > 110 && bocanasoizquierdo > bocanasoderecho){
            cout<<"Siete de espada"<<endl;
          }else{
            cout<<"No haces el siete"<<endl;
          }
        }








        //                cout << "pixel position of second part: " << shape.part(37) << endl;
        // You get the idea, you can get all the face part locations if
        // you want them.  Here we just store them in shapes so we can
        // put them on the screen.
        shapes.push_back(shape);
      }

      // Now let's view our face poses on the screen.
      win.clear_overlay();
      win.set_image(img);
      win.add_overlay(render_face_detections(shapes));

      // We can also extract copies of each face that are cropped, rotated upright,
      // and scaled to a standard size as shown here:
      dlib::array<array2d<rgb_pixel> > face_chips;
      extract_image_chips(img, get_face_chip_details(shapes), face_chips);
      win_faces.set_image(tile_images(face_chips));

      cout << "Hit enter to process the next image..." << endl;
      cin.get();
    }
  }
  catch (exception& e)
  {
    cout << "\nexception thrown!" << endl;
    cout << e.what() << endl;
  }
}

// ----------------------------------------------------------------------------------------
