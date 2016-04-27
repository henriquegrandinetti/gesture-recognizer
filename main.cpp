/*
 * Author: Henrique Grandinetti Barbosa Amaral
 * Email: henriquegrandinetti@gmail.com
 * Computer Science, University of Bristol
 * Image Processing and Computer Vision
 *
 * Gesture Recognizer	
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>


using namespace std;
using namespace cv;

void Square(Mat &image, float size, float angle, Point center, Scalar color);
double** allocateMat (int sizeX, int sizeY);
Size AllocateDerivatives(double ***Ix, double ***Iy, double ***It, Size subimage, Size region, int shift);
double IxRegion (Mat &first, Mat &next, Size regionSize, Point start);
double IyRegion (Mat &first, Mat &next, Size regionSize, Point start);
double ItRegion (Mat &first, Mat &next, Size regionSize, Point start);
void Ixyt (Mat &first, Mat &next, double **IxMat, double **IyMat, double **ItMat, Size region, int shift, Size subimage, Point position);
void FillA (double **Ix, double **Iy, double A[2][2], Size derivativeSize);
void Fillb (double **Ix, double **Iy, double **It, double b[2], Size derivativeSize);
int Inverse (double A[2][2], double Ai[2][2]);
void MultiplyAb (double Ai[2][2], double b[2], double v[2]);
string NextSquare (Vec4f vaverage, int t, float &square_diagonal, float scale_factor, Point &square_center, int speed, float angular_speed, float &angle, Mat &display, bool simple);

int main( int argc, const char** argv ){

    cv::VideoCapture cap;

    if(argc > 1 && strcmp(argv[argc - 1], "simple") != 0){
        cout << "maoi" << endl;
        cap.open(string(argv[1]));}
    else
        cap.open(CV_CAP_ANY);

    if(!cap.isOpened())
        printf("Error: could not load a camera or video.\n");

    int resize_factor = 3;

    ////////////////////////////////////////
    // set operational mode
    // simple -> just left and right
    // !simple -> all gestures + mini-game
    /////////////////////////////////////////
    bool simple;

    if (strcmp(argv[argc - 1], "simple")  == 0)
        simple = true;
    else
        simple = false;

    /////////////////////////////////
    // Images for processing
    ////////////////////////////////

    Mat firstbig, firstgray, first;
    Mat nextbig, nextgray, next;
    Mat display;

    cap >> firstbig;
    cap >> nextbig;

    resize(firstbig, first, Size(firstbig.cols/resize_factor, firstbig.rows/resize_factor));
    resize(nextbig, next, Size(nextbig.cols/resize_factor, nextbig.rows/resize_factor));

    flip(first, first, 1);
    flip(next, next, 1);

    if (firstbig.empty() || nextbig.empty())
        return 0;

    /////////////////////////////////////////
    // Variables for Lucas Kanade Algorithm
    /////////////////////////////////////////

    double A[2][2], b[2], Ai[2][2], v[2];
    double **IxMat, **IyMat, **ItMat;

    int subimages_in_x = 25;
    int subimages_in_y = 18;
    int region_width = 12;
    int region_height = 8;
    int shift = 4;

    Size subimage((int)(first.cols/subimages_in_x),(int)(first.rows/subimages_in_y));
    Size region(region_width,region_height);
    Size derivativesSize = AllocateDerivatives(&IxMat, &IyMat, &ItMat, subimage, region, shift);

    Mat vectors(subimages_in_y, subimages_in_x, CV_32FC2, Scalar(1,1));

    int x, y;



    //////////////////////////////////////
    // Variable for Average Motion
    //////////////////////////////////////

    vector<Vec4f> motion_buffer = *new vector<Vec4f>;
    Vec4f vaverage;
    Vec4f LRmotion;
    Vec4f deleted;
    int t = 7;
    int frames = 15;

    vaverage[0] = 0; vaverage[1] = 0; vaverage[2] = 0; vaverage[3] = 0;
    LRmotion[0] = 0; LRmotion[1] = 0; LRmotion[2] = 0; LRmotion[3] = 0;


    ///////////////////////////////////
    //Variables for Mini-Game
    ///////////////////////////////////

    Point square_center = Point(first.cols/2, first.rows/2);
    float square_size = 30;
    float square_diagonal = square_size * sqrt(2);
    float scale_factor = 1.08;
    float angle = CV_PI/4;
    float angular_speed = CV_PI/50;
    int speed = 3;

    string label;

    namedWindow("video", 1);

    while(1){

        cvWaitKey(10);

        display = first.clone();

        cvtColor(first, firstgray, CV_BGR2GRAY);
        cvtColor(next, nextgray, CV_BGR2GRAY);

        LRmotion[0] = 0;
        LRmotion[1] = 0;
        LRmotion[2] = 0;
        LRmotion[3] = 0;

        int k = 0, m = 0;

        //loop through all subimages
        for (int j = 0; j <= first.rows - subimage.height; j += subimage.height) {
            for (int i = 0; i <= first.cols - subimage.width; i += subimage.width) {


                Ixyt(firstgray, nextgray, IxMat, IyMat, ItMat, region, shift, subimage, Point(i,j));

                FillA(IxMat, IyMat, A, derivativesSize);
                Fillb(IxMat, IyMat, ItMat, b, derivativesSize);

                if (Inverse(A, Ai) == 0){
                    vectors.at<Vec2f>(m, k)[0] = (vectors.at<Vec2f>(m, k)[0] + 0) /2;
                    vectors.at<Vec2f>(m, k)[1] = (vectors.at<Vec2f>(m, k)[1] + 0) /2;
                }

                //have solution then we scale the vector
                else{
                    MultiplyAb(Ai, b, v);
                    if (sqrt(pow(v[0],2) + pow(v[1],2)) <= subimage.height/2) {
                        if (sqrt(pow(v[0],2) + pow(v[1],2)) < 5) {
                            vectors.at<Vec2f>(m, k)[0] = (vectors.at<Vec2f>(m, k)[0] + 0) /2;
                            vectors.at<Vec2f>(m, k)[1] = (vectors.at<Vec2f>(m, k)[1] + 0) /2;
                        }
                        else{
                            vectors.at<Vec2f>(m, k)[0] = (vectors.at<Vec2f>(m, k)[0] + v[0]) /2;
                            vectors.at<Vec2f>(m, k)[1] = (vectors.at<Vec2f>(m, k)[1] + v[1]) /2;
                        }
                    }
                    else {
                        vectors.at<Vec2f>(m, k)[0] = v[0]*subimage.height*0.5/sqrt(pow(v[0],2) + pow(v[1],2));
                        vectors.at<Vec2f>(m, k)[1] = v[1]*subimage.height*0.5/sqrt(pow(v[0],2) + pow(v[1],2));
                    }
                }

                x = i + subimage.width/2;
                y = j + subimage.height/2;

                line(display, Point(x, y), Point(x + vectors.at<Vec2f>(m, k)[0], y + vectors.at<Vec2f>(m, k)[1]), Scalar(0,255,0), 1);

                if (k < subimages_in_x/2) {
                    LRmotion[0] += vectors.at<Vec2f>(m, k)[0];
                    LRmotion[1] += vectors.at<Vec2f>(m, k)[1];
                }
                else{
                    LRmotion[2] += vectors.at<Vec2f>(m, k)[0];
                    LRmotion[3] += vectors.at<Vec2f>(m, k)[1];
                }

                k++;
            }
            k = 0;
            m++;
        }
        m = 0;

        if (motion_buffer.size() < frames){
            motion_buffer.push_back(LRmotion);
            vaverage[0] += LRmotion[0]/frames;
            vaverage[1] += LRmotion[1]/frames;
            vaverage[2] += LRmotion[2]/frames;
            vaverage[3] += LRmotion[3]/frames;
        }
        else{
            deleted = motion_buffer.at(0);
            vaverage[0] -= deleted[0]/frames;
            vaverage[1] -= deleted[1]/frames;
            vaverage[2] -= deleted[2]/frames;
            vaverage[3] -= deleted[3]/frames;

            vaverage[0] += LRmotion[0]/frames;
            vaverage[1] += LRmotion[1]/frames;
            vaverage[2] += LRmotion[2]/frames;
            vaverage[3] += LRmotion[3]/frames;
            motion_buffer.erase(motion_buffer.begin());
            motion_buffer.push_back(LRmotion);
        }

        label = NextSquare(vaverage, t, square_diagonal, scale_factor, square_center, speed, angular_speed, angle, display, simple);

        if (!simple)
            Square(display, square_diagonal, angle, Point(square_center.x, square_center.y), Scalar(255, 0, 0));

        putText(display, label, Point(30, display.rows - 20), CV_FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(0,0,255), 1.0);

        imshow("video", display);


        first = next.clone();
        cap >> nextbig;

        if(nextbig.empty())
            break;

        resize(nextbig, next, Size(nextbig.cols/resize_factor, nextbig.rows/resize_factor));
        flip(next, next, 1);
    }
}



double** allocateMat (int sizeX, int sizeY){
    double **Mat = new double*[sizeX];
    for(int i = 0; i < sizeX; ++i)
        Mat[i] = new double[sizeY];
    return Mat;
}

Size AllocateDerivatives(double ***Ix, double ***Iy, double ***It, Size subimage, Size region, int shift){
    int width  = (int)(subimage.width - region.width)/shift + 1;
    int height = (int)(subimage.height - region.height)/shift + 1;

    *It = allocateMat(width, height);
    *Ix = allocateMat(width, height);
    *Iy = allocateMat(width, height);

    return Size(width,height);
}

void Ixyt (Mat &first, Mat &next, double **IxMat, double **IyMat, double **ItMat, Size region, int shift, Size subimage, Point position){
    int x = 0, y = 0;

    for (int j = position.y; j <= position.y + subimage.height - region.height; j += shift){
        for (int i = position.x; i <= position.x + subimage.width - region.width; i += shift) {
            IxMat[x][y] = IxRegion(first, next, region, Point(i,j));
            IyMat[x][y] = IyRegion(first, next, region, Point(i,j));
            ItMat[x][y] = ItRegion(first, next, region, Point(i,j));
            //cout << x << " " << y <<endl;
            x++;
        }
        x = 0;
        y++;
    }
}

double IxRegion (Mat &first, Mat &next, Size regionSize, Point start){
    double Ix = 0;

    for (int j = 0; j < regionSize.height; j++) {
        for (int i = 0; i < regionSize.width - 1; i++) {
            Ix += first.at<uchar>(Point(start.x + i + 1,start.y + j)) - first.at<uchar>(Point(start.x + i,start.y + j));
            Ix += next.at<uchar>(Point(start.x + i + 1,start.y + j)) - next.at<uchar>(Point(start.x + i,start.y + j));
        }
    }

    return Ix/(2 * regionSize.height * (regionSize.width - 1));
}

double IyRegion (Mat &first, Mat &next, Size regionSize, Point start){
    double Iy = 0;

    for (int i = 0; i < regionSize.width; i++) {
        for (int j = 0; j < regionSize.height - 1; j++) {
            Iy += first.at<uchar>(Point(start.x + i,start.y + j + 1)) - first.at<uchar>(Point(start.x + i,start.y + j));
            Iy += next.at<uchar>(Point(start.x + i,start.y + j + 1)) - next.at<uchar>(Point(start.x + i,start.y + j));
        }
    }

    return Iy/(2 * regionSize.width * (regionSize.height - 1));
}

double ItRegion (Mat &first, Mat &next, Size regionSize, Point start){
    double It = 0;

    for (int j = 0; j < regionSize.height; j++) {
        for (int i = 0; i < regionSize.width; i++) {
            It += next.at<uchar>(Point(start.x + i, start.y + j)) - first.at<uchar>(Point(start.x + i,start.y + j));
        }
    }

    return It/(regionSize.height * regionSize.width);
}

void FillA (double **Ix, double **Iy, double A[2][2], Size derivativeSize){
    //zero out A
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            A[i][j] = 0;
        }
    }

    for (int i = 0; i < derivativeSize.width; i++) {
        for (int j = 0; j < derivativeSize.height; j++) {
            A[0][0] += pow(Ix[i][j], 2);
            A[1][1] += pow(Iy[i][j], 2);
            A[1][0] += Ix[i][j] * Iy[i][j];
        }
    }
    A[0][1] = A[1][0];
}

void Fillb (double **Ix, double **Iy, double **It, double b[2], Size derivativeSize){
    b[0] = 0;
    b[1] = 0;

    for (int i = 0; i < derivativeSize.width; i++) {
        for (int j = 0; j < derivativeSize.height; j++) {
            b[0] -= It[i][j] * Ix[i][j];
            b[1] -= It[i][j] * Iy[i][j];
        }
    }
}

int Inverse (double A[2][2], double Ai[2][2]){
    //infiseable
    if ((A[0][0] * A[1][1]) - (A[0][1]*A[1][0]) < 1)
        return 0;

    Ai[0][0] =  A[1][1]/(A[0][0]*A[1][1] - A[0][1]*A[1][0]);
    Ai[1][0] = -A[1][0]/(A[0][0]*A[1][1] - A[0][1]*A[1][0]);
    Ai[0][1] = -A[0][1]/(A[0][0]*A[1][1] - A[0][1]*A[1][0]);
    Ai[1][1] =  A[0][0]/(A[0][0]*A[1][1] - A[0][1]*A[1][0]);

    return 1;
}

void MultiplyAb (double Ai[2][2], double b[2], double v[2]){
    v[0] = Ai[0][0]*b[0] + Ai[0][1]*b[1];
    v[1] = Ai[1][0]*b[0] + Ai[1][1]*b[1];
}

void Square(Mat &image, float size, float angle, Point center, Scalar color){
    for (int i = 0; i < 4; i++) {
        line(image, Point(center.x + cos(angle) * size/2, center.y + sin(angle) * size/2), Point(center.x + cos(angle + CV_PI/2) * size/2, center.y + sin(angle + CV_PI/2) * size/2), color, 2);
        angle += CV_PI/2;
    }
}

string NextSquare (Vec4f vaverage, int t, float &square_diagonal, float scale_factor, Point &square_center, int speed, float angular_speed, float &angle, Mat &display, bool simple){
    string text = "";

    if (simple){
        if (vaverage[0] + vaverage[2] > 2*t && vaverage[0] + vaverage[2] > abs(vaverage[1]) + abs(vaverage[3])) {
            square_center.x += speed;
            text = "right";
        }
        else if (vaverage[0] + vaverage[2] < -2*t && vaverage[0] + vaverage[2] < -abs(vaverage[1]) - abs(vaverage[3])){
            square_center.x -= speed;
            text = "left";
        }
    }

    else {
        if (vaverage[0] > t && vaverage[2] < -t && abs(vaverage[0]) + abs(vaverage[2]) > abs(vaverage[1]) + abs(vaverage[3])){
            square_diagonal /= scale_factor;
            text = "zoom-out";
        }
        else if (vaverage[0] < -t && vaverage[2] > t && abs(vaverage[0]) + abs(vaverage[2]) > abs(vaverage[1]) + abs(vaverage[3])){
            square_diagonal *= scale_factor;
            text = "zoom-in";
        }
        else if (vaverage[0] + vaverage[2] > 2*t && vaverage[0] + vaverage[2] > abs(vaverage[1]) + abs(vaverage[3])) {
            square_center.x += speed;
            text = "right";
        }
        else if (vaverage[0] + vaverage[2] < -2*t && vaverage[0] + vaverage[2] < -abs(vaverage[1]) - abs(vaverage[3])){
            square_center.x -= speed;
            text = "left";
        }
        else if (vaverage[1] > t && vaverage[3] < -t){
            angle -= angular_speed;
            text = "rotate ACW";
        }
        else if (vaverage[1] < -t && vaverage[3] > t){
            angle += angular_speed;
            text = "rotate CW";
        }
        else if (vaverage[1] + vaverage[3] > 2*t) {
            square_center.y += speed;
            text = "down";
        }
        else if (vaverage[1] + vaverage[3] < -2*t){
            square_center.y -= speed;
            text = "up";
        }
    }

    if (square_center.x < 0) {
        square_center.x = 0;
    }
    if (square_center.y < 0) {
        square_center.y = 0;
    }
    if (square_center.x > display.cols) {
        square_center.x = display.cols;
    }
    if (square_center.y > display.rows) {
        square_center.y = display.rows;
    }
    if (square_diagonal > display.rows * sqrt(2)) {
        square_diagonal = display.rows * sqrt(2);
    }
    if (square_diagonal < 5 * sqrt(2)) {
        square_diagonal = 5;
    }

    return text;
}
