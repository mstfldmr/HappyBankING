//source: https://bitbucket.org/omigamedev/smiledetector/src/a1dfa58307e44dbe702b1e753e7438d9d492bc46?at=default

#include <stdio.h>

// OpenCV
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <cvaux.h>

// number of training images
#define M 2
// dimention of the image vector (computed later)
static int N = 0;

using namespace cv;
using namespace std;

int main (int argc, const char * argv[])
{
    //system("pwd");

    IplImage *img_happy = cvLoadImage("data/happy.jpg");
    IplImage *img_sad = cvLoadImage("data/sad.jpg");

    // read the images size
    CvSize sz = cvGetSize(cvLoadImage("data/happy00.png"));
    N = sz.width * sz.height; // compute the vector image length

    // read the training set images
    char file[64];
    Mat I = Mat(N, M, CV_32FC1);
    Mat S = Mat::zeros(N, 1, CV_32FC1);
/*
    for(int i = 0; i < M/2; i++)
    {
        sprintf(file, "data/happy%02d.png", i);
        Mat m = cvLoadImageM(file, CV_LOAD_IMAGE_GRAYSCALE);
        m = m.t();
        m = m.reshape(1, N);
        m.convertTo(m, CV_32FC1);
        m.copyTo(I.col(i));
        S = S + m;
    }
    for(int i = 0; i < M/2; i++)
    {
        sprintf(file, "data/sad%02d.png", i);
        Mat m = cvLoadImageM(file, CV_LOAD_IMAGE_GRAYSCALE);
        m = m.t();
        m = m.reshape(1, N);
        m.convertTo(m, CV_32FC1);
        m.copyTo(I.col(i + M/2));
        S = S + m;
    }
*/
    // calculate eigenvectors
    Mat Mean = S / (float)M;
    Mat A = Mat(N, M, CV_32FC1);
    for(int i = 0; i < M; i++) A.col(i) = I.col(i) - Mean;
    Mat C = A.t() * A;
    Mat V, L;
    eigen(C, L, V);

    // compute projection matrix Ut
    Mat U = A * V;
    Mat Ut = U.t();

    // project the training set to the faces space
    Mat trainset = Ut * A;

    // prepare for face recognition
    CvMemStorage *storage = cvCreateMemStorage(0);
    CvHaarClassifierCascade *cascade = (CvHaarClassifierCascade*)cvLoad("data/haarcascade_frontalface_default.xml");

    printf("starting cam\n");
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
    if(capture == 0)
    {
        printf("No webcam where found");
        return 1;
    }

    CvScalar red = CV_RGB(250,0,0);

    // counters used for frame capture
    int happy_count = 0;
    int sad_count = 0;

    printf("starting recognition\n");
    while(1)
    {
        // Get one frame
        IplImage *img = cvQueryFrame(capture);

        // read the keyboard
        int c = cvWaitKey(50);
        if((c & 255) == 27) break; // quit the program when ESC is pressed

        // find the face
        CvSeq *rects = cvHaarDetectObjects(img, cascade, storage, 1.3, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(50, 50));
        if(rects->total == 0)
        {
            cvShowImage("result", img);
            continue;
        }
        CvRect *roi = 0;
        for(int i = 0; i < rects->total; i++)
        {
            CvRect *r = (CvRect*) cvGetSeqElem(rects, i);

            // draw rect
            CvPoint p1 = cvPoint(r->x, r->y);
            CvPoint p2 = cvPoint(r->x + r->width, r->y + r->height);
            cvRectangle(img, p1, p2, red, 3);

            // biggest rect
            if(roi == 0 || (roi->width * roi->height < r->width * r->height)) roi = r;
        }

        // copy the face in the biggest rect
        int suby = roi->height * 0.6;
        roi->height -= suby;
        roi->y += suby;
        int subx = (roi->width - roi->height) / 2 * 0.7;
        roi->width -= subx * 2;
        roi->x += subx;
        cvSetImageROI(img, *roi);
        IplImage *subimg = cvCreateImage(cvSize(100, 100 * 0.7), 8, 3);
        IplImage *subimg_gray = cvCreateImage(cvGetSize(subimg), IPL_DEPTH_8U, 1);
        cvResize(img, subimg);
        cvCvtColor(subimg, subimg_gray, CV_RGB2GRAY);
        cvEqualizeHist(subimg_gray, subimg_gray);
        cvResetImageROI(img);

        switch(c) // capture a frame when H (happy) or S (sad) is pressed
        {
            char file[32];
            case 'h':
                sprintf(file, "data/happy%02d.png", happy_count);
                cvSaveImage(file, subimg_gray);
                happy_count++;
                cvWaitKey(1000);
                break;
            case 's':
                sprintf(file, "data/sad%02d.png", sad_count);
                cvSaveImage(file, subimg_gray);
                sad_count++;
                cvWaitKey(1000);
                break;
        }

        // recognize mood
        double min = 1000000000000000.0;
        int mini;
        Mat subimgmat = subimg_gray;
        subimgmat = subimgmat.t();
        subimgmat = subimgmat.reshape(1, N);
        subimgmat.convertTo(subimgmat, CV_32FC1);
        Mat proj = Ut * subimgmat;
        // find the minimum distance vector
        for(int i = 0; i < M; i++)
        {
            double n = norm(proj - trainset.col(i));
            if(min > n)
            {
                min = n;
                mini = i;
            }
        }
        printf("%02d\n", mini);
        if(mini == 1) cvShowImage("logo", img_sad);
        else cvShowImage("logo", img_happy);
        cvMoveWindow("logo", 670, 200);

        // show results
        cvShowImage("face", subimg_gray);
        cvMoveWindow("face", 670, 0);
        cvShowImage("result", img);
        cvMoveWindow("result", 0, 0);

    }

    // cleanup
    cvReleaseCapture(&capture);
    cvDestroyWindow("result");
    cvDestroyWindow("logo");
    cvDestroyWindow("face");
    cvReleaseHaarClassifierCascade(&cascade);
    cvReleaseMemStorage(&storage);

    printf("done");
    return 0;
}