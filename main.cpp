#include <stdio.h>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat src_8uc3_img = cv::imread( "images/lena.png", CV_LOAD_IMAGE_COLOR ); 

    if (src_8uc3_img.empty()) {
        printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
    }
    // cv::imshow( "LENA", src_8uc3_img );

    cv::Mat gray_8uc1_img; 
    cv::Mat gray_32fc1_img; 

    cv::cvtColor(src_8uc3_img, gray_8uc1_img, CV_BGR2GRAY); 
    gray_8uc1_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0/255.0); 

    int x = 10, y = 15; 

    uchar p1 = gray_8uc1_img.at<uchar>(y, x); 
    float p2 = gray_32fc1_img.at<float>(y, x); 
    cv::Vec3b p3 = src_8uc3_img.at<cv::Vec3b>(y, x); 

    printf("p1 = %d\n", p1);
    printf("p2 = %f\n", p2);
    printf("p3[ 0 ] = %d, p3[ 1 ] = %d, p3[ 2 ] = %d\n", p3[ 0 ], p3[ 1 ], p3[ 2 ]);

    gray_8uc1_img.at<uchar>(y, x) = 0; 

    cv::rectangle(gray_8uc1_img, cv::Point(65, 84), cv::Point(75, 94),
                  cv::Scalar(50), CV_FILLED);


    cv::Mat gradient_8uc1_img(50, 256, CV_8UC1);

    for (int y = 0; y < gradient_8uc1_img.rows; y++) {
        for (int x = 0; x < gradient_8uc1_img.cols; x++) {
            gradient_8uc1_img.at<uchar>(y, x) = x;
        }
    }

    cv::Mat output_32fc1_img(gray_32fc1_img.rows,gray_32fc1_img.cols,CV_32FC1);
    cv::Mat copy_32fc1_img = gray_32fc1_img.clone();
    float sigma =0.015f;
    float lamd =0.1;
    for (int t = 0; t < 100; t++) {
         for (int y = 1; y < gray_32fc1_img.rows-1; y++) {
            for (int x = 1; x < gray_32fc1_img.cols-1; x++) {
                float north =0.0f;
                float south =0.0f;
                float west =0.0f;
                float east =0.0f;
                north = exp(-pow(copy_32fc1_img.at<float>(y,x-1)-copy_32fc1_img.at<float>(y,x),2)/pow(sigma,2));
                south = exp(-pow(copy_32fc1_img.at<float>(y,x+1)-copy_32fc1_img.at<float>(y,x),2)/pow(sigma,2));
                west = exp(-pow(copy_32fc1_img.at<float>(y-1,x)-copy_32fc1_img.at<float>(y,x),2)/pow(sigma,2));
                east = exp(-pow(copy_32fc1_img.at<float>(y+1,x)-copy_32fc1_img.at<float>(y,x),2)/pow(sigma,2));
                copy_32fc1_img.at<float>(y,x)=(copy_32fc1_img.at<float>(y,x)*(1-(0.1)*(north+south+west+east))+0.1*(north*copy_32fc1_img.at<float>(y,x-1)+
                                                                                                south*copy_32fc1_img.at<float>(y,x+1)+west*copy_32fc1_img.at<float>(y-1,x)+
                                                                                                east*copy_32fc1_img.at<float>(y+1,x)));

            }
        }
    }





    // diplay images
    // cv::imshow("Gradient", gradient_8uc1_img);
    // cv::imshow( "Lena gray", gray_8uc1_img);
    cv::imshow( "Lena gray 32f", gray_32fc1_img);
    cv::imshow( "Example", copy_32fc1_img);


    cv::waitKey( 0 ); 

    return 0;
}
