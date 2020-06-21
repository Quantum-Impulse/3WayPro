/* 3WayPro.cpp : This file contains the 'main' function. Program execution begins and ends there.
 Authors: 
	* Enrique R.
	* Lance E.
	* Kieran
*/
#include <iostream>
#include <future>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

Mat aTT(Mat mat)
{

	adaptiveThreshold(mat, mat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 49, 0);

	return mat;
}

Mat analyzeHands(Mat &mat, bool &temp)
{
	//medianBlur(mat, mat, 3);
	Mat combined;

	Mat bgr[3];
	split(mat, bgr);

	future<Mat> b = async(launch::async, aTT, bgr[0]);
	future<Mat> g = async(launch::async, aTT, bgr[1]);
	future<Mat> r = async(launch::async, aTT, bgr[2]);

	while (!(b.wait_for(chrono::seconds(0)) == future_status::ready)
		|| !(g.wait_for(chrono::seconds(0)) == future_status::ready) || !(r.wait_for(chrono::seconds(0)) == future_status::ready)) {}

	combined = b.get() + g.get() + r.get();

	temp = false;

	return combined;
}

int main()
{
	Mat img, hand1, hand2;

	bool getHands = true, gotHands = false;

	VideoCapture cam(0);
	cam.set(CAP_PROP_EXPOSURE, 20);
	cam.set(CAP_PROP_AUTO_EXPOSURE, 0);
	cam.set(CAP_PROP_AUTO_WB, 0);

	while (true)
	{
		cam.read(img);
		Mat boxImg;
		if (getHands)
		{
			float xmid, ymid;
			xmid = img.cols / 2;
			ymid = img.rows / 2;
			Rect myRect(xmid * .6, ymid * .5, xmid * .8, ymid * 1.5);
			boxImg = Mat(img, myRect).clone();

			rectangle(img, myRect, Scalar(255, 0, 0), 2, LINE_AA);

			imshow("BoxInfo", boxImg);	

			if (waitKey(1) == 122)
			{
				gotHands = true;
				getHands = false;
			}
		}

		Mat temp;
		if (gotHands)
		{
			temp = analyzeHands(boxImg, gotHands).clone();
		}

		if (temp.empty() == false)
		{
			imshow("Hands", temp);
		}
		imshow("original", img);

		if (waitKey(1) == 27)
		{
			cout << "ESC PRESSED" << endl;
			break;
		}
	}
}
