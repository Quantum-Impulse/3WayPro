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

	adaptiveThreshold(mat, mat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 69, -6);

	return mat;
}

Mat analyzeHands(Mat& mat, bool& temp)
{
	medianBlur(mat, mat, 1);
	Mat combined;

	Mat bgr[3];
	split(mat, bgr);

	future<Mat> b = async(launch::async, aTT, bgr[0]);
	future<Mat> g = async(launch::async, aTT, bgr[1]);
	future<Mat> r = async(launch::async, aTT, bgr[2]);

	while (!(b.wait_for(chrono::seconds(0)) == future_status::ready)
		|| !(g.wait_for(chrono::seconds(0)) == future_status::ready) || !(r.wait_for(chrono::seconds(0)) == future_status::ready)) {
	}

	combined = b.get() + g.get() + r.get();

	morphologyEx(combined, combined, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(7, 7)));
	morphologyEx(combined, combined, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(7, 7)));
	morphologyEx(combined, combined, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(7, 7)));
	morphologyEx(combined, combined, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(7, 7)));

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
	Mat temp;

	while (true)
	{
		cam.read(img);
		Mat boxImg;
		float xmid, ymid;
		xmid = img.cols / 2;
		ymid = img.rows / 2;

		if (getHands)
		{
			Rect myRect(Point(xmid * .6, ymid * .3), Point(xmid * 1.4, ymid * 1.7));
			boxImg = Mat(img, myRect).clone();

			rectangle(img, myRect, Scalar(255, 0, 0), 2, LINE_AA);

			imshow("BoxInfo", boxImg);

			if (waitKey(1) == 122)
			{
				gotHands = true;
				getHands = false;
			}
		}

		if (gotHands)
		{
			temp = analyzeHands(boxImg, gotHands).clone();
		}

		vector<Point> temp2;
		vector<vector<Point>> contours;
		vector<vector<Point>> hand;

		if (!temp.empty())
		{
			findContours(temp, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			drawContours(img, contours, -1, Scalar(0, 255, 0), 2, LINE_AA, noArray(), 214783647, Point(xmid * .6, ymid * .3));
			vector<Point> hull;
			for (size_t t = 0; t < contours.size(); t++)
			{
				approxPolyDP(contours[t], temp2, arcLength(contours[t], true) * .001, true);
				if (fabs(contourArea(temp2)) > 1000)
				{
					hand.push_back(temp2);
				}
				//convexHull(contours[t], );

			}

			//for (size_t a = 0; a < hull.size(); a++)
			//{
			//	Point* p = &hull[a];
			//	//*p += Point(xmid * .6, ymid * .3);
			//	int n = hull.size();
			//	polylines(img, &p, &n, 1, true, Scalar(0, 255, 100), 1, LINE_AA);
			//}
			

			imshow("Hands", temp);
		}

		for (size_t i = 0; i < hand.size(); i++)
		{
			Point* p = &hand[i][0];
			//*p += Point(xmid * .6, ymid * .3);
			//int n = hand[i].size();
			//polylines(img, &p, &n, 1, true, Scalar(128, 0, 128), 1, LINE_AA);
			drawContours(img, hand, -1, Scalar(128, 0, 128), 2, LINE_AA, noArray(), 214783647, Point(xmid * .6, ymid * .3));
		}


		imshow("original", img);

		if (waitKey(1) == 27)
		{
			cout << "ESC PRESSED" << endl;
			break;
		}
	}
}
