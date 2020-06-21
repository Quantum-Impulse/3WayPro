/* 3WayPro.cpp : This file contains the 'main' function. Program execution begins and ends there.
 Authors: 
	* Enrique R.
	* Lance E.
	* Kieran
*/
#include <iostream>
#include <future>
#include <chrono>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const double PI = 3.14159265;

using namespace std;
using namespace cv;

bool CASort(vector<Point> lhs, vector<Point> rhs) { return (contourArea(lhs, true) > contourArea(rhs, true)); }

bool LTRSort(Point lhs, Point rhs) { return lhs.x < rhs.x; }

inline double distance(Point pt0, Point pt1)
{
	return sqrt(pow(pt1.x - pt0.x, 2) + pow(pt1.y - pt0.y, 2));
}

inline double distance(double x1, double y1, double x2, double y2)
{
	return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

bool disSort(Point lhs, Point rhs) { return (distance(Point(0, 0), lhs) < distance(Point(0, 0), rhs)); }

vector<Point> generalizeHull(vector<Point> hull)
{
	vector<Point> g;

	for (size_t t = 0; t < hull.size(); t++)
	{
		double d;

		auto& p1 = hull[t];
		for (size_t a = 0; a < hull.size(); a++)
		{
			auto& p2 = hull[a];
			d = distance(p1, p2);

			if (a == t)
				continue;

			if (d > 10)
			{
				continue;
			}

			else
			{
				hull.erase(hull.begin() + a);
				t = 0;
				break;
			}
		}

	}

	return hull;
}

//returns the angle measure of the middle point as a double and degree
double dAngle(Point pt0, Point pt1, Point pt2)
{
	double a = distance(pt0, pt1);
	double b = distance(pt1, pt2);
	double c = distance(pt2, pt0);
	// C2 = A2 + B2 - 2AB cos(c) law of cosine, lowercase are sides, large case are angles. C is opposite angle of c side.
	double C = acos((pow(a, 2) + pow(b, 2) - pow(c, 2)) / (2 * a * b)) * (180 / PI);
	return C;
}


Mat aTT(Mat mat)
{

	adaptiveThreshold(mat, mat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 79, 0 );

	return mat;
}

Mat analyzeHands(Mat& mat, bool& temp)
{
	medianBlur(mat, mat, 1);
	//cvtColor(mat, mat, COLOR_BGR2HSV);	
	Mat combined;

	Mat bgr[3];
	split(mat, bgr);

	future<Mat> b = async(launch::async, aTT, bgr[0]);
	future<Mat> g = async(launch::async, aTT, bgr[1]);
	future<Mat> r = async(launch::async, aTT, bgr[2]);

	while (!(b.wait_for(chrono::seconds(0)) == future_status::ready)
		|| !(g.wait_for(chrono::seconds(0)) == future_status::ready) || !(r.wait_for(chrono::seconds(0)) == future_status::ready)) {}

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
				gotHands = !gotHands;
				getHands = !getHands;
			}
		}

		if (gotHands)
		{
			temp = analyzeHands(boxImg, gotHands).clone();
		}

		vector<Point> temp2;
		vector<vector<Point>> contours;
		vector<vector<Point>> hand;
		vector<vector<Point>> hulls;

		if (!temp.empty())
		{
			medianBlur(temp, temp, 5);
			findContours(temp, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			drawContours(img, contours, -1, Scalar(0, 255, 0), 2, LINE_AA, noArray(), 214783647, Point(xmid * .6, ymid * .3));
			vector<Point> hull;
			for (size_t t = 0; t < contours.size(); t++)
			{
				approxPolyDP(contours[t], temp2, arcLength(contours[t], true) * .001, true);
				if (fabs(contourArea(temp2)) > 1000)
				{
					hand.push_back(temp2);
					convexHull(contours[t], hull);
					hulls.push_back(hull);
				}

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

		drawContours(img, hand, -1, Scalar(128, 0, 128), 2, LINE_AA, noArray(), 214783647, Point(xmid * .6, ymid * .3));
		sort(hulls.begin(), hulls.end(), CASort);
		if (!hulls.empty() && hulls[0].size() > 6)
		{
			drawContours(img, hulls, 0, Scalar(50, 50, 128), 2, LINE_AA, noArray(), 214783647, Point(xmid * .6, ymid * .3));
			auto pinkangle = dAngle(hulls[0][7], hulls[0][0], hulls[0][1]);
			auto pointangle = dAngle(hulls[0][0], hulls[0][1], hulls[0][2]);
			auto middleangle = dAngle(hulls[0][1], hulls[0][2], hulls[0][3]);
			auto ringangle = dAngle(hulls[0][2], hulls[0][3], hulls[0][4]);
			auto thumbangle = dAngle(hulls[0][3], hulls[0][4], hulls[0][5]);

			double angles[5] = { thumbangle, pointangle, middleangle, ringangle, thumbangle };

			vector<vector<Point>> fingerpoints(1);

			auto g = generalizeHull(hulls[0]);
			sort(g.begin(), g.end(), LTRSort);
			size_t j = g.size() - 5;

			for (size_t i = g.size() - 1; i > j; i--)
			{ /*
				double x, y, s;
				double ra;

				auto &p1 = hulls[0][i], p2 = hulls[0][i + 1];
				switch (i)
				{
				case 0:
					ra = 360 - angles[i];;
					break;
				case 1:
					ra = 360 - angles[i];
					break;
				case 2:
					if (tan(atan2(p1.y, p2.y) * (180 / PI)) * (p2.x - p1.x) > tan(atan2(p2.x, -p2.y) * (180 / PI)))
					{
						ra = 90 + tan(atan2(p1.y, p2.y) * (180 / PI));
					}
					else
					{
						ra = 90 - tan(atan2(p2.x, -p2.y));
					}
					break;
				case 5:


				default:
					ra = angles[i];
					break;
				}
				auto a = (p1.x - p2.x) * (tan((angles[0] / 2) * (PI / 180)));
				auto y3 = a + p1.y;
				x = p2.x; y = y3;

				line(img, p1, Point(x, y), Scalar(0, 100, 100), 2, LINE_AA);
				*/


				auto& x1 = g[i].x;
				auto& y1 = g[i].y;
				auto& x2 = g[i - 1].x;
				auto& y2 = g[i - 1].y;
				auto& x3 = g[i - 2].x;
				auto& y3 = g[i - 2].y;

				double m1, m2;

				if (x2 - x1 == 0 || x3 - x2 == 0)
				{
					continue;
				}

				m1 = (y2 - y1) / (x2 - x1);
				m2 = (y3 - y2) / (x3 - x2);

				double b1, b2;
				b1 = y1 - x1 * m1;
				b2 = y1 - x1 * m1;

				double bt = sqrt(pow(m1, 2) + 1);
				double nm1 = m1 * bt, nc = b1 * bt;
				double ny = (nm1 * x3 + nc);

				fingerpoints[0].push_back(g[i - 1]);
				fingerpoints[0].push_back(Point(x3, ny));


			}
			
			drawContours(img, fingerpoints, -1, Scalar(200, 10, 10), 2, LINE_AA, noArray(), 214783647, Point(xmid * .6, ymid * .3));

		}
		

		imshow("original", img);

		if (waitKey(1) == 27)
		{
			cout << "ESC PRESSED" << endl;
			break;
		}
	}
}
