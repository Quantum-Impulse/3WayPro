/* 3WayPro.cpp : This file contains the 'main' function. Program execution begins and ends there.
 Authors: 
	* Enrique R.
	* Lance E.
	* Kieran H.
*/
#include <iostream>
#include <future>
#include <chrono>
#include <algorithm>
#include "resource.h"
#include <fstream>
#include <string>
#include <Windows.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const double PI = 3.14159265;

using namespace std;
using namespace cv;

double xmid, ymid;

ofstream ino;
ofstream outo;

//based on https://mklimenko.github.io/english/2018/06/23/embed-resources-msvc/
class Resource
{
public:
	HGLOBAL data;
	HRSRC h;
	Resource(int id, LPWSTR type)
	{
		h = FindResource(nullptr, MAKEINTRESOURCE(id), type);
		data = LoadResource(nullptr, h);
	}

	void* loadFont(Resource fontResource)
	{
		return LockResource(fontResource.data);
	}

	void* loadImage(Resource imageResource)
	{
		return LockResource(imageResource.data);
	}
};

vector<Rect2d> findFingers(Mat binary)
{
	return binary;
}

vector<Point> FingerSort(vector<Point> finger1, Rect handbox)
{
	vector<Point> nps;
	for (size_t i = 0; i < finger1.size(); i++)
	{
		finger1[i].x += xmid * .6;
		finger1[i].y += ymid * .3;
		nps.push_back(finger1[i]);
	}

	double yb = handbox.y + handbox.height / 2;

	for (size_t t = 0; t < nps.size(); t++)
	{
		if (nps[t].y > yb)
		{
			nps.erase(nps.begin() + t);
			t = 0;
			continue;
		}
	}

	return nps;
}

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

	adaptiveThreshold(mat, mat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 79, -4 );

	return mat;
}

Mat analyzeHands(Mat mat, bool& temp)
{
	medianBlur(mat, mat, 1);
	cvtColor(mat, mat, COLOR_BGR2HSV);	
	Mat combined;


	auto xmid = mat.cols / 2;
	auto ymid = mat.rows / 2;

	Vec3b color;
	color = mat.at<cv::Vec3b>(cv::Point(xmid, ymid));
	
	cout << color.val[0] << ' ' << color.val[1] << ' ' << color.val[2];


	Scalar values((int)color.val[0], (int) color.val[1], (int) color.val[2]);
	Mat thresholded;
	inRange(mat, values - Scalar(115, 55, 55), values + Scalar(115, 55, 55), thresholded);

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

	return thresholded;
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

	Resource handOutlineR(102, RT_BITMAP);
	//Mat handOutline(200, 200, CV_8UC4, handOutlineR.loadImage(handOutlineR));

	while (true)
	{
		cam.read(img);
		Mat boxImg;
		xmid = img.cols / 2;
		ymid = img.rows / 2;

		if (getHands)
		{
			Rect myRect(Point(xmid * .6, ymid * .3), Point(xmid * 1.4, ymid * 1.7));
			boxImg = Mat(img, myRect).clone();

			circle(img, Point(xmid, ymid), 3, Scalar(0, 255, 0), 2, LINE_AA);

			rectangle(img, myRect, Scalar(255, 0, 0), 2, LINE_AA);

			//img.setTo(Scalar(50, 50, 200), handOutline);


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
			Mat temp5;
			medianBlur(temp, temp5, 5);
			findContours(temp5, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
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

			vector<vector<Point>> fingerpoints;

			auto g = generalizeHull(hulls[0]);
			sort(g.begin(), g.end(), LTRSort);
			size_t j = g.size() - 5;
			fingerpoints.push_back(g);
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

				double a1, a2;
				a1 = atan(m1);
				a2 = atan(m2);

				double bis1, bis2;
				bis1 = (a1 + a2) / 2;
				bis2 = (a1 + a2 + PI) / 2;

				double m3, m4;
				m3 = tan(bis1);
				m4 = tan(bis2);

				double newEq1, newEq2;
				newEq1 = m3*(2)+y2;

				double b1, b2; 
				b1 = y2 - (m1 * x2);
				b2 = y2 - (m2 * x2);

				double A = m1, a = m2, B = 1, b = 1, C = b1, c = b2;

				double bottom1 = sqrt(pow(m1, 2) + 1);
				double bottom2 = sqrt(pow(m2, 2) + 1);

				double total1 = (A * x2 + B * y2 + C) / (bottom1);
				double total2 = (a * x2 + b * y2 + c) / (bottom2);

				double nY1 = (total2 / bottom1) - (A * x2) - (y2);
				double nY2 = (total1 / bottom2) - (a * x2) - (y2);
				double nY3 = (-total2 / bottom1) - (A * x2) - (y2);
				double nY4 = (-total1 / bottom2) - (a * x2) - (y2);

				fingerpoints[0].push_back(g[i - 1]);
				fingerpoints[0].push_back(Point(x3, nY1));


				Rect bbox = boundingRect (g);
				bbox.x += (xmid * .6);
				bbox.y += (ymid * .3);

				circle(img, Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2), 3, Scalar(0, 0, 0), 3, LINE_AA);
				string data = "data.csv";
				

				ino.open(data, ios::out);

				auto fp = FingerSort(g, bbox);
				for (size_t i = 0; i < fp.size(); i++)
				{
					circle(img, fp[i], 3, Scalar(255, 255, 0), 3, LINE_AA);
					ino << fp[i].x << ',' << fp[i].y << ',';
				}
				ino.close();
				

			}
			
			//drawContours(img, fingerpoints, -1, Scalar(100, 60, 10), 2, LINE_AA, noArray(), 214783647, Point(xmid * .6, ymid * .3));

		}
		

		imshow("original", img);

		if (waitKey(1) == 27)
		{
			cout << "ESC PRESSED" << endl;
			break;
		}
	}
}
