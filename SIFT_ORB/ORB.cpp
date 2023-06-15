
#include <stdlib.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
int main()
{
    // 读取彩色图像
    cv::Mat image1_color = cv::imread("/Users/chengshuai/Documents/work/test/1.png", cv::IMREAD_COLOR);
    cv::Mat image2_color = cv::imread("/Users/chengshuai/Documents/work/test/2.png", cv::IMREAD_COLOR);

    // 读取灰度图像
    cv::Mat image1_gray = cv::imread("/Users/chengshuai/Documents/work/test/1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image2_gray = cv::imread("/Users/chengshuai/Documents/work/test/1.png", cv::IMREAD_GRAYSCALE);
    
    //计时
    double start = static_cast<double>(cv::getTickCount());

    // 创建ORB对象
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // 检测关键点和计算描述子
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    orb->detectAndCompute(image1_gray, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(image2_gray, cv::noArray(), keypoints2, descriptors2);


    // 创建FLANN匹配器
    //注意：BruteForce_HAMMING匹配类型适用于二进制特征描述子，如ORB（Oriented FAST and Rotated BRIEF）和Brief。
    //这种匹配类型使用汉明距离（Hamming distance）作为特征描述子之间的距离度量方式。汉明距离是计算两个二进制向量之间不同位的数量。
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    // 进行特征匹配
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // 筛选匹配结果
    std::vector<cv::DMatch> goodMatches;
    double minDist = 100.0;
    double maxDist = 0.0;
    for (int i = 0; i < descriptors1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < minDist)
            minDist = dist;
        if (dist > maxDist)
            maxDist = dist;
    }
    double thresholdDist = 0.6 * maxDist;
    for (int i = 0; i < descriptors1.rows; i++)
    {
        if (matches[i].distance < thresholdDist)
            goodMatches.push_back(matches[i]);
    }

    double time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
    cout<<"cost time: "<<time<<"s"<<endl;
    
    // 绘制匹配结果
    cv::Mat result;
    cv::drawMatches(image1_color, keypoints1, image2_color, keypoints2, goodMatches, result);

    // 保存匹配结果
    cv::imwrite("result.jpg", result);

    return 0;
}
