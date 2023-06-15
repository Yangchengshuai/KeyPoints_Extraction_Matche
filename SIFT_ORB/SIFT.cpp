#include <stdlib.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;

int main()
{
    //图像名，自己实践时替换成自己的路径
    string image_name1="/Users/chengshuai/Documents/work/test/1.png";
    string image_name2="/Users/chengshuai/Documents/work/test/2.png";

    //先读一个彩色图像用于后续绘制特征点匹配对
    cv::Mat color_img1 = cv::imread(image_name1, 1);
    cv::Mat color_img2 = cv::imread(image_name2, 1);

    //将图像转换为灰度图像，用于SIFT特征提取和匹配
    cv::Mat gray_img1 = cv::imread(image_name1, 0);
    cv::Mat gray_img2 = cv::imread(image_name2, 0);

    //计算SIFT特征检测和匹配的时间
    double start = static_cast<double>(cv::getTickCount());
    //提取两幅图像的SIFT特征点并筛选出匹配的特征点
    vector<cv::KeyPoint> keypoints1,keypoints2;
    cv::Mat descriptors1,descriptors2;
    cv::Ptr<cv::FeatureDetector> detector=cv::SiftFeatureDetector::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor=cv::SiftDescriptorExtractor::create();
    cv::Ptr<cv::DescriptorMatcher> matcher=cv::DescriptorMatcher::create("BruteForce");
    
    //----------------------------------------------------------------------------------------------------//
    //opencv3里提供了两种匹配算法，分别是BruteForce和FlannBased，BruteForce是暴力匹配，FlannBased是基于近似最近邻的匹配。
    //BruteForce：通过计算两个特征描述子之间的欧氏距离或其他相似性度量来确定匹配程度。
    //BruteForce_L1:这种匹配类型使用L1范数（曼哈顿距离）作为特征描述子之间的距离度量方式。L1范数是将两个向量各个对应元素的差的绝对值求和作为距离的度量方式。
    //BruteForce_Hamming:这种匹配类型使用汉明距离作为特征描述子之间的距离度量方式。汉明距离是将两个向量各个对应元素的差的绝对值求和作为距离的度量方式。
    //BruteForce_HammingLUT:这种匹配类型使用汉明距离作为特征描述子之间的距离度量方式。汉明距离是将两个向量各个对应元素的差的绝对值求和作为距离的度量方式。这种匹配类型使用了查找表（LUT）来加速汉明距离的计算。
    //BruteForce_SL2:这种匹配类型使用平方欧氏距离作为特征描述子之间的距离度量方式。平方欧氏距离是将两个向量各个对应元素的差的平方求和作为距离的度量方式。
    //FlannBased：基于近似最近邻的匹配，使用快速最近邻搜索包（FLANN）来计算。
    //----------------------------------------------------------------------------------------------------//

    detector->detect(gray_img1,keypoints1);
    detector->detect(gray_img2,keypoints2);

    descriptor->compute(gray_img1,keypoints1,descriptors1);
    descriptor->compute(gray_img2,keypoints2,descriptors2);

    //匹配
    vector<cv::DMatch> matches;
    matcher->match(descriptors1,descriptors2,matches);
    //筛选匹配点
    double min_dist=10000,max_dist=0;
    //找出所有匹配之间的最小距离和最大距离,即是最相似的和最不相似的两组点之间的距离
    for(int i=0;i<descriptors1.rows;i++)
    {
        double dist=matches[i].distance;
        if(dist<min_dist) min_dist=dist;
        if(dist>max_dist) max_dist=dist;
    }
    cout<<"-- Max dist : "<<max_dist<<endl;
    cout<<"-- Min dist : "<<min_dist<<endl;

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<cv::DMatch> filteredMatches;
    for(int i=0;i<descriptors1.rows;i++)
    {
        if(matches[i].distance<=max(2*min_dist,30.0))
        {
            filteredMatches.push_back(matches[i]);
        }
    }
    double time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
    cout<<"cost time: "<<time<<"s"<<endl;
    //输出匹配点对数
    cout<<"good matches:"<<filteredMatches.size()<<endl;

    //画出匹配的特征点
    cv::Mat img_matches;
    //拼接两幅图像作为画布
    cv::hconcat(color_img1,color_img2,img_matches);
    for(int i=0;i<filteredMatches.size();i++)
    {
        cv::Point2f pt1=keypoints1[filteredMatches[i].queryIdx].pt;
        cv::Point2f pt2=keypoints2[filteredMatches[i].trainIdx].pt;
        //特征点坐标需要根据图像偏移量进行修正
        pt2.x+=color_img1.cols;
        cv::line(img_matches,pt1,pt2,cv::Scalar(0,255,0),2);
    }
    
    //保存图片到指定路径
    cv::imwrite("/Users/chengshuai/Documents/work/test/KeyPointsExtractionAndMatche/img_matches.png", img_matches);

    return 0;
}
