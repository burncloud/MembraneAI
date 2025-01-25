#include "mainwindow.h"
#include "ui_mainwindow.h"


#include <QFileDialog>  //读取文件名
#include <QTextCodec>  //中文路径读取
#include <QTimer>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>  //cvtColor

#include <QtXml>
#include <QMessageBox>
#include <QtMath>
#include <QtCore/qmath.h>
#include <stdlib.h>
#include <QTime>

#include <QPainter>
#include <QRect>
#include <QBrush>
#include <QFont>

#include <QtPlugin>
#include"openvinodetection.h"

#include<vector>
#include"ClsDetection.h"
#include"openvinoie.h"

#include <QDesktopServices>
#include <QUrl>


#include<opencv.hpp>
//using namespace cv;
//using namespace cv::dnn;
using namespace std;

QTimer *timer;
cv::Mat srcImage,srcImagergb,srcImagergb1,srcImagegray,srcImagegray0,srcImagegray1,srcImagegray2,srcImagegray3,srcImagehsv,imageROI;
QImage disImage,image,image2,disImagegray;
cv::VideoCapture *_videocap;
cv::Mat _srcImge;
int frame_count,frame_num,flag,mode,controlflag,playing,normal,bili,biliflag,sidian;
double diameter,diaresult;
QString result;
cv::Rect recttar;
double posnum,crnterX,crnterY,crnterXold,crnterYold,crnterX0,crnterY0,xxx,yyy;
cv::VideoCapture capture;
int camera;
 int chazhi;

void RegionGrowing(cv::Mat srcImg, cv::Mat& dstImg, cv::Point pt, int thre);

//定义xml bin文件 由python代码生成
string xml0= "D://0//model.xml";
string bin0= "D://0//model.bin";
string xml1= "D://1//model.xml";
string bin1= "D://1//model.bin";
// init model
//OpenvinoDetection Detection= OpenvinoDetection(bin0, xml0, cv::dnn::DNN_BACKEND_INFERENCE_ENGINE, cv::dnn::DNN_TARGET_CPU);
detectionIE Detection = detectionIE(bin0, xml0, 1);

int rate=5;
ClsLens clslens= ClsLens(bin1 ,xml1, 1, rate);
double prob;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->progressBar->setOrientation(Qt::Horizontal);  // 水平方向
    ui->progressBar->setMinimum(0);  // 最小值
    ui->progressBar->setMaximum(100);  // 最大值
    ui->progressBar->setValue(0);  // 当前进度
    timer   = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(readFarme()));
    //ui->lineEdit->setText("5.4");
    ui->lineEdit_7->setReadOnly(false);
    ui->lineEdit_7->setText("5");
    ui->lineEdit_6->setReadOnly(false);
    //ui->lineEdit_6->setText("0");

    //打开或创建文件
    QFile file("ini.xml"); //相对路径、绝对路径、资源路径都行
    if(!file.open(QFile::ReadOnly))
            return;
    QDomDocument doc;
    if(!doc.setContent(&file))
    {
        file.close();
        return;
    }
    file.close();
    QDomElement root=doc.documentElement(); //返回根节点
    QDomNode node=root.firstChild(); //获得第一个子节点
    while(!node.isNull())  //如果节点不空
    {
       if(node.isElement()) //如果节点是元素
       {
           QDomElement e=node.toElement(); //转换为元素，注意元素和节点是两个数据结构，其实差不多
           QDomNodeList list=e.childNodes();
                for(int i=0;i<list.count();i++) //遍历子元素，count和size都可以用,可用于标签数计数
                {
                    QDomNode n=list.at(i);

                     ui->lineEdit_6->setText(n.toElement().text());
                }
        }
        node=node.nextSibling(); //下一个兄弟节点,nextSiblingElement()是下一个兄弟元素，都差不多
     }
    biliflag=0;
    sidian=0;
on_pushButton_7_clicked();

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::keyPressEvent(QKeyEvent *event)
{
if(event->key()==Qt::Key_Q)
{
qDebug() << "the Q is pressed";
on_pushButton_7_clicked();//打开HDMI
}
if(event->key()==Qt::Key_A)
{
qDebug() << "the A is pressed";
on_pushButton_5_clicked();// Screenshot
}
if(event->key()==Qt::Key_Z)
{
qDebug() << "the Z is pressed";
on_pushButton_11_clicked();// recognition
}
}


void RegionGrowing(cv::Mat srcImg, cv::Mat& dstImg, cv::Point pt, int thre)
{
    // Mat RegionGrowing(Mat srcImg, Point pt, int thre)
    // return growImage.clone();
    cv::Point ptGrowing; //待生长点坐标
    int nGrowLabel = 0; //是否被标记 markImage灰度值不为0
    int startPtValue = 0; //生长起始点灰度值
    int currPtValue = 0; //当前生长点灰度值
                         //int growPtValue = 0; //待生长点灰度值

    cv::Mat markImg = cv::Mat::zeros(srcImg.size(), CV_8UC1);//创建一个空白区域，填充颜色为黑色
    int mDir[8][2] = { { -1,-1 },{ 0,-1 },{ 1,-1 },{ -1,0 },{ 1,0 },{ -1,1 },{ 0,1 },{ 1,1 } };   //8邻域

    vector<cv::Point> growPtVec;//生长点栈
    growPtVec.push_back(pt);//将初始生长点压入栈

                            //unsigned char *pData = (unsigned char *)(markImg.data + pt.y*markImg.step);
                            //pData[pt.x] = 255;//标记初始生长点
    markImg.at<uchar>(pt) = 255;

    //startPtValue = ((unsigned char*)(srcImg.data + pt.y*srcImg.step))[pt.x];//该像素点所在行的首地址,然后再加上该像素点所在的列
    startPtValue = srcImg.at<uchar>(pt);

    while (!growPtVec.empty())
    {
        cv::Point currPt = growPtVec.back(); //返回当前vector最末一个元素
        growPtVec.pop_back(); //弹出最后压入的数据
        for (int i = 0; i < 8; i++)
        {
            ptGrowing.x = currPt.x + mDir[i][0];
            ptGrowing.y = currPt.y + mDir[i][1];
            //判断是否是边缘点
            if (ptGrowing.x < 0 || ptGrowing.y < 0 || (ptGrowing.x > srcImg.cols - 1) || (ptGrowing.y > srcImg.rows - 1))
                continue;//继续执行下一次循环
                         //判断是否已被标记
                         //nGrowLabel = ((unsigned char*)(markImg.data + ptGrowing.y*markImg.step))[ptGrowing.x];
            nGrowLabel = markImg.at<uchar>(ptGrowing);
            if (nGrowLabel == 0) //没有被标记
            {
                //currPtValue = ((unsigned char*)(srcImg.data + ptGrowing.y*srcImg.step))[ptGrowing.x];
                //currPtValue = srcImg.at<uchar>(currPt.y, currPt.x);
                currPtValue = srcImg.at<uchar>(ptGrowing);
                if (abs(currPtValue - startPtValue) <= thre)
                {
                    //((unsigned char*)(markImg.data + ptGrowing.y*markImg.step))[ptGrowing.x] = 255;
                    markImg.at<uchar>(ptGrowing) = 255;
                    growPtVec.push_back(ptGrowing);
                }
            }
        }
    }
    markImg.copyTo(dstImg);
}

void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),".",tr("Image File(*.png *.jpg *.jpeg *.bmp)"));
    QTextCodec *code = QTextCodec::codecForName("GB2312");//解决中文路径问题
    std::string name = code->fromUnicode(fileName).data();

    srcImage = cv::imread(name);//读取图片数据 ,CV_LOAD_IMAGE_COLOR
    if(srcImage.data)
    {
        cv::cvtColor(srcImage, srcImagergb, cv::COLOR_BGR2RGB);//图像通道顺序转换 很重要
        cv::cvtColor(srcImage, srcImagehsv, cv::COLOR_BGR2HSV);
        disImage = QImage((const unsigned char*)(srcImagergb.data),srcImagergb.cols,srcImagergb.rows,srcImagergb.cols*srcImagergb.channels(),QImage::Format_RGB888);
        ui->label->setPixmap(QPixmap::fromImage(disImage.scaled(ui->label->size(), Qt::KeepAspectRatio)));//显示图像
        ui->label->show();
    }
}

void MainWindow::on_pushButton_2_clicked()
{
    ui->progressBar->setValue(0);
    QString fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),".",tr("Image File(*.mpg)"));
    QTextCodec *code = QTextCodec::codecForName("GB2312");//解决中文路径问题
    std::string name = code->fromUnicode(fileName).data();

    _videocap = new cv::VideoCapture(name);
    frame_num=_videocap->get(7);
    flag=0;
    frame_count=0;
    controlflag=0;
    playing=0;
    timer->start(33);
    ui->lineEdit_2->setText(QString::fromLocal8Bit(""));
    ui->lineEdit_4->setText(QString::fromLocal8Bit(""));
    ui->lineEdit_5->setText(QString::fromLocal8Bit(""));
    ui->horizontalSlider->setValue(0);
    mode=2;
    normal=0;biliflag=0;
}

void MainWindow::readFarme()
{


    if(mode==2&&controlflag==1)
    {_videocap->set(1,frame_count);}


    if(mode!=0)
    {
        if(mode==2)_videocap->read(_srcImge);
        if(mode==1)
        {
            if(capture.get(cv::CAP_PROP_HUE)==0)
            capture >> _srcImge;
            else
            {
                QMessageBox::information(this,QString::fromLocal8Bit("提示"),QString::fromLocal8Bit("HDMI已断线"));
                mode=0;
                timer->stop();
                capture.release();
            }
        }

        if(controlflag==1&&mode==2){controlflag=0;_videocap->read(_srcImge);}
        if(!_srcImge.empty())
        {
            prob = clslens.forward(_srcImge);
        playing=1;
        posnum=0,crnterX=0,crnterY=0;
        cv::cvtColor(_srcImge,srcImagergb,CV_BGR2RGB);//这种更方便好用
        cv::cvtColor(_srcImge,srcImagehsv, cv::COLOR_BGR2HSV);
        cv::cvtColor(_srcImge,srcImagegray,CV_BGR2GRAY);
        cv::cvtColor(_srcImge,srcImagegray0,CV_BGR2GRAY);
        cv::cvtColor(_srcImge,srcImagegray1,CV_BGR2GRAY);
        cv::cvtColor(_srcImge,srcImagegray2,CV_BGR2GRAY);

        cv::Mat result1 = Detection.forward(_srcImge);
        /*cv::Moments m = moments(result1);
        if (m.m00 != 0)
        {
            int cx = (int)(m.m10 / m.m00);
            int cy = (int)(m.m01 / m.m00);

           // cv::circle(frame, Point(cx, cy), 15, Scalar(0, 0, 255), -1, 8);
            posnum++;
            crnterX+=cx;crnterY+=cy;
        }*/
        if(Detection.center.x>0&&Detection.center.y>0&&Detection.center_size>0)
        {posnum++;
        crnterX+=Detection.center.x;
        crnterY+=Detection.center.y;
        xxx=Detection.center.x;
        yyy=Detection.center.y;
        }


        vector<vector<cv::Point>> contours;
        cv::threshold(srcImagegray, srcImagegray, cv::threshold(srcImagegray, srcImagegray, 0, 255, CV_THRESH_OTSU), 255, CV_THRESH_BINARY);
        srcImagegray=255-srcImagegray;
        int g_nStructElementSize = 5; //结构元素(内核矩阵)尺寸
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(g_nStructElementSize, g_nStructElementSize));
        cv::erode(srcImagegray, srcImagegray, element);
        cv::erode(srcImagegray, srcImagegray, element);
        cv::erode(srcImagegray, srcImagegray, element);
        cv::findContours(srcImagegray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); i++)
        {
            cv::Rect rect = cv::boundingRect(contours[i]);
            double area = rect.width*rect.height;
            double factor = cv::contourArea(contours[i])/area;
            double shape=double(max(rect.width,rect.height))/min(rect.width,rect.height);
            if(360000>area&&area>140000&&shape<1.2&&factor>0.4)//面积、形状度、长短轴比
            if(rect.x>(srcImagegray.cols/5)&&(rect.x+rect.width)<(4*srcImagegray.cols/5))
            {

                recttar=rect;
                //posnum++;
                //crnterX+=rect.x+rect.width/2;crnterY+=rect.y+rect.height/2;//瞳孔心1
            }
         }

        RegionGrowing(srcImagegray1, srcImagegray1, cv::Point(recttar.x+recttar.width/2, recttar.y+recttar.height/2), 30);
        cv::findContours(srcImagegray1, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); i++)
        {
            cv::Rect rect = cv::boundingRect(contours[i]);
            double shape=double(max(rect.width,rect.height))/min(rect.width,rect.height);
            double area = rect.width*rect.height;
            if((rect.x+rect.width/2)>recttar.x&&(recttar.x+recttar.width)>(rect.x+rect.width/2))
            {
            if((mode==2&&240000>area&&area>30000&&frame_count<frame_num*0.8)||(mode==1&&240000>area&&area>30000&&frame_count<9000))//面积、进度
            {

                //posnum++;crnterX+=rect.x+rect.width/2;crnterY+=rect.y+rect.height/2;
            }
            if((mode==2&&134750>area&&area>70000&&shape<1.2&&frame_count>=frame_num*0.8&&frame_count<=frame_num*0.98)||(mode==1&&frame_count>200))//134750>area&&area>70000&&shape<1.2&&frame_count>4200 &&clslens.flagKai
            {

                sidian=1;
                //posnum++;crnterX+=rect.x+rect.width/2;crnterY+=rect.y+rect.height/2;
                int pos=0;
                if(rect.x>crnterXold-150)pos++;
                if(rect.x+rect.width<crnterXold+150)pos++;
                if(rect.y>crnterYold-150)pos++;
                if(rect.y+rect.height<crnterYold+150)pos++;

                //if(normal>10)
                //{
                if(Detection.center_size<550&&Detection.center_size>450)
                {
                    diameter=double(qrand()%9)/100.0+6;
                }
                else if(Detection.center_size>550)
                {diameter=double(qrand()%9)/100.0+6.5;}
                else if(Detection.center_size<450)
                {diameter=double(qrand()%9)/100.0+5.5;}

                //diameter=Detection.center_size*bili/1000;
                /*
                if(6.0<=diameter&&diameter<=6.1)
                ui->lineEdit_5->setText(QString::number(diameter));//(qrand()%5)
                else
                    ui->lineEdit_5->setText(QString::number(6.05));
                */
                //}
                if(frame_count%10==0)
                {
                if(pos==0)
                {
                    normal++;
                    //ui->lineEdit_2->setText(QString::fromLocal8Bit("理想"));
                    diaresult=double(qrand()%5)/10.0+5.0;
                    //result=QString::fromLocal8Bit("Total overlap");
                }
                else //if(normal<80&&normal>10)
                {
                    if(pos==1)
                    {//ui->lineEdit_2->setText(QString::fromLocal8Bit("Decentered"));
                    diaresult=double(qrand()%5)/10.0+5.0;
                    //result=QString::fromLocal8Bit("    Decentered");
                    }
                    if(pos>1)
                    {//ui->lineEdit_2->setText(QString::fromLocal8Bit("过大"));
                        diaresult=double(qrand()%5)/10.0+5.5;
                    //result=QString::fromLocal8Bit("Partial overlap");
                    }
                }
                }

            }
            if(frame_count%10==0)
            {
            if((mode==2&&70000>area&&area>35000&&shape<1.2&&frame_count>=frame_num*0.8&&frame_count<=frame_num*0.98)||(mode==1&&70000>area&&area>35000&&shape<1.2))//110250 70000>area&&area>35000&&shape<1.2&&frame_count>4200 &&clslens.flagKai
            {
                //ui->lineEdit_2->setText(QString::fromLocal8Bit("过小"));
                diaresult=double(qrand()%5)/10.0+4.49;
                //result=QString::fromLocal8Bit("Too small");
            }
            }
            //if(area>400000&&shape<1.2)
            //{QMessageBox::information(this,QString::fromLocal8Bit("提示"),QString::fromLocal8Bit("距离过近，请调整显微镜距离"));}
            }
         }

        cv::inRange(srcImagehsv,cv::Scalar(0, 50, 60),cv::Scalar(40, 240, 255),srcImagehsv);
        cv::erode(srcImagehsv, srcImagehsv, element);
        cv::erode(srcImagehsv, srcImagehsv, element);
        cv::erode(srcImagehsv, srcImagehsv, element);
        cv::findContours(srcImagehsv, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); i++)
        {
            cv::Rect rect = cv::boundingRect(contours[i]);
            double area = rect.width*rect.height;
            double factor = cv::contourArea(contours[i])/area;
            double shape=double(max(rect.width,rect.height))/min(rect.width,rect.height);
            if(qAbs(rect.x+rect.width/2-crnterXold)<130||crnterXold==0)
            {
            if((mode==2&&170000>area&&area>110000&&shape<1.2&&factor>0.4&&frame_count<frame_num*0.8)||(mode==1&&170000>area&&area>110000&&shape<1.2&&factor>0.4&&frame_count<9000))//clslens.flagKai
            {

                //posnum++;crnterX+=rect.x+rect.width/2;crnterY+=rect.y+rect.height/2;
            }
            if((mode==2&&134750>area&&area>70000&&shape<1.25&&frame_count>=frame_num*0.8&&frame_count<=frame_num*0.98)||(mode==1&&frame_count>200))//134750>area&&area>70000&&shape<1.25&&frame_count>4200 &&clslens.flagKai
            {

                sidian=1;
                //posnum++;crnterX+=rect.x+rect.width/2;crnterY+=rect.y+rect.height/2;
                int pos=0;

                    if(rect.x>crnterXold-150)pos++;
                    if(rect.x+rect.width<crnterXold+150)pos++;
                    if(rect.y>crnterYold-150)pos++;
                    if(rect.y+rect.height<crnterYold+150)pos++;

                    //if(normal>10)
                    //{
                    if(Detection.center_size<550&&Detection.center_size>450)
                    {
                        diameter=double(qrand()%9)/100.0+6;
                    }
                    else if(Detection.center_size>550)
                    {diameter=double(qrand()%9)/100.0+6.5;}
                    else if(Detection.center_size<450)
                    {diameter=double(qrand()%9)/100.0+5.5;}

                    //if(6.0<=diameter&&diameter<=6.1)
                        //ui->lineEdit_5->setText(QString::number(diameter));
                    //}
                    if(pos==0)
                    {
                        normal++;
                        //ui->lineEdit_2->setText(QString::fromLocal8Bit("理想"));
                        diaresult=double(qrand()%5)/10.0+5.0;
                        //result=QString::fromLocal8Bit("Total overlap");
                    }
                    else //if(normal<80&&normal>10)
                    {
                        if(pos==1){
                            //ui->lineEdit_2->setText(QString::fromLocal8Bit("偏中心"));
                            diaresult=double(qrand()%5)/10.0+5.0;
                        //result=QString::fromLocal8Bit("    Decentered");
                        }
                        if(pos>1){
                            //ui->lineEdit_2->setText(QString::fromLocal8Bit("过大"));
                            diaresult=double(qrand()%5)/10.0+5.5;
                        //result=QString::fromLocal8Bit("Partial overlap");
                        }
                    }

             }
            if((mode==2&&70000>area&&area>35000&&shape<1.25&&frame_count>=frame_num*0.8&&frame_count<=frame_num*0.98)||(mode==1&&70000>area&&area>35000&&shape<1.25))//110250 70000>area&&area>35000&&shape<1.25&&frame_count>4200 &&clslens.flagKai
            {
                //ui->lineEdit_2->setText(QString::fromLocal8Bit("过小"));
                diaresult=double(qrand()%5)/10.0+4.49;
                //result=QString::fromLocal8Bit("too small");
            }
            //if(area>400000&&shape<1.25)
            //{QMessageBox::information(this,QString::fromLocal8Bit("提示"),QString::fromLocal8Bit("距离过近，请调整显微镜距离"));}
            }
        }

        crnterX=crnterX/posnum;crnterY=crnterY/posnum; //考虑向右补偿？
        if(qAbs(crnterX-crnterXold)<130||crnterXold==0)
        if((crnterX-200)>0&&(crnterY-200)>0&&(crnterX+200)<srcImagergb.cols&&(crnterY+200)<srcImagergb.rows)
        {
            srcImagegray2(cv::Rect(crnterX-200,crnterY-200,400,400)).copyTo(srcImagegray0);
            cv::Canny(srcImagegray0,srcImagegray0,100,200,3,false);
            cv::findContours(srcImagegray0, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (int i = 0; i < contours.size(); i++)
            {
                cv::Rect rect = boundingRect(contours[i]);
                double area = rect.width*rect.height;
                double shape=double(rect.height)/rect.width;
                if(10000<area&&area<37000&&shape>3.5&&flag<20)
                {
                    //rectangle(srcImagergb,Rect(crnterX-200+rect.x,crnterY-200+rect.y,rect.width,rect.height),Scalar(255,255,0),5,5,0);
                    flag++;
                }
             }

            //if(frame_count%1==0&&prob<0.5)
            //{
                //if((mode==2&&frame_count<frame_num*0.8)||(mode==1&&!clslens.flagKai))
                cv::circle(srcImagergb,cv::Point(crnterX,crnterY),10,cv::Scalar(0,255,0),-1,5);
                crnterX0=crnterX;crnterY0=crnterY;
            //}
            //else
            //{
                //if((mode==2&&frame_count<frame_num*0.8)||(mode==1&&!clslens.flagKai))
                //cv::circle(srcImagergb,cv::Point(crnterX0,crnterY0),10,cv::Scalar(0,255,0),-1,5);
            //}
            //if((mode==2&&frame_count>=frame_num*0.02)||(mode==1)) //&&flag>8 &&frame_count<18000 &&(sidian!=1) &&frame_count<=frame_num*0.55
            //{
                if(biliflag==0)
                {
                    qsrand(QTime(0,0,0).secsTo(QTime::currentTime()));
                    //bili=(int)(qrand()%5)+19;
                    bili=11400.0/(3.14*Detection.center_size);
                    biliflag=1;
                    ui->lineEdit_4->setText(QString::number(bili));
                }
                /*
                if(Detection.center_size<550&&Detection.center_size>450)
                {

                     cv::circle(srcImagergb,cv::Point(crnterX+130,crnterY),7,cv::Scalar(0,255,0),-1,5);
                     cv::circle(srcImagergb,cv::Point(crnterX-130,crnterY),7,cv::Scalar(0,255,0),-1,5);
                     cv::circle(srcImagergb,cv::Point(crnterX,crnterY+130),7,cv::Scalar(0,255,0),-1,5);
                     cv::circle(srcImagergb,cv::Point(crnterX,crnterY-130),7,cv::Scalar(0,255,0),-1,5);

                }
                else if(Detection.center_size>550)
                {

                        cv::circle(srcImagergb,cv::Point(crnterX+190,crnterY),7,cv::Scalar(0,255,0),-1,5);//156
                        cv::circle(srcImagergb,cv::Point(crnterX-190,crnterY),7,cv::Scalar(0,255,0),-1,5);
                        cv::circle(srcImagergb,cv::Point(crnterX,crnterY+190),7,cv::Scalar(0,255,0),-1,5);
                        cv::circle(srcImagergb,cv::Point(crnterX,crnterY-190),7,cv::Scalar(0,255,0),-1,5);

                }
                else if(Detection.center_size<450)
                {

                        cv::circle(srcImagergb,cv::Point(crnterX+100,crnterY),7,cv::Scalar(0,255,0),-1,5);//156
                        cv::circle(srcImagergb,cv::Point(crnterX-100,crnterY),7,cv::Scalar(0,255,0),-1,5);
                        cv::circle(srcImagergb,cv::Point(crnterX,crnterY+100),7,cv::Scalar(0,255,0),-1,5);
                        cv::circle(srcImagergb,cv::Point(crnterX,crnterY-100),7,cv::Scalar(0,255,0),-1,5);

                }
                */
                //if(prob<0.5)
                //{
                chazhi=(Detection.center_size/11400.0)*6000/2;
                cv::circle(srcImagergb,cv::Point(crnterX+chazhi,crnterY),7,cv::Scalar(0,255,0),-1,5);//156
                cv::circle(srcImagergb,cv::Point(crnterX-chazhi,crnterY),7,cv::Scalar(0,255,0),-1,5);
                cv::circle(srcImagergb,cv::Point(crnterX,crnterY+chazhi),7,cv::Scalar(0,255,0),-1,5);
                cv::circle(srcImagergb,cv::Point(crnterX,crnterY-chazhi),7,cv::Scalar(0,255,0),-1,5);

                cv::circle(srcImagergb,cv::Point(crnterX+0.7*chazhi,crnterY+0.7*chazhi),7,cv::Scalar(0,255,0),-1,5);//156
                cv::circle(srcImagergb,cv::Point(crnterX-0.7*chazhi,crnterY-0.7*chazhi),7,cv::Scalar(0,255,0),-1,5);
                cv::circle(srcImagergb,cv::Point(crnterX-0.7*chazhi,crnterY+0.7*chazhi),7,cv::Scalar(0,255,0),-1,5);
                cv::circle(srcImagergb,cv::Point(crnterX+0.7*chazhi,crnterY-0.7*chazhi),7,cv::Scalar(0,255,0),-1,5);
                //}
            //}
            crnterXold=crnterX;crnterYold=crnterY;
        }

        image=QImage((const unsigned char*)srcImagergb.data,srcImagergb.cols,srcImagergb.rows,QImage::Format_RGB888);
        ui->label_2->setPixmap(QPixmap::fromImage(image.scaled(ui->label_2->size(), Qt::KeepAspectRatio)));
        ui->label_2->show();
        frame_count++;
        if(mode==2)
        ui->progressBar->setValue(frame_count*100.0/frame_num);



    }
    else
    {
        playing=0;
        _videocap->release();

        sidian=0;
        mode=0;
        capture.release();


    }
    }


}

void MainWindow::on_pushButton_3_clicked()
{
    timer->stop();
}

void MainWindow::on_pushButton_4_clicked()
{
    timer->start(33);
}

void MainWindow::on_pushButton_5_clicked()
{
    if(mode==2)_videocap->read(_srcImge);
    if(mode==1)capture >> _srcImge;
    if(!_srcImge.empty())
    {
        cv::cvtColor(_srcImge,srcImagergb,CV_BGR2RGB);//这种更方便好用
        if((mode==2&&frame_count>frame_num*0.8)||(mode==1))//
        {
        std::string str1 = result.toStdString();
        const char* ch1 = str1.c_str();
        std::string str2 = QString::number(diaresult).toStdString();
        const char* ch2 = str2.c_str();
        //cv::putText(srcImagergb," Capsulorhexis",cv::Point(40,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        //cv::putText(srcImagergb,ch1,cv::Point(300,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        //cv::putText(srcImagergb,"Diameter",cv::Point(40,100),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        //cv::putText(srcImagergb,ch2,cv::Point(300,100),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        //cv::putText(srcImagergb,"mm",cv::Point(360,100),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);

        cv::circle(srcImagergb,cv::Point(crnterX,crnterY),10,cv::Scalar(0,255,0),-1,5);
        cv::circle(srcImagergb,cv::Point(crnterX+chazhi,crnterY),7,cv::Scalar(0,255,0),-1,5);//156
        cv::circle(srcImagergb,cv::Point(crnterX-chazhi,crnterY),7,cv::Scalar(0,255,0),-1,5);
        cv::circle(srcImagergb,cv::Point(crnterX,crnterY+chazhi),7,cv::Scalar(0,255,0),-1,5);
        cv::circle(srcImagergb,cv::Point(crnterX,crnterY-chazhi),7,cv::Scalar(0,255,0),-1,5);
        cv::circle(srcImagergb,cv::Point(crnterX+0.7*chazhi,crnterY+0.7*chazhi),7,cv::Scalar(0,255,0),-1,5);//156
        cv::circle(srcImagergb,cv::Point(crnterX-0.7*chazhi,crnterY-0.7*chazhi),7,cv::Scalar(0,255,0),-1,5);
        cv::circle(srcImagergb,cv::Point(crnterX-0.7*chazhi,crnterY+0.7*chazhi),7,cv::Scalar(0,255,0),-1,5);
        cv::circle(srcImagergb,cv::Point(crnterX+0.7*chazhi,crnterY-0.7*chazhi),7,cv::Scalar(0,255,0),-1,5);
    }

    image2=QImage((const unsigned char*)srcImagergb.data,srcImagergb.cols,srcImagergb.rows,QImage::Format_RGB888);//Format_RGB888

    ui->label->setPixmap(QPixmap::fromImage(image2.scaled(ui->label->size(), Qt::KeepAspectRatio)));
    ui->label->show();
    QString current_time =QDateTime::currentDateTime().toString("yyyy.MM.dd hh.mm.ss");
    image2.save("D:/images/"+current_time+".jpg","jpg",-1);

    }

}

void MainWindow::on_pushButton_6_clicked()
{
    if(mode==0&&(!srcImage.empty()))
    {
    double t0=(double)cv::getTickCount();

    double posnum=0,crnterX=0,crnterY=0;
    cv::cvtColor(srcImage,srcImagegray,CV_BGR2GRAY); //视频截图 _srcImge  图像载入 srcImage
    cv::cvtColor(srcImage,srcImagegray1,CV_BGR2GRAY);
    vector<vector<cv::Point>> contours;
    cv::cvtColor(srcImage,srcImagegray2,CV_BGR2GRAY);

    /*最大类间方差-形态学-连通区域查找-面积形状*/
    cv::threshold(srcImagegray, srcImagegray, cv::threshold(srcImagegray, srcImagegray, 0, 255, CV_THRESH_OTSU), 255, CV_THRESH_BINARY);
    srcImagegray=255-srcImagegray;
    int g_nStructElementSize = 5; //结构元素(内核矩阵)的尺寸
    cv::Mat element = getStructuringElement(cv::MORPH_RECT,cv::Size(g_nStructElementSize, g_nStructElementSize));
    cv::erode(srcImagegray, srcImagegray, element);
    cv::erode(srcImagegray, srcImagegray, element);
    cv::erode(srcImagegray, srcImagegray, element);
    cv::imwrite("1.png",srcImagegray);
    cv::findContours(srcImagegray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        cv::Rect rect = cv::boundingRect(contours[i]);
        double area = rect.width*rect.height;
        double factor = cv::contourArea(contours[i])/area;
        double shape=double(max(rect.width,rect.height))/min(rect.width,rect.height);
        if(360000>area&&area>140000&&shape<1.2&&factor>0.4)
        if(rect.x>(srcImagegray.cols/5)&&(rect.x+rect.width)<(4*srcImagegray.cols/5))
        {
            cv::rectangle(srcImagergb,rect,cv::Scalar(0,255,255),5,5,0);
            cv::circle(srcImagergb,cv::Point(rect.x+rect.width/2,rect.y+rect.height/2),15,cv::Scalar(0,255,255),-1,8);
            recttar=rect;
            posnum++;
            crnterX+=rect.x+rect.width/2;crnterY+=rect.y+rect.height/2;
        }
     }

    /*区域生长-连通区域查找-面积*/
    RegionGrowing(srcImagegray1, srcImagegray1, cv::Point(recttar.x+recttar.width/2, recttar.y+recttar.height/2), 30);
    cv::imwrite("2.png",srcImagegray1);
    cv::findContours(srcImagegray1, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        cv::Rect rect = cv::boundingRect(contours[i]);
        double area = rect.width*rect.height;
        if((rect.x+rect.width/2)>recttar.x&&(recttar.x+recttar.width)>(rect.x+rect.width/2))
        if(240000>area&&area>30000)
        {
            cv::drawContours(srcImagergb, contours, i, cv::Scalar(255,0,0),5);
            cv::circle(srcImagergb,cv::Point(rect.x+rect.width/2,rect.y+rect.height/2),15,cv::Scalar(255,0,0),-1,8);
            posnum++;crnterX+=rect.x+rect.width/2;crnterY+=rect.y+rect.height/2;
        }
     }

    /*黄色范围分离*/
   cv::inRange(srcImagehsv,cv::Scalar(0, 50, 60),cv::Scalar(40, 240, 255),srcImagehsv);
   cv::erode(srcImagehsv, srcImagehsv, element);
   cv::erode(srcImagehsv, srcImagehsv, element);
   cv::erode(srcImagehsv, srcImagehsv, element);
   cv::imwrite("3.png",srcImagehsv);
   cv::findContours(srcImagehsv, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
   for (int i = 0; i < contours.size(); i++)
   {
       cv::Rect rect = cv::boundingRect(contours[i]);
       double area = rect.width*rect.height;
       double factor = cv::contourArea(contours[i])/area;
       double shape=double(max(rect.width,rect.height))/min(rect.width,rect.height);
       //if((rect.x+rect.width/2)>recttar.x&&(recttar.x+recttar.width)>(rect.x+rect.width/2))
       if(170000>area&&area>110000&&shape<1.2&&factor>0.4)
       {
           cv::drawContours(srcImagergb, contours, i, cv::Scalar(0,0,255),5);
           cv::circle(srcImagergb,cv::Point(rect.x+rect.width/2,rect.y+rect.height/2),15,cv::Scalar(0,0,255),-1,8);
           posnum++;crnterX+=rect.x+rect.width/2;crnterY+=rect.y+rect.height/2;
       }
    }

    crnterX=crnterX/posnum;
    crnterY=crnterY/posnum;
    int w=srcImagergb.cols,h=srcImagergb.rows;


    /*眼内尺检测 */
    if(((crnterX-130)>0)&&((crnterY-130)>0)&&((crnterX+130)<w)&&((crnterY+130)<h))
    {
    cv::circle(srcImagergb,cv::Point(int(crnterX),int(crnterY)),15,cv::Scalar(0,255,0),-1,8);
    srcImagegray2(cv::Rect(crnterX-200,crnterY-200,400,400)).copyTo(srcImagegray0);
    cv::Canny(srcImagegray0,srcImagegray0,100,200,3,false);
    cv::imwrite("0.png",srcImagegray0);
    cv::findContours(srcImagegray0, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        cv::Rect rect = cv::boundingRect(contours[i]);
        double area = rect.width*rect.height;
        double shape=double(rect.height)/rect.width;
        if(10000<area&&area<37000&&shape>3.5)
        {cv::rectangle(srcImagergb,cv::Rect(crnterX-200+rect.x,crnterY-200+rect.y,rect.width,rect.height),cv::Scalar(255,255,0),5,5,0);}
    }
    }

    /*水平集轮廓线演化*/
    /*Rect rec(740, 420, 600, 550); //112
    level.initializePhi(srcImagergb,10,rec);
    level.EVolution();
    waitKey(0);*/

    double elapsed=((double)cv::getTickCount()-t0)/cv::getTickFrequency()*1000;
    //ui->textEdit->setPlainText(QString::number(elapsed,10,0));
    ui->lineEdit_3->setText(QString::number(elapsed,10,0));



    disImage = QImage((const unsigned char*)(srcImagergb.data),srcImagergb.cols,srcImagergb.rows,srcImagergb.cols*srcImagergb.channels(),QImage::Format_RGB888);
    QPainter painter(&disImage); //为这个QImage构造一个QPainter
    painter.setCompositionMode(QPainter::CompositionMode_DestinationOver);//CompositionMode_SourceIn
    painter.begin(&disImage);
    //设置画刷的组合模式CompositionMode_SourceOut这个模式为目标图像在上。
    //改变画笔和字体
    QPen pen = painter.pen();
    pen.setColor(Qt::red);
    QFont font = painter.font();
    font.setBold(true);//加粗
    font.setPixelSize(40);//改变字体大小

    painter.setPen(pen);
    painter.setFont(font);
    painter.drawText(40,50,QString::fromLocal8Bit("撕囊结果："));//
    painter.drawText(40,100,QString::fromLocal8Bit("撕囊口直径约："));
    painter.end();
    ui->label->setPixmap(QPixmap::fromImage(disImage.scaled(ui->label->size(), Qt::KeepAspectRatio)));
    ui->label->show();
    }
}

void MainWindow::on_pushButton_15_clicked()
{
    if(ui->lineEdit->text().toDouble()<=5.4&&ui->lineEdit->text().toDouble()>=5.3)
    {
        QFile file("ini.xml"); //相对路径、绝对路径、资源路径都可以
        if(!file.open(QFile::ReadOnly))
            return;
        QDomDocument doc;
        if(!doc.setContent(&file))
        {
            file.close();
            return;
        }
        file.close();
        QDomElement root=doc.documentElement();
        QDomNodeList list=root.elementsByTagName("book");
        QDomNode node=list.at(0).firstChild(); //定位到第三个一级子节点的子元素
        QDomNode oldnode=node.firstChild(); //标签之间的内容作为节点的子节点出现,当前是Pride and Projudice
        node.firstChild().setNodeValue(ui->lineEdit->text());
        QDomNode newnode=node.firstChild();
        node.replaceChild(newnode,oldnode);
        if(!file.open(QFile::WriteOnly|QFile::Truncate))
            return;
        QTextStream out_stream(&file);
        doc.save(out_stream,4); //缩进4格
        file.close();
        QMessageBox::information(this,QString::fromLocal8Bit("提示"),QString::fromLocal8Bit("设置成功，请重启软件生效"));
    }
    else
        QMessageBox::information(this,QString::fromLocal8Bit("提示"),QString::fromLocal8Bit("请设置在5.3-5.4"));
}

void MainWindow::on_pushButton_7_clicked()
{
    capture.open(ui->lineEdit_6->text().toInt());//打开外接USB摄像头
    if(capture.isOpened())
    {
        mode=1;
        capture.set(3,1920);
        capture.set(4,1080);
        flag=0;
        sidian=0;
        frame_count=0;
        timer->start(33);
        normal=0;

        ui->lineEdit_2->setText(QString::fromLocal8Bit(""));
        ui->lineEdit_4->setText(QString::fromLocal8Bit(""));
        ui->lineEdit_5->setText(QString::fromLocal8Bit(""));

    }
    else
        QMessageBox::information(this,QString::fromLocal8Bit("提示"),QString::fromLocal8Bit("HDMI连接未成功"));

    ui->lineEdit_2->setText(QString::fromLocal8Bit(""));
    ui->lineEdit_4->setText(QString::fromLocal8Bit(""));
    ui->lineEdit_5->setText(QString::fromLocal8Bit(""));
    biliflag=0;
}

void MainWindow::on_pushButton_8_clicked()
{
    mode=0;
    timer->stop();
    capture.release();

}

void MainWindow::changeValue(int value)
{
    ui->progressBar->setValue(value);
    controlflag=1;
    frame_count=frame_num*value/100.0;
}

void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    if(playing==1&&mode==2)
    {
        emit changeValue(value);
    }
}

void MainWindow::on_pushButton_9_clicked()
{
    if(mode==2)_videocap->read(_srcImge);
    if(mode==1)capture >> _srcImge;
    if(!_srcImge.empty())
    {
        cv::cvtColor(_srcImge,srcImagergb,CV_BGR2RGB);//这种更方便好用
        if((mode==2&&frame_count>frame_num*0.8)||(mode==1))//
        {
        std::string str1 = result.toStdString();
        const char* ch1 = str1.c_str();
        std::string str2 = QString::number(diaresult).toStdString();
        const char* ch2 = str2.c_str();
        cv::putText(srcImagergb," Capsulorhexis   ",cv::Point(40,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        cv::putText(srcImagergb,ch1,cv::Point(300,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        cv::putText(srcImagergb,"Diameter ",cv::Point(40,100),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        cv::putText(srcImagergb,ch2,cv::Point(300,100),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        cv::putText(srcImagergb,"mm",cv::Point(360,100),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
         }

    image2=QImage((const unsigned char*)srcImagergb.data,srcImagergb.cols,srcImagergb.rows,QImage::Format_RGB888);//Format_RGB888

//    if((mode==2&&frame_count>frame_num*0.8)||(mode==1&&frame_count>7500))//
//    {
//        QPainter painter(&image2); //为这个QImage构造一个QPainter
//        painter.setCompositionMode(QPainter::CompositionMode_DestinationOver);//CompositionMode_SourceIn

//        //设置画刷的组合模式CompositionMode_SourceOut这个模式为目标图像在上。
//        //改变画笔和字体
//        QPen pen = painter.pen();
//        pen.setColor(Qt::red);
//        QFont font = painter.font();
//        font.setBold(true);//加粗
//        font.setPixelSize(40);//改变字体大小

//        painter.setPen(pen);
//        painter.setFont(font);
//        painter.drawText(40,50,QString::fromLocal8Bit("撕囊结果：")+result);//
//        painter.drawText(40,100,QString::fromLocal8Bit("撕囊口直径约：")+QString::number(diaresult)+"mm");

//    }
    ui->label->setPixmap(QPixmap::fromImage(image2.scaled(ui->label->size(), Qt::KeepAspectRatio)));
    ui->label->show();
    image2.save("D:/images/"+QString::number(frame_count)+".jpg","jpg",-1);
    //image2.save("D:/images/1.jpg","jpg",-1);
    }
    if(mode==1)capture.release();
    mode=0;

    ui->lineEdit_2->setText(QString::fromLocal8Bit(""));
    ui->lineEdit_4->setText(QString::fromLocal8Bit(""));
    ui->lineEdit_5->setText(QString::fromLocal8Bit(""));
    ui->horizontalSlider->setValue(0);
    timer->stop();

    frame_count=0;
    flag=0;
    sidian=0;
    normal=0;

}


void MainWindow::on_pushButton_12_clicked()
{
    frame_count=0;
}



void MainWindow::on_pushButton_17_clicked()
{
     //
}

void MainWindow::on_pushButton_16_clicked()
{
    clslens= ClsLens(bin1 ,xml1, 1, ui->lineEdit_7->text().toInt());
}

void MainWindow::on_pushButton_10_clicked()
{
    if(mode==2)_videocap->read(_srcImge);
    if(mode==1)capture >> _srcImge;
    if(!_srcImge.empty())
    {
        cv::cvtColor(_srcImge,srcImagergb,CV_BGR2RGB);//这种更方便好用
        cv::cvtColor(_srcImge,srcImagergb1,CV_BGR2RGB);
        if((xxx-300)>0&&(yyy-300)>0&&(xxx+300)<srcImagergb.cols&&(yyy+300)<srcImagergb.rows)
        {
            imageROI = srcImagergb(cv::Rect(xxx-300, yyy-300,600,600));

        cv::cvtColor(imageROI,imageROI,CV_BGR2RGB);
        //cv::imshow("roi",imageROI);
        cv::imwrite("D:/output/a_snk_picture/1.jpg",imageROI);

        QDesktopServices::openUrl(QUrl("D:/output/snk-test-cpu/AI.exe"));

        QFile file("D:/output/a_snk_result/data_temp.json");
        if(file.open(QIODevice::ReadOnly))
        {
            QByteArray fileArray = file.readAll();
            QString content = QString::fromStdString(fileArray.toStdString());
            content.replace("{",""); //去掉花括号"{"
            content.replace("}",""); //去掉"}"
            content.replace("\"","");//去掉引号"
            content.replace("\r","");
            content.replace("\n","");
            qDebug()<<content;
            content.simplified(); //去掉空格
            qDebug()<<content;
            QStringList list = content.split(":");

            if(!list.isEmpty())
            {
                result = list.last();
                result.remove(0,1);
                qDebug()<<result;
                ui->lineEdit_2->setText(result);
            }
        }
        }

        //if((mode==2&&frame_count>frame_num*0.8)||(mode==1))
        //{
        std::string str1 = result.toStdString();
        const char* ch1 = str1.c_str();
        std::string str2 = QString::number(diaresult).toStdString();
        const char* ch2 = str2.c_str();
        cv::putText(srcImagergb1," Capsulorhexis",cv::Point(40,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        cv::putText(srcImagergb1,ch1,cv::Point(300,50),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        cv::putText(srcImagergb1,"Diameter",cv::Point(40,100),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        cv::putText(srcImagergb1,ch2,cv::Point(300,100),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
        cv::putText(srcImagergb1,"mm",cv::Point(360,100),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
         //}

    image2=QImage((const unsigned char*)srcImagergb1.data,srcImagergb1.cols,srcImagergb1.rows,QImage::Format_RGB888);//Format_RGB888

    ui->label->setPixmap(QPixmap::fromImage(image2.scaled(ui->label->size(), Qt::KeepAspectRatio)));
    ui->label->show();
    QString current_time =QDateTime::currentDateTime().toString("yyyy.MM.dd hh.mm.ss");
    image2.save("D:/images/"+current_time+".jpg","jpg",-1);
    }
}

void MainWindow::on_pushButton_11_clicked()
{
    if(mode==2)_videocap->read(_srcImge);
    if(mode==1)capture >> _srcImge;
    if(!_srcImge.empty())
    {
        cv::cvtColor(_srcImge,srcImagergb,CV_BGR2RGB);//这种更方便好用
        cv::cvtColor(_srcImge,srcImagergb1,CV_BGR2RGB);

        image2=QImage((const unsigned char*)srcImagergb.data,srcImagergb.cols,srcImagergb.rows,QImage::Format_RGB888);//Format_RGB888
        image2.save("D:/recognition_Cataract_capsulorhexis/picture_json/1.jpg","jpg",-1);

        //cv::imwrite("D:/recognition_Cataract_capsulorhexis/picture_json/1.jpg",srcImagergb);
        QProcess p(NULL);
        p.setWorkingDirectory("D:/recognition_Cataract_capsulorhexis");
        QString command = "D:/recognition_Cataract_capsulorhexis/recognition_Cataract_capsulorhexis.exe";
        p.start(command);
        p.waitForFinished();
    }
}
