﻿#include "mainwindow.h"
#include <QApplication>




int main(int argc, char *argv[])
{
    //QTextCodec::setCodecForTr(QTextCodec::codecForName("gbk"));
    //QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
