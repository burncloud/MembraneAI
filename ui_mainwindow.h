/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.6.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QTabWidget *tabWidget;
    QWidget *tab;
    QLabel *label_2;
    QWidget *tab_2;
    QLabel *label;
    QTabWidget *tabWidget_2;
    QWidget *tab_3;
    QGroupBox *groupBox;
    QPushButton *pushButton_2;
    QPushButton *pushButton_3;
    QPushButton *pushButton_4;
    QPushButton *pushButton_5;
    QProgressBar *progressBar;
    QLabel *label_5;
    QLabel *label_7;
    QSlider *horizontalSlider;
    QGroupBox *groupBox_4;
    QPushButton *pushButton_7;
    QPushButton *pushButton_8;
    QPushButton *pushButton_12;
    QLineEdit *lineEdit_6;
    QGroupBox *groupBox_5;
    QLabel *label_8;
    QLineEdit *lineEdit_4;
    QPushButton *pushButton_9;
    QPushButton *pushButton_11;
    QWidget *tab_4;
    QGroupBox *groupBox_2;
    QPushButton *pushButton_6;
    QPushButton *pushButton;
    QLabel *label_6;
    QLineEdit *lineEdit_3;
    QGroupBox *groupBox_3;
    QLabel *label_3;
    QLineEdit *lineEdit;
    QPushButton *pushButton_15;
    QLabel *label_10;
    QLineEdit *lineEdit_7;
    QGroupBox *groupBox_6;
    QPushButton *pushButton_17;
    QPushButton *pushButton_16;
    QLabel *label_9;
    QLineEdit *lineEdit_5;
    QLabel *label_4;
    QLineEdit *lineEdit_2;
    QPushButton *pushButton_10;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1800, 958);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(10, 0, 1531, 901));
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        label_2 = new QLabel(tab);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(10, 10, 1501, 851));
        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        label = new QLabel(tab_2);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(10, 10, 1501, 851));
        tabWidget->addTab(tab_2, QString());
        tabWidget_2 = new QTabWidget(centralWidget);
        tabWidget_2->setObjectName(QStringLiteral("tabWidget_2"));
        tabWidget_2->setGeometry(QRect(1560, 0, 221, 901));
        tab_3 = new QWidget();
        tab_3->setObjectName(QStringLiteral("tab_3"));
        groupBox = new QGroupBox(tab_3);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 280, 201, 151));
        pushButton_2 = new QPushButton(groupBox);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));
        pushButton_2->setGeometry(QRect(14, 20, 81, 23));
        pushButton_3 = new QPushButton(groupBox);
        pushButton_3->setObjectName(QStringLiteral("pushButton_3"));
        pushButton_3->setGeometry(QRect(20, 50, 75, 23));
        pushButton_4 = new QPushButton(groupBox);
        pushButton_4->setObjectName(QStringLiteral("pushButton_4"));
        pushButton_4->setGeometry(QRect(110, 50, 75, 23));
        pushButton_5 = new QPushButton(groupBox);
        pushButton_5->setObjectName(QStringLiteral("pushButton_5"));
        pushButton_5->setGeometry(QRect(110, 20, 75, 23));
        progressBar = new QProgressBar(groupBox);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setGeometry(QRect(80, 90, 121, 20));
        progressBar->setValue(24);
        label_5 = new QLabel(groupBox);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(0, 90, 71, 16));
        label_7 = new QLabel(groupBox);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(0, 120, 71, 16));
        horizontalSlider = new QSlider(groupBox);
        horizontalSlider->setObjectName(QStringLiteral("horizontalSlider"));
        horizontalSlider->setGeometry(QRect(80, 120, 111, 22));
        horizontalSlider->setMaximum(100);
        horizontalSlider->setOrientation(Qt::Horizontal);
        groupBox_4 = new QGroupBox(tab_3);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        groupBox_4->setGeometry(QRect(10, 150, 201, 81));
        pushButton_7 = new QPushButton(groupBox_4);
        pushButton_7->setObjectName(QStringLiteral("pushButton_7"));
        pushButton_7->setGeometry(QRect(20, 20, 75, 23));
        pushButton_8 = new QPushButton(groupBox_4);
        pushButton_8->setObjectName(QStringLiteral("pushButton_8"));
        pushButton_8->setGeometry(QRect(110, 20, 75, 23));
        pushButton_12 = new QPushButton(groupBox_4);
        pushButton_12->setObjectName(QStringLiteral("pushButton_12"));
        pushButton_12->setGeometry(QRect(110, 50, 71, 23));
        lineEdit_6 = new QLineEdit(groupBox_4);
        lineEdit_6->setObjectName(QStringLiteral("lineEdit_6"));
        lineEdit_6->setGeometry(QRect(22, 50, 71, 20));
        groupBox_5 = new QGroupBox(tab_3);
        groupBox_5->setObjectName(QStringLiteral("groupBox_5"));
        groupBox_5->setGeometry(QRect(10, 40, 201, 71));
        label_8 = new QLabel(groupBox_5);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setGeometry(QRect(10, 30, 131, 16));
        lineEdit_4 = new QLineEdit(groupBox_5);
        lineEdit_4->setObjectName(QStringLiteral("lineEdit_4"));
        lineEdit_4->setGeometry(QRect(140, 30, 51, 20));
        lineEdit_4->setReadOnly(true);
        pushButton_9 = new QPushButton(tab_3);
        pushButton_9->setObjectName(QStringLiteral("pushButton_9"));
        pushButton_9->setGeometry(QRect(14, 678, 191, 41));
        pushButton_11 = new QPushButton(tab_3);
        pushButton_11->setObjectName(QStringLiteral("pushButton_11"));
        pushButton_11->setGeometry(QRect(10, 600, 201, 41));
        tabWidget_2->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QStringLiteral("tab_4"));
        groupBox_2 = new QGroupBox(tab_4);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 20, 201, 111));
        pushButton_6 = new QPushButton(groupBox_2);
        pushButton_6->setObjectName(QStringLiteral("pushButton_6"));
        pushButton_6->setGeometry(QRect(110, 20, 75, 23));
        pushButton = new QPushButton(groupBox_2);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(20, 20, 75, 23));
        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(10, 60, 151, 16));
        lineEdit_3 = new QLineEdit(groupBox_2);
        lineEdit_3->setObjectName(QStringLiteral("lineEdit_3"));
        lineEdit_3->setGeometry(QRect(20, 80, 91, 20));
        lineEdit_3->setReadOnly(true);
        groupBox_3 = new QGroupBox(tab_4);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        groupBox_3->setGeometry(QRect(10, 130, 201, 181));
        label_3 = new QLabel(groupBox_3);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(10, 20, 111, 16));
        lineEdit = new QLineEdit(groupBox_3);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        lineEdit->setGeometry(QRect(120, 20, 71, 20));
        pushButton_15 = new QPushButton(groupBox_3);
        pushButton_15->setObjectName(QStringLiteral("pushButton_15"));
        pushButton_15->setGeometry(QRect(120, 50, 75, 23));
        label_10 = new QLabel(groupBox_3);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setGeometry(QRect(10, 90, 61, 16));
        lineEdit_7 = new QLineEdit(groupBox_3);
        lineEdit_7->setObjectName(QStringLiteral("lineEdit_7"));
        lineEdit_7->setGeometry(QRect(70, 90, 121, 20));
        lineEdit_7->setReadOnly(false);
        groupBox_6 = new QGroupBox(groupBox_3);
        groupBox_6->setObjectName(QStringLiteral("groupBox_6"));
        groupBox_6->setGeometry(QRect(-230, -10, 201, 181));
        pushButton_17 = new QPushButton(groupBox_6);
        pushButton_17->setObjectName(QStringLiteral("pushButton_17"));
        pushButton_17->setGeometry(QRect(120, 130, 75, 23));
        pushButton_16 = new QPushButton(groupBox_3);
        pushButton_16->setObjectName(QStringLiteral("pushButton_16"));
        pushButton_16->setGeometry(QRect(120, 120, 75, 23));
        label_9 = new QLabel(tab_4);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setGeometry(QRect(20, 340, 181, 16));
        lineEdit_5 = new QLineEdit(tab_4);
        lineEdit_5->setObjectName(QStringLiteral("lineEdit_5"));
        lineEdit_5->setGeometry(QRect(20, 360, 181, 20));
        lineEdit_5->setReadOnly(true);
        label_4 = new QLabel(tab_4);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(20, 400, 201, 16));
        lineEdit_2 = new QLineEdit(tab_4);
        lineEdit_2->setObjectName(QStringLiteral("lineEdit_2"));
        lineEdit_2->setGeometry(QRect(20, 420, 181, 20));
        lineEdit_2->setReadOnly(true);
        pushButton_10 = new QPushButton(tab_4);
        pushButton_10->setObjectName(QStringLiteral("pushButton_10"));
        pushButton_10->setGeometry(QRect(20, 470, 75, 23));
        tabWidget_2->addTab(tab_4, QString());
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1800, 23));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);
        tabWidget_2->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Real-time guidance of capsulorhexis system", 0));
        label_2->setText(QApplication::translate("MainWindow", "TextLabel", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "Vidio", 0));
        label->setText(QApplication::translate("MainWindow", "TextLabel", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "Image", 0));
        groupBox->setTitle(QApplication::translate("MainWindow", "Offline video analysis", 0));
        pushButton_2->setText(QApplication::translate("MainWindow", "Upload video", 0));
        pushButton_3->setText(QApplication::translate("MainWindow", "Pause", 0));
        pushButton_4->setText(QApplication::translate("MainWindow", " Play", 0));
        pushButton_5->setText(QApplication::translate("MainWindow", "Screenshot", 0));
        label_5->setText(QApplication::translate("MainWindow", "Progress control\357\274\232", 0));
        label_7->setText(QApplication::translate("MainWindow", "Play control\357\274\232", 0));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "Real-time video operation", 0));
        pushButton_7->setText(QApplication::translate("MainWindow", "Open HDMI", 0));
        pushButton_8->setText(QApplication::translate("MainWindow", "Close HDMI", 0));
        pushButton_12->setText(QApplication::translate("MainWindow", "Clear", 0));
        groupBox_5->setTitle(QApplication::translate("MainWindow", "Video analysis", 0));
        label_8->setText(QApplication::translate("MainWindow", "1\343\200\201Pixel size (\316\274m):", 0));
        pushButton_9->setText(QApplication::translate("MainWindow", "End", 0));
        pushButton_11->setText(QApplication::translate("MainWindow", "recognition", 0));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_3), QApplication::translate("MainWindow", "Surgery video", 0));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "Image analysis", 0));
        pushButton_6->setText(QApplication::translate("MainWindow", "location", 0));
        pushButton->setText(QApplication::translate("MainWindow", "Load Image", 0));
        label_6->setText(QApplication::translate("MainWindow", "Single frame time(ms):", 0));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "Parameter setting", 0));
        label_3->setText(QApplication::translate("MainWindow", "Ruler length\357\274\210mm\357\274\211\357\274\232", 0));
        pushButton_15->setText(QApplication::translate("MainWindow", "save", 0));
        label_10->setText(QApplication::translate("MainWindow", "rate\357\274\232", 0));
        groupBox_6->setTitle(QApplication::translate("MainWindow", " \345\217\202\346\225\260\350\256\276\347\275\256", 0));
        pushButton_17->setText(QApplication::translate("MainWindow", " \347\241\256\345\256\232", 0));
        pushButton_16->setText(QApplication::translate("MainWindow", "save", 0));
        label_9->setText(QApplication::translate("MainWindow", "2\343\200\201IOL diameter (mm)\357\274\232", 0));
        label_4->setText(QApplication::translate("MainWindow", "3\343\200\201Evaluation of capsulorhexis\357\274\232", 0));
        pushButton_10->setText(QApplication::translate("MainWindow", "Evaluate", 0));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_4), QString());
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
