#-------------------------------------------------
#
# Project created by QtCreator 2020-03-31T09:21:43
#
#-------------------------------------------------
QT+=xml
QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MembraneAI
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    openvinodetection.h \
    ClsDetection.h \
    OpenvinoIE.h

FORMS    += mainwindow.ui

#INCLUDEPATH+=D:/opencv/build/include
#INCLUDEPATH+=D:/opencv/build/include/opencv2
#LIBS +=-LD:/opencv/build/x64/vc14/lib \
#-lopencv_world420

## opencv
INCLUDEPATH += $$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/include)
INCLUDEPATH += $$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/include/opencv2)

LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_calib3d440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_core440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_dnn440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_features2d440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_flann440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_gapi440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_highgui440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_imgcodecs440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_imgproc440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_ml440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_objdetect440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_photo440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_stitching440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_video440
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/opencv/lib) -lopencv_videoio440

## openvino
INCLUDEPATH += $$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/bin/intel64/Release)
INCLUDEPATH += $$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/include)

LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/lib/intel64/Release) -linference_engine
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/lib/intel64/Release) -linference_engine_c_api
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/lib/intel64/Release) -linference_engine_ir_reader
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/lib/intel64/Release) -linference_engine_legacy
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/lib/intel64/Release) -linference_engine_lp_transformations
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/lib/intel64/Release) -linference_engine_onnx_reader
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/lib/intel64/Release) -linference_engine_preproc
LIBS += -L$$quote(C:/Program Files (x86)/IntelSWTools/openvino/deployment_tools/inference_engine/lib/intel64/Release) -linference_engine_transformations



