#-------------------------------------------------
#
# Project created by QtCreator 2018-01-31T11:07:59
#
#-------------------------------------------------
include(fvd.pri)
QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = demo
TEMPLATE = app


SOURCES += main.cpp\
    alg/cascadedetect.cpp \
    alg/m_arith.cpp \
    alg/VC_DSP_TRANSFORM.cpp \
        mainwindow.cpp videosrc.cpp    videowidget.cpp tool1.cpp videoprocessor.cpp

HEADERS  += mainwindow.h\
    alg/cascade_day_xml.h \
    alg/cascade_dusk_xml.h \
    alg/cascade_night_xml.h \
    alg/cascade_xml.h \
    alg/cascadedetect.h \
    alg/DSPARMProto.h \
    alg/m_arith.h \
   videosrc.h  videowidget.h tool1.h videoprocessor.h


FORMS    += mainwindow.ui
CONFIG +=c++11
unix:DEFINES+=IS_UNIX
