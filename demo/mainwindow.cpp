#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "videowidget.h"
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
//    VideoWidget *w=new VideoWidget(this);
    //his->setCentralWidget(w);
    v=new VideoThread("rtsp://192.168.1.216:8554/test1",ui->widget);
   //   v=new VideoThread("test.mp4",ui->widget);
    v->start();
}

MainWindow::~MainWindow()
{
    delete ui;
}
