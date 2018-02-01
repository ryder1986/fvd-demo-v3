#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include "videowidget.h"
#include "videosrc.h"
#include "tool1.h"
#include "videoprocessor.h"
#include "common_routine.h"
class VideoThread:public QThread{
public:
    VideoThread(QString src,VideoWidget *w):vs(src.toStdString().data()),wgt(w)
    {
        wgt->set_points(p.pt,p.pt_rear);
        w->setFixedWidth(640);
        w->setFixedHeight(480);

    }
    void run()
    {
        int tick=0;
        Mat m;
        Mat gray_frame;
        while(1){

            bool ret=vs.fetch_frame(m);
            if(ret&&m.rows>0){

         //       wgt->update_mat(m);

                 //   common_routine::read_yuv_file(yuv_buf,640,480,tick++%200,"./200frame.yuv");
                        wgt->update_mat(m);
//                    Mat m1(640,480,CV_8UC1,yuv_buf);
//                          wgt->update_mat(m1);
                 //   p.handle_frame(m.data);
                     cvtColor(m, gray_frame, CV_BGR2GRAY);
                    p.handle_frame((unsigned char *)gray_frame.data);
                     //wgt->update();
            }else{
                //prt(info,"2");
               // QThread::msleep(100);
            }
            // QThread::sleep(1);
            //   QThread::msleep(40);

        }
    }
private:
    VideoSrc vs;
    VideoWidget *wgt;
    VideoProcessor p;
     char yuv_buf[640*480*3/2];
};

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    VideoThread *v;

};

#endif // MAINWINDOW_H
