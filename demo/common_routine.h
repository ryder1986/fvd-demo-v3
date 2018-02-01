#ifndef COMMON_ROUTINE_H
#define COMMON_ROUTINE_H
//#include <QtCore>
#include <stdio.h>
#include <stdlib.h>

class common_routine
{
public:
    common_routine();
    static unsigned char  CONVERT_ADJUST(double tmp);
    //YUV420P to RGB24
    static  void CONVERT_YUV420PtoRGB24(unsigned char* yuv_src,unsigned char* rgb_dst,int nWidth,int nHeight);
    static void read_yuv_file(char  *yuv_buf1,int video_width,int video_height,int frame_no,char *filename=NULL);
};

#endif // COMMON_ROUTINE_H
