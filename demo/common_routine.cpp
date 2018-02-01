#include "common_routine.h"
#include <stdio.h>
#include <malloc.h>
#include <string.h>
//#include <QtCore>
common_routine::common_routine()
{
}

//void common_routine::read_yuv_file(char *yuv_buf1,int video_width,int video_height,int frame_no,char *filename)
//{
//    // QFile file("D://1.yuv");
//     QFile file("./200frame.yuv");
////    if(filename==NULL)
////         QFile file("D://1.yuv");
////    else
////    {
////         QFile file(filename);
////    }
//    if (!file.open(QIODevice::ReadOnly ))
//        printf(" open err\n");
//    memset(yuv_buf1,0,video_width*video_height*3/2);
//    file.seek(frame_no*video_width*video_height*3/2);
//    file.read(yuv_buf1,video_width*video_height*3/2);
//    file.close();
//}
void common_routine::read_yuv_file(char *yuv_buf1,int video_width,int video_height,int frame_no,char *filename)
{
    // QFile file("D://1.yuv");
  //   QFile file("./200frame.yuv");
//    if(filename==NULL)
//         QFile file("D://1.yuv");
//    else
//    {
//         QFile file(filename);
//    }

    FILE *fp;

     fp=fopen(filename,"rb" );
     if(fp==NULL)
        {printf(" open err\n");
     return ;
     }
    memset(yuv_buf1,0,video_width*video_height*3/2);
    fseek(fp,frame_no*video_width*video_height*3/2,SEEK_SET);
    fread(yuv_buf1,video_width*video_height*3/2,1,fp);
    fclose(fp);
}
unsigned char common_routine::CONVERT_ADJUST(double tmp)
{
    return (unsigned char)((tmp >= 0 && tmp <= 255)?tmp:(tmp < 0 ? 0 : 255));
}
//YUV420P to RGB24
 void common_routine::CONVERT_YUV420PtoRGB24(unsigned char* yuv_src,unsigned char* rgb_dst,int nWidth,int nHeight)
{
    unsigned char *tmpbuf=(unsigned char *)malloc(nWidth*nHeight*3);
    unsigned char Y,U,V,R,G,B;
    unsigned char* y_planar,*u_planar,*v_planar;
    int rgb_width , u_width;
    rgb_width = nWidth * 3;
    u_width = (nWidth >> 1);
    int ypSize = nWidth * nHeight;
    int upSize = (ypSize>>2);
    int offSet = 0;

    y_planar = yuv_src;
    u_planar = yuv_src + ypSize;
    v_planar = u_planar + upSize;

    for(int i = 0; i < nHeight; i++)
    {
        for(int j = 0; j < nWidth; j ++)
        {
            // Get the Y value from the y planar
            Y = *(y_planar + nWidth * i + j);
            // Get the V value from the u planar
            offSet = (i>>1) * (u_width) + (j>>1);
            V = *(u_planar + offSet);
            // Get the U value from the v planar
            U = *(v_planar + offSet);

            // Cacular the R,G,B values
            // Method 1
            R = CONVERT_ADJUST((Y + (1.4075 * (V - 128))));
            G = CONVERT_ADJUST((Y - (0.3455 * (U - 128) - 0.7169 * (V - 128))));
            B = CONVERT_ADJUST((Y + (1.7790 * (U - 128))));
            /*
              // The following formulas are from MicroSoft' MSDN
              int C,D,E;
              // Method 2
              C = Y - 16;
              D = U - 128;
              E = V - 128;
              R = CONVERT_ADJUST(( 298 * C + 409 * E + 128) >> 8);
              G = CONVERT_ADJUST(( 298 * C - 100 * D - 208 * E + 128) >> 8);
              B = CONVERT_ADJUST(( 298 * C + 516 * D + 128) >> 8);
              R = ((R - 128) * .6 + 128 )>255?255:(R - 128) * .6 + 128;
              G = ((G - 128) * .6 + 128 )>255?255:(G - 128) * .6 + 128;
              B = ((B - 128) * .6 + 128 )>255?255:(B - 128) * .6 + 128;
              */
            offSet = rgb_width * i + j * 3;

            rgb_dst[offSet] = B;
            rgb_dst[offSet + 1] = G;
            rgb_dst[offSet + 2] = R;
        }
    }
    free(tmpbuf);
}
