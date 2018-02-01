//#include "stdafx.h"
#include "stdio.h"
#include "stdlib.h" /// div_t I??????
#include <string.h>
#include "cascadedetect.h"
#include "m_arith.h"
#include "math.h"
#include "DSPARMProto.h"

//////////////////////////////////////////////////////////////////////////////////////

#define	MAX_SPEEDDETECTOR_DOTS	768*576
#define MaxDotsInDetect 768*576

#define SPEED_STARTER_STARTLINE		1
#define SPEED_STARTER_ENDLINE		40//10
#define SPEED_COUNTER_ENDLINE		1
#define SPEED_COUNTER_STARTLINE		42//12

//#define FULL_COLS  					(720)
//#define FULL_ROWS  					(576)
////#define COLS  					(200)
////#define ROWS  					(100)
//
////?
//#define MAX_CALIBRATION_POINTS 100   //?
//int calibration_points_num; //???
////?
//float mapping_matrix[9]={0};
////????,???
//float image_actual[FULL_ROWS][FULL_COLS][2]={0};
//float actual_point[MAX_CALIBRATION_POINTS][2]={0,0,-7.5,0,0,15,-7.5,15,0,30,-7.5,30,0,45,-7.5,45,0,60,-7.5,60};
//float image_point[MAX_CALIBRATION_POINTS][2]={377,518,147,529,304,393,141,398,263,322,137,328,239,281,135,284,221,250,133,255};
////?

#define		THIS_LANE_COUNTER	0
#define		LEFT_LANE_COUNTER	1
#define		RIGHT_LANE_COUNTER	2
#define		VEHICLE_MIN_ROW		16//6
//










int alg_mem_malloc(m_args *p)
{

	int ret = -1;
	int size;
	//杈撳嚭鍐呭瓨鍒嗛厤



//	ALGPARAMS *Params;
	p->p_outbuf = NULL;
	p->p_outbuf = (OUTBUF*) malloc(sizeof(OUTBUF));
	if (p->p_outbuf == NULL) {
		printf("alg malloc err\n");
	}
	memset(p->p_outbuf, 0, sizeof(OUTBUF));

	//鍙傛暟閰嶇疆鍐呭瓨鍒嗛厤
	p->pParams = NULL;
	p->pParams = (ALGPARAMS*) malloc(sizeof(ALGPARAMS));
	if (p->pParams == NULL) {
		printf("alg malloc err\n");
	}
	memset(p->pParams, 0, sizeof(ALGPARAMS));
	p->pParams->puPointNewImage = NULL;

	p->pParams->puPointNewImage = (Uint8*) malloc(
			DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 15 * sizeof(Uint8));
	if (p->pParams->puPointNewImage == NULL) {
		printf("alg malloc err\n");
	}

	memset(p->pParams->puPointNewImage, 0,
			DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 15 * sizeof(Uint8));

	memset(p->pParams->puPointNewImage, 0, 1000);

	size = 800 * 800;
	p->pParams->m_pPointArray = NULL;
	p->pParams->m_pPointArray = (CPoint *) malloc(size * sizeof(CPoint));
	if (p->pParams->m_pPointArray == NULL) {
		printf("alg malloc err\n");
	}

	memset(p->pParams->m_pPointArray, 0, size * sizeof(CPoint));
	p->pParams->ForegroundArray = NULL;
	p->pParams->ForegroundArray = (CForeground_Array*) malloc(
			sizeof(CForeground_Array));
	if (p->pParams->ForegroundArray == NULL) {
		printf("alg malloc err\n");
	}

	p->pParams->ForegroundArray1 = NULL;
	p->pParams->ForegroundArray1 = (CForeground_Array*) malloc(
			sizeof(CForeground_Array));
	if (p->pParams->ForegroundArray1 == NULL) {
		printf("alg malloc err\n");
	}

	memset(p->pParams->ForegroundArray1, 0, sizeof(CForeground_Array));

	//閰嶇疆鍐呭瓨鍒嗛厤
	p->pCfgs = NULL;
	p->pCfgs = (ALGCFGS*) malloc(sizeof(ALGCFGS));
	if (p->pCfgs == NULL) {
		printf("alg malloc err\n");
	}else{
		printf("alg malloc ok \n");
	}

	memset(p->pCfgs, 0, sizeof(ALGCFGS));
	p->pCfgs->CameraLocalPara.CameraCfgEntry = NULL;
	size = DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * sizeof(Uint32);
	size = sizeof(CAMERA_STRUCT) + sizeof(LANE_STRUCT_DEF) * MAX_LANE
			+ MAX_LANE * sizeof(SPEED_DETECT_STRUCT) + size;
	p->pCfgs->CameraLocalPara.CameraCfgEntry = (Uint8*) malloc(size);
	if (p->pCfgs->CameraLocalPara.CameraCfgEntry == NULL) {
		printf("alg malloc err\n");
	}

	memset(p->pCfgs->CameraLocalPara.CameraCfgEntry, 0, size);
	p->pCfgs->CameraLocalPara.ImageStorageEntry = NULL;
	size = DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 6;
	p->pCfgs->CameraLocalPara.ImageStorageEntry = (Uint8*) malloc(size);
	if (p->pCfgs->CameraLocalPara.ImageStorageEntry == NULL) {
		printf("alg malloc err\n");
	}

	memset(p->pCfgs->CameraLocalPara.ImageStorageEntry, 0, size);

	return 0;
}

int alg_mem_free(m_args *arg_arg)
{
	if (arg_arg->pCfgs) {
		if (arg_arg->pCfgs->CameraLocalPara.ImageStorageEntry) {
			free(arg_arg->pCfgs->CameraLocalPara.ImageStorageEntry);
			arg_arg->pCfgs->CameraLocalPara.ImageStorageEntry = NULL;
		}

		if (arg_arg->pCfgs->CameraLocalPara.CameraCfgEntry) {
			free(arg_arg->pCfgs->CameraLocalPara.CameraCfgEntry);
			arg_arg->pCfgs->CameraLocalPara.CameraCfgEntry = NULL;
		}

		free(arg_arg->pCfgs);
		arg_arg->pCfgs = NULL;
	}
	//閿熼叺鏀捐鎷烽敓鏂ゆ嫹
	if ( arg_arg->pParams) {
		if (arg_arg->pParams->ForegroundArray) {
			free(arg_arg->pParams->ForegroundArray);
			arg_arg->pParams->ForegroundArray = NULL;
		}

		if (arg_arg->pParams->ForegroundArray1) {
			free(arg_arg->pParams->ForegroundArray1);
			arg_arg->pParams->ForegroundArray1 = NULL;
		}
		if (arg_arg->pParams->m_pPointArray) {
			free(arg_arg->pParams->m_pPointArray);
			arg_arg->pParams->m_pPointArray = NULL;
		}
		if (arg_arg->pParams->puPointNewImage) {
			free(arg_arg->pParams->puPointNewImage);
			arg_arg->pParams->puPointNewImage = NULL;
		}
		free(arg_arg->pParams);
		arg_arg->pParams = NULL;
	}
	if (arg_arg->p_outbuf) {
		free(arg_arg->p_outbuf);
		arg_arg->p_outbuf = NULL;
	}

	return 0;
}







int day_flag=0;
int night_flag=0;
int AverageGray_num=0;
float AverageGray_sum=0;
float former_tAverageGray=0;
int Fchange_speed_tAverageGray_num=0;
//float percent=0;
bool IsQueue=0;
/************************************************************************/
/* switch_day_night_flag                                                                     */
/************************************************************************/
//	globals
//
#define max(a,b) (((a)>(b)) ? (a):(b))
#define min(a,b) (((a)>(b)) ? (b):(a))
void data_convertion_mat(unsigned char *imgBuf,int width,int height,int nchannel,Mat& image,int scale)//????
{
	if(!imgBuf)
		return;
	height=height/scale;
	width=width/scale;
	if(image.empty())
	{
		if(nchannel==1)
		{
			image.create(height, width, CV_8U);
		}
		else
		{
			image.create(height,width,CV_8UC3);
		}
	}

	if(image.rows!=height||image.cols!=width)
	{

		resize(image,image,Size(width,height));
	}
	unsigned char* p=image.data;
	for (int k=0;k<height;k++)
	{
		//unsigned char* p=image.ptr<unsigned char>(k);
		for(int l=0;l<width;l++)
		{
			if(nchannel==1)
			{
			//	 //printf("before cppy %d  (%d  %d)\n",m,width,height);
				//p[l]=*(imgBuf+k*scale*scale*width+l*scale);
			    *p=*(imgBuf+k*scale*scale*width+l*scale);

			  //  //printf("after cppy %d\n",m);
				p++;
			}
			//???????
			else
			{
				/*p[3*l]=*(imgBuf+(height-1-k)*3*width*scale*scale+3*l*scale+0);
				p[3*l+1]=*(imgBuf+(height-1-k)*3*width*scale*scale+3*l*scale+1);
				p[3*l+2]=*(imgBuf+(height-1-k)*3*width*scale*scale+3*l*scale+2);*/
				p[0]=*(imgBuf+(height-1-k)*3*width*scale*scale+3*l*scale+0);
				p[1]=*(imgBuf+(height-1-k)*3*width*scale*scale+3*l*scale+1);
				p[2]=*(imgBuf+(height-1-k)*3*width*scale*scale+3*l*scale+2);
				p+=3;
			}

		}
	}
}

void RECT_Init(CRect handle)
{
}
void RECT_Free(CRect handle)
{
}
int RECT_Width(CRect handle)
{
	return abs(handle.right - handle.left + 1);
}
int RECT_Height(CRect handle)
{
	return abs(handle.bottom - handle.top + 1);
}
bool ArithInit(Uint16 ChNum, CFGINFOHEADER *pCfgHeader, SPEEDCFGSEG *pDetectCfgSeg, ALGCFGS *pCfgs, ALGPARAMS *pParams)//,CPoint m_ptend[],CPoint LineUp[]
{
	bool 	bValidCfg;

	if(TRUE == CfgStructParse(ChNum, pCfgHeader, pDetectCfgSeg, pCfgs, pParams))//,m_ptend,LineUp
	{
		bValidCfg = TRUE;
	}
	else
	{
		bValidCfg = FALSE;
	}
	if(bValidCfg == TRUE)
	{
		if( ArithVariableInit(ChNum, pCfgs, pParams) == TRUE )
		{

			return TRUE;
		}
		else
		{
			return FALSE;
		}

	}
	else
	{
		return FALSE;
	}
}

bool CfgStructParse(Uint16 ChNum, CFGINFOHEADER *pCfgHeader, SPEEDCFGSEG *pDetectCfgSeg, ALGCFGS *pCfgs, ALGPARAMS *pParams)//,CPoint m_ptend[],CPoint LineUp[]
{
	Int32	i,j,k,idx;
	float min_value=1000;
	Uint16 	uRegin_width;
	Uint16 	uRegin_height;
	//printf("@@@@@@@@@@@1@@@@@@@@@@\n");fflush(NULL);
	struct 	cSpeedDetectStruct 	*pSpeedDetect = NULL;
	struct	LanesStruct 		*pLane = NULL;
	ZENITH_SPEEDDETECTOR 		*pDownSpeedDetect = NULL;
	Uint8 						*pCurLaneImageEntry = NULL;
	bool 	bDetectBackRun[8];
	CPoint	ptFourCorner[4];
	CPoint	ptRowAndCol;
	CPoint  ptRowAndColQ;
	CPoint	ptRowAndColF;
	CPoint	ptCornerQ[2];  ////?
	CPoint	ptCornerQA[2];////?
	CPoint	ptCornerLB[2];  ////0918
	CPoint	ptCornerRB[2];////0918
	CPoint  pt1[2];
	
	int x1=FULL_COLS,x2=0,y1=FULL_ROWS,y2=0;
	pDownSpeedDetect = (ZENITH_SPEEDDETECTOR*)pDetectCfgSeg->uSegData;

	pCfgs->CameraCfg.LaneAmount = pDownSpeedDetect->uLaneTotalNum;
	pCfgs->bAuto = pDownSpeedDetect->uEnvironment; //TRUE;
	pCfgs->CameraCfg.uEnvironmentStatus = pDownSpeedDetect->uEnvironmentStatus;
	pCfgs->CameraCfg.uDayNightJudgeMinContiuFrame = pDownSpeedDetect->uDayNightJudgeMinContiuFrame; //no used
	pCfgs->CameraCfg.uGetBackMinContiuFrame = pDownSpeedDetect->uComprehensiveSens;  //no used
	pCfgs->CameraCfg.uVehicleHeadMinRow = pDownSpeedDetect->uDetectSens1;
	pCfgs->CameraCfg.uInThreshold = pDownSpeedDetect->uDetectSens2; 
	pCfgs->CameraCfg.uSquareThreshold = pDownSpeedDetect->uStatisticsSens1;
	pCfgs->CameraCfg.guSubImageThreshold = pDownSpeedDetect->uStatisticsSens2;
	pCfgs->CameraCfg.uSobelThreshold = pDownSpeedDetect->uSobelThreshold;

	pCfgs->CameraCfg.bCameraWorkInCross = pCfgHeader->uDetectPosition;  
	pCfgs->CameraLocalPara.bNormalDetectEnable = pCfgHeader->uDetectFuncs[0]; 
	if(pCfgs->CameraCfg.uEnvironmentStatus==1||pCfgs->CameraCfg.uEnvironmentStatus==3) 
		pCfgs->CameraCfg.uSquareThreshold=(pCfgs->CameraCfg.uSquareThreshold<20)? pCfgs->CameraCfg.uSquareThreshold:20; //dawn dusk
    for(i=0;i<6;i++)
	{
		pCfgs->exlight[i]=0;
	}
	//?

	//????
	/*for(i=0;i<CALIBRATION_POINT_NUM;i++)
	{
	pCfgs->actual_point[i][0]=pDownSpeedDetect->ptactual[i].x;
	pCfgs->actual_point[i][1]=pDownSpeedDetect->ptactual[i].y;
	pCfgs->image_point[i][0]=pDownSpeedDetect->ptimage[i].x;
	pCfgs->image_point[i][1]=pDownSpeedDetect->ptimage[i].y;

	}*/
	//camera_calibration(pCfgs->actual_point,pCfgs->image_point,pCfgs->mapping_matrix,CALIBRATION_POINT_NUM,pCfgs);//20150608
	//  //////printf("actual_point=[%f,%f],[%f,%f],[%f,%f],[%f,%f],[%f,%f],[%f,%f],[%f,%f],[%f,%f]\n",pCfgs->actual_point[0][0],pCfgs->actual_point[0][1],pCfgs->actual_point[1][0],pCfgs->actual_point[1][1],pCfgs->actual_point[2][0],pCfgs->actual_point[2][1],pCfgs->actual_point[3][0],pCfgs->actual_point[3][1],pCfgs->actual_point[4][0],pCfgs->actual_point[4][1],pCfgs->actual_point[5][0],pCfgs->actual_point[5][1],pCfgs->actual_point[6][0],pCfgs->actual_point[6][1],pCfgs->actual_point[7][0],pCfgs->actual_point[7][1]);
	//  //////printf("image_point=[%f,%f],[%f,%f],[%f,%f],[%f,%f],[%f,%f],[%f,%f],[%f,%f],[%f,%f]\n",pCfgs->image_point[0][0],pCfgs->image_point[0][1],pCfgs->image_point[1][0],pCfgs->image_point[1][1],pCfgs->image_point[2][0],pCfgs->image_point[2][1],pCfgs->image_point[3][0],pCfgs->image_point[3][1],pCfgs->image_point[4][0],pCfgs->image_point[4][1],pCfgs->image_point[5][0],pCfgs->image_point[5][1],pCfgs->image_point[6][0],pCfgs->image_point[6][1],pCfgs->image_point[7][0],pCfgs->image_point[7][1]);
	//	img_to_actual(pCfgs->mapping_matrix,pCfgs);
	////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////

	for(i=0; i<MAX_LANE; i++) 
	{
		pCfgs->CameraLocalPara.LanesCfgEntry[i] = (Uint8 *)pCfgs->CameraLocalPara.CameraCfgEntry+ \
			sizeof(CAMERA_STRUCT) +    \
			i*sizeof(LANE_STRUCT_DEF);
		memset((void*)pCfgs->CameraLocalPara.LanesCfgEntry[i], 0, sizeof(LANE_STRUCT_DEF)); //? 		

		((LANE_STRUCT_DEF *)pCfgs->CameraLocalPara.LanesCfgEntry[i])->AbnormalFunction.AbnormalFunctionBit.BackRunDef = 0;
		bDetectBackRun[i] = FALSE;
	}

	for(i=0; i<pCfgs->CameraCfg.LaneAmount; i++) //
	{
		pLane = (struct	LanesStruct *)pCfgs->CameraLocalPara.LanesCfgEntry[i];			

		memcpy((void *)&pParams->LaneParameter, (void *)pLane, sizeof(struct LanesStruct));
		InitThisLanesPara(&pParams->LaneParameter);
		memcpy((void *)pLane, (void *)&pParams->LaneParameter, sizeof(struct LanesStruct));

		if(2 == pDownSpeedDetect->SpeedEachLane[i].uDetectDerection)	//?2  ??? 	bDetectBackRun[i] = FALSE;
		{
			pLane->AbnormalFunction.AbnormalFunctionBit.BackRunDef = 1;
			////?
			bDetectBackRun[i] = FALSE;
		}
		else if(1 == pDownSpeedDetect->SpeedEachLane[i].uDetectDerection)	//?1  ?? 
		{
			pLane->AbnormalFunction.AbnormalFunctionBit.BackRunDef = 0;
			bDetectBackRun[i] = TRUE;
		}



	}
	pSpeedDetect = (struct cSpeedDetectStruct *) ((char*)pCfgs->CameraLocalPara.CameraCfgEntry + \
		sizeof(CAMERA_STRUCT) +  \
		MAX_LANE*sizeof(LANE_STRUCT_DEF) \
		);
	pCurLaneImageEntry = (Uint8 *)pCfgs->CameraLocalPara.ImageStorageEntry;


	for(i=0; i<pCfgs->CameraCfg.LaneAmount; i++)	//??I???3I缂侇垽鎷�?????y
	{

		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr = pSpeedDetect;
		memset((void*)pSpeedDetect, 0, sizeof(struct cSpeedDetectStruct));

		pSpeedDetect->CoordinatePointer = (Uint32 *) ((Uint8 *)pSpeedDetect +sizeof(SPEED_DETECT_STRUCT));
		//pSpeedDetect->CoordinatePointer1 = (Uint32 *) ((Uint8 *)pSpeedDetect +sizeof(SPEED_DETECT_STRUCT)+720*576);//?
		//pSpeedDetect->CoordinatePointer1 = new Uint32[720*576];
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner, (void*)pDownSpeedDetect->SpeedEachLane[i].ptFourCorner, 4*sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ, (void*)pDownSpeedDetect->SpeedEachLane[i].ptCornerQ, 2*sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQA, (void*)pDownSpeedDetect->SpeedEachLane[i].ptCornerQA, 2*sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerLB, (void*)pDownSpeedDetect->SpeedEachLane[i].ptCornerLB, 2*sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerRB, (void*)pDownSpeedDetect->SpeedEachLane[i].ptCornerRB, 2*sizeof(CPoint) );
		if( bDetectBackRun[i] == TRUE )
		{
			ptFourCorner[0].x = pDownSpeedDetect->SpeedEachLane[i].ptFourCorner[3].x;
			ptFourCorner[0].y = pDownSpeedDetect->SpeedEachLane[i].ptFourCorner[3].y;
			ptFourCorner[1].x = pDownSpeedDetect->SpeedEachLane[i].ptFourCorner[2].x;
			ptFourCorner[1].y = pDownSpeedDetect->SpeedEachLane[i].ptFourCorner[2].y;
			ptFourCorner[2].x = pDownSpeedDetect->SpeedEachLane[i].ptFourCorner[1].x;
			ptFourCorner[2].y = pDownSpeedDetect->SpeedEachLane[i].ptFourCorner[1].y;
			ptFourCorner[3].x = pDownSpeedDetect->SpeedEachLane[i].ptFourCorner[0].x;
			ptFourCorner[3].y = pDownSpeedDetect->SpeedEachLane[i].ptFourCorner[0].y;
		}
		else
		{
			memcpy( (void*)ptFourCorner, (void*)pDownSpeedDetect->SpeedEachLane[i].ptFourCorner, 4*sizeof(CPoint) );
			memcpy( (void*)ptCornerQ, (void*)pDownSpeedDetect->SpeedEachLane[i].ptCornerQ, 2*sizeof(CPoint) );
			memcpy( (void*)ptCornerQA, (void*)pDownSpeedDetect->SpeedEachLane[i].ptCornerQA, 2*sizeof(CPoint) );
			memcpy( (void*)ptCornerLB, (void*)pDownSpeedDetect->SpeedEachLane[i].ptCornerLB, 2*sizeof(CPoint) );
			memcpy( (void*)ptCornerRB, (void*)pDownSpeedDetect->SpeedEachLane[i].ptCornerRB, 2*sizeof(CPoint) );
		}

//		printf("@@@@@@@@@@a@@@@@@@@@@@\n");fflush(NULL);
#ifdef	COORD_DERECT
		memcpy( (void*)pSpeedDetect->CoordinatePointer, (void*)ptFourCorner, 4*sizeof(CPoint) );

		uRegin_width = abs(((CPoint *)pSpeedDetect->CoordinatePointer)[1].x - ((CPoint *)pSpeedDetect->CoordinatePointer)[0].x);
		uRegin_height = abs(((CPoint *)pSpeedDetect->CoordinatePointer)[2].y - ((CPoint *)pSpeedDetect->CoordinatePointer)[1].y);


#else


		if((ptFourCorner[0].x>1000)||(ptFourCorner[1].x>1000)||(ptFourCorner[2].x>1000)||(ptFourCorner[3].x>1000)
			||(ptCornerRB[0].x>1000)||(ptCornerRB[1].x>1000))
			return FALSE;
		if(pCfgs->bAuto==2)
		{
			ptRowAndCol = ptGetDot( &ptFourCorner[0], &ptFourCorner[1], &ptFourCorner[2],	\
				&ptFourCorner[3], 200,(Uint32*)(pSpeedDetect->CoordinatePointer) );




			////20140707  queue
			pSpeedDetect->CoordinatePointerQ=pSpeedDetect->CoordinatePointer+ptRowAndCol.x*ptRowAndCol.y;
			ptRowAndColQ= ptGetDot( &ptCornerLB[0], &ptCornerRB[0], &ptCornerRB[1],	\
				&ptCornerLB[1], 200,(Uint32*)(pSpeedDetect->CoordinatePointerQ) );

		}
		else
		{
			ptRowAndCol = ptGetDot( &ptFourCorner[0], &ptFourCorner[1], &ptFourCorner[2],	\
				&ptFourCorner[3], 20,(Uint32*)(pSpeedDetect->CoordinatePointer) );


			////20140707  queue
			pSpeedDetect->CoordinatePointerQ=pSpeedDetect->CoordinatePointer+ptRowAndCol.x*ptRowAndCol.y;

			ptRowAndColQ= ptGetDot( &ptCornerLB[0], &ptCornerRB[0], &ptCornerRB[1],	\
				&ptCornerLB[1], 200,(Uint32*)(pSpeedDetect->CoordinatePointerQ) );
		}			
		//0707
		pSpeedDetect->ColightPointer=pSpeedDetect->CoordinatePointerQ+ptRowAndColQ.x*ptRowAndColQ.y;
		if(!i)
		{
			ptRowAndColF=ptGetDot( &ptFourCorner[0], &ptFourCorner[1], &ptFourCorner[2],	\
				&ptFourCorner[3], 200,(Uint32*)(pSpeedDetect->ColightPointer) );
		}
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner, (void*)ptFourCorner, 4*sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ, (void*)ptCornerQ, 2*sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQA, (void*)ptCornerQA, 2*sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerLB, (void*)ptCornerLB, 2*sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerRB, (void*)ptCornerRB, 2*sizeof(CPoint) );
		//0707

		if((ptRowAndCol.x==0) && (ptRowAndCol.y==0))
		{ 
			return	FALSE;	
		}
		if((ptRowAndCol.x * ptRowAndCol.y) > MAX_SPEEDDETECTOR_DOTS)
		{
			return	FALSE;	
		}

		if((ptRowAndColQ.x==0) && (ptRowAndColQ.y==0))
		{ 
			return	FALSE;	
		}
		if((ptRowAndColQ.x * ptRowAndColQ.y) > MAX_SPEEDDETECTOR_DOTS)
		{
			return	FALSE;	
		}
		if(i==0&&(ptRowAndColF.x==0) && (ptRowAndColF.y==0))
		{ 
			return	FALSE;	
		}
		if((ptRowAndColF.x * ptRowAndColF.y) > MAX_SPEEDDETECTOR_DOTS)
		{
			return	FALSE;	
		}

		uRegin_width = ptRowAndCol.x;
		uRegin_height = ptRowAndCol.y;
		//////printf("uRegin_width is:%d\t",uRegin_width);
		//////printf("uRegin_height is:%d\t",uRegin_height);
		////20140707  queue
		pCfgs->team_width = ptRowAndColQ.x;
		pCfgs->team_height = ptRowAndColQ.y;
		//////printf("team_width is:%d\t",pCfgs->team_width);
		//////printf("team_height is:%d\t",pCfgs->team_height);
		pCfgs->light_width=ptRowAndColF.x;
		pCfgs->light_height=ptRowAndColF.y;
		//////printf("light_width is:%d\t",pCfgs->light_width);
		//////printf("light_height is:%d\n",pCfgs->light_height);
		pSpeedDetect->QueDetectDots =  pCfgs->team_width * pCfgs->team_height;
		pSpeedDetect->QueDetectLines=pCfgs->team_height;
		pSpeedDetect->QueDetectColumns=pCfgs->team_width;
		////////printf("que_height is %d,que_height is %d\n",pSpeedDetect->QueDetectLines,pSpeedDetect->QueDetectColumns);
		////////printf("virtual_height is %d,virtual_height is %d\n",uRegin_height,uRegin_width);



#endif		//#ifdef COORD_DERECT
		memset(pSpeedDetect->position,0,30*sizeof(int));
		memset(pSpeedDetect->preposition,0,30*sizeof(int));
		memset(pSpeedDetect->pastlength,0,4);
		pSpeedDetect->percent=0;
		pSpeedDetect->sptimes=0;
		pSpeedDetect->DetectDots =  uRegin_width * uRegin_height;

		pSpeedDetect->DetectLines = uRegin_height;
		pSpeedDetect->DetectColumns = uRegin_width;

		pSpeedDetect->thresholdValue = 1;  //ADDED 20130827 FOR IVD2.0
		pCfgs->thresholdValue =1;
		pSpeedDetect->nSpeedInfoTop = -1;
		pSpeedDetect->nSpeedInfoMaxSize = SPEEDINFO_MAXSIZE;
		pSpeedDetect->nLengthInfoTop = -1;
		pSpeedDetect->nLengthInfoMaxSize = LENGTHINFO_MAXSIZE;
		pSpeedDetect->uSLKThreshold	= 4;
		pCfgs->CameraLocalPara.DetectCfgEntry[i] = (Uint8 *)pSpeedDetect;

		//added by david 20131025
		pSpeedDetect->wLastVehicleSpeed = 30;
		pSpeedDetect->wLastVehicleLength = 4;


		pSpeedDetect->uTransFactor = pDownSpeedDetect->SpeedEachLane[i].uTransFactor;

		pSpeedDetect->uSpeedCounterChangedThreshold = pDownSpeedDetect->SpeedEachLane[i].uSpeedCounterChangedThreshold; //0
		pSpeedDetect->uSpeedCounterChangedThreshold1 = pDownSpeedDetect->SpeedEachLane[i].uSpeedCounterChangedThreshold1; //1
		pSpeedDetect->uSpeedCounterChangedThreshold2 = pDownSpeedDetect->SpeedEachLane[i].uSpeedCounterChangedThreshold2;//2
		pSpeedDetect->uGraySubThreshold = pDownSpeedDetect->SpeedEachLane[i].uGraySubThreshold;


		pSpeedDetect->uSpeedCounterStartRow = pSpeedDetect->DetectLines - SPEED_COUNTER_STARTLINE;

		if(pCfgs->bAuto==2)
		{
			pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight =TRUE;
			pSpeedDetect->SpeedDetectorWorkMode.bDay = FALSE; 	//FALSE; 	//???2?3?????E??????I?
			pSpeedDetect->SpeedDetectorWorkMode.bNight = TRUE; //TRUE;		//???2?3?????E??????I?
			pCfgs->bNight = TRUE;
		}
		else
		{
			pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight =FALSE;
			pSpeedDetect->SpeedDetectorWorkMode.bDay = TRUE; 	//FALSE; 	//???2?3?????E??????I?
			pSpeedDetect->SpeedDetectorWorkMode.bNight = FALSE; //TRUE;		//???2?3?????E??????I?
			pCfgs->bNight = FALSE;
		}

		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->CurrentImagePointer =
			(Uint16 *)pCurLaneImageEntry;

		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->BackImagePointer =
			(Uint16 *)( (Uint8 *)pCurLaneImageEntry + (uRegin_width*uRegin_height) );//txl.0701

		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->puTempImage =
			(Uint16 *)( (Uint8 *)pCurLaneImageEntry + 2*(uRegin_width*uRegin_height) );




		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->SobelBackImagePointer =
			(Uint16 *)( (Uint8 *)pCurLaneImageEntry + 3*(uRegin_width*uRegin_height) );
		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->BackImagePlusPointer =
			(Uint16 *)( (Uint8 *)pCurLaneImageEntry + 4*(uRegin_width*uRegin_height) );


		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->CurrQueueImage =
			(Uint16 *)( (Uint8 *)pCurLaneImageEntry + 5*(uRegin_width*uRegin_height) );
		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->PreQueueImage =
			(Uint16 *)( (Uint8 *)pCurLaneImageEntry + 5*(uRegin_width*uRegin_height)+(pCfgs->team_width*pCfgs->team_height) );
		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->SobelQueueImage =
			(Uint16 *)( (Uint8 *)pCurLaneImageEntry + 5*(uRegin_width*uRegin_height)+2*(pCfgs->team_width*pCfgs->team_height));
		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->BackQueueImage =
			(Uint16 *)( (Uint8 *)pCurLaneImageEntry +5*(uRegin_width*uRegin_height)+3*(pCfgs->team_width*pCfgs->team_height) );
		pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->FirstLaneImage =
			(Uint16 *)( (Uint8 *)pCurLaneImageEntry +5*(uRegin_width*uRegin_height)+4*(pCfgs->team_width*pCfgs->team_height) );//0707xyx
		//added by david 20130826 for alg IVD2.0


		pCurLaneImageEntry += 5*(uRegin_width*uRegin_height)+4*(pCfgs->team_width*pCfgs->team_height)+(pCfgs->light_width*pCfgs->light_height);


		pSpeedDetect = 	(struct cSpeedDetectStruct*)( (char*)pSpeedDetect + sizeof(SPEED_DETECT_STRUCT) + uRegin_width*uRegin_height*sizeof(Uint32)
			+pCfgs->team_width*pCfgs->team_height*sizeof(Uint32)+pCfgs->light_width*pCfgs->light_height*sizeof(Uint32) );
	//	printf("@@@@@@@@@@@end @@@@@@@@@@\n");fflush(NULL);
	}
	for(i=0;i<4;i++)
	{
		pCfgs->calibration_point[i][0]=pDownSpeedDetect->ptimage[i].x;
		pCfgs->calibration_point[i][1]=pDownSpeedDetect->ptimage[i].y;

	}
	for(i=4;i<6;i++)
	{
		pCfgs->base_line[i-4][0]=pDownSpeedDetect->ptimage[i].x;
		pCfgs->base_line[i-4][1]=pDownSpeedDetect->ptimage[i].y;
	}
	pCfgs->base_length=pDownSpeedDetect->base_length;
	pCfgs->near_point_length=pDownSpeedDetect->near_point_length;

	camera_calibration(pCfgs->base_line,pCfgs->base_length,pCfgs->calibration_point,pCfgs->near_point_length,pCfgs->CameraCfg.LaneAmount,pCfgs);
	for(i=0; i<pCfgs->CameraCfg.LaneAmount; i++)
	{
		pt1[0].y=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[0].y+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[1].y)/2;
		pt1[1].y=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[2].y+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[3].y)/2;
		pCfgs->uActualTailLength[i]=int(abs(pCfgs->actual_distance[i][pt1[0].y]-pCfgs->actual_distance[i][pt1[1].y])+0.5);
		pt1[0].y=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[0].y+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[1].y)/2;
		pt1[1].y=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2].y+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3].y)/2;
		pCfgs->uActualDetectLength[i]=int(abs(pCfgs->actual_distance[i][pt1[0].y]-pCfgs->actual_distance[i][pt1[1].y])+0.5);
	}
//	printf("@@@@@@@@@@@@x@@@@@@@@@\n");fflush(NULL);
	for(i=0;i<20;i++)
	{
		pCfgs->degreepoint[i][0]=0;
		pCfgs->degreepoint[i][1]=0;
	}
	j=((int)(pCfgs->near_point_length+9)/10+1)*10;
	k=0;
	for(i=FULL_ROWS-1;i>=0;i--)
	{
		if(abs(pCfgs->actual_degree_length[i]-j)<min_value)
		{
			min_value=abs(pCfgs->actual_degree_length[i]-j);
			idx=i;
		}
		if((int)(pCfgs->actual_degree_length[i]+0.5)==j)
		{
			pCfgs->degreepoint[k][0]=idx;
			pCfgs->degreepoint[k][1]=j;
			////////printf("degree_y=%d,len=%d\n",pCfgs->degreepoint[k][0],pCfgs->degreepoint[k][1]);
			min_value=1000;
			j=j+10;
			k++;


		}
		if(k==20)
		{
			break;
		} 
	}
	//pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uDegreeLength=pCfgs->actual_degree_length;
    ////////////////////////////////////////////////////detect params
	if(pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight==FALSE)
	{
		ADBOOST_DETECT  AdboostDetect;
		if( !AdboostDetect.face_cascade.load_file(pCfgs->CameraCfg.uEnvironmentStatus) )
		{ 
			//printf("--(!)Error loading\n");
			return FALSE; 
		}
		int testStage=15;
		int nstages = (int)AdboostDetect.face_cascade.data.stages.size();
		if(testStage>nstages)
			testStage=nstages;
		if(testStage<nstages)
		{
			vector<CascadeClassifier::Data::Stage> stage;
			for(int i=0;i<testStage;++i)
			{
				stage.push_back(AdboostDetect.face_cascade.data.stages[i]);
			}
			AdboostDetect.face_cascade.data.stages=stage;
		}
		pCfgs->AdboostDetect.face_cascade=AdboostDetect.face_cascade;
		pCfgs->AdboostDetect.min_width=32;
		pCfgs->AdboostDetect.min_height=40;
		pCfgs->AdboostDetect.max_width=120;
		pCfgs->AdboostDetect.max_height=150;                                        
		pCfgs->AdboostDetect.scaleFactor=1.1;
		pCfgs->AdboostDetect.minNeighbor=2;
		pCfgs->detectROI.x=0;
		pCfgs->detectROI.y=0;
		pCfgs->detectROI.width=FULL_COLS/2;
		pCfgs->detectROI.height=FULL_ROWS/2;
		if(pCfgs->CameraCfg.LaneAmount)
		{
			for(i=0; i<pCfgs->CameraCfg.LaneAmount; i++) 
			{
				for(j=0;j<4;j++)
				{
					x1=(x1<pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[j].x)? x1:pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[j].x;
					x2=(x2>pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[j].x)? x2:pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[j].x;
					y1=(y1<pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[j].y)? y1:pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[j].y;
					y2=(y2>pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[j].y)? y2:pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[j].y;
					pCfgs->detectPosition=y2;
				}
			}
			//x1=(x1>0)? x1:0;
			//y1=(y1-300>40)? (y1-300):40;
			//x2=(x2<FULL_COLS)? x2:FULL_COLS;
			y2=((y2+40)<FULL_ROWS)? (y2+40):FULL_ROWS;
			//txl@20160323
			/*if(pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight)
			{
				y1=(y1-100)>60?(y1-100):60;
			}
			else
			{
				y1=(y2-350>40)? (y2-350):40;
			}*/
			y1=(y2-340>40)? (y2-340):40;		
			y2=(y2<y1)? y1:y2;
			pCfgs->detectROI.x=0;
			pCfgs->detectROI.y=y1/2;
			pCfgs->detectROI.width=FULL_COLS/2;
			pCfgs->detectROI.height=(y2-y1)/2;
			pCfgs->detectPosition=pCfgs->detectPosition/2;
		}
		pCfgs->target_id=1;
		pCfgs->targets_size=0;
		pCfgs->candidate_targets_size=0;
		memset(pCfgs->targets,0,100*sizeof(CTarget));
		memset(pCfgs->candidate_targets,0,100*sizeof(CTarget));
	
	}
//	printf("@@@@@@@@@@@2@@@@@@@@@@\n");fflush(NULL);
	return	TRUE;

}

bool ArithVariableInit(Uint16 ChNum, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	Int32	i,j;
	CPoint	ptFourCorner[4];
	float k1,b1,k2,b2;
	Int32 startX,endX;
	unsigned char* p;
	pCfgs->gThisFrameTime = 0; /////?????缁绢叏鎷�???????A缂佸鍠嗛崝澶愭晸閻ｅ瞼浠�???D??3y
	pCfgs->guMinCountIntervalFrame=5;////25
	pCfgs->guAdmitBreakFrame = 2; ///2  
	//pParams->puPointNewImage=Image[0];
	pParams->puPointSobelImage =(Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX;//?????
	pParams->puSobelTempImageX =(Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*2;
	pParams->puSobelTempImageY = (Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*3;
//	printf("@@@@@@@@@@@@@@@@@@@@@\n");fflush(NULL);
	////20131218 zcj
	pParams->puPointNextImage = (Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*4;

	////	pParams->puAlgTempImage = (Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*4;

	pParams->puBackImagePlus =(Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*5; //20130830 by david for ivd2.0

	pParams->puPointBackSobelImage = (Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*6;
	//pParams->PointNewImage=(Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*7;

	//txl,0630
	pParams->CurrQueueImage = (Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*7;
	pParams->PreQueueImage = (Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*8;
	pParams->SobelQueueImage = (Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*9;
	pParams->BackQueueImage = (Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*10;
	//	pParams->MidImage=(Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*11;
	pParams->FirstLaneImage=(Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*11;
	pParams->CurrFullImage=(Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*12;
	pParams->PrevFullImage=(Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*13;
	pParams->MaskFullImage = (Uint8 *)pParams->puPointNewImage + DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX*14;
	for( i = 0; i < 8; i++ )
	{
		memset( (void*)&pCfgs->CameraLocalPara.guSpeedCounterStatus[i], 0, sizeof(SPEED_COUNTER_STATUS) );
	}

	for(i=0; i<MAX_LANE; i++)
	{
		pCfgs->uDetectInSum[i] = 0;
		pCfgs->uDetectOutSum[i] = 0;
		pCfgs->VehicleTeamSave[i]=1;////20140728
		pCfgs->firstx[i]=0;
		pCfgs->secondx[i]=0;
	}
	memset((void*)pParams->MaskFullImage,0,sizeof(unsigned char)*DETECTRECT_WIDTH_MAX*DETECTRECT_HEIGHT_MAX);
	if(pCfgs->CameraCfg.LaneAmount)
	{
		ptFourCorner[0]=pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[0];
		ptFourCorner[1]=pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[1];
		ptFourCorner[2]=pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[2];
		ptFourCorner[3]=pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[3];
		for(i=1;i<pCfgs->CameraCfg.LaneAmount;i++)
		{
			ptFourCorner[0]=(ptFourCorner[0].x<pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[0].x)? ptFourCorner[0]:pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[0];
			ptFourCorner[1]=(ptFourCorner[1].x>pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[1].x)? ptFourCorner[1]:pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[1];
			ptFourCorner[2]=(ptFourCorner[2].x<pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2].x)? ptFourCorner[2]:pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2];
			ptFourCorner[3]=(ptFourCorner[3].x>pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3].x)? ptFourCorner[3]:pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3];
		}
		ptFourCorner[0].x=ptFourCorner[0].x/2;
		ptFourCorner[0].y=ptFourCorner[0].y/2;
		ptFourCorner[1].x=ptFourCorner[1].x/2;
		ptFourCorner[1].y=ptFourCorner[1].y/2;
		ptFourCorner[2].x=ptFourCorner[2].x/2;
		ptFourCorner[2].y=ptFourCorner[2].y/2;
		ptFourCorner[3].x=ptFourCorner[3].x/2;
		ptFourCorner[3].y=ptFourCorner[3].y/2;
		if(ptFourCorner[0].x!=ptFourCorner[2].x)
		{
			k1=(float)(ptFourCorner[0].y-ptFourCorner[2].y)/(ptFourCorner[0].x-ptFourCorner[2].x);
			b1=ptFourCorner[0].y-k1*ptFourCorner[0].x;
		}
		if(ptFourCorner[1].x!=ptFourCorner[3].x)
		{
			k2=(float)(ptFourCorner[1].y-ptFourCorner[3].y)/(ptFourCorner[1].x-ptFourCorner[3].x);
			b2=ptFourCorner[1].y-k2*ptFourCorner[1].x;
		}
		for(i=pCfgs->detectROI.y;i<(pCfgs->detectROI.y+pCfgs->detectROI.height);i++)
		{
			if(ptFourCorner[0].x!=ptFourCorner[2].x)
				startX=(int)((i-b1)/k1);
			else
				startX=ptFourCorner[0].x;
			startX=(startX-50)<0? 0:(startX-50);
			if(ptFourCorner[1].x!=ptFourCorner[3].x)
				endX=(int)((i-b2)/k2);
			else
				endX=ptFourCorner[1].x;
			endX=(endX)>(pCfgs->detectROI.x+pCfgs->detectROI.width)? (pCfgs->detectROI.x+pCfgs->detectROI.width):(endX);
			p=pParams->MaskFullImage+(i-pCfgs->detectROI.y)*FULL_COLS/2-pCfgs->detectROI.x;
			for(j=startX;j<endX;j++)
			{
				*(p+j)=1;
			}
		}
	}
	////20131219	
	pParams->NormalDetectCfg.cSpeedDesc.bGetBackFlag=0;
	pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged=TRUE;
	pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1=TRUE;
	pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged2=TRUE;
	pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged3=TRUE;
	pParams->NormalDetectCfg.cSpeedDesc.bBottomNull=TRUE;


	return TRUE;  //ADDED BY DAVID 20131023

}

void InitThisLanesPara(struct LanesStruct *LanePtr)
{
	LanePtr->AccumulatedFrameTime=0;
	LanePtr->bArbitration = 0;
	LanePtr->bIsVehicleInSpeed = 0;
	LanePtr->CounterValue = 0;
	LanePtr->LengthGetTimes = 0;
	LanePtr->LengthSum = 0;
	LanePtr->nReserve[0] = 0;
	LanePtr->nReserve[1] = 0;
	LanePtr->nReserve[2] = 0;
	//			LanePtr->bLaneEventFlag = 0xFFFF;
	LanePtr->SpeedGetTimes = 0;
	LanePtr->SpeedSum = 0;
	LanePtr->SpeedValue = 0;
	LanePtr->StatisticStartFrameTime = 0;
	LanePtr->TotalFrameCount = 0;
	LanePtr->TotalVehicleFlow = 0;
	LanePtr->uCarTypeTransformFactorDownLimit = 0;
	LanePtr->uCarTypeTransformFactorUpLimit = 0;
	LanePtr->uCounterWrapChangedThreshold = 0;
	LanePtr->uDetectorInSignalFrame = 0;
	LanePtr->uEndRow = 0;
	LanePtr->uFrameCount = 0;
	LanePtr->uGetVehiclePileLengthCount = 0;
	LanePtr->uLaneVehicleMaxReportLength = 0;
	LanePtr->uSobelRowIsChangedRatio = 0;
	LanePtr->uSobelRowIsChangedThreshHold = 0;
	LanePtr->uStartRow = 0;
	LanePtr->uTimeDifferenceRatioThreshHold = 0;
	LanePtr->uVehicleMinLengthInPresence = 0;
	LanePtr->uVehiclePileLengthSum = 0;
	LanePtr->VehicleExistFrameCount = 0;
	LanePtr->VehicleLength = 0;
}
bool GetDetectRegion(const unsigned char * in_data, int nWidth, int nHeight, unsigned char * out_data_y, unsigned char * out_data_uv, int startLine, int endLine,
	int startCol, int endCol)
{
	int i,j,k;
	int nIndexStart;
	int nIndexEnd;

	k = 0;
	if(out_data_y==NULL)
	{
		return FALSE;
	}

	nIndexStart = startCol*2;
	nIndexEnd = endCol*2;

	for(i=startLine; i<endLine; i++)
	{
		for(j=nIndexStart; j<nIndexEnd; j+=2)
		{
			//out_data_y[k] = in_data[i*nWidth*2+j+1];
			memcpy((void *)&out_data_y[k],(void *)&in_data[i*nWidth*2+j+1], sizeof(unsigned char));
			// if(out_data_uv != NULL)
			// {
			// out_data_uv[k] = in_data[i*nWidth*2+j];
			// }
			k++;
		}
	}

	return TRUE;
}
void StoreCurImage(Uint8 *inBuf, ALGCFGS *pCfgs, ALGPARAMS	*pParams)
{
	Uint32	i,j;
	Uint32	Elements,QElements;
	Uint32 	index;
	Uint32	*pCoordinatePointer;
	struct 	cSpeedDetectStruct *pSpeedDetect = NULL;
	//unsigned int w_data=0,h_data=0;
	for( i = 0; i < pCfgs->CameraCfg.LaneAmount; i++ )
	{
		pSpeedDetect = pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr;
		pCoordinatePointer = pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->CoordinatePointer;
		Elements = pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->DetectDots;
		QElements = pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->QueDetectDots;
		//w_data=pSpeedDetect->DetectColumns;
		//h_data=pSpeedDetect->DetectLines;
		if (inBuf)
		{
			for(j=0; j<Elements; j++)
			{
				index = *(pCoordinatePointer+j);
				/*index = index*2 + 1;*/
				*((Uint8*)(pSpeedDetect->CurrentImagePointer)+j) = *(inBuf+index);
			}
			//
			pCoordinatePointer = pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->CoordinatePointerQ;
			//Elements = team_width*team_height;
			//w_data=pSpeedDetect->DetectColumns;
			//h_data=pSpeedDetect->DetectLines;
			for(j=0; j<QElements; j++)
			{
				index = *(pCoordinatePointer+j);
				/*index = index*2 + 1;*/
				*((Uint8*)(pSpeedDetect->CurrQueueImage)+j) = *(inBuf+index);//txl,0701
			}
			//
			////0707xyx
			if(!i)
			{
				pCoordinatePointer = pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr->ColightPointer;
				Elements = pCfgs->light_width*pCfgs->light_height;

				for(j=0; j<Elements; j++)
				{
					index = *(pCoordinatePointer+j);
					//index = index*2 + 1;
					*((Uint8*)(pSpeedDetect->FirstLaneImage)+j) = *(inBuf+index);

				}	
			}
		}
		///
	}
	return;

}
//int sptimes=0;
Uint16 ArithProc(Uint16 ChNum, IVIDEO1_BufDescIn * inBuf, RESULTMSG* outBuf, Int32 outSize, ALGCFGS *pCfgs, ALGPARAMS	*pParams)//,CPoint LineUp1[],CPoint m_ptend1[]
{
	//printf("1111111111111\n");
	Int32 i;
	Uint16	FrameWidth;
	Uint16	FrameHeight;
	float tAverageGray=0.0,AverageGray=0.0,change_speed_tAverageGray=0.0;
	bool	bSpeedEnable, bPresenceEnable;
	struct 	cSpeedDetectStruct *pSpeedDetect = NULL;
	IVIDEO1_BufDescIn *inBufs;

	float temp=0,temp1=0;


	CPoint m_ptend[12];
	CPoint LineUp[2];
	float point1[2]={0};
	float point2[2]={0};
	int j=0,k=0;//txl,20160105
	int thr=10;
    int flag=0;
	Uint16 width=0,height=0;
	unsigned char *p,*p0,*p1,*p2,*p3,*p4;
	inBufs = (IVIDEO1_BufDescIn *)inBuf;

	FrameWidth = inBufs->frameWidth;
	FrameHeight = inBufs->frameHeight;
	////printf("uEnvironmentStatus1 is %d/t",pCfgs->CameraCfg.uEnvironmentStatus);
	//0709xyx
	if((pCfgs->CameraCfg.uEnvironmentStatus==4)&&pCfgs->gThisFrameTime%199==1&&pCfgs->targets_size<10)//daytime
	{
		thr=8;
		pCfgs->up_visib_value++;
		pCfgs->fuzzydegree=fuzzy(inBufs->bufDesc[0].buf , FrameWidth, FrameHeight);//624xyx
		//txl,20160105
		for (j=VISIB_LENGTH-1;j>0;j--)
		{
			pCfgs->visib_value[j]=pCfgs->visib_value[j-1];
		}
		pCfgs->visib_value[0]=(int)(pCfgs->fuzzydegree);
		if (pCfgs->up_visib_value>VISIB_LENGTH)
		{
			pCfgs->visibility=visible_judge(pCfgs->visib_value,VISIB_LENGTH,thr);
		} 
		else
		{
			pCfgs->visibility=FALSE;
		}
		//TRACE("pCfgs->fuzzydegree=%d",pCfgs->fuzzydegree);
		//pCfgs->avegray=StandardDeviation (inBufs->bufDesc[0].buf , FrameWidth,FrameHeight,0, FrameHeight,1);
		//AveGray(inBufs->bufDesc[0].buf , FrameWidth, FrameHeight,pCfgs);//702xyx
		//TRACE("pCfgs->avegray=%d",pCfgs->avegray);
		//if(Visibility(inBufs->bufDesc[0].buf , FrameWidth, FrameHeight,pCfgs))
		//	pCfgs->visitimes=0;
		//else
		//pCfgs->visitimes++;
		//if(pCfgs->fuzzydegree<=10)
		//pCfgs->fuzzytimes++;
		//else
		//pCfgs->fuzzytimes=0;
		//if(pCfgs->fuzzytimes>5)
		//	pCfgs->fuzzyflag=true; 
		//pCfgs->visibility=pCfgs->fuzzytimes>7?true:false;// true:not visible
		//if (pCfgs->fuzzytimes>5)
		//	pCfgs->visibility=true;	
		////////printf("fuzzydegree is: %d, pCfgs->fuzzytimes is:%d\n",pCfgs->fuzzydegree,pCfgs->fuzzytimes);
	}
	else if(pCfgs->CameraCfg.uEnvironmentStatus==2||pCfgs->CameraCfg.uEnvironmentStatus==1||pCfgs->CameraCfg.uEnvironmentStatus==3)//night dawn 
	{
		pCfgs->visibility=FALSE;
	}
	//0709xyx
	////printf("visibility calculate is ok");
#ifdef COORD_DERECT
	//?A????1???-?????D????a??I???????A??I?
	for( i = 0; i < pCfgs->CameraCfg.LaneAmount; i++ )
	{
		pSpeedDetect = pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr;

		startLine = ((CPoint *)pSpeedDetect->CoordinatePointer)[0].y;
		endLine = ((CPoint *)pSpeedDetect->CoordinatePointer)[2].y;
		startCol = ((CPoint *)pSpeedDetect->CoordinatePointer)[0].x;
		endCol = ((CPoint *)pSpeedDetect->CoordinatePointer)[1].x;

		bStatus = GetDetectRegion(inBufs->bufDesc[0].buf , FrameWidth, FrameHeight, pParams->puPointNewImage, NULL, startLine, endLine, startCol, endCol);

		if(bStatus==FALSE)
		{
			return 0;		//failed
		}

		memcpy((void *)pSpeedDetect->CurrentImagePointer, (void *)pParams->puPointNewImage, pSpeedDetect->DetectDots);
	}

#else
	StoreCurImage((Uint8 *)inBufs->bufDesc[0].buf, pCfgs, pParams);
	////printf("StoreCurImage is ok");
#endif	

	//txl@20160323
	if(!pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight)
	{
		width=FrameWidth/2;
		height=FrameHeight/2;
		for(k=pCfgs->detectROI.x;k<(pCfgs->detectROI.x+pCfgs->detectROI.width);k++)
		{
			*(pParams->CurrFullImage+k)=*(inBufs->bufDesc[0].buf+k*2);
			*(pParams->CurrFullImage+(height-1)*width+k)=*(inBufs->bufDesc[0].buf+(height-1)*FrameWidth*2+k*2);
		}
		for(j=pCfgs->detectROI.y;j<(pCfgs->detectROI.y+pCfgs->detectROI.height);j++)
//		char tt[1000];
//		while(1)
		{
			*(pParams->CurrFullImage+j*width)=*(inBufs->bufDesc[0].buf+j*FrameWidth*2);
			*(pParams->CurrFullImage+j*width+width-1)=*(inBufs->bufDesc[0].buf+j*FrameWidth*2+(width-1)*2);
		}
		for(j=pCfgs->detectROI.y+1;j<(pCfgs->detectROI.y+pCfgs->detectROI.height-1);j++)
		{
			p=inBufs->bufDesc[0].buf+j*FrameWidth*2;
			for(k=pCfgs->detectROI.x+1;k<(pCfgs->detectROI.x+pCfgs->detectROI.width-1);k++)
			{
				p0=p+2*k;
				p1=p0-1;
				p2=p0+1;
				p3=p0-FrameWidth;
				p4=p0+FrameWidth;
				*(pParams->CurrFullImage+j*width+k)=(*p0+*p1+*p2+*p3+*p4)/5;
			}
		}
		//get_target(currFrame,prevFrame,AdboostDetect,pCfgs,candidate_roi,1);
		get_target(pParams->CurrFullImage,pParams->PrevFullImage,pParams->MaskFullImage,pParams->puSobelTempImageX,width,height,pCfgs,pParams,1);


//		static int flg=0;
//		char tt[1000];
//		while(1)
//		{
//			memcpy(pParams->PrevFullImage+flg*1000,tt,1000);
//
//			//printf("test %d\n",flg++);
//
//		}while(1) ;

		//memcpy((void *)pParams->PrevFullImage, (void *)inBufs->bufDesc[0].buf, FrameWidth*FrameHeight);
		memcpy((void *)pParams->PrevFullImage, (void *)pParams->CurrFullImage, width*height);
	}


	pCfgs->gThisFrameTime++;
	memset((void *)&pCfgs->ResultMsg, 0, sizeof(pCfgs->ResultMsg));
	memcpy((void *)&pCfgs->ResultMsg, (void *)outBuf, outSize);
	bSpeedEnable = (bool)(pCfgs->CameraLocalPara.bNormalDetectEnable & 0x00000001);
	bPresenceEnable = (bool)(pCfgs->CameraLocalPara.bNormalDetectEnable & 0x00000002);

	for( i = 0; i < pCfgs->CameraCfg.LaneAmount; i++ )
	{
		pCfgs->ResultMsg.uResultInfo.LaneSum = pCfgs->CameraCfg.LaneAmount;
		memcpy((void *)&pParams->LaneParameter, (void *)pCfgs->CameraLocalPara.LanesCfgEntry[i], sizeof(struct	LanesStruct));

		if(bSpeedEnable == TRUE)
		{
			pSpeedDetect = pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr;
			pCfgs->team_height=pSpeedDetect->QueDetectLines;
			pCfgs->team_width=pSpeedDetect->QueDetectColumns;
			//pCfgs->sptimes=pSpeedDetect->sptimes;
			//memset(pCfgs->sptimes,0,5*sizeof(int));
			pCfgs->percent=pSpeedDetect->percent;
			//exchange 
			memcpy((void *)&pParams->NormalDetectCfg, (void *)pSpeedDetect, sizeof(struct	cSpeedDetectStruct));

			memcpy((void *)pParams->puPointNewImage, (void *)pSpeedDetect->CurrentImagePointer, pSpeedDetect->DetectDots);

			memcpy((void *)pParams->puPointSobelImage, (void *)pSpeedDetect->puTempImage, pSpeedDetect->DetectDots);

			//memcpy((void *)pParams->puSobelTempImageX, (void *)pSpeedDetect->BackImagePointer, pSpeedDetect->DetectDots);

			memcpy((void *)pParams->puBackImagePlus, (void *)pSpeedDetect->BackImagePlusPointer, pSpeedDetect->DetectDots);
			//memcpy((void *)pParams->puSobelTempImageX, (void *)pSpeedDetect->CurrQueueImage, pSpeedDetect->QueDetectDots);//
			//memset((void *)pSpeedDetect->CurrQueueImage,0,pSpeedDetect->QueDetectDots);
			//txl,0630
			memcpy((void *)pParams->CurrQueueImage, (void *)pSpeedDetect->CurrQueueImage, pSpeedDetect->QueDetectDots);//960

			memcpy((void *)pParams->PreQueueImage, (void *)pSpeedDetect->PreQueueImage, pSpeedDetect->QueDetectDots);

			//memcpy((void *)pParams->SobelQueueImage, (void *)pSpeedDetect->CurrQueueImage, pSpeedDetect->QueDetectDots);

			//memcpy((void *)pParams->BackQueueImage, (void *)pSpeedDetect->CurrQueueImage, pSpeedDetect->QueDetectDots);
			memcpy((void *)pParams->BackQueueImage, (void *)pSpeedDetect->BackQueueImage, pSpeedDetect->QueDetectDots);
			////printf("memcpy is ok");
			////20150918  TXL
			//////printf("ptCornerLB[0]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerLB[0].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerLB[0].y);
			//////printf("ptCornerLB[1]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerLB[1].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerLB[1].y);
			//////printf("ptCornerRB[0]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerRB[0].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerRB[0].y);
			//////printf("ptCornerRB[1]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerRB[1].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerRB[1].y);

			//////printf("ptCornerQ[0]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerQ[0].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerQ[0].y);
			//////printf("ptCornerQ[1]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerQ[1].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerQ[1].y);
			//////printf("ptCornerQA[0]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerQA[0].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerQA[0].y);
			//////printf("ptCornerQA[1]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerQA[1].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerQA[1].y);

			//////printf("ptFourCorner[0]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[0].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[0].y);
			//////printf("ptFourCorner[1]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[1].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[1].y);
			//////printf("ptFourCorner[2]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[2].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[2].y);
			//////printf("ptFourCorner[3]is:%d,%d\n",pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[3].x,pCfgs->DownSpeedCfg.SpeedEachLane[0].ptFourCorner[3].y);


			m_ptend[0]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerLB[0]; ////??
			m_ptend[1]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerRB[0];
			m_ptend[2]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerRB[1];  ////????
			m_ptend[3]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerLB[1];
			m_ptend[4]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[0];  ////????
			m_ptend[5]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[1];
			m_ptend[6]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQA[0];  ////
			m_ptend[7]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQA[1];  ////		
			m_ptend[8]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[0];  ////????
			m_ptend[9]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[1];
			m_ptend[10]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3];  ////????
			m_ptend[11]= pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2]; 
			// if (pCfgs->gThisFrameTime%100==0){
			//////printf("m_ptend[0] is:%d,%d\n",m_ptend[0].x,m_ptend[0].y);
			//////printf("m_ptend[3] is:%d,%d\n",m_ptend[3].x,m_ptend[3].y);
			//////printf("m_ptend[1] is:%d,%d\n",m_ptend[1].x,m_ptend[1].y);
			//////printf("m_ptend[2] is:%d,%d\n",m_ptend[2].x,m_ptend[2].y);//}
			SpeedCaculate( i, ChNum, pParams->puPointNewImage, outSize,pCfgs, pParams,LineUp,pParams->CurrQueueImage,m_ptend);

			////printf("SpeedCaculate is OK\t");
			//////printf("percent is %f\t",pParams->NormalDetectCfg.cSpeedDesc.percent*100);
			//////printf("IsCarInTailFlag is %d\t",pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail);
			//////printf("uVehicleQueueLength is %d\t",pParams->NormalDetectCfg.cSpeedDesc.percent*pCfgs->team_height);
			//if (pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail)
			//{
				//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.AlarmLineflag=2;
			//	pCfgs->sptimes[i]++;

			//}
			//else
			//{
				//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.AlarmLineflag=1;
			//	pCfgs->sptimes[i]=0;

			//}
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.getQueback_flag=pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag;//txl,20160104
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.IsCarInTailFlag=pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail;
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleHeadtime=pParams->NormalDetectCfg.cSpeedDesc.headtime;
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength=pParams->NormalDetectCfg.cSpeedDesc.percent*pCfgs->team_height;
			/*if (pCfgs->gThisFrameTime%25==0)
			{*/
			/*pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].x=0;
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y=0;
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].x=0;
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y=0;

			LineUp[0]=pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0];
			LineUp[1]=pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1];*/

			pCfgs->percent=pParams->NormalDetectCfg.cSpeedDesc.percent;
			////////printf("pCfgs->percent is: %f\n",pCfgs->percent*100);
			//	pCfgs->IsCarInTail=pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail;

			//	LineUp[0].x=(m_ptend[2].x+m_ptend[3].x)/2;

			//	if (m_ptend[2].x-m_ptend[3].x)
			//		LineUp[0].y=(m_ptend[2].y+m_ptend[3].y)/2;//(LineUp[0].x-m_ptend[3].x)*(m_ptend[2].y-m_ptend[3].y)/(m_ptend[2].x-m_ptend[3].x)+m_ptend[3].y;
			//if((m_ptend[0].x-m_ptend[3].x))
			//{	//LineUp[1].x=m_ptend[3].x+(LineUp[1].y-LineUp[0].y)*(m_ptend[0].x-m_ptend[3].x)/(m_ptend[0].y-m_ptend[3].y);
			//	LineUp[1].x=LineUp[0].x+((m_ptend[0].x+m_ptend[1].x)/2-LineUp[0].x)*pCfgs->percent;
			//	if((m_ptend[0].x+m_ptend[1].x)/2-LineUp[0].x)
			//		LineUp[1].y=LineUp[0].y-(float)(LineUp[0].y-(m_ptend[0].y+m_ptend[1].y)/2)/(float)(LineUp[0].x-(m_ptend[0].x+m_ptend[1].x)/2)*(LineUp[0].x-LineUp[1].x);
			//	else
			//		LineUp[1].y=LineUp[0].y-pCfgs->percent*pCfgs->team_height;	



			//	pParams->NormalDetectCfg.cSpeedDesc.sptimes=pCfgs->sptimes;
			if(m_ptend[0].x-m_ptend[3].x)
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].x=m_ptend[3].x+(m_ptend[0].x-m_ptend[3].x)*pCfgs->percent;
			else
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].x=m_ptend[3].x;
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y=m_ptend[3].y+(m_ptend[0].y-m_ptend[3].y)*pCfgs->percent;
			if(m_ptend[1].x-m_ptend[2].x)
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].x=m_ptend[2].x+(m_ptend[1].x-m_ptend[2].x)*pCfgs->percent;
			else
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].x=m_ptend[2].x;
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y=m_ptend[2].y+(m_ptend[1].y-m_ptend[2].y)*pCfgs->percent;
			//pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0] = LineUp[0];
			//pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1] = LineUp[1];
			//get_actual_point(point1,(int)LineUp[1].y,(int)LineUp[1].x,0,pCfgs);
			//get_actual_point(point2,(int)LineUp[0].y,(int)LineUp[0].x,0,pCfgs);	
			//pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength=int(distance_two(point1,point2)+0.5);
			//pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength=int(pCfgs->actual_distance[i][LineUp[1].y]+0.5);	
			//////printf("LineUp[0] is:%d,%d\n",pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].x,pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y);
			//////printf("LineUp[1] is:%d,%d\n",pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].x,pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y);
			////////printf("m_ptend[1] is:%d,%d\n",m_ptend[1].x,m_ptend[1].y);
			////////printf("m_ptend[2] is:%d,%d\n",m_ptend[2].x,m_ptend[2].y);
			temp=pCfgs->actual_distance[i][(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y+pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y)/2];			
			if(pCfgs->bNight)
			{
				//////printf("tempn is %f\t",temp);
				temp1=temp+4.5;
				while(temp<temp1&&pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y>m_ptend[0].y&&pCfgs->percent)
				{
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y=pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y-1;
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y=pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y-1;
					temp=pCfgs->actual_distance[i][(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y+pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y)/2];
				}
				if(m_ptend[0].y-pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y)
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].x=m_ptend[3].x+(float)(m_ptend[0].y-pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y)*(float)(m_ptend[0].x-m_ptend[3].x)/(float)(m_ptend[0].y-m_ptend[3].y);
				else
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].x=m_ptend[0].x;
				if(m_ptend[1].y-pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y)
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].x=m_ptend[2].x+(float)(m_ptend[1].y-pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y)*(float)(m_ptend[1].x-m_ptend[2].x)/(float)(m_ptend[1].y-m_ptend[2].y);
				else
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].x=m_ptend[1].x;
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength=(int)(temp+0.5);
			}
			else
			{
				//////printf("tempd is %d\t",temp);
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength=(int)(temp+0.5);

			}
			if(pParams->NormalDetectCfg.cSpeedDesc.percent<1e-6)
			{
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength=0;
			}
			////printf("uVehicleQueueLength is %d\t",pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength);
			//   //////printf("lineup[0]:[%d,%d],lineup[1]:[%d,%d]\n",LineUp[0].x,LineUp[0].y,LineUp[1].x,LineUp[1].y);
			//   //////printf("lineup[0] is: %d,%d\n",pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].x,pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y);
			////////printf("lineup[1] is: %d,%d\n",pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].x,pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y);
			//if (pCfgs->sptimes[i]>10)
			//{
			//	pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.AlarmLineflag=2;

			//	if(pCfgs->VehicleTeamSave[i]==1)
			//	{
			//		pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.bInfoValid=1;
			//		pCfgs->VehicleTeamSave[i] =2;
			//	}
			//}
			//else 
			//{
			//	pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.AlarmLineflag=1;
			//	if(pCfgs->VehicleTeamSave[i]==2)
			//	{
			//		pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.bInfoValid=1;
			//		pCfgs->VehicleTeamSave[i] =1;
			//	}
			//}
			//}
			//
			//////printf("calcu is over");


			//??缂侇垽鎷�?A??A??I??闁荤媴绠撻崹锟�?
			memcpy((void *)pSpeedDetect->puTempImage, (void *)pParams->puPointNewImage, pParams->NormalDetectCfg.cSpeedDesc.DetectDots);//A??A??缂侇垽鎷�?


			memcpy((void *)pCfgs->CameraLocalPara.dSpeedIdIndex[i].ExtStorageAddr, (void *)&pParams->NormalDetectCfg, sizeof(struct	cSpeedDetectStruct));
			//??缂侇垽鎷�?
			memcpy((void *)pSpeedDetect->BackImagePlusPointer, (void *)pParams->puBackImagePlus, pSpeedDetect->DetectDots);


			////20140708  queue
			memcpy((void *)pSpeedDetect->PreQueueImage, (void *)pParams->CurrQueueImage, pParams->NormalDetectCfg.cSpeedDesc.QueDetectDots);
			memcpy((void *)pSpeedDetect->BackQueueImage, (void *)pParams->BackQueueImage, pParams->NormalDetectCfg.cSpeedDesc.QueDetectDots);

		}
		//if(bPresenceEnable == TRUE)
		//{
			//reserved

		//}//bPresenceEnable END

		memcpy((void *)pCfgs->CameraLocalPara.LanesCfgEntry[i], (void *)&pParams->LaneParameter, sizeof(struct	LanesStruct));

	}//for ends
	/////0707 rate
	if((((pCfgs->CameraCfg.uEnvironmentStatus==1)&&(pCfgs->bAuto==1))
		||((pCfgs->CameraCfg.uEnvironmentStatus==3)&&(pCfgs->bAuto==2)))&&pCfgs->CameraCfg.LaneAmount)//
	{

		pSpeedDetect = pCfgs->CameraLocalPara.dSpeedIdIndex[0].ExtStorageAddr;
		memcpy((void *)pParams->FirstLaneImage, (void *)pSpeedDetect->FirstLaneImage, (pCfgs->light_width*pCfgs->light_height));//
		//////printf("cpy light detec area is ok");
		if(pCfgs->calctimes)
		{

			if(Existlight(pParams->FirstLaneImage,pCfgs->light_width,pCfgs->light_height,pCfgs,pParams)&&(pCfgs->uDetectInSum[0]!=pCfgs->carin))
			{
				pCfgs->existtimes++;
				pCfgs->carin=pCfgs->uDetectInSum[0];
			}
			if(pCfgs->uDetectOutSum[0]!=pCfgs->carout)
			{
				pCfgs->carout=pCfgs->uDetectOutSum[0];
				pCfgs->carcount++;
				pCfgs->calctimes=0;//0617xyx
				pCfgs->carin=pCfgs->uDetectInSum[0];
			}
		}
		//else
			//pCfgs->carin=pCfgs->uDetectInSum[1];
		if(pCfgs->gThisFrameTime%1500==1&&((pCfgs->CameraCfg.uEnvironmentStatus==1&&pCfgs->carcount)||pCfgs->CameraCfg.uEnvironmentStatus==3))
		{
			//pCfgs->guTestD[0]=pCfgs->carcount;
			//pCfgs->guTestD[1]=pCfgs->existtimes;
			if (pCfgs->carcount&&((float)pCfgs->existtimes/pCfgs->carcount>0.9))
				flag=1;
			else
				flag=0;

			if(pCfgs->exlight[5]==5)
			{
				for(i=0;i<4;i++)
				{
					pCfgs->exlight[i]=pCfgs->exlight[i+1];
				}
					pCfgs->exlight[4]=flag;
			}
			else
			{
				pCfgs->exlight[pCfgs->exlight[5]]=flag;
				pCfgs->exlight[5]++;
			}

			/*	if (pCfgs->carcount&&((float)pCfgs->existtimes/pCfgs->carcount>0.5)&&(!pCfgs->bNight))
			{
			pCfgs->bAuto=2;
			pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight =TRUE ;
			pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bDay =FALSE ;
			//	pCfgs->existtimes=0;
			pCfgs->bNight = TRUE;
			//	pCfgs->carcount=0;
			}
			else if (pCfgs->carcount&&((float)pCfgs->existtimes/pCfgs->carcount<0.2)&&pCfgs->bNight)
			{
			//	pCfgs->carcount=0;
			//	pCfgs->existtimes=0;
			pCfgs->bAuto=1;
			pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight =FALSE;
			pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bDay =TRUE;
			pCfgs->bNight = FALSE;
			}
			*/
			//615
			pCfgs->carcount=0;
			pCfgs->existtimes=0;
			//615
		}
		if(pCfgs->gThisFrameTime%1500==1&&pCfgs->exlight[5]>=3)
		{
			flag=0;
			for(i=0;i<5;i++)
			{
				if(pCfgs->exlight[i])
					flag++;
			}
			if(flag>=3&&(!pCfgs->bNight))
			{
				pCfgs->bAuto=2;
				pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight =TRUE ;
				pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bDay =FALSE ;
				pCfgs->bNight = TRUE;
			}
			else if(flag<3&&pCfgs->bNight)
			{
				pCfgs->bAuto=1;
				pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight =FALSE;
				pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bDay =TRUE;
				pCfgs->bNight = FALSE;
			}
		}

	}
	else if (pCfgs->CameraCfg.uEnvironmentStatus==2)
	{
		pCfgs->bAuto=2;
		pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight =TRUE ;
		pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bDay =FALSE ;
		pCfgs->bNight = TRUE;
		//pCfgs->guTestC[1]=pCfgs->bAuto;
	}
	else if(pCfgs->CameraCfg.uEnvironmentStatus==4)
	{
		pCfgs->bAuto=1;
		pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight =FALSE;
		pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bDay =TRUE;
		pCfgs->bNight = FALSE;
		//pCfgs->guTestC[2]=pCfgs->bAuto;
	}
	/////0707
	////20131217   zcj
	/*pParams->NormalDetectCfg.cSpeedDesc.track_flag++;
	if(pParams->NormalDetectCfg.cSpeedDesc.track_flag>=20) 
		pParams->NormalDetectCfg.cSpeedDesc.track_flag=0;*/


	pCfgs->ResultMsg.uMsgHeader.uFlag = 0x55;
	pCfgs->ResultMsg.uMsgHeader.uCmd = 0xE0E0;
	pCfgs->ResultMsg.uResultInfo.uEnvironmentStatus = pCfgs->bAuto; //added by david 20131014

	memcpy((void *)outBuf, (void *)&pCfgs->ResultMsg, outSize);
	return 1;

}
int sort_aver(int* array,int array_size,int num)
{
	int i=0,j=0;
	int temp;
	int array_temp[20]={0};
	for(i=0;i<array_size;i++)
	{
		array_temp[i]=array[i];
	}
	for(i=0;i<=num/2;i++)
	{
		for(j=i+1;j<num;j++)
		{
			if(array_temp[i]>array_temp[j])
			{
				temp=array_temp[i];
				array_temp[i]=array_temp[j];
				array_temp[j]=temp;
			}
		}
	}
	return array_temp[num/2];
}
void get_averback(ALGPARAMS *pParams,ALGCFGS *pCfgs)
{
	int aver_idx=0;
	int aver_back=0;
	int i=0;
	if(pCfgs->gThisFrameTime==0)
	{
		memset(pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back,0,10*sizeof(int));
		pParams->NormalDetectCfg.cSpeedDesc.aver_back=0;

	}
	if(pParams->NormalDetectCfg.cSpeedDesc.bGet1backflag&&pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[4])
	{
		if(pParams->NormalDetectCfg.cSpeedDesc.outflag%3000==0&&pParams->NormalDetectCfg.cSpeedDesc.m_uContiuCount>8)
		{
            memset(pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back,0,10*sizeof(int));
			pParams->NormalDetectCfg.cSpeedDesc.aver_back=0;
		}
		if(pCfgs->gThisFrameTime%100==0&&(pParams->NormalDetectCfg.cSpeedDesc.outflag==0||(pParams->NormalDetectCfg.cSpeedDesc.outflag>100&&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount==FALSE))&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.updateParams==FALSE)
		{
			pParams->NormalDetectCfg.cSpeedDesc.updateParams=TRUE;
			aver_back=AveGray(pParams->puBackImagePlus,pCfgs->m_iWidth,pCfgs->m_iHeight,0,pCfgs->m_iHeight,pCfgs);
			aver_idx=pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back[9];
			if(aver_idx<9)
			{
				pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back[aver_idx]=aver_back;
				pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back[9]+=1;
				if(aver_idx>4)
				{
					pParams->NormalDetectCfg.cSpeedDesc.aver_back=sort_aver(pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back,10,aver_idx);
				}
			}
			else
			{
				for(i=0;i<9;i++)
				{
					pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back[i]=pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back[i+1];
				}
				pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back[8]=aver_back;
			}
		}
		if(pCfgs->gThisFrameTime%1000==0)
		{
			aver_idx=pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back[9];
			aver_back=sort_aver(pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back,10,aver_idx);
			if(aver_idx>=9)
			{
				if(pParams->NormalDetectCfg.cSpeedDesc.aver_back>1)
				{
					if(abs(pParams->NormalDetectCfg.cSpeedDesc.aver_back-aver_back)>30)
					{
                        memset(pParams->NormalDetectCfg.cSpeedDesc.stat_aver_back,0,10*sizeof(int));
						pParams->NormalDetectCfg.cSpeedDesc.aver_back=0;
						}
					else
					{
						//pParams->NormalDetectCfg.cSpeedDesc.aver_back=0.9*pParams->NormalDetectCfg.cSpeedDesc.aver_back+0.1*aver_back;
						pParams->NormalDetectCfg.cSpeedDesc.aver_back = (pParams->NormalDetectCfg.cSpeedDesc.aver_back * 922 + aver_back * 102) >> 10;
					}
				}
				else
					pParams->NormalDetectCfg.cSpeedDesc.aver_back=aver_back;

			}
            else if(pParams->NormalDetectCfg.cSpeedDesc.aver_back>1&&aver_idx)
			{
				pParams->NormalDetectCfg.cSpeedDesc.aver_back=aver_back;
			}

		}
	}

}



void SpeedCaculate(Uint16 LaneID, Uint16 nChannelNum, Uint8* inBuf,  Int32 outSize,ALGCFGS *pCfgs, ALGPARAMS *pParams,CPoint LineUp[],Uint8* inBuf1,CPoint m_ptend[])
{
	//	int detectYposition=0;
	int standard_deviation=0;
	Uint16 uLines;
	Uint16 bSpeedCalculate = TRUE;
	SPEED_COUNTER_STATUS SpeedStatus;
	struct 	cSpeedDetectStruct *pSpeedDetect = NULL;
	//int txlposition1 = 0;
	//int txlposition2 = 0;
	//txl,201506
	int detectQAposition=0,detectQposition=0,detectTNposition=0,detectBNposition=0;//20150918
	//float tail_std=0.0,back_bottom_std=0;
	//int part_ave_height=0;
	//int part_start_line=0,part_stop_line=0;
	//bool part_has_car[8]={0};
	//int last_car_start_line=0,last_car_stop_line=0;
	//bool find_car_part=FALSE;
	float cur_part_std[10];
	int i=0,j=0;
	int calperiodTh=0;
	//bool bWuChe = TRUE;
	int Elements;
	float teamlength=0;
	//float threshold=0;
	int row=0,col=0;
	//unsigned char val=0;
	//bool bNonVehicle=FALSE;
	Uint8* inBufQ;
	bool cur_has_car=TRUE;
	int aver_curr=0,aver_back=0;
	//bool bNearLane=FALSE;
	//float std_fg=100;
	//bool isShadow=FALSE;
	bool bCarExit=FALSE;
	bool bBack=FALSE;
	int car_confidence=1;
    int x1=0,x2=0,x3=0;
    int vehicle_bottom=0;
	int detectId=0;
    bool bCarIn=FALSE,bCarExist=FALSE;
    int flag=0;
	if(pCfgs->visibility&&pCfgs->bNight==FALSE)
	{
		pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold =(pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold >10)? 10:pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold; 
		pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold1 = (pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold1>10)? 10:pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold1; 
		pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold2 = (pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold2 >10)? 10:pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold2;
		pCfgs->CameraCfg.uInThreshold=(pCfgs->CameraCfg.uInThreshold>10)? 10:pCfgs->CameraCfg.uInThreshold;
		pCfgs->CameraCfg.uSquareThreshold=(pCfgs->CameraCfg.uSquareThreshold>15)? 15:pCfgs->CameraCfg.uSquareThreshold ;
		pCfgs->CameraCfg.uSobelThreshold=(pCfgs->CameraCfg.uSobelThreshold>3)? 3:pCfgs->CameraCfg.uSobelThreshold;
		pCfgs->CameraCfg.guSubImageThreshold=(pCfgs->CameraCfg.guSubImageThreshold>3)? 3:pCfgs->CameraCfg.guSubImageThreshold;
		car_confidence=0;

	}
	//if(pCfgs->CameraCfg.uEnvironmentStatus==1||pCfgs->CameraCfg.uEnvironmentStatus==3)
	//{
		car_confidence=0;
	//}
	inBufQ = pParams->CurrQueueImage;
	uLines = pParams->NormalDetectCfg.cSpeedDesc.DetectLines;
	Elements = pParams->NormalDetectCfg.cSpeedDesc.DetectDots;

	pCfgs->m_iHeight = pParams->NormalDetectCfg.cSpeedDesc.DetectLines;
	pCfgs->m_iWidth = pParams->NormalDetectCfg.cSpeedDesc.DetectColumns;
	//TRACE("frame is:%d\n",frame);
	pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight = pCfgs->bNight;
	detectQposition=(m_ptend[4].y+m_ptend[5].y)/2-(m_ptend[0].y+m_ptend[1].y)/2;//20150918
	detectQAposition=(m_ptend[6].y+m_ptend[7].y)/2-(m_ptend[0].y+m_ptend[1].y)/2;//20150918
	detectTNposition=(m_ptend[8].y+m_ptend[9].y)/2-(m_ptend[0].y+m_ptend[1].y)/2;//20150918
	detectBNposition=(m_ptend[10].y+m_ptend[11].y)/2-(m_ptend[0].y+m_ptend[1].y)/2;//20150918
	////printf("detectposition calcu is ok");
	//if(pCfgs->gThisFrameTime%100==1)
	//{
	////////printf("is Night?:%d\n",pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight);
	////////printf("lineID=%d,[%d,%d],[%d,%d],[%d,%d],[%d,%d]\n",LaneID,m_ptend[4].x,m_ptend[4].y,m_ptend[5].x,m_ptend[5].y,m_ptend[3].x,m_ptend[3].y,m_ptend[2].x,m_ptend[2].y);
	//}


	////20131218  
	if(pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight==FALSE)  ////day
	{
		//detectYposition=(m_ptend[6].y+m_ptend[7].y)/2-(m_ptend[0].y+m_ptend[1].y)/2;//txl,20150311
		//part_ave_height=(int)(pCfgs->team_height-detectYposition-pCfgs->m_iHeight)/3;
		//pCfgs->guTestD[0]=m_ptend[6].y;
		//pCfgs->guTestD[1]=m_ptend[7].y;
		//pCfgs->guTestD[2]=part_ave_height;
		//pCfgs->guTestD[3]=detectYposition;
		//txl,201506
		//if(pCfgs->gThisFrameTime%10==1)
		//{
		//   //////printf("part_ave_height is :%d\n",part_ave_height);
		//   //////printf("detectYposition is :%d\n",detectYposition);
		//}

		//SobelCalculate(inBuf,pParams->puPointNextImage,100,pCfgs->m_iWidth,pCfgs->m_iHeight);
		ShallowLightRemovePreProcess( inBuf, pParams->puPointSobelImage, pParams->puSobelTempImageX, \
			pParams->puSobelTempImageY, &(pParams->NormalDetectCfg.cSpeedDesc),nChannelNum, pCfgs, pParams );
		//////printf("shallow is ok");
		pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged = bIsSpeedCounterChanged( pParams->puPointSobelImage, \
			pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterStartRow, \
			&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus, \
			0, pCfgs, pParams);	
		//////printf(" bSpeedCounterChanged is ok");

		/////20131217 bIsSpeedCounterChanged1
		pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1 = bIsSpeedCounterChanged1( pParams->puPointSobelImage, \
			SPEED_STARTER_ENDLINE, \
			&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus,\
			0, LaneID,pCfgs, pParams );
        calperiodTh=200;
		//calperiodTh=pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag==TRUE?6000:200;
		//	updateMinStd(pParams->CurrQueueImage,0,pCfgs->team_width,pCfgs->team_height,0,pCfgs->team_height,pParams,calperiodTh);
		//	updateMinStd(pParams->CurrQueueImage,1,pCfgs->team_width,pCfgs->team_height,0,detectQAposition,pParams,calperiodTh);//20150918
		//	updateMinStd(pParams->CurrQueueImage,2,pCfgs->team_width,pCfgs->team_height,detectQAposition,detectTNposition,pParams,calperiodTh);//20150918
		updateMinStd(pParams->CurrQueueImage,4,pCfgs->team_width,pCfgs->team_height,detectTNposition,pCfgs->team_height,pCfgs,pParams,calperiodTh);//20150918
		//updateMinStd(pParams->CurrQueueImage,5,pCfgs->team_width,pCfgs->team_height,detectTNposition,pCfgs->team_height,pCfgs,pParams,calperiodTh);
		//updateMinStd(pParams->CurrQueueImage,4,pCfgs->team_width,pCfgs->team_height,detectQposition,detectQAposition,pParams,calperiodTh);//20150918
        //calperiodTh=pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[9]==TRUE? 3000:400;
		updateMinStd(inBuf,9,pCfgs->m_iWidth,pCfgs->m_iHeight,0,pCfgs->m_iHeight,pCfgs,pParams,calperiodTh);
		//////printf("updateMinStd is ok");
		//	if (pParams->NormalDetectCfg.cSpeedDesc.yellowflag>0
		//		&&pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[0]==TRUE&&(pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[0]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[0])<5
		//		&&pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[1]==TRUE&&(pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[1]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[1])<5
		//		&&pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[2]==TRUE&&(pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[2]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[2])<5
		//		&&pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[3]==TRUE&&(pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[3]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[3])<5
		//		&&pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[4]==TRUE&&(pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[4])<5)//0930
		//	{
		//		//pParams->NormalDetectCfg.cSpeedDesc.min_tail_std=pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[1];
		//		memcpy((void *)pParams->BackQueueImage,(void *)pParams->CurrQueueImage,pCfgs->team_width*pCfgs->team_height);//pSpeedDetect->puTempImage

		//		pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag=TRUE;

		//	/*	pParams->NormalDetectCfg.cSpeedDesc.ini_min_std[0]=pParams->NormalDetectCfg.cSpeedDesc.min_part_std[0];
		//		pParams->NormalDetectCfg.cSpeedDesc.ini_min_std[1]=pParams->NormalDetectCfg.cSpeedDesc.min_part_std[1];
		//		pParams->NormalDetectCfg.cSpeedDesc.ini_min_std[9]=pParams->NormalDetectCfg.cSpeedDesc.min_part_std[9];*/

		//	}	

		cur_part_std[9]=pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[9];



	} 
	//if(pCfgs->gThisFrameTime%100==1)
	//{
	////////printf("getqueueback_flag is:%d\n",pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag);
	////////printf("jgtime:%d\n",pCfgs->CameraCfg.uGetBackMinContiuFrame);
	//}
	//
	pParams->LaneParameter.TotalFrameCount++;

	// 
	if(pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight ==TRUE )
	{	 	////printf("########night######## \n" );fflush(NULL);
		////printf("start nightlihtdetect!\t");
		pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag=TRUE;
		NightLightDetect(inBuf,pParams->puPointSobelImage,pParams->puPointNewImage,pCfgs,pParams);
		////printf("finish nightlihtdetect!\t");
		ProjectSubAndSobelEdge(pParams->puPointSobelImage,pCfgs,pParams);
		//pCfgs->guTestB[4]= pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold;
		//pCfgs->guTestB[5]=pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold1;
		//pCfgs->guTestB[6]=pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold2;
		////20140707  
		/*if (pCfgs->gThisFrameTime%25==0)
		{*/	
		////printf("star lengthdetect!\t");
		teamlength=100*TeamLength1(pParams->CurrQueueImage,pParams->PreQueueImage,pCfgs,pParams,detectQposition,detectQAposition);//20150918,txl
		/*if (pCfgs->gThisFrameTime<200)
		{
		pParams->NormalDetectCfg.cSpeedDesc.percent=1;
		}
		else if (pCfgs->gThisFrameTime%200==0)
		{
		pParams->NormalDetectCfg.cSpeedDesc.percent=pParams->NormalDetectCfg.cSpeedDesc.percent*0.9;
		}*/
		////////printf("teamlength is :%f,",teamlength);
		//detectQAposition=(m_ptend[6].y+m_ptend[7].y)/2-(m_ptend[0].y+m_ptend[1].y)/2;//txl,20150311
		//detectYposition=(m_ptend[6].y+m_ptend[7].y)/2;//1203
		////////printf("detectYposition is :%d,",detectYposition);
		////////printf("Lastpos is :%d,",pParams->NormalDetectCfg.cSpeedDesc.Lastpos);
		///////////printf("Is Car In Tail? :%d\n,",pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail);+
		//if (pParams->NormalDetectCfg.cSpeedDesc.Lastpos<detectQAposition)
		//{
		//	pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail=TRUE;
		//}
		//else
		//	pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail=FALSE;
		//}  



	}
	else
	{
		//txl,201506
		/*if(pParams->NormalDetectCfg.cSpeedDesc.bGet1backflag)
		{

		if(pCfgs->gThisFrameTime%1000==0)
		{
		if(pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[2]>0)
		{
		pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[1]/=pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[2];
		pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[0]=pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[0]*0.9+pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[1]*0.1;
		}
		pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[1]=0;
		pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[2]=0;
		}
		if(pCfgs->gThisFrameTime%100==0)
		{
		aver_back=AveGray(pParams->puBackImagePlus,pCfgs->m_iWidth,pCfgs->m_iHeight,0,pCfgs->m_iHeight,pCfgs);
		pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[1]+=aver_back;
		pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[2]+=1;
		}
		}*/
		get_averback(pParams,pCfgs);
        if(LaneID==0)
		{
			pCfgs->guBackGrey=(int)pParams->NormalDetectCfg.cSpeedDesc.aver_back;
		}
		if ((cur_part_std[9]!=0)&&(cur_part_std[9]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[9])<10/*&&((pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[5]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[5])<10)*/
			&&cur_part_std[9]<50&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged == FALSE)&&(bIsSobleBlockUnChanged( pParams->puPointSobelImage, SPEED_STARTER_STARTLINE, SPEED_STARTER_ENDLINE ,0, pCfgs, pParams)))//(pParams->NormalDetectCfg.cSpeedDesc.yellowflag>5)&&
		{
			if(pCfgs->CameraCfg.uEnvironmentStatus==4)
			{
				if(pParams->NormalDetectCfg.cSpeedDesc.yellowflag==0)//20151203add 
				{
					aver_curr=AveGray(inBuf,pCfgs->m_iWidth,pCfgs->m_iHeight,0,pCfgs->m_iHeight,pCfgs); 
				}
				if(pParams->NormalDetectCfg.cSpeedDesc.aver_back>1)
				{
					if(pCfgs->visibility==FALSE)
					{
						bBack=((pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[4])<10&&pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[4]&&pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]<60&&pParams->NormalDetectCfg.cSpeedDesc.outflag>10&&abs(aver_curr-pParams->NormalDetectCfg.cSpeedDesc.aver_back)<20)||(pParams->NormalDetectCfg.cSpeedDesc.yellowflag>0);
					}
					else
					{
						bBack=((pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[4])<10&&pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[4]&&pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]<35&&pParams->NormalDetectCfg.cSpeedDesc.outflag>10&&abs(aver_curr-pParams->NormalDetectCfg.cSpeedDesc.aver_back)<20)||(pParams->NormalDetectCfg.cSpeedDesc.yellowflag>0);
					}

				}
				else
				{

					if(pCfgs->visibility==FALSE)
					{
						bBack=((pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[4])<10&&pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]<60&&pParams->NormalDetectCfg.cSpeedDesc.yellowflag==0)||(pParams->NormalDetectCfg.cSpeedDesc.yellowflag>0);
					}
					else
					{
						bBack=((pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[4])<10&&pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]<35&&pParams->NormalDetectCfg.cSpeedDesc.yellowflag==0)||(pParams->NormalDetectCfg.cSpeedDesc.yellowflag>0);
					}

				}
				if(bBack)
				{
					//pParams->NormalDetectCfg.cSpeedDesc.nullflag=0;
					memcpy((void *)pParams->puBackImagePlus,(void *)inBuf , Elements);//pSpeedDetect->puTempImage
					pParams->NormalDetectCfg.cSpeedDesc.bGet1backflag=TRUE;
					//////printf(" getbackflag is true");
					pParams->NormalDetectCfg.cSpeedDesc.bGetBackFlag=1;
					cur_has_car=FALSE;
				}
			}	
			else
			{
				if(((pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[4])<15&&pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]<50&&pParams->NormalDetectCfg.cSpeedDesc.yellowflag==0)||pParams->NormalDetectCfg.cSpeedDesc.yellowflag>0)
					//pParams->NormalDetectCfg.cSpeedDesc.yellowflag = 0;
				{
					//pParams->NormalDetectCfg.cSpeedDesc.nullflag=0;
					memcpy((void *)pParams->puBackImagePlus,(void *)inBuf , Elements);//pSpeedDetect->puTempImage
					pParams->NormalDetectCfg.cSpeedDesc.bGet1backflag=TRUE;
					//////printf(" getbackflag is true");
					pParams->NormalDetectCfg.cSpeedDesc.bGetBackFlag=1;
					cur_has_car=FALSE;
				}
			}
		}
		////printf("laneID=%d,cur_std=%f,min_std=%f,cur_std[4]=%f,min_tail_std=%f,bBack=%d\n",LaneID,pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[9],pParams->NormalDetectCfg.cSpeedDesc.min_part_std[9],pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4],pParams->NormalDetectCfg.cSpeedDesc.min_tail_std,bBack);
		pParams->NormalDetectCfg.cSpeedDesc.Headposition=pCfgs->team_height;
		flag=0;

		//printf("\n\n %d,%d\n",m_ptend[8].x,m_ptend[8].y);
		//printf("\n\n %d,%d\n",m_ptend[9].x,m_ptend[9].y);
		//printf("\n\n %d,%d\n",m_ptend[10].x,m_ptend[10].y);
		//printf("\n\n %d,%d\n",m_ptend[11].x,m_ptend[11].y);

		if(pCfgs->targets_size)
		{
			for(j=0;j<pCfgs->targets_size;j++)
			{
			    vehicle_bottom=(pCfgs->targets[j].vehicle.y+pCfgs->targets[j].vehicle.height)*2;
				if(vehicle_bottom*2>(m_ptend[8].y+m_ptend[9].y))
				{
					x1=(float)(vehicle_bottom-m_ptend[0].y)*(float)(m_ptend[10].x-m_ptend[0].x)/(float)(m_ptend[10].y-m_ptend[0].y)+m_ptend[0].x;
					x2=(float)(vehicle_bottom-m_ptend[1].y)*(float)(m_ptend[11].x-m_ptend[1].x)/(float)(m_ptend[11].y-m_ptend[1].y)+m_ptend[1].x;
					x3=min(x2,(pCfgs->targets[j].vehicle.x+pCfgs->targets[j].vehicle.width)*2)-max(x1,2*pCfgs->targets[j].vehicle.x);




					if(x3>0&&((float)x3/(pCfgs->targets[j].vehicle.width*2)>0.5||(vehicle_bottom*2>(detectTNposition+detectBNposition)&&((float)x3/(pCfgs->targets[j].vehicle.width*2)>0.3))))
					{					
						if(pCfgs->targets[j].flow_count==false&&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount==FALSE&&vehicle_bottom*2<(m_ptend[10].y+m_ptend[11].y))
						{
							bCarIn=true;
							pCfgs->targets[j].flow_count=true;
							pParams->NormalDetectCfg.cSpeedDesc.DetectId=pCfgs->targets[j].vehicle_id;
						}
					}
				}
				if(pCfgs->targets[j].vehicle_id==pParams->NormalDetectCfg.cSpeedDesc.DetectId)
				{
					flag=1;
					bCarExist=true;
				}
				x1=(float)(pCfgs->targets[j].vehicle.y*2+pCfgs->targets[j].vehicle.height-m_ptend[0].y)*(float)(m_ptend[3].x-m_ptend[0].x)/(float)(m_ptend[3].y-m_ptend[0].y)+m_ptend[0].x;
				x2=(float)(pCfgs->targets[j].vehicle.y*2+pCfgs->targets[j].vehicle.height-m_ptend[1].y)*(float)(m_ptend[2].x-m_ptend[1].x)/(float)(m_ptend[2].y-m_ptend[1].y)+m_ptend[1].x;
				x3=min(x2,(pCfgs->targets[j].vehicle.x+pCfgs->targets[j].vehicle.width)*2)-max(x1,2*pCfgs->targets[j].vehicle.x);
				if(x3>0&&(float)x3/(pCfgs->targets[j].vehicle.width*2)>0.5)
				{
					if(pCfgs->targets[j].vehicle.y*2<pParams->NormalDetectCfg.cSpeedDesc.Headposition)
					{
						pParams->NormalDetectCfg.cSpeedDesc.Headposition=pCfgs->targets[j].vehicle.y*2;
					}
				}

			}
		}
		if(flag==0)
		{
			bCarExit=true;
		}
		//if(pParams->NormalDetectCfg.cSpeedDesc.bGetBackFlag==TRUE)
		//{
		//back_bottom_std=StandardDeviation (pParams->puBackImagePlus,pCfgs->m_iWidth,pCfgs->m_iHeight,0.6*pCfgs->m_iHeight,pCfgs->m_iHeight,1);//txl,0930
		//}
		if (pCfgs->DownSpeedCfg.SpeedEachLane[0].ptCornerQ[0].x!=0)
		{
			iSubStractImage(pParams->CurrQueueImage,pParams->PreQueueImage,3, 0, pCfgs->team_height,pCfgs->team_width,pCfgs->team_height);//????PreQueueImage
			pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[0]=StandardDeviation (pParams->PreQueueImage,pCfgs->team_width,pCfgs->team_height,0,pCfgs->team_height/4,1);
			pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[1]=StandardDeviation (pParams->PreQueueImage,pCfgs->team_width,pCfgs->team_height,pCfgs->team_height/4,pCfgs->team_height/2,1);
			pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[2]=StandardDeviation (pParams->PreQueueImage,pCfgs->team_width,pCfgs->team_height,pCfgs->team_height/2,pCfgs->team_height*3/4,1);
			pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[3]=StandardDeviation (pParams->PreQueueImage,pCfgs->team_width,pCfgs->team_height,pCfgs->team_height*3/4,pCfgs->team_height,1);

			//20151226
			if (pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[0]>10)
			{
				pParams->NormalDetectCfg.cSpeedDesc.cur_maxstd_pos=0;
			}
			else if (pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[1]>10)
			{
				pParams->NormalDetectCfg.cSpeedDesc.cur_maxstd_pos=1;
			}
			else if (pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[2]>10)
			{
				pParams->NormalDetectCfg.cSpeedDesc.cur_maxstd_pos=2;
			}
			else if (pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[3]>10)
			{
				pParams->NormalDetectCfg.cSpeedDesc.cur_maxstd_pos=3;
			}
			else 
			{
				pParams->NormalDetectCfg.cSpeedDesc.cur_maxstd_pos=4;
			}
			if (pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[0]<5&&
				pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[1]<5&&
				pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[2]<5&&
				pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[3]<5)
			{
				pParams->NormalDetectCfg.cSpeedDesc.nullflag++;
			} 
			else
			{
				pParams->NormalDetectCfg.cSpeedDesc.nullflag=0;
			}
			if ((/*pCfgs->up_visib_value>VISIB_LENGTH&&*/pCfgs->visibility==FALSE)&&pParams->NormalDetectCfg.cSpeedDesc.nullflag>5&&pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[4]==TRUE
				&&(pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[4])<10
				&&((pParams->NormalDetectCfg.cSpeedDesc.cur_maxstd_pos==4&&pParams->NormalDetectCfg.cSpeedDesc.last_maxstd_pos==3)||(pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.Tailposition>3000)))//txl,20160110pParams->NormalDetectCfg.cSpeedDesc.not_up_back->pParams->NormalDetectCfg.cSpeedDesc.Tailposition


			{
				memcpy((void *)pParams->BackQueueImage,(void *)pParams->CurrQueueImage,pCfgs->team_width*pCfgs->team_height);//
				pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag=TRUE;//
				memcpy((void *)pParams->PreQueueImage,(void *)pParams->CurrQueueImage,pCfgs->team_width*pCfgs->team_height);
				pParams->NormalDetectCfg.cSpeedDesc.LastCarposition=pCfgs->team_height;
				pParams->NormalDetectCfg.cSpeedDesc.linenum=0;
				pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail=FALSE;
				pParams->NormalDetectCfg.cSpeedDesc.percent=0;
				pParams->NormalDetectCfg.cSpeedDesc.Tailposition=0;//txl,20160110
			}
			else
			{
				//txl,201506
				if(pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag==FALSE)
				{
					memset((void *)pParams->BackQueueImage,0,sizeof(unsigned char) * pCfgs->team_width *pCfgs->team_height);
				}
				pParams->NormalDetectCfg.cSpeedDesc.Tailposition++;
				memset((void *)pParams->PreQueueImage,0,sizeof(unsigned char) * pCfgs->team_width *pCfgs->team_height);
				memcpy((void *)pParams->PreQueueImage,(void *)pParams->BackQueueImage,pCfgs->team_width*pCfgs->team_height);//
				iSubStractImage(pParams->CurrQueueImage,pParams->PreQueueImage,3, 0, pCfgs->team_height,pCfgs->team_width,pCfgs->team_height);//	
				if(pCfgs->visibility||(pCfgs->guBackGrey<50&&pCfgs->guBackGrey))
					SobelCalculate(pParams->CurrQueueImage,pParams->SobelQueueImage,50,pCfgs->team_width,pCfgs->team_height);//SobelQueueImage???????
				else
				    SobelCalculate(pParams->PreQueueImage,pParams->SobelQueueImage,100,pCfgs->team_width,pCfgs->team_height);

				//////printf(" sobelcalculate ok");

				SobelConectivity(pParams->SobelQueueImage,pParams->PreQueueImage,pCfgs, pParams,detectQposition,detectQAposition);//20150918	
				memcpy((void *)pParams->PreQueueImage,(void *)pParams->CurrQueueImage,pCfgs->team_width*pCfgs->team_height);//prePreQueueImage
				if(pCfgs->up_visib_value>VISIB_LENGTH&&pCfgs->visibility==TRUE)
				{  
					pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag=TRUE;
				}
			}
			//20151226
			if (pParams->NormalDetectCfg.cSpeedDesc.last_maxstd_pos<4)
			{
				pParams->NormalDetectCfg.cSpeedDesc.last_maxstd_pos=pParams->NormalDetectCfg.cSpeedDesc.cur_maxstd_pos<4?pParams->NormalDetectCfg.cSpeedDesc.cur_maxstd_pos:pParams->NormalDetectCfg.cSpeedDesc.last_maxstd_pos;
			}



		}

	}

	// parse the result, check the obj, calc the speed and length
	//
	if( pCfgs->CameraCfg.bCameraWorkInCross == 1 )//?A???????: ???????????a
	{


	}
	else if(pCfgs->CameraCfg.bCameraWorkInCross == 2) //?A??: ?????????a,3I???I?A?I缂侇垽鎷�????a
	{
		while( bSpeedCalculate == TRUE )
		{
			bSpeedCalculate = FALSE;

			if( pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight == FALSE )
			{

				if( pParams->NormalDetectCfg.cSpeedDesc.uCountIntervalFrame < pCfgs->guMinCountIntervalFrame )
				{
					pParams->NormalDetectCfg.cSpeedDesc.uCountIntervalFrame++;
					break;
				}
				else if( pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount == FALSE )
				{

					//pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1 = bIsSpeedCounterChanged1( pParams->puPointSobelImage, \
					//SPEED_STARTER_ENDLINE, \
					//&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus,\
					//0,LaneID, pCfgs, pParams );
					////////printf("changed1 is :%d\n",pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1);
					////////printf("bIsSobleBlockUnChanged is :%d\n",bIsSobleBlockUnChanged( pParams->puPointSobelImage, SPEED_STARTER_STARTLINE, SPEED_STARTER_ENDLINE ,0, pCfgs, pParams));
					//if(pCfgs->CameraCfg.uEnvironmentStatus==1&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1&&pCfgs->visibility==FALSE)
					//{
						//bNonVehicle=remove_nonvehicle(pParams->CurrQueueImage,pParams->BackQueueImage,detectTNposition,pCfgs,pParams,0);
						/*if(bNonVehicle==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1 ==TRUE)
						{
						bNonVehicle=remove_nonvehicle1(pParams->CurrQueueImage,pParams->BackQueueImage,detectTNposition,pCfgs,pParams);
						}*/
					//}
					if(/*( bIsSobleBlockUnChanged( pParams->puPointSobelImage, SPEED_STARTER_STARTLINE, SPEED_STARTER_ENDLINE ,0, pCfgs, pParams) == FALSE )
					   &&*//*(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1 ==TRUE&&(bNonVehicle==FALSE)*/bCarIn
					   )
					{
						//if(pParams->NormalDetectCfg.cSpeedDesc.bGet1backflag&&pCfgs->CameraCfg.uEnvironmentStatus==4&&pCfgs->visibility==FALSE)
						//{
							//isShadow=remove_shadow(inBuf,pParams->puBackImagePlus,pCfgs,10);
						//}
						//if(isShadow==FALSE)
						//{
                        if(LaneID==0)
						     pCfgs->calctimes=1;
							pParams->NormalDetectCfg.cSpeedDesc.lxtime++;
                            pParams->NormalDetectCfg.cSpeedDesc.m_uContiuCount++;
						//}
						//else
						//{
						//	pParams->NormalDetectCfg.cSpeedDesc.lxtime=0;
						//	pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=0;
						//	pParams->NormalDetectCfg.cSpeedDesc.calarflag = 0;
						//	pParams->NormalDetectCfg.cSpeedDesc.yellowflag++;
						//	break;
						//}
					}
					else
					{  
						pParams->NormalDetectCfg.cSpeedDesc.lxtime=0;
						//pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=0;
						//pParams->NormalDetectCfg.cSpeedDesc.calarflag = 0;
						//pParams->NormalDetectCfg.cSpeedDesc.yellowflag++;
						//break;

					} 
					if(pParams->NormalDetectCfg.cSpeedDesc.lxtime>0)
					{
						if(pCfgs->gThisFrameTime-pParams->NormalDetectCfg.cSpeedDesc.jgtime<pCfgs->CameraCfg.uGetBackMinContiuFrame)
							break;
						else
						{
							pParams->NormalDetectCfg.cSpeedDesc.headtime=(pCfgs->gThisFrameTime-pParams->NormalDetectCfg.cSpeedDesc.jgtime)/15;//frame 15
							//pCfgs->thresholdValue=pParams->NormalDetectCfg.cSpeedDesc.headtime;
							pParams->NormalDetectCfg.cSpeedDesc.jgtime=pCfgs->gThisFrameTime;
						}
						pParams->NormalDetectCfg.cSpeedDesc.updateParams=FALSE;
						pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount = TRUE;
						pParams->NormalDetectCfg.cSpeedDesc.calarflag = 1;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=1;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.car_in++;
						pCfgs->uDetectInSum[LaneID]++;
						pParams->NormalDetectCfg.cSpeedDesc.speedflag = 1;
						pParams->NormalDetectCfg.cSpeedDesc.firstline = pCfgs->gThisFrameTime;
						pParams->NormalDetectCfg.cSpeedDesc.lengthflag = 1;
						//pParams->NormalDetectCfg.cSpeedDesc.secondline = pCfgs->gThisFrameTime;
						pParams->NormalDetectCfg.cSpeedDesc.yellowflag = 0;
                        if(pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting==FALSE)
						{
                            pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting=TRUE;
						}

						pCfgs->ResultMsg.uMsgHeader.uFlag = 0x55;
						pCfgs->ResultMsg.uMsgHeader.uCmd = 0xE0E0;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].uLaneID = LaneID;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bInfoValid = 1; 	
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bVehicleSta = 1;
					}
					//else
					//{		
						//pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=0;
						//pParams->NormalDetectCfg.cSpeedDesc.calarflag = 0;
						//pParams->NormalDetectCfg.cSpeedDesc.yellowflag++;
						//break;
					//}

				}
                if(pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount)
				{
					if(bCarExit)
					{
						pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount=FALSE;
                        pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.car_out++;
						pCfgs->uDetectOutSum[LaneID]++;
						pParams->NormalDetectCfg.cSpeedDesc.uTrafficFlow++;
						pCfgs->ResultMsg.uMsgHeader.uFlag = 0x55;
						pCfgs->ResultMsg.uMsgHeader.uCmd = 0xE0E0;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].uLaneID = LaneID;
					    pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed = (float)(pParams->NormalDetectCfg.cSpeedDesc.uTransFactor) / ((pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) * 0.067) * 3.6;//15 frame
						if (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed > 60)
						{
							pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed =30+rand()%20;// 100 - (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed - 100) * (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed - 100) / 100;
						}
						else if(pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed<15)
							pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed =15+rand()%20;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleSpeed \
							=((unsigned int) pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed);

						pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = pParams->NormalDetectCfg. \
							cSpeedDesc.wVehicleSpeed * (pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) * 0.067 / 3.6 - (float)(pParams->NormalDetectCfg.cSpeedDesc.uTransFactor);//15 frame
						if( (pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength > 4) \
							&&(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength < 7) )
						{
							pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = 4 + \
								(7 - pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength) * \
								(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength - 4) / 4;
						}
						else if (pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength > 7)
						{
							pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = rand() % 3 + 4;
						}
						else if(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength<3)
						{
							pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength=rand() % 2+3;
						}
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleLength  \
							= ((unsigned int )pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength);
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bInfoValid = 1; 
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bVehicleSta = 0;
					}

				}



				//sobel_rectify(pParams->puPointNextImage,pParams->puPointBackSobelImage,pCfgs);	

				if(pParams->NormalDetectCfg.cSpeedDesc.bGetBackFlag==TRUE)
				{
					pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged2 = bIsSpeedCounterChanged3( pParams->puPointNextImage,pParams->puPointSobelImage, \
						inBuf, \
						pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterStartRow, \
						pParams->puBackImagePlus, pCfgs, pParams );
					//pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged3 = bIsSpeedCounterChanged3( pParams->puPointNextImage,pParams->puPointSobelImage, \
						inBuf,0,pParams->puBackImagePlus, pCfgs, pParams );
				}
				else	
				{
					pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged2 = bIsSpeedCounterChanged2(pParams->puPointNextImage , pParams->puPointSobelImage, \
						pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterStartRow, \
						&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus, \
						pCfgs, pParams );
					//pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged3 = bIsSpeedCounterChanged2(pParams->puPointNextImage , pParams->puPointSobelImage, \
						0,&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus, pCfgs, pParams );
				}

				//txlposition1 = pParams->NormalDetectCfg.cSpeedDesc.position1;
				//txlposition2 = pParams->NormalDetectCfg.cSpeedDesc.position2;
				if((pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged2||bCarExist)&&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting)
				{
					pParams->NormalDetectCfg.cSpeedDesc.outflag++;
				}
				else 
				{
                    pParams->NormalDetectCfg.cSpeedDesc.m_uContiuCount=0;
					pParams->NormalDetectCfg.cSpeedDesc.yellowflag++;
					pParams->NormalDetectCfg.cSpeedDesc.calarflag = 0;
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=0;
					pParams->NormalDetectCfg.cSpeedDesc.outflag=0;
					pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting=FALSE;

				}


				//?a???3?3I缂佹唻鎷�???3I?D?
				//if( pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount == TRUE )
				//{

				//	if(pCfgs->CameraCfg.uEnvironmentStatus==1)
				//		bNonVehicle=remove_nonvehicle(pParams->CurrQueueImage,pParams->BackQueueImage,detectTNposition,pCfgs,pParams,1);
				//	//3I???D??
				//	if( (pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1 == TRUE) &&
				//		(pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting == FALSE) )
				//	{
				//		//now, vehicle has come in

				//		//??3I缂侇垽鎷�??D??
				//		if( (pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bLeftDestroyed == TRUE) &&
				//			(pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bRightDestroyed == TRUE) ){
				//				pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting = TRUE;
				//				pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum = 1;
				//				pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame = 0;
				//		}
				//		else if( pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bLeftDestroyed == TRUE )
				//		{

				//			SpeedStatus = GetSpeedCounterStatus( LEFT_LANE_COUNTER, pParams->LaneParameter.LaneId, \
				//				nChannelNum, pCfgs, pParams );
				//			if( (SpeedStatus.bIsCounting == FALSE) ||
				//				((SpeedStatus.bIsCounting == TRUE) && (SpeedStatus.bRightDestroyed == FALSE)) )
				//			{
				//				pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting = TRUE;
				//				pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum = 1;
				//				pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame = 0;
				//			}
				//		}
				//		else if( pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bRightDestroyed == TRUE )
				//		{
				//			SpeedStatus = GetSpeedCounterStatus( RIGHT_LANE_COUNTER, pParams->LaneParameter.LaneId, \
				//				nChannelNum, pCfgs, pParams );
				//			if( (SpeedStatus.bIsCounting == FALSE) ||
				//				((SpeedStatus.bIsCounting == TRUE) && (SpeedStatus.bLeftDestroyed == FALSE)) )
				//			{
				//				pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting = TRUE;
				//				pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum = 1;
				//				pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame = 0;
				//			}
				//		}
				//		else

				//		{
				//			pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting = TRUE;
				//			pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum = 1;
				//			pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame = 0;
				//		}

				//		//by david 20131105, clear outflag, outflag starts to increase from now on.
				//		//pParams->NormalDetectCfg.cSpeedDesc.outflag = 0;


				//		if(nChannelNum == 0 || nChannelNum == 1)
				//		{
				//			if(pCfgs->visibility==FALSE)
				//			{
				//				memcpy((void *)pParams->puSobelTempImageY,(void*)pParams->PreQueueImage,pCfgs->team_width*pCfgs->team_height);
				//				if(pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag)
				//				{
				//					threshold=AveGray(pParams->puBackImagePlus,pCfgs->m_iWidth,pCfgs->m_iHeight,0,pCfgs->m_iHeight,pCfgs);
				//					threshold=threshold/4;
				//					threshold=(threshold>=5)? threshold:5;
				//					threshold=(threshold<=25)? threshold:25;
				//					for(row=detectTNposition-51;row<=detectTNposition+11;row++)
				//					{
				//						for(col=0;col<pCfgs->team_width;col++)
				//						{
				//							val=abs(GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,pParams->CurrQueueImage)-GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,pParams->BackQueueImage));
				//							val=(val>threshold)? val:0;
				//							SetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,val,pParams->puSobelTempImageY);
				//						}
				//					}
				//				}
				//				pParams->NormalDetectCfg.cSpeedDesc.is_bicycle=remove_bicycle(pParams->puSobelTempImageY,pParams->puPointSobelImage,pParams->SobelQueueImage,detectTNposition-50,detectTNposition+10,pCfgs,pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag);
				//			}
				//			else
				//			{
				//				pParams->NormalDetectCfg.cSpeedDesc.is_bicycle=FALSE;
				//			}
				//			//pParams->NormalDetectCfg.cSpeedDesc.is_bicycle=FALSE;
				//			pCfgs->guTestB[10]=pParams->NormalDetectCfg.cSpeedDesc.is_bicycle;
				//			if(pParams->NormalDetectCfg.cSpeedDesc.is_bicycle==FALSE)
				//			{
				//				pCfgs->uDetectInSum[LaneID]++;
				//				pParams->NormalDetectCfg.cSpeedDesc.calarflag = 1;
				//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=1;
				//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.car_in++;
				//				pParams->NormalDetectCfg.cSpeedDesc.outflag=0;
				//			}
				//			else if(/*pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag==FALSE&&*/pParams->NormalDetectCfg.cSpeedDesc.outflag >=10)                    {
				//				pParams->NormalDetectCfg.cSpeedDesc.outflag=0;
				//			}
				//			else
				//			{
				//				pParams->NormalDetectCfg.cSpeedDesc.outflag++;
				//				pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting=FALSE;
				//				pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum = 0;
				//			}
				//			//	pCfgs->uDetectInSum[LaneID]++;



				//		}


				//		if(pParams->NormalDetectCfg.cSpeedDesc.is_bicycle==FALSE)
				//		{
				//			pCfgs->ResultMsg.uMsgHeader.uFlag = 0x55;
				//			pCfgs->ResultMsg.uMsgHeader.uCmd = 0xE0E0;
				//			//pCfgs->ResultMsg.uMsgHeader.uMsgLength = ;
				//			pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].uLaneID = LaneID;


				//			pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bInfoValid = 1; 	
				//			pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bVehicleSta = 1;	
				//		}

				//	}
				//	else if ( (pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting == TRUE) \
				//		&& (pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged == TRUE) \
				//		&& (pParams->NormalDetectCfg.cSpeedDesc.outflag < 3000)&&(bNonVehicle==FALSE))  
				//	{

				//		if (pParams->NormalDetectCfg.cSpeedDesc.speedflag == 1)
				//		{
				//			if ((pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) > 2  \
				//				&&(pCfgs->gThisFrameTime -  pParams->NormalDetectCfg.cSpeedDesc.firstline) < 20)
				//			{

				//				pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed = (float)(pParams->NormalDetectCfg.cSpeedDesc.uTransFactor) / ((pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) * 0.04) * 3.6;

				//				pParams->NormalDetectCfg.cSpeedDesc.speedflag = 0;

				//				if (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed > 60)
				//				{
				//					pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed =30+rand()%20;// 100 - (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed - 100) * (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed - 100) / 100;
				//				}
				//				else if(pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed<15)
				//					pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed =15+rand()%20;
				//				//
				//				//save speed
				//				pParams->NormalDetectCfg.cSpeedDesc.wLastVehicleSpeed = pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed;
				//			}
				//			else
				//			{
				//				//else use the previous value
				//				pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed = 15+rand()%15 ;
				//			}
				//		}
				//		if(bWuChe == FALSE)
				//			pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame++;
				//		if(pParams->NormalDetectCfg.cSpeedDesc.bGet1backflag == FALSE)
				//			pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame++;

				//		pParams->NormalDetectCfg.cSpeedDesc.outflag++;
				//	}
				//	else if((((pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame >= pCfgs->guAdmitBreakFrame)\
				//		&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged == FALSE)) \

				//		|| (pParams->NormalDetectCfg.cSpeedDesc.outflag >= 3000)||(pParams->NormalDetectCfg.cSpeedDesc.outflag >= 50&&pParams->NormalDetectCfg.cSpeedDesc.is_bicycle)||bNonVehicle==TRUE)

				//		&& (pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting == TRUE) \
				//		)//&&(bSpeedCounterChanged3==FALSE) )	
				//	{
				//		if(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==FALSE&&(cur_part_std[9]-pParams->NormalDetectCfg.cSpeedDesc.min_part_std[9])>15&&(pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[4]-pParams->NormalDetectCfg.cSpeedDesc.min_tail_std)>15)
				//		{
				//			bCarExit=TRUE;
				//		}
				//		if(pParams->NormalDetectCfg.cSpeedDesc.bGet1backflag&&pCfgs->CameraCfg.uEnvironmentStatus==4&&pParams->NormalDetectCfg.cSpeedDesc.outflag>20&&pCfgs->visibility==FALSE)
				//		{
				//			isShadow=remove_shadow(inBuf,pParams->puBackImagePlus,pCfgs,5);
				//		}
				//		if(LaneID==0)
				//		{
				//			//printf("cur_std=%f,min_std=%f,bSpeedCounterChanged1",cur_part_std[9],pParams->NormalDetectCfg.cSpeedDesc.min_part_std[9]);
				//			//printf("bCarExit=%d,bSpeedCounterChanged2=%d,isShadow=%d",bCarExit,pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged2,isShadow);

				//		}
				//		if( ((/*(bWuChe == TRUE) && */bCarExit==FALSE&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged2 == FALSE||isShadow))&&pParams->NormalDetectCfg.cSpeedDesc.outflag>15)

				//			||(pParams->NormalDetectCfg.cSpeedDesc.outflag >=3000)||(pParams->NormalDetectCfg.cSpeedDesc.outflag>100&&pParams->NormalDetectCfg.cSpeedDesc.is_bicycle)||(bNonVehicle==TRUE&&pParams->NormalDetectCfg.cSpeedDesc.outflag >30))  //3???A???闁活剨鎷�3?3I?

				//		{

				//			pParams->NormalDetectCfg.cSpeedDesc.outflag=0;
				//			pParams->NormalDetectCfg.cSpeedDesc.m_bSpdCnterHaveVehicle = FALSE;//?

				//			if (pParams->NormalDetectCfg.cSpeedDesc.lengthflag == 1)
				//			{
				//				if( ((pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.secondline) > 5) \
				//					&& (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed > 0) )
				//				{

				//					pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = pParams->NormalDetectCfg. \
				//						cSpeedDesc.wVehicleSpeed * (pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) * 0.04 / 3.6 - (float)(pParams->NormalDetectCfg.cSpeedDesc.uTransFactor);

				//					pParams->NormalDetectCfg.cSpeedDesc.lengthflag = 0;
				//					//TRACE("detect the vehicle out %d frame!",pCfgs->gThisFrameTime);

				//					//
				//					if( (pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength > 4) \
				//						&&(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength < 7) )
				//					{
				//						pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = 4 + \
				//							(7 - pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength) * \
				//							(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength - 4) / 4;
				//					}
				//					else if (pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength > 7)
				//					{
				//						pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = rand() % 3 + 4;
				//					}
				//					else if(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength<3)
				//					{
				//						pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength=rand() % 2+3;
				//					}


				//					//save length
				//					pParams->NormalDetectCfg.cSpeedDesc.wLastVehicleLength = pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength;
				//				}
				//				else
				//				{
				//					//else use the previous value
				//					pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = rand() % 2+3;
				//				}
				//			}

				//			memset((void*)&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus, \
				//				0, sizeof(SPEED_COUNTER_STATUS));

				//			pParams->NormalDetectCfg.cSpeedDesc.uTrafficFlow++;
				//			//pParams->NormalDetectCfg.cSpeedDesc.calarflag = 2;	
				//			//pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=2;
				//			pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame = 0;
				//			pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum = 0;

				//			if(nChannelNum == 0 || nChannelNum == 1)
				//			{
				//				if(pParams->NormalDetectCfg.cSpeedDesc.is_bicycle==FALSE)
				//				{
				//					pParams->NormalDetectCfg.cSpeedDesc.uTrafficFlow++;
				//					pParams->NormalDetectCfg.cSpeedDesc.calarflag = 2;	
				//					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=2;
				//					pCfgs->uDetectOutSum[LaneID]++;
				//					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.car_out++;
				//				}
				//				//	pCfgs->uDetectOutSum[LaneID]++;




				//			}
				//			if(pParams->NormalDetectCfg.cSpeedDesc.is_bicycle==FALSE)
				//			{
				//				pCfgs->ResultMsg.uMsgHeader.uFlag = 0x55;
				//				pCfgs->ResultMsg.uMsgHeader.uCmd = 0xE0E0;
				//				//pCfgs->ResultMsg.uMsgHeader.uMsgLength = ;
				//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].uLaneID = LaneID;
				//				//	pCfgs->guTestA[1]=(unsigned int) pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed;
				//				//	pCfgs->guTestA[2]=(unsigned int )pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength;
				//				if(pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed<20)
				//					pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed=15+rand()%10;
				//				else
				//					pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed = pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed - \
				//					(int)pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed % 10 + rand() % 10;
				//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleSpeed \
				//					=((unsigned int) pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed);
				//				if(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength<3)
				//					pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength=3+rand()%3;
				//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleLength  \
				//					= ((unsigned int )pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength);


				//				//???????????????????????????????
				//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bInfoValid = 1; 	//????????????
				//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bVehicleSta = 0;	//??????????????
				//			}		
				//		}
				//		else
				//		{
				//			pParams->NormalDetectCfg.cSpeedDesc.outflag++;
				//		}

				//	}
				//	else if(((pCfgs->gThisFrameTime-pParams->NormalDetectCfg.cSpeedDesc.jgtime)>150||(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged2==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.outflag>20))&&
				//		pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged3==FALSE)
				//	{
				//		pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame++;
				//		pParams->NormalDetectCfg.cSpeedDesc.outflag++;
				//	}
				//	else 
				//	{
				//		pParams->NormalDetectCfg.cSpeedDesc.outflag++;
				//	}

				//}
			}
			else //night
			{

				if( pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount == FALSE )
				{
					pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged = bIsSpeedCounterChanged( pParams->puPointSobelImage, \
						pCfgs->m_iHeight/2, \
						&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus, \
						1, pCfgs, pParams );
					if( bIsSobleBlockUnChanged( pParams->puPointSobelImage, SPEED_STARTER_STARTLINE, 0.6*pCfgs->m_iHeight/*SPEED_STARTER_ENDLINE*/,1, pCfgs, pParams) == FALSE
						&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged==FALSE))
					{
						pParams->NormalDetectCfg.cSpeedDesc.lxtime++;
					}
					else
					{
						pParams->NormalDetectCfg.cSpeedDesc.lxtime=0;
						//pParams->NormalDetectCfg.cSpeedDesc.calarflag=0;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=0;
						break;
					}
					if(pParams->NormalDetectCfg.cSpeedDesc.lxtime>0)
					{
						if(pCfgs->gThisFrameTime-pParams->NormalDetectCfg.cSpeedDesc.jgtime<20)
							break;
						else
						{
							pParams->NormalDetectCfg.cSpeedDesc.headtime=(pCfgs->gThisFrameTime-pParams->NormalDetectCfg.cSpeedDesc.jgtime)/15;//15 frame
							pParams->NormalDetectCfg.cSpeedDesc.jgtime=pCfgs->gThisFrameTime;
						}
						pParams->NormalDetectCfg.cSpeedDesc.outflag = 0;
						pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount = TRUE;
						//pParams->NormalDetectCfg.cSpeedDesc.calarflag = 1;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=1;
						pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum++;
						pParams->NormalDetectCfg.cSpeedDesc.speedflag=1;
						pParams->NormalDetectCfg.cSpeedDesc.firstline = pCfgs->gThisFrameTime;
						pParams->NormalDetectCfg.cSpeedDesc.lengthflag = 1;
						pParams->NormalDetectCfg.cSpeedDesc.secondline = pCfgs->gThisFrameTime;
                        if(LaneID==0)
						     pCfgs->calctimes=1;//0710xyx
					}
					else
					{		
						//pParams->NormalDetectCfg.cSpeedDesc.calarflag=0;
						break;
					}
				}	
				pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged = bIsSpeedCounterChanged( pParams->puPointSobelImage, \
					pCfgs->m_iHeight/2, \
					&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus, \
					1, pCfgs, pParams );

				//pCfgs->guTestD[0]=(Uint16)pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged;

				pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1 = bIsSpeedCounterChanged1( pParams->puPointSobelImage, \
					pCfgs->m_iHeight/2, \
					&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus, \
					1,LaneID, pCfgs, pParams );
				//pCfgs->guTestD[1]=	(Uint16)pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1;

				if((pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1 == TRUE) )\
					//||(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==TRUE) 
				{
					pParams->LaneParameter.VehicleExistFrameCount++;
					pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum++;
					pParams->NormalDetectCfg.cSpeedDesc.outflag = 0;
					//pParams->NormalDetectCfg.cSpeedDesc.calarflag=1;
					pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame=0;
				}
				else if (pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum==0)//3I???2???????闁充紮鎷�???DD???
				{
					//pParams->NormalDetectCfg.cSpeedDesc.calarflag=0;//??3I缁绢叏鎷�??????
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=0;
				}
				/*if (pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount == TRUE &&
					(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged ==TRUE))
				{
					pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bbottomappearedcar=TRUE;
				}*/
				if( (pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bCanBeginCount == TRUE) &&
					(pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting == FALSE)
					/*(pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bbottomappearedcar==TRUE||pParams->LaneParameter.VehicleExistFrameCount>40)*/)
				{
					pParams->NormalDetectCfg.cSpeedDesc.outflag = 0;
					pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting = TRUE;
					pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum = 1;
					pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame = 0;


					if(nChannelNum == 0 || nChannelNum == 1)
					{
						pCfgs->uDetectInSum[LaneID]++;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.car_in++;


						//	TRACE("3I?????????????y??o%d\n ",LanePtr->TotalFrameCount);							
					}

					//D????闁荤噦鎷�?
					pCfgs->ResultMsg.uMsgHeader.uFlag = 0x55;
					pCfgs->ResultMsg.uMsgHeader.uCmd = 0xE0E0;
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].uLaneID = LaneID;
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bInfoValid = 1; 	//???a???1??闁活厾妾矨?
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bVehicleSta = 1;	//3I???????????a??

				}	
				else if ( (pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting == TRUE) \
					&& (pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged == TRUE)
					&& (pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1 == FALSE)
					)
				{
					if (pParams->NormalDetectCfg.cSpeedDesc.speedflag == 1)
					{
						if ( ((pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) > 2)
							&& ((pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) < 20) )  ////10 ????
						{
							pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed = (float)(pParams->NormalDetectCfg.cSpeedDesc.uTransFactor) /  \
								((pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) * 0.067) * 3.6;	

							pParams->NormalDetectCfg.cSpeedDesc.speedflag = 0;
							if (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed > 60)
							{
								pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed =30+rand()%20;// 100 - (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed - 100) * (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed - 100) / 100;
							}
							else if(pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed <10)
								pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed =15+rand()%20;

							//save speed
							pParams->NormalDetectCfg.cSpeedDesc.wLastVehicleSpeed = pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed;
						}
						else
						{
							//else use the previous value
							pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed = 14+rand()%15;
						}
					}
					pParams->NormalDetectCfg.cSpeedDesc.outflag=0;
					pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame++;
				}
				else if((pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus.bIsCounting == TRUE)\
					&& (pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged == FALSE)
					&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1== FALSE)||
					(pParams->NormalDetectCfg.cSpeedDesc.outflag>4000)
					)
				{
					pParams->NormalDetectCfg.cSpeedDesc.outflag++;
					if( (pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame < 1) \
						&& (pParams->NormalDetectCfg.cSpeedDesc.outflag < 20)||
						(pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame >0)
						&&(pParams->NormalDetectCfg.cSpeedDesc.outflag <3))    /////<15 ???
					{
						break;
					}
					if (pParams->NormalDetectCfg.cSpeedDesc.lengthflag == 1)
					{
						if( ((pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) > 5)  \
							&& (pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed > 0) )
						{
							pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength =  \
								pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed * \
								(pCfgs->gThisFrameTime - pParams->NormalDetectCfg.cSpeedDesc.firstline) * 0.067 / 3.6 - \
								(float)(pParams->NormalDetectCfg.cSpeedDesc.uTransFactor);

							pParams->NormalDetectCfg.cSpeedDesc.lengthflag = 0;
							if( (pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength > 4) \
								&&(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength < 7) )
							{
								pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = 4 + \
									(7 - pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength) * \
									(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength - 4) / 4;
							}
							else if (pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength > 7)
							{
								pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = rand() % 2 + 4;
							}
							else if(pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength<3)
							{
								pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength=rand() % 2+3;
							}

							//save length
							pParams->NormalDetectCfg.cSpeedDesc.wLastVehicleLength = pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength;
						}
						else
						{
							//else use the previous value
							pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength = rand() % 2+3;
						}
					}

					memset((void*)&pParams->NormalDetectCfg.cSpeedDesc.SpeedCounterStatus, 0, sizeof(SPEED_COUNTER_STATUS));
					pParams->NormalDetectCfg.cSpeedDesc.uTrafficFlow++;

					//pParams->NormalDetectCfg.cSpeedDesc.calarflag = 2;
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.calarflag=0;
					pParams->NormalDetectCfg.cSpeedDesc.outflag = 0;
					pParams->NormalDetectCfg.cSpeedDesc.uBreakFrame = 0;
					pParams->NormalDetectCfg.cSpeedDesc.uCountFrameSum = 0;
					pParams->NormalDetectCfg.cSpeedDesc.uCountIntervalFrame = 0;
					if(nChannelNum == 0 || nChannelNum == 1)
					{						
						pCfgs->uDetectOutSum[LaneID]++;
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.car_out++;

					}
					//D????闁荤噦鎷�?
					pCfgs->ResultMsg.uMsgHeader.uFlag = 0x55;
					pCfgs->ResultMsg.uMsgHeader.uCmd = 0xE0E0;
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].uLaneID = LaneID;
					if(pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed<20)
						pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed=15+rand()%10;
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleSpeed \
						=((unsigned int) pParams->NormalDetectCfg.cSpeedDesc.wVehicleSpeed);

					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleLength  \
						= ((unsigned int )pParams->NormalDetectCfg.cSpeedDesc.wVehicleLength);
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bInfoValid = 1; 
					pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.bVehicleSta = 0;	

				}

			}

		} //while( bSpeedCalculate == TRUE ) ends


	}//if,cross
}

bool wuche(unsigned char* GrayImage,unsigned char* BackImage,unsigned int start,unsigned int end, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	int row,col;
	int value=0;
	//float value=0;
	//float value1=0;
	//float graysub=0;
	int count=0,rcount=0;
	int GrayImageValue=0,BackImageValue=0;
	/*for (row=start;row<end;row++)
	{
	for (col=(0.25*pCfgs->m_iWidth);col<(0.75*pCfgs->m_iWidth);col++)//txl,20130821,for (col=0;col<pCfgs->m_iWidth;col++)
	{
	value+=GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,GrayImage);
	value1+=GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,BackImage);
	count++;
	}
	}
	if (count)
	{
	value /= count;
	value1 /= count;
	}
	graysub = abs(value - value1);
	//if(graysub > 140)
	if(graysub > pParams->NormalDetectCfg.cSpeedDesc.uGraySubThreshold)  //by david 20130905 
	return FALSE;
	else
	return TRUE;*/
	for (row=pCfgs->m_iHeight/2;row<pCfgs->m_iHeight;row++)
	{

		count=0;
		for (col=0;col<pCfgs->m_iWidth;col++)
		{
			GrayImageValue=GetImagePointValue(col, row, pCfgs->m_iHeight, pCfgs->m_iWidth, GrayImage);
			BackImageValue=GetImagePointValue(col, row, pCfgs->m_iHeight, pCfgs->m_iWidth, BackImage); 
			value=abs(GrayImageValue - BackImageValue) ;

			if (value>20)//20
				count++;
		}
		if(count>pCfgs->m_iWidth/4)
			rcount++;
		if(rcount>pCfgs->m_iHeight/4) 
			return FALSE;  
	}
	return TRUE;
}

bool bIsBottomNull( unsigned char* puSobelImage, Uint16 uSpeedCounterStartRow, SPEED_COUNTER_STATUS* pSpeedCounterStatus, ALGCFGS *pCfgs, ALGPARAMS *pParams )
{

	int  nRow;
	int nUnChangedRow = 0;
	unsigned int uThreshold = 20;//1
	//	int nullRowNum=0,maxNullRowPos=0,maxNullRow=0;
	//txl
	for( nRow = uSpeedCounterStartRow + 2; nRow < pCfgs->m_iHeight -1; nRow++ )
	{
		if( GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) > uThreshold )
		{
			nUnChangedRow++;
		}
		////20131218  1?5
		if(nUnChangedRow>pCfgs->m_iHeight/8)//6
		{
			//TRACE("?????闁活叏鎷�3I缂侇垰鍤�??o%d\n ",nUnChangedRow);
			return FALSE;

		}

	}

	return TRUE;//
}

bool BackGrayCharge(unsigned char* GrayImage,unsigned char* BackImage,unsigned int start,unsigned int end, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	int row,col;
	float value = 0;
	float value1 = 0;
	int count = 0;
	for (row = start; row < end; row++)
	{
		for (col = 0; col < pCfgs->m_iWidth; col++)
		{
			value += GetImagePointValue(col,row, pCfgs->m_iHeight,pCfgs->m_iWidth,GrayImage);
			value1 += GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,BackImage);
			count++;
		}
	}
	if (count)
	{
		value /= count;
		value1 /= count;
	}
	if( abs(value - value1) >20 )
		return TRUE;
	else
		return FALSE;
}
////20140107 
void sobel_rectify(unsigned char* puPointNextImage,unsigned char* puPointBackSobelImage,ALGCFGS *pCfgs)
{
	unsigned int i,j;
	for(i=0;i<pCfgs->m_iHeight;i++)
		for(j=0;j<pCfgs->m_iWidth;j++)
		{
			if((GetImagePointValue(j, i, pCfgs->m_iHeight, pCfgs->m_iWidth, puPointNextImage)-GetImagePointValue(j, i, pCfgs->m_iHeight, pCfgs->m_iWidth, puPointBackSobelImage))<=0)
			{
				*(puPointNextImage+i*pCfgs->m_iWidth+j)=0;

			}

		}
}

bool estimate_background(unsigned char *puPointBackSobelImage,ALGCFGS *pCfgs)
{ 
	unsigned int i,j,sum=0;

	for(i=0;i<pCfgs->m_iHeight;i++)
		for(j=0;j<pCfgs->m_iWidth;j++)
		{
			if(*(puPointBackSobelImage+i*pCfgs->m_iWidth+j)>0)
				++sum;
		}
		if(sum>(pCfgs->m_iWidth*pCfgs->m_iHeight)/3) 
		{
			return FALSE;
		}
		else 
		{
			return TRUE; 
		}
}
////20131218 add
bool bIsSpeedCounterChanged3(unsigned char* puSobelImage1, unsigned char* puSobelImage, unsigned char* GrayImage, unsigned short uSpeedCounterStartRow, unsigned char* BackImage, ALGCFGS *pCfgs, ALGPARAMS *pParams )
{
	int nRow=0,nCol=0;
	int value=0/*,svalue=0*/;
	int rcount=0,count=0;

	//int srcount=0,scount=0;
	//int nchangedRow=0;
	unsigned int uThreshold = 20; 
	//unsigned int uThreshold = pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold2; 
	int GrayImageValue=0,BackImageValue=0;

	int startline;
	//if (uSpeedCounterStartRow)
	//{
	//	startline=0/*pCfgs->m_iHeight/2*/;
	//}
	//else
		startline=0;
	uThreshold = (pCfgs->visibility)? 10 : 20;
	for (nRow=startline;nRow<pCfgs->m_iHeight;nRow++)
	{

		/*if (GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) >uThreshold)
		{
			nchangedRow++;
		}*/
		for (nCol=0;nCol<pCfgs->m_iWidth;nCol++)
		{
			GrayImageValue=GetImagePointValue(nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, GrayImage);
			BackImageValue=GetImagePointValue(nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, BackImage); 
			/*	value=abs(GetImagePointValue(nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, GrayImage)
			-GetImagePointValue(nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, BackImage));
			*/
			value=abs(GrayImageValue - BackImageValue) ;

			if (value<uThreshold)//20
				count++;
			//svalue=GetImagePointValue(nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage1);
			//if (svalue)
			//scount++;
		}
		if (count>(pCfgs->m_iWidth*0.75))
		{
			/*for (nCol=0;nCol<pCfgs->m_iWidth;nCol++)
			{
			SetImagePointValue(nCol,nRow,pCfgs->m_iHeight,pCfgs->m_iWidth,0,GrayImage);
			}*/
			count=0;
			//rcount++;
		}
		else
		{
			count=0;
			//rcount=0;
			rcount++;
		}



		//if (scount>(pCfgs->m_iWidth>>1))
		//{
		/*for (nCol=0;nCol<pCfgs->m_iWidth;nCol++)
		{
		SetImagePointValue(nCol,nRow,pCfgs->m_iHeight,pCfgs->m_iWidth,0,GrayImage);
		}*/
		//scount=0;
		//srcount++;
		//}
		//else
		//{
		//scount=0;
		//srcount=0;//421
		//}

	}
	////20140107
	if(pParams->NormalDetectCfg.cSpeedDesc.bGet1backflag)//
	{		
		if ((rcount<pCfgs->m_iHeight/4)/*&&(nchangedRow<pCfgs->m_iHeight/3)&&(srcount<pCfgs->m_iHeight/7)*/)//(rcount>pCfgs->m_iHeight/5)
		{
			return FALSE;//leave
		}

		else
			return TRUE;//exist
	}	

	else
	{
		if ((rcount<pCfgs->m_iHeight/4)/*)&&(nchangedRow<(pCfgs->m_iHeight/3))&&(srcount<(pCfgs->m_iHeight/5))*/)//(rcount>(pCfgs->m_iHeight/5)
		{
			return FALSE;//leave
		}
		else
			return TRUE;//exist
	}

}
bool bIsSpeedCounterChanged1( unsigned char* puSobelImage, Uint16 EndLine, SPEED_COUNTER_STATUS* pSpeedCounterStatus ,int flag, Uint16 lane_id, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	int nLeftBlockColumn = pCfgs->m_iWidth / 3, nRightBlockColumn = pCfgs->m_iWidth * 2 / 3;
	int nPosSum = 0, nRowCount = 0, nRow;
	int nUnChangedRow = 0;
	unsigned int uThreshold ;//1
	unsigned int i;
	int k;
	//////printf("m_iheight is: %d\t",pCfgs->m_iHeight);
	//////printf("EndLine is: %d\t",EndLine);
	if (flag == 0)
	{
		//uThreshold = 10; //by david 20130905 from 5 to 2
		uThreshold = pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold1; //by david 20130905
		//////printf("uThreshold is: %d\t",uThreshold);
		//////printf("uVehicleHeadMinRow is: %d\t",pCfgs->CameraCfg.uVehicleHeadMinRow );
		k = 2;

		for( nRow = 1; nRow < EndLine; nRow++ )
		{
			if( GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) < uThreshold )
			{
				//if (!nUnChangedRow)
				//{
				//Rowposition1 = nRow;  //?A?????闂佺鎷�???, by david
				//}
				nUnChangedRow++;
			}
			if( (nUnChangedRow * k) > pCfgs->CameraCfg.uVehicleHeadMinRow )
			{
				/*pSpeedCounterStatus->bRightDestroyed = FALSE;
				pSpeedCounterStatus->bLeftDestroyed = FALSE;
				//////printf("changed1 is: FALSE\n");
				////20131218  zcj
				for(i=0;i<20;i++)
				{  
					pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[i][lane_id][0]=FALSE;//???闁稿鎷�????
					pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[i][lane_id][1]=0;
				}*/
				return FALSE;
			}
		}
	}

	else////night 
	{

		uThreshold=pCfgs->m_iWidth / 5;
		// 	uThreshold = pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold1; 

		k = 1;
		for( nRow = 0; nRow < EndLine; nRow++ )
		{
			if( GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) < uThreshold )
			{
				nUnChangedRow++;
			}
			if( (nUnChangedRow*k) >EndLine-3 )//why multipled by 2? david
			{
				pSpeedCounterStatus->bRightDestroyed = FALSE;
				pSpeedCounterStatus->bLeftDestroyed = FALSE;
				for(i=0;i<20;i++)
				{  
					pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[i][lane_id][0]= FALSE;
					pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[i][lane_id][1]=0;
					//rectify_lane_direction[i][lane_id][0]=false;//??
					//rectify_lane_direction[i][lane_id][1]=0;
				}
				return FALSE;
			}
		}
	}
	if( pSpeedCounterStatus->bIsCounting == FALSE ){
		for( nRow = 1; nRow < SPEED_STARTER_ENDLINE; nRow++ )
		{
			if(GetImagePointValue(pCfgs->m_iWidth-1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ))
			{
				nPosSum += GetImagePointValue(0, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage );
				nRowCount++;
			}
		}
		if( (nPosSum * 1.0 / nRowCount) > nRightBlockColumn )
		{
			/*if(pParams->NormalDetectCfg.cSpeedDesc.track_flag==0)  
				pParams->NormalDetectCfg.cSpeedDesc.track_flag=20;*/
			////20131220  pCfgs->m_iWidth/4  ?pCfgs->m_iWidth/6
			if((( pCfgs->m_iWidth-(nPosSum*1.0 / nRowCount))>(pCfgs->m_iWidth/4))/*&&(pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[pParams->NormalDetectCfg.cSpeedDesc.track_flag-1][lane_id][0]==FALSE)*/)
			{   
				//	TRACE("\n2b[%d][%d][21]=%d\n",frame_track-1,lane_id,rectify_lane_direction[frame_track-1][lane_id][1]);
				pSpeedCounterStatus->bRightDestroyed = TRUE;
				pSpeedCounterStatus->bLeftDestroyed = FALSE;

			}
			else 
			{
				pSpeedCounterStatus->bRightDestroyed = FALSE;
				pSpeedCounterStatus->bLeftDestroyed = FALSE;

				//pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[pParams->NormalDetectCfg.cSpeedDesc.track_flag][lane_id][0]=TRUE;
				//pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[pParams->NormalDetectCfg.cSpeedDesc.track_flag][lane_id][1]+=1;
				////	TRACE("\nrectify_lane_direction[%d][%d][21]=%d\n",frame_track,lane_id,rectify_lane_direction[frame_track][lane_id][1]);
				return FALSE;
			}
		} 
		else if( (nPosSum * 1.0 / nRowCount) < nLeftBlockColumn )
		{    
			if(pParams->NormalDetectCfg.cSpeedDesc.track_flag==0) pParams->NormalDetectCfg.cSpeedDesc.track_flag=20;
			//if(((nPosSum*1.0 / nRowCount)>(pCfgs->m_iWidth/4))&&(pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[pParams->NormalDetectCfg.cSpeedDesc.track_flag-1][lane_id][0]==FALSE))
			////20131220		pCfgs->m_iWidth/4  ?3
			if(((nPosSum*1.0 / nRowCount)>(pCfgs->m_iWidth/4))/*&&(pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[pParams->NormalDetectCfg.cSpeedDesc.track_flag-1][lane_id][0]==FALSE)*/)
			{

				//	TRACE("\n2b1[%d][%d][21]=%d\n",(frame_track-1),lane_id,rectify_lane_direction[frame_track-1][lane_id][1]);
				pSpeedCounterStatus->bLeftDestroyed = TRUE;
				pSpeedCounterStatus->bRightDestroyed = FALSE;


			}			
			else{
				pSpeedCounterStatus->bLeftDestroyed = FALSE;
				pSpeedCounterStatus->bRightDestroyed = FALSE;

				//pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[pParams->NormalDetectCfg.cSpeedDesc.track_flag][lane_id][0]=TRUE;
				//pParams->NormalDetectCfg.cSpeedDesc.rectify_lane_direction[pParams->NormalDetectCfg.cSpeedDesc.track_flag][lane_id][1]+=1;
				return FALSE;
			}
		}
		else{
			pSpeedCounterStatus->bRightDestroyed = TRUE;
			pSpeedCounterStatus->bLeftDestroyed = TRUE;
		}
	}
	return TRUE;

}

////20131218  ?????
/*
bool bIsSpeedCounterChanged1( unsigned char* puSobelImage, Uint16 EndLine, SPEED_COUNTER_STATUS* pSpeedCounterStatus ,int flag, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
int nLeftBlockColumn = pCfgs->m_iWidth / 3, nRightBlockColumn = pCfgs->m_iWidth * 2 / 3;
int nPosSum = 0, nRowCount = 0, nRow;
int nUnChangedRow = 0;
unsigned int uThreshold ;//1
float k;
if (flag == 0)
{
//uThreshold = 10; //by david 20130905 from 5 to 2
uThreshold = pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold1; //by david 20130905
k = 2;
}
else{
uThreshold=pCfgs->m_iWidth / 5;
k = 0.8;
}

//?D?????A??闁活叏鎷�3I???闁绘﹩鍋撻幏锟�?
for( nRow = 0; nRow < EndLine; nRow++ )
{
if( GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) < uThreshold )
{
if (!nUnChangedRow)
{
//Rowposition1 = nRow;  //闂佺鎷�?A?????闂佺鎷�????, by david
}
nUnChangedRow++;
}
if( (nUnChangedRow * k) > pCfgs->CameraCfg.uVehicleHeadMinRow )
{
pSpeedCounterStatus->bRightDestroyed = FALSE;
pSpeedCounterStatus->bLeftDestroyed = FALSE;

return FALSE;
}
}

//?????D???????????a????闁活偓鎷�???闁活叏鎷�??1?????闁活偓鎷�?,??????D??
if( pSpeedCounterStatus->bIsCounting == FALSE ){
for( nRow = 0; nRow < SPEED_STARTER_ENDLINE; nRow++ )
{
nPosSum += GetImagePointValue(0, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage );
nRowCount++;
}
if( (nPosSum * 1.0 / nRowCount) > nRightBlockColumn )
pSpeedCounterStatus->bRightDestroyed = TRUE;
else if( (nPosSum * 1.0 / nRowCount) < nLeftBlockColumn )
pSpeedCounterStatus->bLeftDestroyed = TRUE;
else{
pSpeedCounterStatus->bRightDestroyed = TRUE;
pSpeedCounterStatus->bLeftDestroyed = TRUE;
}
}
return TRUE;

}
*/
////20131218  add
bool bIsSpeedCounterChanged2(unsigned char* puSobelImage1, unsigned char* puSobelImage, Uint16 uSpeedCounterStartRow, SPEED_COUNTER_STATUS* pSpeedCounterStatus, ALGCFGS *pCfgs, ALGPARAMS *pParams )
{
	int nRow=0,nCol=0;
	//int value=0;
	//int rcount=0,*ccount=NULL;
	//int hig=0,bel=0;
	int nUnChangedRow = 0;
	//	unsigned int uThreshold = 10; //by david 20130905 from 1 to 10
	unsigned int uThreshold = pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold2; //by david 20130905 
	//	int middleThreshold = 90;
	int startline;
	int count=0,rcount=0;
	startline = (uSpeedCounterStartRow)? pCfgs->m_iHeight/2 : 0;


	for( nRow =startline ; nRow < pCfgs->m_iHeight; nRow++ )
	{
		if( GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) >uThreshold )//
			nUnChangedRow++;
		for (nCol=0;nCol<pCfgs->m_iWidth;nCol++)
		{
			if (GetImagePointValue( nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage1 ) >uThreshold)
				count++;

		}
		if(count>pCfgs->m_iWidth/6)
			rcount++;
		count=0;
		////20140107 

		if(pParams->NormalDetectCfg.cSpeedDesc.bGet1backflag)//take_back_flag??:system?I?????3??????????????
		{
			if( (nUnChangedRow) > pCfgs->m_iHeight/5||rcount>pCfgs->m_iHeight/4)
			{
				return TRUE;//exist
			}
		}
		else
		{
			////20140108
			//	if( (nUnChangedRow) > pCfgs->m_iHeight/5||rcount>pCfgs->m_iHeight/5)//
			if( (nUnChangedRow) > pCfgs->m_iHeight/5||rcount>5*pCfgs->m_iHeight/12)//
			{
				return TRUE;//exist
			}
		}
	}


	return FALSE;
}

////
//***********************************************

void ShallowLightRemovePreProcess(Uint8* puNewImage,Uint8* puTempImage,Uint8*puBackImage,Uint8* puTempStoreY,struct cSpeedDetectStruct* pSpeedDetectStruct,Uint16 uChanlNum,ALGCFGS *pCfgs, ALGPARAMS	*pParams)
{
	//	SOBELINPUT SobelInput;
	float dTempGrey;
	//	bool bUpdateBack = FALSE;

	if( pCfgs->CameraCfg.bCameraWorkInCross == 1 )//?A???
	{


	}//?A???1A????????
	else if(pCfgs->CameraCfg.bCameraWorkInCross == 2)//?A??
	{
		//to add code not working in cross
		iSubStractImage( puNewImage, puTempImage, pCfgs->CameraCfg.guSubImageThreshold, 0, pCfgs->m_iHeight, pCfgs->m_iWidth, pCfgs->m_iHeight);////

		/*teamlength=30*TeamLength(pParams->puPointNextImage,puTempImage,pCfgs,pParams,0);
		pCfgs->thresholdValue=teamlength;
		pSpeedDetectStruct->m_bInNight = FALSE;*/
		//pSpeedDetectStruct->SpeedDetectorWorkMode.bAuto = FALSE;
		//pSpeedDetectStruct->SpeedDetectorWorkMode.bDay = FALSE;

		SLKCalculate( puTempImage, pCfgs->CameraCfg.uSobelThreshold, pCfgs, pParams );
		//513
		//memcpy(puTempStoreY,puTempImage,pCfgs->m_iHeight*pCfgs->m_iWidth);

		//513
		//////////////////get night or day info
		/*if( bIsBlockAllEmpty( puTempImage, 1, SPEED_STARTER_ENDLINE, 1, 10, pCfgs, pParams ) == TRUE )
		{
		//		fSetDetectStateFlag( NormalDetectPtr->cSpeedDesc.DetectId, FALSE );
		if( pSpeedDetectStruct->m_uContiuCount == 0 )
		pSpeedDetectStruct->m_uPassTimeCount++;
		pSpeedDetectStruct->m_uAvergreyMax = 0;
		pSpeedDetectStruct->m_uContiuCount = 1;
		}
		else if( pSpeedDetectStruct->m_uContiuCount != 0 )
		{
		//		fSetDetectStateFlag( NormalDetectPtr->cSpeedDesc.DetectId, TRUE );
		dTempGrey = dAverageGreyInBlock( puNewImage, 1, SPEED_STARTER_ENDLINE, pCfgs, pParams );
		if( dTempGrey > (float)pSpeedDetectStruct->m_uAvergreyMax )
		{
		pSpeedDetectStruct->m_uAvergreyMax = (Uint16)dTempGrey;
		pSpeedDetectStruct->m_uContiuCount++;
		}
		else
		{
		if( pSpeedDetectStruct->m_uContiuCount > pCfgs->CameraCfg.uDayNightJudgeMinContiuFrame )//Threshold can adjust
		pSpeedDetectStruct->m_uLightCount++;
		pSpeedDetectStruct->m_uAvergreyMax = 0;
		pSpeedDetectStruct->m_uContiuCount = 0;
		}
		}
		if( pSpeedDetectStruct->m_uPassTimeCount > (pSpeedDetectStruct->uDayNightJudgeCicleCount+1)*20 )
		{
		pSpeedDetectStruct->m_bInNight = FALSE;
		if( pSpeedDetectStruct->m_uLightCount > (pSpeedDetectStruct->uDayNightJudgeCicleCount+1) )
		pSpeedDetectStruct->m_bInNight = TRUE;
		if( pSpeedDetectStruct->uDayNightJudgeCicleCount > 30 )
		pSpeedDetectStruct->uDayNightJudgeCicleCount = 0;
		else
		pSpeedDetectStruct->uDayNightJudgeCicleCount += 2;

		pSpeedDetectStruct->m_uPassTimeCount = 0;
		pSpeedDetectStruct->m_uAvergreyMax = 0;
		pSpeedDetectStruct->m_uContiuCount = 0;
		}*/

	}
}
/*
inline void SobelThreshold
(
Uint8  *Buf,   
Uint16 dotNum,
Uint16 threshold
)
{
Int32	i;
for(i=0;i<dotNum;i++)
{
if( Buf[i] < threshold )
{
Buf[i] = 0;
}
else if(Buf[i] > 0x3f )
{
Buf[i] = 0x3f;
} 

}
}
*/

void nSubstractImage(Uint8* puNewImage,Uint8* puTempImage,Uint8* puTargetImage, Int16 nStartLine,Int16 nEndLine, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	Int16 nRow, nCol;
	bool bFoundStart = FALSE;
	Uint16 nGrey;

	//I??????I????
	SetImagePointValue( 0, 0, pCfgs->m_iHeight, pCfgs->m_iWidth, 0, puTargetImage );

	//????I缂侇垽鎷�??????I?
	for( nRow = nStartLine; nRow < nEndLine; nRow++ )
	{
		for( nCol = 0; nCol < pCfgs->m_iWidth; nCol++ )
		{
			nGrey = (*(puNewImage + nRow*pCfgs->m_iWidth + nCol)) - (*(puTempImage + nRow*pCfgs->m_iWidth + nCol));
			if( abs( nGrey ) < 18 )
			{
				nGrey = 0;
			}
			SetImagePointValue( nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, (Uint16)(abs(nGrey)), puTargetImage );
		}
	}
	for( nCol = 0; nCol < pCfgs->m_iWidth; nCol++ )
	{
		bFoundStart = FALSE;
		for( nRow = nEndLine - 1; nRow > nStartLine; nRow-- )
		{
			if( (*(puTargetImage + nRow*pCfgs->m_iWidth + nCol)) != 0 && bFoundStart == FALSE )
			{
				bFoundStart = TRUE;
				SetImagePointValue( nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, 0, puTargetImage );
				SetImagePointValue( nCol, nRow - 1, pCfgs->m_iHeight, pCfgs->m_iWidth, 0, puTargetImage );
				nRow--;
			}
			else if( bFoundStart == TRUE )
				SetImagePointValue( nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, 150, puTargetImage );
		}
	}
}

void ImageCopy(Uint8* pucSource, Uint8* pucTarget, Int16 nStartLine, Int16 nEndLine, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	Int16 nRow, nCol;
	for( nRow = nStartLine; nRow < nEndLine; nRow++ )
	{
		for( nCol = 0; nCol < pCfgs->m_iWidth; nCol++ )
		{
			SetImagePointValue( nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth,
				GetImagePointValue( nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, pucSource ), pucTarget );
		}
	}
}

void iSubStractImage(Uint8 *puSourceImage,Uint8 *puTargetImage, Uint32 nThreshold, Int16 nFromLine, Int16 nToLine, Int16 width, Int16 height)
{
	Int32 iRow,iCol,nCompareResult;

	for( iRow = nFromLine; iRow < nToLine; iRow++ )
	{
		for( iCol = 0; iCol < width; iCol++ )
		{
			nCompareResult =  *( puSourceImage + iCol + width * iRow )  -  *( puTargetImage + iCol + width * iRow ) ;
			if( abs(nCompareResult) < nThreshold )
				nCompareResult = 0;
			SetImagePointValue( iCol, iRow, height, width, (Uint16)abs(nCompareResult), puTargetImage );
		}
	}
}

bool bIsBlockAllEmpty(Uint8* puImage, Int16 nStartRow, Int16 nEndRow, Int16 nChangedPt, Uint16 uThreshold, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	Int16 nChangeCount = 0;
	Int16 nRow,nCol;

	for( nRow = nStartRow; nRow < nEndRow; nRow++ )
	{
		for( nCol = 1; nCol < pCfgs->m_iWidth - 1; nCol++ )
		{
			if( GetImagePointValue( nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puImage ) != 0 )
				nChangeCount++;
		}
	}

	if( nChangeCount > nChangedPt )
		return FALSE;

	return TRUE;
}
void IsolatePointDelete(Uint8* puSourceImage, Uint16 uLines, Uint16 uColumns, Uint16 uThreshold)//1????闁绘洩鎷�???3y?????I?E?
{
	int iRow, iCol, nGrey;
	for( iRow = 1; iRow < uLines - 1; iRow++ ){
		for( iCol = 1; iCol < uColumns - 1; iCol++ ){
			nGrey =(int)(GetImagePointValue( iCol - 1, iRow - 1, uLines, uColumns, puSourceImage )
				+ GetImagePointValue(iCol-1, iRow, uLines, uColumns, puSourceImage ) +
				GetImagePointValue( iCol - 1, (iRow + 1), uLines, uColumns, puSourceImage)+
				GetImagePointValue(iCol+1, iRow-1, uLines, uColumns, puSourceImage)
				+GetImagePointValue(iCol+1, iRow, uLines, uColumns, puSourceImage)+
				GetImagePointValue(iCol+1, iRow+1, uLines, uColumns, puSourceImage)
				+GetImagePointValue(iCol, iRow-1, uLines, uColumns, puSourceImage)+
				GetImagePointValue(iCol, iRow+1, uLines, uColumns, puSourceImage));
			nGrey = nGrey / 8;
			if( nGrey <= uThreshold )
				nGrey = 0;
			SetImagePointValue( iCol, iRow, uLines, uColumns, (Uint16)nGrey, puSourceImage );
		}
	}

}

void ProjectSubAndSobelEdge(Uint8* puImage, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	Int16 iRow, iCol, nPtCount, nValidPtCount, nPosSum;
	Uint16 nGrey;

	for( iRow = 1; iRow < pCfgs->m_iHeight - 1; iRow++ )
	{
		if( GetImagePointValue( 0, iRow , pCfgs->m_iHeight, pCfgs->m_iWidth, puImage ) > 0 )
		{ 
			SetImagePointValue( pCfgs->m_iWidth - 1, iRow, pCfgs->m_iHeight, pCfgs->m_iWidth, 180, puImage);
			SetImagePointValue( 0, iRow, pCfgs->m_iHeight, pCfgs->m_iWidth, (Uint16)pCfgs->m_iWidth/2, puImage);
		}
		else
		{
			SetImagePointValue( pCfgs->m_iWidth - 1, iRow, pCfgs->m_iHeight, pCfgs->m_iWidth, 0, puImage);
			SetImagePointValue( 0, iRow, pCfgs->m_iHeight, pCfgs->m_iWidth, 0, puImage);
		}
		/*nGrey = 0;
		nPtCount = 0;
		for( iCol = 1; iCol < pCfgs->m_iWidth - 1; iCol++ )
		{
		if( GetImagePointValue( iCol, iRow , pCfgs->m_iHeight, pCfgs->m_iWidth, puImage ) > 0 )
		{
		nPtCount++;
		nGrey += GetImagePointValue( iCol, iRow , pCfgs->m_iHeight, pCfgs->m_iWidth, puImage );
		}
		}
		nValidPtCount = 0;
		nPosSum = 0;
		for( iCol = 1; iCol < pCfgs->m_iWidth - 1; iCol++ )
		{
		if( GetImagePointValue( iCol, iRow , pCfgs->m_iHeight, pCfgs->m_iWidth, puImage ) > 0 )
		{
		nValidPtCount++;
		nPosSum += iCol;
		}
		}
		if( nValidPtCount > 0 )
		{
		nPosSum = (Int16)(nPosSum / nValidPtCount);
		}

		SetImagePointValue( 0, iRow, pCfgs->m_iHeight, pCfgs->m_iWidth, (Uint16)nPosSum, puImage ); //?????DD?A????I??????D

		if (nGrey > 255)
		SetImagePointValue( pCfgs->m_iWidth - 1, iRow, pCfgs->m_iHeight, pCfgs->m_iWidth, 180, puImage);
		else
		SetImagePointValue( pCfgs->m_iWidth - 1, iRow, pCfgs->m_iHeight, pCfgs->m_iWidth, nGrey, puImage);//?????????1?闂佺鎷�???????(uint)
		//SetImagePointValue( m_iWidth - 1, iRow,m_iHeight, m_iWidth, (uint)nPtCount, puImage);//A??????D????y闂佺鎷�?闂佺偨鍎撮锟�??*/

	}
}


void SLKTemplateCalculate(Uint8* puSourceImage, Uint8* uTargetImage, Uint16 uLines, Uint16 uColumns, bool bVertical)
{
	Int16 iRow, iCol;
	Uint16 uGrey;
	if( !bVertical )
	{
		for( iRow = 0; iRow < uLines; iRow++ )
		{
			for( iCol = 0; iCol < uColumns; iCol++ )
			{
				if( (iRow == 0 ) || (iRow == uLines - 1 ))
					SetImagePointValue( iCol, iRow, uLines, uColumns, 0, uTargetImage );
				else if( (iCol == 0) || (iCol == uColumns - 1 ) )
					SetImagePointValue( iCol, iRow, uLines, uColumns, 0, uTargetImage );
				else
				{
					uGrey = (abs(-GetImagePointValue( iCol - 1, iRow - 1, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol-1, iRow, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol-1, iRow+1, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol+1, iRow-1, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol+1, iRow, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol+1, iRow+1, uLines, uColumns, puSourceImage)
						+2*GetImagePointValue( iCol, iRow-1, uLines, uColumns, puSourceImage)
						+2*GetImagePointValue(iCol, iRow, uLines, uColumns, puSourceImage)
						+2*GetImagePointValue(iCol, iRow+1, uLines, uColumns, puSourceImage)) );
					SetImagePointValue( iCol, iRow, uLines, uColumns, uGrey, uTargetImage );
				}
			}
		}
	}
	else
	{
		for( iRow = 0; iRow < uLines; iRow++ )
		{
			for( iCol = 0; iCol < uColumns; iCol++ )
			{
				if( (iRow == 0 ) || (iRow == uLines - 1 ))
					SetImagePointValue( iCol, iRow, uLines, uColumns, 0, uTargetImage );
				else if( (iCol == 0) || (iCol == uColumns - 1 ) )
					SetImagePointValue( iCol, iRow, uLines, uColumns, 0, uTargetImage );
				else
				{
					uGrey = (abs(-GetImagePointValue(iCol-1, iRow-1, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol, iRow-1, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol+1, iRow-1, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol-1, iRow+1, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol, iRow+1, uLines, uColumns, puSourceImage)
						-GetImagePointValue(iCol+1, iRow+1, uLines, uColumns, puSourceImage)
						+ 2*GetImagePointValue(iCol-1, iRow, uLines, uColumns, puSourceImage)
						+ 2*GetImagePointValue(iCol, iRow, uLines, uColumns, puSourceImage)
						+ 2*GetImagePointValue(iCol+1, iRow, uLines, uColumns, puSourceImage)) );
					SetImagePointValue( iCol, iRow, uLines, uColumns, uGrey, uTargetImage );
				}
			}
		}
	}
}
void SLKCalculate(Uint8* puSourceImage, Uint16 uThreshold, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	//	bool bEq = FALSE;

	//SLKTemplateCalculate( puSourceImage, pParams->puSobelTempImageX, pCfgs->m_iHeight, pCfgs->m_iWidth, FALSE );	//????A????
	//SLKTemplateCalculate( puSourceImage, pParams->puSobelTempImageY, pCfgs->m_iHeight, pCfgs->m_iWidth, TRUE );		//闂佺鎷�??A闁烩晪鎷�????

	SquareOfTwoImage( pParams->puSobelTempImageX, pParams->puSobelTempImageY, puSourceImage, pCfgs->m_iHeight, pCfgs->m_iWidth, uThreshold, pCfgs,  pParams); 


	// memcpy((void *)pParams->puAlgTempImage, (void *)puSourceImage, pCfgs->m_iWidth*pCfgs->m_iHeight);
	// memcpy((void *)pParams->puSobelTempImageX, (void *)puSourceImage, pCfgs->m_iWidth*pCfgs->m_iHeight);

	IsolatePointDelete( puSourceImage, pCfgs->m_iHeight, pCfgs->m_iWidth, uThreshold );  

	ProjectSLKToEdge( puSourceImage, pCfgs->m_iHeight, pCfgs->m_iWidth, uThreshold );  

}

inline void	SetImagePointValue(Int16 iCol,Int16 iRow,Uint16 Lines,Uint16 Columns,Uint16 Value,Uint8 *ImagePtr)
{
	*(ImagePtr+iRow*Columns+iCol)=Value;

}

void SquareOfTwoImage(Uint8 *puSourceImageX, Uint8 *puSourceImageY, Uint8 *puTargetImage, Uint16 uLines, Uint16 uColumns, Uint16 uThreshold, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	Int32 iRow, iCol;
	Uint16 uSourceGrey, uTargetGrey, uResult;
	Uint16 uGrey1,uGrey2,val0,val1,val2,val3,val4,val5,val6,val7,val8;
	/*for( iRow = 0; iRow < uLines ; iRow++ )
	{
	for( iCol = 0; iCol < uColumns; iCol++ )
	{
	if( (iRow == 0) || (iRow == uLines - 1) )
	SetImagePointValue( iCol, iRow, uLines, uColumns, 0, puTargetImage );
	else if( (iCol == 0) || (iCol == uColumns - 1) )
	SetImagePointValue( iCol, iRow, uLines, uColumns, 0, puTargetImage );
	else
	{
	uSourceGrey = *( puSourceImageX + iRow * uColumns + iCol );
	uTargetGrey = *( puSourceImageY + iRow * uColumns + iCol );
	uResult = ( uSourceGrey + uTargetGrey ) >> 2;
	if( uResult > pCfgs->CameraCfg.uSquareThreshold )
	uResult = 100;
	else if( uResult < uThreshold )
	uResult = 0;
	SetImagePointValue( iCol, iRow, uLines, uColumns, uResult, puTargetImage );
	}
	}
	}*/

	for( iRow = 0; iRow < uLines ; iRow++ )
	{
		for( iCol = 0; iCol < uColumns; iCol++ )
		{
			if( (iRow == 0) || (iRow == uLines - 1) )
			{
				SetImagePointValue( iCol, iRow, uLines, uColumns, 0, puSourceImageX );
				SetImagePointValue( iCol, iRow, uLines, uColumns, 0, puSourceImageY);
			}
			else if( (iCol == 0) || (iCol == uColumns - 1) )

			{
				SetImagePointValue( iCol, iRow, uLines, uColumns, 0, puSourceImageX );
				SetImagePointValue( iCol, iRow, uLines, uColumns, 0, puSourceImageY);
			}
			else
			{
				val0=GetImagePointValue(iCol, iRow, uLines, uColumns, puTargetImage);
				val1=GetImagePointValue(iCol-1, iRow-1, uLines, uColumns, puTargetImage);
				val2=GetImagePointValue(iCol-1, iRow, uLines, uColumns, puTargetImage);
				val3=GetImagePointValue(iCol-1, iRow+1, uLines, uColumns, puTargetImage);
				val4=GetImagePointValue( iCol, iRow-1, uLines, uColumns, puTargetImage);
				val5=GetImagePointValue(iCol, iRow+1, uLines, uColumns, puTargetImage);
				val6=GetImagePointValue(iCol+1, iRow-1, uLines, uColumns, puTargetImage);
				val7=GetImagePointValue(iCol+1, iRow, uLines, uColumns, puTargetImage);
				val8=GetImagePointValue(iCol+1, iRow+1, uLines, uColumns, puTargetImage);
				uGrey1=abs(-val1-val2-val3-val6-val7-val8+2*val4+2*val0+2*val5);
				//SetImagePointValue( iCol, iRow, uLines, uColumns, uGrey1, puSourceImageX );
				uGrey2=abs(-val1-val3-val4-val5-val6-val8+2*val2+2*val0+2*val7);
				//SetImagePointValue( iCol, iRow, uLines, uColumns, uGrey2, puSourceImageY );
				uResult = ( uGrey1 + uGrey2 ) >> 2;
				if( uResult > pCfgs->CameraCfg.uSquareThreshold )
					uResult = 100;
				else if( uResult < uThreshold )
					uResult = 0;
				SetImagePointValue( iCol, iRow, uLines, uColumns, uResult, puSourceImageX );
			}
		}
	}
	memcpy(puTargetImage,puSourceImageX,uLines*uColumns);
}

void ProjectSLKToEdge(Uint8* puImage, Uint16 uLines, Uint16 uColumns, Uint16 uThreshold)
{
	Uint16 uGrey2 = 0, uGrey = 0;
	Int16 m = 0, k = 0, j, nEndRow, nPosSum = 0, nValidPtCount = 0, nValidLine = 0;
	Int16 iRow, iCol;
	bool bBeginFind = FALSE;
	Int16 nChangedRow = 0;

	for( iRow = 1; iRow < uLines - 1 ; iRow++ )
	{
		nValidPtCount = 0;
		nPosSum = 0;
		uGrey2 = 0;
		m = k = 0;
		for( iCol = 1; iCol < uColumns - 1; iCol++ )
		{
			uGrey = GetImagePointValue(iCol, iRow, uLines, uColumns, puImage);
			if( uGrey > 0 )//113///10
			{
				uGrey2 = uGrey2 + uGrey;
				m++;
				k += iCol;
			}
			/*else if ( m >= 3 )
			{
			break;
			}*/
			if( GetImagePointValue( iCol, iRow , uLines, uColumns, puImage ) > 0 )
			{
				nValidPtCount++;
				nPosSum += iCol;//505+????
			}
		}

		//calculate the threshold value
		if ( m >= 3 )
		{
			uGrey2 = uGrey2 / m;
			k = k / m;
			if( uGrey2 < uThreshold)
				uGrey2 = 0;
		}
		else
		{
			uGrey2 = 0;
			m = k = 0;
		}
		/*??A?I??????DD?*/
		if( nValidPtCount >= 2 && ( nValidPtCount < (uColumns - 6) ) )
		{
			nPosSum = (Int16)nPosSum / nValidPtCount;
		}
		else if( nValidPtCount >= (uColumns - 6) )
			nPosSum = (Int16)(uColumns / 2);
		SetImagePointValue( 0, iRow, uLines, uColumns, (Uint16)nPosSum, puImage );
		//SetImagePointValue( 0, iRow, uLines, uColumns, m, puImage );
		/////////////////////////////
		SetImagePointValue( uColumns - 1, iRow, uLines, uColumns, uGrey2, puImage );
		/*??I?3I?闁荤噦鎷�????*/
		if( uGrey2 > 0 )
		{
			nValidLine++;
			if( nValidLine > VEHICLE_MIN_ROW )
				SetImagePointValue( 0, 0, uLines, uColumns, (Uint16)iRow, puImage );
		}
		else
			nValidLine = 0;
		/////////////////////////////////////////////////////
	}

	/*for( iRow = uLines - 2; iRow > 10; iRow-- )
	{
		if( !bBeginFind )
		{
			uGrey = GetImagePointValue( uColumns - 1, iRow , uLines, uColumns, puImage );
			if( GetImagePointValue( uColumns - 1, iRow , uLines, uColumns, puImage ) != 0 )
				bBeginFind = TRUE;
		}
		else
		{
			if( GetImagePointValue( uColumns - 1, iRow , uLines, uColumns, puImage ) == 0 )
			{
				if( nChangedRow < 5 )
				{
					nEndRow = iRow + 5;
					if( nEndRow > uLines - 2 )
						nEndRow = uLines - 2;
					for( j = iRow + 1; j < nEndRow; j++ )
						SetImagePointValue( uColumns - 1, iRow, uLines, uColumns, 0, puImage );
				}
				break;
			}
			else
				nChangedRow++;
		}
	}*/
}


///////////////////////////////////////////////

/*
void	fSetDetectStateFlag(Uint16 DetectNum,Uint16 Flag, Uint16 nChannelNum,LOCALBUF0 *localBuf0,  void *outArgs){
div_t	Position;
Int16		Bit=1;
Position=div(((Int16)DetectNum-1),16);
Bit=Bit<<Position.rem;
if(Flag==TRUE){
localBuf0->CameraLocalPara[nChannelNum].DetectStateFlag[Position.quot]|=Bit;
}
else{
localBuf0->CameraLocalPara[nChannelNum].DetectStateFlag[Position.quot]&=~Bit;
}

////2005 7 Pulse trigger set value
localBuf0->UploadResultsBuf.DetectorStaFlag[Position.quot] \
= localBuf0->CameraLocalPara[nChannelNum].DetectStateFlag[Position.quot];

}
*/
inline Uint16	GetImagePointValue(Int16 iCol,Int16 iRow,Uint16 Lines,Uint16 Columns,Uint8 *ImagePtr)
{
	return (Uint16)*(ImagePtr+iRow*Columns+iCol);
}

SPEED_COUNTER_STATUS GetSpeedCounterStatus( Uint16 uCounter, Uint16 uThisLaneID, Uint16 nChannelNum ,ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	SPEED_COUNTER_STATUS SpeedCounterStatus;
	memset((void*)&SpeedCounterStatus, 0, sizeof(SPEED_COUNTER_STATUS));
	switch( uCounter )
	{
	case THIS_LANE_COUNTER:
		SpeedCounterStatus = pCfgs->CameraLocalPara.guSpeedCounterStatus[uThisLaneID-1];
		break;
	case LEFT_LANE_COUNTER:
		{
			if( uThisLaneID > 1 )
				SpeedCounterStatus = pCfgs->CameraLocalPara.guSpeedCounterStatus[uThisLaneID-2];
			break;
		}
	case RIGHT_LANE_COUNTER:
		{
			if( uThisLaneID < pCfgs->CameraCfg.LaneAmount )
				//SpeedCounterStatus = pCameraLocalPara[nChannelNum]->guSpeedCounterStatus[uThisLaneID+1];  //
				SpeedCounterStatus = pCfgs->CameraLocalPara.guSpeedCounterStatus[uThisLaneID];  //  20130708
			break;
		}
	default:
		break;
	}

	return SpeedCounterStatus;
}

Int16 nLengthInfoGetSize(struct cSpeedDetectStruct* pSpeedDetectStruct)
{
	return (Int16)(pSpeedDetectStruct->nLengthInfoTop + 1 );
}

Int16 nLengthGetVehicleHeadRow( Uint8* puSobelImage, Uint16* pnStartCalcuRow, VEHICLE_LENGTHINFO* pVehicleLengthInfo, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	int nContinueRow = 0, nStartRow, nEndRow,  nRow;
	unsigned int uThreshold;
	int khcount=0;
	uThreshold = 1;//1

	nStartRow = 0;
	nEndRow = 0;

	if( pCfgs->m_iHeight - (*(pnStartCalcuRow)) < 30 )//5
		return 0;

	for( nRow = (*(pnStartCalcuRow)); nRow < pCfgs->m_iHeight-1 ; nRow++ )
	{
		if( GetImagePointValue( pCfgs->m_iWidth -1 , nRow , pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage) > uThreshold )
		{
			if( nStartRow == 0 )
				nStartRow = nRow;
			nContinueRow++;
			nEndRow = nRow;//0;
			khcount=0;
		}
		/*else if( (nStartRow != 0) && (nContinueRow > VEHICLE_MIN_ROW) )//??1????aI???1?I?DD???A????3?
		{
		nEndRow = nRow - 1;
		pVehicleLengthInfo->uEndRow = nStartRow;
		pVehicleLengthInfo->fLengthSum += (float)(nEndRow - nStartRow);
		pVehicleLengthInfo->uLengthCount++;//3I???3A??????????y??

		return nEndRow;
		}
		else if( nStartRow != 0 )//???闁活叏鎷�???aI???1?I?DD????闁绘繐鎷�?
		nEndRow = nRow;*/
		else 
		{
			khcount++;
		}
		if ( (nStartRow != 0) &&(khcount>5))
		{
			nEndRow = nRow - 1;
			pVehicleLengthInfo->uEndRow = nStartRow;
			pVehicleLengthInfo->fLengthSum += (float)(nEndRow - nStartRow);
			pVehicleLengthInfo->uLengthCount++;//3I???3A??????????y??

			return nEndRow;
		}
	}
	pVehicleLengthInfo->uEndRow = nStartRow;
	pVehicleLengthInfo->fLengthSum += (float)(nEndRow - nStartRow);
	pVehicleLengthInfo->uLengthCount++;//3I???3A??????????y??
	return nEndRow;

}

void LengthInfoAdd(struct cSpeedDetectStruct* pSpeedDetectStruct, VEHICLE_LENGTHINFO* pVehicleLengthInfo)

{
	pSpeedDetectStruct->nLengthInfoTop++;
	if( pSpeedDetectStruct->nLengthInfoTop > (LENGTHINFO_MAXSIZE - 1) )
	{
		pSpeedDetectStruct->nLengthInfoTop = LENGTHINFO_MAXSIZE - 1;
		return;
	}
	memcpy((void *)&pSpeedDetectStruct->VehicleLengthInfo[pSpeedDetectStruct->nLengthInfoTop], (void *)pVehicleLengthInfo, sizeof(VEHICLE_LENGTHINFO));
}

void LengthGetVehicleInInfo( Uint8* puSobelImage, struct cSpeedDetectStruct* pSpeedDetectStruct, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	unsigned uThreshold;//, uMaxChangedRow;
	Int16 i, nRow, nValidLineCount;//, nInvalidLineCount, nChangedRow;

	bool bGetEmptyRow;
	Int16 nStartAddInRow = 5;

	uThreshold = 1;


	bGetEmptyRow = FALSE;
	if( pSpeedDetectStruct->uLengthBeginInRows > 10 )  //5
	{
		bGetEmptyRow = TRUE;
	}
	else
		pSpeedDetectStruct->uLengthBeginInRows = 0;

	for( nRow = 1 + nStartAddInRow ; nRow < 45 + nStartAddInRow ; nRow++ )  //8
	{
		if( (GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) > uThreshold) &&
			(bGetEmptyRow == FALSE) )
		{
			if( nRow == 1 + nStartAddInRow )
				pSpeedDetectStruct->uLengthBeginInRows = 1;
			else if( pSpeedDetectStruct->uLengthBeginInRows == 0 )
				return;
			else
				pSpeedDetectStruct->uLengthBeginInRows++;
		}
		else if( (GetImagePointValue( pCfgs->m_iWidth - 1 , nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) < uThreshold) &&
			(bGetEmptyRow == TRUE) )
		{
			pSpeedDetectStruct->uLengthEndEmptyRow++;
			if( pSpeedDetectStruct->uLengthEndEmptyRow >= 10 ) 

			{
				nValidLineCount = 0;  
				for( i = nRow; i < (Int16)pCfgs->m_iHeight; i++ )
				{
					if( GetImagePointValue( pCfgs->m_iWidth - 1, i, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) > uThreshold )
					{
						nValidLineCount++;
						pSpeedDetectStruct->uLengthBeginInRows = i;
					}
					else if( nValidLineCount != 0 )
						break;
				}
				if( pSpeedDetectStruct->uLengthBeginInRows >= (pCfgs->m_iHeight - 10) )
				{
					//pSpeedDetectStruct->uLengthEndEmptyRow = 0;
					//pSpeedDetectStruct->uLengthBeginInRows = 0;
				}
			}
		}
	}
}

void LengthInfoRemoveAt( struct cSpeedDetectStruct* pSpeedDetectStruct, Int16 nIndex )
{
	Int16 i;
	if( nIndex <= pSpeedDetectStruct->nLengthInfoTop )
	{
		for( i = nIndex; i <= (pSpeedDetectStruct->nLengthInfoTop - 1); i++ )
		{

			memcpy((void *)&pSpeedDetectStruct->VehicleLengthInfo[i], (void *)&pSpeedDetectStruct->VehicleLengthInfo[i+1], sizeof(VEHICLE_LENGTHINFO));
		}
		pSpeedDetectStruct->VehicleLengthInfo[pSpeedDetectStruct->nLengthInfoTop].uEndRow = 0;
		pSpeedDetectStruct->VehicleLengthInfo[pSpeedDetectStruct->nLengthInfoTop].uLengthCount = 0;
		pSpeedDetectStruct->VehicleLengthInfo[pSpeedDetectStruct->nLengthInfoTop].fLengthSum = 0;
		pSpeedDetectStruct->VehicleLengthInfo[pSpeedDetectStruct->nLengthInfoTop].uThreshold = 0;
		pSpeedDetectStruct->nLengthInfoTop--;
	}
}

Uint16 uGetMaxUnEmptyRow(Uint8* puImage, Int16 nStartLine,Int16 nEndLine, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	Int16 nRow;

	for( nRow = nEndLine - 1; nRow >= nStartLine; nRow-- )
	{
		if( GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puImage ) != 0 )
			return  (Uint16)nRow;
	}
	return 1;
}


bool bIsSobleBlockUnChanged(Uint8 *pucSobelCurrentImage, int nStartLine, int nEndLine,int flag, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	int nSpeedStartLine = nStartLine;
	int nSpeedEndLine = nEndLine;
	int nRow;
	int k;
	int nUnChangedRow = 0;
	Uint16 value;//113
	Uint16 ucThreshold;

	if (flag == 0)
	{
		//ucThreshold = 10;
		ucThreshold = pCfgs->CameraCfg.uInThreshold;  // by david 20130910
		k = 1;
	}
	else
	{
		ucThreshold =1;

		k = 1;

	}
	////20140113	
	nUnChangedRow = 0;
	value =0;

	if (flag==1)//night
	{
		for(  nRow = 0; nRow <0.6*pCfgs->m_iHeight; nRow++ )//txl,20151117
		{
			value = GetImagePointValue( pCfgs->m_iWidth -1 , nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, pucSobelCurrentImage );
			if( value < ucThreshold )
			{
				nUnChangedRow++;
			}

			if( (nUnChangedRow * k) >= 0.6*pCfgs->m_iHeight-3 )//3
			{
				return TRUE;
			}
		}
	}
	else
	{
		for(  nRow = nSpeedStartLine; nRow < nSpeedEndLine; nRow++ )
		{
			value = GetImagePointValue( pCfgs->m_iWidth -1 , nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, pucSobelCurrentImage );
			//value1=GetImagePointValue( 0 , nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, pucSobelCurrentImage );//113
			if( value < ucThreshold )//||value1<pCfgs->m_iWidth/2)
			{
				nUnChangedRow++;
			}
			if( nUnChangedRow * k >= (nSpeedEndLine - nSpeedStartLine-10) )//3
			{
				return TRUE;
			}
		}
	}


	return FALSE;
}


bool bIsSobleBlockChanged(Uint8 *pucSobelCurrentImage, Int16 nStartLine, Int16 nEndLine, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	//	Int16 nSpeedStartLine = nStartLine;
	//	Int16 nSpeedEndLine = nEndLine;
	Int16 nRow;
	Uint16 ucThreshold = 1;
	Int16 nUnChangedRow = 0;
	//	Int16 nChangedRow = 0;
	Uint16 value = 0;

	for(  nRow = nStartLine; nRow < nEndLine; nRow++ )
	{
		value = GetImagePointValue( pCfgs->m_iWidth -1 , nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, pucSobelCurrentImage );

		if( value < ucThreshold )
		{
			nUnChangedRow++;
		}
		if( nUnChangedRow * 4 > (nEndLine - nStartLine) )
			//if( nUnChangedRow*2 > (nEndLine - nStartLine) )
		{
			return FALSE;
		}
	}

	return TRUE;
}

float dAverageGreyInBlock(Uint8 *puImage, Int16 nFromBlockID, Int16 nToBlockID, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	Int16 nPointSum=0, nStartRow=0, nEndRow=0, nStartCol=0, nEndCol=0, iRow=0, iCol=0;
	float 	gdAverGrey=0,gdSum=0;

	nStartRow = nFromBlockID;
	nEndRow = nToBlockID;
	nStartCol = 1;



	nEndCol = pCfgs->m_iWidth-1;
	gdSum = 0;


	for( iRow = nStartRow; iRow < nEndRow; iRow++ )
	{
		for( iCol = nStartCol; iCol < nEndCol; iCol++ )
		{
			gdAverGrey = (float)GetImagePointValue(iCol, iRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puImage );
			gdSum += gdAverGrey;
			nPointSum++;
		}
	}
	return (float)(gdSum/nPointSum);
}

Int16 nSpeedGetVehicleHeadRow( Uint8* puSobelImage, Uint16* pnStartCalcuRow, VEHICLE_SPEEDINFO* pVehicleSpeedInfo, int flag, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	int nContinueRow = 0;
	int nStartRow, nEndRow,  nRow;
	unsigned int uThreshold;
	int khcount=0;

	uThreshold = 5;//1;
	nStartRow = 0;
	nEndRow = 0;


	if( pCfgs->m_iHeight - (*(pnStartCalcuRow)) < 5 )//5
		return 0;
	if (flag==0)
	{
		for( nRow = (*(pnStartCalcuRow)); nRow < pCfgs->m_iHeight-1 ; nRow++ )
		{
			if( GetImagePointValue( pCfgs->m_iWidth -1 , nRow , pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage) > uThreshold )
			{
				if( nStartRow == 0 )
					nStartRow = nRow;
				nContinueRow++;
				nEndRow = nRow;//0;
				khcount=0;
			}
			else if(nContinueRow>0)
			{
				khcount++;
			}

			if( (nContinueRow > 10) &&(khcount>4) ) //???3D?D?3I缂侇垽鎷�?3I缂侇垰鍤�??yA?闂佺鎷�???khcount?D?I??D?3I?I???闁活叏鎷�??闁荤儑鎷�?閻忓繑鐗幏锟�??
			{
				nEndRow = nRow - 1;///MOVE ADDED HEAD?
				//pVehicleSpeedInfo->uHeadRow = nStartRow;
				return nEndRow;
			}
		}
	}
	else
	{
		uThreshold = pCfgs->m_iWidth / 5;
		for( nRow = (*(pnStartCalcuRow)); nRow <pCfgs->m_iHeight-1; nRow++ )
		{
			if (nRow>250)
			{
				int a=GetImagePointValue( pCfgs->m_iWidth -1 , nRow , pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage);
			}
			if( GetImagePointValue( pCfgs->m_iWidth -1 , nRow , pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage) > uThreshold )
			{
				if( nStartRow == 0 )
					nStartRow = nRow;
				nContinueRow++;
				nEndRow = nRow;//0;
				khcount=0;
			}
			else if( nStartRow != 0 ) //???D?D?3I缂侇垽鎷�?3I缂侇垰鍤�??yA?闂佺鎷�??khcount?D?I??D?3I?I??缂佹鎷�??闁荤儑鎷�?閻忓繑鐗幏锟�??
			{
				nEndRow = nRow-1;///MOVE ADDED HEAD?
				//pVehicleSpeedInfo->uHeadRow = nStartRow;
				return nEndRow;
			}
		}
	}
	return nEndRow;

}

bool bVehicleHeadIn(Uint8* puSourceImage, Uint8* puTargetImage, Int16 nStartLine,Int16 nEndLine, Int16 nRowThreshold, Int16 nColThreshold, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	Int16 nValidRowCount = 0, nValidColPtCount = 0;
	Int16 nRow, nCol;
	bool bVehicleCome = FALSE;

	for( nRow = nStartLine; nRow < nEndLine; nRow++ )
	{
		nValidColPtCount = 0;
		for( nCol = 0; nCol < pCfgs->m_iWidth; nCol++ )
		{
			if( ( (*(puSourceImage + nRow*pCfgs->m_iWidth + nCol)) * (*(puTargetImage + nRow*pCfgs->m_iWidth + nCol)) ) == 0 )
				SetImagePointValue( nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, 0, puTargetImage );
			else///setpoint?
				nValidColPtCount++;
		}
		if( nValidColPtCount < nColThreshold )
		{
			if( nValidRowCount < nRowThreshold )
				nValidRowCount = 0;
		}
		else
			nValidRowCount++;
	}
	if( nValidRowCount >= nRowThreshold )
		bVehicleCome = TRUE;

	if( bVehicleCome == FALSE )
	{
		for( nRow = nStartLine; nRow < nEndLine; nRow++ )
		{
			for( nCol = 0; nCol < pCfgs->m_iWidth; nCol++ )
			{
				SetImagePointValue( nCol, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, 0, puTargetImage );
			}
		}
	}
	return bVehicleCome;
}

void SpeedInfoAdd(struct cSpeedDetectStruct* pSpeedDetectStruct, VEHICLE_SPEEDINFO* pVehicleSpeedInfo)
{
	pSpeedDetectStruct->nSpeedInfoTop++;

	if( pSpeedDetectStruct->nSpeedInfoTop > (SPEEDINFO_MAXSIZE - 1) )
	{
		pSpeedDetectStruct->nSpeedInfoTop = SPEEDINFO_MAXSIZE - 1;
		return;
	}
	memcpy((void *)&pSpeedDetectStruct->VehicleSpeedInfo[pSpeedDetectStruct->nSpeedInfoTop], (void *)pVehicleSpeedInfo, sizeof(VEHICLE_SPEEDINFO));

}

Int16 nSpeedInfoGetSize(struct cSpeedDetectStruct* pSpeedDetectStruct)
{
	return (Int16)( pSpeedDetectStruct->nSpeedInfoTop + 1 );
}

void SpeedInfoRemoveAt( struct cSpeedDetectStruct* pSpeedDetectStruct, Int16 nIndex )
{
	pSpeedDetectStruct->nSpeedInfoTop = -1;
}

void SpeedGetVehicleInInfo( Uint8* puSobelImage, struct cSpeedDetectStruct* pSpeedDetectStruct, int flag, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	unsigned uThreshold;
	int i, nRow, nInvalidLineCount;
	bool bGetEmptyRow;
	int nStartAddInRow = 5;//30

	uThreshold = 5;//1

	bGetEmptyRow = FALSE;
	if (flag==0)
	{
		if( pSpeedDetectStruct->uSpeedBeginInRows > 10 )//5
		{
			bGetEmptyRow = TRUE;
		}
		else
			pSpeedDetectStruct->uSpeedBeginInRows = 0;

		for( nRow = nStartAddInRow + 1; nRow < nStartAddInRow + 48; nRow++ )//8,31-78,48
		{
			if( (GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) > uThreshold) &&
				(bGetEmptyRow == FALSE) ) 
			{
				if( nRow == 1 + nStartAddInRow )
					pSpeedDetectStruct->uSpeedBeginInRows = 1;
				else if( pSpeedDetectStruct->uSpeedBeginInRows == 0 )
					return;
				else
					pSpeedDetectStruct->uSpeedBeginInRows++;
			}

			else if( (GetImagePointValue( pCfgs->m_iWidth - 1 , nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage )> uThreshold) &&
				(bGetEmptyRow == TRUE) ) 
			{
				pSpeedDetectStruct->uSpeedEndEmptyRow++;

				if( pSpeedDetectStruct->uSpeedEndEmptyRow >= 10 )
				{
					nInvalidLineCount = 0;
					for( i = nRow; i< pCfgs->m_iHeight - 1; i++ )
					{
						if( GetImagePointValue( pCfgs->m_iWidth - 1, i, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) > uThreshold )
						{
							nInvalidLineCount++;
							//pSpeedDetectStruct->uSpeedBeginInRows = i;
							pSpeedDetectStruct->uSpeedalsoInRows=i;
						}
						else if( nInvalidLineCount != 0 )

							break;
					}

					if( pSpeedDetectStruct->uSpeedalsoInRows >= (pCfgs->m_iHeight - 10) ){//10
						pSpeedDetectStruct->uSpeedEndEmptyRow = 0;
						pSpeedDetectStruct->uSpeedalsoInRows = 0;
						//pSpeedDetectStruct->uSpeedBeginInFrame=0;
					}
				}
			}
		}
	}
	else
	{
		uThreshold=5;
		if( pSpeedDetectStruct->uSpeedBeginInRows > 5 )//5
		{
			bGetEmptyRow = TRUE;
		}
		else
			pSpeedDetectStruct->uSpeedBeginInRows = 0;
		for( nRow = nStartAddInRow + 1; nRow < nStartAddInRow + 48; nRow++ )//8,31-78,48
		{
			if( (GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) > uThreshold) &&
				(bGetEmptyRow == FALSE) ) /*??3?3I???????I?DD??y*///I????DD2??a??
			{
				if( nRow == 1 + nStartAddInRow )
					pSpeedDetectStruct->uSpeedBeginInRows = 1;
				else if( pSpeedDetectStruct->uSpeedBeginInRows == 0 )
					return;
				else
					pSpeedDetectStruct->uSpeedBeginInRows++;
			}
			else if((bGetEmptyRow == TRUE) ) 
			{
				for( i = nRow; i< pCfgs->m_iHeight - 1; i++ )
				{
					if( GetImagePointValue( pCfgs->m_iWidth - 1, i, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) > uThreshold )
					{
						pSpeedDetectStruct->uSpeedEndEmptyRow++;
						nInvalidLineCount++;
						pSpeedDetectStruct->uSpeedBeginInRows = i;
						pSpeedDetectStruct->uSpeedalsoInRows=i;
					}
					else if( nInvalidLineCount != 0 )
						break;
				}

				if( pSpeedDetectStruct->uSpeedalsoInRows >= (pCfgs->m_iHeight - 10) ){//10
					pSpeedDetectStruct->uSpeedEndEmptyRow = 0;
					pSpeedDetectStruct->uSpeedalsoInRows = 0;
					//pSpeedDetectStruct->uSpeedBeginInFrame=0;
				}
			}
		}
	}
}

bool bIsSpeedCounterChanged( Uint8* puSobelImage, Uint16 uSpeedCounterStartRow, SPEED_COUNTER_STATUS* pSpeedCounterStatus,int flag,  ALGCFGS *pCfgs, ALGPARAMS *pParams )
{
	//	int nLeftBlockColumn = pCfgs->m_iWidth / 3, nRightBlockColumn = pCfgs->m_iWidth * 2 / 3;
	//int nPosSum = 0, nRowCount = 0;
	int nRow;
	int nUnChangedRow = 0;
	//int startrow = 0;
	//int nullRowNum = 0, maxNullRow = 0;
	//	int  maxNullRowPos = 0;
	unsigned int uThreshold ;//1
	int k;

	if (flag == 0)
	{
		//uThreshold = 2;  //by david 20130905 from 5 to 20
		uThreshold = pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold;  //by david 20130905 
		k = 2;
		//startrow = uSpeedCounterStartRow;


		for( nRow = uSpeedCounterStartRow + 2; nRow < pCfgs->m_iHeight -1; nRow++ )
		{
			if( GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) < uThreshold )
			{


				//nullRowNum++;
				nUnChangedRow++;
			}
			/*else
			{
				if (nullRowNum > maxNullRow){
					maxNullRow = nullRowNum;
					//pParams->NormalDetectCfg.cSpeedDesc.position1 = nRow-maxNullRow+1;
					//pParams->NormalDetectCfg.cSpeedDesc.position2 = nRow;
				}
				nullRowNum=0;
			}*/

		}

		//if( nUnChangedRow * k > pParams->NormalDetectCfg.cSpeedDesc.uVehicleHeadMinRow )
		if( (nUnChangedRow * k) > (pCfgs->m_iHeight - uSpeedCounterStartRow - 2) ) 
		{
			//pSpeedCounterStatus->bRightDestroyed = FALSE;
			//pSpeedCounterStatus->bLeftDestroyed = FALSE;
			return FALSE;	
		}
	}
	else
	{
		////20140110 zcj
		uThreshold=pCfgs->m_iWidth/5;
		////	    uThreshold = pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold;   

		k=1;
		for( nRow = uSpeedCounterStartRow + 2; nRow < pCfgs->m_iHeight -1; nRow++ )
		{
			if( GetImagePointValue( pCfgs->m_iWidth - 1, nRow, pCfgs->m_iHeight, pCfgs->m_iWidth, puSobelImage ) < uThreshold )
				nUnChangedRow++;
		}
		//startrow=m_iHeight-100;
		if( nUnChangedRow*k > (pCfgs->m_iHeight-uSpeedCounterStartRow-5))//
		{
			return FALSE;
		}
		//startrow=m_iHeight-100;

	}
	return TRUE;

}

float GetLenOfTwoPoint(CPoint* Cmp_Dot1, CPoint* Cmp_Dot2)
{
	/*??????Cmp_Dot1??Cmp_Dot2I?3A???*/
	float i,j;
	j = 0;
	i=-1;
	i = (float)((Cmp_Dot1->x - Cmp_Dot2->x) * (Cmp_Dot1->x - Cmp_Dot2->x));
	i = (float)( i + (Cmp_Dot1->y - Cmp_Dot2->y) * (Cmp_Dot1->y - Cmp_Dot2->y));

	j = (float) ( sqrt( i ) );
	return (float)j;
}

/*???a??DT?y??A缂佸鎷�?*/
CPoint ptGetDot(CPoint* ptUpLeft, CPoint* ptUpRight, CPoint* ptDownRight, CPoint* ptDownLeft, Int16 nColNum, Uint32 * ptStorePlace)
{
	CPoint ptRowAndCol, ptTemp, ptOne, ptTwo;
	Uint16	MaxX,MaxY,/*cNewX,*/ uMaxLines;
	int iRow_Num;
	int iRow, iCol;//DD????D
	float fSlope;//DA???
	float temp1, temp2;
	float	fBDivder,fDivder;
	Uint32 	ImageAddr;
	Uint8 	cState;

	cState=0;
	ptOne.x = ptUpLeft->x;
	ptOne.y = ptUpLeft->y;
	ptTwo.x = ptUpRight->x;
	ptTwo.y = ptUpRight->y;

	temp1 =GetLenOfTwoPoint( &ptOne, &ptTwo );//????

	ptOne.x = ptDownLeft->x;
	ptOne.y = ptDownLeft->y;
	ptTwo.x = ptDownRight->x;
	ptTwo.y = ptDownRight->y;
	temp2 =GetLenOfTwoPoint( &ptOne, &ptTwo );//?I??

	if	(temp1 >= temp2)
	{
		cState =0x80;
	};
	if	((cState & 0x80) ==0x80)//???I?
	{
		if	((ptUpLeft->x - ptUpRight->x) ==0)
		{
			fSlope = 2.0;
		}
		else
		{
			if( abs(ptUpLeft->y - ptUpRight->y) > abs(ptUpLeft->x - ptUpRight->x) )//??1
				fSlope = 1.5;
			else
				fSlope = 0;

		};

		ptOne.x = ptDownLeft->x;
		ptOne.y = ptDownLeft->y;
		ptTwo.x = ptDownRight->x;
		ptTwo.y = ptDownRight->y;

		temp1 =GetLenOfTwoPoint( &ptOne, &ptTwo );//?I??

		// by david
		if ( (temp1 > 0) && (temp1 < (nColNum - 1)))
		{
			nColNum = (int)( temp1 + 1 );//???
		};

		// if (temp1 > 0)
		// {
		//nColNum = (int)( temp1 + 1 );//??????D??y
		// nColNum = (int)( temp1 );//??????D??y, by david 20130910
		// }
	}
	else//????I?
	{
		//I?婵℃鎷�?1???
		if	((ptDownRight->x - ptDownLeft->x) ==0)
		{
			fSlope = 2.0;
		}
		else
		{
			//			fSlope =(float)((float)(ptDownLeft->y - ptDownRight->y)/(float)(ptDownLeft->x - ptDownRight->x));
			if( abs(ptDownLeft->y - ptDownRight->y) > abs(ptDownLeft->x - ptDownRight->x) )//??1
				fSlope = 1.5;
			else
				fSlope = 0;
		};

		ptOne.x = ptUpLeft->x;
		ptOne.y = ptUpLeft->y;
		ptTwo.x = ptUpRight->x;
		ptTwo.y = ptUpRight->y;

		temp1 =GetLenOfTwoPoint( &ptOne, &ptTwo );//????

		// by david
		if ( (temp1 > 0) && (temp1 < (nColNum -1)))
		{
			nColNum = (int)( temp1 + 1 );//???
		};
	};

	uMaxLines = (Uint16)( MaxDotsInDetect / nColNum );
	cState =0x80;
	iRow_Num =abs(ptUpLeft->y -ptDownLeft->y);
	if	(iRow_Num > abs(ptUpRight->y - ptDownRight->y))//??
	{
		iRow_Num =abs(ptUpRight->y - ptDownRight->y);//??
		cState =cState +0x40;//2?闁绘洩鎷�?1?a3A妞ゃ儻鎷�?
	};
	//	if	(abs(fSlope) >= 1)
	//	{
	//		//DA????????I?=1
	//		iRow_Num =abs(ptUpLeft->x -ptDownLeft->x);
	//		if	(iRow_Num >= abs(ptUpRight->x - ptDownRight->x))//??
	//		{
	//			iRow_Num =abs(ptUpRight->x - ptDownRight->x);//??
	//			cState =cState +0x40;//2?闁绘洩鎷�?1?a3A妞ゃ儻鎷�?
	//		};
	//	}
	//	else//???1
	//	{
	//		//DA????????I?1
	//		iRow_Num =abs(ptUpLeft->y -ptDownLeft->y);
	//		if	(iRow_Num >= abs(ptUpRight->y - ptDownRight->y))//??
	//		{
	//			iRow_Num =abs(ptUpRight->y - ptDownRight->y);//??
	//			cState =cState +0x40;//2?闁绘洩鎷�?1?a3A妞ゃ儻鎷�?
	//		};
	///	};
	//iRow_Num++;    //by david
	if( iRow_Num > uMaxLines )
		iRow_Num = uMaxLines;

	//??????闁荤儑鎷�?婵℃鎷�?o??A??闁绘洩鎷�???I???A??
	switch (cState &0xc0)
	{
	case	0x00:
		//I?婵℃鎷�?1??妞ゆ帪鎷�??2?闁绘洩鎷�?1???
		fBDivder=(((float)ptDownRight->y -(float)ptUpLeft->y)*((float)ptDownRight->x -(float)ptDownLeft->x)*((float)ptUpRight->x -(float)ptDownRight->x)\
			+ ((float)ptDownRight->y -(float)ptDownLeft->y)*((float)ptUpRight->x -(float)ptDownRight->x)*(float)ptUpLeft->x \
			- ((float)ptUpRight->y -(float)ptDownRight->y)*((float)ptDownRight->x -(float)ptDownLeft->x)*(float)ptDownRight->x);
		fDivder=(((float)ptDownRight->y -(float)ptDownLeft->y)*((float)ptUpRight->x -(float)ptDownRight->x) - ((float)ptUpRight->y -(float)ptDownRight->y)*((float)ptDownRight->x -(float)ptDownLeft->x));
		ptUpRight->x =(unsigned int)(fBDivder/fDivder);

		fBDivder=(((float)ptDownRight->x -(float)ptUpLeft->x)*((float)ptDownRight->y -ptDownLeft->y)*((float)ptUpRight->y -(float)ptDownRight->y)\
			+ ((float)ptDownRight->x -(float)ptDownLeft->x)*((float)ptUpRight->y -(float)ptDownRight->y)*(float)ptUpLeft->y \
			- ((float)ptUpRight->x -(float)ptDownRight->x)*((float)ptDownRight->y -(float)ptDownLeft->y)*(float)ptDownRight->y);
		fDivder=(((float)ptDownRight->x -(float)ptDownLeft->x)*((float)ptUpRight->y -(float)ptDownRight->y) - ((float)ptUpRight->x -(float)ptDownRight->x)*((float)ptDownRight->y -(float)ptDownLeft->y));
		ptUpRight->y =(Uint16)(fBDivder/fDivder);
		break;
	case	0x40:
		fBDivder=(((float)ptDownLeft->y -(float)ptUpRight->y)*((float)ptDownRight->x -(float)ptDownLeft->x)*((float)ptDownLeft->x -(float)ptUpLeft->x)\
			+ ((float)ptDownRight->y -(float)ptDownLeft->y)*((float)ptDownLeft->x -(float)ptUpLeft->x)*(float)ptUpRight->x \
			- ((float)ptDownLeft->y -(float)ptUpLeft->y)*((float)ptDownRight->x -(float)ptDownLeft->x)*(float)ptDownLeft->x);
		fDivder=(((float)ptDownRight->y -(float)ptDownLeft->y)*((float)ptDownLeft->x -(float)ptUpLeft->x) - ((float)ptDownLeft->y -(float)ptUpLeft->y)*((float)ptDownRight->x -(float)ptDownLeft->x));
		ptUpLeft->x =(unsigned int)(fBDivder/fDivder);

		fBDivder=(((float)ptDownLeft->x -(float)ptUpRight->x)*((float)ptDownRight->y -(float)ptDownLeft->y)*((float)ptDownLeft->y -(float)ptUpLeft->y)\
			+ ((float)ptDownRight->x -(float)ptDownLeft->x)*((float)ptDownLeft->y -(float)ptUpLeft->y)*(float)ptUpRight->y \
			- ((float)ptDownLeft->x -(float)ptUpLeft->x)*((float)ptDownRight->y -(float)ptDownLeft->y)*(float)ptDownLeft->y);
		fDivder=(((float)ptDownRight->x -(float)ptDownLeft->x)*((float)ptDownLeft->y -(float)ptUpLeft->y) - ((float)ptDownLeft->x -(float)ptUpLeft->x)*((float)ptDownRight->y -(float)ptDownLeft->y));
		ptUpLeft->y =(Uint16)(fBDivder/fDivder);

		break;
	case	0x80:
		fBDivder=(((float)ptUpRight->y -(float)ptDownLeft->y)*((float)ptUpRight->x -(float)ptUpLeft->x)*((float)ptDownRight->x -(float)ptUpRight->x)\
			+ ((float)ptUpRight->y -(float)ptUpLeft->y)*((float)ptDownRight->x -(float)ptUpRight->x)*(float)ptDownLeft->x \
			- ((float)ptDownRight->y -(float)ptUpRight->y)*((float)ptUpRight->x -(float)ptUpLeft->x)*(float)ptUpRight->x);
		fDivder=(((float)ptUpRight->y -(float)ptUpLeft->y)*((float)ptDownRight->x -(float)ptUpRight->x) - ((float)ptDownRight->y -(float)ptUpRight->y)*((float)ptUpRight->x -(float)ptUpLeft->x));
		ptDownRight->x =(Uint16)(fBDivder/fDivder);

		fBDivder=(((float)ptUpRight->x -(float)ptDownLeft->x)*((float)ptUpRight->y -(float)ptUpLeft->y)*((float)ptDownRight->y -(float)ptUpRight->y)\
			+ ((float)ptUpRight->x -(float)ptUpLeft->x)*((float)ptDownRight->y -(float)ptUpRight->y)*(float)ptDownLeft->y \
			- ((float)ptDownRight->x -(float)ptUpRight->x)*((float)ptUpRight->y -(float)ptUpLeft->y)*(float)ptUpRight->y);
		fDivder=(((float)ptUpRight->x -(float)ptUpLeft->x)*((float)ptDownRight->y -(float)ptUpRight->y) - ((float)ptDownRight->x -(float)ptUpRight->x)*((float)ptUpRight->y -(float)ptUpLeft->y));
		ptDownRight->y =(Uint16)(fBDivder/fDivder);
		break;
	default:
		fBDivder=(((float)ptUpLeft->y -(float)ptDownRight->y)*((float)ptUpRight->x -(float)ptUpLeft->x)*((float)ptDownLeft->x -(float)ptUpLeft->x)\
			+ ((float)ptUpRight->y -(float)ptUpLeft->y)*((float)ptDownLeft->x -(float)ptUpLeft->x)*(float)ptDownRight->x \
			- ((float)ptDownLeft->y -(float)ptUpLeft->y)*((float)ptUpRight->x -(float)ptUpLeft->x)*(float)ptUpLeft->x);
		fDivder=(((float)ptUpRight->y -(float)ptUpLeft->y)*((float)ptDownLeft->x -(float)ptUpLeft->x) - ((float)ptDownLeft->y -(float)ptUpLeft->y)*((float)ptUpRight->x -(float)ptUpLeft->x));
		ptDownLeft->x =(Uint16)(fBDivder/fDivder);

		fBDivder=(((float)ptUpLeft->x -(float)ptDownRight->x)*((float)ptUpRight->y -(float)ptUpLeft->y)*((float)ptDownLeft->y -(float)ptUpLeft->y)\
			+ ((float)ptUpRight->x -(float)ptUpLeft->x)*((float)ptDownLeft->y -(float)ptUpLeft->y)*(float)ptDownRight->y \
			- ((float)ptDownLeft->x -(float)ptUpLeft->x)*((float)ptUpRight->y -(float)ptUpLeft->y)*(float)ptUpLeft->y);
		fDivder=(((float)ptUpRight->x -(float)ptUpLeft->x)*((float)ptDownLeft->y -(float)ptUpLeft->y) - ((float)ptDownLeft->x -(float)ptUpLeft->x)*((float)ptUpRight->y -(float)ptUpLeft->y));
		ptDownLeft->y =(Uint16)(fBDivder/fDivder);

		break;
	};
	//MaxX=720;
	MaxX=FULL_COLS;
	MaxY=FULL_ROWS;
	////printf("row is:%d\t",iRow_Num);
	////printf("col is:%d\t",nColNum);
	////printf("upleft x:%d,y:%d\t",ptUpLeft->x,ptUpLeft->y);
	////printf("upright x:%d,y:%d\t",ptUpRight->x,ptUpRight->y);
	////printf("downleft x:%d,y:%d\t",ptDownLeft->x,ptDownLeft->y);
	////printf("downright x:%d,y:%d\n",ptDownRight->x,ptDownRight->y);
	for( iRow = 0; iRow < iRow_Num; iRow++)
	{
		for( iCol = 0; iCol < nColNum ; iCol++ )
		{
			/*ptTemp.x=(Uint16)( (float)ptUpLeft->x + ((float)ptDownLeft->x - (float)ptUpLeft->x)*iRow/(iRow_Num -1)\
			+((float)ptUpRight->x - (float)ptUpLeft->x)*iCol/(nColNum -1)\
			+((float)ptUpLeft->x + (float)ptDownRight->x - (float)ptUpRight->x - (float)ptDownLeft->x)*iCol/((iRow_Num -1) *(nColNum -1)) );

			ptTemp.y=(Uint16)( (float)ptUpLeft->y + ((float)ptDownLeft->y - (float)ptUpLeft->y)*iRow/(iRow_Num -1)\
			+((float)ptUpRight->y -(float)ptUpLeft->y)*iCol/(nColNum -1)\
			+((float)ptUpLeft->y + (float)ptDownRight->y - (float)ptUpRight->y - (float)ptDownLeft->y)*iCol/((iRow_Num -1) *(nColNum -1)) );*/
			temp1=(float)ptUpRight->x + ((float)ptDownRight->x-(float)ptUpRight->x)*iRow/(iRow_Num -1);
			temp2=(float)ptUpLeft->x + ((float)ptDownLeft->x - (float)ptUpLeft->x)*iRow/(iRow_Num -1);
			ptTemp.x=(Uint16)( temp2+(temp1-temp2)*iCol/(nColNum-1)+((float)ptUpLeft->x + (float)ptDownRight->x - (float)ptUpRight->x - (float)ptDownLeft->x)*iCol/((iRow_Num -1) *(nColNum -1)));

			ptTemp.y=(Uint16)( (float)ptUpLeft->y + ((float)ptDownLeft->y - (float)ptUpLeft->y)*iRow/(iRow_Num -1)\
				+((float)ptUpRight->y -(float)ptUpLeft->y)*iCol/(nColNum -1)\
				+((float)ptUpLeft->y + (float)ptDownRight->y - (float)ptUpRight->y - (float)ptDownLeft->y)*iCol/((iRow_Num -1) *(nColNum -1)) );
			//ptTemp.y=ptUpLeft->y+iRow;
			//if (iRow==0)
			//{
			//	if(iCol ==0)
			//	{
			//		ptUpLeft->x=ptTemp.x;
			//		ptUpLeft->y=ptTemp.y;
			//	}
			//	if (iCol ==(nColNum-1))
			//	{
			//		//temp1=ptTemp.x;
			//		ptUpRight->x=ptTemp.x;
			//		ptUpRight->y=ptTemp.y;
			//	}
			//}
			//if(iRow==(iRow_Num-1))
			//{
			//	if(iCol ==0)
			//	{
			//		ptDownLeft->x=ptTemp.x;
			//		ptDownLeft->y=ptTemp.y;
			//	}
			//	if (iCol ==(nColNum-1))
			//	{
			//		//temp2=ptTemp.x;
			//		ptDownRight->x=ptTemp.x;
			//		ptDownRight->y=ptTemp.y;
			//	}
			//}

			if(ptTemp.x>MaxX||ptTemp.y>MaxY)
			{
				break;
			}
			ImageAddr=((Uint32)ptTemp.y*(Uint32)MaxX)+((Uint32)ptTemp.x);

			*(ptStorePlace+iRow*nColNum+iCol) = ImageAddr;

		}
	}


	ptRowAndCol.x = (Uint16)nColNum;//
	ptRowAndCol.y = iRow_Num;//


	return ptRowAndCol;
}


///////////////////////////////////////////////
//  end
///////////////////////////////////////////////


///////////////////////////////////////////////
bool Existlight(unsigned char* puNewImage,int nWidth,int nHight, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	int row,col,count = 0;
	//float sum = 0;
	//int hsum = 0, lsum = 0;	
	int i,j,l,k,n,n1,n2,t,s,Color=0;
	unsigned char* MidImage;

	int dwArrayCurrentSize, dwArraySize, dwProcessedSize;		
	//int lthreold=1,rthreold=100;//2;1000//50;4000
	int lthreold=100,rthreold=800;//70,800
	CRect rtConnect;
	//CRect fore;

	//int Mean= 0;
	int area=0;
	//float Arearate=0.0;
	//int AreaA,AreaB;
	//int vertical=0,recover=0;
	//int dy=0,dx=0;
	//int center;

	//int position=0;
	//int position1=0;
	int dwForegroundCount=0;
	int candidate_num=0,noncandidate_num=0;
	//int test;
	//ArrayInit((Array *)pParams->ForegroundArray, sizeof(CForeground_Array), FOREGROUND_ARRAY_CAPACITY);////?????

	//
	pCfgs->thresholdValue=pCfgs->CameraCfg.uDayNightJudgeMinContiuFrame;
	pCfgs->thresholdValue=(pCfgs->thresholdValue<150)? 150:pCfgs->thresholdValue;
	/*for(row=0; row<nHight; row++)
	{
	for(col=0; col<nWidth; col++)
	{
	Mean=GetImagePointValue(col,row,nHight,nWidth,puNewImage);//puNewImage

	if (Mean>pCfgs->thresholdValue)
	{
	SetImagePointValue(col,row,nHight,nWidth,255,puNewImage);
	//	area++;
	}
	else
	{
	SetImagePointValue(col,row,nHight,nWidth,0,puNewImage);
	}

	}
	}*/
	//if(area>200)
	//	return FALSE;
	MidImage=pParams->puPointNextImage;
	memset(MidImage, 0, sizeof(unsigned char) * nWidth *nHight);
	memset(pParams->m_pPointArray,0,sizeof(CPoint) * nWidth *nHight);
	pCfgs->guTest2=0;


	for(row=0; row<nHight; row+=3)
	{
		for(col=0; col<nWidth; col+=3)
		{
			if((GetImagePointValue(col,row,nHight,nWidth,puNewImage)>pCfgs->thresholdValue)
				&&(!GetImagePointValue(col,row,nHight,nWidth,MidImage))
				)
			{

				// Got a start point
				pParams->m_pPointArray[0].x =col;
				rtConnect.left = col;
				rtConnect.right = col;

				pParams->m_pPointArray[0].y = row;
				rtConnect.top = row;
				rtConnect.bottom = row;

				dwArrayCurrentSize = 1;
				dwProcessedSize = 0;
				do
				{
					dwArraySize = dwArrayCurrentSize;
					for(l=dwProcessedSize; l<dwArraySize; l++)
					{
						rtConnect.left = (rtConnect.left < pParams->m_pPointArray[l].x)? rtConnect.left:pParams->m_pPointArray[l].x;
						rtConnect.right = (rtConnect.right > pParams->m_pPointArray[l].x)? rtConnect.right:pParams->m_pPointArray[l].x;
						rtConnect.top = (rtConnect.top < pParams->m_pPointArray[l].y)? rtConnect.top:pParams->m_pPointArray[l].y;
						rtConnect.bottom = (rtConnect.bottom > pParams->m_pPointArray[l].y)? rtConnect.bottom:pParams->m_pPointArray[l].y;
						for( t=-1; t<=1; t++)
						{
							for( s=-1; s<=1; s++)
							{
								k = pParams->m_pPointArray[l].y + t;
								j = pParams->m_pPointArray[l].x + s;
								if((k >= 0)
									&&(k < nHight)
									&&(j >= 0)
									&&(j < nWidth)
									)
								{
									if((GetImagePointValue(j,k,nHight,nWidth,puNewImage)>pCfgs->thresholdValue) && 
										!GetImagePointValue(j,k,nHight,nWidth,MidImage))
									{
										SetImagePointValue(j,k,nHight,nWidth,255,MidImage);
										pParams->m_pPointArray[dwArrayCurrentSize].x = j;
										pParams->m_pPointArray[dwArrayCurrentSize].y = k;
										dwArrayCurrentSize++;//I?E??E??????
									}
								}
							}
						}
					}
					dwProcessedSize = dwArraySize;
				}while(dwArrayCurrentSize > dwArraySize);

				area=RECT_Width(rtConnect)*RECT_Height(rtConnect);
				if(area>=lthreold && area<=rthreold)
				{
					candidate_num++;
				}
				else if(area>rthreold)
				{
					noncandidate_num++;
					return FALSE;

				}
				//	pCfgs->guTestD[0]=area;
				//	pCfgs->guTestD[1]=(RECT_Height(rtConnect) + 1);
				//	pCfgs->guTestD[2]=(float)dwArraySize/(float)area*10;
				if(area>lthreold&&area<rthreold
					&&((RECT_Height(rtConnect) )<(int)(nHight/2))
					&&(int)((RECT_Width(rtConnect) )/(RECT_Height(rtConnect)))<3
					&&(int)((RECT_Height(rtConnect) )/(RECT_Width(rtConnect)))<5
					&&((float)((float)dwArraySize/(float)area)>0.3))//
				{
					//fore.left=rtConnect.left;
					//fore.right=rtConnect.right;
					//fore.top=rtConnect.top;
					//fore.bottom=rtConnect.bottom;
					//ArrayAdd((Array *)pParams->ForegroundArray,(void*)&rtConnect);
					dwForegroundCount++;
					pCfgs->guTest2++;     
					// return TRUE;

				}

			}
		}
	}
	//dwForegroundCount = ArrayGetSize((Array *)pParams->ForegroundArray);
	//if(lsum>count/2)
	//	pCfgs->guTestD[3]++;
	if (dwForegroundCount>0&&candidate_num<5&&noncandidate_num<1) //||lsum>count/2 615  lightorgray
		return TRUE;
	else
		return FALSE;
}
bool DayorNight(unsigned char* puNewImage,int nWidth,int nHight, ALGCFGS *pCfgs, ALGPARAMS *pParams,RESULTMSG* outBuf)
{
	int row,col,count = 0;
	float sum = 0;
	int cmax = 0, cmin = 0;
	//float hsum = 0, lsum = 0;
	long long hsum = 0, lsum = 0;
	int hcount = 0, lcount = 0;

	for (row=0;row<nHight;row++)
	{
		for(col=0;col<nWidth;col++)
		{
			count++;
			sum+=GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,puNewImage);
			if (cmax==0)
			{
				cmax=GetImagePointValue(col,row,nHight,nWidth,puNewImage);
			}
			else if (cmax<GetImagePointValue(col,row,nHight,nWidth,puNewImage))
			{
				cmax=GetImagePointValue(col,row,nHight,nWidth,puNewImage);
			}
			if (cmin==0)
			{
				cmin=GetImagePointValue(col,row,nHight,nWidth,puNewImage);
			}
			else if (cmin>GetImagePointValue(col,row,nHight,nWidth,puNewImage))
			{
				cmin=GetImagePointValue(col,row,nHight,nWidth,puNewImage);
			}
		}
	}

	if (count)
	{
		sum=(float)(sum/count);
	}

	for (row=0;row<nHight;row++)
	{
		for (col=0;col<nWidth;col++)
		{
			if (GetImagePointValue(col,row,nHight,nWidth,puNewImage)>(cmax+cmin)/2)
			{
				hsum+=GetImagePointValue(col,row,nHight,nWidth,puNewImage);
				hcount++;
			}
			else
			{
				lsum+=GetImagePointValue(col,row,nHight,nWidth,puNewImage);
				lcount++;
			}
		}
	}

	if (lcount&&hcount)
	{
		hsum /= hcount;
		lsum /= lcount;
	}
	pCfgs->gdWrap=hsum;
	pCfgs->gdSum=lsum;
	//pCfgs->uVehicleQueueLength=sum;
	if ((hsum-lsum)>140&&(sum<60))
	{
		return FALSE;
	}

	return TRUE;
}

void NightLightDetect(unsigned char* puNewImage, unsigned char* puTempImage,unsigned char* puaddImage,ALGCFGS *pCfgs, ALGPARAMS *pParams)
{

	int i,j,l,k,n,n1,n2,t,s,Color=0;
	int col,row;
	unsigned char* MidImage;

	int dwArrayCurrentSize, dwArraySize, dwProcessedSize;		
	int lthreold=pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold*2,rthreold=pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold1;//600
	CRect rtConnect;
	CRect fore;

	int Mean= 0;
	int area;
	float Arearate=0.0,recover=0.0;
	int AreaA,AreaB;
	int vertical=0;
	int dy=0,dx=0;
	int center;

	int position=0;
	int position1=0;
	int dwForegroundCount=0,dwForegroundCount1=0;
	int maxarea=0,maxarea1=0;
	pParams->NormalDetectCfg.cSpeedDesc.fitflag=FALSE;
	////printf("lthreold is:%d,rthreold is:%d\t",lthreold,rthreold);

	pCfgs->thresholdValue =pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold2;
	////printf("000 is ok\t");
	pCfgs->thresholdValue=(pCfgs->thresholdValue<150)? 150:pCfgs->thresholdValue;	
	////printf("thresholdvalue is:%d\t",pCfgs->thresholdValue);
	ArrayInit((Array *)pParams->ForegroundArray, sizeof(CForeground_Array), FOREGROUND_ARRAY_CAPACITY);////?????
	////printf("222 is ok\t");
//	ArrayInit((Array *)pParams->ForegroundArray1, sizeof(CForeground_Array), FOREGROUND_ARRAY_CAPACITY);////?????
	////printf("333 is ok\t");

	for(row=0; row<pCfgs->m_iHeight; row++)
	{
		for(col=0; col<pCfgs->m_iWidth; col++)
		{
			////?????
			Mean =GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,puNewImage);
			/*if (Mean>100)
				SetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,255,puaddImage);
			else
				SetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,0,puaddImage);*/
			if (Mean>pCfgs->thresholdValue)
			{
				Mean=0;
				for(k=0; k<3; k++)
				{
					for(l=0; l<3; l++)
					{
						if((row+l) < pCfgs->m_iHeight && (col+k) < pCfgs->m_iWidth)
						{
							Mean+=GetImagePointValue(col+k,row+l,pCfgs->m_iHeight,pCfgs->m_iWidth,puNewImage);//puNewImage
						}
					}
				}
				Mean/=9;
				if (Mean>pCfgs->thresholdValue)
					SetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,255,puTempImage);
				else
					SetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,0,puTempImage);
			}
			else
			{

				SetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,0,puTempImage);
			}

		}
	}
	//
	//
	////printf("im2bw is ok!\t");
	MidImage=pParams->puPointNextImage;

	memset((void *)MidImage, 0, sizeof(unsigned char) * pCfgs->m_iWidth *pCfgs->m_iHeight);
	memset(pParams->m_pPointArray,0,sizeof(CPoint) * pCfgs->m_iWidth *pCfgs->m_iHeight);
	////printf("start connect_det tempimage\t");
	maxarea=connect_det(puTempImage,MidImage,lthreold,rthreold,pCfgs->m_iWidth ,pCfgs->m_iHeight,pParams,pParams->ForegroundArray,0);
	dwForegroundCount = ArrayGetSize((Array *)pParams->ForegroundArray);

	////printf("finish connect_det tempimage\t");
//	memset(pParams->m_pPointArray,0,sizeof(CPoint) * pCfgs->m_iWidth *pCfgs->m_iHeight);
//	memset((void *)MidImage, 0, sizeof(unsigned char) * pCfgs->m_iWidth *pCfgs->m_iHeight);
	////printf("start connect_det addimage\t");
//	maxarea1=connect_det(puaddImage,MidImage,lthreold,rthreold,pCfgs->m_iWidth,pCfgs->m_iHeight, pParams,pParams->ForegroundArray1,1);
	////printf("finish connect_det addimage\t");

//	dwForegroundCount1 = ArrayGetSize((Array *)pParams->ForegroundArray1);
	////printf("maxarea1 is :%d\t",maxarea1);
	////printf("maxarea is :%d\n",maxarea);
	if (!dwForegroundCount)
	{

		memset(puTempImage, 0, sizeof(unsigned char) * pCfgs->m_iWidth *pCfgs->m_iHeight);
		return;
	}
	if (dwForegroundCount>1)
	{
	//	pCfgs->guTest = dwForegroundCount;
		for (i=0;i<dwForegroundCount-1;i++)
		{
			for (j=i+1;j<dwForegroundCount;j++)
			{
				Arearate=0.0;
				AreaA=0;
				AreaB=0;
				AreaA=(RECT_Width(pParams->ForegroundArray->array[i].m_rtConnect))*(RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect));
				AreaB=(RECT_Width(pParams->ForegroundArray->array[j].m_rtConnect))*(RECT_Height(pParams->ForegroundArray->array[j].m_rtConnect));

				Arearate=(float)((float)min(AreaA,AreaB)/(float)max(AreaB,AreaA));//
				vertical=0;
				recover=0.0;
				vertical=min(pParams->ForegroundArray->array[i].m_rtConnect.bottom,pParams->ForegroundArray->array[j].m_rtConnect.bottom)-max(pParams->ForegroundArray->array[i].m_rtConnect.top,pParams->ForegroundArray->array[j].m_rtConnect.top);
				if(min(RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect), RECT_Height(pParams->ForegroundArray->array[j].m_rtConnect))!=0)
				{
					recover=(float)vertical/(float)min(RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect) ,RECT_Height(pParams->ForegroundArray->array[j].m_rtConnect));//???
				}
				dy=0;
				dx=0;
				dy=max(pParams->ForegroundArray->array[i].m_rtConnect.bottom,pParams->ForegroundArray->array[j].m_rtConnect.bottom)-
					min(pParams->ForegroundArray->array[i].m_rtConnect.top,pParams->ForegroundArray->array[j].m_rtConnect.top);
				dx=max(pParams->ForegroundArray->array[i].m_rtConnect.right,pParams->ForegroundArray->array[j].m_rtConnect.right)-
					min(pParams->ForegroundArray->array[i].m_rtConnect.left,pParams->ForegroundArray->array[j].m_rtConnect.left);//I????
				center=0;
				center=abs(pParams->ForegroundArray->array[i].m_rtConnect.top-pParams->ForegroundArray->array[j].m_rtConnect.top);//???

				/*pCfgs->guTestA[0]=dy;
				pCfgs->guTestA[1]=dx;
				pCfgs->guTestA[2]=center;
				pCfgs->guTestA[3]=min(AreaA,AreaB);
				pCfgs->guTestA[4]=max(AreaB,AreaA);*/


				if (Arearate>=0.3 
					&&recover>=0.2        
					&&dx>=(int)(pCfgs->m_iWidth*0.3)    
					&&dy<=(int)(0.75*pCfgs->m_iHeight)
					&&dy<8*min(RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect),RECT_Height(pParams->ForegroundArray->array[j].m_rtConnect)))
				{
					if (pParams->NormalDetectCfg.cSpeedDesc.fitflag==FALSE)
					{
						pParams->ForegroundArray->array[i].m_rtConnect.bottom=max(pParams->ForegroundArray->array[i].m_rtConnect.bottom,pParams->ForegroundArray->array[j].m_rtConnect.bottom);
						pParams->ForegroundArray->array[i].m_rtConnect.right=max(pParams->ForegroundArray->array[i].m_rtConnect.right,pParams->ForegroundArray->array[j].m_rtConnect.right);
						pParams->ForegroundArray->array[i].m_rtConnect.left=min(pParams->ForegroundArray->array[i].m_rtConnect.left,pParams->ForegroundArray->array[j].m_rtConnect.left);
						pParams->ForegroundArray->array[i].m_rtConnect.top=min(pParams->ForegroundArray->array[i].m_rtConnect.top,pParams->ForegroundArray->array[j].m_rtConnect.top);
						ArrayRemoveAt((Array *)pParams->ForegroundArray,j);

						position=i;
						dwForegroundCount--;
						pParams->NormalDetectCfg.cSpeedDesc.fitflag=TRUE;

						pCfgs->guTest3=6;
					}
					else
					{
						pParams->ForegroundArray->array[i].m_rtConnect.bottom=max(pParams->ForegroundArray->array[i].m_rtConnect.bottom,pParams->ForegroundArray->array[j].m_rtConnect.bottom);
						pParams->ForegroundArray->array[i].m_rtConnect.right=max(pParams->ForegroundArray->array[i].m_rtConnect.right,pParams->ForegroundArray->array[j].m_rtConnect.right);
						pParams->ForegroundArray->array[i].m_rtConnect.left=min(pParams->ForegroundArray->array[i].m_rtConnect.left,pParams->ForegroundArray->array[j].m_rtConnect.left);
						pParams->ForegroundArray->array[i].m_rtConnect.top=min(pParams->ForegroundArray->array[i].m_rtConnect.top,pParams->ForegroundArray->array[j].m_rtConnect.top);
						ArrayRemoveAt((Array *)pParams->ForegroundArray,j);
						position1=i;
						dwForegroundCount--;

						pCfgs->guTest3= 8;

					}
				}
			}
		}
	}

	if (pParams->NormalDetectCfg.cSpeedDesc.fitflag==TRUE)
	{
		pCfgs->guTestC[0]++;
		if (position1!=0                    
			&&position1<ArrayGetSize((Array *)pParams->ForegroundArray)    
			&&position<ArrayGetSize((Array *)pParams->ForegroundArray)     
			)
		{
			if (pParams->ForegroundArray->array[position1].m_rtConnect.top<pParams->ForegroundArray->array[position].m_rtConnect.top)
			{
				position=position1;
			}
		}
		pParams->NormalDetectCfg.cSpeedDesc.fitflag=FALSE;

		/*pCfgs->guTestC[3]= position;
		pCfgs->guTestC[4]= pParams->ForegroundArray->array[position].m_rtConnect.top;
		pCfgs->guTestC[5]= pParams->ForegroundArray->array[position].m_rtConnect.bottom;*/

		for(row=0; row<pCfgs->m_iHeight; row++)
		{
			//for(col=0; col<pCfgs->m_iWidth; col++)
			{
				if (row>pParams->ForegroundArray->array[position].m_rtConnect.top     
					&&row<pParams->ForegroundArray->array[position].m_rtConnect.bottom)  
				{               
					SetImagePointValue(0,row,pCfgs->m_iHeight,pCfgs->m_iWidth,255,puTempImage);
					pCfgs->guTestC[1]++;
				}
				else
				{
					SetImagePointValue(0,row,pCfgs->m_iHeight,pCfgs->m_iWidth,0,puTempImage);
					pCfgs->guTestC[2]++;
				}
			}
		}
	}
	else
	{
		//if (dwForegroundCount>0)
		//{
		//	for (i=0;i<dwForegroundCount;i++)
		//	{
		//		if (RECT_Width(pParams->ForegroundArray->array[i].m_rtConnect)*RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect)>lthreold//40
		//			&&RECT_Width(pParams->ForegroundArray->array[i].m_rtConnect)/(RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect))<3
		//			&&RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect)/(RECT_Width(pParams->ForegroundArray->array[i].m_rtConnect))<3
		//			/*&&RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect)<pCfgs->m_iHeight/3*/)
		//		{
		//			for(row=0; row<pCfgs->m_iHeight; row++)
		//			{
		//				//for(col=0; col<pCfgs->m_iWidth; col++)
		//				{
		//					if (row>pParams->ForegroundArray->array[i].m_rtConnect.top&&row<pParams->ForegroundArray->array[i].m_rtConnect.bottom)
		//					{
		//						SetImagePointValue(0,row,pCfgs->m_iHeight,pCfgs->m_iWidth,255,puTempImage);
		//					}
		//					else
		//						SetImagePointValue(0,row,pCfgs->m_iHeight,pCfgs->m_iWidth,0,puTempImage);
		//				}
		//			}
		//			position=10;
		//			break;
		//		}
		//	}
		//	if (position!=10)
		//	{
		//		memset(puTempImage, 0, sizeof(unsigned char) * pCfgs->m_iWidth *pCfgs->m_iHeight);
		//	}
		//}
		//else
		memset((void *)puTempImage, 0, sizeof(unsigned char) * pCfgs->m_iWidth *pCfgs->m_iHeight);
	}

}
int connect_det(unsigned char *bwImage,unsigned char *MidImage,int lthreold,int rthreold,int width,int height, ALGPARAMS *pParams,CForeground_Array *ForegroundArray,int flag)
{
	int col,row;
	int i,j,l,k,n,n1,n2,t,s,Color=0;
	CRect rtConnect;
	//CRect fore;
	int dwArrayCurrentSize, dwArraySize, dwProcessedSize;
	int area,rtWidth=0,rtHeight=0;
	int maxconArea=0;
	for(row=0; row<height; row+=2)
	{
		for(col=0; col<width; col+=2)
		{
			if(GetImagePointValue(col,row,height,width,bwImage) 
				&&(!GetImagePointValue(col,row,height,width,MidImage))
				)
			{

				// Got a start CPoint
				pParams->m_pPointArray[0].x =col;
				rtConnect.left = col;
				rtConnect.right = col;

				pParams->m_pPointArray[0].y = row;
				rtConnect.top = row;
				rtConnect.bottom = row;

				dwArrayCurrentSize = 1;
				dwProcessedSize = 0;
				do
				{
					dwArraySize = dwArrayCurrentSize;
					for(l=dwProcessedSize; l<dwArraySize; l++)
					{
						rtConnect.left = (rtConnect.left < pParams->m_pPointArray[l].x)? rtConnect.left:pParams->m_pPointArray[l].x;
						rtConnect.right = (rtConnect.right > pParams->m_pPointArray[l].x)? rtConnect.right:pParams->m_pPointArray[l].x;
						rtConnect.top = (rtConnect.top < pParams->m_pPointArray[l].y)? rtConnect.top:pParams->m_pPointArray[l].y;
						rtConnect.bottom = (rtConnect.bottom > pParams->m_pPointArray[l].y)? rtConnect.bottom:pParams->m_pPointArray[l].y;
						for( t=-1; t<=1; t++)
						{
							for( s=-1; s<=1; s++)
							{
								k = pParams->m_pPointArray[l].y + t;
								j = pParams->m_pPointArray[l].x + s;
								if((k >= 0)
									&&(k <height)
									&&(j >= 0)
									&&(j <width)
									)
								{
									if(GetImagePointValue(j,k,height,width,bwImage) && 
										!GetImagePointValue(j,k,height,width,MidImage))
									{
										SetImagePointValue(j,k,height,width,1,MidImage);
										pParams->m_pPointArray[dwArrayCurrentSize].x = j;
										pParams->m_pPointArray[dwArrayCurrentSize].y = k;
										dwArrayCurrentSize++;//?
									}
								}
							}
						}
					}
					dwProcessedSize = dwArraySize;
				}while(dwArrayCurrentSize > dwArraySize);



				rtWidth=RECT_Width(rtConnect);
				rtHeight=RECT_Height(rtConnect);
				area=rtHeight*rtWidth;

				if(flag==0)
				{
					if((rtWidth * rtHeight)>lthreold&&(rtWidth * rtHeight)<rthreold
						&&(rtHeight<(int)(height*0.7))
						&&(int)(rtHeight/rtWidth)<=5//3
						&&(int)(rtWidth/rtHeight)<3
						&&((float)((float)dwArraySize/(float)area)>0.3))//
					{
						ArrayAdd((Array *)ForegroundArray,(void*)&rtConnect);
						maxconArea=area>maxconArea?area:maxconArea;
						//pCfgs->guTest2++;	
					}
				}
				else if(flag==1)
				{
					if((rtWidth * rtHeight)>lthreold)//
					{
						ArrayAdd((Array *)ForegroundArray,(void*)&rtConnect);
						maxconArea=area>maxconArea?area:maxconArea;

						//pCfgs->guTest2++;	
					}
				}
				else
				{
					if((rtWidth * rtHeight)>lthreold
						&&(int)(rtHeight/rtWidth)<=5//3
						&&(int)(rtWidth/rtHeight)<3
						&&((float)((float)dwArraySize/(float)area)>=0.3))//
					{
						ArrayAdd((Array *)ForegroundArray,(void*)&rtConnect);
						maxconArea=area>maxconArea?area:maxconArea;

						//pCfgs->guTest2++;	
					}
				}

			}
		}
	}
	return maxconArea;
}

void SobelCalculate(unsigned char *puPointNewImage,unsigned char *puPointSobelImage,int threshold,int width,int height)
{
	unsigned char *in;   
	unsigned char *out;  
	//int cols;
	//int rows;            
	//int threshold;

	int H, O, V, i,j;
	int i00, i01, i02;
	int i10,      i12;
	int i20, i21, i22;
	//int w;

	in = puPointNewImage;
	out = puPointSobelImage;
	//cols = width;
	//rows = height;
	//threshold = 100;
	//w = cols;
	memset(puPointSobelImage,0,sizeof(unsigned char)*width*height);
	for (i = 1; i < height - 1; i++)
	{
		in=puPointNewImage+i*width;
		out=puPointSobelImage+i*width;
		for(j=1;j<width-1;j++)
		{
			i00=*(in+j-width-1);
			i01=*(in+j-width);
			i02=*(in+j-width+1);
			i10=*(in+j-1);
			i12=*(in+j+1);
			i20=*(in+j+width-1);
			i21=*(in+j+width);
			i22=*(in+j+width+1);

			H = -   i00 - 2*i01 -   i02 +
				+   i20 + 2*i21 +   i22;

			V = -   i00         +   i02
				- 2*i10         + 2*i12
				-   i20         +   i22;

			O = abs(H) + abs(V);
			O = (O < threshold)? 0:255;
			*(out+j) = O;

		}

	}
	/*for (i = 0; i < cols*(rows-2) - 2; i++)
	{
	i00=in[i    ];
	i01=in[i    +1];
	i02=in[i    +2];
	i10=in[i+  w];
	i12=in[i+  w+2];
	i20=in[i+2*w];
	i21=in[i+2*w+1];
	i22=in[i+2*w+2];

	H = -   i00 - 2*i01 -   i02 +
	+   i20 + 2*i21 +   i22;

	V = -   i00         +   i02
	- 2*i10         + 2*i12
	-   i20         +   i22;

	O = abs(H) + abs(V);

	if( O < threshold )
	{
	O = 0;
	}
	else
	{
	O = 255;
	}
	out[i + 1] = O;
	}*/
}
void ArrayInit(Array *nArray, int sizeOfStruct, int capacity)
{
	nArray->sizeOfStruct = sizeOfStruct;
	nArray->sizeOfArray = 0;
	nArray->capacity = capacity;
}

int ArrayGetCapaticy(Array *nArray)
{
	return nArray->capacity;
}

int ArrayGetSize(Array *nArray)
{
	return nArray->sizeOfArray;
}

int ArrayAdd(Array *nArray, void* elem)
{
	int size;
	CForeground_Array *pCForegroundArray = NULL;

	if( nArray->sizeOfArray > (nArray->capacity - 1))
	{
		nArray->sizeOfArray  = nArray->capacity - 1;
		return nArray->sizeOfArray;
	}

	size = nArray->sizeOfStruct;
	switch(size)
	{
	case sizeof(CForeground_Array):
		pCForegroundArray = (CForeground_Array *)nArray;
		memcpy((void*)&pCForegroundArray->array[nArray->sizeOfArray], (void*)elem, sizeof(CRect));
		nArray->sizeOfArray++;
		break;
	default:
		break;
	}

	return nArray->sizeOfArray;

}

void ArrayRemoveAt( Array *nArray, int nIndex )
{
	int i;
	int size;
	CForeground_Array *pCForegroundArray = NULL;
	CRect* SourceAddress;
	CRect* TargetAddress;

	if( nIndex <= (nArray->sizeOfArray - 1) )
	{
		size = nArray->sizeOfStruct;
		switch(size)
		{
		case sizeof(CForeground_Array):
			pCForegroundArray = (CForeground_Array *)nArray;
			for( i = nIndex; i <= (nArray->sizeOfArray - 2); i++ )
			{
				SourceAddress = (CRect *)&pCForegroundArray->array[i+1];
				TargetAddress = (CRect *)&pCForegroundArray->array[i];
				memcpy((void*)TargetAddress, (void*)SourceAddress, sizeof(CRect));
			}
			nArray->sizeOfArray--;
			break;
		default:
			break;
		}

	}
}


void ArraySetSize(Array *nArray, int size)
{
	nArray->sizeOfArray = size;
}

void ArrayRemoveAll(Array *nArray)
{
	nArray->sizeOfArray = 0;
}
int StandardDeviation (unsigned char* puNewImage,int width,int height,int start_line,int stop_line,int flag)
{

	int x=0,y=0;
	// ?
	int average = 0;
	int variance = 0;
	int k=1;
	int l=1;
	int num=0;
	if(flag==0)
	{
		k=2;
		l=1;
	}
	else if(flag==2)
	{  
		k=3;
		l=2;
	}
	else
	{  
		k=2;
		l=2;
	}
	for (y=start_line;y <stop_line;y+=k)
	{
		for (x=0;x <width;x+=l)
		{
			average += *(puNewImage+y*width+x);
			num++;
		}
	} 
	average=(num>0)? (average/num):0;

	// 

	for (y=start_line;y <stop_line;y+=k)
	{
		for (x=0;x <width;x+=l)
		{
			int temp = *(puNewImage+y*width+x);
			variance += (temp - average)*(temp - average);
		}
	}
	variance=(num>0)? (variance/num):0;
	return (int)(sqrt((float)variance)+0.5);

}
//20150918,txl
float TeamLength1(unsigned char* puNewImage,unsigned char* puaddImage,ALGCFGS *pCfgs,ALGPARAMS *pParams,int tailtop,int tailbottom)
{
	/*int position[20];
	int position1=0;*/
	int Mean= 0;
	int area;
	float Arearate=0.0,recover=0.0;
	int AreaA,AreaB;
	int vertical=0;
	int dy=0,dx=0;
	int center;

	int position=0;
	int position1=0;
	int dwForegroundCount=0,dwForegroundCount1=0;
	pParams->NormalDetectCfg.cSpeedDesc.fitflag=FALSE;	


	float percent1=0;
	pCfgs->team_height=pParams->NormalDetectCfg.cSpeedDesc.QueDetectLines;
	pCfgs->team_width=pParams->NormalDetectCfg.cSpeedDesc.QueDetectColumns;
	unsigned char* MidImage;
	MidImage=pParams->BackQueueImage;
	bool stopCal=FALSE;

	//	memset((void *)puaddImage,0,sizeof(unsigned char) * pCfgs->team_width *pCfgs->team_height);

	int i,j,row,col,k,t,s,l,p,count=0,rtwidth=0,rtheight=0;
	int dwArrayCurrentSize, dwArraySize, dwProcessedSize;
	int minsize=pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold,maxsize=pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold1;//600
	CRect rtConnect;
	//int thresholdValue=100;
	bool Islight=FALSE;
	int maxarea;
	//unsigned char* coImage;
	//coImage=pParams->SobelQueueImage;
	//memset(coImage,0,sizeof(unsigned char) * pCfgs->team_width *pCfgs->team_height);
	int thresholdValue =pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold2;
	int thresholdMax=pCfgs->CameraCfg.uVehicleHeadMinRow;
	int thresholdSub=pCfgs->CameraCfg.uInThreshold;
	int sum=0,max_value=0;
	float confidence=0.0,roundness=0.0;
	thresholdValue=(thresholdValue<150)? 150:thresholdValue;
	thresholdMax=(thresholdMax<thresholdValue)? thresholdValue:thresholdMax;
	thresholdSub=(thresholdSub<15)?15:thresholdSub;

	ArrayInit((Array *)pParams->ForegroundArray, sizeof(CForeground_Array), FOREGROUND_ARRAY_CAPACITY);////?????
	//ArrayInit((Array *)pParams->ForegroundArray1, sizeof(CForeground_Array), FOREGROUND_ARRAY_CAPACITY);////?????
	memset(pParams->m_pPointArray,0,sizeof(CPoint) * pCfgs->team_width *pCfgs->team_height);
	//MidImage=new unsigned char[pCfgs->team_width*pCfgs->team_height];
	memset((void *)MidImage,0,sizeof(unsigned char) * pCfgs->team_width *pCfgs->team_height);
	//im2bw(puNewImage,puaddImage,thresholdValue-30,pCfgs->team_width,pCfgs->team_height);
	im2bw(puNewImage,MidImage,thresholdValue,pCfgs->team_width,pCfgs->team_height);
	memset((void *)puNewImage,0,sizeof(unsigned char) * pCfgs->team_width *pCfgs->team_height);
	memcpy((void *)puNewImage,(void *)MidImage, pCfgs->team_width *pCfgs->team_height);
	memset((void *)MidImage,0,sizeof(unsigned char) * pCfgs->team_width *pCfgs->team_height);



	maxarea=connect_det(puNewImage,MidImage,minsize,maxsize,pCfgs->team_width,pCfgs->team_height,pParams,pParams->ForegroundArray,2);
	dwForegroundCount=ArrayGetSize((Array *)pParams->ForegroundArray);
	////printf("maxarea is:%d\t",maxarea);
	//memset((void *)MidImage,0,sizeof(unsigned char) * pCfgs->team_width *pCfgs->team_height);
	//memset(pParams->m_pPointArray,0,sizeof(CPoint) * pCfgs->team_width *pCfgs->team_height);


	if (dwForegroundCount>1)
	{
		//pCfgs->guTest = dwForegroundCount;
		for (i=0;i<dwForegroundCount-1;i++)
		{
			for (j=i+1;j<dwForegroundCount;j++)
			{
				Arearate=0.0;
				AreaA=0;
				AreaB=0;
				AreaA=RECT_Width(pParams->ForegroundArray->array[i].m_rtConnect)*RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect);
				AreaB=RECT_Width(pParams->ForegroundArray->array[j].m_rtConnect)*RECT_Height(pParams->ForegroundArray->array[j].m_rtConnect);

				Arearate=(float)((float)min(AreaA,AreaB)/(float)max(AreaB,AreaA));//
				vertical=0;
				recover=0;
				vertical=min(pParams->ForegroundArray->array[i].m_rtConnect.bottom,pParams->ForegroundArray->array[j].m_rtConnect.bottom)-
					max(pParams->ForegroundArray->array[i].m_rtConnect.top,pParams->ForegroundArray->array[j].m_rtConnect.top);
				if(min(RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect), RECT_Height(pParams->ForegroundArray->array[j].m_rtConnect))!=0)
				{
					recover=(float)vertical/(float)min(RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect) ,RECT_Height(pParams->ForegroundArray->array[j].m_rtConnect));//???
				}
				dy=0;
				dx=0;
				dy=max(pParams->ForegroundArray->array[i].m_rtConnect.bottom,pParams->ForegroundArray->array[j].m_rtConnect.bottom)-
					min(pParams->ForegroundArray->array[i].m_rtConnect.top,pParams->ForegroundArray->array[j].m_rtConnect.top);
				dx=max(pParams->ForegroundArray->array[i].m_rtConnect.right,pParams->ForegroundArray->array[j].m_rtConnect.right)-
					min(pParams->ForegroundArray->array[i].m_rtConnect.left,pParams->ForegroundArray->array[j].m_rtConnect.left);//I????
				center=0;
				center=abs(pParams->ForegroundArray->array[i].m_rtConnect.top-pParams->ForegroundArray->array[j].m_rtConnect.top);//???

				/*pCfgs->guTestA[0]=dy;
				pCfgs->guTestA[1]=dx;
				pCfgs->guTestA[2]=center;
				pCfgs->guTestA[3]=min(AreaA,AreaB);

				pCfgs->guTestA[4]=max(AreaB,AreaA);*/


				if (Arearate>=0.2
					&&recover>=0.2             
					&&dx>=(int)(0.3*pCfgs->team_width))
					//&&dy<8*min(RECT_Height(pParams->ForegroundArray->array[i].m_rtConnect),RECT_Height(pParams->ForegroundArray->array[j].m_rtConnect)))
				{
					//if (pParams->NormalDetectCfg.cSpeedDesc.fitflag==FALSE)
					{
						pParams->NormalDetectCfg.cSpeedDesc.position[count]=(pParams->ForegroundArray->array[i].m_rtConnect.bottom+pParams->ForegroundArray->array[i].m_rtConnect.top)/2;
						count++;
						if(min(tailbottom,pParams->ForegroundArray->array[i].m_rtConnect.bottom)-max(tailtop,pParams->ForegroundArray->array[i].m_rtConnect.top)>0)
						{
							pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail=TRUE;
							stopCal=TRUE;
							break;
						}
						else
						{
							pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail=FALSE;
							//continue;
						}


						ArrayRemoveAt((Array *)pParams->ForegroundArray,j);
						ArrayRemoveAt((Array *)pParams->ForegroundArray,i);
						//position=i;
						dwForegroundCount=dwForegroundCount-2;
						//pParams->NormalDetectCfg.cSpeedDesc.fitflag=TRUE;

						//pCfgs->guTest3=6;
					}

				}

			}
			if(stopCal==TRUE)
			{
				break;
			}
		}
	}	



	if (count>0)
	{
		percent1=1-(float)pParams->NormalDetectCfg.cSpeedDesc.position[0]/pCfgs->team_height;
		percent1=percent1>1?1:percent1;//20150918,txl
		pParams->NormalDetectCfg.cSpeedDesc.percent=percent1;
		pParams->NormalDetectCfg.cSpeedDesc.LastCarposition=pParams->NormalDetectCfg.cSpeedDesc.position[0];
	}
	else
	{
		pParams->NormalDetectCfg.cSpeedDesc.percent=0;	
		pParams->NormalDetectCfg.cSpeedDesc.LastCarposition=pCfgs->team_height;
		pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail=FALSE;//20150918,txl
	}

	//if (pParams->NormalDetectCfg.cSpeedDesc.percent>1)
	//	pParams->NormalDetectCfg.cSpeedDesc.percent=1;
	return pParams->NormalDetectCfg.cSpeedDesc.percent;
}
//
//txl,201506,????sobel???????,??????
void QueLengthSobel(Uint16 LaneID,unsigned char* OriginImage,unsigned char* SobelImage,ALGCFGS *pCfgs,ALGPARAMS *pParams)
{
	//?????????,???????
	int row=0,col=0;
	int sobelnum=0;
	int objTh=pCfgs->team_width/5;
	float aa[1000]={0};
	char subBack_sobel[1000]={0};
	int Lastpos=0,carlinenum=0;
	char car_par_num=0;
	int car_part_last[20]={0};
	int car_part_linenum[20]={0};
	//????
	for(row=0; row<pCfgs->team_height; row++)
	{
		sobelnum=0;
		for(col=2; col<pCfgs->team_width-2; col++)
		{
			if(GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,SobelImage))
			{
				sobelnum++;
			}

		}
		aa[row]=(float)sobelnum/objTh;//?????????????aa?

	}
	//????????5*1??
	for (row=0;row<pCfgs->team_height;row++)
	{
		if (row<3||row>pCfgs->team_height-3)
		{
			subBack_sobel[row]=(char)(aa[row]*10);//????10??????
		} 
		else
		{
			subBack_sobel[row]=(char)((aa[row-2]+aa[row-1]+aa[row]+aa[row+1]+aa[row+2])*10*0.2);//???5?????
		}
	}


	for (row=0;row<pCfgs->team_height;row++)
	{
		if (subBack_sobel[row]>=3)
		{
			if (carlinenum==0)
			{
				Lastpos=row;
				carlinenum++;
			} 
			else
			{
				carlinenum++;
			}			

		} 
		else
		{
			if (carlinenum>=(pCfgs->team_height*0.1))
			{
				car_part_last[car_par_num]=Lastpos;
				car_part_linenum[car_par_num]=carlinenum;
				car_par_num++;

			}
			carlinenum=0;

		}

	}
	if (subBack_sobel[pCfgs->team_height-1]>=5&&carlinenum>10)
	{
		car_part_last[car_par_num]=Lastpos;
		car_part_linenum[car_par_num]=carlinenum;
	}
	pParams->NormalDetectCfg.cSpeedDesc.Lastpos=car_part_last[0];
	pParams->NormalDetectCfg.cSpeedDesc.linenum=car_part_linenum[0];
}

//float QueueLengthDetect(unsigned char* OriginImage,unsigned char* puSubImage,unsigned char* puSobelImage, unsigned char* puSobelConnectImage,ALGCFGS *pCfgs, ALGPARAMS *pParams)
//{
//	bool IsFullQueue=FALSE;
//	bool IsPartQueue[8]={FALSE};
//	int headline=0;
//	int num=0;
//	int quenum=0;
//	int topy=0;
//
//	int diff;
//	int lasttruepart=0;
//	int aveheight=pCfgs->team_height/8;//???
//	float EXPlength=0.0;
//	IsFullQueue = IsVehicleQueue(puSubImage, puSobelImage,0,pCfgs->team_height, pCfgs, pParams,num);//
//
//	//pCfgs->guTestD[3]=IsFullQueue;	
//	pCfgs->guTestD[1]=pCfgs->team_height;
//	pCfgs->guTestD[2]=pCfgs->team_width;
//
//	headline=pCfgs->team_height;
//	if(((headline-aveheight)>=0)&&(headline>0))
//	{
//		for(headline=pCfgs->team_height;(headline-aveheight)>=0;headline=(headline-aveheight))
//		{
//
//			pCfgs->guTestD[0]++;
//			pCfgs->guTestD[4]=num;
//
//			IsPartQueue[num] = IsVehicleQueue(puSubImage, puSobelImage,headline-aveheight,headline, pCfgs, pParams,num);//
//
//			if (IsPartQueue[num] ==TRUE)
//			{
//				quenum++;
//				lasttruepart=num;
//
//				if (num>0 &&IsPartQueue[num-1] ==TRUE&&((num*aveheight-topy)<(aveheight+1)) )//A?
//					//if (num>0 &&IsPartQueue[num-1] ==TRUE&&((num*aveheight-topy)<(2*aveheight+1)) )//A?
//				{
//					topy=aveheight*(num+1);
//				}
//
//			}
//			num++;
//
//
//		}
//
//	}
//
//	if ((quenum>=3)||(IsFullQueue ==TRUE))
//	{
//		topy=aveheight*(lasttruepart+1);
//	}
//
//	if (topy<=5)
//	{
//		topy=0;
//	}
//	if (quenum>1 &&(IsPartQueue[1]==TRUE||IsPartQueue[0]==TRUE) )//A?
//		//if ((IsFullQueue ==TRUE)||((quenum>3)&&(IsPartQueue[3]==TRUE||IsPartQueue[4]==TRUE)) )//A?
//	{
//		pParams->NormalDetectCfg.cSpeedDesc.percent=(float)topy/pCfgs->team_height;
//
//
//
//	} 
//	else
//	{
//		pParams->NormalDetectCfg.cSpeedDesc.percent=0.0;
//
//	}
//	//A?
//	EXPlength=0.4*pParams->NormalDetectCfg.cSpeedDesc.pastlength[3]+0.3*pParams->NormalDetectCfg.cSpeedDesc.pastlength[2]+0.2*pParams->NormalDetectCfg.cSpeedDesc.pastlength[1]+0.1*pParams->NormalDetectCfg.cSpeedDesc.pastlength[0];
//	pParams->NormalDetectCfg.cSpeedDesc.pastlength[0]=pParams->NormalDetectCfg.cSpeedDesc.pastlength[1];
//	pParams->NormalDetectCfg.cSpeedDesc.pastlength[1]=pParams->NormalDetectCfg.cSpeedDesc.pastlength[2];
//	pParams->NormalDetectCfg.cSpeedDesc.pastlength[2]=pParams->NormalDetectCfg.cSpeedDesc.pastlength[3];
//	pParams->NormalDetectCfg.cSpeedDesc.pastlength[3]=pParams->NormalDetectCfg.cSpeedDesc.percent;
//	if ((EXPlength-pParams->NormalDetectCfg.cSpeedDesc.percent)>0.5)
//	{
//		pParams->NormalDetectCfg.cSpeedDesc.percent=EXPlength;
//	}
//
//	pCfgs->guTestC[0]= EXPlength*100;
//	pCfgs->guTestC[2]= quenum;
//	pCfgs->guTestC[3]= topy;
//	pCfgs->guTestC[4]= IsFullQueue;
//	pCfgs->guTestC[5]= 100*pParams->NormalDetectCfg.cSpeedDesc.percent;
//
//
//	return pParams->NormalDetectCfg.cSpeedDesc.percent;
//
//}
//bool IsVehicleQueue(Uint8* puSubImage, Uint8* puSobelImage,Uint16 tail_line,Uint16 head_line, ALGCFGS *pCfgs, ALGPARAMS *pParams,Uint16 num)
//{
//	int nRow,nCol;
//	float subnum=0.0,sobelnum=0.0,overlapnum=0;
//	int nUnChangedRow = 0;
//	int nullRowNum = 0, maxNullRow = 0;
//	unsigned int uThreshold=5 ;
//	float k = 10.0;
//	int edgenum[756]={0},edgenum_row=0,vehicle_line=0;
//	float edge_density=0;
//	int motionRow = 0,subvalue=0,sobelvalue=0;
//	for( nRow =tail_line ; nRow < head_line; nRow++ )
//	{
//		edgenum_row=0;
//		for (nCol =0 ; nCol < pCfgs->team_width; nCol++)
//		{
//			if (GetImagePointValue( nCol, nRow, pCfgs->team_height, pCfgs->team_width, puSubImage))
//			{
//				subnum++;

//			}
//			if (GetImagePointValue( nCol, nRow, pCfgs->team_height, pCfgs->team_width, puSobelImage))
//			{
//				if (GetImagePointValue( nCol, nRow, pCfgs->team_height, pCfgs->team_width,pParams->BackQueueImage))
//				{
//					overlapnum++;
//				}
//				sobelnum++;
//				edgenum_row=edgenum_row+GetImagePointValue( nCol, nRow, pCfgs->team_height, pCfgs->team_width,puSobelImage);
//			}
//		}
//		edgenum[nRow]=edgenum_row;
//		if (edgenum_row>=1000)
//		{
//			vehicle_line++;
//		}		
//	}
//	edge_density=(float)vehicle_line/(head_line-tail_line);
//	if (sobelnum!=0)
//	{
//		overlapnum=overlapnum/sobelnum;
//	}
//
//	if(head_line!=tail_line)
//	{
//		subnum=subnum/(pCfgs->team_width*(head_line-tail_line));
//		sobelnum=sobelnum/(pCfgs->team_width*(head_line-tail_line));
//
//	}      
//
//	if ((overlapnum>=0.7)&&((subnum<0.1&&sobelnum>=0.3)||(subnum<=0.01&&sobelnum>=0.2)))// A?
//		//if (((subnum<0.15&&sobelnum>=0.3)||(subnum<=0.1&&sobelnum>=0.15)||((sobelnum-subnum)>=0.2))&&(edge_density>0.5))//A?
//	{
//		return TRUE;
//	} 
//	else
//	{
//		return FALSE;
//	}
//
//}


//??
void SobelConectivity(unsigned char* puSobelImage,unsigned char* puSobelConnectImage,ALGCFGS *pCfgs, ALGPARAMS *pParams,int tailtop,int tailbottom)
{
	int dwArrayCurrentSize, dwArraySize, dwProcessedSize,k,j,dwForegroundCount;
	CRect rtConnect;
	//CRect fore;
	unsigned char* out;
	int connectnum=1;
	int row,col,l,t,s;
	//int area,totalcol;
	//int frow,fcol;
	int widthTH=0.5*pCfgs->team_width,heightTH=0.1*pCfgs->team_height;
	bool getCar=FALSE,stopCal=FALSE;//20150918
	float posthresh=0.0;
	//ArrayInit((Array *)pParams->ForegroundArray, sizeof(CForeground_Array), FOREGROUND_ARRAY_CAPACITY);////	
	if (puSobelConnectImage==NULL)
	{
		return;
	}
	out=puSobelConnectImage; //new unsigned char[pCfgs->team_width*pCfgs->team_height];
	memset(out,0,sizeof(unsigned char) * pCfgs->team_width *pCfgs->team_height);
	memset(pParams->m_pPointArray,0,sizeof(CPoint)*pCfgs->team_width *pCfgs->team_height);
	for( row=5; row<pCfgs->team_height-5; row+=2)
	{
		for(col=1; col<pCfgs->team_width-1; col+=2)
		{
			// if(0==GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,puSobelImage))
			//{
			////   col+=2;
			//}
			if(GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,puSobelImage) 
				&&(!GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,out))
				)
			{

				// Got a start CPoint
				pParams->m_pPointArray[0].x =col;
				rtConnect.left = col;
				rtConnect.right = col;

				pParams->m_pPointArray[0].y = row;
				rtConnect.top = row;
				rtConnect.bottom = row;

				dwArrayCurrentSize = 1;
				dwProcessedSize = 0;

				do
				{
					dwArraySize = dwArrayCurrentSize;
					for(l=dwProcessedSize; l<dwArraySize; l++)
					{
						rtConnect.left = (rtConnect.left < pParams->m_pPointArray[l].x)? rtConnect.left:pParams->m_pPointArray[l].x;
						rtConnect.right = (rtConnect.right > pParams->m_pPointArray[l].x)? rtConnect.right:pParams->m_pPointArray[l].x;
						rtConnect.top = (rtConnect.top < pParams->m_pPointArray[l].y)? rtConnect.top:pParams->m_pPointArray[l].y;
						rtConnect.bottom = (rtConnect.bottom > pParams->m_pPointArray[l].y)? rtConnect.bottom:pParams->m_pPointArray[l].y;

						for( t=-1; t<=1; t++)
						{
							for( s=-1; s<=1; s++)
							{
								k = pParams->m_pPointArray[l].y + t;
								j = pParams->m_pPointArray[l].x + s;
								if((k >= 5)
									&&(k < pCfgs->team_height-5)
									&&(j >= 1)
									&&(j < pCfgs->team_width-1)
									)
								{
									if(GetImagePointValue(j,k,pCfgs->team_height,pCfgs->team_width,puSobelImage) && 
										!GetImagePointValue(j,k,pCfgs->team_height,pCfgs->team_width,out))
									{
										SetImagePointValue(j,k,pCfgs->team_height,pCfgs->team_width,connectnum,out);
										pParams->m_pPointArray[dwArrayCurrentSize].x = j;
										pParams->m_pPointArray[dwArrayCurrentSize].y = k;
										dwArrayCurrentSize++;//?
									}
								}
							}
						}
					}
					dwProcessedSize = dwArraySize;
				}while(dwArrayCurrentSize > dwArraySize);	
				//	area=RECT_Width(rtConnect)*RECT_Height(rtConnect)+1;
				//20150921
				posthresh=(float)rtConnect.bottom/pCfgs->team_height;
				if(posthresh>0.2)
				{						
					posthresh=0.1;
				}
				else
				{						
					posthresh=posthresh<0.1?0.05:0.07;
				}
				heightTH=posthresh*pCfgs->team_height;
				if((RECT_Width(rtConnect)>widthTH)&&(RECT_Height(rtConnect)>heightTH))//??????
				{		
					if(Sobelfilt(out,connectnum,rtConnect,pCfgs,pParams,getCar))
					{
						//20150918
						if(getCar==FALSE)
						{						
                            rtConnect.top=min(rtConnect.top,pParams->NormalDetectCfg.cSpeedDesc.Headposition);
							getCar=TRUE;
							if(pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag==FALSE)
							{						
								rtConnect.top=pParams->NormalDetectCfg.cSpeedDesc.LastCarposition;
							}
							else
							{						
								pParams->NormalDetectCfg.cSpeedDesc.LastCarposition=rtConnect.top;
							}

							pParams->NormalDetectCfg.cSpeedDesc.linenum=rtConnect.bottom-rtConnect.top;
						}
						if(min(tailbottom,rtConnect.bottom)-max(tailtop,rtConnect.top)>0)
						{						
							pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail=TRUE;
							stopCal=TRUE;
							break;
						}
						else
						{						
							pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail=FALSE;
							continue;
						}
					}			
					/*fore.left=rtConnect.left;
					fore.right=rtConnect.right;
					fore.top=rtConnect.top;
					fore.bottom=rtConnect.bottom;
					ArrayAdd((Array *)pParams->ForegroundArray,(void*)&fore);	*/			

				}
				connectnum =(connectnum>=255)? 1 : (connectnum+1);

			}
		}
		if(stopCal==TRUE)//20150918
		{
			break;
		}
	}

	//	dwForegroundCount = ArrayGetSize((Array *)pParams->ForegroundArray);
	if (getCar==FALSE)
	{
		pParams->NormalDetectCfg.cSpeedDesc.LastCarposition=pCfgs->team_height;
		pParams->NormalDetectCfg.cSpeedDesc.linenum=0;
		pParams->NormalDetectCfg.cSpeedDesc.IsCarInTail=FALSE;//20150918

	}
	pParams->NormalDetectCfg.cSpeedDesc.percent=1-(float)pParams->NormalDetectCfg.cSpeedDesc.LastCarposition/pCfgs->team_height;

}
bool Sobelfilt(unsigned char* SobelImage,int connectnum,CRect rtConnect,ALGCFGS *pCfgs,ALGPARAMS *pParams,bool getcar)
{
	int row=0,col=0;
	int sobelnum=0;
	int car_line=0;
	int objTh=0.3*pCfgs->team_width;
	//	float aa[1000]={0};
	//	char subBack_sobel[1000]={0};
	int left_pos=255,right_pos=0;
	//	bool get_left_pos=FALSE,get_right_pos=FALSE;
	float car_predict=0;
	bool getcarlastpos=FALSE;
	for(row=rtConnect.top; row<=rtConnect.bottom; row++)
	{
		sobelnum=0;
		left_pos=pCfgs->team_width;
		right_pos=0;
		for(col=rtConnect.left; col<=rtConnect.right; col++)
		{
			if(GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,SobelImage)==connectnum)
			{
				left_pos = (left_pos < col)? left_pos : col;
				right_pos = (right_pos > col)? right_pos : col;

			}

		}
		sobelnum=right_pos-left_pos+1;
		if (sobelnum>=objTh)
		{
			car_line++;
			if ((!pParams->NormalDetectCfg.cSpeedDesc.getQueback_flag)&&(!getcar)&&(!getcarlastpos))
			{
				pParams->NormalDetectCfg.cSpeedDesc.LastCarposition=row;
				getcarlastpos=TRUE;

			}
		}
		//	aa[row]=sobelnum;
	}
	car_predict=(float)car_line/(rtConnect.bottom-rtConnect.top+1);
	//pCfgs->guTestD[2]=car_predict*100;
	if ((car_predict>=0.5)||(car_line>=0.1*pCfgs->team_height))
	{
		return TRUE;
	}
	else
	{
		return FALSE;
	}


}
//1203
void matrix_mult( float *result, float *left, float *right, int m, int n, int z )
{
	int i, j, k;
	float sum;
	for( i = 0; i < m; i++ )
		for( j =0; j < z; j++ )
		{
			sum = 0.0;
			for( k = 0; k < n; k++ )
				sum = sum + ( left[ i*n+k ] * right[ k*z+j ] );
			result[i*z+j] = sum;
		}
}
//?
void matrix_transport( float *result, float *mat, int m, int n )
{
	int i, j;
	for( i = 0; i < n; i++ )
		for( j =0; j < m; j++ )
			result[i*m+j] = mat[j*n+i];
}
//?3*3
void matrix_inverse(float *R, float *Ri)
{
	float den=(R[0]*R[4]*R[8] + R[1]*R[5]*R[6] + R[2]*R[3]*R[7])-
		(R[2]*R[4]*R[6] + R[1]*R[3]*R[8] + R[0]*R[5]*R[7]);
	int   i;
	if((den<-0.00000000001)||(den>0.00000000001))
	{
		Ri[0]=(R[4]*R[8]-R[7]*R[5])/den;
		Ri[1]=(R[2]*R[7]-R[1]*R[8])/den;
		Ri[2]=(R[1]*R[5]-R[2]*R[4])/den;
		Ri[3]=(R[5]*R[6]-R[3]*R[8])/den;
		Ri[4]=(R[0]*R[8]-R[2]*R[6])/den;
		Ri[5]=(R[2]*R[3]-R[0]*R[5])/den;
		Ri[6]=(R[3]*R[7]-R[4]*R[6])/den;
		Ri[7]=(R[1]*R[6]-R[0]*R[7])/den;
		Ri[8]=(R[0]*R[4]-R[1]*R[3])/den;
	}
	else 
	{
		for(i=0; i<9; i++) 
		{
			Ri[i]=0.0;

		}
	}
}

//??
bool jacobi(float *a, float *eigen_val, float *eigen_vec,int n)
{
	int k = 0,i,j;
	const float e = 0.00001;		//?
	const int l = 10000;			//
	int p, q;
	float max_value =0;
	float cos_2a, sin_2a, cos_a, sin_a;
	float t, z;
	float a_pp;
	float a_qq;
	float a_pq;
	float a_pi;
	float a_qi;
	float r_ip;
	float r_iq;
	for (i = 0; i < n; i++) 
		eigen_vec[i * n + i] = 1;
	while (1)
	{
		max_value = 0;
		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++)
				if (fabs(a[i * n + j]) > max_value)
				{
					max_value = fabs(a[i * n + j]);
					p = i;
					q = j;
				}
				if (max_value < e || k > l) break;


				if (fabs(a[p * n + p] - a[q * n + q]) == 0)
				{
					sin_2a = 1;
					cos_2a = 0;
					cos_a = 1 / sqrt(2.0);
					sin_a = 1 / sqrt(2.0);
				}
				else 
				{	
					t = 2 * a[p * n + q] / (a[p * n + p] - a[q * n + q]);
					z = (a[p * n + p] - a[q * n + q]) / (2 * a[p * n + q]);

					if (fabs(t) < 1)
					{
						cos_2a = 1 / sqrt(1 + t * t);
						sin_2a = t / sqrt(1 + t * t);
					}
					else 
					{
						cos_2a = fabs(z) / sqrt(1 + z * z);
						sin_2a = (z > 0 ? 1 : (z < 0 ? -1 :0)) / sqrt(1 + z * z);
					}
					cos_a = sqrt((1 + cos_2a) / 2);
					sin_a = sin_2a / (2 * cos_a);
				}
				a_pp =a[p * n + p];
				a_qq = a[q * n + q];
				a_pq = a[p * n + q];
				a[p * n + p] = a_pp * cos_a * cos_a + a_qq * sin_a * sin_a + 2 * a_pq * cos_a * sin_a;
				a[q * n + q] = a_pp * sin_a * sin_a + a_qq * cos_a * cos_a - 2 * a_pq * cos_a * sin_a;

				for (i = 0; i < n; i++)
					if (i != p && i != q)
					{
						a_pi = a[p * n + i];
						a_qi = a[q * n + i];

						a[p * n + i] = a[i * n + p] = a_pi * cos_a + a_qi * sin_a;
						a[q * n + i] = a[i * n + q] = - a_pi * sin_a + a_qi * cos_a;
					}

					a[p * n + q] = a[q * n + p] = (a_qq - a_pp) * sin_2a / 2 + a_pq * cos_2a;
					//?
					for (i = 0; i < n; i++)
					{
						r_ip = eigen_vec[i * n + p];
						r_iq = eigen_vec[i * n + q];
						eigen_vec[i * n + p] = r_ip * cos_a + r_iq * sin_a;
						eigen_vec[i * n + q] = - r_ip * sin_a + r_iq * cos_a;
					}
					k++;
	}

	for (i = 0; i < n; i++) 
		eigen_val[i] = a[i * n + i];
	return TRUE;
}
//svd?
void svd( float *a, int m, int n,  float *d, float v[] )
{
	int i, j, k;
	float aT[2*CALIBRATION_POINT_NUM*9]={0};
	float aT_a[9*9] ={0};
	float tmp;
	float t[9] = {0};
	matrix_transport( aT, a, m, n );
	matrix_mult( aT_a, aT, a, n, m, n );
	jacobi(aT_a, d, v,n);
	//???

	for( i = 0; i < n-1; i++ )
	{
		tmp = d[ i ];
		for( k = 0; k < n; k++ )
			t[ k ] = v[ k * n + i ];
		for( j = i+1; j < n ; j++ )
		{
			if( d[ j ] > tmp )
			{
				d[ i ] = d[ j ];
				d[ j ] = tmp;
				tmp = d[ i ];
				for( k = 0; k < n; k++ )
				{
					v[ k * n + i ] = v[ k * n + j ];
					v[ k * n + j ] = t[ k ];
				}				
			}
		}
	}

}

//lhx,20150608????
/*static void camera_calibration(float actual_point[][2],float img_point[][2],float mapping_matrix[],int calibration_num,ALGCFGS *pCfgs)
{
int sub_calibration_num=4;
int subsection_num=(calibration_num-sub_calibration_num)/2+1;
int num=0;
float tx=0,ty=0,tu=0,tv=0;
float s1=0,s2=0;
float T1[9]={0};
float T2[9]={0};
int i,j;

float normalization_actual_x[4];
float normalization_actual_y[4];
float normalization_img_x[4];
float normalization_img_y[4];
float A[2*4*9]={0};
float v[9*9]={0};
float d[9]={0};
float L[9]={0};
float T1_inv[9]={0};
float temp[9]={0};
int overlap_row1,overlap_row2,flag=0;
//??????
for(num=0;num<subsection_num;num++)
{
//???
tx=ty=tu=tv=0;
s1=s2=0;		
for(i=0;i<sub_calibration_num;i++)
{
j=i+num*2;
tx=tx+actual_point[j][0];
ty=ty+actual_point[j][1];
tu=tu+img_point[j][0];
tv=tv+img_point[j][1];
}
tx=tx/sub_calibration_num;
ty=ty/sub_calibration_num;
tu=tu/sub_calibration_num;
tv=tv/sub_calibration_num;
for(i=0;i<sub_calibration_num;i++)
{
j=i+num*2;
s1=s1+sqrt((img_point[j][0]-tu)*(img_point[j][0]-tu)+(img_point[j][1]-tv)*(img_point[j][1]-tv));
s2=s2+sqrt((actual_point[j][0]-tx)*(actual_point[j][0]-tx)+(actual_point[j][1]-ty)*(actual_point[j][1]-ty));
}
s1=(fabs(s1)<1e-6)?0:sqrt(2.0)/s1;
s2=(fabs(s2)<1e-6)?0:sqrt(2.0)/s2;
T1[1]=T1[3]=T1[6]=T1[7]=0;
T1[0]=T1[4]=s1;
T1[2]=-s1*tu;
T1[5]=-s1*tv;
T1[8]=T2[8]=1;
T2[1]=T2[3]=T2[6]=T2[7]=0;
T2[0]=T2[4]=s2;
T2[2]=-s2*tx;
T2[5]=-s1*ty;
//?????	
for(i=0;i<sub_calibration_num;i++)
{
j=i+num*2;
normalization_img_x[i]=T1[0]*img_point[j][0]+T1[1]*img_point[j][1]+T1[2];
normalization_img_y[i]=T1[3]*img_point[j][0]+T1[4]*img_point[j][1]+T1[5];
normalization_actual_x[i]=T2[0]*actual_point[j][0]+T2[1]*actual_point[j][1]+T2[2];
normalization_actual_y[i]=T2[3]*actual_point[j][0]+T2[4]*actual_point[j][1]+T2[5];

}
//????,?????

for(i=0;i<2*4*9;i++)
{
A[i]=0.0;
}
for(i=0;i<sub_calibration_num;i++)
{
A[(2*i)*9+0]=normalization_actual_x[i];
A[(2*i)*9+1]=normalization_actual_y[i];
A[(2*i)*9+2]=1;
A[(2*i)*9+6]=-1*normalization_img_x[i]*normalization_actual_x[i];
A[(2*i)*9+7]=-1*normalization_img_x[i]*normalization_actual_y[i];
A[(2*i)*9+8]=-1*normalization_img_x[i];
A[(2*i+1)*9+3]=normalization_actual_x[i];
A[(2*i+1)*9+4]=normalization_actual_y[i];
A[(2*i+1)*9+5]=1;
A[(2*i+1)*9+6]=-1*normalization_img_y[i]*normalization_actual_x[i];
A[(2*i+1)*9+7]=-1*normalization_img_y[i]*normalization_actual_y[i];
A[(2*i+1)*9+8]=-1*normalization_img_y[i];

}
for(i=0;i<9*9;i++)
{
v[i]=0.0;
}
for(i=0;i<9;i++)
{
d[i]=0.0;
}
//svd??
svd( A, 2*sub_calibration_num, 9, d, v );
//v???????????????
//	float L[9]={0};
for( i = 0; i < 9; i++ )
L[i] = v[9*i+8];
//???????????inv(T1)*L*T2;
//		float T1_inv[9]={0};
for(i=0;i<9;i++)
{
T1_inv[i]=0.0;
}
matrix_inverse(T1, T1_inv);

for(i=0;i<9;i++)
{
temp[i]=0.0;

}


matrix_mult( temp, T1_inv, L, 3,3,3 );
matrix_mult(mapping_matrix,temp,T2,3,3,3);
flag=0;
if(num==0)
{
flag=1;
overlap_row1=0;
overlap_row2=0;
}
else
{
if(num==subsection_num-1)
flag=2;
overlap_row1=(img_point[num*2][1]+img_point[num*2+1][1])/2;
overlap_row2=(img_point[num*2+sub_calibration_num-4][1]+img_point[num*2+sub_calibration_num-3][1])/2;
}
img_to_actual(mapping_matrix,(img_point[num*2][1]+img_point[num*2+1][1])/2,(img_point[num*2+sub_calibration_num-2][1]+img_point[num*2+sub_calibration_num-1][1])/2,overlap_row1,overlap_row2,flag,pCfgs);
}


}*/
void sort_point(int array[][2],int length)
{
	int i=0,j=0;
	int temp[2];
	for(i=0;i<length-1;i++)
	{
		for(j=i+1;j<length;j++)
		{
			if(array[i][1]>array[j][1])
			{
				temp[0]=array[i][0];
				temp[1]=array[i][1];
				array[i][0]=array[j][0];
				array[i][1]=array[j][1];
				array[j][0]=temp[0];
				array[j][1]=temp[1];
			}
		}
	}
	if(array[0][0]>array[1][0])
	{
		temp[0]=array[0][0];
		temp[1]=array[0][1];
		array[0][0]=array[1][0];
		array[0][1]=array[1][1];
		array[1][0]=temp[0];
		array[1][1]=temp[1];
	}
	if(array[2][0]>array[3][0])
	{
		temp[0]=array[2][0];
		temp[1]=array[2][1];
		array[2][0]=array[3][0];
		array[2][1]=array[3][1];
		array[3][0]=temp[0];
		array[3][1]=temp[1];
	}

}
/*void camera_calibration(int base_line[][2],float base_length,int calibration_point[][2],float near_point_length,int laneNum,ALGCFGS *pCfgs)
{
int i=0,j=0;
//闁哄秴娲ら悾楣冨礌閸濆嫮鍘甸柣鈺侇嚟閸わ拷0闁炽儲鏌￠敓鏂ゆ嫹1闁炽儲鏌￠敓鏂ゆ嫹3闁炽儲鏌￠敓鏂ゆ嫹2闁炽儲鏌￠敓鏂ゆ嫹0
float k01=0,b01=0;
float k13=0,b13=0;
float k32=0,b32=0;
float k20=0,b20=0;
//婵☆偓鎷锋繛鏉戭儓濠у懘鏌嗛幘鑼跺幀闊洤鍟伴崵搴ㄥ棘濠婂懎鑺�
float k_base=0,b_base=0;
float k_road=0,b_road=0;
float pt_x=0,pt_y=0,pt1_x=0,pt1_y=0,pt2_x=0,pt2_y=0,pt3_x=0,pt3_y=0;
float dis[8]={0};//婵絽绻嬮柌婊勬姜閿曞倷澹曞☉鎾抽閻旓拷闁告垵妫涢崵搴☆啅椤旇法鐛�
int temp;
float pixel_length=0;
int start_point=0,end_point=0;
//閺夆晜绋栭、鎴﹀礆閸℃鍋�
int sub_num=6;
int block_num=FULL_ROWS/sub_num;
float ratio_array[FULL_ROWS]={0};
float calibration_base_line[FULL_ROWS]={0};
float temp1=0,temp2=0;
int val=0;
//閻庝絻顫夐悥锝囷拷瑙勮壘鐏忣垶宕洪悢鍝勪化閺夆晜绋栭、鎴﹀箳閹烘垹纰�
sort_point(calibration_point,4);
//婵懓鍊归悥锝囷拷瑙勮壘鐏忣垶宕洪悢鐑樼函缂佺偓瀵ч弸鈺呮偝閸パ勫闁规惌浜ｇ粣锟�
if(calibration_point[0][0]!=calibration_point[1][0])
{
k01=1.0*(calibration_point[0][1]-calibration_point[1][1])/(calibration_point[0][0]-calibration_point[1][0]);
b01=calibration_point[0][1]-k01*calibration_point[0][0];
}
else

{
k01=1e+6;
}
if(calibration_point[1][0]!=calibration_point[3][0])
{
k13=1.0*(calibration_point[1][1]-calibration_point[3][1])/(calibration_point[1][0]-calibration_point[3][0]);
b13=calibration_point[1][1]-k13*calibration_point[1][0];
}
else
{
k13=1e+6;
}
if(calibration_point[3][0]!=calibration_point[2][0])
{
k32=1.0*(calibration_point[3][1]-calibration_point[2][1])/(calibration_point[3][0]-calibration_point[2][0]);
b32=calibration_point[3][1]-k32*calibration_point[3][0];
}
else
{
k32=1e+6;
}
if(calibration_point[2][0]!=calibration_point[0][0])
{
k20=1.0*(calibration_point[2][1]-calibration_point[0][1])/(calibration_point[2][0]-calibration_point[0][0]);
b20=calibration_point[2][1]-k20*calibration_point[2][0];
}
else
{
k20=1e+6;
}
for(i=0;i<block_num;i++)
{
val=i*sub_num;
if(k20<1e+5)
{
temp1=(val-b20)/k20;
}
else
{
temp1=calibration_point[2][0];
}
if(k13<1e+5)
{
temp2=(val-b13)/k13;
}
else
{
temp2=calibration_point[1][0];
}
ratio_array[i]=abs(temp1-temp2);

}
for(i=0;i<block_num;i++)
{
ratio_array[i]=ratio_array[block_num-1]/ratio_array[i];
}
if(base_line[0][1]>base_line[1][1])
{
start_point=base_line[0][1];
end_point=base_line[1][1];
}
else
{
start_point=base_line[1][1];
end_point=base_line[0][1];
}
//閻庣數鎳撻悢锟介柛鎴濇閸ゅ孩绋夋繝鍛閻炴稑鏈悥锝囷拷鐧告嫹
temp1=0,temp2=0;
for(i=start_point;i>=end_point;i--)
{

temp1+=ratio_array[i/sub_num];
}
pixel_length=base_length/temp1;
//闁糕晛鎼崳顖滅棯婢跺摜鐟愭慨锝呯箣闁叉粓宕撹箛鏇狀槺濞寸媴缍�閵嗗啴姊归崹顔碱唺
for(int i=FULL_ROWS-1;i>=0;i--)
{
calibration_base_line[i]=pixel_length*ratio_array[i/sub_num];
}
//濞寸姴瀛╁〒鑸垫媴鎼达綆浼傞悹渚婄磿閻ｅ宕洪崫鍕珯缂佹崘娉曞▓鎴狅拷鍦仱濡绢垶姊归崹顔碱唺
pCfgs->actual_degree_length[FULL_ROWS-1]=near_point_length;
for(i=FULL_ROWS-2;i>=0;i--)
{
pCfgs->actual_degree_length[i]=calibration_base_line[i]+pCfgs->actual_degree_length[i+1];
}

//婵絽绻嬮柌婊勬姜閿曞倷澹曢弶鈺傜椤㈡垿寮介崶褏鏆�
if(base_line[0][0]!=base_line[1][0])
{
k_base=1.0*(base_line[0][1]-base_line[1][1])/(base_line[0][0]-base_line[1][0]);


b_base=base_line[0][1]-k_base*base_line[0][0];
pt_x=(b01-b_base)/(k_base-k01);
pt_y=k01*pt_x+b01;
}
else
{
//婵懓鍊峰锕傛倷閿燂拷
pt_x=base_line[0][0];
pt_y=k01*pt_x+b01;
}

for(i=0;i<laneNum;i++)
{
pt1_x=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[0].x+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[1].x)/2;
pt1_y=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[0].y+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[1].y)/2;
pt2_x=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2].x+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3].x)/2;
pt2_y=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2].y+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3].y)/2;
////////printf("LaneID=%d,[%d,%d],[%d,%d]\n",i,pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[0].x,pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[0].y,pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[1].x,pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[1].y);
// //////printf("LaneID=%d,[%d,%d],[%d,%d]\n",i,pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2].x,pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2].y,pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3].x,pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3].y);
if(pt1_x!=pt2_x)
{
k_road=1.0*(pt1_y-pt2_y)/(pt1_x-pt2_x);
b_road=pt1_y-k_road*pt1_x;
//婵懓鍊峰锕傛倷閿燂拷
pt3_x=(b01-b_road)/(k_road-k01);
pt3_y=k01*pt3_x+b01;
}
else
{
//婵懓鍊峰锕傛倷閿燂拷
pt3_x=pt1_x;
pt3_y=k01*pt3_x+b01;
}
dis[i]=pt3_y-pt_y;
for(j=0;j<FULL_ROWS;j++)
{
temp=(int)(dis[i]+j);
if(temp<0)
continue;
if(temp>=FULL_ROWS)
continue;
pCfgs->actual_distance[i][temp]=pCfgs->actual_degree_length[j];
}
if(dis[i]>0)
{
temp=(int)(dis[i]);
for(j=temp;j>=0;j--)
{
pCfgs->actual_distance[i][j]=pCfgs->actual_distance[i][j+1]+pixel_length*ratio_array[0];
}
}
else
{
temp=(int)(dis[i]+FULL_ROWS);
for(j=temp;j<FULL_ROWS;j++)
{
pCfgs->actual_distance[i][j]=pCfgs->actual_distance[i][j-1]-pixel_length;
}

}
}	
}*/
static void camera_calibration(int base_line[][2],float base_length,int calibration_point[][2],float near_point_length,int laneNum,ALGCFGS *pCfgs)
{
	int i=0,j=0;
	//闁哄秴娲ら悾楣冨礌閸濆嫮鍘甸柣鈺侇嚟閸わ拷0闁炽儲鏌￠敓鏂ゆ嫹1闁炽儲鏌￠敓鏂ゆ嫹3闁炽儲鏌￠敓鏂ゆ嫹2闁炽儲鏌￠敓鏂ゆ嫹0
	float k01=0,b01=0;
	float k13=0,b13=0;
	float k32=0,b32=0;
	float k20=0,b20=0;
	//闁哄秴娲ら悾楣冨礌閸濆嫮鍘靛☉鎾卞�栧顖滅棯鐠恒劍鐣卞ù婧垮�楅崑锟�0-2 1-3
	float x0=0,y0=0;
	float dis=1;
	//婵☆偓鎷锋繛鏉戭儓濠у懘鏌嗛幘鑼跺幀闊洤鍟伴崵搴ㄥ棘濠婂懎鑺�
	float k_road=0,b_road=0;
	//float k_base=0,b_base=0;
	float pt_x=0,pt_y=0,pt1_x=0,pt1_y=0,pt2_x=0,pt2_y=0,pt3_x=0,pt3_y=0;

	int temp;
	int start_point=0,end_point=0;
	float calibration_base_line[FULL_ROWS]={0};
	float dis_pixel[FULL_ROWS]={0};
	float temp1=0,temp2=0;
	//閻庝絻顫夐悥锝囷拷瑙勮壘鐏忣垶宕洪悢鍝勪化閺夆晜绋栭、鎴﹀箳閹烘垹纰�
	sort_point(calibration_point,4);
	//婵懓鍊归悥锝囷拷瑙勮壘鐏忣垶宕洪悢鐑樼函缂佺偓瀵ч弸鈺呮偝閸パ勫闁规惌浜ｇ粣锟�
	if(calibration_point[0][0]!=calibration_point[1][0])
	{
		k01=1.0*(calibration_point[0][1]-calibration_point[1][1])/(calibration_point[0][0]-calibration_point[1][0]);
		b01=calibration_point[0][1]-k01*calibration_point[0][0];
	}
	else
	{
		k01=1e+6;
	}
	if(calibration_point[1][0]!=calibration_point[3][0])
	{
		k13=1.0*(calibration_point[1][1]-calibration_point[3][1])/(calibration_point[1][0]-calibration_point[3][0]);
		b13=calibration_point[1][1]-k13*calibration_point[1][0];
	}
	else
	{
		k13=1e+6;
	}
	if(calibration_point[3][0]!=calibration_point[2][0])
	{
		k32=1.0*(calibration_point[3][1]-calibration_point[2][1])/(calibration_point[3][0]-calibration_point[2][0]);
		b32=calibration_point[3][1]-k32*calibration_point[3][0];
	}
	else
	{
		k32=1e+6;
	}
	if(calibration_point[2][0]!=calibration_point[0][0])
	{
		k20=1.0*(calibration_point[2][1]-calibration_point[0][1])/(calibration_point[2][0]-calibration_point[0][0]);
		b20=calibration_point[2][1]-k20*calibration_point[2][0];
	}
	else
	{
		k20=1e+6;
	}
	if(base_line[0][1]>base_line[1][1])
	{
		start_point=base_line[0][1];
		end_point=base_line[1][1];
	}

	else
	{
		start_point=base_line[1][1];
		end_point=base_line[0][1];
	}
	//濞戞挶鍊栧顖涙姜閿曞倷澹曠紒鐐濡叉悂宕归崒婊勭函缂佹拝鎷�
	if(k20>1e+5&&k13>1e+5)
	{
		dis=base_length/(start_point-end_point+1);
		calibration_base_line[FULL_ROWS-1]=near_point_length;
		for(i=FULL_ROWS-2;i>=0;i--)
		{
			calibration_base_line[i]=near_point_length+dis;
			dis_pixel[i]=dis;
		}
	}
	else
	{
		if(k20>1e+5)
		{
			x0=calibration_point[0][0];
			y0=k13*x0+b13;
		}
		else if(k13>1e+5)
		{
			x0=calibration_point[1][0];
			y0=k20*x0+b20;
		}
		else
		{
			x0=(b13-b20)/(k20-k13);
			y0=k13*x0+b13;
		}
		//閻庣數鎳撻悢锟介柛鎴濇閸ゅ孩绋夋繝鍛閻炴稑鏈悥锝囷拷鐧告嫹
		temp1=0,temp2=0;
		temp1=(float)(FULL_ROWS-1-end_point)/(float)(end_point-y0);
		temp2=(float)(FULL_ROWS-1-start_point)/(float)(start_point-y0);
		dis=(float)((base_line[1][0]-base_line[0][0])*(base_line[1][0]-base_line[0][0]));
		dis=(float)(dis+(base_line[1][1]-base_line[0][1])*(base_line[1][1]-base_line[0][1]));
		dis=sqrt(dis);
		base_length=base_length*(float)(start_point-end_point+1)/dis;
		dis=base_length/(temp1-temp2);
		calibration_base_line[FULL_ROWS-1]=near_point_length;
		for(i=FULL_ROWS-2;i>=0;i--)
		{
			calibration_base_line[i]=near_point_length+dis*(float)(FULL_ROWS-1-i)/(float)(i-y0);
			dis_pixel[i]=calibration_base_line[i]-calibration_base_line[i+1];
		}
	}
	//婵絽绻嬮柌婊勬姜閿曞倷澹曢弶鈺傜椤㈡垿寮介崶褏鏆�
	/*if(base_line[1][0]!=base_line[0][0])
	{
	k_base=1.0*(base_line[1][1]-base_line[0][1])/(base_line[1][0]-base_line[0][0]);
	b_base=base_line[1][1]-k_base*base_line[1][0];
	pt_x=(b01-b_base)/(k_base-k01);
	pt_y=k01*pt_x+b01;

	}
	else
	{
	pt_x=base_line[0][0];
	pt_y=k01*pt_x+b01;
	}*/
	pt_x=x0;
	pt_y=k01*pt_x+b01;
	for(i=0;i<laneNum;i++)
	{

		pt1_x=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[0].x+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[1].x)/2;
		pt1_y=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[0].y+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptCornerQ[1].y)/2;
		pt2_x=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2].x+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3].x)/2;
		pt2_y=(pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[2].y+pCfgs->DownSpeedCfg.SpeedEachLane[i].ptFourCorner[3].y)/2;
		if(pt1_x!=pt2_x)
		{
			k_road=1.0*(pt1_y-pt2_y)/(pt1_x-pt2_x);
			b_road=pt1_y-k_road*pt1_x;
			//婵懓鍊峰锕傛倷閿燂拷
			pt3_x=(b01-b_road)/(k_road-k01);
			pt3_y=k01*pt3_x+b01;
		}
		else
		{
			//婵懓鍊峰锕傛倷閿燂拷
			pt3_x=pt1_x;
			pt3_y=k01*pt3_x+b01;
		}

		temp=(int)(pt3_y+0.5);
        if(temp>479)
            temp=479;
        if(temp<0)
            temp=0;
		pCfgs->actual_distance[i][temp]=calibration_base_line[(int)(pt_y+0.5)];
		for(j=temp+1;j<FULL_ROWS;j++)
		{
			pCfgs->actual_distance[i][j]=pCfgs->actual_distance[i][j-1]-dis_pixel[j-1];
		}
		for(j=temp-1;j>=0;j--)
		{
            float p1=pCfgs->actual_distance[i][j+1];
            float p2=dis_pixel[j];
            pCfgs->actual_distance[i][j]=p1+p2;
        //	pCfgs->actual_distance[i][j]=pCfgs->actual_distance[i][j+1]+dis_pixel[j];
		}
	}
	for(j=0;j<FULL_ROWS;j++)
	{
		pCfgs->actual_degree_length[j]=pCfgs->actual_distance[laneNum-1][j];
	}
}
//????????,lhx,20150608
static void img_to_actual(float mapping_matrix[],int start_row,int end_row,int overlap_row1,int overlap_row2,int flag,ALGCFGS *pCfgs)
{
	int row,col;
	float a1=0,a2=0;
	float b1=0,b2=0;
	float c1=0,c2=0;
	int temp;
	float temp1=0;
	float temp2=0;
	if(flag==1)
	{
		if(start_row>=end_row)
			start_row=FULL_ROWS-1;
		else
			start_row=0;
	}
	if(flag==2)
	{
		if(start_row>=end_row)
			end_row=0;
		else	
			end_row=FULL_ROWS-1;
	}
	if(start_row>end_row)
	{
		temp=start_row;
		start_row=end_row;
		end_row=temp;
	}
	if(overlap_row1>overlap_row2)
	{
		temp=overlap_row1;
		overlap_row1=overlap_row2;
		overlap_row2=temp;
	}
	for(row=start_row;row<=end_row;row++)
	{
		for(col=0;col<FULL_COLS;col++)
		{
			a1=mapping_matrix[0]-mapping_matrix[6]*col;
			a2=mapping_matrix[3]-mapping_matrix[6]*row;
			b1=mapping_matrix[1]-mapping_matrix[7]*col;
			b2=mapping_matrix[4]-mapping_matrix[7]*row;
			c1=mapping_matrix[2]-mapping_matrix[8]*col;
			c2=mapping_matrix[5]-mapping_matrix[8]*row;
			temp1=0;
			temp2=0;
			if(abs(a1*b2-b1*a2)>0.00000000001)
			{
				temp1=(a2*c1-a1*c2)/(a1*b2-b1*a2);
				temp2=(b1*c2-b2*c1)/(a1*b2-b1*a2);
				if(row>=overlap_row1&&row<=overlap_row2&&flag!=1)
				{
					pCfgs->image_actual[row][col][0]+=temp1;
					pCfgs->image_actual[row][col][1]+=temp2;
					pCfgs->image_actual[row][col][0]=pCfgs->image_actual[row][col][0]/2;
					pCfgs->image_actual[row][col][1]=pCfgs->image_actual[row][col][1]/2;
				}
				else
				{
					pCfgs->image_actual[row][col][0]=temp1;
					pCfgs->image_actual[row][col][1]=temp2;
				}
			}
			else
			{
				pCfgs->image_actual[row][col][0]=0;
				pCfgs->image_actual[row][col][1]=0;
			}

		}
	}
}

static void get_actual_point(float actual_point[2],int row,int col,int limit_line,ALGCFGS *pCfgs)
{
	row=(row<limit_line)? limit_line:row;
	actual_point[0]=pCfgs->image_actual[row][col][0];
	actual_point[1]=pCfgs->image_actual[row][col][1];

}

float distance_two(float actual_point1[2],float actual_point2[2])
{
	float dis=0;
	dis=(actual_point1[0]-actual_point2[0])*(actual_point1[0]-actual_point2[0])+(actual_point1[1]-actual_point2[1])*(actual_point1[1]-actual_point2[1]);
	dis=sqrt(dis);
	return dis;

}
//1203
//0707
//void updateMinStd(unsigned char* OriginImage,int num,int width,int height,int startline,int stopline,ALGPARAMS *pParams,int cacPeriod)
//{
//
//	int thr=50;
//	//float standard_deviation=0.0;
//	if(width==20)
//	{
//		pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]=StandardDeviation (OriginImage,width,height,startline,stopline,0);
//	}
//	else if(num==0)//????
//	{
//		pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]=StandardDeviation (OriginImage,width,height,startline,stopline,2);
//	}
//	else
//	{
//		pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]=StandardDeviation (OriginImage,width,height,startline,stopline,1);
//	}
//	if(pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]==0)
//		pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]=100;
//	if(pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]==0)
//		pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]=100;
//	//if(pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num]==0)
//	//pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num]=100;
//	if( pParams->NormalDetectCfg.cSpeedDesc.min_tail_std==0)
//		pParams->NormalDetectCfg.cSpeedDesc.min_tail_std=100;
//	if (pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[num]==FALSE)
//	{
//
//		if (pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]<pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num])
//		{
//			if((num==4||num==9)&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==FALSE)||(num!=4&&num!=9))
//			    pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]=pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num];
//
//		}
//		if(num==4)
//		{
//			pParams->NormalDetectCfg.cSpeedDesc.min_tail_std=pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num];
//			pParams->NormalDetectCfg.cSpeedDesc.lmin_tial_std=0;
//			pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num]=0;
//		}
//	}
//	/*if(pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[num])
//	{
//		if(num!=4)
//		{
//			thr=((pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]+10)<20)? 20:(pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]+10);
//			thr=((pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]+10)>50)? 50:(pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]+10);
//		}
//		else
//		{
//			thr=((pParams->NormalDetectCfg.cSpeedDesc.min_tail_std+10)<20)? 20:(pParams->NormalDetectCfg.cSpeedDesc.min_tail_std+10);
//			thr=((pParams->NormalDetectCfg.cSpeedDesc.min_tail_std+10)>50)? 50:(pParams->NormalDetectCfg.cSpeedDesc.min_tail_std+10);
//		}
//	}*/
//	if (pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]<thr&&pParams->NormalDetectCfg.cSpeedDesc.begin_update_gmin_std[num]==FALSE)
//	{
//		if((num==4||num==9)&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==FALSE)||(num!=4&&num!=9))
//		{
//			pParams->NormalDetectCfg.cSpeedDesc.begin_update_gmin_std[num]=TRUE;
//			pParams->NormalDetectCfg.cSpeedDesc.num_cal_gmin[num]=0;
//			if(num==9)
//			{
//				pParams->NormalDetectCfg.cSpeedDesc.num_cal_lmin=0;
//				pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num]=0;
//			}
//
//		}
//
//	}
//	if (pParams->NormalDetectCfg.cSpeedDesc.begin_update_gmin_std[num]==TRUE)
//	{
//		pParams->NormalDetectCfg.cSpeedDesc.num_cal_gmin[num]=pParams->NormalDetectCfg.cSpeedDesc.num_cal_gmin[num]+1;
//
//		if (pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]>pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num])
//		{
//			if((num==9||num==4)&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==FALSE)||(num!=4&&num!=9))
//				pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]=pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num];
//		}
//		if(num==9&&pParams->NormalDetectCfg.cSpeedDesc.num_cal_gmin[num]%200==0)
//		{
//			if(abs(pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]-100)>1)
//			{
//				pParams->NormalDetectCfg.cSpeedDesc.num_cal_lmin++;
//			    pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num]+=pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num];
//				if(pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[num]==FALSE)
//					pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]=pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num];
//			}
//			pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]=100;
//		}
//		if (pParams->NormalDetectCfg.cSpeedDesc.num_cal_gmin[num]>cacPeriod)
//		{
//			if(num!=9)
//				pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]=pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num];
//			else 
//			{
//				if(pParams->NormalDetectCfg.cSpeedDesc.num_cal_lmin)
//				{
//					pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num]=pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num]/pParams->NormalDetectCfg.cSpeedDesc.num_cal_lmin;
//				    pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]=pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num];	
//				}
//			}
//			if(num==4)
//			{
//
//				if(pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[9]==FALSE)
//				{
//					pParams->NormalDetectCfg.cSpeedDesc.min_tail_std=pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num];
//				}
//
//				pParams->NormalDetectCfg.cSpeedDesc.lmin_tial_std+=pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num];
//				pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num]=pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[num]+1;
//
//			}
//			if(num==9)
//			{
//				if(pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[4])
//				{
//				    pParams->NormalDetectCfg.cSpeedDesc.lmin_tial_std=pParams->NormalDetectCfg.cSpeedDesc.lmin_tial_std/pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[4];
//					pParams->NormalDetectCfg.cSpeedDesc.min_tail_std=pParams->NormalDetectCfg.cSpeedDesc.lmin_tial_std;
//				}
//				pParams->NormalDetectCfg.cSpeedDesc.lmin_tial_std=0;
//				pParams->NormalDetectCfg.cSpeedDesc.gmin_part_std[4]=0;
//			}
//			pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]=100;
//			pParams->NormalDetectCfg.cSpeedDesc.begin_update_gmin_std[num]=FALSE;
//			pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[num]=TRUE;
//
//		}
//	} 
//}
void updateMinStd(unsigned char* OriginImage,int num,int width,int height,int startline,int stopline,ALGCFGS *pCfgs,ALGPARAMS *pParams,int cacPeriod)
{

	int thr=50;
	int i=0;
	if(pCfgs->gThisFrameTime==0)
	{
		for(i=0;i<15;i++)
		{
			pParams->NormalDetectCfg.cSpeedDesc.stat_min_std[num][i]=0;
		}
		pParams->NormalDetectCfg.cSpeedDesc.stat_std_num[num]=0;
	}
	if(width==20)
	{
		pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]=StandardDeviation (OriginImage,width,height,startline,stopline,0);
	}
	else if(num==0)//????
	{
		pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]=StandardDeviation (OriginImage,width,height,startline,stopline,2);
	}
	else
	{
		pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]=StandardDeviation (OriginImage,width,height,startline,stopline,1);
	}
	if(pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]==0)
		pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]=100;
	if(pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]==0)
		pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]=100;
	if (pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[num]==FALSE)
	{

		if (pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]<pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num])
		{
			if((num==4||num==9)&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==FALSE)||(num!=4&&num!=9))
			    pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]=pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num];

		}
	}
	if (pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num]<thr&&pParams->NormalDetectCfg.cSpeedDesc.begin_update_gmin_std[num]==FALSE)
	{
		if((num==4||num==9)&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==FALSE)||(num!=4&&num!=9))
		{
			pParams->NormalDetectCfg.cSpeedDesc.begin_update_gmin_std[num]=TRUE;
			pParams->NormalDetectCfg.cSpeedDesc.num_cal_gmin[num]=0;

		}

	}
	if (pParams->NormalDetectCfg.cSpeedDesc.begin_update_gmin_std[num]==TRUE&&pParams->NormalDetectCfg.cSpeedDesc.uTrafficFlow)
	{
		pParams->NormalDetectCfg.cSpeedDesc.num_cal_gmin[num]=pParams->NormalDetectCfg.cSpeedDesc.num_cal_gmin[num]+1;

		if (pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]>pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num])
		{
			if((num==9||num==4)&&(pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged==FALSE&&pParams->NormalDetectCfg.cSpeedDesc.bSpeedCounterChanged1==FALSE)||(num!=4&&num!=9))
				pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]=pParams->NormalDetectCfg.cSpeedDesc.cur_part_std[num];
		}
	}
	if (pParams->NormalDetectCfg.cSpeedDesc.num_cal_gmin[num]>cacPeriod)
	{
		if(abs(pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]-100)>1)
		{
			if(pParams->NormalDetectCfg.cSpeedDesc.stat_std_num[num]==15)
			{
				for(i=0;i<14;i++)
				{
					pParams->NormalDetectCfg.cSpeedDesc.stat_min_std[num][i]=pParams->NormalDetectCfg.cSpeedDesc.stat_min_std[num][i+1];
					pParams->NormalDetectCfg.cSpeedDesc.stat_min_std[num][14]=pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num];
				}
			}
			else
			{
				pParams->NormalDetectCfg.cSpeedDesc.stat_min_std[num][pParams->NormalDetectCfg.cSpeedDesc.stat_std_num[num]]=pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num];
				pParams->NormalDetectCfg.cSpeedDesc.stat_std_num[num]++;
				pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]=sort_aver(pParams->NormalDetectCfg.cSpeedDesc.stat_min_std[num],15,pParams->NormalDetectCfg.cSpeedDesc.stat_std_num[num]);
			}
		}
		
		pParams->NormalDetectCfg.cSpeedDesc.lmin_part_std[num]=100;
		pParams->NormalDetectCfg.cSpeedDesc.begin_update_gmin_std[num]=FALSE;
		pParams->NormalDetectCfg.cSpeedDesc.has_update_min_std[num]=TRUE;
	}
	if(pParams->NormalDetectCfg.cSpeedDesc.stat_std_num[num]==15&&pCfgs->gThisFrameTime%3000==0)
	{
		pParams->NormalDetectCfg.cSpeedDesc.min_part_std[num]=sort_aver(pParams->NormalDetectCfg.cSpeedDesc.stat_min_std[num],15,pParams->NormalDetectCfg.cSpeedDesc.stat_std_num[num]);
	}
}


//20150619  lhx 
int median(int* array,int length)
{
	int len=length/2;
	int temp;
	int i,j;
	for(i=0;i<=len;i++)
	{
		for(j=i+1;j<length;j++)
		{
			if(*(array+i)>*(array+j))
			{
				temp=*(array+i);
				*(array+i)=*(array+j);
				*(array+j)=temp;
			}
		}
	}
	return array[len];
}
////////////////////////////////////////////
static bool remove_bicycle(unsigned char* fgImg,unsigned char* diffImg,unsigned char* edgeImg,int startLine,int endLine,ALGCFGS *pCfgs,bool bGetbackFlag)
{
	int row,col;
	int k=0;
	int fg_length1[100]={0};
	int fg_length2[100]={0};
	int num=0;
	unsigned char fg_val=0;
	int continue_num=0;
	int near_edge_num=0;
	int val=0,val1=0,val2=0,val3=0;
	unsigned char value=0;
	int sum=0;
	int start_col=0,end_col=0;
	int projecty[200]={0};
	int array1[10][2]={0};
	int array_size=0;
	int max_value=0;
	float threshold1=0.4;
	float threshold2=0.3;
	float threshold3=0.4;
	float threshold4=0.3;
	bool is_car=TRUE;
	//????
	if(bGetbackFlag)
	{
		for(row=endLine;row>startLine;row--)
		{
			start_col=pCfgs->team_width-1;
			end_col=0;
			for(col=1;col<pCfgs->team_width-1;col++)
			{
				value=GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,fgImg);
				if(value)
				{
					sum=0;
					for(k=-1;k<=1;k++)
					{
						if(GetImagePointValue(col+k,row,pCfgs->team_height,pCfgs->team_width,fgImg))
							sum++;
						if(GetImagePointValue(col+k,row-1,pCfgs->team_height,pCfgs->team_width,fgImg))
							sum++;
						if(GetImagePointValue(col+k,row+1,pCfgs->team_height,pCfgs->team_width,fgImg))
							sum++;
					}
					if(sum<4)
					{
						SetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,0,fgImg);

					}
				}

			}
			SetImagePointValue(0,row,pCfgs->team_height,pCfgs->team_width,0,fgImg);
			SetImagePointValue(pCfgs->team_width-1,row,pCfgs->team_height,pCfgs->team_width,0,fgImg);
			for(col=1;col<pCfgs->team_width-1;col++)
			{
				fg_val=GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,fgImg);
				if(fg_val>30)
				{
					continue_num++;
				}
				else
				{
					continue_num=0;
				}
				if(continue_num>5)
				{
					start_col=col-continue_num+1;
					break;  
				}
				if(fg_val&&GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,edgeImg))
				{
					start_col=col;
					break;
				}
			}
			continue_num=0;
			for(col=pCfgs->team_width-2;col>=1;col--)
			{
				fg_val=GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,fgImg);
				if(fg_val>30)
				{
					continue_num++;
				}
				else
				{
					continue_num=0;
				}
				if(continue_num>5)
				{
					end_col=col+continue_num-1;
					break;  
				}
				if(fg_val&&GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,edgeImg))
				{
					end_col=col;
					break;
				}
			}
			for(col=0;col<start_col;col++)
			{
				SetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,0,fgImg);
			}
			for(col=pCfgs->team_width-2;col>end_col;col--)
			{
				SetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,0,fgImg);
			}
			if(end_col>start_col+2)
			{
				fg_length1[num]=end_col-start_col+1;
				num++;
				if(start_col<3||end_col>pCfgs->team_width-4)
				{
					near_edge_num++;
				}
			}
		}
		val1=median(fg_length1,num);
		pCfgs->guTestB[11]=num;
	}
	if(val1==0||num<10||bGetbackFlag==FALSE)
	{
		num=0;
		near_edge_num=0;
		for(row=0;row<SPEED_STARTER_ENDLINE-10;row++)
		{
			start_col=pCfgs->m_iWidth-1;
			end_col=0;
			for(col=1;col<pCfgs->m_iWidth-1;col++)
			{
				if(GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,diffImg)>15)
				{
					start_col=col;
					break;
				}
			}
			for(col=pCfgs->m_iWidth-2;col>0;col--)
			{
				if(GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,diffImg)>15)
				{
					end_col=col;
					break;

				}
			}

			if(end_col>start_col+2)
			{
				fg_length2[num]=end_col-start_col+1;
				num++;
				if(start_col==1||end_col==pCfgs->m_iWidth-2)
				{
					near_edge_num++;
				}
			}
		}
		val2=median(fg_length2,num);
		val=val2;
		/*if(val2>threshold2*pCfgs->m_iWidth)
		{
		for(row=0;row<SPEED_STARTER_ENDLINE;row++)
		{
		for(col=1;col<pCfgs->m_iWidth-1;col++)
		{
		if(GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,diffImg)>20)
		projecty[col]++;
		}

		}
		for(col=0;col<pCfgs->m_iWidth;col++)
		{
		if(projecty[col]&&((projecty[col-1]==0&&col>=1)||col==0))
		{
		array1[array_size][0]=col;
		}
		if(projecty[col]&&((projecty[col+1]==0&&col<pCfgs->m_iWidth-1)||col==pCfgs->m_iWidth-1))
		{
		array1[array_size][1]=col;
		if(array1[array_size][1]-array1[array_size][0]>2)
		{
		array_size++;
		}
		}
		}
		for(k=0;k<array_size;k++)
		{
		val3=array1[k][1]-array1[k][0]+1;
		if(val3>max_value)
		{
		max_value=val3;
		}
		}
		}*/
		is_car=(val>=threshold3*pCfgs->m_iWidth)||(val>=threshold4*pCfgs->m_iWidth&&near_edge_num>num/2);
	}
	else
	{
		val=val1;
		//????????
		if(val1>=threshold2*pCfgs->team_width)
		{
			for(row=endLine;row>startLine;row--)
			{
				for(col=0;col<pCfgs->team_width;col++)
				{
					if(GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,fgImg))
						projecty[col]++;
				}

			}
			for(col=0;col<pCfgs->team_width;col++)
			{
				if(projecty[col])
				{
					if((projecty[col-1]==0&&col>=1)||col==0)
					{
						array1[array_size][0]=col;
					}
					if((projecty[col+1]==0&&col<pCfgs->team_width-1)||col==pCfgs->team_width-1)
					{
						array1[array_size][1]=col;
						if(array1[array_size][1]-array1[array_size][0]>2)
						{
							val3=array1[array_size][1]-array1[array_size][0]+1;
							if(val3>max_value)
							{
								max_value=val3;
							}  
							array_size++;
						}   
					}
				}
			}
		}
		is_car=(val>=threshold1*pCfgs->team_width&&max_value>threshold1*pCfgs->team_width)||(val>=threshold2*pCfgs->team_width&&max_value>pCfgs->team_width/3&&near_edge_num>num/2);
	}
	if(is_car)
		return FALSE;
	else
		return TRUE;

}
//闁告ê顭峰▍搴ｆ崉椤栫偞妗ㄩ柛娆忕Т閸橈拷
static bool remove_nonvehicle(unsigned char* puNewImage,unsigned char* puBackImage,int position,ALGCFGS *pCfgs,ALGPARAMS *pParams,int flag)
{
	int row,col;
	int start_row=0,end_row=0;
	int val1=0,val2=0;
	int lightNum[576]={0};
	int interval_num=0;
	int maxUp=1000,maxDown=0,len=0;
	int thresholdValue =pParams->NormalDetectCfg.cSpeedDesc.uSpeedCounterChangedThreshold2;
	thresholdValue=thresholdValue<220? 220:thresholdValue;
	pCfgs->team_height=pParams->NormalDetectCfg.cSpeedDesc.QueDetectLines;
	pCfgs->team_width=pParams->NormalDetectCfg.cSpeedDesc.QueDetectColumns;
	pCfgs->m_iHeight=pParams->NormalDetectCfg.cSpeedDesc.DetectLines;
	if(flag==1)
	{
		start_row=position-20;
		end_row=position+pCfgs->m_iHeight;
	}
	else
	{
		start_row=position-200;
		end_row=position+50;
	}
	for(row=end_row;row>=start_row;row--)
	{
		for(col=0;col<pCfgs->team_width;col++)
		{
			val1=GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,puNewImage);
			val2=GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,puBackImage);
			if(val1>thresholdValue&&val1-val2>50)
			{
				lightNum[row]++;
			}
		}
	}
	interval_num=0;
	for(row=end_row;row>=start_row;row--)
	{
		len++;
		if(lightNum[row]>3)
		{
			if(interval_num>5)
			{
				if((maxDown-maxUp)<(len-interval_num)&&len>interval_num)
				{
					maxUp=row+interval_num;
					maxDown=row+len;
				}
				if(maxDown-maxUp>50)
					break;
				len=0;
			}
			else if(row==start_row)
			{
				if((maxDown-maxUp)<len)
				{
					maxUp=row;
					maxDown=row+len;
				}
				break;
			}
			interval_num=0;
		}
		else
		{
			if(row==start_row)
			{
				if((maxDown-maxUp)<(len-interval_num))
				{
					maxUp=row+interval_num;
					maxDown=row+len;
				}
				break;
			}
			interval_num++;
		}
	}
	////printf("flag=%d,start_row=%d,end_row=%d,dis=%d\n",flag,start_row,end_row,maxDown-maxUp);
	if(flag==0)
	{

		if(maxDown>(position-50)&&(maxDown-maxUp)>80)
		{
			return TRUE;
		}
		else
		{
			return FALSE;
		}
	}
	else
	{
		if((maxDown-maxUp)>50)
		{
			return TRUE;
		}
		else
		{
			return FALSE;
		}
	}
}
static bool remove_nonvehicle1(unsigned char* puNewImage,unsigned char* puBackImage,int position,ALGCFGS *pCfgs,ALGPARAMS *pParams)
{
	int row,col;
	int start_row=0,end_row=0;
	int val=0,val1=0,val2=0,val3=0;
	int thresholdValue =20;
	int count=0,count1=0,count2=0,sum=0;
	pCfgs->team_height=pParams->NormalDetectCfg.cSpeedDesc.QueDetectLines;
	pCfgs->team_width=pParams->NormalDetectCfg.cSpeedDesc.QueDetectColumns;
	start_row=position-20;
	end_row=position+SPEED_STARTER_ENDLINE-10;//闁哄秷顫夊畵渚�宕犻崫鍕幍濞达絽绉堕悿鍡樻交濞戞粠鏀介悹渚婄磿閻ｏ拷
	for(row=start_row;row<end_row;row++)
	{
		count1=0;
		count2=0;
		for(col=1;col<pCfgs->team_width;col++)
		{
			val=GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,puNewImage);
			val1=GetImagePointValue(col-1,row,pCfgs->team_height,pCfgs->team_width,puNewImage);
			val2=GetImagePointValue(col,row-1,pCfgs->team_height,pCfgs->team_width,puNewImage);
			val3=GetImagePointValue(col,row,pCfgs->team_height,pCfgs->team_width,puBackImage);
			if(abs(val-val3)>20)
			{
				if(abs(val-val1)>thresholdValue||abs(val-val2)>thresholdValue)
				{
					count2++;
				}
				count1++;
			}	
		}
		if(count1>5)
		{

			if((float)count2/count1>0.5)
			{
				count++;
			}
			sum++;
		}
	}
	if(count>sum/2)
	{
		return TRUE;
	}
	else
	{
		return FALSE;
	}

}
bool remove_shadow(unsigned char* puNewImage,unsigned char* puBackImage,ALGCFGS *pCfgs,int threshold)
{
	int row=0,col=0;
	int value1=0,value2=0;
	double val_ratio=0;
	int num1=0,num2=0;
	double average = 0.0;
	double variance = 0.0;
	int aver_back=0,aver_curr=0;
	int end_line=SPEED_STARTER_ENDLINE;
	int array[100]={0};
	aver_curr=AveGray(puNewImage,pCfgs->m_iWidth,pCfgs->m_iHeight,0,pCfgs->m_iHeight,pCfgs);
	aver_back=AveGray(puBackImage,pCfgs->m_iWidth,pCfgs->m_iHeight,0,pCfgs->m_iHeight,pCfgs);
	if(aver_curr<aver_back)
	{
		for(row=0;row<pCfgs->m_iHeight;row++)
		{
			for(col=0;col<pCfgs->m_iWidth;col++)
			{
				value1=GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,puBackImage);
				value2=GetImagePointValue(col,row,pCfgs->m_iHeight,pCfgs->m_iWidth,puNewImage);
				if(abs(value1-value2)>20)
				{
					if((value1-value2)>20)
					{
						val_ratio=(float)value2/(float)value1;
						array[(int)(val_ratio*100)]++;
						num1++;
					}
					num2++;
				}
			}
		}
		for(row=0;row<100;row++)
		{
			if(array[row])
			{
				average+=(float)(row)*array[row];
			}
		}
		average=average/num1;
		for(row=0;row<100;row++)
		{
			if(array[row])
			{
				variance+=((float)(row)-average)*((float)(row)-average)*array[row];
			}
		}
		variance /= num1;
		variance = sqrt(variance);
		if(variance<threshold&&(float)num1/num2>0.5)//15
		{
			return TRUE;
		}
		else
		{
			return FALSE;
		}

	}
	else
		return FALSE;
}

int AveGray(unsigned char* puNewImage,int nWidth,int nHight,int start_line,int end_line,ALGCFGS *pCfgs)//C????鍧�?鎹ｈ?02xyx
{
	int row,col,count = 0;
	int sum = 0;
	int average=0;
	for (row=start_line;row<end_line;row+=2)
	{
		for(col=0;col<nWidth;col+=2)
		{
			count++;
			sum+=GetImagePointValue(col,row,nHight,nWidth,puNewImage);	

		}
	}
	//_CrtDumpMemoryLeaks();
	average=(count>0)? (sum/count):0;
	/*for (row=0;row<nHight;row++)
	{
	for(col=0;col<nWidth;col++)
	{
	sum+=(GetImagePointValue(col,row,nHight,nWidth,puNewImage)-average)*(GetImagePointValue(col,row,nHight,nWidth,puNewImage)-average);
	}
	}
	sum=(float)(sum/count);
	sum=sqrt(sum);*/
	return average;
}
///////////////////////////////////////////////
float fuzzy(unsigned char* puNewImage,int nWidth,int nHight)//E??????缂佸偊鎷�?E?624xyx
{
	float degree=0.0;
	int i,j;
	unsigned char x1,x2,x3;
	float temp=0.0;
	int count=0;
	for(i=0;i<nHight;i+=4)  
	{  
		for(j=0;j<nWidth;j+=4)  
		{  
			x1=*(puNewImage+i*nWidth+j);
			x2=*(puNewImage+(i+1)*nWidth+j);
			x3=*(puNewImage+i*nWidth+j+1);
			degree=(x2-x1)*(x2-x1)+(x3-x1)*(x3-x1);
			temp+=sqrt(degree);
			temp+=abs(x2-x1)+abs(x3-x1);
			count++;
		}  
	}  
	degree=temp/count;
	return degree;
}

void im2bw(unsigned char* originImage,unsigned char* bwImage,int threshold,int width,int height)
{
	int row=0,col=0,Mean=0,l=0,k=0;
	for(row=0; row<height; row++)
	{
		for(col=0; col<width; col++)
		{	
			Mean =GetImagePointValue(col,row,height,width,originImage);	

			if (Mean>threshold)
				SetImagePointValue(col,row,height,width,255,bwImage);
			else
				SetImagePointValue(col,row,height,width,0,bwImage);	

		}
	}
}

float connect_filt(ALGCFGS *pCfgs, ALGPARAMS *pParams,CRect rtConnect0)
{
	int i;
	float Arearate=0.0,coverrate;
	int AreaA,AreaB;
	int vertical=0,recover=0;
	int Array_size=ArrayGetSize((Array *)pParams->ForegroundArray1);
	int subx=0,suby=0;
	float confidence=0.0,maxconfidence=0.0;
	for (i=0;i<Array_size;i++)
	{
		subx=min(pParams->ForegroundArray1->array[i].m_rtConnect.right,rtConnect0.right)-max(pParams->ForegroundArray1->array[i].m_rtConnect.left,rtConnect0.left);
		suby=min(pParams->ForegroundArray1->array[i].m_rtConnect.bottom,rtConnect0.bottom)-max(pParams->ForegroundArray1->array[i].m_rtConnect.top,rtConnect0.top);
		if (subx>0&&suby>0)
		{			
			AreaA=RECT_Width(pParams->ForegroundArray1->array[i].m_rtConnect)*RECT_Height(pParams->ForegroundArray1->array[i].m_rtConnect);
			AreaB=RECT_Width(rtConnect0)*RECT_Height(rtConnect0);
			Arearate=(float)AreaB/AreaA;
			coverrate=(float)(subx*suby)/AreaA;
			confidence=Arearate+coverrate;
			maxconfidence=maxconfidence>confidence?maxconfidence:confidence;

		}

	}
	return maxconfidence;

}
bool Color_deviate(unsigned char* uImage,unsigned char* vImage,int width,int height)
{	
	float ave_a=0,ave_b=0,std_a=0,std_b=0;
	int x=0,y=0;
	float color_deviate=0;
	int pixelnum=0;
	int temp_a,temp_b;
	//pixelnum=width*height;
	for (y=0;y <height;y+=4)
	{
		for (x=0;x <width;x+=4)
		{
			ave_a +=GetImagePointValue( x, y, height, width, uImage )-128;
			ave_b +=GetImagePointValue( x, y, height, width, vImage )-128;
			pixelnum++;
		}
	} 
	ave_a /= pixelnum;
	ave_b /= pixelnum;


	for (y=0;y <height;y+=4)
	{
		for (x=0;x <width;x+=4)
		{
			temp_a = GetImagePointValue( x, y, height, width, uImage )-128;
			std_a += (temp_a - ave_a)*(temp_a - ave_a);
			temp_b = GetImagePointValue( x, y, height, width, vImage )-128;
			std_b += (temp_b - ave_b)*(temp_b - ave_b);
		}
	}
	std_a /= pixelnum;
	std_b /= pixelnum;
	color_deviate=sqrt(ave_a*ave_a+ave_b*ave_b)/sqrt(std_a+std_b);
	//////printf("\ncolor deviate is:%f\n,",color_deviate*10);

	if (color_deviate>=5||color_deviate<0.05)
	{
		// //////printf("color deviate is true\n");
		return TRUE;

	} 
	else
	{
		////////printf("color deviate is false\n");
		return FALSE; 
	}

}
int max_std_probab(ALGPARAMS *pParams,int num)
{
	int i=0,max_ind=0,max_value=pParams->NormalDetectCfg.cSpeedDesc.std_probab[0];
	for(i=1;i<num;i++)
	{
		if(pParams->NormalDetectCfg.cSpeedDesc.std_probab[i]>max_value)
		{
			max_value=pParams->NormalDetectCfg.cSpeedDesc.std_probab[i];
			max_ind=i;
		}
	}
	return max_ind;
}
//txl,20160105
bool visible_judge(Uint16 *a,int visib_length,int threshold)
{
	int i=0,num=0;
	for (i=0;i<visib_length;i++)
	{
		if (a[i]<threshold)
		{
			num++;
		}
		else
		{
			break;
		}

	}
	if (num>(visib_length/2))
	{
		return TRUE;
	} 
	else
	{
		return FALSE;
	}
}
bool Initialize_target(CTarget& target)
{
	target.vehicle=Rect();
	target.vehicle_next=Rect();
	target.tracker_point_num=0;
	target.detected=false;
	target.tracked=false;
	target.flow_count=false;
	target.vehicle_id=0;
	target.num_detected=0;
	target.trajectory_num=0;
	target.vx=0;
	target.vy=0;
	target.continue_num=0;
	target.credit_num=0;
	return true;
}
int round(float f)
{ 
	if ((int)f+0.5>f) 
		return (int)f;
	else 
		return (int)f + 1; 

}
float median(vector<float> v)
{
	int n = floor((float)v.size() / 2);
	nth_element(v.begin(), v.begin()+n, v.end());
	//float val=0;
	//val= (v.size()%2==0)? (v[n-1]+v[n])/2: v[n];
	//return val;
	return v[n];
}
//????
vector<Rect> detectAndDisplay( Mat frame, Mat maskFrame, ADBOOST_DETECT pAdboostDetect)
{
	std::vector<Rect> faces;
	/*Mat frame_gray;

	if(frame.channels()>1)
		cvtColor( frame, frame_gray, CV_BGR2GRAY );
	else
		frame.copyTo(frame_gray);*/
	//int64 c1 = cv::getTickCount();
	pAdboostDetect.face_cascade.detectMultiScale( frame, maskFrame, faces,pAdboostDetect.scaleFactor, pAdboostDetect.minNeighbor, 0, 
		Size(pAdboostDetect.min_width, pAdboostDetect.min_height),Size(pAdboostDetect.max_width,pAdboostDetect.max_height) );
	//int64 c2 = cv::getTickCount();
	//double time = double(c2-c1)/cv::getTickFrequency();
	//cout<<"time ="<<time<<endl;

	return faces;
}
void vehicle_detection(Mat currFrame,Mat maskFrame,ALGCFGS *pCfgs,vector<Rect>& vehicle_roi,vector<Rect> candidate_roi)
{
	int i=0;
	ADBOOST_DETECT  AdboostDetect;
	AdboostDetect=pCfgs->AdboostDetect;
	vehicle_roi.clear();
	for(int i=0;i<candidate_roi.size();i++)
	{
		if(candidate_roi[i].width<30||candidate_roi[i].height<30)
			continue;
		Mat candidate(currFrame(candidate_roi[i]));
		Mat maskROI(maskFrame(candidate_roi[i]));
		//vector<int> one_overlapNum;

		vector<Rect> one_vehicle_roi= detectAndDisplay(candidate, maskROI, AdboostDetect);

		for(int j=0;j<one_vehicle_roi.size();j++)
		{
			one_vehicle_roi[j].x=one_vehicle_roi[j].x+candidate_roi[i].x;
			one_vehicle_roi[j].y=one_vehicle_roi[j].y+candidate_roi[i].y;
			vehicle_roi.push_back(one_vehicle_roi[j]);
		}

	}
	//region fuse
	merge_overlapped_rect(vehicle_roi,50);
	//printf("################%d\n",vehicle_roi.size());
}
//txl@20160323
void vehicle_detection_night(Mat currFrame,Mat maskFrame,ALGCFGS *pCfgs,vector<Rect>& vehicle_roi)
{ 
	int i=0;	
	vehicle_roi.clear();
	vector<int> one_overlapNum;
	ADBOOST_DETECT  AdboostDetect;
	AdboostDetect=pCfgs->AdboostDetect;
	vehicle_roi= detectAndDisplay(currFrame,maskFrame,AdboostDetect);
	//region fuse
	merge_overlapped_rect(vehicle_roi,10);
	for(i=0;i<vehicle_roi.size();i++)
	{
		vehicle_roi[i].x+=pCfgs->detectROI.x;
		vehicle_roi[i].y+=pCfgs->detectROI.y;
	}
}
//?????
int overlapRatio(const Rect& r1,const Rect& r2)
{
	int ratio=0;
	int x1   =   MAX(r1.x, r2.x);
	int y1   =   MAX(r1.y, r2.y);
	int x2   =   MIN(r1.x+r1.width, r2.x+r2.width);
	int y2   =   MIN(r1.y+r1.height, r2.y+r2.height);

	if(x1<x2 && y1<y2) //intersect
	{
		int area_r1=r1.width*r1.height;
		int area_r2=r2.width*r2.height;
		int area_intersection=(x2-x1)*(y2-y1);

		ratio = area_intersection*100/(area_r1+area_r2-area_intersection);
	}
	return ratio;
}
int overlapRatio1(const Rect& r1,const Rect& r2)
{
	int ratio=0;
	int x1   =   MAX(r1.x, r2.x);
	int y1   =   MAX(r1.y, r2.y);
	int x2   =   MIN(r1.x+r1.width, r2.x+r2.width);
	int y2   =   MIN(r1.y+r1.height, r2.y+r2.height);

	if(x1<x2 && y1<y2) //intersect
	{
		int area_r1=r1.width*r1.height;
		int area_r2=r2.width*r2.height;
		int area_intersection=(x2-x1)*(y2-y1);

		ratio = area_intersection*100/max(1,min(area_r1,area_r2));
	}
	return ratio;
}
//??????
void merge_overlapped_rect(vector<Rect>& candidate_roi,int ratio_threshold)
{
	vector<Rect> rect_temp;
	int rect_num = candidate_roi.size();
	for(int i=0; i<rect_num; ++i)
	{
		rect_temp.clear();
		for(int j=1; j<candidate_roi.size(); ++j)
		{
			if(overlapRatio1(candidate_roi[0],candidate_roi[j])>ratio_threshold)
			{
				int x11   =   MIN(candidate_roi[0].x, candidate_roi[j].x);
				int y11   =   MIN(candidate_roi[0].y, candidate_roi[j].y);
				int x22   =   MAX(candidate_roi[0].x+candidate_roi[0].width, candidate_roi[j].x+candidate_roi[j].width);
				int y22   =   MAX(candidate_roi[0].y+candidate_roi[0].height, candidate_roi[j].y+candidate_roi[j].height);

				candidate_roi[0] = Rect(x11,y11,x22-x11,y22-y11);
				rect_num--;
			}	
			else
				rect_temp.push_back(candidate_roi[j]);
		}

		rect_temp.push_back(candidate_roi[0]);

		candidate_roi = rect_temp;
		if(candidate_roi.size() == 1)
			break;
	}
}
//////////////////////////////////////////////////////////////////////////////LK??
void LKTracker(Mat currFrame,Mat prevFrame,CTarget* targets,int targets_size)
{
	int i=0,j=0,k=0;
	vector<Point2f> points1;
	vector<Point2f> points2;
	vector<Point2f> pointsFB;
	vector<Point2f> features;//??????
	int max_count=3;
	double qlevel=0.01; //???????
	double minDist=10;//???????????
	vector<uchar> status; //??????
	vector<float> err; //??????
    cv::Size window_size=Size(15,15);
	int level=5;
	vector<uchar> FB_status;
	vector<float> similarity;
	vector<float> FB_error;
	float simmed;
	float fbmed;
    cv::TermCriteria term_criteria=TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03);;
	float lambda=0.5;
	if(targets_size==0)
		return;
	for(i=0;i<targets_size;i++)
	{

		targets[i].tracked=false;
		targets[i].detected=false;
		unsigned char* p1=currFrame.data;
		unsigned char* p2=prevFrame.data;
		int num=0;
		for(j=targets[i].vehicle.y;j<(targets[i].vehicle.y+targets[i].vehicle.height);j++)
		{
			unsigned char* p11=p1+j*currFrame.cols;
			unsigned char* p21=p2+j*prevFrame.cols;
			for(k=targets[i].vehicle.x;k<(targets[i].vehicle.x+targets[i].vehicle.width);k++)
			{
				if((p11[k]-p21[k]>10)||(p21[k]-p11[k])>10)
				{
                    num++;
				}

			}

		}
		if((float)num/(targets[i].vehicle.height*targets[i].vehicle.width)<0.05)
		{
			targets[i].tracked=true;
			targets[i].vehicle_next=targets[i].vehicle;
			targets[i].vx=0;
			targets[i].vy=0;
			continue;
		}
		////?????????
		if(targets[i].tracker_point_num)
		{
			points1.clear();
			for(k=0;k<targets[i].tracker_point_num;k++)
			{

				if(targets[i].tracker_point[k].x>=targets[i].vehicle.x&&targets[i].tracker_point[k].x<=(targets[i].vehicle.x+targets[i].vehicle.width)&&targets[i].tracker_point[k].y>=targets[i].vehicle.y&&targets[i].tracker_point[k].y<=(targets[i].vehicle.y+targets[i].vehicle.height))
				{
					points1.push_back(targets[i].tracker_point[k]);
				}
			}
			for(k=0;k<points1.size();k++)
			{
				targets[i].tracker_point[k]=points1[k];
			}
			targets[i].tracker_point_num=points1.size();

		}
		Mat vehicle_roi=prevFrame(targets[i].vehicle);
		points1.clear();
		points2.clear();
		pointsFB.clear();
		similarity.clear();
		//???????2?,????????
		if(targets[i].tracker_point_num<2)
		{
			features.clear();
            cv::goodFeaturesToTrack (vehicle_roi,features,max_count,qlevel,minDist);
			for(k=0;k<features.size();k++)
			{
				targets[i].tracker_point[k+targets[i].tracker_point_num]=Point2f(features[k].x+targets[i].vehicle.x,features[k].y+targets[i].vehicle.y);
				//pCfgs->targets[i].tracker_point.push_back(Point2f(features[k].x+pCfgs->targets[i].vehicle.x,features[k].y+pCfgs->targets[i].vehicle.y));
			}
			targets[i].tracker_point_num+=features.size();
			features.clear();
		}
		vehicle_roi.release();
		if(targets[i].tracker_point_num)//???????0
		{
			for(k=0;k<targets[i].tracker_point_num;k++)
			{
				points1.push_back(targets[i].tracker_point[k]);
			}
			//??
            calcOpticalFlowPyrLK( prevFrame,currFrame, points1, points2, status,similarity, window_size, level, term_criteria, lambda, 0);
			//??
            calcOpticalFlowPyrLK( currFrame,prevFrame, points2, pointsFB, FB_status,FB_error, window_size, level, term_criteria, lambda, 0);
			for( j= 0; j<points1.size(); ++j )
			{
				FB_error[j] = norm(pointsFB[j]-points1[j]);
			}
			////////////////////////////////////////////////////////////////////??????????
			//Mat rec0(10,10,CV_8U);
			//Mat rec1(10,10,CV_8U);
			//Mat res(1,1,CV_32F);

			//for (j = 0; j < points1.size(); j++)
			//{
			//	if (status[j] == 1) 
			//	{
			//		getRectSubPix( prevFrame, Size(10,10), points1[j],rec0 );
			//		getRectSubPix( currFrame, Size(10,10), points2[j],rec1);
			//		matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED);
			//		similarity[j] = ((float *)(res.data))[0];
			//	} 
			//	else 
			//	{
			//		similarity[j] = 0.0;
			//	}
			//}
			//rec0.release();
			//rec1.release();
			//res.release();
			/////////////////////////////////////////////////////////////////////??????????????????
			//simmed = median(similarity);
			//for( j=k = 0; j<points2.size(); ++j )
			//{
			//	if( !status[j])
			//		continue;
			//	if(/*similarity[j]> simmed&&*/similarity[j]>0.9)
			//	{
			//		points1[k] = points1[j];
			//		points2[k] = points2[j];
			//		FB_error[k] = FB_error[j];
			//		k++;
			//	}
			//}
			//if (k==0)
			//	targets[j].tracked=false;
			//if(k>0)
			{
				//points1.resize(k);
				//points2.resize(k);
				//FB_error.resize(k);
				//fbmed = median(FB_error);
				for( j=k = 0; j<points2.size(); ++j )
				{
					if(/*FB_error[i] <= fbmed&&*/FB_error[j]<0.1)
					{
						points1[k] = points1[j];
						points2[k] = points2[j];
						k++;
					}
				}
				if(k>0)
				{
					points1.resize(k);
					points2.resize(k);
					targets[i].tracked=true;
					for(j=0;j<k;j++)
					{
						targets[i].tracker_point[j]=points2[j];
					}
					targets[i].tracker_point_num=k;
				}
				else
					targets[i].tracked=false;
			}
			//////////////////////////////////////////////////////////////////////////?????????
			if (targets[i].tracked)
			{
				int npoints = (int)points1.size();
				vector<float> xoff(npoints);
				vector<float> yoff(npoints);
				for (j=0;j<npoints;j++)
				{
					xoff[j]=points2[j].x-points1[j].x;
					yoff[j]=points2[j].y-points1[j].y;
				}
				float dx = median(xoff);
				float dy = median(yoff);
				//float dx = mean(xoff);
				//float dy = mean(yoff);
				//????
				float s;
				if (npoints>1)
				{
					vector<float> d;
					d.reserve(npoints*(npoints-1)/2);
					for (int j=0;j<npoints;j++)
					{
						for (int k=j+1;k<npoints;k++)
						{
							d.push_back(norm(points2[j]-points2[k])/norm(points1[j]-points1[k]));
						}
					}
					s = median(d);
				}
				else 
				{
					s = 1.0;
				}
				float s1 = 0.5*(s-1)*targets[i].vehicle.width;
				float s2 = 0.5*(s-1)*targets[i].vehicle.height;
				targets[i].vehicle_next.x =round(targets[i].vehicle.x + dx -s1);
				targets[i].vehicle_next.y = round(targets[i].vehicle.y + dy -s2);
				targets[i].vehicle_next.width = round(targets[i].vehicle.width*s);
				targets[i].vehicle_next.height = round(targets[i].vehicle.height*s);
				targets[i].vx=dx;
				targets[i].vy=dy;
			}
			else
			{
				/*for(k=0;k<targets[i].tracker_point_num;k++)
				{
					targets[i].tracker_point[k]=Point2f(0,0);
				}*/
				memset(targets[i].tracker_point,0,targets[i].tracker_point_num*sizeof(Point2f));
				targets[i].tracker_point_num=0;
			}
			//////////////////////////////////////////////////////////////////////////

		}
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CTarget* find_nearest_rect(Rect vehicle_roi,CTarget* targets,CTarget* candidate_targets,int targets_size,int candidate_targets_size,int& overlap_ratio1,int& overlap_ratio2)
{

	if(targets_size<1&&candidate_targets_size<1)
	{
		return NULL;
	}
	int i=0;
	int overlap_ratio11=0,overlap_ratio12=0;
	int overlap_ratio21=0,overlap_ratio22=0;
	overlap_ratio1=0,overlap_ratio2=0;
	if(targets_size)
	{
		CTarget* it_max1=&targets[0];
		for(i=0;i<targets_size;i++)
		{
			int ratio=0;
			ratio=MAX(overlapRatio(vehicle_roi,targets[i].vehicle_next),overlapRatio(vehicle_roi,targets[i].vehicle));
			if(ratio>overlap_ratio11)
			{
				overlap_ratio11=ratio;
				it_max1=&targets[i];
			}
		}
		overlap_ratio12=MAX(overlapRatio1(vehicle_roi,it_max1->vehicle_next),overlapRatio1(vehicle_roi,it_max1->vehicle));
		if(overlap_ratio11>10)
		{
			overlap_ratio1=overlap_ratio11;
			overlap_ratio2=overlap_ratio12;
			return it_max1;
		}
	}
	if(candidate_targets_size)
	{
		CTarget* it_max2=&candidate_targets[0];
		for(i=0;i<candidate_targets_size;i++)
		{
			int ratio=0;
			ratio=MAX(overlapRatio(vehicle_roi,candidate_targets[i].vehicle_next),overlapRatio(vehicle_roi,candidate_targets[i].vehicle));
			if(ratio>overlap_ratio21)
			{
				overlap_ratio21=ratio;
				it_max2=&candidate_targets[i];
			}
		}
		overlap_ratio22=MAX(overlapRatio1(vehicle_roi,it_max2->vehicle_next),overlapRatio1(vehicle_roi,it_max2->vehicle));
		if(overlap_ratio21>10)
		{
			overlap_ratio1=overlap_ratio21;
			overlap_ratio2=overlap_ratio22;
			return it_max2;
		}
	}
	return NULL;


}
void get_candidate(const unsigned char* currFrame,unsigned char* prevFrame,unsigned char* tempFrame,int rows,int cols,vector<Rect>& candidate_roi,ALGCFGS *pCfgs,ALGPARAMS *pParams,int threshold)
{
	int scale=2;
	int loopDirect[8][2] = {{-scale, scale}, {0, scale}, {scale, scale}, {scale, 0},
	{scale, -scale}, {0, -scale}, {-scale, -scale}, {-scale, 0}};
	int i=0,j=0,l=0,t=0,s=0;
	int row,col;
	float ratio=0.1;
	const unsigned char* curr_data=currFrame;
	unsigned char* prev_data=prevFrame;
	unsigned char* temp_data=tempFrame;
	unsigned char* fg_data1=prevFrame;
	unsigned char* temp_data1=tempFrame;
	int dwArrayCurrentSize=0,dwArraySize,dwProcessedSize;
	CRect rtConnect;
	int width,height,width_extend,height_extend;
	int x1=pCfgs->detectROI.x;
	int x2=pCfgs->detectROI.x+pCfgs->detectROI.width;
	int y1=pCfgs->detectROI.y;
	int y2=pCfgs->detectROI.y+pCfgs->detectROI.height;
	candidate_roi.clear();
	if(rows&&cols)
	{
		int offset2 = scale*cols;
		memset(tempFrame,0,rows*cols*sizeof(unsigned char));
		//int ratio1=pParams->NormalDetectCfg.cSpeedDesc.SpeedDetectorWorkMode.bNight==TRUE? 2:1;
		for(j=y1;j<y2;j+=scale)
		{
			int offset1 = j * cols;
			curr_data=currFrame + offset1;
			prev_data=prevFrame + offset1;
			temp_data=tempFrame + offset1;
			for(i=x1;i<x2;i+=scale)
			{
				if((curr_data[i]-prev_data[i]>threshold)||(prev_data[i]-curr_data[i]>threshold))
				{
					temp_data[i-offset2]=255;
					temp_data[i+offset2]=255;
					temp_data[i-2*offset2]=255;
					temp_data[i+2*offset2]=255;
					temp_data[i]=255;
				}
				/*else
				prev_data[i]=0;*/
			}
		}
		memcpy(prevFrame,tempFrame,cols*rows*sizeof(unsigned char));
		memset(tempFrame,0,rows*cols*sizeof(unsigned char));
		memset(pParams->m_pPointArray,0,rows*cols*sizeof(CPoint));
		//?????
		for(j=y1; j<y2; j+=3*scale)
		{
			unsigned char* fg_data=prevFrame+j*cols;
			unsigned char* temp_data=tempFrame+j*cols;
			for(i=x1; i<x2; i+=3*scale)
			{
				if(fg_data[i]==255 && temp_data[i]==0)
				{
					// Got a start point 
					pParams->m_pPointArray[0].x = rtConnect.left = rtConnect.right = i;
					pParams->m_pPointArray[0].y = rtConnect.top = rtConnect.bottom = j;
					dwArrayCurrentSize = 1;
					dwProcessedSize = 0;
					do
					{
						dwArraySize = dwArrayCurrentSize;
						for(l=dwProcessedSize; l<dwArraySize; l++)
						{
							/*if(rtConnect.left > pParams->m_pPointArray[l].x)
							{
							rtConnect.left = pParams->m_pPointArray[l].x;
							}
							else if(rtConnect.right < pParams->m_pPointArray[l].x)
							{
							rtConnect.right =pParams->m_pPointArray[l].x;
							}
							if(rtConnect.top > pParams->m_pPointArray[l].y)
							{
							rtConnect.top = pParams->m_pPointArray[l].y;
							}
							else if(rtConnect.bottom < pParams->m_pPointArray[l].y)
							{
							rtConnect.bottom = pParams->m_pPointArray[l].y;
							}*/
							rtConnect.left = (rtConnect.left < pParams->m_pPointArray[l].x)? rtConnect.left:pParams->m_pPointArray[l].x;
							rtConnect.right = (rtConnect.right > pParams->m_pPointArray[l].x)? rtConnect.right:pParams->m_pPointArray[l].x;
							rtConnect.top = (rtConnect.top < pParams->m_pPointArray[l].y)? rtConnect.top:pParams->m_pPointArray[l].y;
							rtConnect.bottom = (rtConnect.bottom > pParams->m_pPointArray[l].y)? rtConnect.bottom:pParams->m_pPointArray[l].y;

							for (t = 0; t < 8; ++t)
							{
								row = pParams->m_pPointArray[l].y + loopDirect[t][1];
								col = pParams->m_pPointArray[l].x + loopDirect[t][0];
								if(row >= 0 && row < rows && 
									col >= 0 && col < cols)
								{
									fg_data1=prevFrame+row*cols;
									temp_data1=tempFrame+row*cols;
									if(fg_data1[col]==255 && temp_data1[col]==0)
									{
										temp_data1[col]= 255;
										pParams->m_pPointArray[dwArrayCurrentSize].x = col;
										pParams->m_pPointArray[dwArrayCurrentSize].y = row;
										dwArrayCurrentSize++;
									}
								}
							}
						}
						dwProcessedSize = dwArraySize;
					}while(dwArrayCurrentSize > dwArraySize);
					width=rtConnect.right-rtConnect.left+1;
					height=rtConnect.bottom-rtConnect.top+1;
					if(width*height >100&&width>30&&height>40)
					{
						//瀵瑰尯鍩熻繘琛屾墿灞�
						width_extend = (width * 205) >> 10; // 0.2 * width
						height_extend = (height * 205) >> 10; // 0.2 * height

						rtConnect.left=rtConnect.left-width_extend;
						rtConnect.top=rtConnect.top-5;
						rtConnect.right=rtConnect.right+width_extend;
						rtConnect.bottom=rtConnect.bottom+height_extend;
						rtConnect.left=(rtConnect.left>0)? rtConnect.left:0;
						rtConnect.top=(rtConnect.top>0)? rtConnect.top:0;
						rtConnect.right=(rtConnect.right>(cols-1))? (cols-1):rtConnect.right;
						rtConnect.bottom=(rtConnect.bottom>(rows-1))? (rows-1):rtConnect.bottom; 
						Rect detect_roi;
						detect_roi.x=rtConnect.left;
						detect_roi.y=rtConnect.top;
						detect_roi.width=rtConnect.right-rtConnect.left+1;
						detect_roi.height=rtConnect.bottom-rtConnect.top+1;
						candidate_roi.push_back(detect_roi);

					}
				}
			}
		}
		//??????
	merge_overlapped_rect(candidate_roi,50);

	}
}
void get_target(unsigned char* currImg,unsigned char* prevImg,unsigned char* maskImg,unsigned char* tempImg,Uint16 width,Uint16 height,ALGCFGS *pCfgs,ALGPARAMS *pParams,Uint16 flag)
{
	//鎷疯礉鏁版嵁
	Mat currFrame,prevFrame,maskFrame;
	currFrame.create(height, width, CV_8U);
	prevFrame.create(height, width, CV_8U);
	maskFrame.create(height, width, CV_8U);
	memcpy(currFrame.data,currImg,width*height);
	memcpy(prevFrame.data,prevImg,width*height);
	memcpy(maskFrame.data,maskImg,width*height);
    ///////////////////////////////////////////////////////////////////////////////////////////
	//LK璺熻釜
	LKTracker(currFrame,prevFrame,pCfgs->targets,pCfgs->targets_size);
	LKTracker(currFrame,prevFrame,pCfgs->candidate_targets,pCfgs->candidate_targets_size);
	//浜害鏆楁椂锛屽抚宸槇鍊奸噰鐢�5锛屽叾浠�10
	int threshold=10;
	pCfgs->guBackGrey=(pCfgs->guBackGrey)? pCfgs->guBackGrey:50;
	threshold=(pCfgs->guBackGrey>30)? 10:5;
	vector<Rect> candidate_roi;
	//妫�娴嬪�欓�夊尯鍩�
	if(pCfgs->guBackGrey>30)
		//get_candidate(inBufs->bufDesc[0].buf,pParams->PrevFullImage,pParams->puSobelTempImageX,FrameHeight,FrameWidth,AdboostDetect,candidate_roi,pParams,threshold);
		get_candidate(currImg,prevImg,tempImg,height,width,candidate_roi,pCfgs,pParams,threshold);
	else
		candidate_roi.push_back(pCfgs->detectROI);
	vector<Rect> vehicleROI;
	//pCfgs->candidateROI[0]=pCfgs->detectROI;
	//妫�娴�
	int i=0,j=0;
	if(flag==1)
	{
		vehicle_detection(currFrame,maskFrame,pCfgs,vehicleROI,candidate_roi);
	}
	else
	{
		vehicle_detection_night(currFrame(pCfgs->detectROI), maskFrame,pCfgs,vehicleROI);
	}
	///////////////////////////////////////////////////////////////////////////////////////////
	//鍒嗘瀽妫�娴嬬粨鏋�
	for(i=0;i<vehicleROI.size();i++)
	{
		if((vehicleROI[i].y+vehicleROI[i].height)>(pCfgs->detectROI.y+pCfgs->detectROI.height))
			continue;
		int overlap_ratio1=0,overlap_ratio2=0;
		CTarget* t;
		t=find_nearest_rect(vehicleROI[i],pCfgs->targets,pCfgs->candidate_targets,pCfgs->targets_size,pCfgs->candidate_targets_size,overlap_ratio1,overlap_ratio2);
		if(t)
		{
			if(overlap_ratio1<20)
			{
				bool is_replace=false;
				is_replace=(overlap_ratio2>90);
				if(is_replace)
				{
					int id=t->vehicle_id;
					int detect_num=t->num_detected;
					Initialize_target(*t);
					t->vehicle =vehicleROI[i];
					t->vehicle_next =vehicleROI[i];
					t->vehicle_id =id;
					t->num_detected=detect_num+1;
					t->detected=true;
				}
				continue;
			}
			t->detected=true;
			t->num_detected++;

			if(t->tracked)
			{
				t->vehicle_next.x = (t->vehicle_next.x+vehicleROI[i].x)/2;  
				t->vehicle_next.y = (t->vehicle_next.y+vehicleROI[i].y)/2;
				t->vehicle_next.width = (t->vehicle_next.width+vehicleROI[i].width)/2;
				t->vehicle_next.height =(t->vehicle_next.height+vehicleROI[i].height)/2;
			}
			else
			{
				t->vehicle_next=vehicleROI[i];

			}
		}
		else
		{

			CTarget nt; 
			Initialize_target(nt);
			nt.vehicle =vehicleROI[i];
			nt.vehicle_next =vehicleROI[i];
			//nt.vehicle_id = target_id++;
			//nt.vehicle_id=0;
			nt.detected=true;
			nt.num_detected++;
			pCfgs->candidate_targets[pCfgs->candidate_targets_size]=nt;
			pCfgs->candidate_targets_size++;
		}
	}

	for(i=0;i<pCfgs->targets_size;i++)
	{
		pCfgs->targets[i].vehicle=pCfgs->targets[i].vehicle_next;
		if((pCfgs->targets[i].vehicle.x+pCfgs->targets[i].vehicle.width)>=(pCfgs->detectROI.x+pCfgs->detectROI.width)||(pCfgs->targets[i].vehicle.y+pCfgs->targets[i].vehicle.height)>=(pCfgs->detectROI.y+pCfgs->detectROI.height)||pCfgs->targets[i].vehicle.x<pCfgs->detectROI.x||pCfgs->targets[i].vehicle.y<pCfgs->detectROI.y)
		{
			if(pCfgs->targets_size>1)
			{
				for(j=i+1;j<pCfgs->targets_size;j++)
				{
					pCfgs->targets[j-1]=pCfgs->targets[j];
				}
			}
			pCfgs->targets_size--;
			i--;
			continue;
		}

		if(pCfgs->targets[i].continue_num>5000)
		{
			if(pCfgs->targets_size>1)
			{
				for(j=i+1;j<pCfgs->targets_size;j++)
				{
					pCfgs->targets[j-1]=pCfgs->targets[j];
				}
			}
			pCfgs->targets_size--;
			i--;
			continue;

		}
		pCfgs->targets[i].vehicle_next.x=round(pCfgs->targets[i].vehicle.x+pCfgs->targets[i].vx);
		pCfgs->targets[i].vehicle_next.y=round(pCfgs->targets[i].vehicle.y+pCfgs->targets[i].vy);
		pCfgs->targets[i].vehicle_next.width=pCfgs->targets[i].vehicle.width;
		pCfgs->targets[i].vehicle_next.height=pCfgs->targets[i].vehicle.height;
		//pCfgs->targets[i].trajectory[pCfgs->targets[i].trajectory_num]=Point2f(pCfgs->targets[i].vehicle.x+pCfgs->targets[i].vehicle.width/2,pCfgs->targets[i].vehicle.y+pCfgs->targets[i].vehicle.height/2);
		//pCfgs->targets[i].trajectory_num++;
		pCfgs->targets[i].continue_num++;
	}
	for(i=0;i<pCfgs->candidate_targets_size;i++)
	{
		pCfgs->candidate_targets[i].vehicle=pCfgs->candidate_targets[i].vehicle_next;
		if((pCfgs->candidate_targets[i].vehicle.x+pCfgs->candidate_targets[i].vehicle.width)>=(pCfgs->detectROI.x+pCfgs->detectROI.width)||(pCfgs->candidate_targets[i].vehicle.y+pCfgs->candidate_targets[i].vehicle.height)>=(pCfgs->detectROI.y+pCfgs->detectROI.height)||pCfgs->candidate_targets[i].vehicle.x<pCfgs->detectROI.x||pCfgs->candidate_targets[i].vehicle.y<pCfgs->detectROI.y)
		{
			if(pCfgs->candidate_targets_size>1)
			{
				for(j=i+1;j<pCfgs->candidate_targets_size;j++)
				{
					pCfgs->candidate_targets[j-1]=pCfgs->candidate_targets[j];
				}
			}
			pCfgs->candidate_targets_size--;
			i--;
			continue;
		}
		pCfgs->candidate_targets[i].vehicle_next.x=round(pCfgs->candidate_targets[i].vehicle.x+pCfgs->candidate_targets[i].vx);
		pCfgs->candidate_targets[i].vehicle_next.y=round(pCfgs->candidate_targets[i].vehicle.y+pCfgs->candidate_targets[i].vy);
		pCfgs->candidate_targets[i].vehicle_next.width=pCfgs->candidate_targets[i].vehicle.width;
		pCfgs->candidate_targets[i].vehicle_next.height=pCfgs->candidate_targets[i].vehicle.height;
		//pCfgs->candidate_targets[i].trajectory[pCfgs->candidate_targets[i].trajectory_num]=Point2f(pCfgs->candidate_targets[i].vehicle.x+pCfgs->candidate_targets[i].vehicle.width/2,pCfgs->candidate_targets[i].vehicle.y+pCfgs->candidate_targets[i].vehicle.height/2);
		//pCfgs->candidate_targets[i].trajectory_num++;
		pCfgs->candidate_targets[i].continue_num++;

		if(pCfgs->candidate_targets[i].num_detected>=2&&(pCfgs->candidate_targets[i].vehicle.y+pCfgs->candidate_targets[i].vehicle.height)<(pCfgs->detectPosition+10)/*(pCfgs->detectROI.y+pCfgs->detectROI.height-30)*/)
		{
			pCfgs->targets[pCfgs->targets_size]=pCfgs->candidate_targets[i];
			pCfgs->targets[pCfgs->targets_size].vehicle_id = pCfgs->target_id++;
			if(pCfgs->target_id>1000)
				pCfgs->target_id=0;
			pCfgs->targets_size++;
			if(pCfgs->candidate_targets_size>1)
			{
				for(j=i+1;j<pCfgs->candidate_targets_size;j++)
				{
					pCfgs->candidate_targets[j-1]=pCfgs->candidate_targets[j];
				}
			}
			pCfgs->candidate_targets_size--;
			i--;
			continue;
		}
	}

}
