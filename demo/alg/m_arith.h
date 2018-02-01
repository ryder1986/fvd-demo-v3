#ifndef __NEW_ARITH_H__
#define __NEW_ARITH_H__
#include "cascadedetect.h"
#include <opencv2/opencv.hpp>
#include "DSPARMProto.h"
using namespace cv;

//BY DAVID 20130322 FOR xDM
//#define		TRUE	1
//#define  	FALSE	0
/*
#ifdef WX_IVD_EXPORTS
#define CIVD_SDK_API extern "C" __declspec(dllexport)
#else
#define CIVD_SDK_API extern "C" __declspec(dllimport)
#endif
*/

#define CIVD_SDK_API

typedef  unsigned int 	Uint32;
typedef  int 			Int32;
typedef  unsigned short Uint16;
typedef  short 			Int16;
//typedef  unsigned short bool;
//typedef  unsigned short bool;
typedef  unsigned char	Uint8;
extern int frame;
//typedef  signed char	Int8;

//#define DETECTRECT_WIDTH_MAX		720//500
#define DETECTRECT_WIDTH_MAX		640//500
#define DETECTRECT_HEIGHT_MAX		480//240


//#define DETECTRECT_WIDTH_MAX		640//500
//#define DETECTRECT_HEIGHT_MAX		480//240

#define DETECTRECT_WIDTH_DEFAULT	160
#define DETECTRECT_HEIGHT_DEFAULT	40
#define LANE_AMOUNT_DEFAULT			3
#define DETECT_WIDTH			200
#define DETECT_HEIGHT			120
#define DETECT_ALGORITHM			3

#ifndef __UPLOAD_DETECT__
#define __UPLOAD_DETECT__

#define TRUE 1
#define FALSE 0
#define	VISIB_LENGTH		10//txl,20160105
typedef struct LaneInitial
{
    Uint16 		uTransFactor;
    Uint32		uGraySubThreshold;
    Uint32 		uSpeedCounterChangedThreshold;
    Uint32 		uSpeedCounterChangedThreshold1;
    Uint32 		uSpeedCounterChangedThreshold2;


    Uint16		uDayNightJudgeMinContiuFrame;//锟叫伙拷时锟斤拷值锟斤拷锟斤拷�??
    Uint16		uComprehensiveSens;//取锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷帧锟斤拷
    Uint16		uDetectSens1;//锟叫讹拷锟角筹拷头锟斤拷锟斤拷小锟斤拷锟斤拷
    Uint16		uDetectSens2;
    Uint16		uStatisticsSens1;
    Uint16 		uStatisticsSens2;	//by david 20130910 from tagCfgs
    Uint16		uSobelThreshold;//sobel锟斤拷�??
    Uint16		uEnvironmentStatus;//锟斤拷锟斤拷锟斤拷锟斤拷锟叫伙拷时锟斤拷锟街碉拷锟轿�?1,2,3,4
    Uint16      uEnvironment;           //1锟斤拷锟斤拷锟斤拷锟斤拷锟�2锟斤拷锟�? 锟斤拷锟较诧拷锟斤拷
        float base_length;//锟斤拷准锟竭筹拷
    float near_point_length;//锟斤拷锟斤拷锟斤拷锟�?
}LANEINISTRUCT;//702

typedef void            XDAS_Void;
typedef Uint8           XDAS_bool;


typedef unsigned char           XDAS_Int8;      /**< Actual size chip dependent. */
typedef Uint8                   XDAS_UInt8;     /**< Actual size chip dependent. */
typedef Int16                   XDAS_Int16;     /**< Actual size of type is 16 bits. */
typedef Uint16                  XDAS_UInt16;    /**< Actual size of type is 16 bits. */
typedef Int32                   XDAS_Int32;     /**< Actual size of type is 32 bits. */
typedef Uint32                  XDAS_UInt32;    /**< Actual size of type is 32 bits. */
typedef struct XDM1_SingleBufDesc {
    XDAS_Int8   *buf;       /**< Pointer to a buffer address. */
    XDAS_Int32  bufSize;    /**< Size of @c buf in 8-bit bytes. */
    XDAS_Int32  accessMask; /**< Mask filled by the algorithm, declaring
                             *   how the buffer was accessed <b>by the
                             *   algorithm processor</b>.
                             *
                             *   @remarks  If the buffer was <b>not</b>
                             *             accessed by the algorithm
                             *             processor (e.g., it was filled
                             *             via DMA or other hardware
                             *             accelerator that <i>doesn't</i>
                             *             write through the algorithm's
                             *             CPU), then no bits in this mask
                             *             should be set.
                             *
                             *   @remarks  It is acceptible (and
                             *             appropriate!)to set several
                             *             bits in this mask if the
                             *             algorithm accessed the buffer
                             *             in several ways.
                             *
                             *   @remarks  This mask is often used by the
                             *             application and/or framework
                             *             to appropriately manage cache
                             *             on cache-based systems.
                             *
                             *   @sa XDM_AccessMode
                             */
} XDM1_SingleBufDesc;
typedef struct IVIDEO1_BufDescIn {
    XDAS_Int32  numBufs;        /**< Number of buffers in bufDesc[]. */
    XDAS_Int32  frameWidth;     /**< Width of the video frame. */
    XDAS_Int32  frameHeight;    /**< Height of the video frame. */
    XDAS_Int32  framePitch;     /**< Frame pitch used to store the frame.
                                 *
                                 *   @remarks   This field can also be used to
                                 *              indicate the padded width.
                                 */
    XDM1_SingleBufDesc bufDesc[8]; /**< Picture buffers. */
} IVIDEO1_BufDescIn;
typedef struct tagDownSpeedLane
{
    //detect region
    CPoint			ptFourCorner[16];
    //direction
    int				uDetectDerection;
    //alg params
    int				uDayNightJudgeMinContiuFrame;	//锟斤拷锟斤拷转锟斤拷锟斤拷锟斤拷锟斤�?
    int				uGetBackMinContiuFrame;			//锟桔猴拷锟斤拷锟斤拷锟斤�?
    int				uVehicleHeadMinRow;				//锟斤拷锟斤拷锟斤拷锟斤拷锟�1
    int				uVehicleHeadMinCol;				//锟斤拷锟斤拷锟斤拷锟斤拷锟�2
    int				uAddCountLineProportion;		//统锟斤拷锟斤拷锟斤拷锟斤�?1
    int				uRepeatFrame;					//统锟斤拷锟斤拷锟斤拷锟斤�?2
    int				uSobelThreshold;				//锟斤拷�?�锟斤拷锟斤拷锟斤拷
    //vehicle length and speed transformation params
    int				SpeedConversionRatio;
    int				SpeedModifyRatio;	//signed
    int				LengthConversionRatio;
    int				LengthModifyRatio;//signed
}DOWN_SPEEDLANE;

typedef struct tagDetectEvent
{
    int	 EventCode;
    bool EventFlag;

}DETECTEVENT;

typedef struct tagOutBuf
{

    int DetectInSum[8];
    int DetectOutSum[8];
    DETECTEVENT DetectEvent[8];
    unsigned int thresholdValue;
    unsigned int frame;
    unsigned int calarflag[8];
    unsigned int AlarmLineflag[8][5];
    CPoint LineUp[8][2];
    bool IsCarInTail[8];//尾锟斤拷占锟斤拷状�??
    //unsigned int LineUp[8][2];
    Uint16	uVehicleSpeed[8];			//锟斤拷锟斤拷
    Uint16	uVehicleLength[8];			//锟斤拷锟斤拷
    Uint16	uVehicleHeight[8];			//锟斤拷锟斤拷
    Uint16 uVehicleQueueLength[8];//锟斤拷锟脚队筹拷锟斤�?
    bool fuzzyflag;//锟斤拷频锟届常状�?
    bool visibility;//锟杰硷拷锟阶刺�
        Uint16 uActualDetectLength[8];
        Uint16 uActualTailLength[8];
        Uint16 uDegreePoint[20][2];
    Uint16	uEnvironmentStatus;
    bool getQueback_flag[8];//txl,20160104
}OUTBUF;

typedef struct tagCAMERASTA
{
    Uint16 Reserved       		:11;
    Uint16 CameraStaCntFlag     :1;
    Uint16 CfgSyntaxSta   		:1;
    Uint16 ImageProcSta			:1;
    Uint16 CfgSta				:1;
    Uint16 VideoSta   			:1;

}CAMERASTA;

typedef struct tagVEHICLEFLOW
{
    struct
    {
        Uint16 VehicleVelocity	:8;		//锟斤拷锟斤拷锟劫讹拷
        Uint16 VehicleLen		:8;		//锟斤拷锟斤拷锟斤拷锟斤拷
    }VehicleVL;
    struct
    {
        Uint16 VehicleExistTimes  	:8;	//锟斤拷锟斤拷锟斤拷时锟斤�?
        Uint16 MotorCadeLen		 	:8;	//锟斤拷锟接筹拷锟斤拷
    }MotorTL;
}VEHICLEFLOW;

typedef struct tagVEHICLEEVENT
{
    Uint16 EventDataValid		:1;
    Uint16 FlowDataValid		:1;///锟斤拷锟斤拷锟斤拷锟斤拷萁峁癸拷锟叫�?
    Uint16 bReserved			:2;

    Uint16 bSignalLightSta		:1;
    Uint16 bTrafficJam			:1;
    Uint16 bOverDelayTimer		:1;
    Uint16 bOverMotorCadeLen  	:1;
    Uint16 bOverLongTimer     	:1;
    Uint16 bRushDash			:1;
    Uint16 bCrossLine			:1;
    Uint16 bRetrograde			:1;
    Uint16 bOverLenVehicle		:1;
    Uint16 bUltraLow			:1;
    Uint16 bOverSpeed			:1;
    Uint16 bVehicleSta			:1;

}VEHICLEEVENT;

typedef struct tagEACHLANERESULTS
{
    Uint16 LaneID;
    VEHICLEFLOW VehicleFlow;
    VEHICLEEVENT VehicleEvent;

}EACHLANERESULTS;

typedef struct tagLANERESULTS
{
    Uint16 LaneSum;
//	Uint16 LaneID;
    EACHLANERESULTS EachLaneResults[8];
}LANERESULTS;

typedef struct tagCAMERARESULTS
{
    //Uint16 CameraID;
    CAMERASTA  CameraSta;
    LANERESULTS LaneResults;
}CAMERARESULTS;

typedef struct tagRESULTSBUF
{

    Uint16 CameraID;////�?路锟斤拷锟斤拷锟斤拷ID锟斤拷锟斤拷锟较达拷锟斤拷锟斤拷锟缴癸拷状�?�时锟斤�?0xffff锟斤拷锟斤拷锟斤拷同CameraNum
    Uint16 DataLen;

    Uint16 BootStatus;
    Uint16 CameraNum;////�?路锟斤拷锟斤拷锟斤拷ID锟紺
    CAMERARESULTS CameraResults;
    Uint16  DetectorStaFlag[4];
}UPLOADRESULTSBUF,*PUPLOADRESULTSBUF;
#endif

/****锟斤拷锟秸碉拷锟斤拷锟斤拷锟斤拷锟侥硷拷锟侥结构****/

/*锟斤拷锟斤拷锟侥结构*/
#define	CAMERA_STRUCT_LENGTH	64
struct	FuncStruct
{
    Uint16	FuncType;
    Uint16	DetectIdNumber[4];
};


struct	StableStruct
{
    Uint16	DetectAmount;
    Uint16	MaxDeviationPixels;
    Uint16	MinDeviationPixels;
    struct FuncStruct	StableFuncType;
};

struct	ContrastStruct
{
    Uint16	DetectAmount;
    Uint16	MaxContrastDiff;
    Uint16	MinContrastDiff;
    struct FuncStruct	ContrastFuncType;
};

typedef struct tagCAMERASTRUCT
{
    Uint16	CameraId;
    Uint16	DetectAmount;
    Uint16	VehicleClass1;
    Uint16	VehicleClass2;
    Uint16	VehicleClass3;
    Uint16	VehicleClass4;
    Uint16	MaxBrightDeviationLimit;
    Uint16	MinBrightDeviationLimit;
    Uint16	MaxBrightDiffLimit;
    Uint16	MinBrightDiffLimit;
    Uint16	MinHeadDistance;
    Uint16	MaxVehicleLength;
    Uint16	MinVehicleLength;
    Uint16	MaxVehicleWidth;
    Uint16	MinVehicleWidth;
    Uint16	Bt835Bright;
    Uint16	Bt835Contrast;
    Uint16	Bt835Sat_U;
    Uint16	Bt835Sat_V;
    Uint16	Bt835Hue;
    Uint16	Resv[15];
    bool			bCameraEventHasSent;
    bool			bCameraEventHappened;
    Uint16	bCounterPole;
    Uint16			bCameraWorkInCross; //txl,20150709,bool->Uint16
    Uint16	LaneAmount;
    Uint16	LanesId[8];
    struct StableStruct	StableCfg;
    struct ContrastStruct	ContrastCfg;

    //added by david 20131011
    Uint16		uEnvironmentStatus;		//added by david 20131011

    Uint16		uDayNightJudgeMinContiuFrame;//取锟狡癸拷时锟斤拷为锟角灯癸拷幕叶锟斤拷锟斤拷锟斤拷锟斤拷拥锟斤拷锟斤拷锟街★拷锟�
    Uint16		uGetBackMinContiuFrame;//取锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷帧锟斤拷
    Uint16		uVehicleHeadMinRow;//锟叫讹拷锟角筹拷头锟斤拷锟斤拷小锟斤拷锟斤拷
    Uint16		uInThreshold;
    Uint16		uSquareThreshold;
    Uint16 		guSubImageThreshold;	//by david 20130910 from tagCfgs
    Uint16		uSobelThreshold;//sobel锟斤拷�??

    //added by david 20131014
    Uint16		bSwitched;

    Uint16		Resv2[1];
}CAMERA_STRUCT;

/* 锟斤拷锟斤拷锟结�?*/

#define DOWNLOAD_LANE_STRUCT_LENGTH 64
#define	NORMAL_LANE_STRUCT_LENGTH	128

struct	MeasurementFunc
{
    Uint16	ExistDetAmount;
    Uint16	ExistFuncName;
    Uint16	ExistFuncType;
    Uint16	ExistId[7];
};

struct	ExistStruct
{
    Uint16	FuncAmount;
    Uint16	Rsvd1;
    struct	MeasurementFunc	MearsureFuncCfg[3];
};

typedef struct	cNormalFunctionDef
{
    Uint16	StatisticFlowDef		:	1;
    Uint16	HugeVehicleFlowDef		:	1;
    Uint16	LargeVehicleFlowDef		:	1;
    Uint16	MiddleVehicleFlowDef	:	1;
    Uint16	SmallVehicleFlowDef		:	1;
    Uint16	SmartVehicleFlowDef		:	1;
    Uint16	AverageLengthDef		:	1;
    Uint16	AverageSpeedDef			:	1;
    Uint16	AverageHeadDistanceDef	:	1;
    Uint16	AverageOccupationDef	:	1;
    Uint16	AverageQuiqueDef		:	1;
    Uint16	AverageDealyDef			:	1;
    Uint16	Reserved4				:	4;
}NORMAL_FUNCTION_DEF;

typedef struct	cAnormalFunctionDef
{
    Uint16	SpeedHighDef		:	1;
    Uint16	SpeedLowDef			:	1;
    Uint16	LengthOverDef		:	1;
    Uint16	OccupyOverDef		:	1;
    Uint16	BackRunDef			:	1;
    Uint16	OverQuiqueDef		:	1;
    Uint16	OverDelayDef		:	1;
    Uint16	CrossLaneDef		:	1;
    Uint16	StopDef				:	1;
    Uint16	RedRunDef			:	1;
    Uint16	Reserved6			:	6;
}ANORMAL_FUNCTION_DEF;

typedef	struct	LanesStruct
{
    Uint16	LaneId;                                      /*0*/
    Uint16	CounterId;                                   /*1*/
    Uint16	SpeedId;                                     /*2*/
    Uint16	LeftRelateCountID;                           /*3*/
    Uint16	RightRelateCountID;                          /*4*/
    Uint16	BackRelateDetect;                            /*5*/
    union	{                                                    /*6*/
                Uint16	NormalFunctionWord;
                NORMAL_FUNCTION_DEF	NormalFunctionBit;
            }NormalFunction;
    union	{                 									/*7*/
                Uint16	AbnormalFunctionWord;
                ANORMAL_FUNCTION_DEF	AbnormalFunctionBit;
            }AbnormalFunction;
    Uint16	MaxOccupation;								/*8*/
    Uint16	MaxVehicleDensity;                          /*9*/
    Uint16	MaxExistTimeInterval;                       /*10*/
    Uint16	MaxVehicleSpeed;                            /*11*/
    Uint16	MinVehicleSpeed;                            /*12*/
    Uint16	uMaxVehicleInSeconds;                       /*13*/
    Uint16	MaxDelay;                                   /*14*/
    Uint16	SpeedConversionRatio;                       /*15*/
    Int16				SpeedModifyRatio;							/*16*/
    Uint16	LengthConversionRatio;                      /*17*/
    Int16				LengthModifyRatio;                          /*18*/
    Uint16	SwitchOutputDefine;							/*19*/
    Uint16	SwitchInputDefine;							/*20*/
/*****************************/
    Uint16    uCarTypeTransformFactorUpLimit;             /*21*/
    Uint16	uCarTypeTransformFactorDownLimit;           /*22*/
    Uint16    uSobelRowIsChangedThreshHold;				/*23*/
    Uint16    uSobelRowIsChangedRatio;					/*24*/
    Uint16    uLaneVehicleMaxReportLength;				/*25*/
    Uint16    uCounterWrapChangedThreshold;				/*26*DEFINE 0---NOT NEED 				GET UNCHANGED LINE STATICTIS,
                                                                        100----DEFINE B				ACK IMAGE IS NOT COMPLEX,USE SLK DIRECTLY*/
    Uint16    uVehicleMinLengthInPresence;				/*27*/
    Uint16    uTimeDifferenceRatioThreshHold;				/*28*/
    Uint16    nReserve[3];								/*29,30,31*/
/***************************/
    struct	ExistStruct	ExistCfg;                               /*32~63*/
    Uint16	StatisticStartFrameTime;                    /*64*/
    Uint16	AccumulatedFrameTime;                       /*65*/
    Uint16	SpeedGetTimes;								/*66*/
    Int16				CounterValue;                               /*67*/
    Int16				SpeedValue;                                 /*68*/      /*锟劫度伙拷准�? 锟斤拷锟斤拷�?锟斤拷统锟狡碉拷平锟斤拷叶锟�?*/
    Uint16	TotalVehicleFlow;                           /*69*/
    float			VehicleLength;                              /*70,71*/   /* 锟斤拷锟饺伙拷准�?*/
    float			SpeedSum;                                   /*72,73*/
    float			LengthSum;									/*74,75*/
    Uint16	TotalFrameCount;                            /*76*/
    Uint16	VehicleExistFrameCount;                     /*77*/
    Uint16	LengthGetTimes;								/*78*/
    Uint16	uVehiclePileLengthSum;                      /*79*/
    Uint16    uGetVehiclePileLengthCount;                 /*80*/
    Uint16    bIsVehicleInSpeed;							/*81*/
    Uint16    bArbitration;								/*82*/	/*TRUE:锟斤拷锟斤拷锟斤拷锟狡讹拷锟斤拷FALSE锟斤拷锟斤拷停止*/
    Uint16	uDetectorInSignalFrame;						/*83*/

/*********OVER LINE DRIVE DEFINE ********/
    Uint16	uStartRow;									/*84*/
    Uint16	uEndRow;									/*85*/
    Uint16	uFrameCount;								/*86*/
/******************/
    Uint16	bLaneEventFlag;								/*87*/
    Uint16 	uReserve[40];								/*88-127*/
}LANE_STRUCT_DEF;

/*DOWNLOAD_DETECT start*/

#define DOWNLOAD_DETECT_LENGTH	16

//#ifndef __ALG_CPOINT__
//#define __ALG_CPOINT__
//typedef struct
//{
//	Uint16 x;
//	Uint16 y;
//}CPoint;
//#endif
////20140102
#ifndef __ALG_CRECT__
#define __ALG_CRECT__



typedef struct RECT_Obj
{
//	struct RECT_Fxns *rectfxns;


    Int16    left;
    Int16    top;
    Int16    right;
    Int16    bottom;
} CRect;
/*
typedef struct RECT_FXNS
{
    void (*Init)(CRect handle);
    void (*Free)(CRect handle);
    int (*Width)(CRect handle);
    int (*Height)(CRect handle);
}RECT_Fxns;
*/
#endif

typedef struct CForeground
{
    CRect m_rtConnect;
}CFORE_GROUND;


#define FOREGROUND_ARRAY_CAPACITY	200
typedef struct CForeground_Array
{
    int sizeOfStruct;
    int sizeOfArray;
    int capacity;
    CFORE_GROUND array[FOREGROUND_ARRAY_CAPACITY];
} CForeground_Array;



typedef struct tagCOUNTADDLINE_RPTFRM
{
    Uint16		uAddCountLineProportion			:8;
    Uint16		uRepeatFrame					:8;
}COUNTADDLINE_RPTFRM;

typedef struct tagSPEED_DETECTOR_WORKMODE
{
    Uint16			uReserve			:13;
    bool			bDay				:1;
    bool			bNight				:1;
    bool			bAuto				:1;
}SPEED_DETECTOR_WORKMODE;

typedef struct tagDOWNLOAD_DETECT
{
    Uint16	uID;
    CPoint			ptFourCorner[8];
    Uint16		uDayNightJudgeMinContiuFrame;
    Uint16		uGetBackMinContiuFrame;
    Uint16		uVehicleHeadMinRow;
    Uint16		uVehicleHeadMinCol;
    Uint16		uSobelThreshold;
    COUNTADDLINE_RPTFRM	CountAddLineAndRptFrm;
    SPEED_DETECTOR_WORKMODE SpeedDetectorWorkMode;
}DOWNLOAD_DETECT;

/*DOWNLOAD_DETECT start*/

/*锟斤拷锟斤拷锟侥硷拷锟侥硷拷锟斤拷锟斤拷峁� start**/
/*锟斤拷锟斤拷锟斤�?*/


/*锟姐法玫锟斤拷慕锟�?/

/*struct	cSpeedStructIndex start */
typedef struct tagSPEED_COUNTER_STATUS						/*锟劫讹拷探锟斤拷锟斤拷募锟斤拷锟阶达�?*/
{
    Uint16			uReserve			:11;
    bool			bBeginCount			:1;					/*锟斤拷头�?锟斤拷末尾锟斤拷锟斤拷锟皆匡拷始锟斤拷锟斤拷*/
    bool			bCanBeginCount		:1;			/*锟斤拷锟街★拷压锟斤拷锟皆匡拷始锟斤拷锟姐车头锟角否到达拷撞锟�*/
    bool			bLeftDestroyed		:1;
    bool			bRightDestroyed		:1;
    bool			bIsCounting			:1;
    bool 			bbootomappercar		:1;		//txl,20151226
}SPEED_COUNTER_STATUS;

typedef struct tagVEHICLE_SPEEDINFO						/*探锟斤拷锟斤拷锟叫筹拷锟斤拷锟劫讹拷锟斤拷息*/
{
    Uint16	uStartFrame;				/*锟斤拷始锟斤拷锟斤拷时帧*/
    Uint16	uEndRow;					/*锟斤拷尾锟斤�?*/
    Uint16	uHeadRow;				/*锟斤拷始锟斤拷锟斤拷时锟斤拷?/
    Uint16 	uThreshold;				/*锟矫筹拷锟斤拷应锟斤拷锟斤拷�?*/
}VEHICLE_SPEEDINFO;

typedef struct tagVEHICLE_LENGTHINFO						/*探锟斤拷锟斤拷锟叫筹拷锟斤拷锟斤拷锟斤拷锟斤拷息*/
{
    float			fLengthSum;					/*锟斤拷锟姐到锟斤拷锟饺猴�?*/
    Uint16	uEndRow;					/*锟斤拷尾锟斤�?*/
    Uint16	uLengthCount;				/*锟斤拷锟姐到锟斤拷锟饺硷拷锟斤拷*/
    Uint16 	uThreshold;					/*锟矫筹拷锟斤拷应锟斤拷锟斤拷�?*/
    Uint16 	uReserve;
}VEHICLE_LENGTHINFO;

#define		SPEEDINFO_MAXSIZE	7
#define		LENGTHINFO_MAXSIZE	4

typedef struct cSpeedDetectStruct
{
    Uint16		DetectId;
    Uint16		DetectDots;
    Uint16	    QueDetectDots;
    Uint16		DetectLines;
    Uint16		DetectColumns;
    Uint16		QueDetectLines;
    Uint16		QueDetectColumns;
    Uint32		*CoordinatePointer;
    Uint32		*CoordinatePointerQ;
    Uint32		*ColightPointer;//0707xyx
    Uint16		*CurrentImagePointer;
    Uint16		*BackImagePointer;
    Uint16		*puTempImage;
    Uint16		*SobelBackImagePointer;
    Uint16		*BackImagePlusPointer;			//added by david 20130826 for IVD2.0
    Int16		wVehicleSpeed;
    Int16		wLastVehicleSpeed;				//added by david 20130929 for IVD2.0
    Int16		wVehicleLength;							/*11*/
    Int16		wLastVehicleLength;				//added by david 20130929 for IVD2.0

    //txl,0630
    int			position[30];
    int			preposition[30];


    float		percent;
    int Lastpos;
    int			sptimes;
    int			recount;
    Uint16		*FirstLaneImage;//0707xyx
    Uint16		*CurrQueueImage;
    Uint16		*PreQueueImage;
    Uint16		*SobelQueueImage;
    Uint16		*BackQueueImage;
    bool        IsCarInTail;//txl,1126

    Uint16		uSpeedBeginInRows;
    Uint16		uSpeedalsoInRows;//锟铰斤拷锟斤�?, by xyx
    Uint16		uSpeedBeginInFrame;
    Uint16		uSpeedEndEmptyRow;
    VEHICLE_SPEEDINFO		VehicleSpeedInfo[SPEEDINFO_MAXSIZE];
    float					fTempSpeed;
    Int16					nSpeedInfoTop;
    Int16					nSpeedInfoMaxSize;						/*35*/

    Uint16		uLengthBeginInRows;
    Uint16		uLengthBeginInFrame;
    Uint16		uLengthEndEmptyRow;
    Int16					nLengthInfoTop;
    VEHICLE_LENGTHINFO		VehicleLengthInfo[LENGTHINFO_MAXSIZE];
    float					fTempLength;							/*31*/
    Int16					nLengthInfoMaxSize;

    Uint16		uCountIntervalFrame;
    SPEED_COUNTER_STATUS	SpeedCounterStatus;
    Uint16		uTrafficFlow;
    Uint16		uCountFrameSum;
    Uint16		uBreakFrame;                            /*5*/

    Uint16		uSpeedNowThreshold;
    Uint16		uLengthNowThreshold;
    Uint16		uStatisThreshold;
    Uint16		uCountNowThreshold;

    Uint16		uStaticAverSpeedCount;
    Uint16		uStaticAverSpeed;
    Uint16		uStaticAverSpeedSum;
    Uint16		uSLKThreshold;
    Uint16		uProjectionStatisCount;
    Uint16		uProjectionStatic[32];					/*41*/
/******speedCounter info*****************************/
    bool		bGetComplexLineInfo;
    float		dSpeedCounterWrap;
    float		dSpeedCounterWrapSum;

    Uint16		uStatisFrameCount;
    bool		bSpeedCounterBackComplex;
    Uint16		uUnComplexLine;
    Uint16		uWrapStatisFrameCount;				/*9*/
/***********************************/
    bool		m_bInNight;
    bool		m_bVehicleBeginEnter;
    bool		m_bVehicleBeginOut;
    bool		m_bSpeedCounterBackValid;
    bool		m_bSpeedStarterBackValid;
    Uint16		m_uPassTimeCount;
    Uint16		m_uAvergreyMax;
    Uint16		m_uContiuCount;
    Uint16		m_uLightCount;
    Uint16		m_uSpeedCounterBackContiuFrame;
    Uint16		m_uSpeedStartBackContiuFrame;		/*11*/

    Uint16		uDayNightJudgeCicleCount;
///////////////////////锟缴碉拷锟斤拷�??
    Uint16		uVehicleHeadMinCol;//锟叫讹拷锟角筹拷头锟斤拷锟斤拷小锟斤拷锟斤拷
    Uint16		uSpeedCounterStartRow;//锟斤拷锟斤拷锟斤拷锟斤拷锟�
    Uint16		uRepeatFrame;//锟斤拷为锟斤拷一锟斤拷锟斤拷锟侥筹拷锟斤拷锟斤拷锟斤拷小�?
////////////////////////////
///////////////////////////////////////////////锟缴碉拷锟斤拷锟斤拷
    SPEED_DETECTOR_WORKMODE SpeedDetectorWorkMode;
////////////////////////
    bool		m_bAntcptMove;//预锟叫匡拷锟角凤拷锟狡讹�?
    Int16		m_nAntcptValidFrame;//预锟叫匡拷锟斤拷锟斤拷帧20
    bool		m_bArbtrnMove;//锟劫裁匡拷锟角凤拷锟狡讹�?
    Int16		m_nArbtrnFrame;//锟劫裁匡拷锟斤拷锟斤拷帧20
    bool		m_bSpdCnterHaveVehicle;//指示锟劫讹拷锟斤拷锟斤拷锟斤拷探锟斤拷锟斤拷锟角凤拷锟叫筹拷

    //ADDED 20130827 FOR IVD2.0
    Int16 		speedflag;
    Int16 		lengthflag;
    Int16 		outflag;
    Int16 		yellowflag;
    Int16 		calarflag;
    Int16 		position1;
    Int16 		position2;
    Int16 		Rowposition1;
    Int16 		thresholdValue;
    Uint16 		firstline;
    Uint16		secondline;
    Uint16		threeline;
    Int16 		jgtime;
    Int16 		headtime;
    Int16 		lxtime;
//	float min_standard_deviation;
//	float gmin_standard_deviation;
    bool has_update_min_std[10];
    bool begin_update_gmin_std[10];
    int num_cal_gmin[10];
    int cur_maxstd_pos;
    int last_maxstd_pos;
    bool getQueback_flag;//txl,20160104
    Uint16 		uTransFactor;
////20131218
    Uint16 		track_flag;    ////20131217
    Uint16    rectify_lane_direction[22][8][2];  ////20131217
    Uint16    nullflag;   ////20131217
    bool    bGetBackFlag;   ////20131217
    bool     fitflag;
    bool bGetsobelbackflag;
    bool    bGet1backflag;


    //
    Uint32 		uSpeedCounterChangedThreshold;
    Uint32 		uSpeedCounterChangedThreshold1;
    Uint32 		uSpeedCounterChangedThreshold2;
    Uint32		uGraySubThreshold;

    //added 20130912 for IVD2.0
    bool 		bSpeedCounterChanged;
    bool 		bSpeedCounterChanged1;
    bool 		bSpeedCounterChanged2;
    bool 		bSpeedCounterChanged3;
    bool 		bBottomNull;
    bool		no_car_flag;
    bool		complex_simple_background_flag;
    //txl.0625
    int Tailposition;
    int Headposition;
    int min_part_std[10];//txl,20150603
    int cur_part_std[10];//txl,20150603

    int min_tail_std;//txl,20150603
    int lmin_part_std[10];
    //float gmin_part_std[10];
    //float ini_min_std[10];
    int std_probab[100];
    //int not_up_back;//txl,20160110
    //bool Has_update_min[10];
    int lmin_tial_std;
    int num_cal_lmin;
    int stat_min_std[10][15];
    int stat_std_num[10];
    int aver_back;
    int stat_aver_back[10];
    int LastCarposition;//txl,201506
    int linenum;//txl,201506
    float       pastlength[4];//txl,0806
    bool is_bicycle;
    bool updateParams;
    Uint16		uReserve[37];
}SPEED_DETECT_STRUCT;		/* total struct occupy 128 words */

struct	cSpeedStructIndex
{
    Uint16	SpeedId;
    struct cSpeedDetectStruct	*ExtStorageAddr;
};
/*struct	cSpeedStructIndex end */


/*struct	cCountStructIndex start */
typedef struct	cCountStateFlags
{
    Uint16	Reserved4				:	4;
    Uint16	bBackImageActiveFlag	:	1;
    Uint16	bSubBackImageActiveFlag	:	1;
    Uint16	bSobelImageActiveFlag	:	1;
    Uint16	bFirstCaptureFlag		:	1;
    Uint16	bMSDEnableFlag			:	1;
    Uint16	bBackComplexFlag		:	1;
    Uint16	bStartedCounting		:	1;
    Uint16	bLeftDestroyedFlag		:	1;
    Uint16	bRightDestroyedFlag		:	1;
/*		Uint16	EnableCountFlag			:	1;   */
    Uint16	CounterStatus			:	3; /* value=0,1,2,4 */
}FLAGS;

typedef struct	cCountDetectStruct
{
    Uint16	DetectId;
    Uint16	DetectDots;
    Uint16	DetectLines;
    Uint16	DetectColumns;
    Uint32	*CoordinatePointer;
    Uint16	*CurrentImagePointer;
    Uint16	*BackImagePointer;
    Uint16	*SubBackImagePointer;
    Uint16	*SobelBackImagePointer;
    Uint16	bThisFrameTime;
    Uint16	bEnterFrameTime;
    Uint16	bSameFrameNumber;
    Uint16	bEnteredBackAnalyseFrameNum;
    Uint16	bVehicleCount;
    float			dNowMaxMeanGrey;
    float			dNowMinMeanGrey;
    float			dNowContrast;
    float			dNowModulation;
    float			wNowGreyMSD;
    float			wNowRelativeMSD;
    Uint16	wNowGreyAverage;
    Uint16	Reserved;
    float			dBackMaxMeanGrey;
    float			dBackMinMeanGrey;
    float			dBackContrast;
    float			dBackModulation;
    float			wBackGreyMSD;
    float			wBackRelatveMSD;
    Uint16	wBackGreyAverage;
    FLAGS			CountState;
    Uint16	Diagram[256];
}COUNT_DETECT_STRUCT;          /*total=256+15=271 words */

struct	cCountStructIndex
{
    Uint16	CountId;
    struct cCountDetectStruct	*ExtStorageAddr;
};
/*struct	cCountStructIndex end */

/*struct	cPresenceStructIndex start*/
typedef	struct cPresenceDetectStruct
{
    Uint16	DetectId;
    Uint16	DetectDots;
    Uint16	DetectLines;
    Uint16	DetectColumns;
    Uint32	*CoordinatePointer;
    Uint16	*CurrentImagePointer;
    Uint16	*BackImagePointer;
    Uint16	*SubBackImagePointer;
    Uint16	*SobelBackImagePointer;/*未锟斤拷*/		/*9*/

    Uint16	uBlocksNumber;/*锟斤拷�?�锟斤拷锟斤拷为0*/

    float           fVehiclePileLength;
    Uint16    uVehicleNonPresenceFrame;/*锟斤拷锟斤拷锟斤拷锟斤拷锟节碉拷锟斤拷锟斤拷帧锟斤拷,锟叫断非凤拷停锟斤拷时锟斤拷*/
    Uint16    uVehiclePresenceFrame;/*锟斤拷锟斤拷锟斤拷锟节碉拷锟斤拷锟斤拷帧锟斤�?,锟叫断非凤拷停锟斤拷时锟斤拷*/
    Uint16    uVehicleInMinContinueBlock;/*锟叫断筹拷锟接筹拷锟斤拷时锟斤拷, 锟接筹拷锟斤拷锟斤拷锟斤拷锟叫硷拷锟斤拷玫锟�?   6 */

    struct
    {
        Uint16	bHasBeenStatistic	:	1;	/*锟斤拷�?�锟斤拷锟斤拷为false*/
        Uint16 	bBackImageValid		:	1;	/*锟斤拷�?�锟斤拷锟斤拷为false*/
        Uint16	bHaveVehicle		:	1;	/*锟叫断筹拷锟斤拷锟斤拷锟斤拷,锟斤拷锟斤拷锟斤拷锟斤拷,锟斤拷�?�锟斤拷锟斤拷为false*/
        Uint16	bFirstCaptureFlag	:	1;	/*锟斤拷�?�锟斤拷锟斤拷为false*/

        Uint16    bVehicleIsStoping   :   1;  /*锟叫断非凤拷停锟斤拷时锟斤拷,锟斤拷�?�锟斤拷锟斤拷为false*/


        Uint16    bVehicleIsPresencing :  1;  /*锟叫断非凤拷停锟斤拷时锟斤拷,锟斤拷�?�锟斤拷锟斤拷为false*/

        Uint16	Reserved12			:	10;
    }cPresenceState;
//    BlockDescribe	BlockDef[MAX_BLOCKS_EXIST];	/*The total is 10*12=120*/
}PRESENCE_DETECT_STRUCT;		/*All words are 120+16=136*/

struct	cPresenceStructIndex
{
    Uint16	PresenceId;
    struct cPresenceDetectStruct	*ExtStorageAddr;
};

/*struct	cPresenceStructIndex end*/

/*PRESENCE_CALCURESULT start */
typedef struct tagPRESENCE_CALCURESULT{
    Uint16	nPresenceID;
    Uint16	nLaneCode;
    Uint16	uVehicleStopSeconds;
    Uint16	bVehicleIsStoping;
    Uint16	bEventHappenedFlag;	///Add by WJS in 2003-5-27
//	Uint16	bEventFlagHasClear;
    float			fVehiclePileLength;
}PRESENCE_CALCURESULT;
/*PRESENCE_CALCURESULT end */

typedef	struct	tagCameraParamaters
{
    struct	cSpeedStructIndex		dSpeedIdIndex[8];
    struct	cCountStructIndex		dCountIdIndex[8];
    struct	cPresenceStructIndex	dPresenceIdIndex[32];
    unsigned char			*LanesCfgEntry[8];
    Uint8			*DetectCfgEntry[32];
    unsigned int			DetectStateFlag[4];
    unsigned int			nPresenceDetectAmount;
    SPEED_COUNTER_STATUS 	guSpeedCounterStatus[8];
    PRESENCE_CALCURESULT	gPresenceCalcuResult[32];

    //extended by david 20130422:
    Uint8		 	*CameraCfgEntry;
    Uint8		 	*ImageStorageEntry;
    Uint16			frameWidth;
    Uint16			frameHeight;
    Uint32			bNormalDetectEnable;		//bit0:锟斤拷锟节硷拷锟斤拷锟绞癸拷锟�; bit1:锟斤拷锟接筹拷锟饺硷拷锟斤拷锟绞癸拷锟�; 锟斤拷锟斤拷锟斤�?
}CAMERA_PARAMETERS;

/*锟斤拷锟斤拷锟侥硷拷锟侥硷拷锟斤拷锟斤拷峁� start**/
union	NormalDetect
{
    struct	cSpeedDetectStruct	cSpeedDesc;
    struct	cCountDetectStruct	cCountDesc;
    struct  cPresenceDetectStruct cPresenceDesc;
};
/*锟斤拷锟斤拷锟侥硷拷锟侥硷拷锟斤拷锟斤拷峁筫nd**/

/*锟斤拷锟藉触锟斤拷锟侥结�? start*/

typedef	struct
{
    Uint16	bActiveFlag;
    Uint16	SwitchBit;
    Uint16	nInt0FrameTimes;
}BIT_FLAG_DEF;

typedef	struct tagBitActiveDesc
{
    BIT_FLAG_DEF	BitFlagDef[8];
}BIT_ACTIVE_DESC;

/*锟斤拷锟藉触锟斤拷锟侥结�? end*/

struct	cDetectStorageStruct
{
    Uint16	DetectId;
    Uint16	DetectDots;
    Uint16	DetectLines;
    Uint16	DetectColumns;
    Uint32	*CoordinatePointer;
    Uint16	*CurrentImagePointer;
    Uint16	*BackImagePointer;
    Uint16	Resered;
};
struct	DetAccessForm
{
    Uint16	DetectDots;
    Uint16	XYPage;
    Uint16	*XYAddrTable;
    Uint16	ImageStorePage;
    Uint16	*ImageStorePtr;
    Uint16	*Buffer;
};

typedef struct
{
    Uint16	Lines;
    Uint16	Columns;
    Uint8	*SourceAddr;
    Uint8	*SobelAddr;
    Uint16	Threshold;
}SOBELINPUT;
typedef struct tagADBOOST_DETECT
{
    CascadeClassifier face_cascade;
    Uint8 min_width;
    Uint8 min_height;
    Uint8 max_width;
    Uint8 max_height;
    float UpScale;
    float scaleFactor;
    int minNeighbor;
    Rect candidateRoi;//锟斤拷锟斤拷锟斤拷锟�?
}ADBOOST_DETECT;
typedef struct CTarget
{
    Rect vehicle;
    Rect vehicle_next;
    Point2f tracker_point[10];
    int tracker_point_num;
    bool tracked;
    bool detected;
    bool flow_count;
    int  num_detected;
    int vehicle_id;
    Point2f trajectory[3000];
    int trajectory_num;
    float vx;//锟劫讹拷
    float vy;
    int continue_num;
    int credit_num;//锟斤拷锟斤拷锟斤拷哦锟�
}CTarget;

typedef struct tagCfgs
{
    CAMERA_STRUCT 			CameraCfg;			//锟斤拷锟斤拷全锟斤拷锟斤拷锟斤拷
    CAMERA_PARAMETERS		CameraLocalPara;	//
    RESULTMSG				ResultMsg;			//
    ZENITH_SPEEDDETECTOR	DownSpeedCfg;
    ADBOOST_DETECT  AdboostDetect;
    Rect  detectROI;//�?测区�?
    CTarget candidate_targets[100];
    CTarget targets[100];
    int target_id;
    int targets_size;
    int candidate_targets_size;
    Uint16 detectPosition;
    Uint16	guBackGrey;
    Uint16 	guCurrentGrey;
    Uint16	gCurrentAlarmName;//锟斤拷前锟斤拷锟斤拷锟斤拷锟斤拷锟斤�?

    Uint16	m_iWidth, m_iHeight;
    Uint16	team_width;
    Uint16  team_height;
    float  percent;
    int Lastpos;
    bool    IsCarInTail[MAX_LANE];//txl,1126
    int sptimes[5];
    Uint16 	guMinCountIntervalFrame;


    Uint16 	guAdmitBreakFrame;
    bool    bNight;
    Uint16  bAuto;


    Uint16  thresholdValue;
    Uint16 	guTest;
    Uint16 	guTest1;
    Uint16 	guTest2;
    Uint16 	guTest3;
    Uint16 	guTestA[10];
    Uint16 	guTestB[10];
    Uint16 	guTestC[10];
    Uint16 	guTestD[10];

    Uint32 	gThisFrameTime;
    Uint32  abnormal_time;

    float 	gdAverGrey;
    float 	gdWrap;
    float	gdSum;
    Uint16 AverageGray_num;
    float AverageGray_sum;
    float former_tAverageGray;
    Uint16 Fchange_speed_tAverageGray_num;
    //added by david 20130422
    Uint32	uDetectInSum[MAX_LANE];		//8锟斤拷锟斤拷锟斤拷锟斤拷锟斤�?
    Uint32	uDetectOutSum[MAX_LANE];	//8锟斤拷锟斤拷锟斤拷锟斤拷锟斤�?
    Uint16 uVehicleQueueLength[MAX_LANE];
    Uint32  VehicleTeamSave[MAX_LANE];
    bool	complex_simple_background_flag;
    Uint16	firstx[8];
    Uint16	secondx[8];
    Uint16	padding[10];
    float mapping_matrix[9];
    //图锟斤拷实锟绞碉拷映锟斤�?,锟斤拷锟酵硷拷锟斤拷锟斤拷玫锟绞碉拷锟斤拷锟斤拷
    float image_actual[FULL_ROWS][FULL_COLS][2];
    float actual_point[CALIBRATION_POINT_NUM][2];
    float image_point[CALIBRATION_POINT_NUM][2];
    int calibration_point[4][2];//锟疥定锟斤拷锟斤拷锟�?
    int base_line[2][2];//锟疥定锟斤拷准锟竭碉�?
    float base_length;//锟斤拷准锟竭筹拷
    float near_point_length;//锟斤拷锟斤拷锟斤拷锟�?
    float actual_distance[8][FULL_ROWS];
    Uint16 uActualDetectLength[8];//锟斤拷锟斤拷锟斤拷锟斤拷锟津长讹�?
    Uint16 uActualTailLength[8];//占锟斤拷状�?�锟斤拷锟津长讹�?
    float actual_degree_length[FULL_ROWS];//锟教讹拷锟斤拷锟较筹拷锟斤拷
    Uint16 degreepoint[20][2];
    //0707
    //0609
    Uint16 calctimes;
    Uint16 carin;//0608
    Uint16 carout;//0608
    Uint16 carcount;//0608
    Uint16 existtimes;//4.1
    Uint16 exlight[6];
    Uint16 Nexlight;//514
    Uint16 light_width;
    Uint16 light_height;
    Uint16 fuzzydegree;//624xyx
    Uint16 fuzzytimes;//630xyx
    Uint16 avegray;
    bool fuzzyflag;
    Uint16 visitimes;
    bool visibility;//锟杰硷拷锟阶刺�
    int up_visib_value;//txl,20160105
    Uint16 visib_value[VISIB_LENGTH];//txl,20160105
    //0609
    //0707

}ALGCFGS;

typedef struct tagParams
{
    struct	LanesStruct		LaneParameter;
    union	NormalDetect	NormalDetectCfg;

    Uint8	*puPointNewImage;   ///锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷谋锟斤拷锟�?
    Uint8	*puPointSobelImage;
    Uint8 	*puSobelTempImageX;
    Uint8 	*puSobelTempImageY;

    Uint8	*puPointBackSobelImage;
////20131218
////	Uint8 	*puAlgTempImage;
    Uint8   *puPointNextImage;
    Uint8 	*puBackImagePlus;	//added by david 20130826 for IVD2.0

    Uint8	*PointNewImage;  ///锟铰硷拷锟斤拷锟斤拷锟斤拷谋锟斤拷锟�?
    Uint8	*PointBackImage;
    //Uint8	*PointTempImage;
    //Uint8	*PointBackSubImage;

    Uint8   *FirstLaneImage;//锟叫伙拷锟斤拷锟斤拷	0707xyx
    //txl,0630
    Uint8	*CurrQueueImage;  //锟脚队硷拷锟侥憋拷锟斤拷
    Uint8	*PreQueueImage;
    Uint8	*SobelQueueImage;
    Uint8	*BackQueueImage;
    Uint8	*PrevFullImage;//全图锟斤拷锟斤拷
    Uint8	*CurrFullImage;
    Uint8   *MaskFullImage;
    //added by david 20130903 for IVD2.0
    CPoint 	*m_pPointArray;

    CForeground_Array *ForegroundArray;
    CForeground_Array *ForegroundArray1;
    unsigned char* MidImage;////20140708

    Uint32	ArraySize;

}ALGPARAMS;

/*  //by david
extern CAMERA_STRUCT CameraCfg[3];
extern struct	LanesStruct		LaneParameter;
extern struct	LanesStruct     *LanePtr;

extern union	NormalDetect	NormalDetectCfg;
extern union	NormalDetect    *NormalDetectPtr;

extern CAMERA_PARAMETERS	*pCameraLocalPara[3];
extern CAMERA_PARAMETERS	CameraLocalPara[3];


extern UPLOADRESULTSBUF UploadResultsBuf,*pUploadResultsBuf;

*/

////20140102 array struct
typedef struct Array {
    int sizeOfStruct;
    int sizeOfArray;
    int capacity;
} Array;

typedef struct tagVIDENC1COPY_TI_OBJ{

    ALGCFGS		*pCfgs;
    ALGPARAMS	*pParams;

} VIDENC1COPY_TI_Obj;

extern  SPEEDCFGSEG    pDetectCfgSeg;
extern  ALGCFGS        pCfgs;
extern  ALGPARAMS      pParams;
extern  CFGINFOHEADER  pCfgHeader;
extern  LANE_STRUCT_DEF LaneStruct[8];
extern  unsigned char	Image[7][1600*1200];
extern  Uint16	ImageStorageEntry[8][8][1600*1200];
extern  unsigned int Coordinate1[8][2048*15];
extern  bool  flag;
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
static void ArrayInit(Array *nArray, int sizeOfStruct, int capacity);
static int ArrayGetCapaticy(Array *nArray);
static int ArrayGetSize(Array *nArray);
static int ArrayAdd(Array *nArray, void* elem);
static void ArrayRemoveAt( Array *nArray, int nIndex );
static void ArraySetSize(Array *nArray, int size);
static void ArrayRemoveAll(Array *nArray);

//static bool transform_init_DSP_VC(CPoint m_ptstart[],bool iniflag);
//static unsigned int transform_Proc_DSP_VC(unsigned char *pInFrameBuf, int nWidth, int nHeight, OUTBUF *outBuf, HANDLE hWrite);
//extern Uint16 ArithProc(Uint16 ChNum);
extern Uint16 ArithProc(Uint16 ChNum, IVIDEO1_BufDescIn * inBuf, RESULTMSG* outBuf, Int32 outSize, ALGCFGS *pCfgs, ALGPARAMS	*pParams);//,CPoint LineUp1[],CPoint m_ptend1[]
extern bool Color_deviate(Uint8 *uImage,Uint8 *vImage,int width,int height);////txl,20150714
extern bool ArithInit(Uint16 ChNum, CFGINFOHEADER *pCfgHeader, SPEEDCFGSEG *pDetectCfgSeg, ALGCFGS *pCfgs, ALGPARAMS *pParams);//,CPoint m_ptend[],CPoint LineUp[]
//extern bool ArithInit(Uint16 ChNum, CFGINFOHEADER *pCfgHeader, SPEEDCFGSEG *pDetectCfgSeg, ALGCFGS *pCfgs, ALGPARAMS *pParams);
//static bool ArithVariableInit(Uint16 ChNum);

extern bool ArithVariableInit(Uint16 ChNum, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void SetCfgUploadFlag(Uint16 CfgParseRight);
static void	InitSetAndSave(unsigned int nChannelNum, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void	InitSet(unsigned int nChannelNum, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void	InitThisLanesPara(struct LanesStruct *LanePtr);
static void	InitPresenceDetectPara(union NormalDetect *NormalDetectPtr);
static bool CfgStructParse(Uint16 ChNum, CFGINFOHEADER *pCfgHeader, SPEEDCFGSEG *pDetectCfgSeg, ALGCFGS *pCfgs, ALGPARAMS *pParams);//,CPoint m_ptend[],CPoint LineUp[]
//static bool CfgStructParse(Uint16 ChNum, CFGINFOHEADER *pCfgHeader, SPEEDCFGSEG *pDetectCfgSeg, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static CPoint ptGetDot(CPoint* ptUpLeft, CPoint* ptUpRight, CPoint* ptDownRight, CPoint* ptDownLeft, Int16 nColNum, Uint32 * ptStorePlace);
static float GetLenOfTwoPoint(CPoint* Cmp_Dot1, CPoint* Cmp_Dot2);


//void	CountSignalOutput(Uint16 nChannelNum);
//void	CountSignalClose(Uint16 nChannelNum);
//static void DigitalOutputVehicleSta(Uint16 CameraID,Uint16 LaneID,Uint16 VehicleSta);
//static void StoreCurImage(Uint16 ChNum);
static void StoreCurImage(Uint8 *inBuf, ALGCFGS *pCfgs, ALGPARAMS	*pParams);
//static void	SpeedCaculate(Uint16 LaneID,Uint16 nChannelNum);
static void SpeedCaculate(Uint16 LaneID, Uint16 nChannelNum, Uint8* inBuf, Int32 outSize, ALGCFGS *pCfgs, ALGPARAMS	*pParams,CPoint LineUp[],Uint8* inBuf1,CPoint m_ptend[]);//LaneID锟斤拷示锟斤拷锟斤拷锟脚ｏ拷nChannelNum锟斤拷示锟斤拷锟斤拷锟斤�?

static void PresenceCaculate(Uint16 nIndex,Uint16 nChannelNum);
static void SaveThisPresenceResult(Int16 nIndex, Uint16 nChannelNum);
static void ProjectSLKTemplateRltToEdge(Uint16 *pucSLKResult, Uint16 ucThreshold);
static bool bIsImageChanged(Uint8* pucBackImage, Uint8* pucNewImage);

//static void ShallowLightRemovePreProcess(Uint16* puNewImage,Uint16* puTempImage,Uint16 *puBackImage,Uint16 * puTempStoreY,struct cSpeedDetectStruct* pSpeedDetectStruct,Uint16 uChanlNum);
//static void ShallowLightRemovePreProcess(Uint8* puNewImage,Uint8* puTempImage,Uint8 *puBackImage,Uint8 * puTempStoreY,struct cSpeedDetectStruct* pSpeedDetectStruct,Uint16 uChanlNum);//by david
void ShallowLightRemovePreProcess(Uint8* puNewImage,Uint8* puTempImage,Uint8*puBackImage,Uint8* puTempStoreY,struct cSpeedDetectStruct* pSpeedDetectStruct,Uint16 uChanlNum,ALGCFGS *pCfgs, ALGPARAMS	*pParams);
static Uint16	GetImagePointValue(Int16 iCol,Int16 iRow,Uint16 Lines,Uint16 Columns,Uint8 *ImagePtr);
static SPEED_COUNTER_STATUS GetSpeedCounterStatus( Uint16 uCounter, Uint16 uThisLaneID, Uint16 nChannelNum ,ALGCFGS *pCfgs, ALGPARAMS *pParams);
static Int16 nLengthInfoGetSize(struct cSpeedDetectStruct* pSpeedDetectStruct);
static void LengthInfoRemoveAt( struct cSpeedDetectStruct* pSpeedDetectStruct, Int16 nIndex );
static void LengthInfoAdd(struct cSpeedDetectStruct* pSpeedDetectStruct, VEHICLE_LENGTHINFO* pVehicleLengthInfo);

static Uint16 uGetMaxUnEmptyRow(Uint8* puImage, Int16 nStartLine,Int16 nEndLine,ALGCFGS *pCfgs, ALGPARAMS *pParams);

static bool bIsSobleBlockChanged(Uint8 *pucSobelCurrentImage, Int16 nStartLine, Int16 nEndLine,ALGCFGS *pCfgs, ALGPARAMS *pParams);
static bool bIsSobleBlockUnChanged(Uint8 *pucSobelCurrentImage, int nStartLine, int nEndLine,int flag, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void SaveSpeedDetectorResult(Uint16 nChannelNum);
static Int16 nSpeedGetVehicleHeadRow( Uint8* puSobelImage, Uint16* pnStartCalcuRow, VEHICLE_SPEEDINFO* pVehicleSpeedInfo, int flag, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void LengthGetVehicleInInfo( Uint8* puSobelImage, struct cSpeedDetectStruct* pSpeedDetectStruct, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void SpeedInfoAdd(struct cSpeedDetectStruct* pSpeedDetectStruct, VEHICLE_SPEEDINFO* pVehicleSpeedInfo);
static void SpeedInfoRemoveAt( struct cSpeedDetectStruct* pSpeedDetectStruct, Int16 nIndex );
static void SpeedGetVehicleInInfo( Uint8* puSobelImage, struct cSpeedDetectStruct* pSpeedDetectStruct, int flag, ALGCFGS *pCfgs, ALGPARAMS *pParams);
bool bIsSpeedCounterChanged( Uint8* puSobelImage, Uint16 uSpeedCounterStartRow, SPEED_COUNTER_STATUS* pSpeedCounterStatus,int flag, ALGCFGS *pCfgs, ALGPARAMS *pParams );
static bool bVehicleHeadIn(Uint8* puSourceImage, Uint8* puTargetImage, Int16 nStartLine,Int16 nEndLine, Int16 nRowThreshold, Int16 nColThreshold, ALGCFGS *pCfgs, ALGPARAMS *pParams);

static void nSubstractImage(Uint8* puNewImage,Uint8* puTempImage,Uint8* puTargetImage, Int16 nStartLine,Int16 nEndLine, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void IsolatePointDelete(Uint8* puSourceImage, Uint16 uLines, Uint16 uColumns, Uint16 uThreshold);//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷�?�锟斤拷;
static void ImageCopy(Uint8* pucSource, Uint8* pucTarget, Int16 nStartLine, Int16 nEndLine,ALGCFGS *pCfgs, ALGPARAMS *pParams);
//static void iSubStractImage(Uint16 *puSourceImage,Uint16 *puTargetImage, Int16 nThreshold, Int16 nFromLine, Int16 nToLine);
static void iSubStractImage(Uint8 *puSourceImage,Uint8 *puTargetImage, Uint32 nThreshold, Int16 nFromLine, Int16 nToLine,Int16 width,Int16 height);

static float dAverageGreyInBlock(Uint8 *puImage, Int16 nFromBlockID, Int16 nToBlockID, ALGCFGS *pCfgs, ALGPARAMS *pParams);

static void ProjectSubAndSobelEdge(Uint8* puImage, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void SLKCalculate(Uint8* puSourceImage, Uint16 uThreshold, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void ProjectSLKToEdge(Uint8* puImage, Uint16 uLines, Uint16 uColumns, Uint16 uThreshold);
static void	SetImagePointValue(Int16 iCol,Int16 iRow,Uint16 Lines,Uint16 Columns,Uint16 Value,Uint8 *ImagePtr);
static void SquareOfTwoImage(Uint8* puSourceImageX, Uint8* puSourceImageY, Uint8* puTargetImage, Uint16 uLines, Uint16 uColumns, Uint16 uThreshold, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void SLKTemplateCalculate(Uint8* puSourceImage, Uint8* uTargetImage, Uint16 uLines, Uint16 uColumns, bool bVertical);


static bool bIsBlockAllEmpty(Uint8* puImage, Int16 nStartRow, Int16 nEndRow, Int16 nChangedPt, Uint16 uThreshold, ALGCFGS *pCfgs, ALGPARAMS *pParams);

static void SaveAllPresenceResult(Uint16	nChannelNum);
static void ConvergeFinalResult(Uint16 ChNum);

static Int16 nLengthGetVehicleHeadRow( Uint8* puSobelImage, Uint16* pnStartCalcuRow, VEHICLE_LENGTHINFO* pVehicleLengthInfo , ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void SpeedInfoRemoveAt( struct cSpeedDetectStruct* pSpeedDetectStruct, Int16 nIndex );
static Int16 nSpeedInfoGetSize(struct cSpeedDetectStruct* pSpeedDetectStruct);

////20131218 add

static bool bIsSpeedCounterChanged1( unsigned char* puSobelImage, Uint16 EndLine, SPEED_COUNTER_STATUS* pSpeedCounterStatus ,int flag, Uint16 lane_id, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static bool bIsSpeedCounterChanged2(unsigned char* puSobelImage1, unsigned char* puSobelImage, Uint16 uSpeedCounterStartRow, SPEED_COUNTER_STATUS* pSpeedCounterStatus, ALGCFGS *pCfgs, ALGPARAMS *pParams );

static bool bIsSpeedCounterChanged3(unsigned char* puSobelImage1, unsigned char* puSobelImage, unsigned char* GrayImage, unsigned short uSpeedCounterStartRow, unsigned char* BackImage, ALGCFGS *pCfgs, ALGPARAMS *pParams );

//added by david 20130828 for IVD2.0
//bool bIsSpeedCounterChanged1( unsigned char* puSobelImage, Uint16 EndLine, SPEED_COUNTER_STATUS* pSpeedCounterStatus ,int flag, ALGCFGS *pCfgs, ALGPARAMS *pParams);
//static bool bIsSpeedCounterChanged2( unsigned char* puSobelImage,unsigned short uSpeedCounterStartRow,SPEED_COUNTER_STATUS* pSpeedCounterStatus, ALGCFGS *pCfgs, ALGPARAMS *pParams );
//static bool bIsSpeedCounterChanged3( unsigned char* GrayImage, unsigned short uSpeedCounterStartRow, unsigned char* BackImage, ALGCFGS *pCfgs, ALGPARAMS *pParams );


static bool wuche(unsigned char* GrayImage,unsigned char* BackImage,unsigned int start,unsigned int end, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static bool bIsBottomNull( unsigned char* puSobelImage, Uint16 uSpeedCounterStartRow, SPEED_COUNTER_STATUS* pSpeedCounterStatus, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static bool BackGrayCharge(unsigned char* GrayImage,unsigned char* BackImage,unsigned int start,unsigned int end, ALGCFGS *pCfgs, ALGPARAMS *pParams);
//void NightLightDetect(unsigned char* puNewImage, unsigned char* puTempImage, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static void ostu(unsigned char* puNewImage,int width,int height, ALGCFGS *pCfgs, ALGPARAMS *pParams);
//bool DayorNight(unsigned char* puNewImage, ALGCFGS *pCfgs, ALGPARAMS *pParams);
static bool DayorNight(unsigned char* puNewImage,int nWidth,int nHight, ALGCFGS *pCfgs, ALGPARAMS *pParams,RESULTMSG* outBuf);
static void NightLightDetect(unsigned char* puNewImage, unsigned char* puTempImage,unsigned char* puaddImage,ALGCFGS *pCfgs, ALGPARAMS *pParams);
////20131218
static void SobelCalculate(unsigned char *puPointNewImage,unsigned char *puPointSobelImage,int threshold,int width,int height);
static void circle180(unsigned char *puPointNewImage,unsigned char *puPointSobelImage,int width,int height);
//static void SobelCalculate(SOBELINPUT* SobelInput);
//static void SobelCalculate(
//    const Uint8  *in,   /* Input image data   */
//    Uint8        *out,  /* Output image data  */
//    Int16 cols, Int16 rows              /* Image dimensions   */
//);
//916
float * mbys(float * im,int imW,int imH,float *tp,int tpW,int tpH);
void OnImgHarris(unsigned char* puNewImage,int width,int height);
//916
void  Gradient(unsigned char* puNewImage,int width,int height);
int StandardDeviation (unsigned char* puNewImage,int width,int height,int start_line,int stop_line,int flag);
//float AveGray(unsigned char* puNewImage,int nWidth,int nHight,ALGCFGS *pCfgs);
static bool estimate_background(unsigned char *puPointBackSobelImage,ALGCFGS *pCfgs);
static void sobel_rectify(unsigned char* puPointNextImage,unsigned char* puPointBackSobelImage,ALGCFGS *pCfgs);

//CIVD_SDK_API bool transform_init_DSP_VC(CPoint	ptactual[8],CPoint ptimage[8],CPoint LineUp[],CPoint m_ptend[],bool iniflag, ALGCFGS *pCfgs, ALGPARAMS *pParams,Uint16 lanecount,LANEINISTRUCT LaneIn);
//CIVD_SDK_API unsigned int transform_Proc_DSP_VC(unsigned char  *pInFrameBuf,unsigned char  *pInuBuf,unsigned char  *pInvBuf,int nWidth, int nHeight, OUTBUF *outBuf, int hWrite, ALGCFGS *pCfgs, ALGPARAMS *pParams);//,CPoint LineUp[],CPoint m_ptend[]

typedef struct args{
    ALGCFGS *pCfgs;
    ALGPARAMS *pParams;
    OUTBUF *p_outbuf;
    SPEEDCFGSEG    pDetectCfgSeg;
    CFGINFOHEADER  pCfgHeader;

    CPoint m_ptEnd[48];
    CPoint ptimage[8];

}m_args;

bool transform_init_DSP_VC(bool iniflag, Uint16 lanecount,
        LANEINISTRUCT LaneIn,RESULTMSG *p_outbuf,m_args *p_arg);
CIVD_SDK_API unsigned int transform_Proc_DSP_VC(unsigned char  *pInFrameBuf,
        unsigned char *pInuBuf,unsigned char *pInvBuf,\
        int nWidth, int nHeight, int hWrite,RESULTMSG *p_outbuf,m_args *p_arg);
int transform_arg_ctrl_DSP_VC(m_args *);
int transform_release_DSP_VC(m_args *);
extern unsigned int transform_Proc_0_DSP_VC(unsigned int* count,float* speed,float* length);
static int AveGray(unsigned char* puNewImage,int nWidth,int nHight,int start_line,int end_line,ALGCFGS *pCfgs);//702xyx
////20140703
//static float QueueLengthDetect(unsigned char* OriginImage,unsigned char* puSlkImage,unsigned char* puSobelImage, unsigned char* puSobelConnectImage,ALGCFGS *pCfgs, ALGPARAMS *pParams);
//static bool IsVehicleQueue(Uint8* puSubImage, Uint8* puSobelImage,Uint16 tail_line,Uint16 head_line, ALGCFGS *pCfgs, ALGPARAMS *pParams,Uint16 num);
static void SobelConectivity(unsigned char* puSobelImage,unsigned char* puSobelConnectImage,ALGCFGS *pCfgs, ALGPARAMS *pParams,int tailtop,int tailbottom);

static float TeamLength1(unsigned char* puNewImage,unsigned char* puaddImage,ALGCFGS *pCfgs,ALGPARAMS *pParams,int tailtop,int tailbottom);//20150918,txl
//1203
//锟斤拷锟斤拷锟疥�?
//映锟斤拷锟斤拷锟�?
//static void camera_calibration(float actual_point[][2],float img_point[][2],float mapping_matrix[],int calibration_num,ALGCFGS *pCfgs);
static void img_to_actual(float mapping_matrix[],int start_row,int end_row,int overlap_row1,int overlap_row2,int flag,ALGCFGS *pCfgs);
static void camera_calibration(int base_line[][2],float base_length,int calibration_point[][2],float near_point_length,int laneNum,ALGCFGS *pCfgs);
//图锟斤拷愕绞碉拷实锟斤拷锟斤�?
static void get_actual_point(float actual_point[2],int row,int col,int limit_line,ALGCFGS *pCfgs);
//锟斤拷锟斤拷实锟绞撅拷锟斤拷
static float distance_two(float actual_point1[2],float actual_point2[2]);
//1203
//0707
static bool Existlight(unsigned char* puNewImage,int nWidth,int nHight, ALGCFGS *pCfgs, ALGPARAMS *pParams);//0609
static void QueLengthSobel(Uint16 LaneID,unsigned char* OriginImage,unsigned char* SobelImage,ALGCFGS *pCfgs,ALGPARAMS *pParams);//txl,201506
//static void updateMinStd(unsigned char* OriginImage,int num,int width,int height,int startline,int stopline,ALGPARAMS *pParams,int cacPeriod);//txl,201506
static void updateMinStd(unsigned char* OriginImage,int num,int width,int height,int startline,int stopline,ALGCFGS *pCfgs,ALGPARAMS *pParams,int cacPeriod);
//static bool remove_bicycle(unsigned char* fgImg,unsigned char* diffImg, unsigned char* edgeImg,ALGCFGS *pCfgs);//20150619
static bool remove_bicycle(unsigned char* fImg,unsigned char* diffImg,unsigned char* edgeImg,int startLine,int endLine,ALGCFGS *pCfgs,bool bGetbackFlag);
static bool Sobelfilt(unsigned char* SobelImage,int connectnum,CRect rtConnect,ALGCFGS *pCfgs,ALGPARAMS *pParams,bool getcar);//txl,201507

static float fuzzy(unsigned char* puNewImage,int nWidth,int nHight);//624xyx
static bool Visibility(unsigned char* puNewImage,int nWidth,int nHight,ALGCFGS *pCfgs);//702xyx
static void im2bw(unsigned char* originImage,unsigned char* bwImage,int threshold,int width,int height);//702xyx
static float connect_filt( ALGCFGS *pCfgs, ALGPARAMS *pParams,CRect rtConnect);
static int connect_det(unsigned char* bwImage,unsigned char* MidImage,int lthreold,int rthreold,int width,int height,ALGPARAMS *pParams,CForeground_Array *ForegroundArray1,int flag);
static bool remove_nonvehicle(unsigned char* puNewImage,unsigned char* puBackImage,int position,ALGCFGS *pCfgs,ALGPARAMS *pParams,int flag);
static bool remove_nonvehicle1(unsigned char* puNewImage,unsigned char* puBackImage,int position,ALGCFGS *pCfgs,ALGPARAMS *pParams);
static int max_std_probab(ALGPARAMS *pParams,int num);
static bool remove_shadow(unsigned char* puNewImage,unsigned char* puBackImage,ALGCFGS *pCfgs,int threshold);
static bool visible_judge(Uint16 *a,int visib_length,int threshold);//txl,20160105
bool Initialize_target(CTarget& target);
int round(float f);
float median(vector<float> v);

vector<Rect> detectAndDisplay( Mat frame ,ADBOOST_DETECT pAdboostDetect);
//fg
void vehicle_detection(Mat currFrame,ADBOOST_DETECT AdboostDetect,vector<Rect>& vehicle_roi,vector<Rect> candidate_roi);
void merge_overlapped_rect(vector<Rect>& candidate_roi,int ratio_threshold);
//锟斤拷锟斤拷寻锟斤拷匹锟斤拷目锟斤拷
int overlapRatio(const Rect& r1,const Rect& r2);
//锟斤拷锟节合诧拷锟斤拷锟斤拷
int overlapRatio1(const Rect& r1,const Rect& r2);
void LKTracker(Mat currFrame,Mat prevFrame,CTarget* targets,int targets_size);
//锟揭碉拷锟斤拷锟斤拷锟斤拷锟�?
CTarget* find_nearest_rect(Rect vehicle_roi,CTarget* targets,CTarget* candidate_targets,int targets_size,int candidate_targets_size,int& overlap_ratio1,int& overlap_ratio2);
void get_candidate(const unsigned char* currFrame,unsigned char* prevFrame,unsigned char* tempFrame,int rows,int cols,vector<Rect>& candidate_roi,ALGCFGS *pCfgs,ALGPARAMS *pParams,int threshold);
void get_target(unsigned char* currImg,unsigned char* prevImg,unsigned char* maskImg,unsigned char* tempImg,Uint16 width,Uint16 height,ALGCFGS *pCfgs,ALGPARAMS *pParams,Uint16 flag);
//0707


//int alg_mem_malloc(RESULTMSG  **outBuf1,ALGCFGS **Cfgs, ALGPARAMS **Params);
int alg_mem_malloc(m_args *);
int alg_mem_free(m_args *arg_arg);
#endif



