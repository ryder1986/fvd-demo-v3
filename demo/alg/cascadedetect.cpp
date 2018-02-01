//#include "stdafx.h"
#define  HAVE_TBB 
#include "cascadedetect.h"
//#include "opencv2/core/internal.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include "cascade_day_xml.h"
#include "cascade_night_xml.h"
#include "cascade_dusk_xml.h"
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

using namespace std;
namespace cv
{

// class for grouping object candidates, detected by Cascade Classifier, HOG etc.
// instance of the class is to be passed to cv::partition (see cxoperations.hpp)
class CV_EXPORTS SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
        return std::abs(r1.x - r2.x) <= delta &&
        std::abs(r1.y - r2.y) <= delta &&
        std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
        std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};


void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps, vector<int>* weights, vector<int>* levelWeights)
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        if( weights )
        {
            size_t i, sz = rectList.size();
            weights->resize(sz);
            for( i = 0; i < sz; i++ )
                (*weights)[i] = 1;
        }
        return;
    }
    vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));

    vector<Rect> rrects(nclasses);
    vector<int> rweights(nclasses, 0);
    vector<int> rejectLevels(nclasses, 0);
    vector<int> rejectWeights(nclasses, 0);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }
    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
    {
        for( i = 0; i < nlabels; i++ )
        {
            int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
        }
    }

    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }

    rectList.clear();
    if( weights )
        weights->clear();
    if( levelWeights )
        levelWeights->clear();

    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
		int n1 = rweights[i];
		//int n1 = levelWeights ? rejectLevels[i] : rweights[i];
        int w1 = rejectWeights[i];
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold)
                continue;
            Rect r2 = rrects[j];

            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
            if( weights )
                weights->push_back(n1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
    }
}

void groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, 0, 0);
}

void groupRectangles(vector<Rect>& rectList, vector<int>& weights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &weights, 0);
}
//used for cascade detection algorithm for ROC-curve calculating
void groupRectangles(vector<Rect>& rectList, vector<int>& rejectLevels, vector<int>& levelWeights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &rejectLevels, &levelWeights);
}

//----------------------------------------------  LBPEvaluator -------------------------------------
/*bool FeatureEvaluator::Feature :: read(const FileNode& node )
{
    FileNode rnode = node[CC_RECT];
    FileNodeIterator it = rnode.begin();
    it >> rect.x >> rect.y >> rect.width >> rect.height;
    return true;
}*/

FeatureEvaluator::FeatureEvaluator()
{
    features = new vector<Feature>();
}
FeatureEvaluator::~FeatureEvaluator()
{
}

/*bool FeatureEvaluator::read( const FileNode& node )
{
	char filename[100]="E:\\image\\cascade_xml.txt";
	FILE* fp;
	fp=fopen(filename,"a+");
	fprintf(fp,"const int NUM_FEATURES=%d;\n",node.size());
	fprintf(fp,"const	Rect features[NUM_FEATURES] = {\n");
	features->resize(node.size());
	featuresPtr = &(*features)[0];
	FileNodeIterator it = node.begin(), it_end = node.end();
	for(int i = 0; it != it_end; ++it, i++)
	{
		if(!featuresPtr[i].read(*it))
			return false;
		else
			fprintf(fp,"{%d,%d,%d,%d},\n",featuresPtr[i].rect.x,featuresPtr[i].rect.y,featuresPtr[i].rect.width,featuresPtr[i].rect.height);
	}
	fprintf(fp,"};\n");
	fclose(fp);
    return true;
}*/
bool FeatureEvaluator::read(int flag)
{
	if(flag==1||flag==3)
	{
		features->resize(cascade_dusk_xml::NUM_FEATURES);
		featuresPtr = &(*features)[0];
		for(int i = 0; i<cascade_dusk_xml::NUM_FEATURES;i++)
		{
			featuresPtr[i]=Feature(cascade_dusk_xml::features[i].x,cascade_dusk_xml::features[i].y,cascade_dusk_xml::features[i].width,cascade_dusk_xml::features[i].height);
		}
	}
	if(flag==2)
	{
		features->resize(cascade_night_xml::NUM_FEATURES);
		featuresPtr = &(*features)[0];
		for(int i = 0; i<cascade_night_xml::NUM_FEATURES;i++)
		{
			featuresPtr[i]=Feature(cascade_night_xml::features[i].x,cascade_night_xml::features[i].y,cascade_night_xml::features[i].width,cascade_night_xml::features[i].height);
		}
	}
	if(flag==4)
	{
		features->resize(cascade_day_xml::NUM_FEATURES);
		featuresPtr = &(*features)[0];
		for(int i = 0; i<cascade_day_xml::NUM_FEATURES;i++)
		{
			featuresPtr[i]=Feature(cascade_day_xml::features[i].x,cascade_day_xml::features[i].y,cascade_day_xml::features[i].width,cascade_day_xml::features[i].height);
		}
	}
	// load features 
	
	return true;

}
Ptr<FeatureEvaluator> FeatureEvaluator::clone() const
{
    FeatureEvaluator* ret = new FeatureEvaluator;
    ret->origWinSize = origWinSize;
    ret->features = features;
    ret->featuresPtr = &(*ret->features)[0];
    ret->sum0 = sum0, ret->sum = sum;
    ret->normrect = normrect;
    ret->offset = offset;
    return ret;
}

bool FeatureEvaluator::setImage( const Mat& image, Size _origWinSize )
{
    int rn = image.rows+1, cn = image.cols+1;
    origWinSize = _origWinSize;

    if( image.cols < origWinSize.width || image.rows < origWinSize.height )
        return false;

    if( sum0.rows < rn || sum0.cols < cn )
        sum0.create(rn, cn, CV_32S);
    sum = Mat(rn, cn, CV_32S, sum0.data);
    integral(image, sum);

    size_t fi, nfeatures = features->size();

    for( fi = 0; fi < nfeatures; fi++ )
        featuresPtr[fi].updatePtrs( sum );
    return true;
}

bool FeatureEvaluator::setWindow( Point pt )
{
    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= sum.cols ||
        pt.y + origWinSize.height >= sum.rows )
        return false;
    offset = pt.y * ((int)sum.step/sizeof(int)) + pt.x;
    return true;
}

Ptr<FeatureEvaluator> FeatureEvaluator::create( int featureType )
{
	return Ptr<FeatureEvaluator>(new FeatureEvaluator);
}
//---------------------------------------- Classifier Cascade --------------------------------------------

CascadeClassifier::CascadeClassifier()
{
}

/*CascadeClassifier::CascadeClassifier(const string& filename)
{
    load(filename);
}*/

CascadeClassifier::~CascadeClassifier()
{
}

bool CascadeClassifier::empty() const
{
    return  data.stages.empty();
}

/*bool CascadeClassifier::load(const string& filename)
{
    data = Data();
    featureEvaluator.release();

    FileStorage fs(filename, FileStorage::READ); 
    if( !fs.isOpened() )
        return false;

    if( read(fs.getFirstTopLevelNode()) )
        return true;

    fs.release();

	return true;
}*/
bool CascadeClassifier::load_file(int flag)
{
	data = Data();
	featureEvaluator.release();
	if( read(flag))
		return true;

	return true;
}
int CascadeClassifier::runAt( Ptr<FeatureEvaluator>& evaluator, Point pt, int& weight)
{


    if( !evaluator->setWindow(pt) )
        return -1;
    if( data.isStumpBased )
    {
        return predictCategoricalStump<FeatureEvaluator>( *this, evaluator, weight);     
    }
    else
    {
       return predictCategorical<FeatureEvaluator>( *this, evaluator, weight); 
    }
}
void CascadeClassifier::cascade_detection(CascadeClassifier *classifier,Size processingRectSize, int stripSize, int yStep, double scalingFactor,
	vector<Rect>& rectangles, vector<int>& rejectLevels, vector<int>& levelWeights, bool outputLevels, Mat mask)
{
	//Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();
	int num=0;
	Size winSize(cvRound(classifier->data.origWinSize.width * scalingFactor), cvRound(classifier->data.origWinSize.height * scalingFactor));
    unsigned char* p=mask.data;
	unsigned char* p1=p;
	for( int y = 0; y < processingRectSize.height; y += yStep )
	{
		num=0;
        p1=p+y*mask.cols;
		for( int x = 0; x < processingRectSize.width; x += yStep )
		{
			if ( (!mask.empty()) && (p1[x]==0)) {
				continue;
			}
			int gypWeight;
			int result = classifier->runAt(classifier->featureEvaluator/*evaluator*/, Point(x, y), gypWeight);
			if( outputLevels)
			{
				if( result == 1 )
					result =  -(int)classifier->data.stages.size();
				if( classifier->data.stages.size() + result <4)//最后几阶  4
				{
					rectangles.push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor), winSize.width, winSize.height));
					rejectLevels.push_back(-result);
					levelWeights.push_back(gypWeight);
				}
			}
			else if( result > 0 )
			{
				rectangles.push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor),winSize.width, winSize.height));
			}
			if( result == 0 )
			{
				x += yStep;
			}
			else
			{
				num++;
			}

		}
		if(num==0)
			y++;
	}
}
bool CascadeClassifier::setImage( Ptr<FeatureEvaluator>& evaluator, const Mat& image )
{
    return empty() ? false : evaluator->setImage(image, data.origWinSize);
}
class CascadeClassifierInvoker : public ParallelLoopBody
{
public:
	CascadeClassifierInvoker( CascadeClassifier& _cc, Size _sz1, int _stripSize, int _yStep, double _factor,
		vector<Rect>& _vec, vector<int>& _levels, vector<int>& _weights, bool outputLevels, const Mat& _image, const Mat& _mask, Mutex* _mtx)
	{
		classifier = &_cc;
		processingRectSize = _sz1;
		stripSize = _stripSize;
		yStep = _yStep;
		scalingFactor = _factor;
		rectangles = &_vec;
		rejectLevels = outputLevels ? &_levels : 0;
		levelWeights = outputLevels ? &_weights : 0;
		image = _image;
		mask = _mask;
		mtx = _mtx;
	}

	void operator()(const Range& range) const
	{
		//Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();
        int num=0;
		Size winSize(cvRound(classifier->data.origWinSize.width * scalingFactor), cvRound(classifier->data.origWinSize.height * scalingFactor));

		int y1 = range.start * stripSize;
		int y2 = min(range.end * stripSize, processingRectSize.height);
		unsigned char* p=image.data;
		unsigned char* p1=p;
		unsigned char* pMask=mask.data;
		unsigned char* p2=pMask;
		for( int y = y1; y < y2; y += yStep )
		{
            num=0;
			p1=p+y1*image.cols;
			p2=pMask+y1*mask.cols;
			for( int x = 0; x < processingRectSize.width; x += yStep )
			{
				if ( (!image.empty()) && (p1[x]==0)) {
					continue;
				}
				if(p2[x]==0)
				{
					continue;
				}
				int gypWeight;
				int result = classifier->runAt(classifier->featureEvaluator, Point(x, y), gypWeight);
				if( rejectLevels )
				{
					if( result == 1 )
						result =  -(int)classifier->data.stages.size();
					if( classifier->data.stages.size() + result <4)//最后几阶  4
					{
						mtx->lock();
						rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor), winSize.width, winSize.height));
						mtx->unlock();
						rejectLevels->push_back(-result);
						levelWeights->push_back(gypWeight);
					}
				}
				else if( result > 0 )
				{
					mtx->lock();
					rectangles->push_back(Rect(cvRound(x*scalingFactor), cvRound(y*scalingFactor),
						winSize.width, winSize.height));
					mtx->unlock();
				}
				if( result == 0 )
					x += yStep;
				else
				{
					num++;
				}

			}
			if(num==0)
				y++;
		}
	}

	CascadeClassifier* classifier;
	vector<Rect>* rectangles;
	Size processingRectSize;
	int stripSize, yStep;
	double scalingFactor;
	vector<int> *rejectLevels;
	vector<int> *levelWeights;
	Mat mask;
	Mat image;
	Mutex* mtx;

	//int stage;
};
bool CascadeClassifier::detectSingleScale( const Mat& image, const Mat& maskImg, int stripCount, Size processingRectSize,
                                           int stripSize, int yStep, double factor, vector<Rect>& candidates,
                                           vector<int>& levels, vector<int>& weights, bool outputRejectLevels)
{
    if( !featureEvaluator->setImage( image, data.origWinSize ) )
        return false;

    //Mat currentMask;
	//image.copyTo(currentMask);

    vector<Rect> candidatesVector;
    vector<int> rejectLevels;
    vector<int> levelWeights;
  //  if( outputRejectLevels )
  //  {
		//cascade_detection(this,processingRectSize, stripSize,yStep, factor,candidatesVector,rejectLevels,levelWeights,true,image/*currentMask*/);
  //      levels.insert( levels.end(), rejectLevels.begin(), rejectLevels.end() );
  //      weights.insert( weights.end(), levelWeights.begin(), levelWeights.end() );
  //  }
  //  else
  //  {
		//cascade_detection(this, processingRectSize, stripSize,yStep, factor,candidatesVector,rejectLevels,levelWeights,false,image/*currentMask*/);
  //  }
	Mutex mtx;
	if( outputRejectLevels )
	{
		parallel_for_(Range(0, stripCount), CascadeClassifierInvoker( *this, processingRectSize, stripSize, yStep, factor,
			candidatesVector, rejectLevels, levelWeights, true, image/*currentMask*/,maskImg, &mtx));
		levels.insert( levels.end(), rejectLevels.begin(), rejectLevels.end() );
		weights.insert( weights.end(), levelWeights.begin(), levelWeights.end() );
	}
	else
	{
		parallel_for_(Range(0, stripCount), CascadeClassifierInvoker( *this, processingRectSize, stripSize, yStep, factor,
			candidatesVector, rejectLevels, levelWeights, false, image/*currentMask*/, maskImg, &mtx));
	}
    candidates.insert( candidates.end(), candidatesVector.begin(), candidatesVector.end() );

    return true;
}

Size CascadeClassifier::getOriginalWindowSize() const
{
    return data.origWinSize;
}

bool CascadeClassifier::setImage(const Mat& image)
{
    return featureEvaluator->setImage(image, data.origWinSize);
}
void CascadeClassifier::detectMultiScale( const Mat& image, const Mat& maskImg, vector<Rect>& objects,
                                          vector<int>& rejectLevels,
                                          vector<int>& levelWeights,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize,
                                          bool outputRejectLevels )
{
    const double GROUP_EPS = 0.2;

    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if( empty() )
        return;
    objects.clear();

    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
        maxObjectSize = image.size();

    Mat grayImage = image;
    if( grayImage.channels() > 1 )
    {
        Mat temp;
        cvtColor(grayImage, temp, CV_BGR2GRAY);
        grayImage = temp;
    }
	
	float factor[10]={1.125,1.25,1.5,1.75,2.0,2.375,2.875,3.375,4.0,4.75};
	float factor_inv[10]={0.889,0.8,0.667,0.571,0.5,0.421,0.348,0.296,0.211};
	int winsize_width[10]={36,40,48,56,64,76,92,108,128,152};
	int winsize_height[10]={45,50,60,70,80,95,115,135,160,190};
    Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);
	Mat maskBuffer(maskImg.rows + 1, maskImg.cols + 1, CV_8U);
    vector<Rect> candidates;
    for( int i=0;i<10 ; i++)
    {
        Size originalWindowSize = getOriginalWindowSize();
        Size windowSize( winsize_width[i],winsize_height[i]);
        Size scaledImageSize( cvRound( grayImage.cols*factor_inv[i] ), cvRound( grayImage.rows*factor_inv[i]) );
        Size processingRectSize( scaledImageSize.width - originalWindowSize.width + 1, scaledImageSize.height - originalWindowSize.height + 1 );

        if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )
            break;
        if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height )
            break;
        if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
            continue;

        Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
        resize( grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );
		Mat scaleMask( scaledImageSize, CV_8U, maskBuffer.data);
		resize( maskImg, scaleMask, scaledImageSize, 0, 0, CV_INTER_LINEAR );
        int yStep;
		yStep = i > 6 ? 1 : 2;
        int stripCount, stripSize;

    #ifdef HAVE_TBB
        const int PTS_PER_THREAD = 1000;
        stripCount = ((processingRectSize.width/yStep)*(processingRectSize.height + yStep-1)/yStep + PTS_PER_THREAD/2)/PTS_PER_THREAD;
        stripCount = std::min(std::max(stripCount, 1), 100);
        stripSize = (((processingRectSize.height + stripCount - 1)/stripCount + yStep-1)/yStep)*yStep;
    #else
        stripCount = 1;
        stripSize = processingRectSize.height;
    #endif

        if( !detectSingleScale( scaledImage, scaleMask, stripCount, processingRectSize, stripSize,yStep, factor[i], candidates,
            rejectLevels, levelWeights, outputRejectLevels ) )
            break;
    }

    objects.resize(candidates.size());
    std::copy(candidates.begin(), candidates.end(), objects.begin());

   if( outputRejectLevels )
    {
        groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
    }
    else
    {
        groupRectangles( objects, rejectLevels,minNeighbors, GROUP_EPS );
    }
}
void CascadeClassifier::detectMultiScale( const Mat& image, const Mat& maskImg, vector<Rect>& objects,
                                          double scaleFactor, int minNeighbors,
                                          int flags, Size minObjectSize, Size maxObjectSize)
{
    vector<int> fakeLevels;
    vector<int> fakeWeights;
    detectMultiScale( image, maskImg, objects,fakeLevels, fakeWeights, scaleFactor,
        minNeighbors, flags, minObjectSize, maxObjectSize, false );
}
/*bool CascadeClassifier::Data::read(const FileNode &root)
{
	char filename[100]="E:\\image\\cascade_xml.txt";
	FILE* fp;
	fp=fopen(filename,"a+");
    static const float THRESHOLD_EPS = 1e-5f;
    // load stage params
    string stageTypeStr = (string)root[CC_STAGE_TYPE];
    if( stageTypeStr == CC_BOOST )
        stageType = BOOST;
    else
        return false;

    string featureTypeStr = (string)root[CC_FEATURE_TYPE];
    if( featureTypeStr == CC_LBP )
        featureType = 1;

    else
        return false;

    origWinSize.width = (int)root[CC_WIDTH];
    origWinSize.height = (int)root[CC_HEIGHT];
    CV_Assert( origWinSize.height > 0 && origWinSize.width > 0 );

    isStumpBased = (int)(root[CC_STAGE_PARAMS][CC_MAX_DEPTH]) == 1 ? true : false;

    // load feature params
    FileNode fn = root[CC_FEATURE_PARAMS];
    if( fn.empty() )
        return false;

    ncategories = fn[CC_MAX_CAT_COUNT];
    int subsetSize = (ncategories + 31)/32,
        nodeStep = 3 + ( ncategories>0 ? subsetSize : 1 );

    // load stages
    fn = root[CC_STAGES];
    if( fn.empty() )
        return false;
	fprintf(fp,"const bool isStumpBased = true;\n");
	fprintf(fp,"const int stageType =%d;\n",stageType);
	fprintf(fp,"const int featureType =%d;\n",featureType);
	fprintf(fp,"const Size origWinSize =Size(%d,%d);\n",origWinSize.width,origWinSize.height);
	fprintf(fp,"const int ncategories =%d;\n",ncategories);
    stages.reserve(fn.size());
    classifiers.clear();
    nodes.clear();

    FileNodeIterator it = fn.begin(), it_end = fn.end();

    for( int si = 0; it != it_end; si++, ++it )
    {
        FileNode fns = *it;
        Stage stage;
        stage.threshold = (int)(((float)fns[CC_STAGE_THRESHOLD] - THRESHOLD_EPS)*1000000);
        fns = fns[CC_WEAK_CLASSIFIERS];
        if(fns.empty())
            return false;
        stage.ntrees = (int)fns.size();
        stage.first = (int)classifiers.size();
        stages.push_back(stage);
        classifiers.reserve(stages[si].first + stages[si].ntrees);
        FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
        for( ; it1 != it1_end; ++it1 ) // weak trees
        {
            FileNode fnw = *it1;
            FileNode internalNodes = fnw[CC_INTERNAL_NODES];
            FileNode leafValues = fnw[CC_LEAF_VALUES];
            if( internalNodes.empty() || leafValues.empty() )
                return false;

            DTree tree;
            tree.nodeCount = (int)internalNodes.size()/nodeStep;
            classifiers.push_back(tree);

            nodes.reserve(nodes.size() + tree.nodeCount);
            leaves.reserve(leaves.size() + leafValues.size());
            if( subsetSize > 0 )
                subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);

            FileNodeIterator internalNodesIter = internalNodes.begin(), internalNodesEnd = internalNodes.end();

            for( ; internalNodesIter != internalNodesEnd; ) // nodes
            {
                DTreeNode node;
                node.left = (int)*internalNodesIter; ++internalNodesIter;
                node.right = (int)*internalNodesIter; ++internalNodesIter;
                node.featureIdx = (int)*internalNodesIter; ++internalNodesIter;
                if( subsetSize > 0 )
                {
                    for( int j = 0; j < subsetSize; j++, ++internalNodesIter )
                        subsets.push_back((int)*internalNodesIter);
                    //node.threshold = 0.f;
					node.threshold=0;
                }
                else
                {
                    //node.threshold = (float)*internalNodesIter; ++internalNodesIter;
					node.threshold = (int)*internalNodesIter; ++internalNodesIter;
                }
                nodes.push_back(node);
            }

            internalNodesIter = leafValues.begin(), internalNodesEnd = leafValues.end();

            for( ; internalNodesIter != internalNodesEnd; ++internalNodesIter ) // leaves
                leaves.push_back((int)(((float)*internalNodesIter)*1000000));
        }
    }
	fprintf(fp,"const int NUM_STAGES = %d;\n",stages.size());
	fprintf(fp,"const CascadeClassifier::Data::Stage stages[NUM_STAGES] = {\n");
	for(int i=0;i<stages.size();i++)
	{
		fprintf(fp,"{%d,%d,%d},\n",stages[i].first,stages[i].ntrees,stages[i].threshold);
	}
	fprintf(fp,"};\n");
	int classifiers_size=classifiers.size();
	fprintf(fp,"const int NUM_CLASSIFIERS = %d;\n",classifiers_size);
	fprintf(fp,"const CascadeClassifier::Data::DTree classifiers[NUM_CLASSIFIERS] = {\n");
	for(int i=0;i<classifiers.size();i++)
	{
		fprintf(fp,"{%d},\n",classifiers[i].nodeCount);
	}
	fprintf(fp,"};\n");
	int nodes_size=nodes.size();
	fprintf(fp,"const int NUM_NODES = %d;\n",nodes_size);
	fprintf(fp,"const CascadeClassifier::Data::DTreeNode nodes[NUM_NODES] = {\n");
	for(int i=0;i<nodes.size();i++)
	{
		fprintf(fp,"{%d,%d,%d,%d},\n",nodes[i].featureIdx,nodes[i].threshold,nodes[i].left,nodes[i].right);
	}
	fprintf(fp,"};\n");
	int leaves_size=leaves.size();
	fprintf(fp,"const int NUM_LEAVES = %d;\n",leaves_size);
	fprintf(fp,"const int leaves[NUM_LEAVES] = {");
	for(int i=0;i<leaves_size-1;i++)
	{
		fprintf(fp,"%d,",leaves[i]);
	}
	fprintf(fp,"%d};\n",leaves[leaves_size-1]);
	int subsets_size=subsets.size();
	fprintf(fp,"const int NUM_SUBSETS = %d;\n",subsets_size);
	fprintf(fp,"const int subsets[NUM_SUBSETS] = {");
	for(int i=0;i<subsets_size-1;i++)
	{
		fprintf(fp,"%d,",subsets[i]);
	}
	fprintf(fp,"%d};\n",subsets[subsets_size-1]);
	fclose(fp);
    return true;
}
bool CascadeClassifier::read(const FileNode& root)
{
	if( !data.read(root) )
		return false;

	// load features
	featureEvaluator = FeatureEvaluator::create(data.featureType);
	FileNode fn = root[CC_FEATURES];
	if( fn.empty() )
		return false;

	return featureEvaluator->read(fn);
}*/
bool CascadeClassifier::read(int flag)
{
	if( !data.read(flag) )
		return false;

	// load features
	featureEvaluator = FeatureEvaluator::create(data.featureType);
	return featureEvaluator->read(flag);
}
bool CascadeClassifier::Data::read(int flag)
{
	if(flag==1||flag==3)
	{
		// load stage params
		stageType = cascade_dusk_xml::stageType;
		featureType =cascade_dusk_xml::featureType;
		origWinSize.width = cascade_dusk_xml::origWinSize.width;
		origWinSize.height = cascade_dusk_xml::origWinSize.height;
		CV_Assert( origWinSize.height > 0 && origWinSize.width > 0 );
		isStumpBased =cascade_dusk_xml::isStumpBased;

		// load feature params
		ncategories = cascade_dusk_xml::ncategories;
		int subsetSize = cascade_dusk_xml::NUM_SUBSETS;
		// load stages
		stages.reserve(cascade_dusk_xml::NUM_STAGES);
		classifiers.clear();
		nodes.clear();
		for(int i=0;i<cascade_dusk_xml::NUM_STAGES;i++)
		{
			stages.push_back(cascade_dusk_xml::stages[i]);
		}
		for(int i=0;i<cascade_dusk_xml::NUM_CLASSIFIERS;i++)
		{
			classifiers.push_back(cascade_dusk_xml::classifiers[i]);
		}
		for(int i=0;i<cascade_dusk_xml::NUM_NODES;i++)
		{
			nodes.push_back(cascade_dusk_xml::nodes[i]);
		}
		for(int i=0;i<cascade_dusk_xml::NUM_LEAVES;i++)
		{
			leaves.push_back(cascade_dusk_xml::leaves[i]);
		}
		for(int i=0;i<cascade_dusk_xml::NUM_SUBSETS;i++)
		{
			subsets.push_back(cascade_dusk_xml::subsets[i]);
		}
	}
	if(flag==2)
	{
		// load stage params
		stageType = cascade_night_xml::stageType;
		featureType =cascade_night_xml::featureType;
		origWinSize.width = cascade_night_xml::origWinSize.width;
		origWinSize.height = cascade_night_xml::origWinSize.height;
		CV_Assert( origWinSize.height > 0 && origWinSize.width > 0 );
		isStumpBased =cascade_night_xml::isStumpBased;

		// load feature params
		ncategories = cascade_night_xml::ncategories;
		int subsetSize = cascade_night_xml::NUM_SUBSETS;
		// load stages
		stages.reserve(cascade_night_xml::NUM_STAGES);
		classifiers.clear();
		nodes.clear();
		for(int i=0;i<cascade_night_xml::NUM_STAGES;i++)
		{
			stages.push_back(cascade_night_xml::stages[i]);
		}
		for(int i=0;i<cascade_night_xml::NUM_CLASSIFIERS;i++)
		{
			classifiers.push_back(cascade_night_xml::classifiers[i]);
		}
		for(int i=0;i<cascade_night_xml::NUM_NODES;i++)
		{
			nodes.push_back(cascade_night_xml::nodes[i]);
		}
		for(int i=0;i<cascade_night_xml::NUM_LEAVES;i++)
		{
			leaves.push_back(cascade_night_xml::leaves[i]);
		}
		for(int i=0;i<cascade_night_xml::NUM_SUBSETS;i++)
		{
			subsets.push_back(cascade_night_xml::subsets[i]);
		}
	}
	if(flag==4)
	{
		// load stage params
		stageType = cascade_day_xml::stageType;
		featureType =cascade_day_xml::featureType;
		origWinSize.width = cascade_day_xml::origWinSize.width;
		origWinSize.height = cascade_day_xml::origWinSize.height;
		CV_Assert( origWinSize.height > 0 && origWinSize.width > 0 );
		isStumpBased =cascade_day_xml::isStumpBased;

		// load feature params
		ncategories = cascade_day_xml::ncategories;
		int subsetSize = cascade_day_xml::NUM_SUBSETS;
		// load stages
		stages.reserve(cascade_day_xml::NUM_STAGES);
		classifiers.clear();
		nodes.clear();
		for(int i=0;i<cascade_day_xml::NUM_STAGES;i++)
		{
			stages.push_back(cascade_day_xml::stages[i]);
		}
		for(int i=0;i<cascade_day_xml::NUM_CLASSIFIERS;i++)
		{
			classifiers.push_back(cascade_day_xml::classifiers[i]);
		}
		for(int i=0;i<cascade_day_xml::NUM_NODES;i++)
		{
			nodes.push_back(cascade_day_xml::nodes[i]);
		}
		for(int i=0;i<cascade_day_xml::NUM_LEAVES;i++)
		{
			leaves.push_back(cascade_day_xml::leaves[i]);
		}
		for(int i=0;i<cascade_day_xml::NUM_SUBSETS;i++)
		{
			subsets.push_back(cascade_day_xml::subsets[i]);
		}
	}
	return true;
}
//----------------------------------------------  predictor functions -------------------------------------


template<class FEval>
inline int predictCategorical( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, int& sum)
{
	int nstages = (int)cascade.data.stages.size();
	int nodeOfs = 0, leafOfs = 0;
	FEval& featureEvaluator = (FEval&)*_featureEvaluator;
	size_t subsetSize = (cascade.data.ncategories + 31)/32;
	int* cascadeSubsets = &cascade.data.subsets[0];
	int* cascadeLeaves = &cascade.data.leaves[0];
	CascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
	CascadeClassifier::Data::DTree* cascadeWeaks = &cascade.data.classifiers[0];
	CascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];

	for(int si = 0; si < nstages; si++ )
	{
		CascadeClassifier::Data::Stage& stage = cascadeStages[si];
		int wi, ntrees = stage.ntrees;
		sum = 0;

		for( wi = 0; wi < ntrees; wi++ )
		{
			CascadeClassifier::Data::DTree& weak = cascadeWeaks[stage.first + wi];
			int idx = 0, root = nodeOfs;
			do
			{
				CascadeClassifier::Data::DTreeNode& node = cascadeNodes[root + idx];
				int c = featureEvaluator(node.featureIdx);
				const int* subset = &cascadeSubsets[(root + idx)*subsetSize];
				idx = (subset[c>>5] & (1 << (c & 31))) ? node.left : node.right;
			}
			while( idx > 0 );
			sum += cascadeLeaves[leafOfs - idx];
			nodeOfs += weak.nodeCount;
			leafOfs += weak.nodeCount + 1;
		}
		if( sum < stage.threshold )
			return -si;
	}
	return 1;
}

template<class FEval>
inline int predictCategoricalStump( CascadeClassifier& cascade, Ptr<FeatureEvaluator> &_featureEvaluator, int& sum )
{
	int nstages = (int)cascade.data.stages.size();
	int nodeOfs = 0, leafOfs = 0;
	FEval& featureEvaluator = (FEval&)*_featureEvaluator;
	size_t subsetSize = (cascade.data.ncategories + 31)/32;
	int* cascadeSubsets = &cascade.data.subsets[0];
	int* cascadeLeaves = &cascade.data.leaves[0];
	CascadeClassifier::Data::DTreeNode* cascadeNodes = &cascade.data.nodes[0];
	CascadeClassifier::Data::Stage* cascadeStages = &cascade.data.stages[0];
	for( int si = 0; si < nstages; si++ )
	{
		CascadeClassifier::Data::Stage& stage = cascadeStages[si];
		int wi, ntrees = stage.ntrees;
		sum = 0;

		for( wi = 0; wi < ntrees; wi++ )
		{
			CascadeClassifier::Data::DTreeNode& node = cascadeNodes[nodeOfs];
			int c = featureEvaluator(node.featureIdx);
			const int* subset = &cascadeSubsets[nodeOfs*subsetSize];
			sum += cascadeLeaves[ subset[c>>5] & (1 << (c & 31)) ? leafOfs : leafOfs+1];
			nodeOfs++;
			leafOfs += 2;
		}
		if( sum < stage.threshold )
			return -si;
	}

	return 1;
}

} // namespace cv
