//
// Created by fmthhadmin on 21.05.19.
//




// Include files to use OpenCV API.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Include files to use the PYLON API.
#include <pylon/PylonIncludes.h>
#include <fstream>
// Use sstream to create image names including integer
#include <sstream>

//For cvtColor()
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>


#include <iostream>
#include <memory>
#include <cuda.h>
#include "NvEncoder/NvEncoderCuda.h"
#include "../Utils/Logger.h"
#include "../Utils/NvEncoderCLIOptions.h"
#include "../Utils/NvCodecUtils.h"



// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using GenApi objects
using namespace GenApi;

// Namespace for using opencv objects.
using namespace cv;

// Namespace for using cout.
using namespace std;

// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 1000;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int status = 0;   //1 means start encoding 2 means stop encoding

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>> & vec) {
    std::vector<T> result;
    for (const auto & v : vec)
        result.insert(result.end(), v.begin(), v.end());
    return result;
}

void ShowEncoderCapability()
{
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    std::cout << "Encoder Capability" << std::endl << std::endl;
    for (int iGpu = 0; iGpu < nGpu; iGpu++) {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
        NvEncoderCuda enc(cuContext, 1280, 720, NV_ENC_BUFFER_FORMAT_NV12);

        std::cout << "GPU " << iGpu << " - " << szDeviceName << std::endl << std::endl;
        std::cout << "\tH264:\t\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                                           NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no" ) << std::endl <<
                  "\tH264_444:\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                                           NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no" ) << std::endl <<
                  "\tH264_ME:\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                                           NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no" ) << std::endl <<
                  "\tH264_WxH:\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID,
                                           NV_ENC_CAPS_WIDTH_MAX) ) << "*" <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_HEIGHT_MAX) ) << std::endl <<
                  "\tHEVC:\t\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                           NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no" ) << std::endl <<
                  "\tHEVC_Main10:\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                           NV_ENC_CAPS_SUPPORT_10BIT_ENCODE) ? "yes" : "no" ) << std::endl <<
                  "\tHEVC_Lossless:\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                           NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE) ? "yes" : "no" ) << std::endl <<
                  "\tHEVC_SAO:\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                           NV_ENC_CAPS_SUPPORT_SAO) ? "yes" : "no" ) << std::endl <<
                  "\tHEVC_444:\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                           NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no" ) << std::endl <<
                  "\tHEVC_ME:\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                           NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no" ) << std::endl <<
                  "\tHEVC_WxH:\t" << "  " <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID,
                                           NV_ENC_CAPS_WIDTH_MAX) ) << "*" <<
                  ( enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_HEIGHT_MAX) ) << std::endl;

        std::cout << std::endl;

        enc.DestroyEncoder();
        ck(cuCtxDestroy(cuContext));
    }
}

void ShowHelpAndExit(const char *szBadOption = NULL)
{
    bool bThrowError = false;
    std::ostringstream oss;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "-i           Input file path" << std::endl
        << "-o           Output file path" << std::endl
        << "-s           Input resolution in this form: WxH" << std::endl
        << "-if          Input format: iyuv nv12 yuv444 p010 yuv444p16 bgra bgra10 ayuv abgr abgr10" << std::endl
        << "-gpu         Ordinal of GPU to use" << std::endl
            ;
    oss << NvEncoderInitParam().GetHelpMessage() << std::endl;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
    else
    {
        std::cout << oss.str();
        ShowEncoderCapability();
        exit(0);
    }
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, int &nWidth, int &nHeight,
                      NV_ENC_BUFFER_FORMAT &eFormat, char *szOutputFileName, NvEncoderInitParam &initParam, int &iGpu)
{
    std::ostringstream oss;
    int i;
    for (i = 1; i < argc; i++)
    {
        if (!_stricmp(argv[i], "-h"))
        {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-o"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-o");
            }
            sprintf(szOutputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-s"))
        {
            if (++i == argc || 2 != sscanf(argv[i], "%dx%d", &nWidth, &nHeight))
            {
                ShowHelpAndExit("-s");
            }
            continue;
        }
        std::vector<std::string> vszFileFormatName =
                {
                        "iyuv", "nv12", "yv12", "yuv444", "p010", "yuv444p16", "bgra", "bgra10", "ayuv", "abgr", "abgr10"
                };
        NV_ENC_BUFFER_FORMAT aFormat[] =
                {
                        NV_ENC_BUFFER_FORMAT_IYUV,
                        NV_ENC_BUFFER_FORMAT_NV12,
                        NV_ENC_BUFFER_FORMAT_YV12,
                        NV_ENC_BUFFER_FORMAT_YUV444,
                        NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
                        NV_ENC_BUFFER_FORMAT_YUV444_10BIT,
                        NV_ENC_BUFFER_FORMAT_ARGB,
                        NV_ENC_BUFFER_FORMAT_ARGB10,
                        NV_ENC_BUFFER_FORMAT_AYUV,
                        NV_ENC_BUFFER_FORMAT_ABGR,
                        NV_ENC_BUFFER_FORMAT_ABGR10,
                };
        if (!_stricmp(argv[i], "-if"))
        {
            if (++i == argc) {
                ShowHelpAndExit("-if");
            }
            auto it = std::find(vszFileFormatName.begin(), vszFileFormatName.end(), argv[i]);
            if (it == vszFileFormatName.end())
            {
                ShowHelpAndExit("-if");
            }
            eFormat = aFormat[it - vszFileFormatName.begin()];
            continue;
        }
        if (!_stricmp(argv[i], "-gpu"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        // Regard as encoder parameter
        if (argv[i][0] != '-')
        {
            ShowHelpAndExit(argv[i]);
        }
        oss << argv[i] << " ";
        while (i + 1 < argc && argv[i + 1][0] != '-')
        {
            oss << argv[++i] << " ";
        }
    }
    initParam = NvEncoderInitParam(oss.str().c_str());
}


int encodeVideo(char* cameraSerialNumber, int argc, char** argv) {

    // The exit code of the sample application.
    int exitCode = 0;


    /******************************************************
     *
     * Parsing command prompt input
     * Setting output file paths
     *
     * ***************************************************/

    /*******************************************************************************************************************************************************************/
    char szInFilePath[256] = "",
            szOutFilePath[256] = "";
    int nWidth = 0, nHeight = 0;
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    int iGpu = 0;
    NvEncoderInitParam encodeCLIOptions;

    //Parsing command line arguments
    ParseCommandLine(argc, argv, szInFilePath, nWidth, nHeight, eFormat, szOutFilePath, encodeCLIOptions, iGpu);
    CheckInputFile(szInFilePath);
    ValidateResolution(nWidth, nHeight);

    //Forming the output file path for camera using camera serial number
    char endFile[20];
    std::strcat(endFile, cameraSerialNumber);
    std::strcat(endFile,".264");
    std::strcat(szOutFilePath,endFile);


    if (!*szOutFilePath) {
        sprintf(szOutFilePath, encodeCLIOptions.IsCodecH264() ? "out.h264" : "out.hevc");
    }

    /*******************************************************************************************************************************************************************/


   /******************************************************
 *
 * Boiler plate program to initialise CUDA Encoder
 *
 * ***************************************************/


    /*******************************************************************************************************************************************************************/
    //Encoder initialization. Boiler plate code
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        return 1;
    }
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));
    NvEncoderCuda enc(cuContext, nWidth, nHeight, eFormat);
    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;
    enc.CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);
    enc.CreateEncoder(&initializeParams);
    int nFrameSize = enc.GetFrameSize();
    std::cout << "FRAME LENGTH";
    std::cout << nFrameSize << "\n";
    /*******************************************************************************************************************************************************************/



    /******************************************************
    *
    * Setting up the Basler Camera to capture frames
    *
    * ***************************************************/

    /*******************************************************************************************************************************************************************/
    // Automagically call PylonInitialize and PylonTerminate to ensure the pylon runtime system
    // is initialized during the lifetime of this object.
    Pylon::PylonAutoInitTerm autoInitTerm;

    try
    {
        // Create an instant camera object with the camera device found first.
        cout << "Creating Camera..." << endl;
        // CInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice());

        // or use a device info object to use a specific camera
        CDeviceInfo info;


        info.SetSerialNumber(cameraSerialNumber);
        CInstantCamera camera( CTlFactory::GetInstance().CreateFirstDevice(info));
        cout << "Camera Created." << endl;
        // Print the model name of the camera.
        cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;
        // The parameter MaxNumBuffer can be used to control the count of buffers
        // allocated for grabbing. The default value of this parameter is 10.
        camera.MaxNumBuffer = 10;

        // create pylon image format converter and pylon image
        CImageFormatConverter formatConverter;
        formatConverter.OutputPixelFormat= PixelType_BGR8packed;
        CPylonImage pylonImage;

        // Create an OpenCV image
        Mat openCvImage;
        Mat yuvImage;
        // Start the grabbing of c_countOfImagesToGrab images.
        // The camera device is parameterized with a default configuration which
        // sets up free-running continuous acquisition.
        camera.StartGrabbing();

        // Output for ca
        CGrabResultPtr ptrGrabResult;

        //Encoder output file
        std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary | std::ofstream::trunc);
        if (!fpOut)
        {

            std::ostringstream err;
            err << "Unable to open output file: " << szOutFilePath << std::endl;
            throw std::invalid_argument(err.str());
        }

        //Keeps count of the number of frames encoded so far
        int counter = 0;
        int nFrame = 0;

        // Camera.StopGrabbing() is called automatically by the RetrieveResult() method
        // when c_countOfImagesToGrab images have been retrieved.
        while ( camera.IsGrabbing())
        {


            // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            camera.RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_ThrowException);

            // Image grabbed successfully?
            if (ptrGrabResult->GrabSucceeded())
            {
                // Access the image data.
               // cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
                //cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;
                const uint8_t *pImageBuffer = (uint8_t *) ptrGrabResult->GetBuffer();
                //cout << "Gray value of first pixel: " << (uint32_t) pImageBuffer[0] << endl << endl;

                // Convert the grabbed buffer to pylon imag
                formatConverter.Convert(pylonImage, ptrGrabResult);
                // Create an OpenCV image out of pylon image
                openCvImage= cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t *) pylonImage.GetBuffer());
                //Converting the opencv image to YUV420 format. If you specify the input format in command prompt to something else
                //Don't forget to change the Constant in cvtColor to convert opencvImage to the desired format
                cvtColor(openCvImage, yuvImage, CV_BGR2YUV_I420);

                // Create a display window
                namedWindow( cameraSerialNumber, WINDOW_NORMAL);//AUTOSIZE //FREERATIO
                // Display the current image with opencv
                imshow(cameraSerialNumber, openCvImage);
                /******************************************************
                *
                * The captured frame stored in the yuvImage is passed to the encoder and it encodes it
                *
                ***************************************************/
                //starts encoding if user inputs 1 as input
                if(status != 0) {

                    try
                    {
                        // For receiving encoded packets
                        std::vector<std::vector<uint8_t>> vPacket;
                        if (status == 1)
                        {
                            const NvEncInputFrame* encoderInputFrame = enc.GetNextInputFrame();

                            NvEncoderCuda::CopyToDeviceFrame(cuContext, (uint8_t*)yuvImage.data, 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                                             (int)encoderInputFrame->pitch,
                                                             enc.GetEncodeWidth(),
                                                             enc.GetEncodeHeight(),
                                                             CU_MEMORYTYPE_HOST,
                                                             encoderInputFrame->bufferFormat,
                                                             encoderInputFrame->chromaOffsets,
                                                             encoderInputFrame->numChromaPlanes);
                            enc.EncodeFrame(vPacket);

                        }
                        else
                        {
                            cout << "Stopping encoding for: " << cameraSerialNumber;
                            enc.EndEncode(vPacket);
                            camera.StopGrabbing();
                        }

                        nFrame += (int)vPacket.size();
                        for (std::vector<uint8_t> &packet : vPacket)
                        {
                            // For each encoded packet

                            fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
                        }


                    }


                    catch (const std::exception &ex)
                    {

                        std::cout << ex.what();

                    }
                }



                waitKey(1);

            }
            else
            {
                cout << "Error: " << ptrGrabResult->GetErrorCode() << " " << ptrGrabResult->GetErrorDescription() << endl;
            }

            counter += 1;
        }

        std::cout << "Total frames encoded: " << nFrame << std::endl << "Saved in file " << szOutFilePath << std::endl;
        enc.DestroyEncoder();
        fpOut.close();


    }
    catch (GenICam::GenericException &e)
    {
        // Error handling.
        cerr << "An exception occurred." << endl
             << e.GetDescription() << endl;
        exitCode = 1;
    }

    // Comment the following two lines to disable waiting on exit.

    cerr << endl << "Press Enter to exit." << endl;
    while( cin.get() != '\n');

    return exitCode;

}

int msleep(unsigned long milisec)
{
    struct timespec req={0};
    time_t sec=(int)(milisec/1000);
    milisec=milisec-(sec*1000);
    req.tv_sec=sec;
    req.tv_nsec=milisec*1000000L;
    while(nanosleep(&req,&req)==-1)
        continue;
    return 1;
}

void getUserInput() {

    msleep(3000);
    cout << "\n" << "Press 1 to start encoding";
    std::cin >> status;
    cout << "Press 2 to stop encoding";
    std::cin >> status;

}


/******************************************************
    *
    * Main function
    * Separate threads are created for each camera
    * Manually edit the ids
    * You can find the ids for the Basler cameras in PylonViewerApp
    *
***************************************************/



int main(int argc, char** argv)
{
    char* camera1SerialNumber = "40024100";
    char* camera2SerialNumber = "40024099";
    char* camera3SerialNumber = "40024969";
    char* camera4SerialNumber = "40024971";
    char* camera5SerialNumber = "40024299";


    std::thread t1(encodeVideo,std::ref(camera1SerialNumber),std::ref(argc),std::ref(argv));
    std::thread t2(encodeVideo,std::ref(camera2SerialNumber),std::ref(argc),std::ref(argv));
    std::thread t3(encodeVideo,std::ref(camera3SerialNumber),std::ref(argc),std::ref(argv));
    std::thread t4(encodeVideo,std::ref(camera4SerialNumber),std::ref(argc),std::ref(argv));
    std::thread t5(encodeVideo,std::ref(camera5SerialNumber),std::ref(argc),std::ref(argv));

    std::thread t6(getUserInput);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    return 1;
}
