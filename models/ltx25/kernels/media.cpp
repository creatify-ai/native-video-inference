// Native counterpart of the frozen Pillow RGB JPEG + PyAV single-frame CRF
// conditioning path. Links the reference wheels' codec libraries, not Python.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <setjmp.h>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <jpeglib.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
namespace {
thread_local std::string error;
void check(int code,const char* op){if(code<0){char detail[AV_ERROR_MAX_STRING_SIZE];av_strerror(code,detail,sizeof(detail));throw std::runtime_error(std::string(op)+": "+detail);}}
struct JpegError {jpeg_error_mgr manager;jmp_buf jump;char text[JMSG_LENGTH_MAX];};
void jpeg_fail(j_common_ptr jpeg){auto* e=reinterpret_cast<JpegError*>(jpeg->err);jpeg->err->format_message(jpeg,e->text);longjmp(e->jump,1);}
std::vector<uint8_t> jpeg_rgb(const uint8_t* raw,size_t length,int& width,int& height){
    jpeg_decompress_struct jpeg{};JpegError state{};jpeg.err=jpeg_std_error(&state.manager);state.manager.error_exit=jpeg_fail;
    if(setjmp(state.jump)){jpeg_destroy_decompress(&jpeg);throw std::runtime_error(state.text);}
    jpeg_create_decompress(&jpeg);jpeg_mem_src(&jpeg,raw,length);
    jpeg_save_markers(&jpeg,JPEG_APP0+1,0xffff);jpeg_save_markers(&jpeg,JPEG_APP0+2,0xffff);
    jpeg_read_header(&jpeg,TRUE);
    for(auto* marker=jpeg.marker_list;marker;marker=marker->next){
        if((marker->marker==JPEG_APP0+1 && marker->data_length>=6 && !std::memcmp(marker->data,"Exif\0\0",6))
            || (marker->marker==JPEG_APP0+2 && marker->data_length>=12 && !std::memcmp(marker->data,"ICC_PROFILE\0",12))){
            jpeg_destroy_decompress(&jpeg);throw std::runtime_error("EXIF/ICC image branch has not been admitted");
        }
    }
    if(jpeg.num_components!=3){jpeg_destroy_decompress(&jpeg);throw std::runtime_error("only RGB JPEG conditioning is admitted");}
    jpeg.out_color_space=JCS_RGB;jpeg_start_decompress(&jpeg);width=jpeg.output_width;height=jpeg.output_height;
    std::vector<uint8_t> rgb(size_t(width)*height*3);
    while(jpeg.output_scanline<jpeg.output_height){JSAMPROW row=rgb.data()+size_t(jpeg.output_scanline)*width*3;jpeg_read_scanlines(&jpeg,&row,1);}
    jpeg_finish_decompress(&jpeg);jpeg_destroy_decompress(&jpeg);return rgb;
}
struct Frame {
    AVFrame* p=av_frame_alloc();bool allocated=false;
    Frame(){if(!p)throw std::bad_alloc();}
    ~Frame(){if(allocated)av_freep(&p->data[0]);av_frame_free(&p);}
    void image(int width,int height,AVPixelFormat format){p->width=width;p->height=height;p->format=format;
        check(av_image_alloc(p->data,p->linesize,width,height,format,16),"allocate image");allocated=true;}
};
struct Codec {AVCodecContext* p;explicit Codec(const AVCodec* codec):p(avcodec_alloc_context3(codec)){if(!p)throw std::bad_alloc();}~Codec(){avcodec_free_context(&p);}};
struct Packet {AVPacket* p=av_packet_alloc();Packet(){if(!p)throw std::bad_alloc();}~Packet(){av_packet_free(&p);}};
struct Scale {SwsContext* p;Scale(int width,int height,AVPixelFormat from,AVPixelFormat to):
    p(sws_getContext(width,height,from,width,height,to,SWS_BILINEAR,nullptr,nullptr,nullptr)){if(!p)throw std::runtime_error("swscale context");}~Scale(){sws_freeContext(p);}};
std::vector<uint8_t> recompress(const std::vector<uint8_t>& source,int source_width,int source_height,int crf,int& width,int& height){
    if(crf==0 || source_width<2 || source_height<2){width=source_width;height=source_height;return source;}
    width=source_width/2*2;height=source_height/2*2;
    const AVCodec* encoder=avcodec_find_encoder_by_name("libx264");const AVCodec* decoder=avcodec_find_decoder(AV_CODEC_ID_H264);
    if(!encoder || !decoder)throw std::runtime_error("reference H264 codecs unavailable");
    Codec enc(encoder),dec(decoder);
    enc.p->width=width;enc.p->height=height;enc.p->pix_fmt=AV_PIX_FMT_YUV420P;
    enc.p->bit_rate=0;enc.p->bit_rate_tolerance=128000;enc.p->framerate={1,1};enc.p->time_base={1,1};
    enc.p->thread_count=0;enc.p->thread_type=FF_THREAD_SLICE;
    enc.p->qmin=-1;enc.p->qmax=-1;enc.p->flags|=AV_CODEC_FLAG_GLOBAL_HEADER|AV_CODEC_FLAG_CLOSED_GOP;
    AVDictionary* options=nullptr;av_dict_set(&options,"crf",std::to_string(crf).c_str(),0);av_dict_set(&options,"preset","veryfast",0);
    int status=avcodec_open2(enc.p,encoder,&options);av_dict_free(&options);check(status,"open libx264");
    AVCodecParameters* parameters=avcodec_parameters_alloc();if(!parameters)throw std::bad_alloc();
    status=avcodec_parameters_from_context(parameters,enc.p);
    if(status>=0)status=avcodec_parameters_to_context(dec.p,parameters);avcodec_parameters_free(&parameters);check(status,"H264 parameters");
    dec.p->thread_count=0;dec.p->thread_type=FF_THREAD_SLICE;check(avcodec_open2(dec.p,decoder,nullptr),"open H264 decoder");
    Frame rgb,yuv,decoded;rgb.image(width,height,AV_PIX_FMT_RGB24);yuv.image(width,height,AV_PIX_FMT_YUV420P);
    for(int row=0;row<height;row++)std::memcpy(rgb.p->data[0]+row*rgb.p->linesize[0],source.data()+size_t(row)*source_width*3,width*3);
    Scale down(width,height,AV_PIX_FMT_RGB24,AV_PIX_FMT_YUV420P);
    check(sws_scale(down.p,rgb.p->data,rgb.p->linesize,0,height,yuv.p->data,yuv.p->linesize),"RGB to YUV420");
    for(int plane=0;plane<3;plane++){int rows=plane?(height+1)/2:height,visible=plane?(width+1)/2:width;
        for(int row=0;row<rows;row++)std::memset(yuv.p->data[plane]+row*yuv.p->linesize[plane]+visible,0,yuv.p->linesize[plane]-visible);}
    yuv.p->pts=0;yuv.p->time_base={1,1};check(avcodec_send_frame(enc.p,yuv.p),"send conditioning frame");
    check(avcodec_send_frame(enc.p,nullptr),"flush conditioning encoder");
    Packet packet;bool found=false;
    while((status=avcodec_receive_packet(enc.p,packet.p))>=0){
        check(avcodec_send_packet(dec.p,packet.p),"decode conditioning packet");av_packet_unref(packet.p);
        int result=avcodec_receive_frame(dec.p,decoded.p);if(result>=0){found=true;break;}if(result!=AVERROR(EAGAIN))check(result,"receive conditioning frame");
    }
    if(!found){check(avcodec_send_packet(dec.p,nullptr),"flush conditioning decoder");check(avcodec_receive_frame(dec.p,decoded.p),"receive conditioning frame");}
    Frame result;result.image(width,height,AV_PIX_FMT_RGB24);
    Scale up(width,height,static_cast<AVPixelFormat>(decoded.p->format),AV_PIX_FMT_RGB24);
    check(sws_scale(up.p,decoded.p->data,decoded.p->linesize,0,height,result.p->data,result.p->linesize),"YUV420 to RGB");
    std::vector<uint8_t> output(size_t(width)*height*3);
    for(int row=0;row<height;row++)std::memcpy(output.data()+size_t(row)*width*3,result.p->data[0]+row*result.p->linesize[0],width*3);
    return output;
}
}
extern "C" const char* ltx25_media_error(){return error.c_str();}
extern "C" void ltx25_media_free(void* pointer){std::free(pointer);}
extern "C" int ltx25_media_decode(const uint8_t* source,size_t length,int crf,uint8_t** decoded,uint8_t** preprocessed,int* width,int* height){
    try{
        if(avcodec_version()!=LIBAVCODEC_VERSION_INT || swscale_version()!=LIBSWSCALE_VERSION_INT)throw std::runtime_error("media headers differ from the reference runtime");
        if(length<2 || source[0]!=0xff || source[1]!=0xd8)throw std::runtime_error("only admitted JPEG inputs are supported");
        int original_width,original_height;auto rgb=jpeg_rgb(source,length,original_width,original_height);
        auto output=recompress(rgb,original_width,original_height,crf,*width,*height);
        if(original_width!=*width || original_height!=*height)throw std::runtime_error("odd image dimensions have not been admitted");
        *decoded=static_cast<uint8_t*>(std::malloc(rgb.size()));*preprocessed=static_cast<uint8_t*>(std::malloc(output.size()));
        if(!*decoded || !*preprocessed)throw std::bad_alloc();
        std::memcpy(*decoded,rgb.data(),rgb.size());std::memcpy(*preprocessed,output.data(),output.size());return 0;
    }catch(const std::exception& e){error=e.what();return -1;}
}
