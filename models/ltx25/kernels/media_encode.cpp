// Native SDR H264/AAC delivery. Latent and decoded-pixel parity are checked
// before this lossy delivery boundary; MP4 bytes are reported independently.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
namespace {
thread_local std::string error;
void check(int status,const char* operation){if(status<0){char message[AV_ERROR_MAX_STRING_SIZE];av_strerror(status,message,sizeof(message));throw std::runtime_error(std::string(operation)+": "+message);}}
struct Frame {AVFrame* p=av_frame_alloc();Frame(){if(!p)throw std::bad_alloc();}~Frame(){av_frame_free(&p);}};
struct Mux {
    AVFormatContext* format=nullptr;AVCodecContext *video=nullptr,*audio=nullptr;AVStream *vs=nullptr,*as=nullptr;SwsContext* scale=nullptr;
    explicit Mux(const std::string& path,int width,int height,int fps,int rate){
        check(avformat_alloc_output_context2(&format,nullptr,"mp4",path.c_str()),"create MP4");
        auto* vc=avcodec_find_encoder_by_name("libx264");auto* ac=avcodec_find_encoder(AV_CODEC_ID_AAC);
        if(!vc||!ac)throw std::runtime_error("H264/AAC delivery codec missing");
        vs=avformat_new_stream(format,vc);as=avformat_new_stream(format,ac);video=avcodec_alloc_context3(vc);audio=avcodec_alloc_context3(ac);
        if(!vs||!as||!video||!audio)throw std::bad_alloc();
        video->width=width;video->height=height;video->pix_fmt=AV_PIX_FMT_YUV420P;video->framerate={fps,1};video->time_base={1,fps};
        video->bit_rate=0;video->bit_rate_tolerance=128000;video->thread_count=0;video->thread_type=FF_THREAD_FRAME;
        video->colorspace=AVCOL_SPC_BT709;video->color_range=AVCOL_RANGE_MPEG;video->qmin=-1;video->qmax=-1;video->flags|=AV_CODEC_FLAG_CLOSED_GOP;
        audio->sample_rate=rate;audio->sample_fmt=AV_SAMPLE_FMT_FLTP;audio->time_base={1,rate};audio->bit_rate=0;av_channel_layout_default(&audio->ch_layout,2);
        if(format->oformat->flags&AVFMT_GLOBALHEADER){video->flags|=AV_CODEC_FLAG_GLOBAL_HEADER;audio->flags|=AV_CODEC_FLAG_GLOBAL_HEADER;}
        AVDictionary* options=nullptr;av_dict_set(&options,"crf","19",0);av_dict_set(&options,"preset","veryfast",0);
        int status=avcodec_open2(video,vc,&options);av_dict_free(&options);check(status,"open delivery H264");check(avcodec_open2(audio,ac,nullptr),"open delivery AAC");
        check(avcodec_parameters_from_context(vs->codecpar,video),"video stream parameters");check(avcodec_parameters_from_context(as->codecpar,audio),"audio stream parameters");
        vs->time_base=video->time_base;vs->avg_frame_rate=video->framerate;as->time_base=audio->time_base;
        check(avio_open(&format->pb,path.c_str(),AVIO_FLAG_WRITE),"open MP4 destination");check(avformat_write_header(format,nullptr),"write MP4 header");
        scale=sws_getContext(width,height,AV_PIX_FMT_RGB24,width,height,AV_PIX_FMT_YUV420P,SWS_BILINEAR,nullptr,nullptr,nullptr);
        if(!scale)throw std::runtime_error("delivery colorspace context");
        const int* coefficients=sws_getCoefficients(SWS_CS_ITU709);
        check(sws_setColorspaceDetails(scale,coefficients,1,coefficients,0,0,1<<16,1<<16),"delivery BT709 range");
    }
    ~Mux(){sws_freeContext(scale);avcodec_free_context(&video);avcodec_free_context(&audio);if(format){if(format->pb)avio_closep(&format->pb);avformat_free_context(format);}}
    void encode(AVCodecContext* codec,AVStream* stream,AVFrame* frame){
        check(avcodec_send_frame(codec,frame),"send delivery frame");AVPacket* packet=av_packet_alloc();if(!packet)throw std::bad_alloc();
        for(;;){int status=avcodec_receive_packet(codec,packet);if(status==AVERROR(EAGAIN)||status==AVERROR_EOF)break;
            if(status<0){av_packet_free(&packet);check(status,"delivery packet");}
            av_packet_rescale_ts(packet,codec->time_base,stream->time_base);packet->stream_index=stream->index;
            status=av_interleaved_write_frame(format,packet);av_packet_unref(packet);if(status<0){av_packet_free(&packet);check(status,"mux delivery packet");}}
        av_packet_free(&packet);
    }
    void frames(const float* source,size_t count,int width,int height){
        std::vector<uint8_t> rgb(size_t(width)*height*3);Frame yuv;yuv.p->width=width;yuv.p->height=height;yuv.p->format=AV_PIX_FMT_YUV420P;
        yuv.p->colorspace=AVCOL_SPC_BT709;yuv.p->color_range=AVCOL_RANGE_MPEG;check(av_frame_get_buffer(yuv.p,32),"delivery frame allocation");
        for(size_t index=0;index<count;index++){
            check(av_frame_make_writable(yuv.p),"writable delivery frame");
            for(size_t i=0;i<rgb.size();i++){
                float value=source[index*rgb.size()+i];if(!std::isfinite(value))throw std::runtime_error("nonfinite decoded RGB");
                rgb[i]=static_cast<uint8_t>(std::nearbyint(std::clamp(value,0.f,1.f)*255.f));
            }
            const uint8_t* data[4]={rgb.data(),nullptr,nullptr,nullptr};int lines[4]={width*3,0,0,0};
            check(sws_scale(scale,data,lines,0,height,yuv.p->data,yuv.p->linesize),"delivery RGB to YUV");
            yuv.p->pts=index;encode(video,vs,yuv.p);
        }
        encode(video,vs,nullptr);
    }
    void waveform(const float* source,size_t count,int rate){
        size_t offset=0;int block=audio->frame_size;if(block<=0)throw std::runtime_error("invalid AAC frame size");
        while(offset<count){Frame frame;int samples=std::min(size_t(block),count-offset);frame.p->nb_samples=samples;
            frame.p->format=AV_SAMPLE_FMT_FLTP;frame.p->sample_rate=rate;frame.p->pts=offset;av_channel_layout_default(&frame.p->ch_layout,2);
            check(av_frame_get_buffer(frame.p,0),"delivery audio allocation");
            for(int channel=0;channel<2;channel++)for(int i=0;i<samples;i++){
                float value=source[size_t(channel)*count+offset+i];if(!std::isfinite(value))throw std::runtime_error("nonfinite decoded audio");
                int16_t pcm=static_cast<int16_t>(std::clamp(value,-1.f,1.f)*32767.f);
                reinterpret_cast<float*>(frame.p->data[channel])[i]=float(pcm)*(1.f/32768.f);
            }
            encode(audio,as,frame.p);offset+=samples;
        }
        encode(audio,as,nullptr);
    }
    void finish(){check(av_write_trailer(format),"write MP4 trailer");check(avio_closep(&format->pb),"close delivery MP4");}
};
}
extern "C" const char* ltx25_encode_error(){return error.c_str();}
extern "C" int ltx25_encode_mp4(const char* path,const float* rgb,size_t frames,int width,int height,int fps,const float* audio,size_t samples,int rate){
    std::string temporary=std::string(path)+".partial.mp4";
    try{{Mux mux(temporary,width,height,fps,rate);mux.frames(rgb,frames,width,height);mux.waveform(audio,samples,rate);mux.finish();}
        if(std::rename(temporary.c_str(),path))throw std::runtime_error("publish MP4");return 0;
    }catch(const std::exception& e){error=e.what();std::remove(temporary.c_str());return -1;}
}
