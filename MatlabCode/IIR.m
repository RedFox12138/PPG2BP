function [ output ] = IIR( data,rate,frequency )
%陷波器
LL=length(data);%数据长度
%一次性执行，可以初始化完成 第一次50 Hz
 fh=frequency;fs=rate;
 wh = fh * pi / fs;
 Q = tan(wh);%角频率
 p=5;%品质因数
 A=1;
 m = 1.0 + Q/p + Q*Q;
 a0=(1+Q*Q)*A/m;
 a1=2.0*(Q*Q-1.0)*A/m;
 a2=(Q*Q+1.0)*A/m;
 b1 = 2.0*(Q*Q-1.0) / m;
 b2 = (1.0 - Q/p + Q*Q) / m;
 y1=data(1);
 y2=data(2);
 x1=a0*y1;
 x2=a0*y2+a1*y1-b1*x1;
 I2(2)=x2;
 I2(1)=x1;
    for i=3:LL
        y3=data(i);
        y2=data(i-1);
        y1=data(i-2);
        x3=a0*y3+a1*y2+a2*y1-b1*x2-b2*x1;
        I2(i)=x3;
        x1=x2;
        x2=x3;
    end
   data=I2;
% %%%%%%%%%%%%%%%%%%%%%IIR陷波 实时执行%%%%%%%%%%%%%%%%%%%%%%%%
%一次性执行，可以初始化完成 第二次50 Hz
 fh=frequency;fs=rate;
 wh = fh * pi / fs;
 Q = tan(wh);%角频率
 p=5;%品质因数
 A=1;
 m = 1.0 + Q/p + Q*Q;
 a0=(1+Q*Q)*A/m;
 a1=2.0*(Q*Q-1.0)*A/m;
 a2=(Q*Q+1.0)*A/m;
 b1 = 2.0*(Q*Q-1.0) / m;
 b2 = (1.0 - Q/p + Q*Q) / m;
 y1=data(1);
 y2=data(2);
 x1=a0*y1;
 x2=a0*y2+a1*y1-b1*x1;
 I2(2)=x2;
 I2(1)=x1;
  for i=3:LL
        y3=data(i);
        y2=data(i-1);
        y1=data(i-2);
        x3=a0*y3+a1*y2+a2*y1-b1*x2-b2*x1;
        I2(i)=x3;
        x1=x2;
        x2=x3;
  end
   output=I2;
end

