function [ output_args ] = LPF( data,rate,frequency )
%%%%%低通%%%%%
 LL=length(data);
 fh=frequency;fs=rate;
 wh = fh * pi / fs;                    
 Q = tan(wh);%角频率
 p=0.707;%品质因数
 m = 1.0 + Q/p + Q*Q;
 a = 1.414*Q*Q / m;
 b1 = 2.0*(Q*Q-1.0) / m;
 b2 = (1.0 - Q/p + Q*Q) / m;
 y1=data(1);
 y2=data(2);
 x1=a*y1;
 x2=a*(y2+2.0*y1)-b1*x1;
 data2(2)=x2;
 data2(1)=x1;
 %模拟AD进入的
 for i=3:LL 
     % 直接低通滤波
      y3=data(i);
      y2=data(i-1);
      y1=data(i-2);
      x3=a*(y3+2.0*y2+y1)-b1*x2-b2*x1;
      data2(i)=x3;
      x1=x2;
      x2=x3;
 end
 i=1;
 for i=1:length(data2)-30
 output_args(i)=data2(i+30);
 i=i+1;
 end