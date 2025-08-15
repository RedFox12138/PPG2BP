function [ output ] = HPF( data,rate,frequency )
%%%%%%��ͨ%%%%%% 
 LL=length(data);%���ݳ���
 %һ����ִ�У����Գ�ʼ����� ��һ��60 Hz
 fh=frequency;fs=rate;
 wh = fh * pi / fs;
 Q = tan(wh);%��Ƶ��
 p=0.707;%Ʒ������
 m = 1.0 + Q/p + Q*Q;
 a =1 / m;
 b1 = 2.0*(Q*Q-1.0) / m;
 b2 = (1.0 - Q/p + Q*Q) / m;
 y1=data(1);
 y2=data(2);
 x1=a*y1;
 x2=a*(y2-2.0*y1)-b1*x1;
 I3(2)=x2;
 I3(1)=x1;
 %ģ��AD�����
for i=3:LL 
     % ֱ�Ӹ�ͨ�˲�
      y3=data(i);
      y2=data(i-1);
      y1=data(i-2);
      x3=a*(y3-2.0*y2+y1)-b1*x2-b2*x1;
      I3(i)=x3;
      x1=x2;
      x2=x3;
    
end
data=I3;

%%%%%�ڶ��θ�ͨ%%%%%
 fh=frequency;fs=rate;
 wh = fh * pi / fs;
 Q = tan(wh);%��Ƶ��
 p=0.707;%Ʒ������
 m = 1.0 + Q/p + Q*Q;
 a =1 / m;
 b1 = 2.0*(Q*Q-1.0) / m;
 b2 = (1.0 - Q/p + Q*Q) / m;
 y1=data(1);
 y2=data(2);
 x1=a*y1;
 x2=a*(y2-2.0*y1)-b1*x1;
 I3(2)=x2;
 I3(1)=x1;
 %ģ��AD�����
for i=3:LL 
     % ֱ�Ӹ�ͨ�˲�
      y3=data(i);
      y2=data(i-1);
      y1=data(i-2);
      x3=a*(y3-2.0*y2+y1)-b1*x2-b2*x1;
      I3(i)=x3;
      x1=x2;
      x2=x3;
    
end
output=I3;

end

