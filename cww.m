clc;

clear;

SampFreq = 30;

t=1/SampFreq:1/SampFreq:4;

sig = sin(12*pi*t);

sig(1:end/2) = sig(1:end/2) + sin(6*pi*t(1:end/2));

sig(end/2+1:end) = sig(end/2+1:end) + sin(18*pi*t(end/2+1:end));
figure(1)
plot(t,sig)

fmax = 0.5;

% ��߷���Ƶ��(��һ��Ƶ��)

fmin = 0.005;

% ��ͷ���Ƶ��(��һ��Ƶ��)

fb = 4 ;

% ȡcmor4-2С������ʵ�飬�������Ϊ4

fc = 2;

% ����Ƶ��2Hz

totalscal = 512;

% ��ȡ�߶ȵ���Ŀ

FreqBins = linspace(fmin,fmax,totalscal);

% ��Ƶ�����ڷ�����Χ�ڵȼ������

Scales = fc./ FreqBins;



RealFreqBins = FreqBins * SampFreq;
[MWT,f]=wsst(sig,SampFreq);

figure(2)

pcolor(t,f,abs(MWT))



shading interp;
axis tight;
  



axis([min(t) max(t) min(RealFreqBins) max(RealFreqBins)]);
axis tight;
  
ylabel('Frequency / Hz');

xlabel('Time / sec');