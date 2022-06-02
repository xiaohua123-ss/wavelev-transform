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

% 最高分析频率(归一化频率)

fmin = 0.005;

% 最低分析频率(归一化频率)

fb = 4 ;

% 取cmor4-2小波进行实验，带宽参数为4

fc = 2;

% 中心频率2Hz

totalscal = 512;

% 所取尺度的数目

FreqBins = linspace(fmin,fmax,totalscal);

% 将频率轴在分析范围内等间隔划分

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