close all; clear; clc;
fs = 1000;
t = 0:1/fs:4-1/fs;
L=4000;

signal=10*cos(2*pi*20*t).*(t>=0 & t<1)+80*cos(2*pi*80*(t-1)).*(t>1 & t<=2)+60*cos(2*pi*60*(t-2)).*(t>2 & t<=3)+40*cos(2*pi*40*(t-3)).*(t>3 & t<=4);  





len =  [32, 64, 128, 256];
for i = 1:4
    wlen = len(i);
    hop = wlen/4;
    nfft = wlen;
    
    win = blackman(wlen, 'periodic');
    [S, f, t] = spectrogram(signal, win, wlen - hop, nfft, fs);
    subplot(2, 2, i);
    PlotSTFT(t,f,S,win);
    [m, n] = size(S);
    t = sprintf('Wlen = %d, S: [%d, %d]', wlen, m, n);
    title(t);
end
