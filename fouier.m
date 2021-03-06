clear;
clc;
N = 1000;
fs = 1000;
dt = 1/fs;
t = 0:dt:N*dt-dt;
xt1 = sin(t*50*pi);
t1 = 101:200;
xt1(101:200) = cos((t1-100)*pi/100);
xt2 = sin(t*50*pi);
t2 = 801:900;
xt2(801:900) = cos((t2-800)*pi/100);
xf1 = fft(xt1);
xf2 = fft(xt2);
figure(1);
subplot(2,2,1);plot(t,xt1);grid on;title("ʱ??1");
subplot(2,2,3);plot(abs(xf1));grid on;title("Ƶ??ͼ1");

subplot(2,2,2);plot(t,xt2);grid on;title("ʱ??2");
subplot(2,2,4);plot(abs(xf2));grid on;title("Ƶ??ͼ2");
