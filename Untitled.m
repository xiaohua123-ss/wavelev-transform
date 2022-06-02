
data1=load('2.s2p');

nout=1.9403e-14 	 ;
S21_real=data1(6,4);
S21_imag=data1(6,5);
G=abs(S21_real+1j*S21_imag);
10*log10(G)
match(nout,G)


