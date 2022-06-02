function PlotSTFT(T, F, S, win)
    wlen = length(win);
    C = sum(win)/wlen;
    S = abs(S)/wlen/C;
    
    S = 20*log10(S + 1e-6);

    %figure(1)
    surf(T, F, S)
    shading interp;
    axis tight;
    view(0,90);
    %set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);
    xlabel('Time, s');
    ylabel('Frequency, Hz');
    title('Amplitude spectrogram of the signal');
    
    hcol = colorbar;
    %set(hcol, 'FontName', 'Times New Roman', 'FontSize', 14);
    ylabel(hcol, 'Magnitude, dB');
end
