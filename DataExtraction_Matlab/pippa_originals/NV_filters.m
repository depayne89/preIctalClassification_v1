% SET FILTERS FOR NV FEATURES

Ford = 1;
% Wideband Filter
Wc1 = 1;
Wc2 = 140;
W1 = Wc1/(Fs/2);
W2 = Wc2/(Fs/2);
[b,a] = butter(Ford,[W1 W2],'bandpass');
filter_wb = [b;a];
% Notch Filter
Wc1 = 45;
Wc2 = 55;
W1 = Wc1/(Fs/2);
W2 = Wc2/(Fs/2);
[b,a] = butter(Ford,[W1 W2],'stop');
filter_notch = [b;a];
% Filter One
Wc1 = 8;
Wc2 = 16;
W1 = Wc1/(Fs/2);
W2 = Wc2/(Fs/2);
[b,a] = butter(Ford,[W1 W2],'bandpass');
filter1 = [b;a];
% Filter Two
Wc1 = 16;
Wc2 = 32;
W1 = Wc1/(Fs/2);
W2 = Wc2/(Fs/2);
[b,a] = butter(Ford,[W1 W2],'bandpass');
filter2 = [b;a];
% Filter Three
Wc1 = 32;
Wc2 = 64;
W1 = Wc1/(Fs/2);
W2 = Wc2/(Fs/2);
[b,a] = butter(Ford,[W1 W2],'bandpass');
filter3 = [b;a];
% Filter Four
Wc1 = 64;
Wc2 = 128;
W1 = Wc1/(Fs/2);
W2 = Wc2/(Fs/2);
[b,a] = butter(Ford,[W1 W2],'bandpass');
filter4 = [b;a];

filters = [filter1 ; filter2 ; filter3 ; filter4];