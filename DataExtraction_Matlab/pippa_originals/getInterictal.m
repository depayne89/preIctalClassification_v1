clear
close all
clc

% IEEG LOGIN HERE
login = 'depayne';
pword = 'dep_ieeglogin.bin';

%% Patients

Patient{1} = '23_002';   % 18 ??
Patient{2} = '23_003';   % 4
Patient{3} = '23_004';   % 32   ***
Patient{4} = '23_005';   % 11
Patient{5} = '23_006';   % 2
Patient{6} = '23_007';   % 18 ??

Patient{7} = '24_001';  % 16
Patient{8} = '24_002';  % 45  ***
Patient{9} = '24_004';  % 44  ***
Patient{10} = '24_005'; % 70 ***

Patient{11} = '25_001'; % 26 ***
Patient{12} = '25_002'; % 0
Patient{13} = '25_003'; % 63 ***
Patient{14} = '25_004'; % 0
Patient{15} = '25_005'; % 22 ***

mkdir('TrainingData')


%% parameters
Fs = 400;  % for filtering
iCh = 1:16;  % channels to take

% Get type 3 seizures = 1
Type3 = 0;

% Time in seconds to save
DataWin = 60*60; 
% gap between seizure for interictal
InterWin = 6*60*60; 
Tbefore = 30*60;    

% training data time cutoff (days)
start_cutoff = 100;
end_cutoff = 200;

%% Feature parameters
Nfeatures = 80;
time = 5;                          % get 5 s data to calculate features
shift = 1;                          % shift by 1 s at a time
extra = 0.5;                      % extra time for filter edges

% FILTERS
NV_filters

%% load data
for iPt = [9, 10, 11, 13, 15]
% for iPt = 1:15
    
    save_path = ['TrainingData/' Patient{iPt}];
    
    %% link to portal session
    curPt = Patient{iPt};
    patient = IEEGSession(['NVC1001_' curPt '_2'],login,pword);
    Fs_actual = patient.data.sampleRate;
    
    timeS = round(Fs_actual*time);
    extraS = round(Fs_actual*extra);
    
    % load([curPt '_DataInfo']);
    % trial_t0 = datenum(MasterTimeKey(1,:));
    load(['Portal Annots/' curPt '_Annots']);
    load('Portal Annots/portalT0');
    trial_t0 = datenum(startDateTime(iPt));
    
    % chron. order
    [SzTimes,I] = sort(SzTimes);
    SzType = SzType(I);
    SzDur = SzDur(I);
    SzInd = SzInd(I);
    
    % circadian times
    SzCirc = trial_t0 + SzTimes/1e6/86400;
    SzCirc = datevec(SzCirc);
    SzCirc = SzCirc(:,4);
    
    %% get seizure index
    ISI = diff(SzTimes)/1e6;
    ISI = [0 ISI];
    if ~Type3
        remove = SzType == 3;
        SzType(remove) = [];
        ISI(remove) = [];
        SzTimes(remove) = [];
        SzCirc(remove) = [];
    end
    
    % find places where we have at least InterWin hours free of seizure before AND
    % after
    train_inter = find(ISI > 2*InterWin);
    % this is where we can start interictal windows from (InterWin hours after the seizure before the one with the long ISI)
    InterInd = ceil(SzTimes(train_inter-1)/1e6) + InterWin;   % in s
    InterCirc = mod(SzCirc(train_inter-1) + InterWin/60/60,24);
    % length the interictal window can extend for (InterWin hours before next seizure)
    InterLength = floor((ISI(train_inter)-2*InterWin));  % in s
    
    % get rid of times outside training period
    training = InterInd > (start_cutoff*24*60*60) & InterInd <= (end_cutoff*24*60*60);
    InterLength = InterLength(training);
    InterInd = InterInd(training);
    InterCirc = InterCirc(training);
    
    InterIctalTimes = [];
    InterCircTimes = [];
    for m = 1:length(InterInd)
        Time = InterInd(m):DataWin:DataWin*floor((InterInd(m)+InterLength(m))/DataWin);
        CircTime = InterCirc(m):InterCirc(m)+length(Time)-1;    % HACK ONLY WORKS IF DATAWIN IS ONE HOUR
        CircTime = mod(CircTime,24);
        InterIctalTimes = [InterIctalTimes Time];
        InterCircTimes = [InterCircTimes CircTime];
    end
    
    %% get rid of times that aren't in the troughs
    % load([save_path 'SzProb']);
    % lowSeizure = find(SzProb < prctile(SzProb,50))-1;
    % valid = ismember(InterCircTimes,lowSeizure);
    % InterIctalTimes = InterIctalTimes(valid);
    % InterCircTimes = InterCircTimes(valid);
    % InterIctalTimes = [InterIctalTimes InterIctalTimes + Tbefore];  % hack to get the extra half an hour windows
    % InterCircTimes = [InterCircTimes InterCircTimes];
    
    %%  get the same amount of interictal data
    data = matfile([save_path 'TrainingSeizures']);
    N = data.N;
    randomInd = randperm(length(InterIctalTimes));
    segment_length = Tbefore + time + 2*extra;
    
    nseg = 0;
    interIctal = zeros(Nfeatures,Tbefore,N);
    interIctalCirc = zeros(1,N);
    interIctalDropouts = zeros(Tbefore,N);
    tic
    
    while nseg < N
        
        if isempty(randomInd)
            display('no more data')
            break;
        end
        
        t0 = InterIctalTimes(randomInd(1));  % interictal window time
        circT = InterCircTimes(randomInd(1));
        randomInd(1) = [];
        
        % get the data from the portal
        try
            Data = getvalues(patient.data,t0 * 1e6,segment_length * 1e6,iCh);
        catch
            % try again
            try
                Data = getvalues(patient.data,t0 * 1e6,segment_length * 1e6,iCh);
            catch
                % maybe lost connection.
                display('early termination');
                continue;
            end
        end
        
        % check dropouts
        if sum(isnan(Data(:,1))) > 0.25*length(Data)
            display('too many dropouts');
            continue;
        end
        Data(isnan(Data(:,1)),:) = 0;
        
        % we can use the segment
        nseg = nseg+1;
        % pre-filter
        Data = filtfilt(filter_wb(1,:),filter_wb(2,:),Data);
        
        % need to grab the data in segments
        mu = zeros(Nfeatures,1800,2);
        for nn = 0:shift:(Tbefore/shift-shift);
            
            ind1 = floor(Fs_actual*nn)+1;
            curSeg = Data(ind1:ind1+timeS+2*extraS,:);
            
            if sum(curSeg(:,1).^2) < 1e-16
                % ignore dropout sections
                interIctalDropouts(nn/shift+1,nseg) = 1;
                continue;
            end
            
            % calculate features from data (all 80 features)
            features = calculate_features(curSeg,1:80,filters,extraS);
            
            if sum(isnan(features))
                display('update code')
                return;
            end
            
            % save into the data matrix
            interIctal(:,nn/shift+1,nseg) = features;
            
        end % end feature segments
        
        % save the time of day
        interIctalCirc(nseg) = circT;
        
        temp = toc;
        fprintf('%d of %d segments processed, time elapsed: %.2f s\n',nseg,N,temp)
        
    end
    
    save([save_path 'TrainingNonSeizures'],'interIctalDropouts','interIctalCirc','interIctal','-v7.3')
    
end