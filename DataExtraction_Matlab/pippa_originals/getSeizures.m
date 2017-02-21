%% getSeizures
% gets pre-ictal segments from iEEG portal
% NB: at the moment only works for segments of less than 2000 s

clear
close all
clc

% IEEG LOGIN HERE
login = 'depayne';
pword = 'dep_ieeglogin.bin';

% Patients
Patient{1} = '23_002';
Patient{2} = '23_003';
Patient{3} = '23_004';
Patient{4} = '23_005';
Patient{5} = '23_006';
Patient{6} = '23_007';

Patient{7} = '24_001';
Patient{8} = '24_002';
Patient{9} = '24_004';
Patient{10} = '24_005';

Patient{11} = '25_001';
Patient{12} = '25_002';
Patient{13} = '25_003';
Patient{14} = '25_004';
Patient{15} = '25_005';

mkdir('TrainingData')

%% parameters
Fs = 400;  % for filtering
iCh = 1:16;

% Only save seizures with at least XX as seizure-free time beforehand (lead
% time)
% Lead time to include the seizures in seconds
LeadTime = 5*60*60;

% Get type 3 seizures = 1
Type3 = 0;

% Time before & after sz in seconds to save
Tbefore = 30*60;
Toffset = 60;

% training data time cutoff (days)
start_cutoff = 14;
%end_cutoff = 200;

%% Feature parameters
Nfeatures = 80;
time = 5;                       % get 5 s data to calculate features
shift = 1;                      % shift by 1 s at a time
extra = 0.5;                    % extra time for filter edges

% FILTERS
NV_filters

for iPt = 1:15
    
    %% link to portal session
    save_path = ['TrainingData/' Patient{iPt}];
    curPt = Patient{iPt};
    patient = IEEGSession(['NVC1001_' curPt '_2'],login,pword);
    
    Fs_actual = patient.data.sampleRate;
    timeS = round(Fs_actual*time);
    extraS = round(Fs_actual*extra);
    
    %% load information
    load(['Portal Annots/' curPt '_Annots']);
    load('Portal Annots/portalT0');
    trial_t0 = datenum(startDateTime(iPt));
    
    % chron. order
    [SzTimes,I] = sort(SzTimes);
    SzType = SzType(I);
    SzDur = SzDur(I);
    
    % circadian times
    SzCirc = trial_t0 + SzTimes/1e6/86400;
    SzCirc = datevec(SzCirc);
    SzCirc = SzCirc(:,4);
    
    %% get seizure index
    ISI = diff(SzTimes)/1e6;
    ISI = [LeadTime+1 ISI];
    if ~Type3
        remove = SzType == 3;
        SzType(remove) = [];
        ISI(remove) = [];
        SzTimes(remove) = [];
        SzCirc(remove) = [];
        SzDur(remove) = [];
    end
    
    % save seizures that have a leading interval of LeadTime and within
    % training period
    SzDay = ceil(SzTimes/1e6/60/60/24);
    training = SzDay > start_cutoff & SzDay < end_cutoff;
    SzInd = find(ISI > LeadTime & training);
    
    
    %% start grabbing data
    N = length(SzInd);
    preIctal = zeros(Nfeatures,Tbefore,N);
    preIctalDropouts = zeros(Tbefore,N);
    preIctalCirc = zeros(1,N);
    segment_length = Tbefore + time + 2*extra;
    
    meanTime = ceil(Fs_actual*segment_length);
    
    for n = 1:N
        
        % intialize start time
        t0 = SzTimes(SzInd(n))/1e6 - Tbefore - Toffset - (time-shift) - extra;
        
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
        
        % pre-filter
        Data(isnan(Data(:,1)),:) = 0;
        Data = filtfilt(filter_wb(1,:),filter_wb(2,:),Data);
        
        % need to grab the data in segments
        mu = zeros(Nfeatures,1800,2);
        for nn = 0:shift:(Tbefore/shift-shift);
            
            ind1 = floor(Fs_actual*nn)+1;
            curSeg = Data(ind1:ind1+timeS+2*extraS,:);
            
            
            if sum(curSeg(:,1).^2) < 1e-16
                % ignore dropout sections
                preIctalDropouts(nn/shift+1,n) = 1;
                continue;
            end
            
            % calculate features from data (all 80 features)
            features = calculate_features(curSeg,1:80,filters,extraS);
            
            if sum(isnan(features))
                display('update code')
                return;
            end
            
            % save into the data matrix
            preIctal(:,nn/shift+1,n) = features;
            
        end % end feature segments
        
        % save the time of day
        preIctalCirc(n) = SzCirc(n);
        
        fprintf('%d of %d seizures processed\n',n,N)
    end % end seizure loop
    
    fprintf('saving training data ... \n');
    save([save_path 'TrainingSeizures'],'preIctalDropouts','preIctalCirc', 'preIctal','N','-v7.3')
    
end