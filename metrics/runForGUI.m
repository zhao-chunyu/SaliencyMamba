% add path
addpath(genpath(pwd));  

% inint
data_cls=''
path1=''
path2=''
path3=''
params = {data_cls, path1, path2, path3}

% check cell
if ~iscell(params)
    error('Input must be a cell array');
end

% get params
num_params = length(params);
fprintf('Total number of parameters: %d\n', num_params);

% print params
fprintf('[%d] category of dataset : %s\n', 1, params{1});
fprintf('[%d] modelOutput matPath : %s\n', 2, params{2});
fprintf('[%d] fixations datapath  : %s\n', 3, params{3});
fprintf('[%d] test data datapath  : %s\n', 4, params{4});

dataset_cls = params{1}

if strcmp(dataset_cls, 'TrafficGaze')
    calcForTrafficGaze(params{2}, params{3}, params{4})
elseif strcmp(dataset_cls, 'DrFixD_rainy')
    calcForDrFixDrainy(params{2}, params{3}, params{4})
else
    calcForBDDA(params{2}, params{3}, params{4})
end