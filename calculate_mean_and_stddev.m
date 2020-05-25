% % Author: Arun Asokan Nair
% % Date created: 2020-01-02
% % Date modified: 2020-01-02
% % Purpose: This script takes the data that's been split into train, val,
% % and test and calculates normalization values (mean, std-dev) on the
% % training set to apply to the data

clc; clearvars; close all;
addpath('./utils');

%% Chose data directory that has just been split into train, val, and test and calculated mean & std-dev on the training set
% ONLY USE THE TRAINING SET FOR THIS! (otherwise you're peeking at test
% data)
root_dir = '20200102_Sim3_0.05';
data_dir = fullfile(root_dir, '/train');
mat_files = dir(fullfile(data_dir,'0*.mat'));
ch1_means = zeros(1,length(mat_files));   % Array to store means of feature channel 1
ch1_stddevs = zeros(1,length(mat_files)); % Array to store std-devs of feature channel 1
ch2_means = zeros(1,length(mat_files));   % Array to store means of feature channel 2
ch2_stddevs = zeros(1,length(mat_files)); % Array to store std-devs of feature channel 2

for idx = 1:length(mat_files)
    curr_file_name = mat_files(idx).name;
    load(fullfile(mat_files(idx).folder, curr_file_name));
    ch1 = input_data(1,:,:);
    ch2 = input_data(2,:,:);
    ch1_means(idx) = mean(ch1(:));
    ch1_stddevs(idx) = std(ch1(:));
    ch2_means(idx) = mean(ch2(:));
    ch2_stddevs(idx) = std(ch2(:));
end
overall_ch1_mean = mean(ch1_means);
overall_ch2_mean = mean(ch2_means);
overall_ch1_stddev = calc_stddevs(ch1_stddevs, ch1_means);
overall_ch2_stddev = calc_stddevs(ch2_stddevs, ch2_means);

% Save calculated mean and std-dev to a mat file
save(fullfile(root_dir,'normalization_params.mat'), 'overall_ch1_mean', 'overall_ch1_stddev',  'overall_ch2_mean', 'overall_ch2_stddev');

%% Normalize the dataset by the calculated mean and standard deviation
load(fullfile(root_dir,'normalization_params.mat'));
created_folder_names = {'train', 'val', 'test'};
% start parallel pool if not already running
p = gcp; 

for created_folder_idx = 1:length(created_folder_names)
    curr_created_folder_name = created_folder_names{created_folder_idx};
    data_dir = fullfile(root_dir, curr_created_folder_name);
    mat_files = dir(fullfile(data_dir,'0*.mat'));
    parforProgress(length(mat_files));
    parfor idx=1:length(mat_files)
        curr_file_name = fullfile(mat_files(idx).folder, mat_files(idx).name);
        img_struct = load(curr_file_name);
        img_struct.input_data(1,:,:) = (img_struct.input_data(1,:,:)-overall_ch1_mean)/overall_ch1_stddev;
        img_struct.input_data(2,:,:) = (img_struct.input_data(2,:,:)-overall_ch2_mean)/overall_ch2_stddev;
        parsave(curr_file_name, img_struct.file_name_idx, img_struct.input_data, img_struct.target_output);
        parforProgress;
    end
    parforProgress(0);
    sprintf('Finished normalizing %s', curr_created_folder_name)
end
sprintf('Finished normalizing all the files')
% kill parallel pool
delete(p);