% % Author: Arun Asokan Nair
% % Date created: 2020-01-02
% % Date modified: 2020-01-02
% % Purpose: This script takes the preprocessed data and splits it into
% % train, validation, and test data. 

% NOTE: Stuff marked as %%**SET**%% should probably be changed between
% simulations
%% Chose data directory that has just been generated and needs to be split into train-val-test
chosen_dir = '20200102_Sim3_0.05'; %%**SET**%%
mat_files = dir(fullfile(chosen_dir,'0*.mat'));
sprintf('Splitting %s into train, val and test', chosen_dir)
%% Create train, val and test dirs in it
created_folder_names = {'train', 'val', 'test'};
mkdir(fullfile(chosen_dir, created_folder_names{1})); % Just returns a warning if it exists already
mkdir(fullfile(chosen_dir, created_folder_names{2}));
mkdir(fullfile(chosen_dir, created_folder_names{3}));

%% Do the splitting and moving
% Seed the randon number generator
rng(1337);
index = randperm(numel(mat_files), numel(mat_files));

% Assume a 80%-10%-10% train-val-test split
idx_cell = {index(1:round(0.8*length(index))), index(round(0.8*length(index))+1:round(0.9*length(index))), ...
    index(round(0.9*length(index))+1:end)};

% Move the training data to the correct directory
for created_folder_idx = 1:length(created_folder_names)
    curr_created_folder_name = created_folder_names{created_folder_idx};
    curr_idx_array = idx_cell{created_folder_idx};
    for curr_idx = curr_idx_array
        source_file_name = fullfile(chosen_dir, mat_files(curr_idx).name);
        target_file_name = fullfile(chosen_dir, curr_created_folder_name, mat_files(curr_idx).name);
        movefile(source_file_name, target_file_name);
    end
    sprintf('Finished moving files into %s', curr_created_folder_name)
end
sprintf('Finished moving files into respective directories')