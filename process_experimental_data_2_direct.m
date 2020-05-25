% clc; clearvars; close all
addpath('./utils');

% Build sensing matrix and measurements
load('A.mat'); 

% Permute array dimensions to ensure it's numpats x height x width
At = permute(A, [3 1 2]);

% Load the randomly projected samples
load('fs_2avg.mat');

% Load the params file
load_dir = '20200102_Sim3_0.05';
load(fullfile(load_dir, 'params.mat'));

%--------------------------------------------------------------------------
% Try 1: Directly apply pinv
% % So that didn't work, surprisingly, out of the box. Setting all pixels<0
% % to 0 helps, but still a pretty crappy image
target_output = apply_pinv(params, At, f);
target_output(target_output<0)=0;

%--------------------------------------------------------------------------
% Try 2: Follow traditional pipeline for oversampled case to get 
% target output image, which is then reprocessed to match training data
% statistics

path(path,genpath('../ModifiedDungCode'));
addpath(genpath('/data/Dropbox/Arun/Research/NIH-Imaging/Toolboxes'));

%--------------------------------------------------------------------------
% Problem size
dimy = 100; % y-dimension (number of image columns)                             
dimx = 100; % x-dimension (number of image rows)
numpats = 5; % Number of random patters

% build sensing matrix and measurements
load('A.mat'); % Loads a matrix A that is 100X100X200 i.e., dimy x dimx x numpats
% A = zeros(numpats,dimx,dimy);
At = zeros(numpats,dimx,dimy);
for k = 1:numpats
    temp = squeeze(A(:,:,k));
    At(k,:,:) = temp;
end

% f = zeros(numpats,dimy);
load('fs_2avg.mat');
f = f(1:numpats,:);
f = circshift(f,[0 0]); % This does not do anything...

fprintf('Finished building sensing matrix \n');

%% patch parameters
lambda_lci = 0.5e4; % Jeff mostly adjusts this hyperparameter
rho = 1;
alpha = 1;
max_iter   = 30;

verbose    = 1;
abs_tol    = 1e-6;
rel_tol    = abs_tol*1e-3;
different  = true;
smooth     = false;
% parallel   = true;
parallel   = false;
nonnegative   = false;
patch_size_1 = 8;
patch_size_2 = 8;

%% run patch
Xr = zeros(dimy,dimx);

Xr = Arun_lci_patch_l1(f, At, ...
    lambda_lci, rho, alpha, ...
    'verbose'         , verbose   , ...
    'max_iter'        , max_iter  , ...
    'abs_tol'         , abs_tol   , ...
    'rel_tol'         , rel_tol   , ...
    'different'       , different , ...
    'parallel'        , parallel  , ...
    'nonnegative'        , nonnegative  , ...
    'patch_size_1'      , patch_size_1, ...
    'patch_size_2'      , patch_size_2, ...
    'initial_estimate', Xr        , ...
    'smooth'          , smooth    );

%% display results

figure;
imshow(Xr,[]);

b = makeColorMap([0 0 0],[0.2 0.2 1],[1 1 1],2^8);
colormap(b);

Xr_padded = padarray(Xr,[14 14],0,'both');

%% Now, reproject this recovered image to get data to pass to the network - Experimental
% Update At based on the set sampling ratio to select only a subset of patterns
At = At(1:floor(params.sampling_ratio*size(At,2)),:,:);

resized_img = Xr/max(Xr(:))*255; % Rescale it to be at 255

% % Apply random projections on it to get f
% f = apply_random_projections(params, At, resized_img);
f_2 = f;

% Apply the pseudo inverse on it to get back pinv_img
pinv_img_2 = apply_pinv(params, At, f_2);
    
% Apply the matrix transpose on it to get back transposed_img
transposed_img_2 = apply_transpose(params, At, f_2);
    
% Pad the arrays with zeros to get them to be 128x128 images
resized_img_padded = padarray(resized_img,[14 14],0,'both');    
pinv_img_padded = padarray(pinv_img_2,[14 14],0,'both');
transposed_img_padded = padarray(transposed_img_2,[14 14],0,'both');
    
% Stack pinv_img and transposed_img into one tensor
input_data = zeros(2,size(resized_img_padded,1), size(resized_img_padded,2));
input_data(1,:,:) = pinv_img_padded(:,:);
input_data(2,:,:) = transposed_img_padded(:,:);

% Normalize the input_data
load(fullfile(load_dir, 'normalization_params.mat'));
input_data(1,:,:) = (input_data(1,:,:)-overall_ch1_mean)/overall_ch1_stddev;
input_data(2,:,:) = (input_data(2,:,:)-overall_ch2_mean)/overall_ch2_stddev;

i = 077779;
target_output = uint8(resized_img_padded/255); % TODO: Verify
target_output = reshape(target_output, 1, size(target_output,1), size(target_output,2));
file_name = fullfile(params.output_dir, sprintf('%06d.mat',i-1));
file_name_idx = i-1;
parsave(file_name, file_name_idx, input_data, target_output);

%% Now, reproject this recovered image to get data to pass to the network - Experimental - with scaling
% Update At based on the set sampling ratio to select only a subset of patterns
At = At(1:floor(params.sampling_ratio*size(At,2)),:,:);

resized_img = Xr/max(Xr(:))*255; % Rescale it to be at 255

% % Apply random projections on it to get f
% f = apply_random_projections(params, At, resized_img);
f_3 = f;

% Apply the pseudo inverse on it to get back pinv_img
pinv_img_3 = apply_pinv(params, At, f_3);
pinv_img_3 =pinv_img_3/2;
    
% Apply the matrix transpose on it to get back transposed_img
transposed_img_3 = apply_transpose(params, At, f_3);
transposed_img_3 = transposed_img_3*3/4;
    
% Pad the arrays with zeros to get them to be 128x128 images
resized_img_padded = padarray(resized_img,[14 14],0,'both');    
pinv_img_padded = padarray(pinv_img_3,[14 14],0,'both');
transposed_img_padded = padarray(transposed_img_3,[14 14],0,'both');
    
% Stack pinv_img and transposed_img into one tensor
input_data = zeros(2,size(resized_img_padded,1), size(resized_img_padded,2));
input_data(1,:,:) = pinv_img_padded(:,:);
input_data(2,:,:) = transposed_img_padded(:,:);

% Normalize the input_data
load(fullfile(load_dir, 'normalization_params.mat'));
input_data(1,:,:) = (input_data(1,:,:)-overall_ch1_mean)/overall_ch1_stddev;
input_data(2,:,:) = (input_data(2,:,:)-overall_ch2_mean)/overall_ch2_stddev;

i = 077780;
target_output = uint8(resized_img_padded/255); % TODO: Verify
target_output = reshape(target_output, 1, size(target_output,1), size(target_output,2));
file_name = fullfile(params.output_dir, sprintf('%06d.mat',i-1));
file_name_idx = i-1;
parsave(file_name, file_name_idx, input_data, target_output);

%% Now, reproject this recovered image to get data to pass to the network - Original method
% Update At based on the set sampling ratio to select only a subset of patterns
At = At(1:floor(params.sampling_ratio*size(At,2)),:,:);

resized_img = Xr/max(Xr(:))*255; % Rescale it to be at 255

% Apply random projections on it to get f
f = apply_random_projections(params, At, resized_img);
    
% Apply the pseudo inverse on it to get back pinv_img
pinv_img = apply_pinv(params, At, f);
    
% Apply the matrix transpose on it to get back transposed_img
transposed_img = apply_transpose(params, At, f);
    
% Pad the arrays with zeros to get them to be 128x128 images
resized_img_padded = padarray(resized_img,[14 14],0,'both');    
pinv_img_padded = padarray(pinv_img,[14 14],0,'both');
transposed_img_padded = padarray(transposed_img,[14 14],0,'both');
    
% Stack pinv_img and transposed_img into one tensor
input_data = zeros(2,size(resized_img_padded,1), size(resized_img_padded,2));
input_data(1,:,:) = pinv_img_padded(:,:);
input_data(2,:,:) = transposed_img_padded(:,:);

% Normalize the input_data
load(fullfile(load_dir, 'normalization_params.mat'));
input_data(1,:,:) = (input_data(1,:,:)-overall_ch1_mean)/overall_ch1_stddev;
input_data(2,:,:) = (input_data(2,:,:)-overall_ch2_mean)/overall_ch2_stddev;

i = 077778;
target_output = uint8(resized_img_padded/255); % TODO: Verify
target_output = reshape(target_output, 1, size(target_output,1), size(target_output,2));
file_name = fullfile(params.output_dir, sprintf('%06d.mat',i-1));
file_name_idx = i-1;
parsave(file_name, file_name_idx, input_data, target_output);

% 
% after getting pinv_img, pinv_img_2, pinv_img_3 and the same for
% transposed
figure; subplot(131); imagesc(pinv_img); colormap gray; subplot(132); imagesc(pinv_img_2); colormap gray; subplot(133); imagesc(pinv_img_3); colormap gray;
figure; subplot(131); imagesc(transposed_img); colormap gray; subplot(132); imagesc(transposed_img_2); colormap gray; subplot(133); imagesc(transposed_img_3); colormap gray;

figure; subplot(131); imagesc(squeeze(DNN_output)); title('Dung Algorithm reprojected'); colormap gray; subplot(132); imagesc(squeeze(DNN_output)); title('Direct'); colormap gray; subplot(133); imagesc(squeeze(DNN_output)); colormap gray;
% figure; subplot(131); imagesc(transposed_img); colormap gray; subplot(132); imagesc(transposed_img_2); colormap gray; subplot(133); imagesc(transposed_img_3); colormap gray;