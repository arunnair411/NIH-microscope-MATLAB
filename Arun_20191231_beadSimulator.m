clc; clearvars; close all
addpath('./utils');
%--------------------------------------------------------------------------
% Attempt 1:
% 1) Randomly rotated **triangles** of beads 
% 2) Allows for possible overlap between beads - check temp_store{25} for
% an example, quite rare in practice...
% TODO: Also include Dung's processing code to generate a third channel for
% this process
% TODO: Can eventually include other shape templates in addition to triangles

% NOTE: Stuff marked as %%**SET**%% should probably be changed between
% simulations

%% Set simulation parameters

% Build sensing matrix and measurements
load('A.mat'); 
% Permute array dimensions to ensure it's numpats x height x width
At = permute(A, [3 1 2]);


% Set the total number of simulations to run
params.num_simulations = 20000; %%**SET**%%
% Set the sampling ratio (SR) - Can be oversampled!!! i.e. up to 2.0
params.sampling_ratio = 0.05; %%**SET**%%
% Update At based on the set sampling ratio to select only a subset of patterns
At = At(1:floor(params.sampling_ratio*size(At,2)),:,:);
% Set the possible number of beads per cluster
params.beads_per_cluster = [1, 3, 6, 10, 15];
% Set the probability distribution function (pdf) over the possible number 
% of beads per cluster
params.cluster_pdf = [0.5, 0.25, 0.125, 0.0625, 0.0625];
% Usingan interbeadspacing multiplier of slightly greater than 1 otherwise
% the images blend into each other. 1 is the tightly packed case.
params.inter_bead_spacing_multiplier = 1.1; 
params.img_height_input = 1000;         % [pixels]
params.img_width_input = 1000;          % [pixels]
params.bead_radius = 30;                % [pixels] - this is w.r.t. input scale
params.img_height_output = size(At,2);  % [pixels]
params.img_width_output = size(At,3);   % [pixels]

% Check for consistency of parameters
assert(length(params.beads_per_cluster)==length(params.cluster_pdf), 'Length of the pdf vector has to match the length of possible bead configurations' );
assert(sum(params.cluster_pdf)==1, 'pdf vector has to sum to 1' );

% Create template images for each of the number of beads per cluster
params.templates = create_templates(params);

% Set the parent save directory - this is the directory the output mat
% files are sent to
params.output_dir = '20200102_Sim3_0.05' %%**SET**%%
if ~exist(fullfile(params.output_dir), 'dir')
    mkdir(fullfile(params.output_dir));
end

% Save params to the directory
save(fullfile(params.output_dir, 'params.mat'), 'params');
%% Start image generation
% Load params file
load(fullfile(params.output_dir, 'params.mat'));
% Start timing
tic;

% start parallel pool if not already running
p = gcp;
 
disp('Now saving processed data..............');
sprintf('%s data samples to process at a sampling ratio of %s ..........',...
    num2str(params.num_simulations), num2str(params.sampling_ratio))
parforProgress(params.num_simulations);
parfor i=1:params.num_simulations
    % Create a baseline all zeros image of the right size
    img_generated_input = uint8(zeros(params.img_height_input, params.img_width_input));
    
    % Set the seed for the random number generator to ensure
    % reproducibility
    rng(i);
    
    % Randomly choose number of clusters to be between 1 and max_num)clusters, 
    % uniformly distributed    
    max_num_clusters = 3;
    num_clusters = floor(rand(1,1)*max_num_clusters)+1; % ~Zero probability that it will choose exactly max_num_clusters - verified for a million data points
    
    for j = 1:num_clusters
        
        % Randomly choose number of beads in the cluster
        beads_in_curr_cluster = randsample(params.beads_per_cluster, 1, true, params.cluster_pdf);
        
        % Find idx for the chosen number of cluster beads and retrieve the
        % corresponding template image
        [~, I] = find(params.beads_per_cluster==beads_in_curr_cluster);
        temp_img = params.templates{I};
        
        % Randomly choose a rotation angle for the image and rotate the
        % image
        temp_rot_ang_d = floor(rand(1,1)*360);
        temp_img_rotated = imrotate(temp_img, temp_rot_ang_d, 'bilinear', 'crop');
        
        % Randomly choose a circular shift for the image and circshift it
        % in both x and y directions - Upto +-25% shift of total image size
        temp_circ_shift_x = floor(-params.img_width_input/4  + rand(1,1)*params.img_width_input/2 );
        temp_circ_shift_y = floor(-params.img_height_input/4 + rand(1,1)*params.img_height_input/2);
        temp_img_rotated_circ_shifted = circshift(temp_img_rotated, [temp_circ_shift_y, temp_circ_shift_x]);
        
        % Add the processed template to the basline image - since it should
        % be typecasted to uint8, it should saturate at 255 if there is
        % overlap...
        img_generated_input = img_generated_input + temp_img_rotated_circ_shifted;
    end
    % Resize and store the image - REMEMBER THE ANTIALIASING! (It's worse
    % without it)
    resized_img = imresize(img_generated_input, [params.img_height_output, params.img_width_output]);
    
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
    
    target_output = resized_img_padded/255;
    target_output = reshape(target_output, 1, size(target_output,1), size(target_output,2));
    file_name = fullfile(params.output_dir, sprintf('%06d.mat',i-1));
    file_name_idx = i-1;
    parsave(file_name, file_name_idx, input_data, target_output);
    
    
    parforProgress;
end
parforProgress(0);
disp('Processing done!');

% kill parallel pool
delete(p);

% Total time
time = toc;
sprintf('Total time taken for data pre-processing is %d seconds', time)

%% SUCCESS - Below, I try to understand the procedure to generate the data by applying the A matrix on my simulated data 
% and trying to use Dung's code to recover it - SUCCESS!
% sanity_check_img = double(temp_store{23});
% numpats = size(At,1);
% f = zeros(size(At,1),size(At,2));
% for i=1:numpats
%     curr_A = squeeze(At(i,:,:));
% %     f(i,:) = (sum(curr_A.*sanity_check_img,2)).'; % Trial 1 - sum along columns after elementwise product
%     f(i,:) = (sum(curr_A.*sanity_check_img,1)); % Trial 2 - sum along rows after elementwise product - THIS IS DA WAE!
% end
% save('Arun_20191231_fs.mat', 'f');



%% SUCCESS - Below, I try to do the same thing as above using matrix operations 
% (more amenable to finding a pseudo-inverse down the road, and using for CS)
% sanity_check_img = double(temp_store{23});
% numpats = size(At,1);
% f = zeros(size(At,1),size(At,2));
% for i=1:numpats
%     curr_A = squeeze(At(i,:,:));
%     % Step 1 - Will need to construct a matrix that achieves the same as Trial
%     % 2 above - worked out on paper what matrix you need. 
%     curr_A_transposed = curr_A.';
%     redq_big_mat = zeros(size(curr_A_transposed,1), size(curr_A_transposed,1)*size(curr_A_transposed,2));
%     for j =1:size(curr_A_transposed,1)
%         redq_big_mat(j, (j-1)*size(curr_A_transposed,2)+1:j*size(curr_A_transposed,2)) = curr_A_transposed(j,:);        
%     end
%     % Step 2 - Simply do the multiplication
%     
%     f(i,:) = (redq_big_mat*sanity_check_img(:)).'; % Trial 2 from above in matrix form
% end

%% SUCCESS - Now, I write code to do a pseudoinverse to recover each column 
% % from from its observations just to see how it looks - this is likely what 
% % I'll be feeding into my DNN. Also verify if I can use it to recover the 
% % image using Dung's code
% sanity_check_img = double(temp_store{23});
% numcols = size(At,3);
% f = zeros(size(At,1),size(At,2));
% for i=1:numcols
%     curr_A = squeeze(At(:,:,i));
%     f(:,i) = curr_A*sanity_check_img(:,i);
% end
% % Try to reconstruct the image columnwise using a pseudoinverse and see
% % what it looks like...
% % NOTE: pinv() and MLDIVIDE(\) behave differently for underdetemined
% % systems! Read online for more info...
% recon_img = zeros(size(sanity_check_img));
% for i=1:numcols
%     curr_A = squeeze(At(:,:,i));
%     recon_img(:,i) = pinv(curr_A)*f(:,i);
% %     recon_img(:,i) = curr_A\f(:,i);
% end

%% SUCCESS - Now, I write code to do a matrix transpose to regenerate a column shaped
% from from its observations just to see how it looks - this is another
% popular option to feed into my DNN. It's the same projection code from 
% above, so Dung's code still successfully recorvers it.
% sanity_check_img = double(temp_store{23});
% numcols = size(At,3);
% f = zeros(size(At,1),size(At,2));
% for i=1:numcols
%     curr_A = squeeze(At(:,:,i));
%     f(:,i) = curr_A*sanity_check_img(:,i);
% end
% % Try to regenrate the image columnwise simply the transpose of the matrix
% % and see what it looks rike
% regen_img = zeros(size(sanity_check_img));
% for i=1:numcols
%     curr_A = squeeze(At(:,:,i));
%     regen_img(:,i) = curr_A'*f(:,i);
% end
% 
% % Maybe, as a first step provide both?
% 
% % Save the result to file
% save('Arun_20191231_fs_2avg.mat', 'f');

function [templates] = create_templates(params)
    templates = {}; % Cell array to store the templates
    
    % Create a meshgrid of coordinates to use for generating the templates
    [X,Y] = meshgrid(-params.img_width_input/2:params.img_width_input/2-1, -params.img_height_input/2:params.img_height_input/2-1);
    
    % Create a list of offsets for the different beads per cluster relative
    % to the lowest left bead, and another list for coordinates of lowest
    % left bead center
    bead_offsets = [];
    lowest_left_bead_coords = [];
    for idx =1:length(params.beads_per_cluster)
        if idx==1
            bead_offsets = [bead_offsets; [0,0]];
            lowest_left_bead_coords = [lowest_left_bead_coords, [0,0]];
        else    
            bottom_coords  = [2*(idx-1)*params.inter_bead_spacing_multiplier*params.bead_radius, 0];
            bead_offsets = [bead_offsets; bottom_coords];
            top_coords     = [2*(idx-1)*params.inter_bead_spacing_multiplier*params.bead_radius*cosd(60), 2*(idx-1)*params.inter_bead_spacing_multiplier*params.bead_radius*sind(60)];
            bead_offsets = [bead_offsets; top_coords];
            lowest_left_bead_coords = [lowest_left_bead_coords; [-bottom_coords(1)/2,-top_coords(2)/2]];
            for j=1:idx-2
                curr_coords = [bottom_coords(1)+ j*(top_coords(1)-bottom_coords(1))/(idx-2+1), ...
                    bottom_coords(2)+ j*(top_coords(2)-bottom_coords(2))/(idx-2+1)];
                bead_offsets = [bead_offsets; curr_coords];
            end
        end            
    end    
   
    for idx = 1:length(params.beads_per_cluster)
        curr_image = zeros(params.img_height_input, params.img_width_input);
        curr_beads_per_cluster = params.beads_per_cluster(idx);
        for j=1:curr_beads_per_cluster
            curr_image = curr_image + ...
            ( ( (X-(lowest_left_bead_coords(idx,1)+bead_offsets(j,1))).^2 + ...
                (Y-(lowest_left_bead_coords(idx,2)+bead_offsets(j,2))).^2 ) <= params.bead_radius^2);
        end
        curr_image(curr_image>0)=255;
        
        % Typecast it to uint8
        templates{idx} = uint8(curr_image);
    end
end