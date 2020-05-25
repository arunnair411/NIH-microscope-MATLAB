% Need to run process_experimental_data
sim_mat_file_names = {'/data/Dropbox/Arun/Research/NIH-Imaging/microscope/DNN/email_data/Sim1/019994.mat', ...
    '/data/Dropbox/Arun/Research/NIH-Imaging/microscope/DNN/email_data/Sim2/019994.mat', ...
    '/data/Dropbox/Arun/Research/NIH-Imaging/microscope/DNN/email_data/Sim3/019994.mat'
    };
real_mat_file_names = {'/data/Dropbox/Arun/Research/NIH-Imaging/microscope/DNN/email_data/Sim1/077777.mat', ...
    '/data/Dropbox/Arun/Research/NIH-Imaging/microscope/DNN/email_data/Sim2/077777.mat', ...
    '/data/Dropbox/Arun/Research/NIH-Imaging/microscope/DNN/email_data/Sim3/077777.mat'
    };

real_optimization_mat_file_name = '/data/Dropbox/Arun/Research/NIH-Imaging/microscope/DNN/email_data/optimization_data.mat';
load(real_optimization_mat_file_name);
real_optimization_data{1} = Xr_padded_20;
real_optimization_data{2} = Xr_padded_10;
real_optimization_data{3} = Xr_padded_05;

sampling_percentages = {20, 10, 5};
for idx = 1:3
    figure (idx);
    load(sim_mat_file_names{idx});
    subplot(232);
    imagesc(squeeze(target_output)); colormap gray;
    title(sprintf('Target Image (Simulated)'));
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]); set(gca,'FontSize',16);

%     subplot(232);
%     imagesc(squeeze(Xr_padded)); colormap gray;
%     title(sprintf('Optimization Output \n (Simulated, %d%% sampling)', sampling_percentages{idx}));
%     set(gca,'XTickLabel',[]);
%     set(gca,'YTickLabel',[]);
%     set(gca,'FontSize',16);

    subplot(233);
    imagesc(squeeze(DNN_output)); colormap gray;
    title(sprintf('DNN Output (Simulated, \n %d%% sampling)', sampling_percentages{idx}));
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'FontSize',16);

    load(real_mat_file_names{idx});
    subplot(234);
    imagesc(squeeze(Xr_padded_200)); colormap gray;
    title(sprintf('Target Image (Experimental, \n 200%% sampling)'));
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'FontSize',16);
    
    subplot(235);
    imagesc(squeeze(real_optimization_data{idx})); colormap gray;
    title(sprintf('Optimization Output \n (Experimental, %d%% sampling)', sampling_percentages{idx}));
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'FontSize',16);

    subplot(236);
    imagesc(squeeze(DNN_output)); colormap gray;
    title(sprintf('DNN Output (Experimental, \n %d%% sampling)', sampling_percentages{idx}));
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'FontSize',16);
end