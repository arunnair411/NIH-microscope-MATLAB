function [stddev_val] = calc_stddevs(stddev_list, mean_list)
%CALC_MEANS_STDDEVS Calculate overall standard deviation - simplify life
%   Detailed explanation goes here
    stddev_val = [];
    if ~isempty(stddev_list)
        stddev_val = sqrt( sum(stddev_list.^2+mean_list.^2)/length(stddev_list) - (sum(mean_list)/length(stddev_list))^2);
    end
end