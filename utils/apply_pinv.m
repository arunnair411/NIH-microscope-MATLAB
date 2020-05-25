function [pinv_img] = apply_pinv(params, At, f)
    numcols = params.img_width_output;
    pinv_img = zeros(params.img_height_output, numcols);
    for i=1:numcols
        curr_A = squeeze(At(:,:,i));
        pinv_img(:,i) = pinv(curr_A)*f(:,i);
    %     pinv_img(:,i) = curr_A\f(:,i);
    end
end
