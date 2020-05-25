function [transposed_img] = apply_transpose(params, At, f)
    numcols = params.img_width_output;
    transposed_img = zeros(params.img_height_output, numcols);
    for i=1:numcols
        curr_A = squeeze(At(:,:,i));
        transposed_img(:,i) = curr_A'*f(:,i);
    end
end
