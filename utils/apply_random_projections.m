function [f] = apply_random_projections(params, At, img)
    numcols = params.img_width_output;
    numpats = size(At,1);
    f = zeros(numpats, numcols);
    for i=1:numcols
        curr_A = squeeze(At(:,:,i));
        f(:,i) = curr_A*double(img(:,i));
    end
end