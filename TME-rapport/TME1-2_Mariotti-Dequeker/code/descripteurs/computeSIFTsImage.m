function t_sift = computeSIFTsImage(I)

    size_patch=16;
    delta_samp = 8;
    r = denseSampling(I, size_patch, delta_samp);
    nb_patches = size(r, 2);
    
    hx = [-1 0 1];
    hy = 1/4*[1; 2; 1];
    Ix = convolution_separable(I, hx, hy);
    Iy = convolution_separable(I, hy', hx');
    Ig = sqrt(Ix.^2 + Iy.^2);
    Ior = orientation(Ix, Iy, Ig);

    Mg = gaussSIFT(size_patch);
    t_sift = [];
    
    for i = 1:nb_patches
        startPatch_x = r(1, i);
        startPatch_y = r(2, i);

        dim_x_Ig = startPatch_x:startPatch_x+size_patch-1;
        dim_y_Ig = startPatch_y:startPatch_y+size_patch-1;
        patch_Ig = Ig(dim_x_Ig, dim_y_Ig);

        dim_x_Ior = startPatch_x:startPatch_x+size_patch-1;
        dim_y_Ior = startPatch_y:startPatch_y+size_patch-1;
        patch_Ior = Ior(dim_x_Ior, dim_y_Ior);

        sift = computeSIFT(size_patch, patch_Ig, patch_Ior, Mg);
        t_sift = [t_sift sift];
    end
    % TODO: Write t_sift for each image in the database
    % --> computeSIFTBase
    % drawPatches(I, r, size_patch, t_sift)

end
