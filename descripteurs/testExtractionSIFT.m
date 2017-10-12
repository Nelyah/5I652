% I = marche()
% I = imread('pictures/tools.gif')
I = imread('pictures/Scene/MITmountain/image_0363.jpg');
size_patch=16;

startPatch_x = 173
startPatch_y = 125

hx = [-1 0 1]
hy = 1/4*[1; 2; 1]
Ix = convolution_separable(I,hx,hy)
Iy = convolution_separable(I,hy',hx')

Ig = sqrt(Ix.^2+Iy.^2)
Ior = orientation(Ix,Iy,Ig)
Mg = gaussSIFT(size_patch)


topleft_x_Ig = startPatch_x+1:startPatch_x+size_patch
topleft_y_Ig = startPatch_y+1:startPatch_y+size_patch
patch_Ig = Ig(topleft_x_Ig,topleft_y_Ig)
   
topleft_x_Ior = startPatch_x+1:startPatch_x+size_patch
topleft_y_Ior = startPatch_y+1:startPatch_y+size_patch
patch_Ior = Ior(topleft_x_Ior,topleft_y_Ior)

sift=computeSIFT(size_patch,patch_Ig,patch_Ior,Mg)


%t_sifts = computeSIFTsImage(I);
visuSIFT( I, Ig, Ior, [startPatch_x;startPatch_y] , 'test', size_patch, sift )
