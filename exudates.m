clc;
clear all;

I = imread('image019.png');
res_img = imresize(I,[256 256]);
h = fspecial('unsharp');
filt_img = imfilter(res_img(:,:,2),h);
h = fspecial('disk',0.8);
filt_img = imfilter(filt_img,h);

adjst_img = imadjust(filt_img);
A = adapthisteq(adjst_img,'clipLimit',0.02,'Distribution','rayleigh');
SE = strel('disk',1);
%afterOpening = imopen(A,SE);
BW2 = imerode(A,SE);
BW3 = imdilate(BW2,SE);
% GLCM = graycomatrix(filt_img);
% stats = GLCM_Features4(GLCM,0)
% stats.contr
% stats. energ
% stats. homom
% greenChannel = I(:, :, 2);
% I2 = imcrop(greenChannel);
% I3 = imresize(I2, 3);
% img = imadjust(filt_img);
% G = fspecial('gaussian',[5 5],2);
% Ig = imfilter(img,G,'same');

[mu, mask] = adaptcluster_kmeans(BW3);
% figure,imshow(mu==2,[]);
% figure,imshow(mu==3,[]);
% figure,imshow(mu==4,[]);
% figure,imshow(I); 

b = uint8(mu == 1);
b1 = b*255;
% GLCM2 = graycomatrix(b1,'Offset',[2 0;0 2]); 
% stats = graycoprops(GLCM2,{'contrast','homogeneity'})
% stats = graycoprops(b1)
% b2 = bwareaopen(b1, 1);
% GLCM = graycomatrix(filt_img);
% stats = GLCM_Features4(GLCM,0);
% whtpixl = nnz(b1)
% stats.contr
% stats. energ
% stats. homom

maxImage = imregionalmax(b1);
maxImage = imdilate(maxImage,strel('disk',0));

% BWfill = imfill(maxImage, 'holes');
% holes = BWfill &~ maxImage; 

% [B,L,N] = bwboundaries(maxImage);
% figure; imshow(maxImage); hold on;
% for k=1:length(B),
%     boundary = B{k};
%     if(k > N)
%         plot(boundary(:,2),...
%             boundary(:,1),'g','LineWidth',2);
%     else
%         plot(boundary(:,2),...
%             boundary(:,1),'r','LineWidth',2);
%     end
% end

% BW = edge(b1,'canny');
% figure, imshow(BW), title('IMG');
% Rmin = 1;
% Rmax = 5;
% 
% % Find all the bright circles in the image
% [centersBright, radiiBright] = imfindcircles(BW,[Rmin Rmax], ...
%                             'ObjectPolarity','bright');
% viscircles(centersBright, radiiBright,'EdgeColor','b');

binaryImage = maxImage - bwareaopen(maxImage, 350);
refill = imfill(binaryImage,'holes');
L = medfilt2(refill,[4 4]);
% figure, imshow(b1), title('b1 IMG');
figure, imshow(L), title('Classification for object one');

L1 = bwlabel(b1);
stats1 = regionprops(L1, 'Centroid', 'Area', 'BoundingBox');
area_values1 = [stats1.Area];
sprintf('The area of class object one is %d', area_values1);

% c = uint8(mu == 2);
% c1 = c*255;
% figure, imshow(c1), title('Classification for object two');
% 
% L2 = bwlabel(c1);
% stats2 = regionprops(L2, 'Centroid', 'Area', 'BoundingBox');
% area_values2 = [stats2.Area];
% sprintf('The area of class object one is %d', area_values2);
% 
% g = uint8(mu == 3);
% g1 = g*255;
% figure, imshow(g1), title('Classification for object three');
% 
% L3 = bwlabel(g1);
% stats1 = regionprops(L3, 'Centroid', 'Area', 'BoundingBox');
% area_values1 = [stats1.Area];
% sprintf('The area of class object one is %d', area_values1);

h = uint8(mu == 4);
h1 = h*255;
% GLCM3 = graycomatrix(h1,'Offset',[2 0;0 2]); 
% stats3 = graycoprops(GLCM3,{'contrast','homogeneity'})
stats = graycoprops(h1);
% whtpixl_ex = nnz(h1)
figure, imshow(h1), title('Classification for object four');

L4 = bwlabel(h1);
stats1 = regionprops(L4, 'Centroid', 'Area', 'BoundingBox');
area_values1 = [stats1.Area];
sprintf('The area of class object one is %d', area_values1);
