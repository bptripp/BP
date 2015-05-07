% load Middlebury images

im0 = imread('~/code/BP/data/middlebury-adirondack/im0.png');
im1 = imread('~/code/BP/data/middlebury-adirondack/im1.png');

cropped0 = im0(501:1500, 251:1750);
cropped1 = im1(501:1500, 251:1750);

down0 = cropped0(5:10:end,5:10:end);
down1 = cropped1(5:10:end,5:10:end);

figure
subplot(2,1,1), imshow(down0)
subplot(2,1,2), imshow(down1)
