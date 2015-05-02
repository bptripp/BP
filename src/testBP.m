% testing modifications of BP for neurons

im = zeros(10,10,'int32');
im(:,1:5) = 50;
im(:,6:10) = 100;
im = im + int32(10*randn(size(im)));
imagesc(im);

levels = int32(1:128);
bp = BP(im, levels);
for i = 1:30
    tic, bp.iterate(); toc
    result = getMAP(bp);
    figure(1), set(gcf, 'Position', [217   374   560   420])
    mesh(result), pause(.01)
%     imagesc(result), pause(.01)
end

