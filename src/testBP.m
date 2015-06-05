% testing modifications of BP for neurons

ideal = zeros(10,10,'int32');
ideal(:,1:5) = 50;
ideal(:,6:end) = 100;
im = ideal + int32(10*randn(size(ideal)));
mesh(im);

levels = int32(1:128);
bp = SumProductBP(im, levels, 10, 5);
steps = 300;
results = zeros(size(im,1),size(im,2),steps);
for i = 1:steps
    tic, bp.iterate(); toc
    result = getMAP(bp);
    figure(1), set(gcf, 'Position', [217   374   560   420])
    mesh(result), pause(.01)
    results(:,:,i) = result;
%     imagesc(result), pause(.01)
end

figure
subplot(2,2,1), mesh(ideal), title('ideal'), set(gca, 'ZLim', [20 140])
subplot(2,2,2), mesh(im), title('input'), set(gca, 'ZLim', [20 140])
subplot(2,2,3), mesh(mean(results(:,:,1:5), 3)), title('early'), set(gca, 'ZLim', [20 140])
subplot(2,2,4), mesh(mean(results(:,:,end-4:end), 3)), title('late'), set(gca, 'ZLim', [20 140])
