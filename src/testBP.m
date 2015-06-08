% testing modifications of BP for neurons

% clear classes 

ideal = zeros(10,10,'int32');
ideal(:,1:5) = 2;
ideal(:,6:end) = 8;
im = ideal + int32(1*randn(size(ideal)));
mesh(im);

levels = int32(1:10);

bp = NeuralBP(im, levels, 1, .2);
steps = 400;
time = (1:steps)*bp.dt;
results = zeros(size(im,1),size(im,2),steps);
figure(1), set(gcf, 'Position', [229 373 1012 420])
for i = 1:steps
    time(i)-bp.dt
    tic, bp.iterate(time(i)-bp.dt); toc
%     result = getMAP(bp);
%     mesh(result)
    subplot(1,2,1), mesh(getMAP(bp))
    subplot(1,2,2), mesh(getProbeMAP(bp))
    pause(.01)
    results(:,:,i) = getProbeMAP(bp);
%     imagesc(result), pause(.01)
end

figure
subplot(2,2,1), mesh(ideal), title('ideal'), set(gca, 'ZLim', [-2 12])
subplot(2,2,2), mesh(im), title('input'), set(gca, 'ZLim', [-2 12])
subplot(2,2,3), mesh(mean(results(:,:,1:5), 3)), title('early'), set(gca, 'ZLim', [-2 12])
subplot(2,2,4), mesh(mean(results(:,:,end-4:end), 3)), title('late'), set(gca, 'ZLim', [-2 12])
