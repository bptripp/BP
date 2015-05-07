% exploring generation of encoders that span many dimensions but with
% spatial focus

[offsetX, offsetY, offsetD] = meshgrid(-4:4, -4:4, -2:2);

distance = (offsetX.^2 + offsetY.^2 + (2*offsetD).^2).^(1/2);
radii = max(0,5-distance)/5;

r = radii(:);
n = 100;
points = Population.genRandomPoints(n, r, 1);

figure
subplot(2,1,1), plot(1:length(r), points, '.')
subplot(2,1,2), plot(1:length(r), r)
