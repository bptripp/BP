% playing with alternative bases for PDFs 

x = -5:.01:5;
sigma = .04;
% sigma = .1:.05:.5;
% offset = 1/4;
offset = .1:.1:3;
% offset = .5;

% overlap = zeros(size(sigma));
overlap = zeros(size(offset));
for i = 1:length(offset)
    s = sigma;
    o = offset(i);
    a = cos(2*pi*x) .* exp(-(x-0).^2 / 2 / s);
%     b = cos(2*pi*x+pi/2) .* exp(-(x+o).^2 / 2 / s);
    c = cos(2*pi*(x-o)) .* exp(-(x-o).^2 / 2 / s);
    plot(x, a, 'r', x, c, 'b'), pause
%     plot(x, a, 'r', x, b, 'g', x, c, 'b')
    overlap(i) = sum(a .* c);
%     overlap(i) = sum(a .* b);
end

plot(offset, overlap);
