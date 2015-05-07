% testing how well we can decode submessages 

x = -10:.1:0;

f_i = -5;

f = @(x) exp(x+f_i);
% f = @(x) x;

n = 100;
r = 5;
sg = LIFSpikeGenerator(.001, .002, .02, -1+2*rand(1,n), 100+200*rand(1,n), 0);
p = CosinePopulation([r], sg, 'h');

do = DecodedOrigin('m_i', f, p);
points = -10+10*rand(1,300);
findDecoders(do, points);

origin = addOrigin(p, 'm_i', f, do);
plotDecodedOrigin(p, 'm_i');
