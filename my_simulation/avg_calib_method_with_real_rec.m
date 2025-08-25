close all; clc; clear all;
%% parameters 
fs = 16e3; %[Hz]
c = 340; %[m/s]
M = 6; % number of microphones
ref_mic = 4;
B = 19; % number of partitions for AETF
P = 1; % number of partitions for RETF
R = 128;
B_nc = 3;
mu = 0.1; % parameter for AETF gradient descent 
beta = exp(-R / (0.075*fs)); % forgetting factor 
epsilon = 1e-3; % reg for AEC 

window_length = 512;
overlap = 0.75*window_length; % 75% overlap in stft
window_type = hamming(window_length);

sigma_loudspeaker = sqrt(0.5*0.05);
sigma_v = sqrt(0.5*0.00005); % thermal noise

% calibration stage 
%% estimating noise covariance matix averaging over head orientations
% [audio, f_audio] = audioread("fb_signals_front.wav");
% audio_resample = resample(audio, fs, f_audio);
% yt_est = audio_resample.';

calib_time = 10; % [sec] 
orientation={'front' 'down' 'left'  'right'  'up'};  % Head orientations
num_ort = 5;
yt_est = zeros(num_ort, M, calib_time*fs);
for ort_idx=1:num_ort
    % creating feedback signal at each mic 
    sname=['fb_signals_' orientation{ort_idx} '.wav'];  % FB signals
    [tmp, f_audio] = audioread(sname);
    tmp_resample = resample(tmp, fs, f_audio);
    tmp_resample = tmp_resample(1:calib_time*fs, :);
    yt_est(ort_idx,:,:) = tmp_resample.';
end

[audio, f_audio] = audioread("desired_signals.wav");
audio_resample = resample(audio, fs, f_audio);
st_est = audio_resample.';

yf_est = cell(M, num_ort);
sf_est = cell(M,1);
tmp1 = zeros(M, calib_time*fs);
for m=1:M
    for ort_idx=1:num_ort
        tmp1(m,:) = yt_est(ort_idx, m, :);
        yf_est{m,ort_idx} = stft(tmp1(m, :), fs, "Window", window_type, "OverlapLength", ...
            overlap, "FFTLength", window_length);
    end
    sf_est{m} = stft(st_est(m, :), fs, "Window", window_type, "OverlapLength", ...
        overlap, "FFTLength", window_length);
end

phi_noise = cell(window_length, 1);
for k=1:window_length
    phi_noise{k} = zeros(M);
end
%phi_noise_ort = cell(num_ort, window_length);
tmp2 = cell(M, 1);
for ort_idx = 1:num_ort
    for m=1:M
        tmp2{m} = yf_est{m,ort_idx};
    end
    tmp3 = est_cov_mat(tmp2, window_length, M);
    for k=1:window_length
        %phi_noise_ort{ort_idx, k} = tmp3{k};
        phi_noise{k} = phi_noise{k} + tmp3{k};
    end
end
%phi_noise = est_cov_mat(yf_est, window_length, M);
phi_noise_inv = cell(window_length, 1);
for k=1:window_length
    phi_noise{k} = (1/window_length) * phi_noise{k};
    phi_noise_inv{k} = inv(phi_noise{k});
end

%% estimating the REFT's 
a_hat = cell(M, window_length);
for k=1:window_length
    for m=1:M
        a_hat{m, k} = zeros(P, 1);
    end 
end

n_tf = size(yf_est{1, 1}, 2);
for k=1:window_length
    for ort_idx=1:num_ort
        denom = (1/n_tf) * conj(yf_est{ref_mic, ort_idx}(k, :)) * yf_est{ref_mic, ort_idx}(k, :).'; 
        for m=1:M
            cross =  (1/n_tf) * conj(yf_est{ref_mic, ort_idx}(k, :)) * yf_est{m, ort_idx}(k, :).';
            a_hat{m, k} = a_hat{m, k} + (1/num_ort)*(cross / denom);
        end
    end
end
%% estimating steering vector 
load("C:\Users\anat\OneDrive - Technion\Notability\Project\first research simulation\RIRs.mat", ...
    "g_glasses");

tmp = randn(1, size(st_est, 2));
for m=1:M
    st_est(m, :) = filter(g_glasses(m, :), 1, tmp);
    sf_est{m} = stft(st_est(m, :), fs, "Window", window_type, "OverlapLength", ...
        overlap, "FFTLength", window_length);
end
sv = zeros(M, window_length);
for k=1:window_length
    denom = (1/n_tf) * conj(sf_est{ref_mic}(k, :)) * sf_est{ref_mic}(k, :).';
    for m=1:M
        cross =  (1/n_tf) * conj(sf_est{ref_mic}(k, :)) * sf_est{m}(k, :).';
        sv(m, k) = cross / denom; 
    end
end
% real time stage 
%% test - using not recorded audio for s(t)
load("C:\Users\anat\OneDrive - Technion\Notability\Project\first research simulation\RIRs.mat", ...
    "g_glasses");
[audio, f_audio] = audioread('cropped_audio.mp3');
audio_resample = resample(audio, fs, f_audio);
%talker_sig_full = zeros(sig_time*fs, 1);
%talker_sig_full(1:length(audio_resample)) = audio_resample;
%talker_sig_full = randn(1, fs*10);
st_test = zeros(M, length(audio_resample));
for m=1:M
    st_test(m, :) = 0.1 * filter(g_glasses(m, :), 1, audio_resample);
end

%% creating input signals
[x_audio, f_audio] = audioread("CalibrationWhiteNoise.wav");
x_partial = resample(x_audio, fs, f_audio);
orientation={'front' 'down' 'left'  'right'  'up'};  % Head orientations
num_ort = 5;
d_t = [];
s_t = [];
x = [];
for ort_idx=1:num_ort
    % creating feedback signal at each mic 
    sname=['fb_signals_' orientation{ort_idx} '.wav'];  % FB signals
    [tmp, f_audio] = audioread(sname);
    tmp_resample = resample(tmp, fs, f_audio);
    tmp_resample = tmp_resample(1:round(size(tmp_resample, 1)/2), :);
    d_t = [d_t, tmp_resample.'];
    % creating original feedback signal x 
    d_ref = tmp_resample(:, ref_mic).';
    sig_len = length(d_ref);
    [c, lags] = xcorr(d_ref, x_partial, 4500); % cross correlation to align x with the recorded signal
    [~, idx_max] = max(abs(c));
    lag_max = lags(idx_max);
    x_aligned = circshift(x_partial, lag_max);
    x_tiled = repmat(x_aligned, ceil(sig_len/length(x_aligned)), 1);
    x_tiled = x_tiled(1:sig_len);
    x = [x x_tiled.'];
    % creating desired signal recived at each mic
    tmp = zeros(M, R*500);
    s_t = [s_t tmp st_test(:, 1:(length(x_tiled)-size(tmp,2)))];
end

y_t = d_t + s_t;

%%
x_f = stft(x, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);

d_f = cell(1, M);
s_f = cell(1, M);
y_f = cell(1, M);

for m=1:M
    d_f{m} = stft(d_t(m, :), fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
    s_f{m} = stft(s_t(m, :), fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
    y_f{m} = stft(y_t(m, :), fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
end

%% estimating the AETF for refrence mic (4)
h1_est = cell(size(x_f));
x_f_vec = create_vec(x_f, B);
d_est_f = cell(1, M);
for k=1:window_length
    h1_est{k, 1} = zeros(B, 1);
    psi_last = M*eye(B);
    for n=2:size(x_f, 2)
        [step, psi_last] = calc_step_mat(beta, x_f_vec{k, n-1}, psi_last, mu, B, epsilon);
        h1_est{k, n} = grad_descent_step(y_f{ref_mic}(k, n-1), x_f_vec{k, n-1}, h1_est{k, n-1}, step);
    end
end

d_est_f{ref_mic} = zeros(size(x_f));
for k=1:window_length
    for l=1:size(x_f, 2)
        d_est_f{ref_mic}(k, l) = h1_est{k, l}' * x_f_vec{k, l};
    end
end

d1_est_f_vec = create_vec(d_est_f{ref_mic}, P);

% figure;
% subplot(311)
% plot(abs(ha_f(1, :)));
% title("real h1");
% subplot(312)
% plot(abs(h1_est(:, end)));
% title("estimated h1");
% subplot(313)
% plot(abs(ha_f(1, :) - h1_est(:, end).'));
% title("diffrence between real and estimation");

%% estimating echo for rest of mics
for m=1:M
    if m == ref_mic
        continue
    end
    d_est_f{m} = zeros(size(x_f));
    for k=1:window_length
        for l=1:size(x_f, 2)
            d_est_f{m}(k, l) = a_hat{m, k}.' * d1_est_f_vec{k, l};
        end
    end
end

%% AEC - echo cancelling for each mic 
e_f = cell(M, 1);
u_f = cell(M, 1);
for m=1:M
    e_f{m} = y_f{m} - d_est_f{m};
    u_f{m} = d_f{m} - d_est_f{m};
end

%% plotting fig 5
u_t = cell(1, M);
e_t = cell(1, M);
for m=1:M
    u_t{m} = istft(u_f{m}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
    e_t{m} = istft(e_f{m}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
end

l = 0:1:(length(y_t(2, :))/R - 1);
indices = R*l + 1;
y2_plot = y_t(2, indices);
e2_plot = e_t{2}(indices);
u2_plot = u_t{2}(indices);
figure;
plot(y2_plot);
%ylim([-1, 1]);
xlabel("$lR$", 'Interpreter', 'latex');
hold on;
plot(e2_plot);
hold on;
plot(u2_plot);
legend("$y_2(lR)$", "$e_2(lR)$", "$u_2(lR)$", 'Interpreter', 'latex');
hold off;
%% calculating erle after AEC
erle = zeros(M, length(u_t{1})/R);
for l=0:length(erle(1, :))-1
    for m=1:M
        d_vec = d_t(m, (l*R+1):(R*l+R));
        u_vec = u_t{m}((l*R+1):(R*l+R));

        tmp = norm(d_vec)^2 / norm(u_vec)^2;
        erle(m, l+1) = 10*log10(tmp);
    end
end

%%
seg_len = round(size(tmp_resample, 1)/R);
i = 0;
for idx=1:num_ort
    segs(idx) = i+seg_len;
    i = segs(idx);
end

figure;
subplot(321)
plot(erle(1, :));
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("ERLE [dB]");
ylim([-10, 25]);
xlim([0, length(erle(4, :))]);
grid on;
hold on;
for idx = 1:num_ort
    xline(segs(idx), "k--", 'LineWidth', 1);
end
legend("n=1", Location="northwest");
subplot(322)
plot(erle(2, :));
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("ERLE [dB]");
ylim([-10, 25]);
xlim([0, length(erle(4, :))]);
grid on;
hold on;
for idx = 1:num_ort
    xline(segs(idx), "k--", 'LineWidth', 1);
end
legend("n=2", Location="northwest");
subplot(323)
plot(erle(3, :));
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("ERLE [dB]");
ylim([-10, 25]);
xlim([0, length(erle(4, :))]);
grid on;
hold on;
for idx = 1:num_ort
    xline(segs(idx), "k--", 'LineWidth', 1);
end
legend("n=3", Location="northwest");
subplot(324)
plot(erle(4, :));
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("ERLE [dB]");
ylim([-10, 25]);
xlim([0, length(erle(4, :))]);
grid on;
hold on;
for idx = 1:num_ort
    xline(segs(idx), "k--", 'LineWidth', 1);
end
legend("n=4", Location="northwest");
subplot(325)
plot(erle(5, :));
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("ERLE [dB]");
ylim([-10, 25]);
xlim([0, length(erle(4, :))]);
grid on;
hold on;
for idx = 1:num_ort
    xline(segs(idx), "k--", 'LineWidth', 1);
end
legend("n=5", Location="northwest");
subplot(326)
plot(erle(6, :));
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("ERLE [dB]");
ylim([-10, 25]);
xlim([0, length(erle(4, :))]);
grid on;
hold on;
for idx = 1:num_ort
    xline(segs(idx), "k--", 'LineWidth', 1);
end
legend("n=6", Location="northwest");

%%
figure;
subplot(211);
plot(erle(4, :));
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("ERLE [dB]");
ylim([-10, 25]);
xlim([0, length(erle(4, :))]);
grid on;
hold on;
for idx = 1:num_ort
    xline(segs(idx), "k--", 'LineWidth', 1);
end
legend("n=4", Location="northwest");
subplot(212)
plot(erle(3, :));
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("ERLE [dB]");
ylim([-10, 25]);
xlim([0, length(erle(4, :))]);
grid on;
hold on;
for idx = 1:num_ort
    xline(segs(idx), "k--", 'LineWidth', 1);
end
legend("n=3", Location="northwest");
%% using MVDR beamformer
h_mvdr = zeros(M, window_length);

for f_idx=1:window_length
    denominator = sv(:, f_idx)' * phi_noise_inv{f_idx} * sv(:, f_idx);
    numerator = phi_noise_inv{f_idx} * sv(:, f_idx);
    h_mvdr(:, f_idx) = numerator / denominator;
end

% making e_vec 
e_f_vec = cell(size(e_f{1}, 2), 1);
u_f_vec = cell(size(u_f{1}, 2), 1);
for l=1:size(e_f{1}, 2)
    e_f_vec{l} = zeros(M, window_length);
    u_f_vec{l} = zeros(M, window_length);
    for m=1:M
        e_f_vec{l}(m ,:) = e_f{m}(:, l);
        u_f_vec{l}(m ,:) = u_f{m}(:, l);
    end
end

% applying beamformer
output_f = zeros(size(e_f{1}));
residual_echo_f = zeros(size(u_f{1}));
for l=1:size(e_f{1}, 2)
    tmp_mat = h_mvdr' * e_f_vec{l};
    output_f(:, l) = diag(tmp_mat);

    tmp_mat = h_mvdr' * u_f_vec{l};
    residual_echo_f(:, l) = diag(tmp_mat);
end
%%
figure;
subplot(311)
plot(abs(istft(e_f{1}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length)));
title("mic 3 output after AEC");
subplot(312)
plot(abs(istft(output_f, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length)));
title("MVDR output");
subplot(313)
plot(abs(istft(y_f{1}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length)));
title("input in mic 1");
%%
output = istft(output_f, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
residual_echo = istft(residual_echo_f, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
s_t_mic1 = istft(s_f{1}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
y_t_mic1 = istft(y_f{1}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);

figure;
plot(real(e_t{4}(indices)));
xlabel("$lR$", 'Interpreter', 'latex');
ylim([-0.2, 0.2])
hold on;
plot(real(output(indices)));
hold on;
plot(real(residual_echo(indices)));
hold on;
legend("$e_4(lR)$", "$\hat{s}_4(lR)$", "$\bar{u}(lR)$", 'Interpreter', 'latex');
hold off;

%% erle after mvdr
final_erle = zeros(1, length(output)/R);
for l=0:length(final_erle)-1
    d_vec = d_t(1, (l*R+1):(R*l+R));
    residual_echo_vec = residual_echo((l*R+1):(R*l+R));
    tmp = norm(d_vec)^2 / norm(residual_echo_vec)^2;
    final_erle(l+1) = 10*log10(tmp);
end

seg_len = round(size(tmp_resample, 1)/R);
i = 0;
for idx=1:num_ort
    segs(idx) = i+seg_len;
    i = segs(idx);
end
fig = figure;
plot(final_erle);
xlabel("$l$", 'Interpreter', "latex")
ylabel("ERLE [dB]");
ylim([-5, 45]);
xlim([0, segs(num_ort)]);
grid on;
hold on;
for idx = 1:num_ort
    xline(segs(idx), "k--", 'LineWidth', 1);
end
exportgraphics(fig, 'final_erle.pdf', 'ContentType', 'vector');

%% calculating average final erle for each orientation
avg_final_erle = zeros(1, num_ort);
converaging_time = 500;
seg_end = round(size(tmp_resample, 1)/R);
i = 0;
for idx=1:num_ort
    start_idx = i+converaging_time;
    end_idx = i+seg_end;
    avg_final_erle(idx) = mean(final_erle(start_idx:end_idx));
    i = i + seg_end;
end
T = array2table(avg_final_erle, 'VariableNames', orientation);
disp(T)
%% saving audio
audiowrite('ref_mic_imput.wav', y_t_mic1, fs);
audiowrite('output.wav', real(output), fs);

%% functions 
function phi = est_cov_mat(x, window_length, M)    
    % creating cell for each frequency of X
    num_segments = length(x{1}(1, :));
    phi = cell(window_length, 1);
    x_freq_mat = cell(window_length, 1);
    for k=1:window_length
        x_freq_mat{k} = zeros(M, num_segments);
        for m=1:M
            x_freq_mat{k}(m, :) = x{m}(k, :);
        end
        phi{k} = zeros(M, M);
        for n=1:num_segments
            phi{k} = phi{k} + x_freq_mat{k}(:, n)*x_freq_mat{k}(:, n)';
        end
        phi{k} = (1/num_segments)*phi{k};
    end
end

% function for creating vector for convolution
function x_vec = create_vec(x, vec_len)
    x_vec = cell(size(x));
    for k=1:size(x, 1)
        for l=1:size(x,2)
            vec = zeros(vec_len, 1);
            for b=0:vec_len-1
                if l-b >= 1
                    vec(b+1) = x(k, l-b);
                end
            end
            x_vec{k, l} = vec;
        end
    end
end

function x_mat = create_mat(x, vec_len)
    x_mat = cell(1, size(x, 1));
    x_vec = create_vec(x, vec_len);
    for k=1:size(x, 1)
        x_mat{k} = zeros(size(x, 2), vec_len);
        for l=1:size(x, 2)
            x_mat{k}(l, :) = x_vec{k, l};
        end
    end
end

% input: y1 - the last frame input at mic 1
%        x - the last frame loudspeaker signal
%        h - the last estimation for the AETF
function h_next = grad_descent_step(y1, x, h, step)
    % calculating error signal
    E = y1 - h'*x;
    % calculating next step
    h_next = h + step*x*conj(E);
end

function [M, psi] = calc_step_mat(beta, x, psi_prev, mu, b, epsilon)
    xxH = x*x';
    Ib = eye(b);
    psi = beta*psi_prev + (1-beta)*(Ib .* xxH);
    psi_reg = psi + epsilon*eye(size(psi));

    M = (mu / b) * (psi_reg \ eye(size(psi)));
end

function y_ransac = my_ransac(n, x)
    % RANSAC parameters
    numIter = 100;       % number of iterations
    threshold = 1.5;     % inlier distance threshold
    bestInliers = [];
    bestModel = [];
    
    for i = 1:numIter
        % 1. Randomly pick 2 points
        idx = randperm(length(n), 2);
        x1 = n(idx(1)); 
        y1 = x(idx(1));
        x2 = n(idx(2)); 
        y2 = x(idx(2));
        
        % Skip if vertical line
        if x1 == x2, continue; end
        
        % 2. Fit line through them: y = a*x + b
        a = (y2 - y1) / (x2 - x1);
        b = y1 - a*x1;
        
        % 3. Compute distances of all points to line
        y_est = a*n + b;
        dist = abs(x - y_est);
        
        % 4. Count inliers
        inliers = find(dist < threshold);
        
        % 5. Update best model
        if length(inliers) > length(bestInliers)
            bestInliers = inliers;
            bestModel = [a b];
        end
    end
    
    % Final line using best model
    a = bestModel(1);
    b = bestModel(2);
    y_ransac = a*n + b;
end
