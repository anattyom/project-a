close all; clc; clear all;
% here the talker signal is audio with change in location
% the mvdr steering vector is far field approx (not real RTF) 
%% parameters 
fs = 16e3; %[Hz]
c = 340; %[m/s]
M = 4; % number of microphones
B = 1; % number of partitions for AETF
P = 4; % number of partitions for RETF
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

sig_time = 24; % sec
%% loading RIR's
load("C:\Users\anat\OneDrive - Technion\Notability\Project\first research simulation\RIRs.mat", ...
    "g_a", "g_b", "h_a", "h_b", "doa", "array_radius", "relative_pos_a");
% calibration stage 
%% estimating noise covariance matix 
estimation_length = 15; %[sec] - estimating the noise covariance matrix for both the mic and the background noise

x_loudspeaker = sigma_loudspeaker * randn(1, estimation_length*fs); % for estimation loudpeaker signal is WGN
dt_calib = zeros(M, estimation_length*fs); % signal recived at each mic
for m=1:M
    dt_calib(m, :) = filter(h_a(m, :), 1, x_loudspeaker);
end
 
vt = sigma_v * randn(M, length(dt_calib(1, :)));

yt_est = dt_calib + vt; %input signal at each mic
yf_est = cell(M, 1);
vf = cell(M, 1);
for m=1:M
    yf_est{m} = stft(yt_est(m, :), fs, "Window", window_type, "OverlapLength", ...
        overlap, "FFTLength", window_length);
    vf{m} = stft(vt(m, :), fs, "Window", window_type, "OverlapLength", ...
        overlap, "FFTLength", window_length);
end

phi_noise = est_cov_mat(yf_est, window_length, M);
phi_noise_inv = cell(window_length, 1);
for k=1:window_length
    phi_noise_inv{k} = inv(phi_noise{k});
end

%% estimating the REFT's 
y1_est_mat = create_mat(yf_est{1}, P);
a_hat = cell(M, window_length);
for m=2:M
    %ym_est = create_vec(yf_est{m}, P);
    for k=1:window_length
        a_hat{m,k} = lsqr(y1_est_mat{k}, yf_est{m}(k, :).');
        a_hat{m, k} = conj(a_hat{m, k});
    end
end

% real time stage 
%% creating input signals
x = sigma_loudspeaker * randn(1, sig_time*fs); % loudspeaker signal 
no_echo_indices = [(500*R+1):1:(750*R+1); 
    (2200*R+1):1:(2450*R+1)];
%x(no_echo_indices(1, :)) = 0;
x(no_echo_indices(2, :)) = 0;
v_t = sigma_v * randn(M, length(x));

[audio, f_audio] = audioread('cropped_audio.mp3');
audio_resample = resample(audio, fs, f_audio);
talker_sig_full = zeros(sig_time*fs, 1);
talker_sig_full(1:length(audio_resample)) = audio_resample;
talker_sig_full(1:750*R+1) = 0;
talker_sig_full(1200*R+1:1600*R+1) = 0;
talker_sig = talker_sig_full(1:length(x)).';

d_t = zeros(M, length(x));
s_t = zeros(M, length(x));
y_t = zeros(M, length(x));
for m=1:M
    d_t(m, :) = [filter(h_a(m, :), 1, x(1:round(length(x)/2))), ...
        filter(h_b(m, :), 1, x(round(length(x))/2+1:end))];
    s_t(m, :) = [filter(g_a(m, :), 1, talker_sig(1:round(length(talker_sig)/2))),...
        filter(g_b(m, :), 1, talker_sig(round(length(talker_sig)/2)+1:end))]; 
    y_t(m, :) = v_t(m, :) + d_t(m, :) + s_t(m, :);
end
%%
figure;
subplot(311)
plot(talker_sig);
subplot(312);
plot(s_t(1, :));
subplot(313);
plot(d_t(1, :));

%%
x_f = stft(x, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
talker_sig_f = stft(talker_sig, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);

ha_f = zeros(M, window_length);
ga_f = zeros(M, window_length);
hb_f = zeros(M, window_length);
gb_f = zeros(M, window_length);
h_f = zeros(M, 2*window_length);
g_f = zeros(M, 2*window_length);

d_f = cell(1, M);
s_f = cell(1, M);
v_f = cell(1, M);
y_f = cell(1, M);

for m=1:M
    ha_f(m, :) = fftshift(fft(h_a(m, :), window_length));
    ga_f(m, :) = fftshift(fft(g_a(m, :), window_length));
    hb_f(m, :) = fftshift(fft(h_b(m, :), window_length));
    gb_f(m, :) = fftshift(fft(g_b(m, :), window_length));

    h_f(m, :) = [ha_f(m, :), hb_f(m, :)];
    g_f(m, :) = [ga_f(m, :), gb_f(m, :)];

    d_f{m} = stft(d_t(m, :), fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
    s_f{m} = stft(s_t(m, :), fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
    y_f{m} = stft(y_t(m, :), fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
    v_f{m} = stft(v_t(m, :), fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);

end

%% estimating the AETF for refrence mic (1)
h1_est = 0 * ones(size(x_f));
for k=1:window_length
    for n=2:size(x_f, 2)
        h1_est(k, n) = grad_descent_step(y_f{1}(k, n-1), x_f(k, n-1), h1_est(k, n-1));
    end
end
h1_est = conj(h1_est);
d1_est = h1_est .* x_f;

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

%% AEC - echo cancelling for each mic 
e_f = cell(M, 1);
u_f = cell(M, 1);
e_f{1} = y_f{1} - d1_est;
u_f{1} = d_f{1} - d1_est;
d1_est_vec = create_vec(d1_est, P);
for m=2:M
    dm_est = zeros(size(d1_est));
    for n=1:size(d1_est, 2)
        for k=1:window_length
            dm_est(k, n) = a_hat{m, k}' * d1_est_vec{k, n};
        end
    end
    e_f{m} = y_f{m} - dm_est;
    u_f{m} = d_f{m} - dm_est;
end

%% plotting fig 5
u_t = zeros(size(d_t));
e_t = zeros(size(d_t));
for m=1:M
    u_t(m, :) = istft(u_f{m}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
    e_t(m, :) = istft(e_f{m}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
end

l = 0:1:(sig_time*fs/R - 1);
indices = R*l + 1;
y2_plot = y_t(2, indices);
e2_plot = e_t(2, indices);
u2_plot = u_t(2, indices);
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
erle = zeros(M, length(u_t(1, :)) - R);
for l=0:length(erle(1, :))/R-1
    for m=1:M
        d_vec = d_t(m, (l*R+1):(R*l+R));
        u_vec = u_t(m, (l*R+1):(R*l+R));

        tmp = norm(d_vec)^2 / norm(u_vec)^2;
        erle(m, l+1) = 10*log10(tmp);
    end
end

figure;
subplot(221)
plot(erle(1, :));
xlim([1, 1500]);
ylabel("$erle [dB]$", 'Interpreter','latex');
xlabel("$lR$", 'Interpreter','latex');
grid on;
legend("n=1", Location="northwest");
subplot(222)
plot(erle(2, :));
xlim([1, 1500]);
ylabel("$erle [dB]$", 'Interpreter','latex');
xlabel("$lR$", 'Interpreter','latex');
grid on;
legend("n=2", Location="northwest");
subplot(223)
plot(erle(3, :));
xlim([1, 1500]);
ylabel("$erle [dB]$", 'Interpreter','latex');
xlabel("$lR$", 'Interpreter','latex');
grid on;
legend("n=3", Location="northwest");
subplot(224)
plot(erle(4, :));
xlim([1, 1500]);
ylabel("$erle [dB]$", 'Interpreter','latex');
xlabel("$lR$", 'Interpreter','latex');
grid on;
legend("n=4", Location="northwest");
%% using MVDR beamformer
h_mvdr = zeros(M, window_length);

% creating steering vector 
sv = zeros(M, window_length);
for f_idx = 1:window_length
    freq = ((f_idx-1)/window_length - 0.5) * fs;
    %sv(:, f_idx) = steering_vector(M, array_radius, doa, freq, c);
    sv(:, f_idx) = steering_vector(M, doa, freq, c, relative_pos_a);
end

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
plot(real(e_t(1, indices)));
xlabel("$lR$", 'Interpreter', 'latex');
hold on;
plot(real(output(indices)));
hold on;
plot(real(residual_echo(indices)));
hold on;
legend("$e_1(lR)$", "$\hat{s}_1(lR)$", "$\bar{u}(lR)$", 'Interpreter', 'latex');
hold off;

%% erle after mvdr
residual_echo = istft(residual_echo_f, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);

%final_erle = d_t(1, :).^2 ./ abs(residual_echo').^2;
%residual_echo_lpf = filter(avg_filter, 1, abs(residual_echo).^2);
%final_erle = d_t_lpf ./ residual_echo_lpf.';

%figure;
% subplot(211);
% plot(abs(residual_echo));
% subplot(212);
plot(10*log10(final_erle));
xlabel("l")
ylabel("ERLE [dB]")

%%
e2_t = istft(e_f{2}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
figure;
plot(y_t(2, :));
xlabel("l");
ylabel("amplitude")
hold on;
plot(e2_t);
hold on;
plot(u_t(2, :));
hold off;

%% 
e1_t = istft(e_f{1}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);

figure;
plot(e1_t);
xlabel("l");
ylabel("amplitude");
hold on;
plot(real(output));
hold on;
plot(real(residual_echo));
hold off;

%% saving audio
audiowrite('output.wav', real(output), fs);

%%
test = [1 2 3 4 5 6 7 8 9 10];
test_vec = create_vec(test, 3);
test_mat = create_mat(test, 3);
for i=1:size(test_vec, 2)
    test_vec{i}
end
test_mat{1}
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
function h_next = grad_descent_step(y1, x, h)
    step = 1e-2; 
    % calculating error signal
    E = y1 - conj(h)*x;
    % calculating next step
    h_next = h + step*x*conj(E);
end

function a = steering_vector(M, doa, freq, c, rel_pos)
    phi_rad = deg2rad(doa(1));
    theta_rad = deg2rad(doa(2));
    d_hat = [cos(theta_rad)*cos(phi_rad); 
        cos(theta_rad)*sin(phi_rad); 
        sin(theta_rad)];
    lambda = c / freq;
    k = 2 * pi / lambda;
    phase_delay = k * (rel_pos(2:end, :) * d_hat);
    a = ones(M, 1);
    a(2:end) = exp(-1j*phase_delay);
end
