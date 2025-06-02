close all; clc; clear all; 
%% parameters
fs = 16e3; %[Hz]
c = 340; %[m/s]
M = 6; % number of microphones
ref_mic = 4; % refrence mic 
B = 19; % number of partitions of AETF
P = 4; % number of partitions of RETF
rho = B / P; % overestimation factor 
R = 128;
B_nc = 3;
mu = 0.1; %parameter for AETF gradient decsent 
eta = 0.09; %parameter for RETF gradient decsent
beta = exp(-R/(0.075*fs)); % forgetting factor
alpha = exp(-R/(0.04*fs)); % forgetting factor for residual echo

epsilon = 1e-3; % reg for AEC

window_length = 512;
overlap = 0.75*window_length; % 75% overlap in stft
window_type = hamming(window_length);

sigma_loudspeaker = sqrt(0.5*0.05);
sigma_v = sqrt(0.5*0.00005); % thermal noise

sig_time = 24; % sec

%% estimating steering vector  
[x_talker, f_audio] = audioread("desired_signals.wav");
x_talker_resample = resample(x_talker, fs, f_audio);
sf_est = cell(M, 1);
for m=1:M
    sf_est{m} = stft(x_talker_resample(:, m), fs, "Window", window_type, "OverlapLength", ...
        overlap, "FFTLength", window_length);
end

% estimating sv
sv = zeros(M, window_length);
n_tf = size(sf_est{ref_mic}, 2);
for k=1:window_length
    denom = (1/n_tf) * conj(sf_est{ref_mic}(k, :)) * sf_est{ref_mic}(k, :).';
    for m=1:M
        cross =  (1/n_tf) * conj(sf_est{ref_mic}(k, :)) * sf_est{m}(k, :).';
        sv(m, k) = cross / denom; 
    end
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
    tmp_resample = tmp_resample(1:round(size(tmp_resample, 1)/6), :);
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
    s_t = [s_t x_talker_resample(1:length(x_tiled), :).'];
end


y_t = zeros(M, length(d_t));
for m=1:M
    y_t(m, :) = d_t(m, :);% + s_t(m, :);
end

figure;
subplot(311)
plot(s_t(1, :));
title("s_1(t)");
subplot(312)
plot(d_t(1, :))
title("d_1(t)")
subplot(313)
plot(y_t(1, :));
title("y_1(t)")

x_f = stft(x, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
x_f_vec = create_vec(x_f, B);

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

%% estimating AETF for ref mic
h1_est = cell(size(x_f));
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

%% estimating RETF
a_est = cell(size(x_f));
for m=1:M
    if m == ref_mic
        continue
    end
    test = m
    for k=1:window_length
        a_est{k, 1} = zeros(P, 1);
        psi_prev_retf = M*eye(P);
        for n=2:size(x_f, 2)
            [step_retf, psi_prev_retf] = calc_step_mat_retf(beta, d1_est_f_vec{k, n-1}, psi_prev_retf, eta, P, epsilon);
            a_est{k, n} = grad_descent_step(y_f{m}(k, n-1), d1_est_f_vec{k, n-1}, a_est{k, n-1}, step_retf);
        end
    end
    
    d_est_f{m} = zeros(size(x_f));
    for k=1:window_length
        for l=1:size(x_f, 2)
            d_est_f{m}(k, l) = a_est{k, l}' * d1_est_f_vec{k, l};
        end
    end
end

%% echo cancelling
e_f = cell(1, M);
u_f = cell(1, M);
for m=1:M
    e_f{m} = y_f{m} - d_est_f{m};
    u_f{m} = d_f{m} - d_est_f{m};
end

e_t = cell(1, M);
u_t = cell(1, M);
for m=1:M
    e_t{m} = istft(e_f{m}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
    u_t{m} = istft(u_f{m}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
end

figure;
subplot(211)
plot(e_t{3})
title("e3")
ylim([-0.5, 0.5])
subplot(212)
plot(e_t{1})
title("e1")
ylim([-0.5, 0.5])

%% calculating erle
erle = zeros(M, length(u_t{1})/R);
for l=0:length(erle(1, :))-1
    for m=1:M
        d_vec = d_t(m, (l*R+1):(R*l+R));
        u_vec = u_t{m}((l*R+1):(R*l+R));

        tmp = norm(d_vec)^2 / norm(u_vec)^2;
        erle(m, l+1) = 10*log10(tmp);
    end
end

figure;
subplot(221)
plot(erle(1, :));
%ylim([-30, 40]);
%xlim([1, 1500]);
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("$erle [dB]$", 'Interpreter', 'latex');
grid on;
legend("n=1", Location="northwest");
subplot(222)
plot(erle(2, :));
%ylim([-30, 40]);
%xlim([1, 1500]);
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("$erle [dB]$", 'Interpreter', 'latex');
grid on;
legend("n=2", Location="northwest");
subplot(223)
plot(erle(3, :));
%ylim([-30, 40]);
%xlim([1, 1500]);
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("$erle [dB]$", 'Interpreter', 'latex');
grid on;
legend("n=3", Location="northwest");
subplot(224)
plot(erle(4, :));
%ylim([-30, 40]);
%xlim([1, 1500]);
xlabel("$lR$", 'Interpreter', 'latex');
ylabel("$erle [dB]$", 'Interpreter', 'latex');
grid on;
legend("n=4", Location="northwest");

%% plotting fig 5 time graph
l = 0:1:(sig_time*fs/R - 1);
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

%% residual echo estimation
% refrence mic - using AETF
delta_h1 = cell(size(x_f));
u_hat = cell(1, M);
u_hat{ref_mic} = zeros(size(x_f));
for k=1:window_length
    delta_h1{k, 1} = zeros(B, 1);
    psi_last = M*eye(B);
    cross_psi = zeros(B, 1); % check this 
    for n=1:size(x_f, 2)
        [~, psi_last] = calc_step_mat(beta, x_f_vec{k, n}, psi_last, mu, B, epsilon);
        psi_last_reg = epsilon*eye(size(psi_last)) + psi_last;
        cross_psi = beta*cross_psi + (1-beta)*x_f_vec{k, n}*e_f{ref_mic}(k, n);

        delta_h1{k ,n} = (psi_last_reg \ eye(size(psi_last_reg)))*cross_psi;

        u_hat{ref_mic}(k, n) = delta_h1{k, n}'*x_f_vec{k, n};
    end
end

% plotting estimated u1
u1_hat_t = istft(u_hat{ref_mic}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
figure;
plot(real(u1_hat_t))
hold on
plot(u_t{ref_mic})
legend('$\hat{u}_4$', "$u_4$", 'Interpreter', 'latex');
%%
% rest mics - using RETF
for m=1:M
    if m == ref_mic
        continue
    end
    delta_am = cell(size(d_est_f{ref_mic}));
    u_hat{m} = zeros(size(d_est_f{ref_mic}));
    for k=1:window_length
        %delta_h1{k, 1} = zeros(P, 1);
        psi_last = M*eye(P);
        cross_psi = zeros(P, 1); % check this 
        for n=1:size(x_f, 2)
            [~, psi_last] = calc_step_mat_retf(beta, d1_est_f_vec{k, n}, psi_last, eta, P, epsilon);
            psi_last_reg = epsilon*eye(size(psi_last)) + psi_last;
            cross_psi = beta*cross_psi + (1-beta)*d1_est_f_vec{k, n}*e_f{m}(k, n);

            delta_am{k ,n} = (B/P)*(psi_last_reg \ eye(size(psi_last_reg)))*cross_psi;
            %delta_am{k ,n} = (psi_last_reg \ eye(size(psi_last_reg)))*cross_psi;
    
            u_hat{m}(k, n) = delta_am{k, n}'*d1_est_f_vec{k, n};
        end
    end
end

% plotting estimated u2
u2_hat_t = istft(u_hat{2}, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
figure;
plot(real(u2_hat_t))
hold on
plot(u_t{2})
legend('$\hat{u}_2$', "$u_2$", 'Interpreter', 'latex');

%% estimating residual echo covariance matrix
u_hat_vec = cell(size(x_f));
for k=1:window_length
    for n=1:size(x_f, 2)
        u_hat_vec{k, n} = zeros(M, 1);
        for m=1:M
            u_hat_vec{k, n}(m) = u_hat{m}(k, n);
            %u_hat_vec{k, n}(m) = u_f{m}(k, n);
        end
    end
end

phi_u_hat = cell(size(x_f));
for k=1:window_length
    phi_u_hat{k, 1} = eye(M);
    for n=2:size(x_f, 2)
        phi_u_hat{k, n} = alpha*phi_u_hat{k, n-1} + (1-alpha)*u_hat_vec{k, n}*u_hat_vec{k, n}';
    end
end

phi_v = sigma_v^2 * eye(M); % for now - change later
%% bemforming 
% % creating steering vector using far field approximation
% sv = zeros(M ,window_length);
% for k=1:window_length
%     freq = ((k-1)/window_length - 0.5) * fs;
%     sv(:, k) = steering_vector(M, doa, freq, c, relative_pos_a);
% end
% 
% sv2 = zeros(M, window_length);
% for m=1:M
%      sv(m, :) = ga_f(m, :) ./ ga_f(1, :);
%      sv2(m, :) = gb_f(m, :) ./ gb_f(1, :);
% end

% creating MVDR beamformer
test_phi_u = sigma_loudspeaker^2 * eye(M);
h_mvdr = cell(size(x_f, 2), 1);
for f_idx=1:window_length
    for n=1:size(x_f, 2)
        phi_total = phi_v + phi_u_hat{f_idx, n};
        phi_total_inv = phi_total \ eye(size(phi_total));
        denominator = sv(:, f_idx)' * phi_total_inv * sv(:, f_idx);
        numerator = phi_total_inv * sv(:, f_idx);
        h_mvdr{n}(:, f_idx) = numerator / denominator;
    end
    % for n=round(size(x_f, 2)/2)+1:size(x_f, 2)
    %     phi_total = phi_v + phi_u_hat{f_idx, n};
    %     phi_total_inv = phi_total \ eye(size(phi_total));
    %     denominator = sv2(:, f_idx)' * phi_total_inv * sv2(:, f_idx);
    %     numerator = phi_total_inv * sv2(:, f_idx);
    %     h_mvdr{n}(:, f_idx) = numerator / denominator;
    % end
end

% creating input format for beamformer
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
    tmp_mat = h_mvdr{l}' * e_f_vec{l};
    output_f(:, l) = diag(tmp_mat);

    tmp_mat = h_mvdr{l}' * u_f_vec{l};
    residual_echo_f(:, l) = diag(tmp_mat);
end
%% plotting fig 10
output = istft(output_f, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);
residual_echo = istft(residual_echo_f, fs, "Window", window_type, "OverlapLength", overlap, ...
   "FFTLength", window_length);

e1_plot = e_t{ref_mic}(indices);
s1_hat_plot = output(indices);
residual_echo_plot = residual_echo(indices);
figure;
plot(real(e1_plot));
ylim([-0.2, 0.2]);
xlabel("$lR$", 'Interpreter', 'latex');
hold on;
plot(real(s1_hat_plot));
hold on;
plot(real(residual_echo_plot));
legend("$e_4(lR)$", "$\hat{s}_4(lR)$", "$\bar{u}(lR)$", 'Interpreter', 'latex');
hold off;

%% saving audio
audiowrite('output.wav', real(output), fs);

%% calculating final erle
final_erle = zeros(1, length(output)/R);
for l=0:length(final_erle)-1
    d_vec = d_t(1, (l*R+1):(R*l+R));
    residual_echo_vec = residual_echo((l*R+1):(R*l+R));
    tmp = norm(d_vec)^2 / norm(residual_echo_vec)^2;
    final_erle(l+1) = 10*log10(tmp);
end

figure;
plot(final_erle);
grid on;
xlabel("$lR$", 'Interpreter', "latex")
ylabel("$erle [dB]$", "Interpreter", "latex");

%% functions 
% funtion to create signal vec for convolution
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

function [M, psi] = calc_step_mat_retf(beta, d, psi_prev, eta, p, epsilon)
    ddH = d*d';
    psi = beta*psi_prev + (1-beta)*ddH;
    psi_reg = psi + epsilon*eye(size(psi));

    M = (eta / p) * (psi_reg \ eye(size(psi)));
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

