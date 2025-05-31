% (1) 单精度求和
sum_single = single(0);
current_sum = sum_single;

tic; % 计时开始

for n = 1:10000000 % 设置一个较大的上限
    term = single(1) / single(n);
    current_sum = sum_single + term;
    if current_sum == sum_single
        break; 
    end
    sum_single = current_sum;
end

elapsedTime = toc; % 计时结束

n_part1 = n - 1;
disp(['(1) 单精度求和停止变化的n值: ', num2str(n_part1)]);
disp(['    此时的结果为: ', num2str(sum_single)]);

% 理论估计n值
eps_single = eps('single'); % 1.1921e-07
n_estimate_single = 1e6; % 初始猜测
for iter = 1:5
    S = log(n_estimate_single) + 0.5772;
    threshold = eps_single * S / 2;
    if 1/n_estimate_single < threshold
        break;
    end
    n_estimate_single = n_estimate_single * 2;
end
disp(['    单精度求和停止变化的n估计值: ~', num2str(n_estimate_single)]);

% (2) 双精度求和
sum_double = 0;
for k = 1:n_part1
    sum_double = sum_double + 1/k;
end
error_abs = double(sum_single) - sum_double;
error_rel = error_abs / sum_double;
disp(['(2) 双精度结果为: ', num2str(sum_double)]);
disp(['    绝对误差: ', num2str(error_abs)]);
disp(['    相对误差: ', num2str(error_rel)]);

% (3) 双精度求和极限时间估计
% 理论估计n值
eps_double = eps('double'); % 2.2204e-16
n_estimate = 1e14; % 初始猜测
for iter = 1:5
    S = log(n_estimate) + 0.5772;
    threshold = eps_double * S / 2;
    if 1/n_estimate < threshold
        break;
    end
    n_estimate = n_estimate * 2;
end

% 计算时间估计
total_time_s = n_estimate / n_estimate_single * elapsedTime;
total_time_year = total_time_s / (60 * 60 * 24 * 365);
disp(['(3) 双精度求和停止变化的n估计值: ~', num2str(n_estimate)]);
disp(['    预计计算时间: ~', num2str(total_time_year), ' 年']);