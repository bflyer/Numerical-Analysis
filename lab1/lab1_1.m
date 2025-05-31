clear; clc; close all;

% 参数设置
M = 1;                % 二阶导绝对值上界
epsilon = 1e-16;      % 计算一次函数值的误差上界
x = 1;
true_deriv = cos(x);  % 真实导数值

% 等间距生成步长h的对数坐标数组
h = logspace(-16, 0, 1000);

% 理论误差计算
truncation_error = (M * h) / 2;        % 截断误差
rounding_error = 2 * epsilon ./ h;     % 舍入误差
total_error_bound = truncation_error + rounding_error;  % 总误差限

% 实际误差计算
approx_deriv = (sin(x + h) - sin(x)) ./ h;
actual_error = abs(approx_deriv - true_deriv);

% 绘图
figure;
loglog(h, truncation_error, 'b--', 'LineWidth', 1.5); hold on;
loglog(h, rounding_error, 'r--', 'LineWidth', 1.5);
loglog(h, total_error_bound, 'k-', 'LineWidth', 2);
loglog(h, actual_error, 'g-', 'LineWidth', 1.5);

% 设置坐标轴和标签
xlabel('步长 h');
ylabel('误差');
title('差商近似导数的误差与步长h的关系');
xlim([1e-16, 1]);
ylim([1e-17, 10]);

% 标记理论最优步长
optimal_h = 2 * sqrt(epsilon / M);
line([optimal_h, optimal_h], ylim, 'Color', [0.5, 0, 0.5], 'LineStyle', ':', 'LineWidth', 1.5); 
text(optimal_h, 1e-14, sprintf('h_{minerr}=%.1e', optimal_h), 'Rotation', 90, 'Color', [0.5, 0, 0.5], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

% 标记实际最小步长
[min_actual_error, min_index] = min(actual_error);
optimal_h_actual = h(min_index);
line([optimal_h_actual, optimal_h_actual], ylim, 'Color', [0, 0.5, 0], 'LineStyle', '-.', 'LineWidth', 1.5); 
text(optimal_h_actual, 1e-16, sprintf('h_{actminerr}=%.1e', optimal_h_actual), 'Rotation', 90, 'Color', [0, 0.5, 0], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

legend('截断误差', '舍入误差', '总误差限', '实际总误差', '理论最优步长', '实际最优步长', 'Location', 'northeast');