import numpy as np
import torch
import random


def generate_continuous_mask(T, n, k):
    # 무작위로 시작 위치 생성
    start_positions = torch.randint(1, T - k + 1, (n,))

    # 범위 생성
    ranges = torch.arange(k).unsqueeze(0)  # (1, 1, k)
    expanded_positions = start_positions.unsqueeze(-1) + ranges  # (B, n, k)

    # 마스킹 텐서 생성
    mask = torch.zeros((T), dtype=torch.float32)  # 마스킹 텐서 초기화
    
    # 마스크 채우기
    mask[expanded_positions] = 1

    return mask

def forward_fill_with_mask(tensor, mask):
    T, C = tensor.shape

    # 마스크를 텐서의 shape에 맞게 확장
    mask = mask.unsqueeze(-1).expand(T, C)

    # 마스크된 영역을 NaN으로 설정
    masked_tensor = tensor.detach().clone()
    masked_tensor[mask == 1] = float('nan')

    # Forward fill 구현
    for i in range(1, T):  # Forward fill (시간 축으로 진행)
        nan_mask = torch.isnan(masked_tensor[i, :])  # 현재 위치가 NaN인 경우
        # print(masked_tensor.shape, nan_mask.shape)
        # masked_tensor[nan_mask, i, :] = masked_tensor[nan_mask, i - 1, :]  # 이전 값 복사
        masked_tensor[i, :][nan_mask] = masked_tensor[i - 1, :][nan_mask]

    return masked_tensor, mask

def zero_fill_with_mask(tensor, mask):
    
    # 마스크를 텐서의 shape에 맞게 확장
    if len(mask.shape) == 1:  # (T)
        mask = mask.unsqueeze(-1).expand_as(tensor)  # (T, C)
    
    # 마스크가 1인 영역을 0으로 설정
    tensor_filled = tensor.detach().clone()
    tensor_filled[mask == 1] = 0.0

    return tensor_filled, mask


def inject_noise(seq, normal_ratio=0.3, noise_prob=0.2, noise_type='point-gaussian', noise_mean=0.0, noise_std=2.0):
    
    if type(seq) == np.ndarray:
        seq = torch.from_numpy(seq)

    random_value = random.random()
    if random_value <= normal_ratio:
        noise_seq = seq.detach().clone()
        noise_mask = torch.zeros_like(seq)
        return seq, noise_seq, noise_mask
         
    T, C = seq.shape  # Batch size, Sequence length, Channel size

    e = None
    if noise_type == 'point-gaussian':
        noise_mask = torch.rand(T, C, device=seq.device) < noise_prob
        gaussian_noise = torch.randn_like(seq) * noise_std + noise_mean  # 평균 noise_mean, 표준 편차 noise_std
        e = torch.zeros_like(seq)
        e[noise_mask] = gaussian_noise[noise_mask]
        
    elif noise_type == 'point-abs-gaussian':
        noise_mask = torch.rand(T, C, device=seq.device) < noise_prob
        gaussian_noise = torch.randn_like(seq) * noise_std + noise_mean  # 평균 noise_mean, 표준 편차 noise_std
        e = torch.zeros_like(seq)
        e[noise_mask] = torch.abs(gaussian_noise[noise_mask])
        
    elif noise_type == 'point-zero':
        # Missing noise, random elements are set to zero based on noise_prob
        noise_mask = torch.rand_like(seq) < noise_prob
        e = torch.zeros_like(seq)
        noise_seq = seq.clone().detach()  # Clone the sequence to apply noise
        noise_seq[noise_mask] = 0  # Apply the missing noise
        return seq, noise_seq, noise_mask
    
    elif noise_type == 'continuous-ffill':
        nk = int(T * noise_prob)
        n, k = 2, nk//2  # 마스킹 개수와 길이 설정
        noise_mask = generate_continuous_mask(T, n, k)  # 마스크 생성
        noise_seq, noise_mask = forward_fill_with_mask(seq, noise_mask)
        return seq, noise_seq, noise_mask
        
    elif noise_type == 'continuous-zero':
        nk = int(T * noise_prob)
        n, k = 2, nk//2  # 마스킹 개수와 길이 설정
        noise_mask = generate_continuous_mask(T, n, k)  # 마스크 생성
        noise_seq, noise_mask = zero_fill_with_mask(seq, noise_mask)
        return seq, noise_seq, noise_mask
    
    else:
        raise ValueError("Unsupported noise type provided.")

    e = e.to(seq.device)  # Ensure 'e' is on the same device as the input sequence
    noise_seq = seq.clone().detach()  # Clone the sequence to apply noise
    noise_seq += e  # Apply the noise

    return seq, noise_seq, noise_mask

def simulate_distshift(in_seq, out_seq, start_t, slope=0.002, b=0.0, shift_type='trend', direction='random'):
    if type(in_seq) == np.ndarray:
        in_seq = torch.from_numpy(in_seq)
    if type(out_seq) == np.ndarray:
        out_seq = torch.from_numpy(out_seq)
    
    I, _ = in_seq.shape
    O, _ = out_seq.shape
    
    seq = torch.cat([in_seq, out_seq], dim=0)
    
    if shift_type == 'trend':
        # start_t: (B,), seq: (B, T, C)
        T, C = seq.shape
        shift_mask = torch.ones_like(seq)
        input_times = start_t + torch.arange(T, device=seq.device)
        input_times = input_times.unsqueeze(-1)
        
        if direction == 'random':
            if random.random() < 0.5:
                dist_seq = seq - (slope * input_times + b)
            else:
                dist_seq = seq + (slope * input_times + b)
        elif direction == 'up':
            dist_seq = seq + (slope * input_times + b)
        elif direction == 'down':
            dist_seq = seq - (slope * input_times + b)
        
        in_dist_seq = dist_seq[:I, ].detach().clone()
        out_dist_seq = dist_seq[-O:, ].detach().clone()
        
        return in_seq, in_dist_seq, shift_mask[:I, ], out_seq, out_dist_seq
    
    elif shift_type == 'seasonal-amp':
        # start_t: (B,), seq: (B, T, C)
        T, C = seq.shape
        shift_mask = torch.ones_like(seq)
        input_times = start_t + torch.arange(T, device=seq.device)
        input_times = input_times.unsqueeze(-1)
        
        seq_mean = seq.mean(dim=0, keepdim=True)
        if direction == 'random':
            if random.random() < 0.5:
                dist_seq = (seq-seq_mean) / (slope*2 * input_times + 1) + seq_mean
            else:
                dist_seq = (seq-seq_mean) * (slope*2 * input_times + 1) + seq_mean
        elif direction == 'up':
            dist_seq = (seq-seq_mean) * (slope*2 * input_times + 1) + seq_mean
        elif direction == 'down':
            dist_seq = (seq-seq_mean) / (slope*2 * input_times + 1) + seq_mean
        
        in_dist_seq = dist_seq[:I, ].detach().clone()
        out_dist_seq = dist_seq[-O:, ].detach().clone()
        
        return in_seq, in_dist_seq, shift_mask[:I, ], out_seq, out_dist_seq
    else:
        raise Exception()