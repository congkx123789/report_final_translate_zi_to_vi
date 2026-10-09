import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Configure matplotlib for crisp vector graphics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.8

# Color palette
COLOR_TF = '#ef4444'       # Crimson Red
COLOR_SSM = '#10b981'      # Emerald Green
COLOR_SSM_BLUE = '#2563eb' # Royal Blue

out_dir = '/home/alida/Documents/IT/all_model/My_alida_model/uet-graduation-thesis'
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 1. FIG_BLEU_COMPARISON.PDF (2 Models Only, No Overlap, Big Fonts)
# =============================================================================
def generate_bleu_chart():
    categories = [
        'Hội thoại\n(Daily)',
        'Tin tức\n(News)',
        'Tiên hiệp 80+\n(Cultivation)',
        'Cổ phong\n(Ancient)',
        'Khoa huyễn\n(Sci-Fi)'
    ]
    
    bleu_tf = [36.2, 34.8, 22.4, 21.1, 23.5]
    bleu_ssm = [37.1, 39.4, 37.6, 36.2, 36.8]
    
    x = np.arange(len(categories))
    width = 0.36
    
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)
    
    rects1 = ax.bar(x - width/2, bleu_tf, width, label='Standard Transformer (68.2M Baseline)', 
                    color='#ef4444', edgecolor='#b91c1c', linewidth=1.5, alpha=0.92)
    rects2 = ax.bar(x + width/2, bleu_ssm, width, label='Evolving-Head Hybrid SSM (71.97M Đề xuất)', 
                    color='#10b981', edgecolor='#047857', linewidth=1.5, alpha=0.95)
    
    ax.set_ylabel('Điểm SacreBLEU', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('So sánh Điểm chất lượng dịch BLEU trên các miền dữ liệu', fontsize=14, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 52)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=11.5, framealpha=0.95, loc='upper left')
    
    # Value labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='normal')
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#047857')
    
    # Highlight cultivation gap (placed in right empty area above bar 2)
    ax.annotate('+15.2 BLEU\n(Vượt trội)', 
                xy=(x[2] + width/2, bleu_ssm[2]), 
                xytext=(x[2] + width/2, bleu_ssm[2] + 7.5),
                arrowprops=dict(facecolor='#047857', shrink=0.08, width=1.5, headwidth=6),
                fontsize=10.5, fontweight='bold', color='#047857', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ecfdf5', edgecolor='#10b981', lw=1))
                
    plt.tight_layout()
    pdf_path = os.path.join(out_dir, 'fig_bleu_comparison.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: {pdf_path}")

# =============================================================================
# 2. FIG_LATENCY_COMPARISON.PDF (Evenly Spaced X-Axis, No Overlapping Ticks)
# =============================================================================
def generate_latency_chart():
    seq_labels = ['32', '64', '128', '256', '512', '1024', '2048']
    x_indices = np.arange(len(seq_labels))
    
    tf_latency = np.array([0.45, 0.62, 0.95, 1.82, 3.85, 8.42, 16.80])
    ssm_latency = np.array([0.38, 0.51, 0.72, 1.15, 1.85, 2.85, 4.40])
    
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)
    
    ax.plot(x_indices, tf_latency, 'o-', color='#ef4444', linewidth=3.0, markersize=8, 
            label='Standard Transformer ($O(N^2)$ Quadratic)')
    ax.plot(x_indices, ssm_latency, 's-', color='#2563eb', linewidth=3.0, markersize=8, 
            label='Evolving-Head Hybrid SSM ($O(N)$ Linear - Đề xuất)')
    
    ax.set_xlabel('Độ dài chuỗi ngữ cảnh sinh (Tokens)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Độ trễ sinh từ (ms/token)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('So sánh Độ trễ Suy luận theo Độ dài Chuỗi (Latency)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(seq_labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 19.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11.5, framealpha=0.95, loc='upper left')
    
    # Annotate values at 2048 tokens
    ax.annotate('Transformer: 16.8 ms\n(Chậm gấp 3.8x)', 
                xy=(x_indices[-1], 16.80), xytext=(x_indices[-1] - 1.6, 15.5),
                arrowprops=dict(facecolor='#ef4444', shrink=0.08, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', color='#b91c1c',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fef2f2', edgecolor='#ef4444', lw=1))
                
    ax.annotate('Hybrid SSM: 4.4 ms\n(Nhanh hơn 3.8x)', 
                xy=(x_indices[-1], 4.40), xytext=(x_indices[-1] - 1.6, 5.8),
                arrowprops=dict(facecolor='#2563eb', shrink=0.08, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', color='#1d4ed8',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#eff6ff', edgecolor='#2563eb', lw=1))
                
    plt.tight_layout()
    pdf_path = os.path.join(out_dir, 'fig_latency_comparison.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: {pdf_path}")

# =============================================================================
# 3. FIG_VRAM_COMPARISON.PDF (Evenly Spaced X-Axis, No Overlapping Ticks)
# =============================================================================
def generate_vram_chart():
    seq_labels = ['32', '64', '128', '256', '512', '1024', '2048']
    x_indices = np.arange(len(seq_labels))
    
    tf_vram = np.array([580, 650, 820, 1250, 2200, 4150, 7420])
    ssm_vram = np.array([550, 590, 680, 850, 1150, 1550, 2056])
    
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)
    
    ax.plot(x_indices, tf_vram, 'o-', color='#ef4444', linewidth=3.0, markersize=8, 
            label='Standard Transformer (KV-Cache $O(N)$)')
    ax.plot(x_indices, ssm_vram, 's-', color='#10b981', linewidth=3.0, markersize=8, 
            label='Evolving-Head Hybrid SSM (Nén trạng thái $O(1)$)')
    
    ax.set_xlabel('Độ dài chuỗi ngữ cảnh sinh (Tokens)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Bộ nhớ VRAM tiêu thụ đỉnh (MB)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('So sánh Tiêu thụ Bộ nhớ Đồ họa Đỉnh (Peak VRAM)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(seq_labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 8500)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11.5, framealpha=0.95, loc='upper left')
    
    ax.annotate('Transformer: 7,420 MB\n(Chiếm 46.4% GPU 16GB)', 
                xy=(x_indices[-1], 7420), xytext=(x_indices[-1] - 1.9, 6800),
                arrowprops=dict(facecolor='#ef4444', shrink=0.08, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', color='#b91c1c',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fef2f2', edgecolor='#ef4444', lw=1))
                
    ax.annotate('Hybrid SSM: 2,056 MB\n(Tiết kiệm 72.3% VRAM)', 
                xy=(x_indices[-1], 2056), xytext=(x_indices[-1] - 1.9, 3100),
                arrowprops=dict(facecolor='#10b981', shrink=0.08, width=1.5, headwidth=6),
                fontsize=11, fontweight='bold', color='#047857',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ecfdf5', edgecolor='#10b981', lw=1))
                
    plt.tight_layout()
    pdf_path = os.path.join(out_dir, 'fig_vram_comparison.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: {pdf_path}")

# =============================================================================
# 4. FIG_SLIDE_TRAINING_COMP.PDF (Clean 2-Panel Stacked Chart for Slide 12)
# =============================================================================
def parse_training_log(filepath, max_step=200000):
    steps, losses, ppls = [], [], []
    pattern = re.compile(r'Step \[\s*(\d+).*?\] \| Loss:\s*([\d\.]+) \| PPL:\s*([\d\.]+)')
    last_step = -1
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                ppl = float(m.group(3))
                if step > last_step and step <= max_step:
                    steps.append(step)
                    losses.append(loss)
                    ppls.append(ppl)
                    last_step = step
    return np.array(steps), np.array(losses), np.array(ppls)

def ema_smooth(data, alpha=0.1):
    smoothed = []
    val = data[0] if len(data) > 0 else 0
    for d in data:
        val = alpha * d + (1 - alpha) * val
        smoothed.append(val)
    return np.array(smoothed)

def generate_slide_training_comp():
    log_tf = '/home/alida/Documents/IT/all_model/My_alida_model/logs_training/train_transformer_baseline.log'
    log_ssm = '/home/alida/Documents/IT/all_model/My_alida_model/logs_training/train_full_scratch.log'
    
    tf_steps, tf_losses, tf_ppls = parse_training_log(log_tf, max_step=200000)
    ssm_steps, ssm_losses, ssm_ppls = parse_training_log(log_ssm, max_step=200000)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 6.6), dpi=300, sharex=True)
    
    # Subplot 1: Loss Convergence
    ax1.plot(ssm_steps, ssm_losses, color='#93c5fd', alpha=0.35, linewidth=0.8)
    ax1.plot(ssm_steps, ema_smooth(ssm_losses, 0.12), color='#1d4ed8', linewidth=2.8, label='Evolving Hybrid SSM')
    ax1.plot(tf_steps, tf_losses, color='#fca5a5', alpha=0.35, linewidth=0.8)
    ax1.plot(tf_steps, ema_smooth(tf_losses, 0.12), color='#b91c1c', linewidth=2.8, label='Standard Transformer')
    
    ax1.set_title('1. Diễn biến Hội tụ Loss (0 – 200,000 Steps)', fontsize=13, fontweight='bold', pad=8)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 10.5)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10.5, framealpha=0.95, loc='upper right')
    
    ax1.annotate('SSM Best: 0.1794', 
                 xy=(ssm_steps[-1], ssm_losses[-1]), 
                 xytext=(ssm_steps[-1] - 75000, ssm_losses[-1] + 1.8),
                 arrowprops=dict(facecolor='#1d4ed8', shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', color='#1d4ed8', fontsize=11)
                 
    ax1.annotate('TF Best: 0.2017', 
                 xy=(tf_steps[-1], tf_losses[-1]), 
                 xytext=(tf_steps[-1] - 75000, tf_losses[-1] + 3.8),
                 arrowprops=dict(facecolor='#b91c1c', shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', color='#b91c1c', fontsize=11)
                 
    # Subplot 2: Perplexity Decay (Log Scale)
    ax2.plot(ssm_steps, ssm_ppls, color='#93c5fd', alpha=0.35, linewidth=0.8)
    ax2.plot(ssm_steps, ema_smooth(ssm_ppls, 0.12), color='#1d4ed8', linewidth=2.8, label='Hybrid SSM (PPL: 1.20)')
    ax2.plot(tf_steps, tf_ppls, color='#fca5a5', alpha=0.35, linewidth=0.8)
    ax2.plot(tf_steps, ema_smooth(tf_ppls, 0.12), color='#b91c1c', linewidth=2.8, label='Transformer (200k PPL: 2.18)')
    
    ax2.set_yscale('log')
    ax2.set_title('2. Độ phức tạp Perplexity (Thang Logarit)', fontsize=13, fontweight='bold', pad=8)
    ax2.set_xlabel('Bước Huấn luyện (Training Steps)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('PPL (Log Scale)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10.5, framealpha=0.95, loc='upper right')
    
    plt.tight_layout()
    pdf_path = os.path.join(out_dir, 'fig_slide_training_comp.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: {pdf_path}")

if __name__ == '__main__':
    generate_bleu_chart()
    generate_latency_chart()
    generate_vram_chart()
    generate_slide_training_comp()
