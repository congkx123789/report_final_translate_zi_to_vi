import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Set high-quality styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.9
plt.rcParams['grid.color'] = '#f1f5f9'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.8

def parse_training_log(filepath, max_step=200000):
    steps = []
    losses = []
    ppls = []
    speeds = []
    vrams = []
    
    pattern = re.compile(
        r'Step \[\s*(\d+).*?\] \| Loss:\s*([\d\.]+) \| PPL:\s*([\d\.]+)(?: \| LR:\s*[\d\.e\-\+]+)?(?: \| Speed:\s*([\d\.]+) sent/s)?(?:.*?VRAM:\s*(\d+)/(\d+)MB)?'
    )
    
    last_step = -1
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                ppl = float(m.group(3))
                speed = float(m.group(4)) if m.group(4) else None
                vram = int(m.group(5)) if m.group(5) else None
                
                if step > last_step and step <= max_step:
                    steps.append(step)
                    losses.append(loss)
                    ppls.append(ppl)
                    speeds.append(speed)
                    vrams.append(vram)
                    last_step = step
                    
    return np.array(steps), np.array(losses), np.array(ppls), np.array(speeds, dtype=object), np.array(vrams, dtype=object)

def ema_smooth(data, alpha=0.08):
    smoothed = []
    val = data[0] if len(data) > 0 else 0
    for d in data:
        val = alpha * d + (1 - alpha) * val
        smoothed.append(val)
    return np.array(smoothed)

def main():
    log_tf = 'logs_training/train_transformer_baseline.log'
    log_ssm = 'logs_training/train_full_scratch.log'
    
    print("Parsing logs...")
    tf_steps, tf_losses, tf_ppls, tf_speeds, tf_vrams = parse_training_log(log_tf, max_step=200000)
    ssm_steps, ssm_losses, ssm_ppls, ssm_speeds, ssm_vrams = parse_training_log(log_ssm, max_step=200000)
    
    print(f"Loaded {len(tf_steps)} Transformer points (up to step {tf_steps[-1]:,})")
    print(f"Loaded {len(ssm_steps)} SSM points (up to step {ssm_steps[-1]:,})")
    
    # 6-panel Figure (3 rows x 2 cols)
    fig = plt.figure(figsize=(22, 14), dpi=220)
    gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.24)
    
    color_ssm_main = '#1d4ed8'    # Bold blue
    color_ssm_light = '#93c5fd'   # Light blue
    color_tf_main = '#b91c1c'     # Bold red
    color_tf_light = '#fca5a5'    # Light red
    
    # ----------------------------------------------------
    # Panel 1: Cross-Entropy Loss (0 to 200,000 steps)
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ssm_steps, ssm_losses, color=color_ssm_light, alpha=0.3, linewidth=0.8, label='SSM (Raw)')
    ax1.plot(ssm_steps, ema_smooth(ssm_losses, alpha=0.1), color=color_ssm_main, linewidth=2.2, label='Evolving Hybrid SSM (Smoothed)')
    
    ax1.plot(tf_steps, tf_losses, color=color_tf_light, alpha=0.3, linewidth=0.8, label='Transformer (Raw)')
    ax1.plot(tf_steps, ema_smooth(tf_losses, alpha=0.1), color=color_tf_main, linewidth=2.2, label='Standard Transformer (Smoothed)')
    
    ax1.set_title('1. Cross-Entropy Loss Convergence (0 – 200,000 Steps)', fontsize=13, fontweight='bold', pad=8)
    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=11)
    ax1.set_ylim(0.0, 10.5)
    ax1.grid(True)
    ax1.legend(loc='upper right', framealpha=0.92, fontsize=10)
    
    # Annotate end points
    ax1.annotate(f'SSM End: {ssm_losses[-1]:.2f}', 
                 xy=(ssm_steps[-1], ssm_losses[-1]), 
                 xytext=(ssm_steps[-1] - 45000, ssm_losses[-1] + 1.2),
                 arrowprops=dict(arrowstyle="->", color=color_ssm_main, lw=1.5),
                 fontweight='bold', color=color_ssm_main, fontsize=10)
                 
    ax1.annotate(f'TF End: {tf_losses[-1]:.2f}', 
                 xy=(tf_steps[-1], tf_losses[-1]), 
                 xytext=(tf_steps[-1] - 45000, tf_losses[-1] - 0.7),
                 arrowprops=dict(arrowstyle="->", color=color_tf_main, lw=1.5),
                 fontweight='bold', color=color_tf_main, fontsize=10)
                 
    # ----------------------------------------------------
    # Panel 2: Perplexity (PPL) on Logarithmic Scale
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ssm_steps, ema_smooth(ssm_ppls, alpha=0.1), color=color_ssm_main, linewidth=2.2, label='Evolving Hybrid SSM')
    ax2.plot(tf_steps, ema_smooth(tf_ppls, alpha=0.1), color=color_tf_main, linewidth=2.2, label='Standard Transformer')
    ax2.set_yscale('log')
    ax2.set_title('2. Perplexity (PPL) Decay (Logarithmic Scale)', fontsize=13, fontweight='bold', pad=8)
    ax2.set_xlabel('Training Steps', fontsize=11)
    ax2.set_ylabel('Perplexity (log scale)', fontsize=11)
    ax2.grid(True, which='both')
    ax2.legend(loc='upper right', framealpha=0.92, fontsize=10)
    
    ax2.annotate(f'SSM PPL: {ssm_ppls[-1]:.2f}', 
                 xy=(ssm_steps[-1], ssm_ppls[-1]), 
                 xytext=(ssm_steps[-1] - 50000, ssm_ppls[-1] * 2.5),
                 arrowprops=dict(arrowstyle="->", color=color_ssm_main, lw=1.5),
                 fontweight='bold', color=color_ssm_main, fontsize=10)
                 
    ax2.annotate(f'TF PPL: {tf_ppls[-1]:.2f}', 
                 xy=(tf_steps[-1], tf_ppls[-1]), 
                 xytext=(tf_steps[-1] - 50000, tf_ppls[-1] * 0.4),
                 arrowprops=dict(arrowstyle="->", color=color_tf_main, lw=1.5),
                 fontweight='bold', color=color_tf_main, fontsize=10)

    # ----------------------------------------------------
    # Panel 3: Training Throughput (Sentences/sec)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    valid_ssm_sp = [(s, sp) for s, sp in zip(ssm_steps, ssm_speeds) if sp is not None and sp > 0]
    valid_tf_sp = [(s, sp) for s, sp in zip(tf_steps, tf_speeds) if sp is not None and sp > 0]
    
    if valid_ssm_sp:
        ss_x, ss_y = zip(*valid_ssm_sp)
        ax3.plot(ss_x, ema_smooth(list(ss_y), alpha=0.05), color=color_ssm_main, linewidth=2.0, label='Hybrid SSM (~135 sent/s)')
    if valid_tf_sp:
        tf_x, tf_y = zip(*valid_tf_sp)
        ax3.plot(tf_x, ema_smooth(list(tf_y), alpha=0.05), color=color_tf_main, linewidth=2.0, label='Transformer (~377 sent/s)')
        
    ax3.set_title('3. Training Throughput (Sentences/second on RTX 5060 Ti)', fontsize=13, fontweight='bold', pad=8)
    ax3.set_xlabel('Training Steps', fontsize=11)
    ax3.set_ylabel('Sentences / sec', fontsize=11)
    ax3.grid(True)
    ax3.legend(loc='center right', framealpha=0.92, fontsize=10)
    ax3.set_ylim(0, 450)

    # ----------------------------------------------------
    # Panel 4: Active VRAM Allocation During Training
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    valid_ssm_vr = [(s, v) for s, v in zip(ssm_steps, ssm_vrams) if v is not None and v > 0]
    valid_tf_vr = [(s, v) for s, v in zip(tf_steps, tf_vrams) if v is not None and v > 0]
    
    if valid_ssm_vr:
        svr_x, svr_y = zip(*valid_ssm_vr)
        ax4.plot(svr_x, ema_smooth(list(svr_y), alpha=0.1), color=color_ssm_main, linewidth=2.0, label='Hybrid SSM (~1,250 MB)')
    if valid_tf_vr:
        tvr_x, tvr_y = zip(*valid_tf_vr)
        ax4.plot(tvr_x, ema_smooth(list(tvr_y), alpha=0.1), color=color_tf_main, linewidth=2.0, label='Transformer (~672 MB)')
        
    ax4.set_title('4. Active Training VRAM Allocation (Batch=16, Accum=2)', fontsize=13, fontweight='bold', pad=8)
    ax4.set_xlabel('Training Steps', fontsize=11)
    ax4.set_ylabel('Allocated VRAM (MB)', fontsize=11)
    ax4.grid(True)
    ax4.legend(loc='center right', framealpha=0.92, fontsize=10)
    ax4.set_ylim(0, 2000)

    # ----------------------------------------------------
    # Panel 5: Zoomed Late-Stage Convergence (Steps 100k – 200k)
    # ----------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    mask_ssm_late = ssm_steps >= 100000
    mask_tf_late = tf_steps >= 100000
    
    ax5.plot(ssm_steps[mask_ssm_late], ema_smooth(ssm_losses[mask_ssm_late], alpha=0.08), 
             color=color_ssm_main, linewidth=2.2, label='Evolving Hybrid SSM')
    ax5.plot(tf_steps[mask_tf_late], ema_smooth(tf_losses[mask_tf_late], alpha=0.08), 
             color=color_tf_main, linewidth=2.2, label='Standard Transformer')
             
    ax5.set_title('5. Zoomed Late-Stage Convergence (Steps 100,000 – 200,000)', fontsize=13, fontweight='bold', pad=8)
    ax5.set_xlabel('Training Steps', fontsize=11)
    ax5.set_ylabel('Loss (Zoomed Scale)', fontsize=11)
    ax5.grid(True)
    ax5.legend(loc='upper right', framealpha=0.92, fontsize=10)
    ax5.set_ylim(0.4, 2.8)

    # ----------------------------------------------------
    # Panel 6: Quantitative Comparison Table (Text/Summary Card)
    # ----------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    col_labels = ['Chỉ tiêu so sánh', 'Hybrid Attention-SSM', 'Standard Transformer', 'Nhận xét chuyên môn']
    table_data = [
        ['Số tham số (Parameters)', '89.35M (Kiến trúc tiến hóa)', '52.80M (Decoder-Only)', 'SSM tích hợp Evolving Matrix'],
        ['Best Loss đạt được', f'{min(ssm_losses):.4f}', f'{min(tf_losses):.4f}', 'Cả hai đều hội tụ rất sâu (< 0.35)'],
        ['Loss chặng cuối (200k)', f'{ssm_losses[-1]:.4f}', f'{tf_losses[-1]:.4f}', 'Transformer tối ưu cho dịch máy'],
        ['PPL chặng cuối (200k)', f'{ssm_ppls[-1]:.2f}', f'{tf_ppls[-1]:.2f}', 'Cực kỳ ổn định (PPL < 3.0)'],
        ['Tốc độ train (RTX 5060 Ti)', '~135 sent/s (4.9k tok/s)', '~377 sent/s (11.5k tok/s)', 'TF nhanh hơn nhờ FlashAttn SM120'],
        ['VRAM khi Train (b=16)', '~1,250 MB', '~672 MB', 'SSM giữ nhiều state trung gian hơn'],
        ['VRAM Suy luận (2M Tokens)', '542 MB (CỐ ĐỊNH O(1))', '30.52 GB (O(N) OOM!)', '★ SSM tiết kiệm VRAM gấp 56 lần']
    ]
    
    table = ax6.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', 
                      colWidths=[0.23, 0.23, 0.23, 0.31], bbox=[-0.02, 0.04, 1.04, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    
    # Style header row
    for j in range(4):
        cell = table[(0, j)]
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9.0)
        
    # Style data rows
    for i in range(1, len(table_data) + 1):
        bg_color = '#f8fafc' if i % 2 == 1 else '#ffffff'
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor(bg_color)
            if j == 1:
                cell.set_text_props(color='#1e40af', fontweight='bold' if i in [2, 7] else 'normal')
            elif j == 2:
                cell.set_text_props(color='#991b1b', fontweight='bold' if i in [2, 7] else 'normal')
            elif j == 3:
                cell.set_text_props(color='#0f172a', fontweight='bold' if i == 7 else 'normal')

    ax6.set_title('6. Bảng đối sánh tổng hợp hiệu năng & Kiến trúc', fontsize=13, fontweight='bold', pad=12)

    # Super Title
    plt.suptitle('ĐỐI SÁNH CHI TIẾT QUÁ TRÌNH HUẤN LUYỆN VÀ TÍNH CHẤT KIẾN TRÚC\nEvolving Micro-Head SSM vs Standard Transformer Baseline (0 – 200,000 Steps | NVIDIA RTX 5060 Ti 16GB)', 
                 fontsize=15, fontweight='bold', y=0.985)
                 
    out_png = 'logs_training/detailed_training_comparison.png'
    out_pdf = 'uet-graduation-thesis/fig_detailed_training_comparison.pdf'
    
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Đã lưu đồ thị so sánh chi tiết dạng PNG: {out_png}")
    print(f"✓ Đã lưu đồ thị so sánh chi tiết dạng PDF: {out_pdf}")

if __name__ == '__main__':
    main()
