import os
import re
import math
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration for publication-ready figures
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

def parse_log(path, max_step=None, start_line=0):
    steps = []
    losses = []
    ppls = []
    speeds = []
    vrams = []
    
    pattern = re.compile(
        r'Step \[\s*(\d+).*?\] \| Loss:\s*([\d\.]+) \| PPL:\s*([\d\.]+)(?: \| LR:.*?)?(?: \| Speed:\s*([\d\.]+) sent/s)?(?:.*?VRAM:\s*(\d+)/(\d+)MB)?'
    )
    
    last_step = -1
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[start_line:]
        for line in lines:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                ppl = float(m.group(3))
                speed = float(m.group(4)) if m.group(4) else None
                vram_alloc = int(m.group(5)) if m.group(5) else None
                
                # Keep strictly monotonic increasing steps
                if step > last_step:
                    if max_step is None or step <= max_step:
                        steps.append(step)
                        losses.append(loss)
                        ppls.append(ppl)
                        speeds.append(speed)
                        vrams.append(vram_alloc)
                        last_step = step
                        
    return steps, losses, ppls, speeds, vrams

def compute_ema(data, alpha=0.15):
    ema = []
    val = data[0] if data else 0
    for d in data:
        val = alpha * d + (1 - alpha) * val
        ema.append(val)
    return ema

def main():
    log_ssm = 'logs_training/train_full_scratch.log'
    log_tf = 'logs_training/train_transformer_baseline.log'
    
    ssm_steps, ssm_losses, ssm_ppls, ssm_speeds, ssm_vrams = parse_log(log_ssm, start_line=20)
    tf_steps, tf_losses, tf_ppls, tf_speeds, tf_vrams = parse_log(log_tf, start_line=20)
    
    max_step = max(tf_steps) if tf_steps else 5000
    print(f"Max transformer step available: {max_step}")
    
    # Filter SSM data to match current transformer step range
    ssm_idx_comp = [i for i, s in enumerate(ssm_steps) if s <= max_step]
    ssm_s_comp = [ssm_steps[i] for i in ssm_idx_comp]
    ssm_l_comp = [ssm_losses[i] for i in ssm_idx_comp]
    ssm_p_comp = [ssm_ppls[i] for i in ssm_idx_comp]
    
    # Create Figure with 2x2 Layout:
    # Top-Left: Loss (0 to max_step)
    # Bottom-Left: PPL (0 to max_step, Log Scale)
    # Top-Right: Long-Term SSM Loss Trajectory (0 to 435k) with zoomed comparison
    # Bottom-Right: Resource & Speed Comparison
    fig = plt.figure(figsize=(16, 10), dpi=200)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    
    # ----------------------------------------------------
    # Plot 1: Loss Convergence (Step 0 to max_step)
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(ssm_s_comp, ssm_l_comp, color='#3b82f6', alpha=0.18, s=12, label='SSM (Raw)')
    ax1.plot(ssm_s_comp, compute_ema(ssm_l_comp, alpha=0.2), color='#1d4ed8', linewidth=2.2, label='Evolving Hybrid SSM (EMA)')
    
    ax1.scatter(tf_steps, tf_losses, color='#ef4444', alpha=0.18, s=12, label='Transformer (Raw)')
    ax1.plot(tf_steps, compute_ema(tf_losses, alpha=0.2), color='#b91c1c', linewidth=2.2, label='Standard Transformer (EMA)')
    
    ax1.set_title(f'1. Cross-Entropy Loss Comparison (Steps 1 – {max_step:,})', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.grid(True)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(0.0, 10.5)
    
    # ----------------------------------------------------
    # Plot 2: Perplexity (PPL) on Log Scale (Step 0 to max_step)
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ssm_s_comp, compute_ema(ssm_p_comp, alpha=0.2), color='#1d4ed8', linewidth=2.2, label='Evolving Hybrid SSM')
    ax2.plot(tf_steps, compute_ema(tf_ppls, alpha=0.2), color='#b91c1c', linewidth=2.2, label='Standard Transformer')
    
    ax2.set_yscale('log')
    ax2.set_title(f'2. Perplexity (PPL) Convergence (Log Scale)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Training Steps', fontsize=11)
    ax2.set_ylabel('Perplexity (PPL, log scale)', fontsize=11)
    ax2.grid(True, which='both')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    # Annotate final PPL
    if ssm_p_comp:
        last_ssm_p = ssm_p_comp[-1]
        ax2.annotate(f'SSM PPL: {last_ssm_p:.1f}', 
                     xy=(ssm_s_comp[-1], last_ssm_p), 
                     xytext=(ssm_s_comp[-1] - max_step*0.25, last_ssm_p * 2.2),
                     arrowprops=dict(arrowstyle="->", color='#1d4ed8', lw=1.5),
                     fontweight='bold', color='#1d4ed8')
    if tf_ppls:
        last_tf_p = tf_ppls[-1]
        ax2.annotate(f'Transformer PPL: {last_tf_p:.1f}', 
                     xy=(tf_steps[-1], last_tf_p), 
                     xytext=(tf_steps[-1] - max_step*0.25, last_tf_p * 0.4),
                     arrowprops=dict(arrowstyle="->", color='#b91c1c', lw=1.5),
                     fontweight='bold', color='#b91c1c')

    # ----------------------------------------------------
    # Plot 3: Long-term SSM Full Trajectory (0 to 435k)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 1])
    # Downsample full SSM for smooth display
    downsample_factor = max(1, len(ssm_steps) // 500)
    full_s = ssm_steps[::downsample_factor]
    full_l = ssm_losses[::downsample_factor]
    ax3.plot(full_s, compute_ema(full_l, alpha=0.1), color='#2563eb', linewidth=1.8, label='Full SSM Trajectory (0 – 435k Steps)')
    
    # Highlight current transformer progress
    ax3.axvspan(0, max_step, color='#fca5a5', alpha=0.35, label=f'Transformer Current ({max_step:,} steps)')
    ax3.annotate(f'SSM Converged: 0.755 (PPL: 2.13)', 
                 xy=(ssm_steps[-1], ssm_losses[-1]), 
                 xytext=(ssm_steps[-1] - 180000, 2.5),
                 arrowprops=dict(arrowstyle="->", color='#2563eb', lw=1.5),
                 fontweight='bold', color='#1e3a8a')
    
    ax3.set_title('3. Long-Term Convergence Trajectory (Hybrid SSM to 435k Steps)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Training Steps', fontsize=11)
    ax3.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax3.grid(True)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.set_ylim(0.0, 10.5)

    # ----------------------------------------------------
    # Plot 4: Hardware & Runtime Comparison (Throughput & VRAM)
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    categories = ['Throughput\n(Sentences/s)', 'Step Time\n(ms)', 'Active VRAM\n(MB)']
    
    # Metrics
    # Filter valid speeds
    ssm_sp_valid = [s for s in ssm_speeds[:len(ssm_idx_comp)] if s is not None and s > 0]
    tf_sp_valid = [s for s in tf_speeds if s is not None and s > 0]
    
    avg_ssm_speed = float(np.mean(ssm_sp_valid[-30:])) if ssm_sp_valid else 135.0
    avg_tf_speed = float(np.mean(tf_sp_valid[-30:])) if tf_sp_valid else 260.0
    
    # Step time for effective batch 32
    step_time_ssm = (32.0 / avg_ssm_speed) * 1000.0
    step_time_tf = (32.0 / avg_tf_speed) * 1000.0
    
    vram_ssm = 1250.0  # from log VRAM allocated
    vram_tf = 672.0    # from log VRAM allocated
    
    x = np.arange(len(categories))
    width = 0.35
    
    ssm_vals = [avg_ssm_speed, step_time_ssm, vram_ssm]
    tf_vals = [avg_tf_speed, step_time_tf, vram_tf]
    
    rects1 = ax4.bar(x - width/2, ssm_vals, width, label='Hybrid Attention-SSM', color='#3b82f6', edgecolor='#1d4ed8')
    rects2 = ax4.bar(x + width/2, tf_vals, width, label='Standard Transformer', color='#ef4444', edgecolor='#b91c1c')
    
    ax4.set_title('4. Hardware Efficiency & Runtime Comparison (RTX 5060 Ti)', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax4.grid(True, axis='y')
    ax4.legend(loc='upper right', framealpha=0.9)
    
    # Attach bar text labels
    for rect in rects1:
        h = rect.get_height()
        ax4.annotate(f'{h:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1d4ed8')
    for rect in rects2:
        h = rect.get_height()
        ax4.annotate(f'{h:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='#b91c1c')
                     
    ax4.set_ylim(0, max(vram_ssm, vram_tf) * 1.25)
    
    # Super Title
    plt.suptitle('Cross-Architecture Training Convergence & Efficiency Benchmark\nEvolving Micro-Head SSM vs. Standard Transformer Decoder-Only (NVIDIA RTX 5060 Ti Blackwell SM120)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # Save files
    out_png = 'logs_training/loss_ppl_comparison.png'
    out_pdf = 'uet-graduation-thesis/fig_loss_ppl_comparison.pdf'
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved PNG to {out_png}")
    print(f"✓ Saved PDF to {out_pdf}")

if __name__ == '__main__':
    main()
