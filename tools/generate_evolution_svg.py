width = 820
height = 430
margin_left = 60
margin_bottom = 75
margin_top = 55
margin_right = 30

plot_w = width - margin_left - margin_right
plot_h = height - margin_top - margin_bottom

phases = [
    ("Phase 1: Baseline", "Dense 32 + SGD/Adam"),
    ("Phase 2: Dropout", "He Init + Regularization"),
    ("Phase 3: BatchNorm", "CosineLR + Class Weights"),
    ("Phase 4: Deep-Wide", "512-Wide + FocalLoss")
]

acc_test = [35.53, 50.20, 66.20, 69.75]
acc_val = [41.20, 54.80, 71.89, 75.87]
acc_mate = [23.90, 52.00, 70.80, 75.70]

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">']
svg.append('<rect width="100%" height="100%" fill="#111827" rx="10"/>')
svg.append('<text x="410" y="32" fill="#F9FAFB" font-size="18" font-weight="bold" text-anchor="middle" font-family="sans-serif">Évolution des Performances Algorithmiques de MyTorch</text>')

# Horizontal grid
for y_val in [20, 40, 60, 80, 100]:
    y_pos = margin_top + plot_h - (y_val / 100.0) * plot_h
    svg.append(f'<line x1="{margin_left}" y1="{y_pos}" x2="{width - margin_right}" y2="{y_pos}" stroke="#374151" stroke-dasharray="4" stroke-width="1"/>')
    svg.append(f'<text x="{margin_left - 10}" y="{y_pos + 4}" fill="#9CA3AF" font-size="11" text-anchor="end" font-family="sans-serif">{y_val}%</text>')

n_groups = len(phases)
group_w = plot_w / n_groups
bar_w = 26

colors = ["#3B82F6", "#10B981", "#F59E0B"]
labels = ["Précision Test Global", "Précision Validation", "Détection Échec & Mat"]

for i in range(n_groups):
    cx = margin_left + i * group_w + group_w / 2
    vals = [acc_test[i], acc_val[i], acc_mate[i]]
    for j, val in enumerate(vals):
        bx = cx - 1.5 * bar_w + j * bar_w + (j - 1) * 2
        bh = (val / 100.0) * plot_h
        by = margin_top + plot_h - bh
        svg.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" fill="{colors[j]}" rx="3" opacity="0.95"/>')
        svg.append(f'<text x="{bx + bar_w/2}" y="{by - 5}" fill="#F3F4F6" font-size="10" font-weight="bold" text-anchor="middle" font-family="sans-serif">{val:.1f}%</text>')

    p1, p2 = phases[i]
    svg.append(f'<text x="{cx}" y="{margin_top + plot_h + 20}" fill="#E5E7EB" font-size="11" font-weight="bold" text-anchor="middle" font-family="sans-serif">{p1}</text>')
    svg.append(f'<text x="{cx}" y="{margin_top + plot_h + 34}" fill="#9CA3AF" font-size="9.5" text-anchor="middle" font-family="sans-serif">{p2}</text>')

leg_y = margin_top + plot_h + 54
leg_x_start = 140
for j, label in enumerate(labels):
    lx = leg_x_start + j * 190
    svg.append(f'<rect x="{lx}" y="{leg_y}" width="12" height="12" fill="{colors[j]}" rx="2"/>')
    svg.append(f'<text x="{lx + 18}" y="{leg_y + 10}" fill="#D1D5DB" font-size="11" font-family="sans-serif">{label}</text>')

svg.append('</svg>')

with open('doc/assets/model_evolution_comparison.svg', 'w', encoding='utf-8') as f:
    f.write('\n'.join(svg))
print('OK')
