import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(
1, 1,
zorder=2, s=120, edgecolors='white',
marker='P', color='#BDD7EC', linewidths=0.3
)

#ax.spines['bottom'].set_color('white')
#ax.spines['top'].set_color('white')
#ax.spines['left'].set_color('white')
#ax.spines['right'].set_color('white')
#ax.xaxis.label.set_color('white')
#ax.yaxis.label.set_color('white')
#ax.tick_params(axis='both', colors='white')
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
ax.set_axis_off()

fig.savefig('../results/plus_marker/marker.pdf', dpi=450, bbox_inches='tight')
