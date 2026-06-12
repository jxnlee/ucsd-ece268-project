import io
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Provided table data
data = """
     Batch  Elems/msg    CPU (ms)    GPU (ms)   Speedup   Match
        64          4       123.8       368.5     0.34x       ✓
        64          8       178.0       420.1     0.42x       ✓
        64         16       271.0       521.4     0.52x       ✓
        64         32       466.3       721.6     0.65x       ✓
        64         64       853.2      1121.9     0.76x       ✓
        64        128      1636.0      1923.0     0.85x       ✓
       256          4       490.2       472.1     1.04x       ✓
       256          8       685.8       522.3     1.31x       ✓
       256         16      1086.6       622.6     1.75x       ✓
       256         32      1869.2       823.3     2.27x       ✓
       256         64      3327.8      1224.2     2.72x       ✓
       256        128      6412.8      2026.1     3.17x       ✓
       512          4       955.0       523.4     1.82x       ✓
       512          8      1377.3       576.1     2.39x       ✓
       512         16      2158.9       679.4     3.18x       ✓
       512         32      3737.6       882.0     4.24x       ✓
       512         64      6929.4      1286.8     5.38x       ✓
       512        128     13032.6      2097.6     6.21x       ✓
     1024          4      1950.8       579.9     3.36x       ✓
     1024          8      2735.7       631.0     4.34x       ✓
     1024         16      4308.0       732.4     5.88x       ✓
     1024         32      7419.0       935.4     7.93x       ✓
     1024         64     13653.6      1341.2    10.18x       ✓
     1024        128     26212.2      2151.3    12.18x       ✓
     2048          4      3883.4       635.7     6.11x       ✓
     2048          8      5451.2       686.9     7.94x       ✓
     2048         16      8616.0       788.1    10.93x       ✓
     2048         32     14828.7       992.9    14.94x       ✓
     2048         64     27331.7      1397.0    19.57x       ✓
     2048        128     52360.5      2206.4    23.73x       ✓
     4096          4      7572.0       719.6    10.52x       ✓
     4096          8     10604.5       816.2    12.99x       ✓
     4096         16     17256.8       903.4    19.10x       ✓
     4096         32     29810.8      1134.0    26.29x       ✓
     4096         64     54867.4      1604.3    34.20x       ✓
     4096        128    104779.4      2552.6    41.05x       ✓
     8192          4     15587.0       896.5    17.39x       ✓
     8192          8     21910.1       932.9    23.49x       ✓
     8192         16     33545.3      1062.6    31.57x       ✓
     8192         32     58096.8      1332.8    43.59x       ✓
     8192         64    108949.6      1875.7    58.08x       ✓
     8192        128    208552.9      3009.6    69.30x       ✓
    16384          4     31186.7      1161.1    26.86x       ✓
    16384          8     43769.9      1263.7    34.64x       ✓
    16384         16     68257.3      1482.3    46.05x       ✓
    16384         32    115794.2      1983.9    58.37x       ✓
    16384         64    218239.7      2907.3    75.07x       ✓
    16384        128    412735.2      4739.8    87.08x       ✓
"""

# Parse rows manually due to multi-word headers
lines = [line.strip().split() for line in data.strip().split('\n')[1:] if line.strip()]
columns = ['Batch', 'Elems/msg', 'CPU_ms', 'GPU_ms', 'Speedup', 'Match']
df = pd.DataFrame(lines, columns=columns)

# Convert types
df['Batch'] = df['Batch'].astype(int)
df['Elems/msg'] = df['Elems/msg'].astype(int)
df['CPU_ms'] = df['CPU_ms'].astype(float)
df['GPU_ms'] = df['GPU_ms'].astype(float)
df['Speedup'] = df['Speedup'].str.replace('x', '').astype(float)

# Set visual style
sns.set_theme(style="whitegrid")

# Plot 1: Speedup vs Batch Size
fig, ax = plt.subplots(figsize=(10, 6))
for elems in df['Elems/msg'].unique():
    subset = df[df['Elems/msg'] == elems]
    ax.plot(subset['Batch'], subset['Speedup'], marker='o', label=f'{elems} Elems/msg')

ax.set_xscale('log', base=2)
ax.set_xticks(df['Batch'].unique())
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Batch Size (log scale)')
ax.set_ylabel('Speedup (x)')
ax.set_title('GPU Speedup over CPU by Batch Size and Elems/msg')
ax.legend(title='Elements per Message')
plt.tight_layout()
plt.savefig('speedup_vs_batch.png', dpi=300)
plt.close()

# Plot 2: CPU vs GPU Execution Time vs Batch Size
fig, ax = plt.subplots(figsize=(10, 6))
selected_elems = [4, 32, 128]
colors = ['blue', 'green', 'red']

for i, elems in enumerate(selected_elems):
    subset = df[df['Elems/msg'] == elems]
    ax.plot(subset['Batch'], subset['CPU_ms'], marker='o', linestyle='-', color=colors[i], label=f'CPU ({elems} Elems/msg)')
    ax.plot(subset['Batch'], subset['GPU_ms'], marker='s', linestyle='--', color=colors[i], label=f'GPU ({elems} Elems/msg)')

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)
ax.set_xticks(df['Batch'].unique())
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Batch Size (log scale)')
ax.set_ylabel('Execution Time (ms, log scale)')
ax.set_title('Execution Time Comparison: CPU vs GPU')
ax.legend(title='Device & Config', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('execution_time_comparison.png', dpi=300)
plt.close()

# Plot 3: Speedup vs Elems/msg
fig, ax = plt.subplots(figsize=(10, 6))
for batch in df['Batch'].unique():
    subset = df[df['Batch'] == batch]
    ax.plot(subset['Elems/msg'], subset['Speedup'], marker='o', label=f'Batch {batch}')

ax.set_xscale('log', base=2)
ax.set_xticks(df['Elems/msg'].unique())
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Elements per Message (log scale)')
ax.set_ylabel('Speedup (x)')
ax.set_title('GPU Speedup by Elements per Message')
ax.legend(title='Batch Size', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('speedup_vs_elems.png', dpi=300)
plt.close()