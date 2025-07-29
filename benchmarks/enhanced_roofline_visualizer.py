#!/usr/bin/env python3
"""
Enhanced roofline visualization with interactive plots and multi-precision support
Part of Omni-Perf-Bench Phase 1 improvements
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch

from utils.metrics_utils import detect_gpu_architecture, calculate_roofline_ceilings

class EnhancedRooflineVisualizer:
    """Advanced roofline visualization with multiple precision levels and memory hierarchies"""
    
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.cuda.current_device()
        
        self.device = device
        self.device_caps = detect_gpu_architecture(device)
        self.device_name = torch.cuda.get_device_name(device)
        
        # Calculate all performance ceilings
        props = torch.cuda.get_device_properties(device)
        self.clock_rate_ghz = props.clock_rate / 1e6
        self.ceilings = calculate_roofline_ceilings(self.device_caps, self.clock_rate_ghz)
        
        # Color scheme for different kernel types
        self.color_map = {
            'layer_norm': '#1f77b4',
            'softmax': '#ff7f0e',
            'diagonal_matmul': '#2ca02c',
            'fused_linear_rowsum': '#d62728',
            'attention': '#9467bd',
            'default': '#7f7f7f'
        }
    
    def create_interactive_roofline(self, results_data: List[Dict], 
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> go.Figure:
        """Create interactive roofline plot using Plotly"""
        
        fig = go.Figure()
        
        # Arithmetic intensity range
        ai_range = np.logspace(-2, 3, 1000)
        
        # Add roofline for different precision levels
        precision_configs = [
            ('FP32', self.ceilings['fp32_peak_tflops'], 
             self.ceilings['dram_bandwidth_gbps'], 'blue', 'solid'),
        ]
        
        if self.device_caps.has_tensor_cores:
            precision_configs.append(
                ('Tensor Core', self.ceilings['tensor_peak_tflops'],
                 self.ceilings['dram_bandwidth_gbps'], 'green', 'dash')
            )
        
        if self.device_caps.has_bf16:
            precision_configs.append(
                ('BF16', self.ceilings.get('bf16_peak_tflops', 
                                          self.ceilings['tensor_peak_tflops'] * 2),
                 self.ceilings['dram_bandwidth_gbps'], 'orange', 'dot')
            )
        
        # Plot rooflines
        for name, peak_tflops, bandwidth_gbps, color, dash in precision_configs:
            memory_bound = bandwidth_gbps * 1e9 * ai_range / 1e12
            compute_bound = np.ones_like(ai_range) * peak_tflops
            roofline = np.minimum(memory_bound, compute_bound)
            
            fig.add_trace(go.Scatter(
                x=ai_range,
                y=roofline,
                mode='lines',
                name=f'{name} Roofline',
                line=dict(color=color, width=3, dash=dash),
                hovertemplate='AI: %{x:.2f}<br>Peak: %{y:.2f} TFLOP/s'
            ))
            
            # Add ridge point
            ridge_point = peak_tflops * 1e12 / (bandwidth_gbps * 1e9)
            fig.add_trace(go.Scatter(
                x=[ridge_point],
                y=[peak_tflops],
                mode='markers',
                name=f'{name} Ridge Point',
                marker=dict(symbol='x', size=12, color=color),
                showlegend=False,
                hovertemplate=f'{name} Ridge Point<br>AI: %{{x:.2f}}<br>Peak: %{{y:.2f}} TFLOP/s'
            ))
        
        # Add memory hierarchy lines
        if 'l2_bandwidth_gbps' in self.ceilings:
            l2_bound = self.ceilings['l2_bandwidth_gbps'] * 1e9 * ai_range / 1e12
            fig.add_trace(go.Scatter(
                x=ai_range,
                y=l2_bound,
                mode='lines',
                name='L2 Cache Limit',
                line=dict(color='purple', width=2, dash='dashdot'),
                visible='legendonly'  # Hidden by default
            ))
        
        # Plot kernel performance points
        kernels_by_type = {}
        for result in results_data:
            if 'arithmetic_intensity' not in result or 'achieved_tflops' not in result:
                continue
            
            kernel_type = result.get('kernel', 'unknown')
            if kernel_type not in kernels_by_type:
                kernels_by_type[kernel_type] = {
                    'ai': [], 'tflops': [], 'text': [], 'sizes': []
                }
            
            kernels_by_type[kernel_type]['ai'].append(result['arithmetic_intensity'])
            kernels_by_type[kernel_type]['tflops'].append(result['achieved_tflops'])
            
            # Create hover text
            hover_text = f"Kernel: {kernel_type}<br>"
            hover_text += f"Size: {result.get('batch_size', '?')}x{result.get('seq_len', '?')}"
            if result.get('hidden_dim'):
                hover_text += f"x{result['hidden_dim']}"
            hover_text += f"<br>AI: {result['arithmetic_intensity']:.2f} FLOP/byte"
            hover_text += f"<br>Performance: {result['achieved_tflops']:.2f} TFLOP/s"
            
            if 'occupancy_pct' in result:
                hover_text += f"<br>Occupancy: {result['occupancy_pct']:.1f}%"
            if 'sm_efficiency_pct' in result:
                hover_text += f"<br>SM Efficiency: {result['sm_efficiency_pct']:.1f}%"
            
            kernels_by_type[kernel_type]['text'].append(hover_text)
            
            # Size for bubble chart effect
            size = 10 + (result.get('batch_size', 32) / 32) * 5
            kernels_by_type[kernel_type]['sizes'].append(size)
        
        # Add kernel points
        for kernel_type, data in kernels_by_type.items():
            color = self.color_map.get(kernel_type, self.color_map['default'])
            
            fig.add_trace(go.Scatter(
                x=data['ai'],
                y=data['tflops'],
                mode='markers',
                name=kernel_type,
                marker=dict(
                    size=data['sizes'],
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                text=data['text'],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=title or f'Interactive Roofline Model - {self.device_name}',
            xaxis=dict(
                title='Arithmetic Intensity (FLOP/byte)',
                type='log',
                gridcolor='lightgray',
                range=[-2, 3]
            ),
            yaxis=dict(
                title='Performance (TFLOP/s)',
                type='log',
                gridcolor='lightgray',
                range=[-2, np.log10(max(self.ceilings['fp32_peak_tflops'] * 2, 100))]
            ),
            hovermode='closest',
            width=1200,
            height=800,
            template='plotly_white'
        )
        
        # Add annotations
        fig.add_annotation(
            x=-1, y=np.log10(self.ceilings['fp32_peak_tflops'] * 0.7),
            text="Memory Bound",
            showarrow=False,
            textangle=-45,
            font=dict(size=14, color='gray')
        )
        
        fig.add_annotation(
            x=2, y=np.log10(self.ceilings['fp32_peak_tflops'] * 0.5),
            text="Compute Bound",
            showarrow=False,
            font=dict(size=14, color='gray')
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive roofline saved to: {save_path}")
        
        return fig
    
    def create_performance_radar(self, results_data: List[Dict],
                               save_path: Optional[str] = None) -> go.Figure:
        """Create radar chart showing multiple performance metrics"""
        
        # Aggregate metrics by kernel type
        kernel_metrics = {}
        
        for result in results_data:
            kernel = result.get('kernel', 'unknown')
            if kernel not in kernel_metrics:
                kernel_metrics[kernel] = {
                    'occupancy': [],
                    'sm_efficiency': [],
                    'memory_efficiency': [],
                    'compute_efficiency': [],
                    'speedup': []
                }
            
            # Collect available metrics
            if 'occupancy_pct' in result:
                kernel_metrics[kernel]['occupancy'].append(result['occupancy_pct'])
            if 'sm_efficiency_pct' in result:
                kernel_metrics[kernel]['sm_efficiency'].append(result['sm_efficiency_pct'])
            if 'memory_efficiency_pct' in result:
                kernel_metrics[kernel]['memory_efficiency'].append(result['memory_efficiency_pct'])
            if 'compute_efficiency_pct' in result:
                kernel_metrics[kernel]['compute_efficiency'].append(result['compute_efficiency_pct'])
            if 'speedup' in result:
                kernel_metrics[kernel]['speedup'].append(min(result['speedup'] * 10, 100))  # Scale speedup
        
        # Calculate averages
        fig = go.Figure()
        
        categories = ['Occupancy', 'SM Efficiency', 'Memory Efficiency', 
                     'Compute Efficiency', 'Speedup (x10)']
        
        for kernel, metrics in kernel_metrics.items():
            values = []
            for metric_key in ['occupancy', 'sm_efficiency', 'memory_efficiency', 
                             'compute_efficiency', 'speedup']:
                if metrics[metric_key]:
                    values.append(np.mean(metrics[metric_key]))
                else:
                    values.append(0)
            
            # Close the radar chart
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=kernel,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Kernel Performance Radar Chart",
            width=800,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Radar chart saved to: {save_path}")
        
        return fig
    
    def create_efficiency_heatmap(self, results_data: List[Dict],
                                save_path: Optional[str] = None) -> go.Figure:
        """Create heatmap showing efficiency across different configurations"""
        
        # Organize data for heatmap
        data_matrix = []
        row_labels = []
        col_labels = []
        
        # Group by kernel and configuration
        grouped_data = {}
        for result in results_data:
            kernel = result.get('kernel', 'unknown')
            config_str = f"{result.get('batch_size', '?')}x{result.get('seq_len', '?')}x{result.get('hidden_dim', '?')}"
            
            if kernel not in grouped_data:
                grouped_data[kernel] = {}
            
            # Use compute efficiency as primary metric
            efficiency = result.get('compute_efficiency_pct', 0)
            grouped_data[kernel][config_str] = efficiency
        
        # Create matrix
        all_configs = sorted(set(config for kernel_data in grouped_data.values() 
                               for config in kernel_data.keys()))
        
        for kernel in sorted(grouped_data.keys()):
            row = []
            for config in all_configs:
                row.append(grouped_data[kernel].get(config, 0))
            data_matrix.append(row)
            row_labels.append(kernel)
        
        col_labels = all_configs
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=col_labels,
            y=row_labels,
            colorscale='Viridis',
            text=[[f'{val:.1f}%' for val in row] for row in data_matrix],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Compute Efficiency %")
        ))
        
        fig.update_layout(
            title="Kernel Compute Efficiency Heatmap",
            xaxis_title="Configuration (Batch x Seq x Hidden)",
            yaxis_title="Kernel Type",
            width=1000,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Efficiency heatmap saved to: {save_path}")
        
        return fig
    
    def create_comparison_dashboard(self, results_data: List[Dict],
                                  save_path: Optional[str] = None) -> go.Figure:
        """Create comprehensive dashboard with multiple visualizations"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance vs Problem Size', 
                          'Efficiency Distribution',
                          'Memory vs Compute Bound', 
                          'Speedup by Data Type'),
            specs=[[{"type": "scatter"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Performance vs Problem Size
        for result in results_data:
            if 'achieved_tflops' not in result:
                continue
                
            problem_size = (result.get('batch_size', 1) * 
                          result.get('seq_len', 1) * 
                          result.get('hidden_dim', 1))
            
            kernel = result.get('kernel', 'unknown')
            color = self.color_map.get(kernel, self.color_map['default'])
            
            fig.add_trace(go.Scatter(
                x=[problem_size],
                y=[result['achieved_tflops']],
                mode='markers',
                name=kernel,
                marker=dict(color=color, size=8),
                showlegend=False
            ), row=1, col=1)
        
        # 2. Efficiency Distribution
        efficiency_data = {}
        for result in results_data:
            kernel = result.get('kernel', 'unknown')
            if kernel not in efficiency_data:
                efficiency_data[kernel] = []
            
            if 'compute_efficiency_pct' in result:
                efficiency_data[kernel].append(result['compute_efficiency_pct'])
        
        for kernel, efficiencies in efficiency_data.items():
            color = self.color_map.get(kernel, self.color_map['default'])
            fig.add_trace(go.Box(
                y=efficiencies,
                name=kernel,
                marker_color=color,
                showlegend=False
            ), row=1, col=2)
        
        # 3. Memory vs Compute Bound Analysis
        for result in results_data:
            if not all(k in result for k in ['arithmetic_intensity', 'achieved_tflops']):
                continue
            
            kernel = result.get('kernel', 'unknown')
            ai = result['arithmetic_intensity']
            
            # Determine if memory or compute bound
            ridge_point = self.ceilings['fp32_ridge_point']
            is_memory_bound = ai < ridge_point
            
            color = 'red' if is_memory_bound else 'blue'
            symbol = 'circle' if is_memory_bound else 'square'
            
            fig.add_trace(go.Scatter(
                x=[ai],
                y=[result['achieved_tflops']],
                mode='markers',
                marker=dict(color=color, symbol=symbol, size=10),
                name='Memory Bound' if is_memory_bound else 'Compute Bound',
                showlegend=False,
                hovertext=f"{kernel}: {'Memory' if is_memory_bound else 'Compute'} Bound"
            ), row=2, col=1)
        
        # 4. Speedup by Data Type
        dtype_speedups = {}
        for result in results_data:
            if 'speedup' not in result:
                continue
                
            dtype = result.get('dtype', 'float32')
            kernel = result.get('kernel', 'unknown')
            
            if dtype not in dtype_speedups:
                dtype_speedups[dtype] = {}
            if kernel not in dtype_speedups[dtype]:
                dtype_speedups[dtype][kernel] = []
            
            dtype_speedups[dtype][kernel].append(result['speedup'])
        
        # Calculate average speedups
        for dtype, kernel_data in dtype_speedups.items():
            kernels = []
            speedups = []
            
            for kernel, values in kernel_data.items():
                kernels.append(kernel)
                speedups.append(np.mean(values))
            
            fig.add_trace(go.Bar(
                x=kernels,
                y=speedups,
                name=dtype,
                showlegend=True
            ), row=2, col=2)
        
        # Update layout
        fig.update_xaxes(title_text="Problem Size", row=1, col=1, type="log")
        fig.update_yaxes(title_text="TFLOP/s", row=1, col=1)
        
        fig.update_yaxes(title_text="Compute Efficiency %", row=1, col=2)
        
        fig.update_xaxes(title_text="Arithmetic Intensity", row=2, col=1, type="log")
        fig.update_yaxes(title_text="TFLOP/s", row=2, col=1, type="log")
        
        fig.update_xaxes(title_text="Kernel", row=2, col=2)
        fig.update_yaxes(title_text="Average Speedup", row=2, col=2)
        
        fig.update_layout(
            title_text="Kernel Performance Dashboard",
            height=1000,
            width=1400,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Dashboard saved to: {save_path}")
        
        return fig
    
    def generate_full_report(self, results_data: List[Dict], 
                           output_dir: str = "benchmarks/visualizations/roofline"):
        """Generate complete set of visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate all visualizations
        print("Generating enhanced visualizations...")
        
        # 1. Interactive roofline
        self.create_interactive_roofline(
            results_data,
            save_path=str(output_path / f"roofline_interactive_{timestamp}.html")
        )
        
        # 2. Performance radar
        self.create_performance_radar(
            results_data,
            save_path=str(output_path / f"performance_radar_{timestamp}.html")
        )
        
        # 3. Efficiency heatmap
        self.create_efficiency_heatmap(
            results_data,
            save_path=str(output_path / f"efficiency_heatmap_{timestamp}.html")
        )
        
        # 4. Comparison dashboard
        self.create_comparison_dashboard(
            results_data,
            save_path=str(output_path / f"performance_dashboard_{timestamp}.html")
        )
        
        # Generate static matplotlib plots as well
        self._generate_static_plots(results_data, output_path, timestamp)
        
        print(f"\nAll visualizations saved to: {output_path}")
        
        # Create index HTML
        self._create_index_html(output_path, timestamp)
    
    def _generate_static_plots(self, results_data: List[Dict], 
                             output_path: Path, timestamp: str):
        """Generate traditional matplotlib plots"""
        
        # Create static roofline
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot rooflines
        ai_range = np.logspace(-2, 3, 1000)
        
        # FP32 roofline
        memory_bound = self.ceilings['dram_bandwidth_gbps'] * 1e9 * ai_range / 1e12
        compute_bound = np.ones_like(ai_range) * self.ceilings['fp32_peak_tflops']
        roofline = np.minimum(memory_bound, compute_bound)
        
        ax.loglog(ai_range, roofline, 'b-', linewidth=3, label='FP32 Roofline')
        
        # Plot kernel points
        for kernel_type in set(r.get('kernel') for r in results_data):
            kernel_results = [r for r in results_data if r.get('kernel') == kernel_type]
            
            ai_values = [r['arithmetic_intensity'] for r in kernel_results 
                        if 'arithmetic_intensity' in r]
            tflops_values = [r['achieved_tflops'] for r in kernel_results 
                           if 'achieved_tflops' in r]
            
            if ai_values and tflops_values:
                color = self.color_map.get(kernel_type, self.color_map['default'])
                ax.scatter(ai_values, tflops_values, 
                         s=100, alpha=0.7, color=color, 
                         label=kernel_type, edgecolors='black')
        
        ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
        ax.set_ylabel('Performance (TFLOP/s)', fontsize=12)
        ax.set_title(f'Roofline Model - {self.device_name}', fontsize=14, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / f"roofline_static_{timestamp}.png", dpi=300)
        plt.close()
    
    def _create_index_html(self, output_path: Path, timestamp: str):
        """Create index HTML page linking all visualizations"""
        
        index_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Omni-Perf-Bench Visualization Report - {timestamp}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .viz-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            transition: transform 0.2s;
        }}
        .viz-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .viz-card h3 {{
            color: #555;
            margin-bottom: 10px;
        }}
        .viz-card a {{
            text-decoration: none;
            color: #007bff;
            font-size: 16px;
        }}
        .viz-card a:hover {{
            text-decoration: underline;
        }}
        .info {{
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Omni-Perf-Bench Visualization Report</h1>
        <p style="text-align: center; color: #666;">Generated: {timestamp}</p>
        <p style="text-align: center; color: #666;">Device: {self.device_name}</p>
        
        <div class="viz-grid">
            <div class="viz-card">
                <h3>Interactive Roofline Model</h3>
                <p>Explore kernel performance relative to hardware limits</p>
                <a href="roofline_interactive_{timestamp}.html" target="_blank">Open Interactive Plot</a>
            </div>
            
            <div class="viz-card">
                <h3>Performance Radar Chart</h3>
                <p>Multi-dimensional performance metrics comparison</p>
                <a href="performance_radar_{timestamp}.html" target="_blank">Open Radar Chart</a>
            </div>
            
            <div class="viz-card">
                <h3>Efficiency Heatmap</h3>
                <p>Compute efficiency across kernels and configurations</p>
                <a href="efficiency_heatmap_{timestamp}.html" target="_blank">Open Heatmap</a>
            </div>
            
            <div class="viz-card">
                <h3>Performance Dashboard</h3>
                <p>Comprehensive performance analysis dashboard</p>
                <a href="performance_dashboard_{timestamp}.html" target="_blank">Open Dashboard</a>
            </div>
            
            <div class="viz-card">
                <h3>Static Roofline Plot</h3>
                <p>Traditional roofline visualization (PNG)</p>
                <a href="roofline_static_{timestamp}.png" target="_blank">View Image</a>
            </div>
        </div>
        
        <div class="info">
            <h3>Navigation Tips:</h3>
            <ul>
                <li>Interactive plots support zoom, pan, and hover for detailed information</li>
                <li>Use the legend to show/hide specific kernel types</li>
                <li>Double-click on the plot to reset the view</li>
                <li>Click and drag to zoom into specific regions</li>
            </ul>
        </div>
    </div>
</body>
</html>
        """
        
        with open(output_path / "index.html", 'w') as f:
            f.write(index_content)
        
        print(f"Visualization index created at: {output_path / 'index.html'}")

def main():
    """Generate enhanced visualizations from benchmark data"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate enhanced roofline visualizations'
    )
    parser.add_argument('--data-dir', type=str, 
                       default='benchmarks/data/unified',
                       help='Directory containing benchmark results')
    parser.add_argument('--output-dir', type=str,
                       default='benchmarks/visualizations/roofline',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load benchmark results
    results = []
    data_path = Path(args.data_dir)
    
    for json_file in data_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # Handle different data formats
            if isinstance(data, list):
                for item in data:
                    if 'metrics' in item:
                        # Unified benchmark format
                        result = item['config'].copy()
                        result.update(item['metrics'])
                        result['kernel'] = item['kernel']
                        results.append(result)
                    else:
                        results.append(item)
            elif 'results' in data:
                results.extend(data['results'])
    
    if not results:
        print(f"No benchmark results found in {args.data_dir}")
        return
    
    print(f"Found {len(results)} benchmark results")
    
    # Create visualizer and generate report
    visualizer = EnhancedRooflineVisualizer()
    visualizer.generate_full_report(results, args.output_dir)

if __name__ == "__main__":
    main()