"""
Visualization Module for Recommender System Analysis.

Provides plotting functions for data exploration and power law analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from collections import Counter


def plot_rating_distribution(
    ratings: np.ndarray,
    title: str = "Rating Distribution",
    figsize: Tuple[int, int] = (10, 6),
    color: str = "#2ecc71",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of rating values.
    
    Args:
        ratings: Array of rating values
        title: Plot title
        figsize: Figure size
        color: Bar color
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Count ratings
    unique, counts = np.unique(ratings, return_counts=True)
    
    ax.bar(unique, counts, width=0.4, color=color, edgecolor='white', linewidth=1.5)
    ax.set_xlabel("Rating", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(unique)
    
    # Add count labels on bars
    for x, y in zip(unique, counts):
        ax.annotate(f'{y:,}', xy=(x, y), ha='center', va='bottom', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_distribution(
    counts: np.ndarray,
    title: str,
    xlabel: str = "Count",
    ylabel: str = "Frequency",
    figsize: Tuple[int, int] = (10, 6),
    color: str = "#3498db",
    log_scale: bool = False,
    bins: int = 50,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of count data (e.g., ratings per user/item).
    
    Args:
        counts: Array of counts
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        color: Histogram color
        log_scale: Whether to use log scale
        bins: Number of histogram bins
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(counts, bins=bins, color=color, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if log_scale:
        ax.set_yscale('log')
    
    # Add statistics
    stats_text = f"Mean: {np.mean(counts):.1f}\nMedian: {np.median(counts):.1f}\nMax: {np.max(counts):,}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_power_law(
    counts: np.ndarray,
    title: str = "Power Law Analysis",
    xlabel: str = "Value (log)",
    ylabel: str = "Frequency (log)",
    figsize: Tuple[int, int] = (10, 6),
    color: str = "#e74c3c",
    fit_line: bool = True,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, Optional[float]]:
    """
    Create log-log plot to analyze power law distribution.
    
    Power law: p(k) ∝ k^(-γ)
    On log-log plot: log(p(k)) = -γ * log(k) + const
    
    Args:
        counts: Array of values (e.g., ratings per user)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        color: Scatter plot color
        fit_line: Whether to fit and plot a line
        save_path: Optional path to save figure
        
    Returns:
        Tuple of (Figure, gamma) where gamma is the power law exponent
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Count frequency of each value
    value_counts = Counter(counts)
    values = np.array(sorted(value_counts.keys()))
    frequencies = np.array([value_counts[v] for v in values])
    
    # Filter out zeros for log
    mask = (values > 0) & (frequencies > 0)
    values = values[mask]
    frequencies = frequencies[mask]
    
    log_values = np.log10(values)
    log_freq = np.log10(frequencies)
    
    ax.scatter(log_values, log_freq, alpha=0.6, s=30, color=color, edgecolors='white', linewidth=0.5)
    
    gamma = None
    if fit_line and len(log_values) > 2:
        # Fit linear regression
        coeffs = np.polyfit(log_values, log_freq, 1)
        gamma = -coeffs[0]  # Power law exponent
        
        fit_line_y = np.polyval(coeffs, log_values)
        ax.plot(log_values, fit_line_y, 'k--', linewidth=2, alpha=0.8,
                label=f'Fit: γ = {gamma:.2f}')
        ax.legend(loc='upper right', fontsize=11)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, gamma


def plot_zipf_rank(
    counts: np.ndarray,
    title: str = "Zipf's Law Analysis",
    xlabel: str = "Rank (log)",
    ylabel: str = "Frequency (log)",
    figsize: Tuple[int, int] = (10, 6),
    color: str = "#9b59b6",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rank vs frequency on log-log scale (Zipf's law).
    
    Zipf's law: frequency ∝ 1/rank
    
    Args:
        counts: Array of counts (e.g., item popularity)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        color: Line color
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort counts in descending order
    sorted_counts = np.sort(counts)[::-1]
    ranks = np.arange(1, len(sorted_counts) + 1)
    
    ax.loglog(ranks, sorted_counts, color=color, linewidth=1.5, alpha=0.8)
    
    # Plot theoretical Zipf line
    zipf_line = sorted_counts[0] / ranks
    ax.loglog(ranks, zipf_line, 'k--', linewidth=1, alpha=0.5, label='1/rank (Zipf)')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_data_exploration_plots(
    ratings: np.ndarray,
    user_counts: np.ndarray,
    item_counts: np.ndarray,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive 2x2 figure with all data exploration plots.
    
    Args:
        ratings: Array of rating values
        user_counts: Number of ratings per user
        item_counts: Number of ratings per item
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Rating distribution
    ax = axes[0, 0]
    unique, counts = np.unique(ratings, return_counts=True)
    ax.bar(unique, counts, width=0.4, color='#2ecc71', edgecolor='white')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating Distribution", fontweight='bold')
    ax.set_xticks(unique)
    
    # 2. User activity distribution
    ax = axes[0, 1]
    ax.hist(user_counts, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_xlabel("Ratings per User")
    ax.set_ylabel("Number of Users")
    ax.set_title("User Activity Distribution", fontweight='bold')
    ax.set_yscale('log')
    
    # 3. Item popularity (Zipf plot)
    ax = axes[1, 0]
    sorted_items = np.sort(item_counts)[::-1]
    ranks = np.arange(1, len(sorted_items) + 1)
    ax.loglog(ranks, sorted_items, color='#9b59b6', linewidth=1.5)
    ax.set_xlabel("Item Rank (log)")
    ax.set_ylabel("Number of Ratings (log)")
    ax.set_title("Item Popularity (Zipf's Law)", fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # 4. Power law analysis for users
    ax = axes[1, 1]
    value_counts = Counter(user_counts)
    values = np.array(sorted(value_counts.keys()))
    frequencies = np.array([value_counts[v] for v in values])
    mask = (values > 0) & (frequencies > 0)
    ax.scatter(np.log10(values[mask]), np.log10(frequencies[mask]), 
               alpha=0.6, s=30, color='#e74c3c', edgecolors='white')
    ax.set_xlabel("log₁₀(Ratings per User)")
    ax.set_ylabel("log₁₀(Frequency)")
    ax.set_title("User Activity Power Law", fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
