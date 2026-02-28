"""
Event-Driven Backtester
-----------------------
Clean, minimal implementation for earnings event-based portfolio evaluation.

Core logic:
1. Sort events by pressure_score into quintiles
2. Compute post-earnings cumulative returns
3. Compare Q5 (info) vs Q1 (liquidity)

Two entry conventions:
- t=0: Enter at first trading day >= earnings_date (at close)
- t=+1: Enter at second trading day >= earnings_date (at open)

Two holding periods:
- [0, +5] trading days
- [0, +20] trading days

CRITICAL FIX: Properly handles after-hours/non-trading-day earnings announcements
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from pathlib import Path


class EventBacktester:
    """
    Event-driven backtester for quintile-based portfolio evaluation.
    
    Focus: Simplicity and interpretability over abstraction.
    """
    
    def __init__(self, event_df: pd.DataFrame, returns_df: pd.DataFrame):
        """
        Parameters
        ----------
        event_df : pd.DataFrame
            Event-level data with columns:
            - event_id: unique identifier
            - permno: stock identifier
            - earnings_date: announcement date (calendar date, may be non-trading day)
            - pressure_score: continuous score in [-1, 1]
            
        returns_df : pd.DataFrame
            Daily returns with columns:
            - permno: stock identifier
            - date: trading date
            - ret: daily return (decimal, not percent)
        """
        self.event_df = event_df.copy()
        self.returns_df = returns_df.copy()
        
        # Validate required columns
        self._validate_inputs()
        
        # Ensure proper date types
        self.event_df['earnings_date'] = pd.to_datetime(self.event_df['earnings_date'])
        self.returns_df['date'] = pd.to_datetime(self.returns_df['date'])
        
        # Sort for efficient lookups
        self.returns_df = self.returns_df.sort_values(['permno', 'date'])
        
        
    def _validate_inputs(self):
        """Validate required columns exist."""
        required_event_cols = ['event_id', 'permno', 'earnings_date', 'pressure_score']
        required_return_cols = ['permno', 'date', 'ret']
        
        missing_event = set(required_event_cols) - set(self.event_df.columns)
        missing_return = set(required_return_cols) - set(self.returns_df.columns)
        
        if missing_event:
            raise ValueError(f"Missing columns in event_df: {missing_event}")
        if missing_return:
            raise ValueError(f"Missing columns in returns_df: {missing_return}")
    
    
    def assign_quintiles(self) -> pd.DataFrame:
        """
        Assign quintile labels to events based on pressure_score.
        
        Q1 = lowest scores (liquidity-dominated)
        Q5 = highest scores (information-dominated)
        
        Returns
        -------
        pd.DataFrame
            Event data with additional 'quintile' column (1-5)
        """
        df = self.event_df.copy()
        
        # Handle NaN scores by excluding them
        valid_mask = ~df['pressure_score'].isna()
        
        if valid_mask.sum() == 0:
            raise ValueError("No valid pressure_score values found")
        
        # Quintile assignment (qcut handles ties automatically)
        df.loc[valid_mask, 'quintile'] = pd.qcut(
            df.loc[valid_mask, 'pressure_score'],
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'  # Handle case where unique values < 5
        )
        
        return df
    
    
    def assign_extreme_bins(self, top_pct: float = 10, bottom_pct: float = 10) -> pd.DataFrame:
        """
        Assign extreme bins instead of quintiles for concentrated signal testing.
        
        Parameters
        ----------
        top_pct : float
            Top percentile (e.g., 10 for top 10%)
        bottom_pct : float
            Bottom percentile (e.g., 10 for bottom 10%)
        
        Returns
        -------
        pd.DataFrame
            Event data with 'bin' column: 'top', 'bottom', or 'middle'
        """
        df = self.event_df.copy()
        
        # Handle NaN scores by excluding them
        valid_mask = ~df['pressure_score'].isna()
        
        if valid_mask.sum() == 0:
            raise ValueError("No valid pressure_score values found")
        
        # Calculate thresholds
        top_threshold = df.loc[valid_mask, 'pressure_score'].quantile(1 - top_pct/100)
        bottom_threshold = df.loc[valid_mask, 'pressure_score'].quantile(bottom_pct/100)
        
        # Assign bins (default to middle)
        df.loc[valid_mask, 'bin'] = 'middle'
        df.loc[valid_mask & (df['pressure_score'] >= top_threshold), 'bin'] = 'top'
        df.loc[valid_mask & (df['pressure_score'] <= bottom_threshold), 'bin'] = 'bottom'
        
        print(f"  Extreme bins ({top_pct}% / {bottom_pct}%):")
        print(f"    Top:    {(df['bin'] == 'top').sum()} events (score >= {top_threshold:.3f})")
        print(f"    Bottom: {(df['bin'] == 'bottom').sum()} events (score <= {bottom_threshold:.3f})")
        print(f"    Middle: {(df['bin'] == 'middle').sum()} events")
        
        return df
    
    
    def compute_event_returns(
        self,
        event_row: pd.Series,
        entry_timing: str,
        horizon: int
    ) -> float:
        """
        Compute cumulative return for a single event.
        
        CRITICAL: Earnings announcements may occur after-hours or on non-trading days.
        We define t=0 as the FIRST trading day >= earnings_date.
        
        This ensures:
        - Weekend/holiday announcements map to next trading day
        - After-hours announcements use next day's open (for t=+1)
        
        Parameters
        ----------
        event_row : pd.Series
            Single row from event dataframe
        entry_timing : str
            't=0' (enter at t=0 close) or 't=+1' (enter at t=+1 open)
        horizon : int
            Number of trading days to hold (e.g., 5 or 20)
            
        Returns
        -------
        float
            Cumulative return, or NaN if insufficient data
        """
        permno = event_row['permno']
        earnings_date = event_row['earnings_date']
        
        # Get return series for this stock
        stock_returns = self.returns_df[
            self.returns_df['permno'] == permno
        ].set_index('date')['ret']
        
        if len(stock_returns) == 0:
            return np.nan
        
        # CRITICAL FIX: Find first trading day >= earnings_date
        # This handles after-hours and non-trading-day announcements correctly
        valid_dates = stock_returns.index[stock_returns.index >= earnings_date]
        
        if len(valid_dates) == 0:
            return np.nan  # No future data available
        
        t0_date = valid_dates[0]  # First trading day on/after earnings announcement
        
        # Determine entry based on timing convention
        if entry_timing == 't=0':
            # Enter at t=0 close (include t=0 return)
            entry_date = t0_date
        elif entry_timing == 't=+1':
            # Enter at t=+1 open (skip t=0 return, start from t=+1)
            if len(valid_dates) < 2:
                return np.nan  # Need at least t=0 and t=+1
            entry_date = valid_dates[1]
        else:
            raise ValueError(f"Invalid entry_timing: {entry_timing}")
        
        # Get position in index for slicing
        start_idx = stock_returns.index.get_loc(entry_date)
        end_idx = start_idx + horizon
        
        if end_idx > len(stock_returns):
            return np.nan  # Not enough forward data
        
        # Extract return window
        returns_slice = stock_returns.iloc[start_idx:end_idx]
        
        if returns_slice.isna().any():
            return np.nan  # Missing returns in window
        
        # Compute cumulative return: (1+r1)*(1+r2)*...-1
        cum_return = (1 + returns_slice).prod() - 1
        
        return cum_return
    
    
    def run_quintile_backtest(
        self,
        entry_timing: str,
        horizon: int
    ) -> pd.DataFrame:
        """
        Run quintile backtest for given entry timing and horizon.
        
        Includes data loss tracking for interpretability.
        
        Parameters
        ----------
        entry_timing : str
            't=0' or 't=+1'
        horizon : int
            Holding period in trading days
            
        Returns
        -------
        pd.DataFrame
            Event-level results with columns:
            - event_id
            - permno
            - earnings_date
            - pressure_score
            - quintile
            - cum_return
        """
        # Assign quintiles
        df = self.assign_quintiles()
        n_total = len(df)
        
        # Compute returns for each event
        df['cum_return'] = df.apply(
            lambda row: self.compute_event_returns(row, entry_timing, horizon),
            axis=1
        )
        
        # Track data loss (important for interpretation)
        n_valid = df['cum_return'].notna().sum()
        pct_loss = 100 * (n_total - n_valid) / n_total
        
        print(f"  Data availability: {n_valid}/{n_total} events ({pct_loss:.1f}% loss)")
        
        # Per-quintile data availability
        quintile_stats = df.groupby('quintile').agg({
            'cum_return': lambda x: f"{x.notna().sum()}/{len(x)}"
        })
        print(f"  By quintile (valid/total):")
        for q, row in quintile_stats.iterrows():
            print(f"    Q{int(q)}: {row['cum_return']}")
        
        # Keep relevant columns
        result_df = df[[
            'event_id', 'permno', 'earnings_date',
            'pressure_score', 'quintile', 'cum_return'
        ]].copy()
        
        return result_df
    
    
    def compute_quintile_summary(
        self,
        backtest_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute summary statistics by quintile.
        
        NOTE: Does NOT include Sharpe ratio.
        Event-based returns with overlapping horizons violate independence assumptions.
        Statistical significance should be evaluated via t-statistics or Fama-MacBeth.
        
        Parameters
        ----------
        backtest_results : pd.DataFrame
            Output from run_quintile_backtest()
            
        Returns
        -------
        pd.DataFrame
            Summary table with:
            - quintile (1-5)
            - n_events: number of valid events
            - mean_return: average cumulative return
            - std_return: standard deviation
        """
        summary = backtest_results.groupby('quintile').agg({
            'cum_return': ['count', 'mean', 'std']
        }).reset_index()
        
        summary.columns = ['quintile', 'n_events', 'mean_return', 'std_return']
        
        return summary
    
    
    def compute_long_short(
        self,
        summary_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute Q5 - Q1 long-short return.
        
        Parameters
        ----------
        summary_df : pd.DataFrame
            Output from compute_quintile_summary()
            
        Returns
        -------
        dict
            {'long_short_return': float, 'q5_return': float, 'q1_return': float}
        """
        q5_return = summary_df[summary_df['quintile'] == 5]['mean_return'].values[0]
        q1_return = summary_df[summary_df['quintile'] == 1]['mean_return'].values[0]
        
        return {
            'long_short_return': q5_return - q1_return,
            'q5_return': q5_return,
            'q1_return': q1_return
        }
    
    
    def run_full_analysis(
        self,
        entry_timings: list = ['t=0', 't=+1'],
        horizons: list = [5, 20]
    ) -> Dict[str, pd.DataFrame]:
        """
        Run complete analysis across all entry timings and horizons.
        
        Parameters
        ----------
        entry_timings : list
            List of entry timing conventions
        horizons : list
            List of holding periods (in trading days)
            
        Returns
        -------
        dict
            Nested dictionary with structure:
            {
                ('t=0', 5): {'backtest': df, 'summary': df, 'long_short': dict},
                ('t=+1', 5): {...},
                ...
            }
        """
        results = {}
        
        for entry in entry_timings:
            for h in horizons:
                print(f"\nRunning: entry={entry}, horizon={h} days")
                
                # Run backtest
                backtest_df = self.run_quintile_backtest(entry, h)
                
                # Compute summary
                summary_df = self.compute_quintile_summary(backtest_df)
                
                # Compute long-short
                ls_stats = self.compute_long_short(summary_df)
                
                # Store results
                key = (entry, h)
                results[key] = {
                    'backtest': backtest_df,
                    'summary': summary_df,
                    'long_short': ls_stats
                }
                
                # Print quick summary
                print(f"  Q1 return: {ls_stats['q1_return']:.4f}")
                print(f"  Q5 return: {ls_stats['q5_return']:.4f}")
                print(f"  Q5-Q1:     {ls_stats['long_short_return']:.4f}")
        
        return results
    
    
    def run_extreme_bins_backtest(
        self,
        entry_timing: str,
        horizon: int,
        top_pct: float = 10,
        bottom_pct: float = 10
    ) -> Dict[str, float]:
        """
        Run backtest on extreme bins (top/bottom percentiles).
        
        Parameters
        ----------
        entry_timing : str
            't=0' or 't=+1'
        horizon : int
            Holding period in trading days
        top_pct : float
            Top percentile
        bottom_pct : float
            Bottom percentile
            
        Returns
        -------
        dict
            {'top_return': float, 'bottom_return': float, 'spread': float,
             'top_n': int, 'bottom_n': int}
        """
        # Assign bins
        df = self.assign_extreme_bins(top_pct, bottom_pct)
        
        # Compute returns for each event
        df['cum_return'] = df.apply(
            lambda row: self.compute_event_returns(row, entry_timing, horizon),
            axis=1
        )
        
        # Calculate statistics for each bin
        top_returns = df[df['bin'] == 'top']['cum_return'].dropna()
        bottom_returns = df[df['bin'] == 'bottom']['cum_return'].dropna()
        
        top_mean = top_returns.mean() if len(top_returns) > 0 else np.nan
        bottom_mean = bottom_returns.mean() if len(bottom_returns) > 0 else np.nan
        spread = top_mean - bottom_mean
        
        return {
            'top_return': top_mean,
            'bottom_return': bottom_mean,
            'spread': spread,
            'top_n': len(top_returns),
            'bottom_n': len(bottom_returns)
        }
    
    
    def plot_car_comparison(
        self,
        backtest_results: pd.DataFrame,
        quintiles_to_plot: list = [1, 5],
        max_days: int = 20
    ):
        """
        Plot Cumulative Average Return (CAR) for selected quintiles.
        
        NOTE: This requires recomputing returns day-by-day.
        For now, this is a placeholder. Implement if needed in Week 4.
        """
        raise NotImplementedError(
            "CAR plotting requires day-by-day return tracking. "
            "Current implementation computes only terminal cumulative returns. "
            "Extend compute_event_returns() to return full time series if needed."
        )


# ============================================================================
# TEST CODE - Runs when script is executed directly
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Event-Driven Backtester - Real Data Test")
    print("="*70)
    
    # ========================================================================
    # Load Real Data
    # ========================================================================
    
    # Load event features (Week 2 output)
    event_file = "Methods_Sung_Cho/D_event_features.parquet"
    print(f"\nLoading event data: {event_file}")
    
    try:
        events_raw = pd.read_parquet(event_file)
        print(f"✓ Loaded {len(events_raw)} rows")
        
        # Handle column naming variations
        date_col = None
        for col in ['earnings_date', 'rdq', 'event_date', 'date']:
            if col in events_raw.columns:
                date_col = col
                break
        
        if date_col and date_col != 'earnings_date':
            events_raw = events_raw.rename(columns={date_col: 'earnings_date'})
        
        # Keep one row per event (in case of daily features)
        if events_raw['event_id'].duplicated().any():
            print(f"  Deduplicating: keeping first row per event_id")
            events_raw = events_raw.drop_duplicates(subset='event_id', keep='first')
        
        # Select required columns
        event_df = events_raw[['event_id', 'permno', 'earnings_date', 'pressure_score']].copy()
        
        print(f"✓ Prepared {len(event_df)} unique events")
        print(f"  Date range: {event_df['earnings_date'].min()} to {event_df['earnings_date'].max()}")
        print(f"  pressure_score range: [{event_df['pressure_score'].min():.3f}, {event_df['pressure_score'].max():.3f}]")
        print(f"  NaN scores: {event_df['pressure_score'].isna().sum()}")
        
    except FileNotFoundError:
        print(f"\n✗ File not found: {event_file}")
        print("  Please ensure event_features.parquet exists in data/processed/")
        print("  Or update the path in this script")
        exit(1)
    except Exception as e:
        print(f"\n✗ Error loading event data: {e}")
        exit(1)
    
    # Load CRSP returns
    crsp_file = "data/processed/crsp_daily.parquet"
    print(f"\nLoading CRSP returns: {crsp_file}")
    
    try:
        # Get required permnos and date range
        permnos = event_df['permno'].unique()
        min_date = pd.to_datetime(event_df['earnings_date'].min()) - pd.Timedelta(days=30)
        max_date = pd.to_datetime(event_df['earnings_date'].max()) + pd.Timedelta(days=30)
        
        # Load and filter
        crsp_raw = pd.read_parquet(crsp_file)
        crsp_raw['date'] = pd.to_datetime(crsp_raw['date'])
        
        returns_df = crsp_raw[
            crsp_raw['permno'].isin(permnos) &
            (crsp_raw['date'] >= min_date) &
            (crsp_raw['date'] <= max_date)
        ][['permno', 'date', 'ret']].copy()
        
        print(f"✓ Loaded {len(returns_df):,} daily return observations")
        print(f"  Stocks: {returns_df['permno'].nunique()}")
        print(f"  Date range: {returns_df['date'].min().date()} to {returns_df['date'].max().date()}")
        print(f"  Mean return: {returns_df['ret'].mean():.5f}")
        
    except FileNotFoundError:
        print(f"\n✗ File not found: {crsp_file}")
        print("  Please ensure crsp_daily.parquet exists in data/processed/")
        print("  Required columns: permno, date, ret")
        exit(1)
    except Exception as e:
        print(f"\n✗ Error loading CRSP data: {e}")
        exit(1)
    
    # ========================================================================
    # Run Backtester
    # ========================================================================
    
    print("\n" + "-"*70)
    print("Initializing EventBacktester...")
    print("-"*70)
    
    bt = EventBacktester(event_df, returns_df)
    
    print("\nRunning full analysis (4 configurations)...")
    print("="*70)
    
    results = bt.run_full_analysis(
        entry_timings=['t=0', 't=+1'],
        horizons=[5, 20]
    )
    
    # ========================================================================
    # Display Results
    # ========================================================================
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for (entry, horizon), res in results.items():
        print(f"\n{'─'*70}")
        print(f"Configuration: entry={entry}, horizon={horizon} days")
        print(f"{'─'*70}")
        
        summary = res['summary']
        ls = res['long_short']
        
        print("\nQuintile Performance:")
        print(summary.to_string(index=False))
        
        print(f"\n📊 Long-Short Analysis:")
        print(f"  Q5 (Info) return:      {ls['q5_return']:>8.4f} ({ls['q5_return']*100:>6.2f}%)")
        print(f"  Q1 (Liquidity) return: {ls['q1_return']:>8.4f} ({ls['q1_return']*100:>6.2f}%)")
        print(f"  Q5-Q1 spread:          {ls['long_short_return']:>8.4f} ({ls['long_short_return']*100:>6.2f}%)")
        
        # Monotonicity check
        returns = summary['mean_return'].values
        monotonic = all(returns[i] <= returns[i+1] for i in range(len(returns)-1))
        print(f"\n  Monotonic pattern: {'✓ YES' if monotonic else '✗ NO'}")
    
    # ========================================================================
    # EXTREME BINS ANALYSIS (Critical Test)
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXTREME BINS ANALYSIS")
    print("="*70)
    print("\nTesting Top/Bottom 10%, 15%, 20% instead of quintiles")
    print("Rationale: Info trading concentrated in extreme events")
    
    for pct in [10, 15, 20]:
        print(f"\n{'─'*70}")
        print(f"Top {pct}% vs Bottom {pct}%")
        print(f"{'─'*70}")
        
        # Get extreme events
        n_extreme = int(len(event_df) * pct / 100)
        top_events = event_df.nlargest(n_extreme, 'pressure_score')['event_id']
        bottom_events = event_df.nsmallest(n_extreme, 'pressure_score')['event_id']
        
        print(f"  Top {pct}%: {len(top_events)} events (pressure_score >= {event_df.nlargest(n_extreme, 'pressure_score')['pressure_score'].min():.3f})")
        print(f"  Bottom {pct}%: {len(bottom_events)} events (pressure_score <= {event_df.nsmallest(n_extreme, 'pressure_score')['pressure_score'].max():.3f})")
        
        # Test on 20-day horizon (most important)
        for entry in ['t=0', 't=+1']:
            backtest_df = results[(entry, 20)]['backtest']
            
            top_returns = backtest_df[backtest_df['event_id'].isin(top_events)]['cum_return']
            bottom_returns = backtest_df[backtest_df['event_id'].isin(bottom_events)]['cum_return']
            
            top_mean = top_returns.mean()
            bottom_mean = bottom_returns.mean()
            spread = top_mean - bottom_mean
            
            print(f"\n  {entry}, 20d horizon:")
            print(f"    Top {pct}%:    {top_mean:>8.4f} ({top_mean*100:>6.2f}%)")
            print(f"    Bottom {pct}%: {bottom_mean:>8.4f} ({bottom_mean*100:>6.2f}%)")
            print(f"    Spread:        {spread:>8.4f} ({spread*100:>6.2f}%) {'✓✓✓' if spread > 0.01 else '✓' if spread > 0.005 else '✗'}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for (entry, horizon), res in results.items():
        filename = f"backtest_results_{entry.replace('=', '')}_h{horizon}.parquet"
        filepath = output_dir / filename
        res['backtest'].to_parquet(filepath, index=False)
    
    print(f"\n✓ Saved results to: {output_dir}/")
    
    # Final interpretation
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
Expected pattern if pressure_score works:
- Q5 (high info) should show DRIFT (positive returns)
- Q1 (high liquidity) should show REVERSION (negative/lower returns)
- Q5-Q1 should be significantly positive
- Pattern should be monotonic (Q1 < Q2 < Q3 < Q4 < Q5)

This is REAL DATA - results matter!
    """)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)