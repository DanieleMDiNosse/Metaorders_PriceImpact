import pandas as pd
import numpy as np
import time
from typing import Dict, List

# --- Mock Data Generation ---
def generate_synthetic_data(n_rows=1000):
    # Create data spanning multiple days
    dates = pd.date_range(start="2025-01-01 09:30:00", periods=n_rows, freq="10s")
    df = pd.DataFrame({
        "Trade Time": dates,
        "ID Member": np.random.randint(1, 5, n_rows),
        "Total Quantity Buy": np.random.randint(10, 100, n_rows),
        "Total Quantity Sell": np.random.randint(10, 100, n_rows),
    })
    return df

def generate_mock_metaorders(n_rows, n_metaorders=20):
    metaorders = {}
    for i in range(n_metaorders):
        member_id = np.random.randint(1, 5)
        # Random start and length
        start = np.random.randint(0, n_rows - 20)
        length = np.random.randint(2, 15)
        indices = list(range(start, start + length))
        
        if member_id not in metaorders:
            metaorders[member_id] = []
        metaorders[member_id].append(indices)
    return {"ISIN_MOCK": metaorders}

# --- Original Logic ---
def run_original(trades, metaorders_dict_all):
    durations = []
    inter_arrivals = []
    meta_volumes = []
    q_over_v = []
    participation_rates = []
    
    isin = "ISIN_MOCK"
    metaorders_dict = metaorders_dict_all.get(isin, {})
    
    for metas in metaorders_dict.values():
        for meta in metas:
            start_idx, end_idx = meta[0], meta[-1]
            start_time = trades.iloc[start_idx]["Trade Time"]
            end_time = trades.iloc[end_idx]["Trade Time"]
            durations.append((end_time - start_time).total_seconds())
            vols = trades.iloc[meta][["Total Quantity Buy", "Total Quantity Sell"]].sum().sum()
            meta_volumes.append(float(vols))
            day_mask = trades["Trade Time"].dt.date == start_time.date()
            day_volume = trades.loc[day_mask, ["Total Quantity Buy", "Total Quantity Sell"]].sum().sum()
            if day_volume != 0:
                q_over_v.append(float(vols / day_volume))
            slice_volume = trades.iloc[start_idx : end_idx + 1][["Total Quantity Buy", "Total Quantity Sell"]].sum().sum()
            if slice_volume != 0:
                participation_rates.append(float(vols / slice_volume))
            if len(meta) > 1:
                times = trades.iloc[meta]["Trade Time"].to_numpy()
                diffs = (times[1:] - times[:-1]).astype("timedelta64[s]").astype(float)
                inter_arrivals.extend(diffs.tolist())
                
    return durations, inter_arrivals, meta_volumes, q_over_v, participation_rates

# --- Optimized Logic ---
def run_optimized(trades, metaorders_dict_all):
    durations = []
    inter_arrivals = []
    meta_volumes = []
    q_over_v = []
    participation_rates = []
    
    isin = "ISIN_MOCK"
    metaorders_dict = metaorders_dict_all.get(isin, {})
    
    # Optimization: Pre-compute numpy arrays
    times = trades["Trade Time"].to_numpy()
    q_buy = trades["Total Quantity Buy"].to_numpy()
    q_sell = trades["Total Quantity Sell"].to_numpy()
    
    # Pre-calculate daily volumes
    daily_vols = trades.groupby(trades["Trade Time"].dt.date)[["Total Quantity Buy", "Total Quantity Sell"]].sum().sum(axis=1).to_dict()

    for metas in metaorders_dict.values():
        for meta in metas:
            if not meta: continue
            start_idx, end_idx = meta[0], meta[-1]
            
            start_time_np = times[start_idx]
            end_time_np = times[end_idx]
            
            # Duration
            dur_seconds = (end_time_np - start_time_np) / np.timedelta64(1, 's')
            durations.append(float(dur_seconds))
            
            # Meta volume
            meta_indices = np.array(meta)
            vols = float(q_buy[meta_indices].sum() + q_sell[meta_indices].sum())
            meta_volumes.append(vols)
            
            # Daily volume
            start_date = pd.Timestamp(start_time_np).date()
            day_volume = daily_vols.get(start_date, 0.0)
            
            if day_volume != 0:
                q_over_v.append(float(vols / day_volume))
            
            # Slice volume
            slice_volume = float(q_buy[start_idx : end_idx + 1].sum() + q_sell[start_idx : end_idx + 1].sum())
            if slice_volume != 0:
                participation_rates.append(float(vols / slice_volume))
            
            # Inter-arrivals
            if len(meta) > 1:
                meta_times = times[meta_indices]
                diffs = (meta_times[1:] - meta_times[:-1]) / np.timedelta64(1, 's')
                inter_arrivals.extend(diffs.tolist())

    return durations, inter_arrivals, meta_volumes, q_over_v, participation_rates

# --- Main Execution ---
if __name__ == "__main__":
    print("Generating data...")
    # Use enough rows to make it interesting
    trades = generate_synthetic_data(n_rows=10000)
    metaorders = generate_mock_metaorders(n_rows=10000, n_metaorders=200)
    
    print("Running Original...")
    t0 = time.time()
    res_orig = run_original(trades, metaorders)
    t1 = time.time()
    print(f"Original took {t1-t0:.4f}s")
    
    print("Running Optimized...")
    t0 = time.time()
    res_opt = run_optimized(trades, metaorders)
    t1 = time.time()
    print(f"Optimized took {t1-t0:.4f}s")
    
    # Compare
    names = ["durations", "inter_arrivals", "meta_volumes", "q_over_v", "participation_rates"]
    all_match = True
    for name, r1, r2 in zip(names, res_orig, res_opt):
        if len(r1) != len(r2):
            print(f"Mismatch in length for {name}: {len(r1)} vs {len(r2)}")
            all_match = False
            continue
        
        # Check values with tolerance
        if not np.allclose(r1, r2, rtol=1e-9, atol=1e-9):
            print(f"Mismatch in values for {name}")
            # Print first mismatch
            for i, (v1, v2) in enumerate(zip(r1, r2)):
                if not np.isclose(v1, v2, rtol=1e-9, atol=1e-9):
                    print(f"  Index {i}: {v1} vs {v2}")
                    break
            all_match = False
        else:
            print(f"{name}: MATCH")
            
    if all_match:
        print("\nSUCCESS: Both approaches give exactly the same results.")
    else:
        print("\nFAILURE: Results differ.")
