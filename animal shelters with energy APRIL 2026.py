import ast 
import colour
import json # read file from steve
import luxpy as lx
import math
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import paxplot # plot results for 3+ objectives
import pyradiance as pr
import subprocess
import time
import traceback

import input_params_june_2025 as userinput
from alphaopics_main import alphaopics as ao

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Binary
from scipy.interpolate import interp1d

# no scientific notation in printed results
np.set_printoptions(suppress=True) 

# adjust the font sizes for the plots
VERY_SMALL_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
plt.rcParams.update({
    'font.size': SMALL_SIZE,
    'axes.labelsize': MEDIUM_SIZE,
    'axes.titlesize': BIGGER_SIZE,
    'xtick.labelsize': SMALL_SIZE,
    'ytick.labelsize': SMALL_SIZE,
    'legend.fontsize': VERY_SMALL_SIZE
})


def cols_to_df(cols, df): 
    # turn select columns from an existing dataframe into rows of a new dataframe (restructuring the data)
    # not transposed directly
    # the data from each column becomes a list to be put into one cell within the new dataframe
    vals = np.empty((len(cols), len(df))) # make an empty array
    for n in range(len(cols)): 
        for m in range(len(df)):
            vals[n][m] = df[cols[n]].iloc[m] 
    new_df = pd.DataFrame({"source": cols, "values": vals.tolist()}).set_index("source")
    return new_df

def calculate_metrics(result_inputs, sources, wavelengths, radiance_lux_h, radiance_lux_v,
                      max_luxpy_power, species, non_human_species=None, reference=False):
    # Calculate the resulting metrics for a given solution
    calc_wavelengths = np.array(wavelengths)
    
    # Check if inputs are per-channel arrays/lists
    is_per_channel = isinstance(radiance_lux_h, (list, np.ndarray)) and len(radiance_lux_h) > 1

    if reference:
        # Intelligently extract the array whether 'sources' is a DataFrame, Series, or array/list
        if hasattr(sources, 'spds'):
            raw_spd = sources['spds'].iloc[0]
        elif hasattr(sources, 'iloc'):
            raw_spd = sources.iloc[0]
        else:
            raw_spd = sources
            
        resulting_spd_power = np.array(raw_spd, dtype=float).flatten()
        spd_on_desk = np.copy(resulting_spd_power)
        spd_at_eye = np.copy(resulting_spd_power)
        
        # Ensure reference values are treated as scalars
        h_rad = radiance_lux_h[0] if isinstance(radiance_lux_h, list) else radiance_lux_h
        v_rad = radiance_lux_v[0] if isinstance(radiance_lux_v, list) else radiance_lux_v
        max_lp = max_luxpy_power[0] if isinstance(max_luxpy_power, list) else max_luxpy_power
        
        spds_ref = np.vstack((calc_wavelengths, resulting_spd_power))
        calc_power = float(np.asarray(lx.spd_to_power(spds_ref, ptype='pu')).item())
        
        safe_max = float(max_lp) if (max_lp is not None and float(max_lp) > 0) else calc_power
        if safe_max == 0: 
            safe_max = 1.0
            
        spd_on_desk *= (float(h_rad) / safe_max)
        spd_at_eye *= (float(v_rad) / safe_max)
        
    else:
        # GA optimization path
        n_sources = len(sources)
        resulting_spd_power = np.zeros(len(calc_wavelengths), dtype=float)
        spd_on_desk = np.zeros(len(calc_wavelengths), dtype=float)
        spd_at_eye = np.zeros(len(calc_wavelengths), dtype=float)
        
        for n in range(n_sources):
            source_spectrum = np.array(sources["spds"].iloc[n], dtype=float).flatten()
            dim_val = result_inputs[f"source_{n}_dim"]
            on_off_val = result_inputs[f"source_{n}_on"]
            actual_dimming = dim_val * on_off_val
            
            resulting_spd_power += (actual_dimming * source_spectrum)
            
            if is_per_channel:
                # Apply per-channel scaling independently
                h_rad = float(radiance_lux_h[n])
                v_rad = float(radiance_lux_v[n])
                max_lp = float(max_luxpy_power[n])
                if max_lp == 0: 
                    max_lp = 1.0
                
                spd_on_desk += (actual_dimming * source_spectrum * (h_rad / max_lp))
                spd_at_eye += (actual_dimming * source_spectrum * (v_rad / max_lp))

        if not is_per_channel:
            # Fallback for unified scaling (when radiance inputs are scalars)
            h_rad = radiance_lux_h[0] if isinstance(radiance_lux_h, list) else radiance_lux_h
            v_rad = radiance_lux_v[0] if isinstance(radiance_lux_v, list) else radiance_lux_v
            max_lp = max_luxpy_power[0] if isinstance(max_luxpy_power, list) else max_luxpy_power
            
            spds_combined = np.vstack((calc_wavelengths, resulting_spd_power))
            calc_power = float(np.asarray(lx.spd_to_power(spds_combined, ptype='pu')).item())
            
            safe_max = float(max_lp) if (max_lp is not None and float(max_lp) > 0) else calc_power
            if safe_max == 0: 
                safe_max = 1.0
            
            spd_on_desk = resulting_spd_power * (float(h_rad) / safe_max)
            spd_at_eye = resulting_spd_power * (float(v_rad) / safe_max)

    # Sort wavelengths in ascending order and mirror the sort on SPDs
    sort_idx = np.argsort(calc_wavelengths)
    calc_wavelengths = calc_wavelengths[sort_idx]
    resulting_spd_power = resulting_spd_power[sort_idx]
    spd_on_desk = spd_on_desk[sort_idx]
    spd_at_eye = spd_at_eye[sort_idx]

    # Remove duplicate wavelength entries if present
    calc_wavelengths, unique_idx = np.unique(calc_wavelengths, return_index=True)
    resulting_spd_power = resulting_spd_power[unique_idx]
    spd_on_desk = spd_on_desk[unique_idx]
    spd_at_eye = spd_at_eye[unique_idx]

    spds = np.vstack((calc_wavelengths, resulting_spd_power))
    
    try:
        # Core Luxpy Calculations
        xyz = lx.spectrum.spd_to_xyz(spds, relative=True)
        cct = lx.color.cct.xyz_to_cct(xyz)
        tm30 = lx.color.cri.iestm30.metrics.spd_to_ies_tm30_metrics(spds)
        
        # Determine specific illuminances from the physically scaled SPDs
        desk_spds = np.vstack((calc_wavelengths, spd_on_desk))
        eye_spds = np.vstack((calc_wavelengths, spd_at_eye))
        
        desk_illum = float(np.asarray(lx.spd_to_power(desk_spds, ptype='pu')).item())
        eye_illum = float(np.asarray(lx.spd_to_power(eye_spds, ptype='pu')).item())
        
        # MEDI Calculation applies exactly to the vertical eye vector
        alphaopic_calc = ao.alphaopic(spd_at_eye, calc_wavelengths, opsin='Mel', lmax=species)
        MEDI = alphaopic_calc["Luminous"]
        
        if non_human_species is not None:
            alphaopic_calc_nh = ao.alphaopic(spd_at_eye, calc_wavelengths, opsin='Mel', lmax=non_human_species)
            MEDI_non_human = alphaopic_calc_nh["Luminous"]
            
        # Dictionary fixed (previous syntax errors removed)
        metrics = {
            'cct': float(np.asarray(cct).item()),
            'horizontal_illuminance': desk_illum,
            'vertical_illuminance': eye_illum,
            'medi': float(np.asarray(MEDI).item()),
            'medi_non_human': float(np.asarray(MEDI_non_human).item()) if non_human_species is not None else None,
            'tm30_rf': float(np.asarray(tm30["Rf"]).item()),
            'tm30_rg': float(np.asarray(tm30["Rg"]).item()),
            'tm30_bin1_rf': float(np.asarray(tm30["Rfi"][0][0]).item()),
            'tm30_bin1_chroma': float(np.asarray(tm30["Rcshj"][0][0]).item())
        }
        
        return metrics
        
    except Exception as e:
        import traceback
        print(f"Error occurred while calculating metrics: {e}")
        traceback.print_exc()
        return None

def get_ref_obj_value(obj_name, metrics):
    """Helper function to map objective names to reference metric values."""
    if obj_name in ["max_medi", "min_medi"]:
        return metrics['medi']
    elif obj_name == "max_tm30_rf":
        return metrics['tm30_rf']
    elif obj_name == "max_tm30_rg":
        return metrics['tm30_rg']
    elif obj_name in ["max_cct", "min_cct"]:
        return metrics['cct']
    elif obj_name in ["max_illuminance", "min_illuminance"]:
        return metrics['horizontal_illuminance']
    elif obj_name == "min_sources":
        return 1.0  # Reference fixture treated as 1 active source
    return 0.0

def plot_results(res, problem, objectives, sources, wavelengths, radiance_lux_h,
                 radiance_lux_v, max_luxpy_power, species, non_human_species=None,
                 reference=None, thresholds=None, ref_scale=False, selected_sol=False):
    
    n_obj = len(objectives)
    label_map = {
        'max_medi': 'mEDI', 'min_medi': 'mEDI', 'max_tm30_rf': 'TM-30 Rf',
        'max_tm30_rg': 'TM-30 Rg', 'max_cct': 'CCT (K)', 'min_cct': 'CCT (K)',
        'max_illuminance': 'Illuminance', 'min_illuminance': 'Illuminance',
        'min_sources': 'Channels'
    }
    
    readable_labels = [label_map.get(obj, obj) for obj in objectives]
    title_goals = []
    for obj in objectives:
        base_label = label_map.get(obj, obj)
        if obj.startswith("max_"):
            title_goals.append(f"Maximize {base_label}")
        elif obj.startswith("min_"):
            title_goals.append(f"Minimize {base_label}")
        else:
            title_goals.append(base_label)
    title_text = f"Results for Objectives: {', '.join(title_goals)}"

    # 1. Transform res.F values back to real physical values
    F_array = np.array(res.F, dtype=float)
    if F_array.ndim == 1:
        F_array = F_array.reshape(1, -1) if hasattr(res.F[0], '__len__') else F_array.reshape(-1, 1)
    F_display = np.zeros_like(F_array)
    
    for col_idx, obj in enumerate(objectives):
        if obj.startswith("max_"):
            F_display[:, col_idx] = -F_array[:, col_idx]
        else:
            F_display[:, col_idx] = F_array[:, col_idx]

    if "min_sources" in objectives and res.X is not None:
        col_idx = objectives.index("min_sources")
        solutions_list = [res.X] if isinstance(res.X, dict) else res.X
        for i, sol in enumerate(solutions_list):
            num_on = sum(1 for n in range(problem.n_sources) if sol[f"source_{n}_on"])
            F_display[i, col_idx] = num_on

    # 2. Compute effective outputs and PRE-CALCULATE metrics for ALL solutions
    effective_outputs = []
    all_metrics = [] # Store metrics for CSV and threshold checking
    
    if res.X is not None:
        solutions_list = [res.X] if isinstance(res.X, dict) else res.X
        for sol in solutions_list:
            total_eff = sum((1 if sol[f"source_{n}_on"] else 0) * sol[f"source_{n}_dim"] for n in range(problem.n_sources))
            effective_outputs.append(total_eff)
            
            # Calculate physical metrics for this specific solution
            m = calculate_metrics(sol, sources, wavelengths, radiance_lux_h, radiance_lux_v, max_luxpy_power, species, non_human_species)
            all_metrics.append(m)
            
    total_effective_outputs = np.array(effective_outputs)

    # --- SAVE TO CSV ---
    save_results_to_csv(res, problem, objectives, F_display, all_metrics)

    # 3. Calculate Reference Metrics
    ref_data = []
    if reference is not None:
        ref_list = reference if isinstance(reference, (list, tuple)) else [reference]
        for ref_name in ref_list:
            try:
                df = pd.read_excel('Light Sources.xlsx', sheet_name=ref_name)
                if isinstance(df, dict):
                    df = pd.concat(df.values(), axis=1)
                df = df.loc[:, ~df.columns.duplicated()]
                ref_wav = df['Wavelength']
                df_sources = df.drop(columns="Wavelength")
                ref_sources = cols_to_df(df_sources.columns, df_sources).rename(columns={"values": "spds"})

                ref_rad_h = radiance_simulation(ref_name, sensor_type='horizontal')
                ref_rad_v = radiance_simulation(ref_name, sensor_type='vertical')
                max_power_ref = lx.spd_to_power(np.vstack((ref_wav,
                                                np.array(ref_sources["spds"].iloc[0]))), ptype='pu')

                # Calculate baseline metrics
                ref_metrics = calculate_metrics(None, ref_sources, ref_wav, ref_rad_h, ref_rad_v,
                                                max_power_ref, species, non_human_species=None, reference=True)

                # NEW SCALING LOGIC:
                if ref_scale and ref_metrics:
                    # Calculate the target scale based on the average of min and max illuminance
                    target_illum = (problem.target_illuminance_h + problem.max_illum_h) / 2.0
                    current_illum = ref_metrics['horizontal_illuminance']
                    
                    if current_illum > 0:
                        scale_factor = target_illum / current_illum
                        ref_rad_h *= scale_factor
                        ref_rad_v *= scale_factor
                        
                        # Recalculate with scaled values (indirectly scales MEDI)
                        ref_metrics = calculate_metrics(None, ref_sources, ref_wav, ref_rad_h, ref_rad_v,
                                                        max_power_ref, species, non_human_species=None, reference=True)

                if ref_metrics:
                    obj_vals = [get_ref_obj_value(obj, ref_metrics) for obj in objectives]
                    ref_data.append({'name': ref_name, 'metrics': ref_metrics, 'obj_values': obj_vals})
                    
            except Exception as e:
                print(f"Error processing reference '{ref_name}': {e}")


# =========================================================================
    # CASE 1: Single Objective
    # =========================================================================
    if n_obj == 1:
        best_score = F_display[0, 0]
        print(f"\nOptimal Solution for Objective: {title_goals[0]}")
        print(f"Objective:\n    {readable_labels[0]}: {best_score:.2f}")

        if res.X is not None:
            print("\nConfiguration")
            sol = res.X if isinstance(res.X, dict) else res.X[0]
            for src_idx in range(problem.n_sources):
                status = "ON" if sol[f"source_{src_idx}_on"] else "OFF"
                multiplier = 1 if sol[f"source_{src_idx}_on"] else 0
                dimming = sol[f"source_{src_idx}_dim"]
                print(f"Source {src_idx}: {status} | Dimming Level: {dimming:.2f}")
                print(f"Effective Output: {multiplier * dimming:.2f} (0=OFF, 1=Full Power)")

            metrics = calculate_metrics(sol, sources, wavelengths, radiance_lux_h, radiance_lux_v, max_luxpy_power, species, non_human_species)
            if metrics:
                print("\nOptimized Metrics")
                print(f"CCT: {metrics['cct']:.0f} K")
                print(f"Horizontal Illuminance: {metrics['horizontal_illuminance']:.2f} lux")
                print(f"Vertical Illuminance: {metrics['vertical_illuminance']:.2f} lux")
                print(f"MEDI: {metrics['medi']:.2f}")
                if non_human_species is not None:
                    print(f"MEDI (Non-Human): {metrics['medi_non_human']:.2f}")
                print(f"TM30 Rf: {metrics['tm30_rf']:.2f}")
                print(f"TM30 Rg: {metrics['tm30_rg']:.2f}")
                print(f"TM30 Bin 1 Rf: {metrics['tm30_bin1_rf']:.2f}")
                print(f"TM30 Bin 1 Chroma Shift: {metrics['tm30_bin1_chroma']*100:.2f}%")

        if ref_data:
            print("\nReference Metrics")
            for r in ref_data:
                print(f"Reference Source: {r['name']}")
                ref_m = r['metrics']
                print(f"CCT: {ref_m['cct']:.0f} K")
                print(f"Horizontal Illuminance: {ref_m['horizontal_illuminance']:.2f} lux")
                print(f"Vertical Illuminance: {ref_m['vertical_illuminance']:.2f} lux")
                print(f"MEDI: {ref_m['medi']:.2f}")
                if non_human_species is not None and 'medi_non_human' in ref_m and ref_m['medi_non_human'] is not None:
                    print(f"MEDI (Non-Human): {ref_m['medi_non_human']:.2f}")
                print(f"TM30 Rf: {ref_m['tm30_rf']:.2f}")
                print(f"TM30 Rg: {ref_m['tm30_rg']:.2f}")
                print(f"TM30 Bin 1 Rf: {ref_m['tm30_bin1_rf']:.2f}")
                print(f"TM30 Bin 1 Chroma Shift: {ref_m['tm30_bin1_chroma']*100:.2f}%")

    # =========================================================================
    # CASE 2: Two Objectives (Pareto Scatter Plot)
    # =========================================================================
    elif n_obj == 2:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(F_display[:, 0], F_display[:, 1], alpha=0.8, c=total_effective_outputs, cmap='viridis')
        
        plt.xlabel(readable_labels[0])
        plt.ylabel(readable_labels[1])

        # Plot Reference Points in Red
        for r in ref_data:
            plt.plot(r['obj_values'][0], r['obj_values'][1], 'ro', markersize=10, label=f"Ref ({r['name']})", markeredgecolor='black')

        plt.title(title_text)
        plt.grid(True, alpha=0.3)
        if ref_data:
            plt.legend()
        plt.colorbar(scatter, label="Total Effective Output")
        plt.tight_layout()
        plt.show()

        # Summary table
        print("\nSolutions on Pareto Front")
        print(f"{'Idx':<4} {readable_labels[0]:<18} {readable_labels[1]:<18}")
        print("-" * 42)
        for i, solution in enumerate(F_display):
            print(f"{i:<4} {solution[0]:<18.2f} {solution[1]:<18.2f}")


    # CASE 3: 3+ Objectives (Paxplot Parallel Coordinates)
    else:
        # Add CCT to the readable labels for the plot
        plot_labels = ['CCT (K)'] + readable_labels 

        # Increase n_axes by 1 to accommodate CCT
        paxfig = paxplot.pax_parallel(n_axes=len(objectives) + 1)
        norm = plt.Normalize(vmin=np.min(total_effective_outputs), vmax=np.max(total_effective_outputs))
        cmap = plt.cm.viridis

        # Evaluate which solutions pass the custom thresholds
        is_better = []
        for m in all_metrics:
            if thresholds and m:
                passes = (
                    m['medi'] >= thresholds.get('medi', 0) and
                    m['tm30_rf'] >= thresholds.get('rf', 0) and
                    m['tm30_rg'] >= thresholds.get('rg', 0) and
                    m['tm30_bin1_rf'] >= thresholds.get('rf_bin1', 0) and
                    m['tm30_bin1_chroma'] >= thresholds.get('rg_bin1', -100) and
                    m['horizontal_illuminance'] >= thresholds.get('illuminance', 0) and
                    m['horizontal_illuminance'] <= thresholds.get('max_illuminance', float('inf')) # <-- NEW MAX LIMIT
                )
                is_better.append(passes)
            else:
                is_better.append(False)

        # Plot GA solutions row by row
        for i, sol_vals in enumerate(F_display):
            # Insert this solution's CCT at the beginning of the values being plotted
            cct_val = all_metrics[i]['cct'] if all_metrics[i] else 0.0
            pax_sol_vals = np.insert(sol_vals, 0, cct_val)

            if thresholds and not is_better[i]:
                # Fails threshold: Plot Gray
                paxfig.plot(np.array([pax_sol_vals]), line_kwargs={'color': 'lightgray', 'alpha': 0.4, 'zorder': 1})
            else:
                # Passes threshold: Plot Gradient
                line_color = cmap(norm(total_effective_outputs[i]))
                paxfig.plot(np.array([pax_sol_vals]), line_kwargs={'color': line_color, 'alpha': 0.8, 'zorder': 2})

        legend_handles = []

        # Add Reference Line(s) in RED
        if ref_data:
            for r in ref_data:
                # Insert the reference CCT at the beginning of the plotted values
                ref_cct = r['metrics']['cct']
                ref_pax_vals = np.insert(r['obj_values'], 0, ref_cct)
                
                ref_line = np.array([ref_pax_vals])
                paxfig.plot(ref_line, line_kwargs={'color': 'red', 'linewidth': 3, 'zorder': 10})
            
            ref_legend = mlines.Line2D([], [], color='red', linewidth=3, label='Reference Source')
            legend_handles.append(ref_legend)

        # Add Gray and Gradient indicators to the legend
        if thresholds:
            better_legend = mlines.Line2D([], [], color=cmap(0.5), linewidth=3, label='Optimal (Better) Solutions')
            gray_legend = mlines.Line2D([], [], color='lightgray', linewidth=3, label='Optimal Solutions')
            legend_handles.extend([better_legend, gray_legend])

        # --- NEW CODE: PLOT SELECTED SOLUTION ---
        if selected_sol:
            # 1. Manually map the exact Telelumen dimming values requested
            # custom_sol = {
            #     'source_0_on': False, 'source_0_dim': 0.0,
            #     'source_1_on': True,  'source_1_dim': 2.08,
            #     'source_2_on': False, 'source_2_dim': 0.0,
            #     'source_3_on': True,  'source_3_dim': 3.23,
            #     'source_4_on': True,  'source_4_dim': 4.08,
            #     'source_5_on': True,  'source_5_dim': 4.16,
            #     'source_6_on': False, 'source_6_dim': 0.0,
            #     'source_7_on': True,  'source_7_dim': 1.53
            # }

            custom_sol = {
                            'source_0_on': False, 'source_0_dim': 0.0,
                            'source_1_on': True,  'source_1_dim': 2.05,
                            'source_2_on': True,  'source_2_dim': 1.8,
                            'source_3_on': True,  'source_3_dim': 3.14,
                            'source_4_on': True,  'source_4_dim': 9.92,
                            'source_5_on': True,  'source_5_dim': 5.35,
                            'source_6_on': False, 'source_6_dim': 0.0,
                            'source_7_on': True,  'source_7_dim': 3.11
                        }
            
            # 2. Run the values through the existing metrics calculator
            sel_metrics = calculate_metrics(custom_sol, sources, wavelengths, radiance_lux_h, 
                                            radiance_lux_v, max_luxpy_power, species, non_human_species)
            
            if sel_metrics:
                sel_obj_vals = []
                for obj in objectives:
                    # Manually enforce the source count since 5 channels are ON
                    if obj == "min_sources":
                        sel_obj_vals.append(5.0) 
                    else:
                        sel_obj_vals.append(get_ref_obj_value(obj, sel_metrics))
                
                # 3. Format it for paxplot (inserting CCT at index 0 to match the others)
                sel_pax_vals = np.insert(sel_obj_vals, 0, sel_metrics['cct'])
                
                # 4. Plot the line (Using a cyan dashed line so it stands out against the gradient and red reference)
                paxfig.plot(np.array([sel_pax_vals]), line_kwargs={'color': 'cyan', 'linewidth': 3, 'linestyle': '--', 'zorder': 12})
                
                # 5. Add it to the legend array
                sel_legend = mlines.Line2D([], [], color='cyan', linewidth=3, linestyle='--', label='Selected Solution')
                legend_handles.append(sel_legend)
        # --- END NEW CODE ---

        if legend_handles:
            paxfig.figure.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.05, 0.95))

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = paxfig.figure.colorbar(sm, ax=paxfig.axes, orientation='vertical', pad=0.08)
        cbar.set_label("Effective Output")

        if 'min_sources' in objectives:
            # Add 1 to the index because CCT is now at index 0
            channel_axis_idx = objectives.index('min_sources') + 1 
            max_channels = len(sources)
            integer_ticks = list(range(1, max_channels + 1))
            paxfig.set_ticks(ax_idx=channel_axis_idx, ticks=integer_ticks)

        paxfig.set_labels(plot_labels)
        plt.suptitle(title_text, y=1.05, fontsize=14, fontweight='bold')
        plt.show()


# def plot_results(res, problem, objectives, sources, wavelengths, radiance_lux_h, radiance_lux_v, max_luxpy_power, species, non_human_species=None, reference=None):
#     # Determine number of objectives
#     n_obj = len(objectives)
    
#     # Dictionary for clean axis labels
#     label_map = {
#         'max_medi': 'MEDI',
#         'min_medi': 'MEDI',
#         'max_tm30_rf': 'TM30 Rf',
#         'max_tm30_rg': 'TM30 Rg',
#         'max_cct': 'CCT (K)',
#         'min_cct': 'CCT (K)',
#         'max_illuminance': 'Illuminance',
#         'min_illuminance': 'Illuminance',
#         'min_sources': 'Number of Channels On'
#     }
    
#     # Translate the backend objectives into readable text for axes
#     readable_labels = [label_map.get(obj, obj) for obj in objectives]
    
#     # --- NEW: Build Title with Maximize/Minimize Goals ---
#     title_goals = []
#     for obj in objectives:
#         base_label = label_map.get(obj, obj)
#         if obj.startswith("max_"):
#             title_goals.append(f"Maximize {base_label}")
#         elif obj.startswith("min_"):
#             title_goals.append(f"Minimize {base_label}")
#         else:
#             title_goals.append(base_label)
            
#     title_text = f"Results for Objectives: {', '.join(title_goals)}"
    
#     # 1. Transform res.F values back to real physical values (un-negate maximized objectives)
#     F_array = np.array(res.F, dtype=float)
#     if F_array.ndim == 1:
#         F_array = F_array.reshape(1, -1) if hasattr(res.F[0], '__len__') else F_array.reshape(-1, 1)

#     F_display = np.zeros_like(F_array)
#     for col_idx, obj in enumerate(objectives):
#         if obj.startswith("max_"):
#             F_display[:, col_idx] = -F_array[:, col_idx]  # Invert back to positive
#         else:
#             F_display[:, col_idx] = F_array[:, col_idx]
            
#     # Override min_sources to show the actual integer count of channels on
#     if "min_sources" in objectives and res.X is not None:
#         col_idx = objectives.index("min_sources")
#         solutions_list = [res.X] if isinstance(res.X, dict) else res.X
#         for i, sol in enumerate(solutions_list):
#             num_on = sum(1 for n in range(problem.n_sources) if sol[f"source_{n}_on"])
#             F_display[i, col_idx] = num_on

#     # 2. Compute effective outputs for solutions
#     effective_outputs = []
#     if res.X is not None:
#         solutions_list = [res.X] if isinstance(res.X, dict) else res.X
#         for sol in solutions_list:
#             total_eff = sum(
#                 (1 if sol[f"source_{n}_on"] else 0) * sol[f"source_{n}_dim"]
#                 for n in range(problem.n_sources)
#             )
#             effective_outputs.append(total_eff)
#     total_effective_outputs = np.array(effective_outputs)

#     # 3. Calculate Reference Metrics (Unified for all scenarios)
#     ref_data = []
#     if reference is not None:
#         ref_list = reference if isinstance(reference, (list, tuple)) else [reference]
#         for ref_name in ref_list:
#             try:
#                 df = pd.read_excel('Light Sources.xlsx', sheet_name=ref_name)
#                 if isinstance(df, dict):
#                     df = pd.concat(df.values(), axis=1)
#                 df = df.loc[:, ~df.columns.duplicated()]
#                 ref_wav = df['Wavelength']
#                 df_sources = df.drop(columns="Wavelength")
#                 ref_sources = cols_to_df(df_sources.columns, df_sources).rename(columns={"values": "spds"})

#                 ref_rad_h = radiance_simulation(ref_name, sensor_type='horizontal')
#                 ref_rad_v = radiance_simulation(ref_name, sensor_type='vertical')
#                 max_power_ref = lx.spd_to_power(np.vstack((ref_wav, np.array(ref_sources["spds"].iloc[0]))), ptype='pu')

#                 ref_metrics = calculate_metrics(None, ref_sources, ref_wav, ref_rad_h, ref_rad_v, max_power_ref, species, non_human_species=None, reference=True)

#                 if ref_metrics:
#                     obj_vals = [get_ref_obj_value(obj, ref_metrics) for obj in objectives]
#                     ref_data.append({
#                         'name': ref_name,
#                         'metrics': ref_metrics,
#                         'obj_values': obj_vals
#                     })
#             except Exception as e:
#                 print(f"Error processing reference '{ref_name}': {e}")

#     # =========================================================================
#     # CASE 1: Single Objective
#     # =========================================================================
#     if n_obj == 1:
#         best_score = F_display[0, 0]
#         print(f"\nOptimal Solution for Objective: {title_goals[0]}")
#         print(f"Objective:\n    {readable_labels[0]}: {best_score:.2f}")

#         if res.X is not None:
#             print("\nConfiguration")
#             sol = res.X if isinstance(res.X, dict) else res.X[0]
#             for src_idx in range(problem.n_sources):
#                 status = "ON" if sol[f"source_{src_idx}_on"] else "OFF"
#                 multiplier = 1 if sol[f"source_{src_idx}_on"] else 0
#                 dimming = sol[f"source_{src_idx}_dim"]
#                 print(f"Source {src_idx}: {status} | Dimming Level: {dimming:.2f}")
#                 print(f"Effective Output: {multiplier * dimming:.2f} (0=OFF, 1=Full Power)")

#             metrics = calculate_metrics(sol, sources, wavelengths, radiance_lux_h, radiance_lux_v, max_luxpy_power, species, non_human_species)
#             if metrics:
#                 print("\nOptimized Metrics")
#                 print(f"CCT: {metrics['cct']:.0f} K")
#                 print(f"Horizontal Illuminance: {metrics['horizontal_illuminance']:.2f} lux")
#                 print(f"Vertical Illuminance: {metrics['vertical_illuminance']:.2f} lux")
#                 print(f"MEDI: {metrics['medi']:.2f}")
#                 if non_human_species is not None:
#                     print(f"MEDI (Non-Human): {metrics['medi_non_human']:.2f}")
#                 print(f"TM30 Rf: {metrics['tm30_rf']:.2f}")
#                 print(f"TM30 Rg: {metrics['tm30_rg']:.2f}")
#                 print(f"TM30 Bin 1 Rf: {metrics['tm30_bin1_rf']:.2f}")
#                 print(f"TM30 Bin 1 Chroma Shift: {metrics['tm30_bin1_chroma']*100:.2f}%")

#         if ref_data:
#             print("\nReference Metrics")
#             for r in ref_data:
#                 print(f"Reference Source: {r['name']}")
#                 ref_m = r['metrics']
#                 print(f"CCT: {ref_m['cct']:.0f} K")
#                 print(f"Horizontal Illuminance: {ref_m['horizontal_illuminance']:.2f} lux")
#                 print(f"Vertical Illuminance: {ref_m['vertical_illuminance']:.2f} lux")
#                 print(f"MEDI: {ref_m['medi']:.2f}")
#                 if non_human_species is not None and 'medi_non_human' in ref_m and ref_m['medi_non_human'] is not None:
#                     print(f"MEDI (Non-Human): {ref_m['medi_non_human']:.2f}")
#                 print(f"TM30 Rf: {ref_m['tm30_rf']:.2f}")
#                 print(f"TM30 Rg: {ref_m['tm30_rg']:.2f}")
#                 print(f"TM30 Bin 1 Rf: {ref_m['tm30_bin1_rf']:.2f}")
#                 print(f"TM30 Bin 1 Chroma Shift: {ref_m['tm30_bin1_chroma']*100:.2f}%")

#     # =========================================================================
#     # CASE 2: Two Objectives (Pareto Scatter Plot)
#     # =========================================================================
#     elif n_obj == 2:
#         plt.figure(figsize=(10, 6))
#         scatter = plt.scatter(F_display[:, 0], F_display[:, 1], alpha=0.8, c=total_effective_outputs, cmap='viridis')
        
#         plt.xlabel(readable_labels[0])
#         plt.ylabel(readable_labels[1])

#         # Plot Reference Points in Red
#         for r in ref_data:
#             plt.plot(r['obj_values'][0], r['obj_values'][1], 'ro', markersize=10, label=f"Ref ({r['name']})", markeredgecolor='black')

#         plt.title(title_text)
#         plt.grid(True, alpha=0.3)
#         if ref_data:
#             plt.legend()
#         plt.colorbar(scatter, label="Total Effective Output")
#         plt.tight_layout()
#         plt.show()

#         # Summary table
#         print("\nSolutions on Pareto Front")
#         print(f"{'Idx':<4} {readable_labels[0]:<18} {readable_labels[1]:<18}")
#         print("-" * 42)
#         for i, solution in enumerate(F_display):
#             print(f"{i:<4} {solution[0]:<18.2f} {solution[1]:<18.2f}")

#     # =========================================================================
#     # CASE 3: 3+ Objectives (Paxplot Parallel Coordinates)
#     # =========================================================================
#     else:
#         # Create Paxplot with axes equal ONLY to the objectives count
#         paxfig = paxplot.pax_parallel(n_axes=len(objectives))

#         # Setup colormap based on Total Effective Output
#         norm = plt.Normalize(vmin=np.min(total_effective_outputs), vmax=np.max(total_effective_outputs))
#         cmap = plt.cm.viridis

#         # Plot GA solutions row by row colored by effective output
#         for i, sol_vals in enumerate(F_display):
#             line_color = cmap(norm(total_effective_outputs[i]))
#             paxfig.plot(np.array([sol_vals]), line_kwargs={'color': line_color, 'alpha': 0.7})

#         # Plot Reference Line(s) in RED
#         if ref_data:
#             for r in ref_data:
#                 ref_line = np.array([r['obj_values']])
#                 paxfig.plot(ref_line, line_kwargs={'color': 'red', 'linewidth': 3, 'zorder': 10})
                
#             # Force a custom legend to show the red reference line
#             ref_legend = mlines.Line2D([], [], color='red', linewidth=3, label='Reference Source')
#             paxfig.figure.legend(handles=[ref_legend], loc='upper left', bbox_to_anchor=(0.05, 0.95))

#         # Colorbar Legend mapped to Effective Output
#         sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#         sm.set_array([])
#         cbar = paxfig.figure.colorbar(sm, ax=paxfig.axes, orientation='vertical', pad=0.08)
#         cbar.set_label("Effective Output (Total Dimming Across All Sources)")

#         target_objective = 'min_sources'

#         if target_objective in objectives:
#             # 1. Dynamically find the axis index
#             channel_axis_idx = objectives.index(target_objective)
            
#             # 2. Determine max channels dynamically from the 'sources' variable
#             max_channels = len(sources) 
            
#             # 3. Create integer ticks from 1 to the total number of sources
#             integer_ticks = list(range(1, max_channels + 1))
            
#             # 4. Apply the custom integer ticks to the paxplot figure
#             paxfig.set_ticks(
#                 ax_idx=channel_axis_idx,
#                 ticks=integer_ticks
#             )

#         paxfig.set_labels(readable_labels)
#         plt.suptitle(title_text, y=1.05, fontsize=14, fontweight='bold')
        
#         plt.show()

# radiance simulation
def _build_ies2rad_kwargs(ies2rad_input): # TODO: remove the need for this ??
    # translate options into pyradiance keyword arguments
    kwargs = {}

    if not ies2rad_input:
        return kwargs

    if isinstance(ies2rad_input, dict):
        if "multiply_factor" in ies2rad_input:
            kwargs["multiply_factor"] = ies2rad_input["multiply_factor"]
        if "lamp_color" in ies2rad_input:
            kwargs["lamp_color"] = tuple(ies2rad_input["lamp_color"])
        return kwargs

    if isinstance(ies2rad_input, (list, tuple)):
        params = list(ies2rad_input)

        if "-m" in params:
            idx = params.index("-m")
            if idx + 1 < len(params):
                kwargs["multiply_factor"] = float(params[idx + 1])

        if "-c" in params:
            idx = params.index("-c")
            if idx + 3 < len(params):
                kwargs["lamp_color"] = tuple(float(v) for v in params[idx + 1:idx + 4])

    return kwargs

def get_standard_material(dataset_name, sample_name, bins=27, start_nm=380, end_nm=780):
    """
    Fetches a standard material from colour-science and interpolates it 
    into a discrete number of bins for Radiance.
    """
    # 1. Fetch the continuous spectral distribution from the library
    # Example dataset: 'ColorChecker N Ohta'
    sd = colour.SDS_COLOURCHECKERS[dataset_name][sample_name]
    
    # 2. Define your target wavelengths (e.g., 27 bins from 380nm to 780nm)
    target_shape = colour.SpectralShape(start_nm, end_nm, (end_nm - start_nm) / (bins - 1))
    
    # 3. Align (interpolate) the spectrum to your exact bins
    sd.align(target_shape)
    
    # 4. Return as a numpy array scaled from 0.0 to 1.0 for Radiance
    return sd.values / 100.0  # (Colour-science stores reflectance as percentages 0-100)


# def radiance_simulation(source_type='telelumen', ies2rad_input=None, sensor_type='horizontal', eye_h=1.6): 
#     # TODO: add options for honeybee + grasshopper
#     # TODO: import obj file

#     room_l = 1.82
#     room_w = 2.23
#     room_h = 2.69
#     desk_l = 1.22
#     desk_w = 0.61
#     desk_h = 0.76
#     # desk_h = room_h - 1 # height of the desk surface from the floor

#     scene_text = f"""
#     # materials

#     # walls 
#     void plastic white_wall
#     0
#     0
#     5 0.9 0.9 0.9 0.01 0.01

#     void plastic acoustic_ceiling 
#     0 
#     0 
#     5 0.88 0.88 0.88 0.02 0.05

#     # floor
#     void plastic carpet_floor
#     0
#     0
#     5 0.2 0.2 0.2 0.01 0.01

#     # wood desk
#     void plastic wood_desk
#     0
#     0
#     5 0.4 0.3 0.2 0.05 0.1

#     # room
#     !genbox white_wall room {room_l} {room_w} {room_h} -i
# """
#     # # desk
#     # !genbox wood_desk desk_surface {desk_l} {desk_w} {desk_h} | xform -t {(room_l - desk_l) / 2} {(room_w - desk_w) / 2} 0
#     # """
#     # write the scene file
#     with open("room.rad", "w") as f:
#         f.write(scene_text)

#     # import the ies file 
#     if source_type == "telelumen":
#         ies_file_name = "telelumen octa ies file 22dec18.ies"
#     elif source_type == "rubik":
#         ies_file_name = "RUBIK_9CS_90CRI_35K_STWH_440LM.ies"
#     elif source_type == "white":
#         ies_file_name = "white.ies"
#     elif source_type == "reference":
#         ies_file_name = "SVT-2x2-3000LM-DO-SD-MVOLT-ST-IES.ies"

#     else:
#         raise ValueError(f"Source type {source_type} not recognized. Please choose from: telelumen, rubik.")
#     ies2rad_kwargs = _build_ies2rad_kwargs(ies2rad_input)
#     if ies2rad_kwargs:
#         pr.ies2rad(ies_file_name, outname="light_source", **ies2rad_kwargs)
#     else:
#         pr.ies2rad(ies_file_name, outname="light_source")

#     # center of the ceiling
#     center_x = room_l / 2
#     center_y = room_w / 2
#     z_height = room_h - 0.05 # Mounted flush/slightly dropped from ceiling

#     # spacing between cells (8 inches = ~0.2 meters)
#     spacing = 0.2 

#     # light placement string 
#     if source_type == "telelumen" or source_type == "reference":
#         light_placement = "# Light Fixture\n"
#         light_placement += f"!xform -t {0.52} {0.8} {z_height} light_source.rad\n"
        
#     elif source_type == "white": # place in center of ceiling, no duplicates
#         # place in center of ceiling, no duplicates
#         light_placement = "# Light Fixture\n"
#         light_placement += f"!xform -t {center_x:.3f} {center_y:.3f} {z_height} light_source.rad\n"

#     elif source_type == "rubik": # place nine cells
#         light_placement = "# 9-Cell Rubik Fixture (2x2 ft)\n"

#         # Loop through X offsets (-0.2, 0, 0.2)
#         for dx in [-spacing, 0, spacing]:
#             # Loop through Y offsets (-0.2, 0, 0.2)
#             for dy in [-spacing, 0, spacing]:
                
#                 # Calculate final coordinates for this specific cell
#                 cell_x = center_x + dx
#                 cell_y = center_y + dy
                
#                 # Append this cell to the Radiance file text
#                 light_placement += f"!xform -t {cell_x:.3f} {cell_y:.3f} {z_height} light_source.rad\n"
#     else:
#         raise ValueError(f"Source type {source_type} not recognized. Please choose from: telelumen, rubik, white.")

#     # write the light placement file
#     with open("luminaires.rad", "w") as f:
#         f.write(light_placement)

#     # compile octree
#     octree_bytes = pr.oconv("room.rad", "luminaires.rad")

#     # write octree to file
#     with open("scene.oct", "wb") as f:
#         f.write(octree_bytes)

#     # define sensor
#     # if source_type == "telelumen":
#     if sensor_type == "horizontal":
#             sensor_string = f"{1.22} {0.8} {desk_h} 0 0 1\n"
#     elif sensor_type == "vertical":
#         sensor_string = f"{1.22} {0.8} {eye_h} -1 0 0\n"
#     # else: 
#     #     if sensor_type == "horizontal":
#     #         sensor_string = f"{room_l / 2} {room_w / 2} {desk_h} 0 0 1\n"
#     #     elif sensor_type == "vertical":
#     #         sensor_string = f"{room_l / 2} {room_w / 2} {eye_h} 0 1 0\n"
#     else:
#         raise ValueError(f"Sensor type {sensor_type} not recognized. Please choose from: horizontal, vertical.")
#     sensor_point = sensor_string.encode('utf-8')

#     # run
#     res_bytes = pr.rtrace(
#         sensor_point, 
#         "scene.oct", 
#         header=False, # header=False ensures it only returns the numbers, not the Radiance header text
#         params=["-I", "-ab", "7", "-ad", "2048", "-as", "1024", "-h"]    
#         )

#     # calculate lux
#     res_str = res_bytes.decode('utf-8') # turn into python string
#     lines = res_str.strip().splitlines() # split the output into lines and remove any leading/trailing whitespace
#     data_line = lines[-1] # grab only the very last line (where the RGB data lives)
#     r, g, b = map(float, data_line.split()) # split that specific line into the 3 numbers 
#     lux = 179 * (0.265 * r + 0.670 * g + 0.065 * b) # calculate illuminance

#     print(f"Calculated Illuminance on Desk: {lux:.2f} Lux")

#     return lux

def radiance_simulation(source_type='telelumen', ies2rad_input=None, sensor_type='horizontal', eye_h=1.6, grid_spacing=0.15, wall_offset=0.15):
    room_l = 1.82
    room_w = 2.23
    room_h = 2.69
    desk_l = 1.22
    desk_w = 0.61
    desk_h = 0.76

    # -------------------------------------------------------------------------
    # Scene and Geometry Generation (Maintained from your calibrated setup)
    # -------------------------------------------------------------------------
    scene_text = f"""
    # materials
    void plastic white_wall 0 0 5 0.9 0.9 0.9 0.01 0.01
    void plastic acoustic_ceiling 0 0 5 0.88 0.88 0.88 0.02 0.05
    void plastic carpet_floor 0 0 5 0.2 0.2 0.2 0.01 0.01
    void plastic wood_desk 0 0 5 0.4 0.3 0.2 0.05 0.1

    # room
    !genbox white_wall room {room_l} {room_w} {room_h} -i
    """

    with open("room.rad", "w") as f:
        f.write(scene_text)

    # Import and compile IES
    if source_type == "telelumen":
        ies_file_name = "telelumen octa ies file 22dec18.ies"
    elif source_type == "rubik":
        ies_file_name = "RUBIK_9CS_90CRI_35K_STWH_440LM.ies"
    elif source_type == "white":
        ies_file_name = "white.ies"
    elif source_type == "reference":
        ies_file_name = "SVT-2x2-3000LM-DO-SD-MVOLT-ST-IES.ies"
    else:
        raise ValueError(f"Source type {source_type} not recognized.")

    ies2rad_kwargs = _build_ies2rad_kwargs(ies2rad_input)
    if ies2rad_kwargs:
        pr.ies2rad(ies_file_name, outname="light_source", **ies2rad_kwargs)
    else:
        pr.ies2rad(ies_file_name, outname="light_source")

    # Light Placement
    center_x = room_l / 2
    center_y = room_w / 2
    z_height = room_h - 0.05
    spacing = 0.2

    if source_type in ["telelumen", "reference"]:
        light_placement = "# Light Fixture\n"
        light_placement += f"!xform -t {0.52} {0.8} {z_height} light_source.rad\n"
    elif source_type == "white":
        light_placement = "# Light Fixture\n"
        light_placement += f"!xform -t {center_x:.3f} {center_y:.3f} {z_height} light_source.rad\n"
    elif source_type == "rubik":
        light_placement = "# 9-Cell Rubik Fixture (2x2 ft)\n"
        for dx in [-spacing, 0, spacing]:
            for dy in [-spacing, 0, spacing]:
                light_placement += f"!xform -t {center_x + dx:.3f} {center_y + dy:.3f} {z_height} light_source.rad\n"

    with open("luminaires.rad", "w") as f:
        f.write(light_placement)

    octree_bytes = pr.oconv("room.rad", "luminaires.rad")
    with open("scene.oct", "wb") as f:
        f.write(octree_bytes)

    # -------------------------------------------------------------------------
    # Plane Grid Generation vs. Single Point Raytracing
    # -------------------------------------------------------------------------
    if sensor_type == "horizontal":
        # Standard Single Point (Desk Height)
        sensor_string = f"{1.22} {0.8} {desk_h} 0 0 1\n"
        sensor_points = [ (1.22, 0.8) ]
    elif sensor_type == "vertical":
        # Standard Single Point (Eye Height)
        sensor_string = f"{1.22} {0.8} {eye_h} -1 0 0\n"
        sensor_points = [ (1.22, 0.8) ]
    elif sensor_type == "grid_horizontal":
        # New Plane Grid: Generate 2D coordinates across the room at desk height
        # Wall offset avoids hitting wall boundaries where geometry edges can cause ray anomalies
        x_coords = np.arange(wall_offset, room_l - wall_offset + grid_spacing/2, grid_spacing)
        y_coords = np.arange(wall_offset, room_w - wall_offset + grid_spacing/2, grid_spacing)
        
        sensor_lines = []
        sensor_points = []
        for x in x_coords:
            for y in y_coords:
                sensor_lines.append(f"{x:.3f} {y:.3f} {desk_h:.3f} 0 0 1")
                sensor_points.append((x, y))
        sensor_string = "\n".join(sensor_lines) + "\n"
    else:
        raise ValueError(f"Sensor type '{sensor_type}' not recognized.")

    # Convert coordinates to binary stream for rtrace
    sensor_bytes = sensor_string.encode('utf-8')

    # Run the raytracer
    res_bytes = pr.rtrace(
        sensor_bytes,
        "scene.oct",
        header=False,
        params=["-I", "-ab", "7", "-ad", "2048", "-as", "1024", "-h"]
    )

    # Parse output lines
    res_str = res_bytes.decode('utf-8')
    lines = res_str.strip().splitlines()
    
    lux_values = []
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        
        # Robustly skip Radiance text headers (any lines that cannot be split into 3 floats)
        try:
            r, g, b = map(float, cleaned_line.split())
            # Calculate photopic lux using Radiance standard V(lambda) weights
            lux = 179 * (0.265 * r + 0.670 * g + 0.065 * b)
            lux_values.append(lux)
        except ValueError:
            # Safely skip non-numeric header lines like '#?RADIANCE' or 'FORMAT=ascii'
            continue

    # -------------------------------------------------------------------------
    # Format and Return Output (Restores 100% backward compatibility)
    # -------------------------------------------------------------------------
    if sensor_type in ["horizontal", "vertical"]:
        # Return a single float value so your existing loops do not break
        return float(lux_values[0]) if lux_values else 0.0
    
    else:
        # Return the rich dictionary for grid metrics
        avg_lux = np.mean(lux_values) if lux_values else 0.0
        min_lux = np.min(lux_values) if lux_values else 0.0
        max_lux = np.max(lux_values) if lux_values else 0.0
        uniformity = min_lux / avg_lux if avg_lux > 0 else 0.0
        
        return {
            "coordinates": sensor_points,
            "lux_grid": lux_values,
            "average_lux": avg_lux,
            "min_lux": min_lux,
            "max_lux": max_lux,
            "uniformity_ratio": uniformity
        }
    
def render_optimized_space_spectral(solution_vars, problem, raw_json_data, wavelengths, base_ies_lumens=2500.0, filename="optimized_spectral_render.hdr"):
    print(f"\n--- Starting N=9 Spectral Render for {filename} ---")
    
    # 1. Generate standard room geometry (same as your current setup)
    room_l, room_w, room_h = 1.83, 2.43, 2.74
    desk_l, desk_w, desk_h = 1.22, 0.61, 0.76
    
#     scene_text = f"""
# # materials (Assuming flat spectra for simplicity, but these should ideally be spectrally defined per pass too)
# void plastic white_wall
# 0
# 0
# 5 0.8 0.8 0.8 0.01 0.01

# void plastic carpet_floor
# 0
# 0
# 5 0.2 0.2 0.2 0.01 0.01

# void plastic wood_desk
# 0
# 0
# 5 0.4 0.3 0.2 0.05 0.1

# !genbox white_wall room {room_l} {room_w} {room_h} -i
# !genbox wood_desk desk_surface {desk_l} {desk_w} {desk_h} | xform -t {(room_l - desk_l)/2} {(room_w-desk_w)/2} 0
# """

    scene_text = f"""
    # =========================================================================
    # MATERIAL DEFINITIONS
    # =========================================================================
    # Bright White Walls (80% reflectance)
    void plastic white_wall 0 0 5  0.80 0.80 0.80  0.01 0.01

    # Acoustic Ceiling Tiles (88% reflectance, rougher diffuse texture)
    void plastic acoustic_ceiling 0 0 5  0.88 0.88 0.88  0.01 0.05

    # Dark Gray Carpeting (12% reflectance, high diffuse roughness)
    void plastic carpet_floor 0 0 5  0.12 0.12 0.12  0.01 0.15

    # Wood Desk
    void plastic wood_desk 0 0 5  0.40 0.30 0.20  0.05 0.10

    # =========================================================================
    # INWARD-FACING ROOM GEOMETRY (Winding order set for inward-pointing normals)
    # =========================================================================
    
    # 1. Floor (Dark Gray Carpet)
    carpet_floor polygon room_floor
    0
    0
    12
        0          0          0
        {room_l}   0          0
        {room_l}   {room_w}   0
        0          {room_w}   0

    # 2. Ceiling (White Acoustic Tiles)
    acoustic_ceiling polygon room_ceiling
    0
    0
    12
        0          0          {room_h}
        0          {room_w}   {room_h}
        {room_l}   {room_w}   {room_h}
        {room_l}   0          {room_h}

    # 3. Back Wall (Y = 0)
    white_wall polygon wall_back
    0
    0
    12
        0          0          0
        0          0          {room_h}
        {room_l}   0          {room_h}
        {room_l}   0          0

    # 4. Front Wall (Y = room_w)
    white_wall polygon wall_front
    0
    0
    12
        0          {room_w}   0
        {room_l}   {room_w}   0
        {room_l}   {room_w}   {room_h}
        0          {room_w}   {room_h}

    # 5. Left Wall (X = 0)
    white_wall polygon wall_left
    0
    0
    12
        0          0          0
        0          {room_w}   0
        0          {room_w}   {room_h}
        0          0          {room_h}

    # 6. Right Wall (X = room_l)
    white_wall polygon wall_right
    0
    0
    12
        {room_l}   0          0
        {room_l}   0          {room_h}
        {room_l}   {room_w}   {room_h}
        {room_l}   {room_w}   0
    """
    with open("room.rad", "w") as f:
        f.write(scene_text)

    # 2. Define the N=9 wavebands (3 images x 3 channels)
    # Target bands: Image 1 [380-510], Image 2 [515-645], Image 3 [650-780]
    # We will interpolate the combined SPD into 9 distinct bins
    target_bins = np.linspace(380, 780, 10) 
    channels = list(raw_json_data.keys())
    
    # Calculate combined total optimized SPD
    total_spd_power = np.zeros(len(wavelengths))
    for n, ch in enumerate(channels):
        if not solution_vars[f"source_{n}_on"]: continue
        dimming = solution_vars[f"source_{n}_dim"]
        ch_lumens = raw_json_data[ch]["100_percent_measurement"]["Colorimetrics"]["LuminousOutput"]
        base_multiplier = ch_lumens / base_ies_lumens
        spd = np.array(raw_json_data[ch]["100_percent_measurement"]["SpdArr"])
        total_spd_power += (spd * base_multiplier * dimming)
        
    # Bin the SPD into 9 wavebands
    binned_spd = []
    for i in range(9):
        mask = (wavelengths >= target_bins[i]) & (wavelengths < target_bins[i+1])
        avg_radiance = np.mean(total_spd_power[mask]) if np.any(mask) else 0.0
        binned_spd.append(avg_radiance)

    center_x, center_y, z_height = room_l/2, room_w/2, room_h - 0.05
    ies_file_name = "telelumen octa ies file 22dec18.ies"
    
    hdr_passes = []
    
    # 3. Spectral Rendering Loop (3 passes)
    for pass_idx in range(3):
        # Map 3 wavebands to R, G, B for this pass
        r_val = binned_spd[pass_idx * 3 + 0]
        g_val = binned_spd[pass_idx * 3 + 1]
        b_val = binned_spd[pass_idx * 3 + 2]
        
        # Max scale to prevent Radiance from clipping, we'll restore scale later
        pass_max = max(r_val, g_val, b_val)
        scale_factor = pass_max if pass_max > 0 else 1.0
        
        r, g, b = r_val/scale_factor, g_val/scale_factor, b_val/scale_factor
        
        # Generate light source for this specific spectral pass
        rad_params = ["-m", f"{scale_factor:.4f}", "-c", f"{r:.4f}", f"{g:.4f}", f"{b:.4f}"]
        rad_kwargs = _build_ies2rad_kwargs(rad_params)
        
        pass_outname = f"light_source_pass_{pass_idx}"
        pr.ies2rad(ies_file_name, outname=pass_outname, **rad_kwargs)
        
        light_placement = f"!xform -t {center_x:.3f} {center_y:.3f} {z_height} {pass_outname}.rad\n"
        with open(f"luminaires_pass_{pass_idx}.rad", "w") as f:
            f.write(light_placement)
            
        # Compile and Render
        octree_bytes = pr.oconv("room.rad", f"luminaires_pass_{pass_idx}.rad")
        oct_file = f"scene_pass_{pass_idx}.oct"
        with open(oct_file, "wb") as f:
            f.write(octree_bytes)
            
        view_args = ["-vtv", "-vp", f"{room_l/2}", "0.2", "1.6", "-vd", "0", "1", "-0.2", "-vh", "90", "-vv", "60"]
        render_options = ["-ab", "5", "-ad", "2048", "-as", "1024", "-x", "800", "-y", "600"]
        
        print(f"Shooting rays for Pass {pass_idx+1}/3 ([{target_bins[pass_idx*3]:.0f} - {target_bins[pass_idx*3+3]:.0f} nm])...")
        image_data = pr.rpict(view=view_args, octree=oct_file, params=render_options)
        
        pass_hdr = f"pass_{pass_idx}.hdr"
        with open(pass_hdr, "wb") as f:
            f.write(image_data)
        hdr_passes.append(pass_hdr)

    # 4. Recombine the Hyperspectral Passes into Tristimulus (XYZ -> RGB)
    # Radiance's 'pcomb' command is used here to apply the CIE color matching functions
    # to the resulting binary images to generate the XYZ image.
    print("Convolving wavebands with color matching functions via pcomb...")
    
    # Note: The constants below represent the integrated CIE 1931 matching functions for these specific 3-bin combinations.
    # In a full implementation, you would calculate these based on luxpy's matching functions (lx._CIE_XYZ_1931_2DEG).
    pcomb_expr = (
        f"pcomb -e \"ro=ri(1)*0.05+ri(2)*0.45+ri(3)*0.50\" " 
        f"-e \"go=gi(1)*0.10+gi(2)*0.80+gi(3)*0.10\" "
        f"-e \"bo=bi(1)*0.95+bi(2)*0.05+bi(3)*0.00\" "
        f"-o {hdr_passes[0]} -o {hdr_passes[1]} -o {hdr_passes[2]} > {filename}"
    )

    subprocess.run(pcomb_expr, shell=True, check=True)
    
    print(f"Spectral render complete! Saved as {filename}")
    
    # 5. Tone map and display (using subprocess for guaranteed execution)
    print("Tone mapping and converting to BMP...")
    
    # Removed the '-h' flag! pcond will now auto-expose the image beautifully.
    pcond_cmd = f"pcond {filename} > tonemapped.hdr"
    subprocess.run(pcond_cmd, shell=True, check=True)
    
    # Convert the tonemapped HDR to a standard BMP image
    bmp_filename = filename.replace(".hdr", ".bmp")
    ra_bmp_cmd = f"ra_bmp tonemapped.hdr > {bmp_filename}"
    subprocess.run(ra_bmp_cmd, shell=True, check=True)
    
    print(f"Image processing complete! Saved tonemapped.hdr and {bmp_filename}")

class optimization_problem(ElementwiseProblem):
    def __init__(self, sources, wavelengths, rad_illuminance_h, rad_illuminance_v, max_lp_illuminance, 
                 target_illuminance_h=750, max_illum_h=1125, target_illuminance_v=0, target_medi=275, 
                 target_medi_nonhuman=0, min_dim=0.2, max_dim=10.0, species="Human", nonhuman_species=None,
                 objective=None, dim_curves=None, sensor_grid=False):
        # initialize variables        
        self.sources = sources # sources dataframe with spds
        self.wavelengths = wavelengths # wavelengths for the spds
        self.n_sources = len(sources) # count of channels
        self.rad_illuminance_h = rad_illuminance_h # illuminance from radiance simulation (horizontal)
        self.rad_illuminance_v = rad_illuminance_v # illuminance from radiance simulation (vertical)
        # print("radi",rad_illuminance)
        self.max_lp_illuminance = max_lp_illuminance # illuminance from luxpy simulation 
        # print("lpi", max_lp_illuminance)
        self.max_illum_h = max_illum_h # maximum illuminance (horizontal)
        self.target_illuminance_h = target_illuminance_h # target illuminance (horizontal)
        self.target_illuminance_v = target_illuminance_v # target illuminance (vertical)    
        self.target_medi = target_medi # target MEDI        
        self.target_medi_nonhuman = target_medi_nonhuman # target MEDI (non-human)
        self.min_dim = min_dim # minimum dimming level per channel
        self.max_dim = max_dim # maximum dimming level per channel
        self.species = species # species to optimize for
        self.objective = objective # objective(s) to optimize for 
        self.dim_curves = dim_curves # dimming curves for each source
        self.sensor_grid = sensor_grid # whether to use sensor grid
        self.species_nonhuman = nonhuman_species # species to optimize for (non-human)

        #define variables
        variables = {}
        for n in range(len(sources)):
            variables[f"source_{n}_on"]= Binary() # on or off
            variables[f"source_{n}_dim"]= Real(bounds=(self.min_dim,self.max_dim)) # percent dim for each source

        sg = 0
        if self.sensor_grid==True:
            sg = 2

        # count of constraints : 2 CCT, 6 TM30, 2 Illuminance (horizontal, +2 if sensor grid is used), 1 MEDI if human (2 if nonhuman species is included), 
        if nonhuman_species is not None:
            total_constraints = 2 + 6 + 2 + sg + 2 
        else:
            total_constraints = 2 + 6 + 2 + sg + 1 

        # objective count
        if self.objective is None:
            raise ValueError("Objective must be specified")
        else:
            num_objectives = len(self.objective)

        super().__init__(vars=variables, n_obj=num_objectives, n_ieq_constr=total_constraints)

    def _evaluate(self, x, out, *args, **kwargs):
#         # objective: reduce number of sources (expanded now)
#         # pymoo convention 
#             # cons <= 0
#             # objective minimized
#         # variable 1 set : 0 or 1 for each source (on/off)
#         # variable 2 set: dimming level for each source (0 to 1)
#         # constraints:
#         # 1. CCT between values
#         # 2. MEDI or alpha opic output within constraints
#         # 3. cri or tm30 within threshold
#         # 4. illuminance within bounds

        if type(self.rad_illuminance_h) == list:
            tmp = False # per channel illuminance
        else:
            tmp = True # per source illuminance

        # define variables
        on_off_variables = np.zeros(self.n_sources)
        dim_variables = np.zeros(self.n_sources)
        
        for n in range(self.n_sources):
            on_off_variables[n] = x[f"source_{n}_on"]
            dim_variables[n] = x[f"source_{n}_dim"]

        wavelengths = self.wavelengths
        constraints = []
                                      
        resulting_spd = np.zeros(len(self.wavelengths))
        spd_on_desk = np.zeros(len(self.wavelengths))
        spd_at_eye = np.zeros(len(self.wavelengths))
        illuminance_h_val = 0.0
        illuminance_v_val = 0.0

        for n in range(self.n_sources):
            dimming = dim_variables[n]
            
            if self.dim_curves is not None and self.dim_curves is not False: # estimated dimming curve
                # Pass the raw 0.0-1.0 guess through the interpolation curve for this channel
                amount_dim = self.dim_curves[n](dimming)
                actual_dimming = amount_dim * on_off_variables[n]
            else: # no dimming curve, assume linear
                actual_dimming = dimming * on_off_variables[n]

            source_spectrum = np.array(self.sources["spds"].iloc[n])
            resulting_spd += (actual_dimming * source_spectrum)
            
            if not tmp: # per channel illuminance, need to loop over channels
                illuminance_h_val += actual_dimming * self.rad_illuminance_h[n]
                h_transfer_factor = self.rad_illuminance_h[n] / self.max_lp_illuminance[n]
                spd_on_desk += (actual_dimming * source_spectrum * h_transfer_factor)

                illuminance_v_val += actual_dimming * self.rad_illuminance_v[n]
                v_transfer_factor = self.rad_illuminance_v[n] / self.max_lp_illuminance[n] # scale
                spd_at_eye += (actual_dimming * source_spectrum * v_transfer_factor)

        if tmp: # source illuminance, no need to loop - only one value
            spds_combined = np.vstack((self.wavelengths, resulting_spd))
            calculated_power = lx.spd_to_power(spds_combined, ptype='pu')
            calculated_power_val = float(np.asarray(calculated_power).item())
            
            h_scaling_factor = self.rad_illuminance_h / self.max_lp_illuminance
            illuminance_h_val = calculated_power_val * h_scaling_factor
            spd_on_desk = resulting_spd * h_scaling_factor

            v_scaling_factor = self.rad_illuminance_v / self.max_lp_illuminance
            illuminance_v_val = calculated_power_val * v_scaling_factor
            spd_at_eye = resulting_spd * v_scaling_factor

        spds = np.vstack((self.wavelengths, resulting_spd))
        
        try: 
            xyz = lx.spectrum.spd_to_xyz(spds, relative=True)
            cct = lx.color.cct.xyz_to_cct(xyz)
            tm30 = lx.color.cri.iestm30.metrics.spd_to_ies_tm30_metrics(spds)
            
            cct_val = float(np.asarray(cct).item())
            tm30_rf_val = float(np.asarray(tm30["Rf"]).item())
            tm30_rg_val = float(np.asarray(tm30["Rg"]).item())
            tm30_bin1_rf_val = float(np.asarray(tm30["Rfi"][0][0]).item())
            tm30_bin1_chroma_val = float(np.asarray(tm30["Rcshj"][0][0]).item())

            alphaopic_calc = ao.alphaopic(spd_at_eye, self.wavelengths, opsin='Mel', lmax=self.species)
            medi_val = float(np.asarray(alphaopic_calc["Luminous"]).item())

            if self.species_nonhuman is not None:
                alphaopic_calc_nonhuman = ao.alphaopic(spd_at_eye, self.wavelengths, opsin='Mel', lmax=self.species_nonhuman)
                medi_nonhuman_val = float(np.asarray(alphaopic_calc_nonhuman["Luminous"]).item())
            
        except (FloatingPointError, ValueError, ZeroDivisionError, Exception): 
            # print(f"\n--- FATAL ERROR IN GENERATION ---")
            # traceback.print_exc()
            # make values zero if there is an error, objective should not choose this solution but the optimization will still proceed
            # rather than failing with a divide by zero error or something
            cct_val = 0
            tm30_rf_val = 0
            tm30_rg_val = 0
            tm30_bin1_rf_val = 0
            tm30_bin1_chroma_val = 100
            illuminance_h_val = 0
            illuminance_v_val = 0
            medi_val = 0
            medi_nonhuman_val = 0


        # constraints
        constraints.append(3500.0 - cct_val)                   # CCT >= 3500
        constraints.append(cct_val - 5500.0)                   # CCT <= 5500
        # constraints.append(85.0 - tm30_rf_val)                   # TM30 Rf >= 85
        # constraints.append(92.0 - tm30_rg_val)                 # TM30 Rg >= 92
        # constraints.append(tm30_rg_val - 110.0)                 # TM30 Rg <= 110
        # constraints.append(85.0 - tm30_bin1_rf_val)            # TM30 Bin 1 Rf >= 85
        constraints.append(95.0 - tm30_rf_val)                   # TM30 Rf >= 70
        constraints.append(97.0 - tm30_rg_val)                 # TM30 Rg >= 97
        constraints.append(tm30_rg_val - 110.0)                 # TM30 Rg <= 110
        constraints.append(85.0 - tm30_bin1_rf_val)            # TM30 Bin 1 Rf >= 78
        constraints.append(-0.07 - tm30_bin1_chroma_val)        # TM30 Bin 1 Chroma >= -7%
        constraints.append(tm30_bin1_chroma_val - 0.07)         # TM30 Bin 1 Chroma <= 7%
        constraints.append(self.target_illuminance_h - illuminance_h_val) # Lux >= Target
        constraints.append(illuminance_h_val - self.max_illum_h) # Lux <= Maximum
        # constraints.append(self.target_illuminance_v - illuminance_v_val) # Lux >= Target
        constraints.append(self.target_medi - medi_val)        # MEDI >= Target
        if self.species_nonhuman is not None:
            constraints.append(self.target_medi_nonhuman - medi_nonhuman_val)
        out["G"] = constraints

        actual_dimming = dim_variables * on_off_variables

        # set up objective function(s)
        obj = []
        for n in range(len(self.objective)):
            if self.objective[n] == "max_medi":
                obj.append(-medi_val) # maximize MEDI
            elif self.objective[n] == "min_medi":
                obj.append(medi_val) # minimize MEDI
            elif self.objective[n] == "max_tm30_rf":
                obj.append(-tm30_rf_val) # maximize TM30 Rf
            elif self.objective[n] == "max_tm30_rg":
                obj.append(-tm30_rg_val) # maximize TM30 Rg
            elif self.objective[n] == "max_cct":
                obj.append(-cct_val) # maximize CCT
            elif self.objective[n] == "min_cct":
                obj.append(cct_val) # minimize CCT
            elif self.objective[n] == "max_illuminance":
                obj.append(-illuminance_h_val) # maximize horizontal illuminance
            elif self.objective[n] == "min_illuminance":
                obj.append(illuminance_h_val) # minimize horizontal illuminance
            elif self.objective[n] == "min_sources":
                obj.append(float(np.sum((on_off_variables*1.05)+actual_dimming))) # minimize number of sources, add a small penalty to encourage less sources
            else:
                raise ValueError(f"Objective {self.objective[n]} not recognized. Please choose from: max_medi, min_medi, max_tm30_fidelity, max_tm30_preference, max_cct, min_cct, max_illuminance, min_illuminance, min_sources.")   
        
        out["F"] = obj
        # print(out["F"])

species = userinput.species # define what species the optimization is for

if userinput.sources != ['telelumen']: # rubik or other source, do not use per channel data
    # print("if")
    if userinput.sources == ['telelumen_lab']:
        wavelengths = pd.read_csv("tlwav.csv", header=None).values.flatten() # input wavelengths for luxpy power calculation
        sources = pd.read_csv("tl.csv") # input spds for luxpy power calculation
        sources["spds"] = sources["spds"].apply(ast.literal_eval) # ensure numpy array is read in as a list of floats, not a string
    else:
        # read in data about spds for the light sources 
        df = pd.read_excel('Light Sources.xlsx', sheet_name = userinput.sources) 
        if isinstance(df, dict):
            # if multiple sheets were read
            df = pd.concat(df.values(), axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        wavelengths = df['Wavelength']
        df = df.drop(columns="Wavelength")
        sources = cols_to_df(df.columns, df)
        sources = sources.rename(columns={"values": "spds"})
        # print(sources)

    # calculate radiance lux
    radiance_lux_h = radiance_simulation(userinput.iesfile[0], sensor_type='horizontal') 
    radiance_lux_v = radiance_simulation(userinput.iesfile[0], sensor_type='vertical') 
    if userinput.sensor_grid == True:
        radiance_lux_grid = radiance_simulation(userinput.iesfile[0], sensor_type='grid_horizontal')

    spd_100_percent = np.sum([np.array(sources["spds"].iloc[i]) for i in range(len(sources))], axis=0) # spd of sources 100% on
    power_inputs = np.vstack((wavelengths, spd_100_percent))
    max_power = lx.spd_to_power(power_inputs, ptype='pu') # max source power from luxpy
    max_power = float(np.asarray(max_power).item()) 

else: # telelumen, use per channel data from json, add dimming curves, etc
    # print("else")
    # input sources and calculate radiance lux
    radiance_lux_h = []
    radiance_lux_v = []
    radiance_lux_gh = []
    file_path = "telelumen_calibration.json"

    with open(file_path, 'r') as f:
        data = json.load(f)

    wavelengths = np.array(data["TestConfig"]["Spectrometer"]["WavelengthArray"])
    np.savetxt("tlwav.csv", wavelengths, delimiter=",")

    measurements = data.get("Data", [])

    raw_json_data = {}
    spds_100 = {}
    dimming_data_raw = {}

    for m in measurements:
        channel = m.get("Channel")
        dim_level = m.get("FracLevelSet")
        
        #  use RadiantOutput to create the dimming curve.
        output_energy = m.get("Colorimetrics", {}).get("RadiantOutput")
        
        if channel and dim_level is not None:

            if dim_level == 1.0:
                raw_json_data[channel] = {"100_percent_measurement": m}
                spds_100[channel] = np.array(m["SpdArr"]) # get spds for the calculations, use 100% on spds

            if output_energy is not None:
                if channel not in dimming_data_raw:
                    dimming_data_raw[channel] = {"x_dim_set": [], "y_measured_out": []} # get data about dimming
                    
                dimming_data_raw[channel]["x_dim_set"].append(dim_level)
                dimming_data_raw[channel]["y_measured_out"].append(output_energy)


    # for n in range(len(spds_100)):
    #     plt.plot(wavelengths, spds_100[list(spds_100.keys())[n]], label=list(spds_100.keys())[n])
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Spectral Power Distribution")
    # plt.title("Spectral Power Distributions")
    # plt.legend()
    # plt.show()
    # set up dimming curves
    dimming_curves = {}

    for channel, points in dimming_data_raw.items():
        # convert to numpy arrays
        x_raw = np.array(points["x_dim_set"])
        y_raw = np.array(points["y_measured_out"])
        
        # sort them in case the JSON measurements were out of order
        sorted_indices = np.argsort(x_raw)
        x_sorted = x_raw[sorted_indices]
        y_sorted = y_raw[sorted_indices]
        
        # normalize the curve
        y_normalized = y_sorted / np.max(y_sorted)
        
        # create the interpolation curve
        curve = interp1d(x_sorted, y_normalized, kind='linear', fill_value="extrapolate")
        dimming_curves[channel] = curve

    sources = pd.DataFrame.from_dict(spds_100, orient='index')
    sources = sources.T
    sources = cols_to_df(sources.columns, sources)
    sources = sources.rename(columns={"values": "spds"})
    sources.to_csv("tl.csv")
    print("TO CSV")

    # print(sources)
    dimming_curves_df = pd.DataFrame.from_dict(dimming_curves, orient='index', columns=['curve'])
    print(dimming_curves_df)

    ordered_curves = []
    for channel_name in sources.index:  
        ordered_curves.append(dimming_curves_df.loc[channel_name, 'curve'])

    base_ies_lumens = 996.12 # 2500.0  # total lumen output from ies file

    radiance_channel_params = {}
    lumens_per_channel = []

    for channel, data in raw_json_data.items():
        # get the lumens for this channel
        ch_lumens = data["100_percent_measurement"]["Colorimetrics"]["LuminousOutput"]
        lumens_per_channel.append(ch_lumens)
        
        # calculate the intensity multiplier
        intensity_multiplier = ch_lumens / base_ies_lumens
        
        # get the raw SPD for this channel
        spd = data["100_percent_measurement"]["SpdArr"]
        spd_array = np.vstack((wavelengths, spd))
        
        # convert SPD to Linear RGB (Radiance requires linear RGB)
        xyz = lx.spectrum.spd_to_xyz(spd_array)
        linear_rgb = lx.color.ctf.colortransforms.xyz_to_srgb(xyz, cspace='sRGB', linear=True)[0]
        
        # clip any negative values to 0 
        linear_rgb = np.clip(linear_rgb, 0, None)
        
        # normalize the RGB for Radiance: Y = 0.265*R + 0.670*G + 0.065*B
        rad_luminance = (0.265 * linear_rgb[0]) + (0.670 * linear_rgb[1]) + (0.065 * linear_rgb[2])
        
        if rad_luminance > 0:
            normalized_rgb = linear_rgb / rad_luminance
        else:
            normalized_rgb = [0, 0, 0]
            
        radiance_channel_params[channel] = {
            "multiplier": intensity_multiplier,
            "R": normalized_rgb[0],
            "G": normalized_rgb[1],
            "B": normalized_rgb[2]
        }

    for ch, params in radiance_channel_params.items():
        m = params['multiplier']
        r, g, b = params['R'], params['G'], params['B']
        ies_file_name = "telelumen octa ies file 22dec18.ies"
        output_prefix = f"fixture_{ch}" # e.g., "fixture_Violet"

        # break the flags into a list 
        rad_params = [
            "-m", f"{m:.4f}", 
            "-c", f"{r:.4f}", f"{g:.4f}", f"{b:.4f}"
        ]
        
        radiance_lux_h += [radiance_simulation(source_type='telelumen', ies2rad_input=rad_params, sensor_type='horizontal')]
        radiance_lux_v += [radiance_simulation(source_type='telelumen', ies2rad_input=rad_params, sensor_type='vertical')]
        if userinput.sensor_grid == True:
            radiance_lux_gh += [radiance_simulation(source_type='telelumen', ies2rad_input=rad_params, sensor_type='grid_horizontal')]


        print ("rl, h", radiance_lux_h)
        print("rl, v", radiance_lux_v)
        # print("rl, gh", radiance_lux_gh)
    max_power = lumens_per_channel
    print("mp", max_power)

# # print(sources)
# # print(dimming_curves_df)
# # print(radiance_lux)
        
# # for n in range(len(sources)):
# #     luxpy_power = lx.spd_to_power(np.vstack((wavelengths, np.array(sources["spds"].iloc[n]))), ptype='pu')
# #     pow_per_channel.append(float(np.asarray(luxpy_power).item()))

# # define objectives
# # objectives = ['max_medi', 'min_medi', 'max_tm30_rf', 'max_tm30_rg', 'max_cct', 'min_cct', "max_illuminance", "min_illuminance", "min_sources"]
# # objectives = ['min_sources']
# # objectives = ['max_tm30_rf', 'max_tm30_rg']
# objectives = ['max_medi', 'max_tm30_rf', 'max_tm30_rg', "max_illuminance", "min_sources"] # objectives i want to consider

# start_time = time.time() # start timer for this optimization run

# print(f"starting optimization for {objectives}")
# print("inputs")
# print("horizontal", radiance_lux_h)
# print("vertical", radiance_lux_v)
# print("max_power", max_power)

# print("total horizontal illuminance from radiance simulation: ", sum(radiance_lux_h))

# problem = optimization_problem(sources, wavelengths, radiance_lux_h, radiance_lux_v, max_power,
#                                target_medi_nonhuman=None,
#                                species=species, objective=objectives, dim_curves=ordered_curves, sensor_grid=userinput.sensor_grid) # set up the problem

# algorithm = NSGA2(
#     pop_size=250,
#     sampling=MixedVariableSampling(),
#     mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
#     eliminate_duplicates=MixedVariableDuplicateElimination()
# ) # define how it will be solved

# res = minimize(problem, algorithm, ('n_gen', 150), seed=1, verbose=True) # run the problem

# print("optimization complete")

# end_time = time.time() # end timer for this optimization run
# elapsed_time = end_time - start_time
# print(f"\nElapsed Time for this optimization: {elapsed_time:.2f} seconds")

# import datetime

# def save_results_to_csv(res, problem, objectives, F_display, all_metrics, filename="optimization_results.csv"):
#     """Saves the optimization inputs, constraints, and outputs to a CSV."""
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     # 1. Capture the Inputs & Constraints from the problem definition
#     inputs_summary = {
#         "Timestamp": timestamp,
#         "Target Illuminance (H)": problem.target_illuminance_h,
#         "Target Illuminance (V)": problem.target_illuminance_v,
#         "Target MEDI": problem.target_medi,
#         "Target MEDI (Non-Human)": problem.target_medi_nonhuman,
#         "Min Dimming": problem.min_dim,
#         "Max Dimming": problem.max_dim,
#         "Objectives Optimized": " | ".join(objectives)
#     }
    
#     data = []
#     solutions_list = [res.X] if isinstance(res.X, dict) else res.X
    
#     # 2. Iterate through all solutions on the Pareto front
#     for i, sol in enumerate(solutions_list):
#         row = {"Solution_Index": i}
#         row.update(inputs_summary) 
        
#         # Add the raw decision variables (on/off and dimming levels)
#         for k, v in sol.items():
#             row[k] = v
            
#         # Add the true (un-negated) objective values
#         for j, obj_name in enumerate(objectives):
#             row[f"Obj: {obj_name}"] = F_display[i, j]
            
#         # Add the physical metrics calculated for this specific solution
#         if all_metrics[i]:
#             for metric_name, metric_val in all_metrics[i].items():
#                 row[f"Metric: {metric_name}"] = metric_val
                
#         data.append(row)
        
#     # 3. Save to CSV
#     df = pd.DataFrame(data)
#     # Generate a unique filename with the timestamp
#     safe_time = timestamp.replace(":", "-").replace(" ", "_")
#     final_filename = f"opt_results_{safe_time}.csv"
    
#     df.to_csv(final_filename, index=False)
#     print(f"\nSaved optimization data to: {final_filename}")

# # my_thresholds = {
# #     'medi': 275.0,
# #     'rf': 86.0,
# #     'rg': 92.0,
# #     'rf_bin1': 85.0,
# #     'rg_bin1': -0.07,
# #     'illuminance': 300, # min illuminance limit
# #     'max_illuminance': 600    # Maximum illuminance limit
# # }

# # # Call the updated plot_results with the thresholds dictionary appended
# # plot_results(res, problem, objectives, sources, wavelengths, radiance_lux_h, radiance_lux_v,
# #              max_power, species, reference=userinput.ref_source, thresholds=my_thresholds) 

# # Plot/print results based on number of objectives
# plot_results(res, problem, objectives, sources, wavelengths, radiance_lux_h, radiance_lux_v, max_power, species, reference=userinput.ref_source, ref_scale=True)

# plot_results(res, problem, objectives, sources, wavelengths, radiance_lux_h, radiance_lux_v,
#              max_power, species, reference=userinput.ref_source, ref_scale=True, selected_sol=True)

# # Render a specific solution (e.g., index 0)
# # If it's a single objective, res.X is a dict. If multiple, it's a list of dicts.
# if isinstance(res.X, dict):
#     best_solution = res.X 
# else:
#     best_solution = res.X[0] # Pick the first solution on the Pareto front (or whichever index you want)

# # render_optimized_space_spectral(best_solution, problem, raw_json_data, wavelengths, filename="my_optimized_room.hdr")