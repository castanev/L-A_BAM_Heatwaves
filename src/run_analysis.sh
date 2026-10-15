#!/bin/bash
#SBATCH -A eapsdept
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G        # leave ~8G for system
#SBATCH --time=0-12:00:00
#SBATCH --output=output_run.txt
#SBATCH --error=error_run.txt

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set echo
  module load anaconda
  environment=/home/castanev/.conda/envs/2024.02-py311/MyEnv
  conda activate $environment
  export PATH=$environment/bin:$PATH

  # Run from this script's directory (config_v2.yaml + numbered .py live here)
  cd "$(dirname "$0")"

  name='ERA5'
  percentile=95
  region="US"
  case="${name}_P${percentile}_${region}_detrended_object"

  name_land='GLEAM'
  path_outputs_land="/scratch/negishi/${USER}/${name_land}/"
  path_outputs_case_land="$path_outputs_land/${case}/"
  path_case_land="/home/castanev/land_atmosphere/${case}/"
  path_figures_land="$path_case_land/Figures/"
  path_figures_all="$path_case_land/../Figures/"

  mkdir -p "$path_case_land" "$path_figures_land" "$path_outputs_case_land" "$path_figures_all"

  # Pipeline order (scripts present in this folder)
  python3 0_data.py --config config_v2.yaml
  python3 1_heatwaves_detection_teng_clusters_object.py --config config_v2.yaml
  python3 2_analysis_EOF_meanPC_AprMay.py --config config_v2.yaml
  python3 3_breakpoints_optimized_LE.py --config config_v2.yaml
  python3 3_map_separation_regimes_LE.py --config config_v2.yaml
  python3 4_analysis_regimes_LE_allHWdays_allHWarea_regional.py --config config_v2.yaml
  python3 5_coupling_conditional.py --config config_v2.yaml
  python3 6_BAM_precondition.py
  python3 6_BAM_phases_persistent_precondition.py
  python3 7_z500_regression_BAM_EOF3.py --config config_v2.yaml
  python3 8_prob_HW_BAM_persistent_precondition_thres125_lateApr_allMay.py
  python3 9_precipitation_evolution_BAM.py
