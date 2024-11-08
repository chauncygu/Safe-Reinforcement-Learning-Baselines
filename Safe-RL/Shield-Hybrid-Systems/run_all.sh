#!/usr/bin/env bash


# $@ is bash for "all arguments"
# 2>&1 | tee SomeFile.txt is bash for redirecting all output to SomeFile.txt
julia "fig-BarbaricMethodAccuracy/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-fig-BarbaricMethodAccuracy.txt
julia "fig-BBGranularityCost/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-fig-BBGranularityCost.txt
julia "fig-BBShieldingResultsGroup/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-fig-BBShieldingResultsGroup.txt
julia "fig-BBShieldRobustness/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-fig-BBShieldRobustness.txt
julia "fig-CCShieldingResultsGroup/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-fig-CCShieldingResultsGroup.txt
julia "fig-DCShieldingResultsGroup/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-fig-DCShieldingResultsGroup.txt
julia "fig-NoRecovery/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-fig-NoRecovery.txt
julia "fig-OPShieldingResultsGroup/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-fig-OPShieldingResultsGroup.txt
julia "fig-RWShieldingResultsGroup/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-fig-RWShieldingResultsGroup.txt
julia "tab-BBSynthesis/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-tab-BBSynthesis.txt
julia "fig-DifferenceRigorousBarbaric/Run Experiment.jl" 2>&1 | tee ~/Results/output-fig-DifferenceRigorousBarbaric.txt
julia "tab-CCSynthesis/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-tab-CCSynthesis.txt
julia "tab-DCSynthesis/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-tab-DCSynthesis.txt
julia "tab-OPSynthesis/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-tab-OPSynthesis.txt
julia "tab-RWSynthesis/Run Experiment.jl" $@ 2>&1 | tee ~/Results/output-tab-RWSynthesis.txt