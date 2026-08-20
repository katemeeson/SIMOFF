from itertools import combinations
import numpy as np
import pandas as pd
import math
import warnings
from scipy.optimize import dual_annealing
from cobra import Model as CobraModel

"""
INPUT
    model: CobraModel in an .xml format. Prior media conditions or omics constraints should be added before running SIMOFF
    input_reactions: List of reactions IDs from CobraModel that user would like to explore as potential reactions in multi-objective function
    qualitative_constraints: Dictionary of reactions IDs (keys) from CobraModel and experimental fluxes (values) converted to -1 (uptake), 0 (minimal exchange)     or +1 (secretion)
    bounds: sequence or Bounds for variables (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html dual_annealing function); default is None
    max_iter: int, optional. The maximum number of global search iterations. Default value is 1000 (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html dual_annealing function)
    initialtemp: float, optional. The initial temperature, use higher values to facilitates a wider search of the energy landscape, allowing dual_annealing to escape local minima that it is trapped in. Default value is 5230. Range is (0.01, 5.e4) (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html dual_annealing function). Default value is 5230
    maxfun: int, optional. Soft limit for the number of objective function calls. If the algorithm is in the middle of a local search, this number will be exceeded, the algorithm will stop just after the local search is done (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html dual_annealing function). Default value is 10000000
    stop_at_first_perfect: bool, optional. If True, SIMOFF stops when the first 100% accurate solution is found. If False, the successful annealing run continues to max_iter/maxfun and returns a dataframe containing all maximum-accuracy solutions.
    
METHOD
    SIMOFF will compare the Flux Balance Analysis flux distribution resulting from each objective function to qualitative_constraints. Dual annealing is used to reach the most appropriate objective function. Hyperparameters should be defined by user

OUTPUT
    dict(zip(objective_reactions, scaled_coeffs)): dictionary of reaction IDs explored in this combination and the coefficient
    accuracy: float, accuracy value achieved by the objective (dict(zip(objective_reactions, scaled_coeffs))) as a decimal

    which of these is the 
    log_df: DataFrame, rows are individual iterations; first column is a list of reaction coefficients for this iteration with the same index as input_reactions; second column is the accuracy (decimal) achieved using these coefficients; third is a dictionary of absolute fluxes (values) predicted for each reaction in qualitative_constraints (keys) for this iteration
    agreement_df: DataFrame, rows are individual reactions in the qualitative_constraints and columns are mismatch values (either 0 for disagreement or 1 for agreement) between the FBA predicted fluxes for each iteration (each iteration is a separate column) and qualitative_criteria
    If stop_at_first_perfect is False, returns (results, max_accuracy_df), where max_accuracy_df contains all iterations that achieved the maximum accuracy observed during the SIMOFF run.
"""

def simoff(
    model,
    input_reactions,
    qualitative_constraints,
    bounds=None,
    max_iter=1000,
    initialtemp=5230,
    maxfun=10000000,
    stop_at_first_perfect=True
):
    """
    Run annealing optimisation on all combinations of input reactions (length>1).
    By default, stops early if perfect accuracy (1.0) is reached.
    
    Returns a dict mapping tuples of reaction IDs to results:
    { combo: (coefficients_dict, accuracy, log_df, agreement_df) }

    If stop_at_first_perfect is False, returns:
    (results, max_accuracy_df)
    """
    warnings.filterwarnings('ignore', 
                           message='DataFrame is highly fragmented', 
                           category=pd.errors.PerformanceWarning)
    
    if not isinstance(model, CobraModel):
        raise TypeError("Model must be a cobra.Model instance")
    
    # Generate all combos of length >1
    all_combos = []
    for r in range(2, len(input_reactions)+1):
        all_combos.extend(combinations(input_reactions, r))
    all_combos = [list(t) for t in all_combos]
    print(f"There are {len(all_combos)} possible combinations of objectives.\n")
    
    results = {}

    # --- Allow uptake if needed ---
    for k, v in qualitative_constraints.items():
        if v == -1 and not model.reactions.get_by_id(k).reversibility:
            model.reactions.get_by_id(k).lower_bound = -1000
            print(f"Updated bounds for {k} to allow uptake")

    # --- Storage ---
    results_summary = []
    perfect_result = None  # to store early stop data

    # ==============================================================
    # 🔹 SINGLE-REACTION TESTS
    # ==============================================================

    print("\n🔸 Running single-reaction objective tests...\n")

    for rxn_id in input_reactions:
        if rxn_id not in qualitative_constraints:
            print(f"   Skipping {rxn_id}: not present in qualitative constraints.")
            continue

        direction = qualitative_constraints[rxn_id]
        rxn = model.reactions.get_by_id(rxn_id)

        # Reset objectives
        for r in model.reactions:
            r.objective_coefficient = 0.0

        # Set direction: +1 → maximise, -1 → minimise
        rxn.objective_coefficient = float(direction)
        sense = "maximisation" if direction == 1 else "minimisation"

        solution = model.optimize()
        if solution is None or solution.status != "optimal":
            accuracy = 0.0
        else:
            fluxes = solution.fluxes[list(qualitative_constraints.keys())]
            expected = np.array([qualitative_constraints[r] for r in list(qualitative_constraints.keys())])
            rounded_fluxes = np.where(np.abs(fluxes) < 1e-6, 0, np.sign(fluxes).astype(int))
            mismatch_count = np.count_nonzero(rounded_fluxes != expected)
            accuracy = 1 - (mismatch_count / len(expected))

        print(f"   Single-objective {rxn_id} ({sense}) → Accuracy: {accuracy:.2%}")
        results_summary.append({
            'combination': [rxn_id],
            'coefficients': {rxn_id: direction},
            'accuracy': accuracy
        })

    print("\n✅ Single-objective tests complete.\n")

    print(f"\n🧊 Optimisation: fitting objective(s) {input_reactions} "
          f"to match {len(qualitative_constraints)} qualitative constraints")

    class PerfectAccuracyFound(Exception):
        pass
    
    def _annealing_single(objective_reactions):
        selected_rxns = list(qualitative_constraints.keys())
        results_log = []
        agreement_matrix = {}
        early_stop_data = None

        def evaluate_solution(c_raw):
            nonlocal early_stop_data
            if np.sum(np.abs(c_raw)) == 0:
                c = np.zeros_like(c_raw)
            else:
                c = c_raw / np.sum(np.abs(c_raw))
            for i, rxn_id in enumerate(objective_reactions):
                model.reactions.get_by_id(rxn_id).objective_coefficient = c[i]
            solution = model.optimize()
            flux_dict = {}
            agreement_dict = {}
            if solution is None:
                accuracy = 0.0
                mismatch_count = len(selected_rxns)
                for rxn_id in selected_rxns:
                    flux_dict[rxn_id] = None
                    agreement_dict[rxn_id] = 0
            else:
                fluxes = solution.fluxes[selected_rxns]
                expected = np.array([qualitative_constraints[r] for r in selected_rxns])
                rounded_fluxes = np.where(np.abs(fluxes) < 1e-6, 0, np.sign(fluxes).astype(int))
                mismatch_count = np.count_nonzero(rounded_fluxes != expected)
                accuracy = 1 - (mismatch_count / len(expected))
                flux_dict = dict(zip(selected_rxns, fluxes))
                agreement_dict = {rxn_id: int(round_f == exp)
                                  for rxn_id, round_f, exp in zip(selected_rxns, rounded_fluxes, expected)}
            results_log.append({'coefficients': c.tolist(), 'accuracy': accuracy, 'fluxes': flux_dict})
            agreement_matrix[len(results_log) - 1] = agreement_dict

            if accuracy == 1.0 and stop_at_first_perfect:
                early_stop_data = (dict(zip(objective_reactions, c)), accuracy)
                raise PerfectAccuracyFound
            return 1 - accuracy

        # Auto temperature
        temp = initialtemp
        if initialtemp == 'auto':
            random_costs = []
            for _ in range(20):
                rand_coeffs = np.random.uniform(-1, 1, len(objective_reactions))
                cost = evaluate_solution(rand_coeffs)
                random_costs.append(cost)
            avg_delta_e = np.mean([abs(a - b) for a, b in zip(random_costs[:-1], random_costs[1:])])
            temp = max(1.0, avg_delta_e / -math.log(1.0))
            print(f"Estimated initial temperature: {temp:.2f}")

        try:
            result = dual_annealing(
                evaluate_solution,
                bounds if bounds is not None else [(-1,1)]*len(objective_reactions),
                maxiter=max_iter,
                initial_temp=temp,
                maxfun=maxfun
            )
        except PerfectAccuracyFound:
            if early_stop_data is not None:
                coeffs, acc = early_stop_data
                log_df = pd.DataFrame(results_log)
                agreement_df = pd.DataFrame.from_dict(agreement_matrix, orient='index').T
                return coeffs, acc, log_df, agreement_df

        # If not perfect, finalize
        if np.sum(np.abs(result.x)) == 0:
            scaled_coeffs = np.zeros_like(result.x)
        else:
            scaled_coeffs = result.x / np.sum(np.abs(result.x))
        for i, rxn_id in enumerate(objective_reactions):
            model.reactions.get_by_id(rxn_id).objective_coefficient = scaled_coeffs[i]

        solution = model.optimize()
        fluxes = solution.fluxes[selected_rxns]
        expected = np.array([qualitative_constraints[r] for r in selected_rxns])
        rounded_fluxes = np.where(np.abs(fluxes) < 1e-6, 0, np.sign(fluxes).astype(int))
        accuracy = 1 - (np.count_nonzero(rounded_fluxes != expected) / len(expected))
        log_df = pd.DataFrame(results_log)
        agreement_df = pd.DataFrame.from_dict(agreement_matrix, orient='index').T

        return dict(zip(objective_reactions, scaled_coeffs)), accuracy, log_df, agreement_df

    def _collect_max_accuracy_solutions(search_results):
        rows = []
        max_accuracy = None

        for combo, combo_result in search_results.items():
            log_df = combo_result[2]
            if log_df.empty:
                continue

            combo_max = log_df['accuracy'].max()
            if max_accuracy is None or combo_max > max_accuracy:
                max_accuracy = combo_max

        if max_accuracy is None:
            return pd.DataFrame(
                columns=['combination', 'iteration', 'coefficients',
                         'coefficient_dict', 'accuracy', 'fluxes']
            )

        for combo, combo_result in search_results.items():
            log_df = combo_result[2]
            if log_df.empty:
                continue

            max_rows = log_df[np.isclose(log_df['accuracy'], max_accuracy)]
            for iteration, row in max_rows.iterrows():
                coefficients = row['coefficients']
                rows.append({
                    'combination': combo,
                    'iteration': iteration,
                    'coefficients': coefficients,
                    'coefficient_dict': dict(zip(combo, coefficients)),
                    'accuracy': row['accuracy'],
                    'fluxes': row['fluxes']
                })

        return pd.DataFrame(
            rows,
            columns=['combination', 'iteration', 'coefficients',
                     'coefficient_dict', 'accuracy', 'fluxes']
        )

    # Run for each combo
    print('🔸 Running SIMOFF across different reaction combinations...')
    for combo in all_combos:
        print(f"\n🔹 Testing combination{combo}...\n")
        results[tuple(combo)] = _annealing_single(combo)
        if results[tuple(combo)][1] == 1.0:
            print(f"SIMOFF suggested coefficients:{results[tuple(combo)][0]}")
            print("🎯 Perfect 100% accuracy achieved")
            break
        else:
            print(f"SIMOFF suggested coefficients:{results[tuple(combo)][0]}")    
            print(f"Accuracy achieved:{results[tuple(combo)][1]*100}%")

    if not stop_at_first_perfect:
        max_accuracy_df = _collect_max_accuracy_solutions(results)
        return results, max_accuracy_df

    return results
