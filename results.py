import os
import pandas as pd
import json
import pickle


def results_append():
    experiment_results = []
    # for exp_no, variable_list in zip([1],[[5,10,20,30,50,70,100]]):
    for exp_no, variable_list in zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13],
                                     [[5, 10, 20, 30, 50, 70, 100], [234, 456, 789, 141516, 333],
                                      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [2, 3, 4, 5], [3, 4, 5, 6],
                                      [3, 4, 5, 6], [''], [''], [''], ['random-normal', 'he-normal', 'xavier'], [''],
                                      ['']]):
        for set_no in range(1, 9):
            for run in range(0, 5):
                for variable in variable_list:
                    prefix = "experiment_" + str(exp_no) + "_" + str(set_no) + "_" + str(variable)
                    r1 = open(os.path.join("models_data", prefix, "results", "prediction_summary.json")).read()
                    results = json.loads(r1)
                    experiment_results.append([exp_no, variable, set_no, run, 'dev', results['dev'][str(run)]['QWK']])
                    experiment_results.append(
                        [exp_no, variable, set_no, run, 'holdout', results['holdout'][str(run)]['QWK']])
    return experiment_results


experiment_results = results_append()

experiment_results_df = pd.DataFrame(experiment_results)
experiment_results_df = experiment_results_df.rename(index=str,
                                                     columns={0: "experiment_number", 1: "variable", 2: "essay_set",
                                                              3: "run", 4: "dev_or_holdout", 5: "QWK"})
experiment_results_final = \
experiment_results_df.groupby(["experiment_number", "variable", "essay_set", "dev_or_holdout"]).mean().reset_index()[
    ["experiment_number", "variable", "essay_set", "dev_or_holdout", 'QWK']]

with open('expt_results.pkl', 'wb') as f:
    pickle.dump(experiment_results_final, f)
