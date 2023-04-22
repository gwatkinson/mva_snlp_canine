# Run all NLI experiments
source scripts/run_nli_exps.sh

# Evaluate all NLI experiments
source scripts/evaluate_nli_exps.sh

# Visualise all NLI experiments
source scripts/visualise_nli_results.sh

# Run all NLI attacks
source scripts/evaluate_nli_attacks.sh

# Add all NLI results to git (ignored by default)
source scripts/add_nli_results.sh
