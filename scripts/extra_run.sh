echo "Running NLI experiments"
echo "========================"


# echo "Running 10k_fr"
# echo "------------------------"
# nli_run_experiment 10k_fr
# nli_evaluate_experiment 10k_fr
# nli_visualise_results 10k_fr

# nli_augmented_dataset 10k_fr --language_subset 'en,fr,es,bg,ru,de,el'
# nli_visualise_results 10k_fr --attacked
# echo "========================"


echo "Running 100k_fr"
echo "------------------------"
nli_run_experiment 100k_fr
nli_evaluate_experiment 100k_fr
nli_visualise_results 100k_fr

nli_augmented_dataset 100k_fr --language_subset 'en,fr,es,bg,ru,de,el'
nli_visualise_results 100k_fr --attacked
echo "========================"


# echo "Running 10k_zh_th"
# echo "------------------------"
# nli_run_experiment 10k_zh_th
# nli_evaluate_experiment 10k_zh_th
# nli_visualise_results 10k_zh_th

# nli_augmented_dataset 10k_zh_th --language_subset 'zh,th,hi,ur,vi'
# nli_visualise_results 10k_zh_th --attacked
# echo "========================"


source scripts/add_nli_results.sh