echo "Running NLI experiments"
echo "========================"


echo "Running 100k_en"
echo "------------------------"
nli_run_experiment 100k_en
echo "========================"


echo "Running 10k_fr"
echo "------------------------"
nli_run_experiment 10k_fr
echo "========================"


echo "Running 300k_fr"
echo "------------------------"
nli_run_experiment 300k_fr
echo "========================"


echo "Running 200k_ar_tr"
echo "------------------------"
nli_run_experiment 200k_ar_tr
echo "========================"


echo "Running 10k_zh"
echo "------------------------"
nli_run_experiment 10k_zh
echo "========================"


echo "Running 100k_zh_th"
echo "------------------------"
nli_run_experiment 100k_zh_th
echo "========================"


echo "Running 300k_zh_th"
echo "------------------------"
nli_run_experiment 300k_zh_th
echo "========================"


echo "Running 300k_all"
echo "------------------------"
nli_run_experiment 300k_all
echo "========================"
