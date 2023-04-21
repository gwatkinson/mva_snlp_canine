echo "Evaluating NLI experiments"
echo "========================"


echo "Evaluating 100k_en"
echo "------------------------"
nli_evaluate_experiment 100k_en
echo "========================"


echo "Evaluating 300k_fr"
echo "------------------------"
nli_evaluate_experiment 300k_fr
echo "========================"


echo "Evaluating 200k_ar_tr"
echo "------------------------"
nli_evaluate_experiment 200k_ar_tr
echo "========================"


echo "Evaluating 100k_zh_th"
echo "------------------------"
nli_evaluate_experiment 100k_zh_th
echo "========================"


echo "Evaluating 300k_zh_th"
echo "------------------------"
nli_evaluate_experiment 300k_zh_th
echo "========================"
