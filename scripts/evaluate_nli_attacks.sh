echo "Evaluating NLI experiments with modified datasets"
echo "========================"


echo "Evaluating modified 100k_en"
echo "------------------------"
nli_augmented_dataset 100k_en --language_subset 'en,fr,es,bg,ru,de,el'
nli_visualise_results 100k_en --attacked
echo "========================"


echo "Evaluating modified 300k_fr"
echo "------------------------"
nli_augmented_dataset 300k_fr --language_subset 'en,fr,es,bg,ru,de,el'
nli_visualise_results 300k_fr --attacked
echo "========================"


echo "Evaluating modified 200k_ar_tr"
echo "------------------------"
nli_augmented_dataset 200k_ar_tr --language_subset 'ar,tr,ur,hi,sw,es'
nli_visualise_results 200k_ar_tr --attacked
echo "========================"


echo "Evaluating modified 100k_zh_th"
echo "------------------------"
nli_augmented_dataset 100k_zh_th --language_subset 'zh,th,hi,ur,vi'
nli_visualise_results 100k_zh_th --attacked
echo "========================"


echo "Evaluating modified 300k_all"
echo "------------------------"
nli_augmented_dataset 300k_all --language_subset 'en,fr,es,bg,ru,de,el'
nli_visualise_results 300k_all --languages 'en, fr, es, bg, ru' --attacked
echo "========================"
