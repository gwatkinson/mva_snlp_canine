echo "git add --force nli_results/**/models/*/README.md \ \n
    nli_results/**/models/*/trainer_state.json \ \n
    nli_results/**/results/*.pdf \ \n
    nli_results/**/results/*.html \n"

git add --force nli_results/**/models/*/README.md \
    nli_results/**/models/*/trainer_state.json \
    nli_results/**/results/*.pdf \
    nli_results/**/results/*.html

echo "git commit -m 'Add NLI results'"

git commit -m "Add NLI results"
