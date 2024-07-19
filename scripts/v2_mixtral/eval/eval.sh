# nohup srun -p MoE --gres gpu:1 bash scripts/eval.sh all /mnt/petrelfs/share_data/quxiaoye/models/Sheared-LLaMA-2.7B True results/Sheared-LLaMA-2.7B 1>logs/eval-all-Sheared-LLaMA-2.7B.log 2>&1 &

mmlu() {
    # MMLU: https://github.com/princeton-nlp/LLM-Shearing/blob/20ebd2645a8ff5fa65874e1347f9891b80e01805/icl_eval/run_eval.sh#L18
    MODEL=$1
    TRUST_REMOTE_CODE=$2
    RESULT_DIR=$3
    mkdir -p $RESULT_DIR

    lm_eval \
        --log_samples \
        --model hf \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks mmlu_computer_security,mmlu_high_school_chemistry,mmlu_philosophy,mmlu_elementary_mathematics,mmlu_prehistory,mmlu_formal_logic,mmlu_high_school_mathematics,mmlu_econometrics,mmlu_moral_scenarios,mmlu_college_mathematics,mmlu_high_school_government_and_politics,mmlu_us_foreign_policy,mmlu_high_school_world_history,mmlu_conceptual_physics,mmlu_college_medicine,mmlu_international_law,mmlu_abstract_algebra,mmlu_logical_fallacies,mmlu_machine_learning,mmlu_medical_genetics,mmlu_public_relations,mmlu_college_biology,mmlu_marketing,mmlu_electrical_engineering,mmlu_anatomy,mmlu_high_school_us_history,mmlu_high_school_biology,mmlu_miscellaneous,mmlu_high_school_psychology,mmlu_sociology,mmlu_business_ethics,mmlu_high_school_geography,mmlu_human_aging,mmlu_high_school_statistics,mmlu_moral_disputes,mmlu_professional_psychology,mmlu_global_facts,mmlu_college_physics,mmlu_nutrition,mmlu_high_school_macroeconomics,mmlu_world_religions,mmlu_professional_medicine,mmlu_high_school_computer_science,mmlu_college_chemistry,mmlu_human_sexuality,mmlu_high_school_microeconomics,mmlu_astronomy,mmlu_professional_accounting,mmlu_high_school_european_history,mmlu_jurisprudence,mmlu_professional_law,mmlu_high_school_physics,mmlu_virology,mmlu_management,mmlu_college_computer_science,mmlu_clinical_knowledge,mmlu_security_studies \
        --num_fewshot 5 \
        --device cuda:0 \
        --batch_size auto \
        --verbosity DEBUG \
        --trust_remote_code \
        --output_path $RESULT_DIR/mmlu
}

bbh() {
    # Big Bench Hard (BBH): https://arxiv.org/pdf/2210.09261.pdf
    MODEL=$1
    TRUST_REMOTE_CODE=$2
    RESULT_DIR=$3
    mkdir -p $RESULT_DIR

    lm_eval \
        --log_samples \
        --model hf \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks bbh_fewshot_boolean_expressions,bbh_fewshot_causal_judgement,bbh_fewshot_date_understanding,bbh_fewshot_disambiguation_qa,bbh_fewshot_dyck_languages,bbh_fewshot_formal_fallacies,bbh_fewshot_geometric_shapes,bbh_fewshot_hyperbaton,bbh_fewshot_logical_deduction_five_objects,bbh_fewshot_logical_deduction_seven_objects,bbh_fewshot_logical_deduction_three_objects,bbh_fewshot_movie_recommendation,bbh_fewshot_multistep_arithmetic_two,bbh_fewshot_navigate,bbh_fewshot_object_counting,bbh_fewshot_penguins_in_a_table,bbh_fewshot_reasoning_about_colored_objects,bbh_fewshot_ruin_names,bbh_fewshot_salient_translation_error_detection,bbh_fewshot_snarks,bbh_fewshot_sports_understanding,bbh_fewshot_temporal_sequences,bbh_fewshot_tracking_shuffled_objects_five_objects,bbh_fewshot_tracking_shuffled_objects_seven_objects,bbh_fewshot_tracking_shuffled_objects_three_objects,bbh_fewshot_web_of_lies,bbh_fewshot_word_sorting \
        --device cuda:0 \
        --batch_size auto \
        --verbosity DEBUG \
        --output_path $RESULT_DIR/bbh
}

reasoning() {
    MODEL=$1
    TRUST_REMOTE_CODE=$2
    RESULT_DIR=$3
    mkdir -p $RESULT_DIR

    lm_eval \
        --log_samples \
        --model hf \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks gsm8k_cot \
        --device cuda:0 \
        --batch_size auto \
        --verbosity DEBUG \
        --output_path $RESULT_DIR/reasoning
}

qa() {
    MODEL=$1
    TRUST_REMOTE_CODE=$2
    RESULT_DIR=$3
    mkdir -p $RESULT_DIR

    lm_eval \
        --log_samples \
        --model hf \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks arc_easy,arc_challenge,boolq \
        --num_fewshot 0 \
        --device cuda:0 \
        --batch_size auto \
        --verbosity DEBUG \
        --output_path $RESULT_DIR/qa
}

extend() {
    MODEL=$1
    TRUST_REMOTE_CODE=$2
    RESULT_DIR=$3
    mkdir -p $RESULT_DIR

    # triviaqa,nq_open
    lm_eval \
        --log_samples \
        --model hf \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks boolq,sciq,piqa,winogrande,asdiv,lambada_openai,openbookqa \
        --num_fewshot 0 \
        --device cuda:0 \
        --batch_size auto \
        --verbosity DEBUG \
        --trust_remote_code \
        --output_path $RESULT_DIR/extend
}

truthfulqa() {
    MODEL=$1
    TRUST_REMOTE_CODE=$2
    RESULT_DIR=$3
    mkdir -p $RESULT_DIR

    lm_eval \
        --log_samples \
        --model hf \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks truthfulqa_mc2 \
        --device cuda:0 \
        --batch_size auto \
        --verbosity DEBUG \
        --output_path $RESULT_DIR/truthfulqa
}

arc() {
    MODEL=$1
    TRUST_REMOTE_CODE=$2
    RESULT_DIR=$3
    mkdir -p $RESULT_DIR

    lm_eval \
        --log_samples \
        --model hf \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --device cuda:0 \
        --batch_size auto \
        --verbosity DEBUG \
        --output_path $RESULT_DIR/arc_25
}

hellaswag() {
    MODEL=$1
    TRUST_REMOTE_CODE=$2
    RESULT_DIR=$3
    mkdir -p $RESULT_DIR

    lm_eval \
        --log_samples \
        --model hf \
        --model_args pretrained=$MODEL,trust_remote_code=$TRUST_REMOTE_CODE \
        --tasks hellaswag \
        --num_fewshot 10 \
        --device cuda:0 \
        --batch_size auto \
        --verbosity DEBUG \
        --output_path $RESULT_DIR/hellaswag_10
}

EVAL_TASK=$1
shift 1
start=$(date +%s)
case $EVAL_TASK in
    mmlu)
        mmlu $* ;;
    bbh)
        bbh $* ;;
    reasoning)
        reasoning $* ;;
    qa)
        qa $* ;;
    extend)
        extend $* ;;
    truthfulqa)
        truthfulqa $* ;;
    arc)
        arc $* ;;
    hellaswag)
        hellaswag $* ;;
    all)
        mmlu $*
        bbh $*
        reasoning $*
        qa $*
        ;;
    smoe)
        mmlu $*
        bbh $*
        reasoning $*
        arc $*
        hellaswag $*
        extend $*
        ;;
    *)
        echo "$EVAL_TASK not recognized!";;
esac
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
