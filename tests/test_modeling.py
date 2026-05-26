from tool_hallucination_detection.modeling import _trainer_kwargs, _training_args_kwargs


class EvalStrategyTrainingArguments:
    def __init__(
        self,
        output_dir,
        per_device_train_batch_size=None,
        per_device_eval_batch_size=None,
        num_train_epochs=None,
        eval_strategy=None,
        save_strategy=None,
        remove_unused_columns=None,
    ):
        pass


class EvaluationStrategyTrainingArguments:
    def __init__(
        self,
        output_dir,
        per_device_train_batch_size=None,
        per_device_eval_batch_size=None,
        num_train_epochs=None,
        evaluation_strategy=None,
        save_strategy=None,
        remove_unused_columns=None,
    ):
        pass


def test_training_args_uses_eval_strategy_when_available(tmp_path):
    kwargs = _training_args_kwargs(EvalStrategyTrainingArguments, tmp_path, batch_size=2, epochs=1)
    assert kwargs["eval_strategy"] == "epoch"
    assert "evaluation_strategy" not in kwargs


def test_training_args_uses_evaluation_strategy_for_old_versions(tmp_path):
    kwargs = _training_args_kwargs(EvaluationStrategyTrainingArguments, tmp_path, batch_size=2, epochs=1)
    assert kwargs["evaluation_strategy"] == "epoch"
    assert "eval_strategy" not in kwargs


class ProcessingClassTrainer:
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
    ):
        pass


class TokenizerTrainer:
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
    ):
        pass


def test_trainer_kwargs_use_processing_class_when_available():
    kwargs = _trainer_kwargs(ProcessingClassTrainer, 1, 2, 3, 4, "tok", 5)
    assert kwargs["processing_class"] == "tok"
    assert "tokenizer" not in kwargs


def test_trainer_kwargs_use_tokenizer_for_old_versions():
    kwargs = _trainer_kwargs(TokenizerTrainer, 1, 2, 3, 4, "tok", 5)
    assert kwargs["tokenizer"] == "tok"
    assert "processing_class" not in kwargs
