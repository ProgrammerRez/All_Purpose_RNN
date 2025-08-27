    X, y, total_words, max_len, tokenizer = preprocess_data(
        filepath=dataset_path, num_words=2000, padding="post"
    )
    model = build_model(total_words=total_words, max_len=max_len, mode="SimpleRNN")

    model, history, loss, acc = train_or_retrain(
        X=X, y=y, model=model, earlystop=False, show_summary=True
    )