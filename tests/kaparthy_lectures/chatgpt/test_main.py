from deep_learning.kaparthy_lectures.chatgpt.main import main


def test_main():
    main(
        batch_size=1,
        train_iters=2,
        eval_iters=1,
        eval_interval=2,
        block_size=1,
        device="cpu",
        additional_predicted_chars=1,
    )
