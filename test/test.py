from pathlib import Path

import matplotlib.pyplot as plt

import guess_right as gr


def random_batch(n: int, shape: list[int], mu: float, sigma: float) -> list[gr.Tensor]:
    return [gr.Tensor.normal(shape, mu, sigma) for _ in range(n)]


def plot_results(
    x_test: list[gr.Tensor],
    y_test: list[gr.Tensor],
    y_pred: list[gr.Tensor],
    train_history: list[float],
    test_history: list[float],
    output_path: Path,
) -> None:
    # funziona solo per input_dim == output_dim == 1 (Tensor scalari -> .item())
    x_vals = [t.item() for t in x_test]
    y_true_vals = [t.item() for t in y_test]
    y_pred_vals = [t.item() for t in y_pred]

    order = sorted(range(len(x_vals)), key=lambda i: x_vals[i])
    x_sorted = [x_vals[i] for i in order]
    y_true_sorted = [y_true_vals[i] for i in order]
    y_pred_sorted = [y_pred_vals[i] for i in order]

    fig, (ax_reg, ax_loss) = plt.subplots(1, 2, figsize=(8, 4), dpi=200)

    ax_reg.scatter(x_sorted, y_true_sorted, label="target", alpha=0.7)
    ax_reg.plot(x_sorted, y_pred_sorted, color="tab:red", label="predizione")
    ax_reg.set_xlabel("x")
    ax_reg.set_ylabel("y")
    ax_reg.set_title("Regressione: target vs predizione (test set)")
    ax_reg.legend()

    epochs = range(1, len(train_history) + 1)
    ax_loss.plot(epochs, train_history, label="train loss")
    ax_loss.plot(epochs, test_history, label="test loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_title("Andamento della loss")
    ax_loss.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.show()
    print(f"plot salvato in {output_path}")


def main() -> None:
    input_dim = 1
    output_dim = 1

    slope = gr.Tensor.normal([input_dim, output_dim], 1, 1)
    intercept = gr.Tensor.normal([output_dim], 2, 0.1)

    # training set
    x_train = random_batch(256, [input_dim], 0, 2)
    y_train = [
        gr.matmul(slope.transpose(), x) + intercept + gr.Tensor.normal([1], 0, 0.4)
        for x in x_train
    ]
    train = gr.Dataset(x_train, y_train)
    train_loader = gr.Loader(train, 32, False)

    # test set
    x_test = random_batch(32, [input_dim], 0, 2)
    y_test = [
        gr.matmul(slope.transpose(), x) + intercept + gr.Tensor.normal([1], 0, 0.4)
        for x in x_test
    ]

    # definizione del modello
    model = gr.Model()
    hidden_dim = 64
    model.add_linear(input_dim, hidden_dim)
    model.add_relu()
    model.add_linear(hidden_dim, hidden_dim)
    model.add_relu()
    model.add_linear(hidden_dim, output_dim)

    sgd = gr.StochasticGradientDescent(model.parameters(), 1e-3)
    loss_fn = gr.MeanSquaredError()

    max_epochs = 200
    train_history: list[float] = []
    test_history: list[float] = []

    # training loop
    for epoch in range(max_epochs):
        for i in range(train_loader.size()):
            # preleva un batch dal dataloader
            batch = train_loader.get()
            x_batch = batch.inputs
            y_batch = batch.targets

            # azzera i gradienti
            sgd.zero_grad()

            # forward pass
            pred = model(x_batch)

            # calcolo della loss
            loss = loss_fn(pred, y_batch)

            # calcolo dei gradienti
            loss.backward()

            # aggiornamento dei parametri
            sgd.step()

            print(
                f"epoch {epoch + 1:>4}, batch {i + 1}/{train_loader.size()},"
                f"  loss: {loss.item():>6.4f}",
                end="\r",
                flush=True,
            )
        if (epoch + 1) % 50 == 0:
            print()

        with gr.no_grad():
            # predizione finale sul dataset completo e storico della loss
            train_loss = loss_fn(model(x_train), y_train)
            train_history.append(train_loss.item())

            test_loss = loss_fn(model(x_test), y_test)
            test_history.append(test_loss.item())

    print(f"final training loss: {train_history[-1]:>6.4f}")
    print(f"final test loss: {test_history[-1]:>6.4f}")

    with gr.no_grad():
        y_pred_test = model(x_test)

    plot_results(
        x_test,
        y_test,
        y_pred_test,
        train_history,
        test_history,
        Path(__file__).parent / "regression_plot.png",
    )


if __name__ == "__main__":
    main()
