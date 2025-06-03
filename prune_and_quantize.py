import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


def prune_model_manual(
    model_path, X_train, y_train, X_val, y_val, sparsity=0.5
):
    """
    Manual implementation của magnitude-based pruning theo MADONNA
    Không cần tensorflow_model_optimization
    """

    model = tf.keras.models.load_model(model_path)
    print(f"Original model parameters: {model.count_params()}")

    pruned_model = tf.keras.models.clone_model(model)
    pruned_model.set_weights(model.get_weights())

    for layer in pruned_model.layers:
        if hasattr(layer, "kernel"):
            weights = layer.get_weights()
            kernel = weights[0]

            flat_weights = np.abs(kernel.flatten())
            threshold = np.percentile(flat_weights, sparsity * 100)

            mask = np.abs(kernel) >= threshold
            pruned_kernel = kernel * mask

            weights[0] = pruned_kernel
            layer.set_weights(weights)

    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Pruned model parameters: {pruned_model.count_params()}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    print("Fine-tuning pruned model...")
    history = pruned_model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    pruned_model_path = "model_pruned.keras"
    pruned_model.save(pruned_model_path)

    return pruned_model_path, pruned_model, history


def analyze_sparsity(model):
    """Phân tích sparsity của model sau pruning"""
    total_params = 0
    zero_params = 0

    for layer in model.layers:
        if hasattr(layer, "kernel"):
            weights = layer.get_weights()[0]
            total_params += weights.size
            zero_params += np.sum(
                np.abs(weights) < 1e-8
            )

    sparsity = zero_params / total_params if total_params > 0 else 0
    print(f"Model sparsity: {sparsity:.2%}")
    print(f"Zero parameters: {zero_params}/{total_params}")

    return sparsity


def quantize_model(model_path, X_test, y_test):
    """
    Quantize model theo phương pháp MADONNA
    - Sử dụng Post-training quantization
    - Chuyển đổi sang TensorFlow Lite với INT8 quantization
    """

    model = tf.keras.models.load_model(model_path)
    print(f"Model size before quantization: {get_model_size(model_path):.2f} MB")

    def representative_data_gen():
        """Generator cho representative dataset"""
        for i in range(min(100, len(X_test))):
            yield [X_test[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    try:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    except:
        print(
            "Warning: Full INT8 quantization not supported, using default optimization"
        )
        pass

    quantized_model = converter.convert()

    quantized_model_path = model_path.replace(".keras", "_quantized.tflite")
    with open(quantized_model_path, "wb") as f:
        f.write(quantized_model)

    print(f"Quantized model size: {get_model_size(quantized_model_path):.2f} MB")

    return quantized_model_path


def get_model_size(model_path):
    """Tính size của model file"""
    return os.path.getsize(model_path) / (1024 * 1024)


def evaluate_quantized_model(quantized_model_path, X_test, y_test):
    """Đánh giá performance của quantized model"""

    interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []

    for i in range(len(X_test)):
        input_data = X_test[i : i + 1].astype(np.float32)

        input_scale, input_zero_point = input_details[0].get("quantization", (0.0, 0))
        if input_scale != 0:
            input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])

        output_scale, output_zero_point = output_details[0].get(
            "quantization", (0.0, 0)
        )
        if output_scale != 0:
            output_data = (
                output_data.astype(np.float32) - output_zero_point
            ) * output_scale

        predictions.append(output_data[0])

    predictions = np.array(predictions)
    y_pred_quantized = (predictions > 0.5).astype(int).flatten()

    accuracy = np.mean(y_pred_quantized == y_test)
    print(f"Quantized model accuracy: {accuracy:.4f}")

    return accuracy, predictions


def compare_models(
    original_model_path, pruned_model_path, quantized_model_path, X_test, y_test
):
    """So sánh performance giữa original, pruned và quantized model"""

    original_model = tf.keras.models.load_model(original_model_path)
    original_pred = original_model.predict(X_test)
    original_acc = np.mean((original_pred > 0.5).astype(int).flatten() == y_test)

    pruned_model = tf.keras.models.load_model(pruned_model_path)
    pruned_pred = pruned_model.predict(X_test)
    pruned_acc = np.mean((pruned_pred > 0.5).astype(int).flatten() == y_test)

    quantized_acc, quantized_pred = evaluate_quantized_model(
        quantized_model_path, X_test, y_test
    )

    original_size = get_model_size(original_model_path)
    pruned_size = get_model_size(pruned_model_path)
    quantized_size = get_model_size(quantized_model_path)

    final_compression_ratio = original_size / quantized_size

    print("\n" + "=" * 60)
    print("MODEL OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Original Model:")
    print(f"  - Size: {original_size:.2f} MB")
    print(f"  - Accuracy: {original_acc:.4f}")
    print(f"  - Parameters: {original_model.count_params()}")

    print(f"\nPruned Model:")
    print(f"  - Size: {pruned_size:.2f} MB")
    print(f"  - Accuracy: {pruned_acc:.4f}")
    print(f"  - Parameters: {pruned_model.count_params()}")

    sparsity = analyze_sparsity(pruned_model)

    print(f"\nPruned + Quantized Model:")
    print(f"  - Size: {quantized_size:.2f} MB")
    print(f"  - Accuracy: {quantized_acc:.4f}")

    print(f"\nOptimization Results:")
    print(f"  - Total size reduction: {final_compression_ratio:.2f}x")
    print(f"  - Final accuracy drop: {(original_acc - quantized_acc):.4f}")
    print(f"  - Relative accuracy: {(quantized_acc/original_acc)*100:.2f}%")
    print(f"  - Model sparsity: {sparsity:.2%}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    models = ["Original", "Pruned", "Pruned+Quantized"]
    sizes = [original_size, pruned_size, quantized_size]
    colors = ["blue", "green", "red"]
    bars = plt.bar(models, sizes, color=colors, alpha=0.7)
    plt.ylabel("Size (MB)")
    plt.title("Model Size Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    for bar, size in zip(bars, sizes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{size:.2f}MB",
            ha="center",
            va="bottom",
        )

    plt.subplot(1, 4, 2)
    accuracies = [original_acc, pruned_acc, quantized_acc]
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.ylim([0.8, 1.0])
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
        )

    plt.subplot(1, 4, 3)
    compression_ratios = [1, original_size / pruned_size, final_compression_ratio]
    bars = plt.bar(models, compression_ratios, color=colors, alpha=0.7)
    plt.ylabel("Compression Ratio")
    plt.title("Compression Ratio")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    for bar, ratio in zip(bars, compression_ratios):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{ratio:.1f}x",
            ha="center",
            va="bottom",
        )

    plt.subplot(1, 4, 4)
    plt.scatter(
        original_pred.flatten(),
        quantized_pred.flatten(),
        alpha=0.6,
        color="red",
        label="Original vs Quantized",
    )
    plt.scatter(
        original_pred.flatten(),
        pruned_pred.flatten(),
        alpha=0.4,
        color="green",
        label="Original vs Pruned",
    )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.8)
    plt.xlabel("Original Predictions")
    plt.ylabel("Optimized Predictions")
    plt.title("Prediction Correlation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "data/prune_quantization_results.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    """Main function để chạy quá trình Prune + Quantize theo MADONNA"""

    df = pd.read_csv("data/domain_dataset_cleaned.csv")
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print("Model Optimization Process: Manual Prune → Quantize")
    print("=" * 55)

    original_model_path = "model_13feat.keras"

    # Step 1: Prune model (manual implementation)
    print("\nStep 1: Pruning model (manual magnitude-based)...")
    pruned_model_path, pruned_model, prune_history = prune_model_manual(
        original_model_path, X_train_final, y_train_final, X_val, y_val, sparsity=0.5
    )

    # Step 2: Quantize pruned model
    print("\nStep 2: Quantizing pruned model...")
    quantized_model_path = quantize_model(pruned_model_path, X_test, y_test)

    # Step 3: Compare all models
    print("\nStep 3: Comparing models...")
    compare_models(
        original_model_path, pruned_model_path, quantized_model_path, X_test, y_test
    )

    print("\nOptimization completed successfully!")
    print(f"Pruned model saved as: {pruned_model_path}")
    print(f"Final optimized model saved as: {quantized_model_path}")


if __name__ == "__main__":
    main()
