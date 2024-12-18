import tensorflow as tf
import numpy as np

def tensorflow_training_task(task_id, epochs=10, batch_size=32):
    """
    Simulates TensorFlow model training as a task.
    """
    print(f"Starting TensorFlow Task {task_id}...")

    # Create dummy data
    x_train = np.random.rand(1000, 10)
    y_train = np.random.randint(2, size=(1000, 1))

    # Define a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    print(f"TensorFlow Task {task_id} completed.")
