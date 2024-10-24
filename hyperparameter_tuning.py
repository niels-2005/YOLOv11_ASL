from model import get_base_model

MODEL = get_base_model()
DATA = "data.yaml"
EPOCHS = 50
ITERATIONS = 50
OPTIMIZER = "AdamW"
PLOTS = False
SAVE = False
VAL = False


# Perform Hyperparameter Tuning on YOLO v11 Model, 50 Iterations for 50 Epochs!
if __name__ == "__main__":
    MODEL.tune(
        data=DATA,
        epochs=EPOCHS,
        iterations=ITERATIONS,
        optimizer=OPTIMIZER,
        plots=PLOTS,
        save=SAVE,
        val=VAL,
    )
