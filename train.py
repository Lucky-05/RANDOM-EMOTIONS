import os

# Correct the paths with either double backslashes or raw string literals
c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
a = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib"
b = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include"
gstreamerPath = r"C:\gstreamer\1.0\msvc_x86_64\bin"

# Adding the required directories to the DLL search path
os.add_dll_directory(gstreamerPath)
os.add_dll_directory(c)
os.add_dll_directory(a)
os.add_dll_directory(b)

# Wrap the main code in the main guard
if __name__ == "__main__":
    from ultralytics import YOLO

    # Load the model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")    # load a pretrained model (recommended for training)

    # Train the model
    model.train(data="data.yaml", epochs=250)
    
    # You can uncomment the following lines to run validation or make predictions:
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
