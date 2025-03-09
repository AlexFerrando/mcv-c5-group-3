from ultralytics import YOLO
# Load a model
model = YOLO("/ghome/c5mcv03/mcv-c5-group-3/outputs/pol/job_outputs/max_resolution_noAugmentation_frozen233/weights/last.pt")  # load a partially trained model
# Resume training
results = model.train(resume=True)