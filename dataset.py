import kagglehub # type: ignore

# Download latest version
path = kagglehub.dataset_download("saugataroyarghya/resume-dataset")

print("Path to dataset files:", path)