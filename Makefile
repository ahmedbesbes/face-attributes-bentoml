train:	
	poetry run python -m src.train.main \
		--csv_data_path ./data/list_attr_celeba.csv \
		--image_folder_path ./data/img_align_celeba/img_align_celeba
